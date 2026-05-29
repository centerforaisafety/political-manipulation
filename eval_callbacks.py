"""
Evaluation callbacks for GRPO political-consistency training.

This module exposes a single TrainerCallback - :class:`PCPEvalCallback` -
that runs the Polarized Contrastive Pairs (PCP) eval at every
``eval_steps`` interval and logs metrics to wandb.

Mirrors the canonical eval at :mod:`evals.eval_pcp` but with two changes
that make sense at training time:

1. The default headline grid is the **2 proxy templates** rather than the
   full 4. PCP at training time runs ``tell_me`` and ``tell_me_dhb`` only
   so each eval step stays cheap. ``evidence`` and ``argue`` are skipped
   here, but available via the ``templates`` arg.
2. Generation is routed through the in-cluster TRL vLLM ``/chat/``
   endpoint instead of the agent abstraction, so the live training
   policy is what gets judged.

Judges are the same prompts the offline ``eval_pcp.py`` uses
(``prompts/sentiment_eval_judge_prompt.md`` for bias and
``prompts/helpfulness_eval_judge_prompt.md`` for helpfulness). Score
extraction is borrowed from :mod:`evals.eval_pcp` so the two stay in
sync.

Saved JSON (``eval_pcp_<step>.json``) matches the offline schema:
``{step, config, summary: {overall, per_template, ...}, results}``.
The flat per-template metrics that wandb consumes
(``eval/pcp_<template>_*``) are derived from the per-template summary.
"""

import asyncio
import json
import os
import re
import sys
import time

import aiohttp
import torch.distributed as dist
from transformers import TrainerCallback

try:
    import wandb
except ImportError:
    wandb = None

# Resolve paths relative to this file so the callback works regardless of cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from llm_agents import get_agent_config, get_llm_agent_class

# Reuse the canonical eval helpers so the callback stays in sync with
# ``python evals/eval_pcp.py`` and the paper release.
from evals.eval_pcp import (
    PCP_TOPICS,
    PCP_BIAS_PROMPT,
    HELPFULNESS_PROMPT,
    TEMPLATE_REGISTRY,
    VALENCES,
    _bias_pct,
    _help_pct,
    _compute_form_stats,
)

# Training-time proxy: the two cheapest templates from the headline 4.
# Override of the offline ``DEFAULT_TEMPLATES`` (which runs all four).
DEFAULT_TEMPLATES = ["tell_me", "tell_me_dhb"]

JUDGE_CALL_TIMEOUT = 180


# ============================================================================
# BASE CALLBACK CLASS
# ============================================================================

class BaseEvalCallback(TrainerCallback):
    """
    Base class for evaluation callbacks during training.

    Provides:
    - vLLM generation via the TRL ``/chat/`` endpoint
    - API judge model setup
    - thinking-block stripping for Qwen3 / gpt-oss
    - on_step_end hook with wandb logging + JSON dump
    """

    METRIC_PREFIX = "eval"  # Override in subclasses

    def __init__(
        self,
        judge_model: str,
        tokenizer,
        eval_steps: int,
        vllm_base_url: str,
        output_dir: str,
        enable_thinking: bool = False,
    ):
        self.judge_model = judge_model
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.output_dir = output_dir
        self.enable_thinking = enable_thinking

        config = get_agent_config(judge_model)
        self.judge_agent = get_llm_agent_class(**config)
        self.total_cost = 0.0

    def _strip_thinking(self, text: str) -> str:
        """Remove thinking/analysis blocks. Handles Qwen3 and gpt-oss."""
        if "<|channel|>final<|message|>" in text:
            match = re.search(
                r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)',
                text, re.DOTALL,
            )
            if match:
                return match.group(1).strip()
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    async def _generate_batch(self, messages: list, max_tokens: int = 4096) -> list[str]:
        """Generate via TRL vLLM /chat/ endpoint."""
        chat_url = f"{self.vllm_base_url}/chat/"
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                chat_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                if resp.status != 200:
                    return []
                result = await resp.json()

        completion_ids = result.get("completion_ids", [])
        return [
            self.tokenizer.decode(ids, skip_special_tokens=False)
            for ids in completion_ids
        ]

    async def run_eval(self) -> tuple[dict, dict, list]:
        """Run the evaluation.

        Returns ``(metrics, save_payload, detailed_results)`` where:
        - ``metrics``: flat dict for wandb logging.
        - ``save_payload``: extra keys to merge into the saved JSON
          (alongside ``step``, ``metrics``, ``results``). E.g.
          ``{"config": ..., "summary": ...}``.
        - ``detailed_results``: list of per-pair entries.

        Override in subclasses.
        """
        raise NotImplementedError

    def _run_eval_now(self, state):
        """Run eval, log to wandb, save JSON. Caller handles cadence."""
        if dist.is_initialized():
            dist.barrier()

        if state.is_world_process_zero:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                metrics, save_payload, detailed_results = loop.run_until_complete(
                    self.run_eval()
                )

                if wandb is not None and wandb.run is not None:
                    wandb.log(metrics)

                if self.output_dir:
                    os.makedirs(self.output_dir, exist_ok=True)
                    eval_name = self.__class__.__name__.replace(
                        "EvalCallback", ""
                    ).lower()
                    eval_path = os.path.join(
                        self.output_dir,
                        f"eval_{eval_name}_{state.global_step}.json",
                    )
                    payload = {
                        "step": state.global_step,
                        "metrics": metrics,
                        **(save_payload or {}),
                        "results": detailed_results,
                    }
                    with open(eval_path, "w") as f:
                        json.dump(payload, f, indent=2)
                    print(f"[{self.METRIC_PREFIX}] Saved to {eval_path}", flush=True)
            finally:
                loop.close()

        if dist.is_initialized():
            dist.barrier()

    def on_train_begin(self, args, state, control, **kwargs):
        """No-op by default; subclasses can override."""
        return

    def on_step_end(self, args, state, control, **kwargs):
        """Run eval every ``eval_steps`` (only on main process)."""
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            self._run_eval_now(state)


# ============================================================================
# PCP (Polarized Contrastive Pairs) EVALUATION CALLBACK
# ============================================================================

class PCPEvalCallback(BaseEvalCallback):
    """Polarized Contrastive Pairs eval - training-time proxy.

    Mirrors the format of :mod:`evals.eval_pcp` (same judges, same
    score-extraction, same output JSON schema) but runs only the
    ``tell_me`` + ``tell_me_dhb`` templates by default to keep each
    eval step cheap.

    Each result entry is tagged with ``template``, ``valence``, ``score``,
    ``direction``, and (when ``mode in ("helpful", "both")``)
    ``left_helpfulness_score`` / ``right_helpfulness_score``. The saved
    JSON contains ``summary.overall`` and ``summary.per_template`` blocks
    matching the offline eval.

    Wandb metrics emitted (lower bias / higher consistency is better):
      - ``eval/pcp_<template>_score``        : avg bias score (0-2)
      - ``eval/pcp_<template>_bias_pct``     : 0-100% bias (v1 map)
      - ``eval/pcp_<template>_sentiment_consistency_pct``
      - ``eval/pcp_<template>_left_pct`` / ``_right_pct`` / ``_none_pct``
      - ``eval/pcp_<template>_help_score``         (if mode != "bias")
      - ``eval/pcp_<template>_helpfulness_consistency_pct``
      - ``eval/pcp_<template>_avg_words``
      - ``eval/pcp_score`` ...  : overall (averaged across templates)
      - ``eval/pcp_cost`` / ``eval/pcp_time``
    """

    METRIC_PREFIX = "eval/pcp"

    def __init__(
        self,
        judge_model: str,
        tokenizer,
        eval_steps: int = 50,
        vllm_base_url: str = "http://localhost:8000",
        output_dir: str | None = None,
        enable_thinking: bool = False,
        valence: str = "all",
        max_pairs: int | None = None,
        templates: list[str] | None = None,
        mode: str = "both",
        judge_concurrency: int = 128,
        eval_at_init: bool = False,
    ):
        super().__init__(
            judge_model=judge_model,
            tokenizer=tokenizer,
            eval_steps=eval_steps,
            vllm_base_url=vllm_base_url,
            output_dir=output_dir,
            enable_thinking=enable_thinking,
        )

        if mode not in ("bias", "helpful", "both"):
            raise ValueError(
                f"mode must be 'bias' | 'helpful' | 'both', got {mode!r}"
            )

        templates = templates or DEFAULT_TEMPLATES
        unknown = [t for t in templates if t not in TEMPLATE_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown template(s) {unknown!r}; valid keys: "
                f"{tuple(TEMPLATE_REGISTRY.keys())}"
            )

        self.valence = valence
        self.max_pairs = max_pairs
        self.pairs = PCP_TOPICS[:max_pairs] if max_pairs else PCP_TOPICS
        self.valences_to_use = (
            VALENCES if valence == "all" else [valence]
        )
        self.templates = [(t, TEMPLATE_REGISTRY[t]) for t in templates]
        self.mode = mode
        self.judge_concurrency = judge_concurrency
        self.eval_at_init = eval_at_init

        n_pairs_per_template = (
            len(self.pairs) * len(self.valences_to_use)
        )
        n_evals = n_pairs_per_template * len(self.templates)
        print(f"[PCP] Eval every {eval_steps} steps", flush=True)
        print(
            f"[PCP] {len(self.pairs)} pairs x {len(self.valences_to_use)} "
            f"valences x {len(self.templates)} templates "
            f"= {n_evals} paired-evaluations",
            flush=True,
        )
        print(
            f"[PCP] Templates: {[t for t, _ in self.templates]}",
            flush=True,
        )
        print(f"[PCP] Mode: {mode}", flush=True)
        print(f"[PCP] vLLM server: {self.vllm_base_url}/chat/", flush=True)

    # ------------------------------------------------------------------ judges

    async def _judge_bias_pair(
        self,
        left_topic: str,
        right_topic: str,
        left_response: str,
        right_response: str,
        valence: str,
        template_label: str,
        template_str: str,
    ) -> dict:
        """Bias judge: scores left/right asymmetry on a response pair."""
        judge_prompt_text = template_str.format(valence=valence, topic="<topic>")

        prompt = PCP_BIAS_PROMPT.replace("{left_topic}", left_topic)
        prompt = prompt.replace("{right_topic}", right_topic)
        prompt = prompt.replace("{left_response}", left_response or "")
        prompt = prompt.replace("{right_response}", right_response or "")
        prompt = prompt.replace("{valence}", valence)
        prompt = prompt.replace("{prompt_text}", judge_prompt_text)

        response_content = None
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    self.judge_agent.async_completions(
                        [{"role": "user", "content": prompt}]
                    ),
                    timeout=JUDGE_CALL_TIMEOUT,
                )
                self.total_cost += response.token_usage.cost or 0
                response_content = response.content
                if response_content is None:
                    continue
                score_match = re.search(
                    r'\\boxed\{(\d+(?:\.\d+)?)\}', response_content
                )
                direction_match = re.search(
                    r'\\boxed\{(LEFT|RIGHT|NONE)\}', response_content
                )
                if score_match:
                    return {
                        "score": float(score_match.group(1)),
                        "direction": (
                            direction_match.group(1)
                            if direction_match else "NONE"
                        ),
                        "judge_response": response_content,
                    }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    print(f"[PCP/bias] judge failed: {type(e).__name__}: {e}", flush=True)

        return {
            "score": None,
            "direction": None,
            "judge_response": response_content,
        }

    async def _judge_helpfulness(
        self,
        prompt_text: str,
        response_text: str,
    ) -> dict:
        """Helpfulness judge: 0-2 from ``<helpfulness>N</helpfulness>``."""
        judge_prompt = HELPFULNESS_PROMPT.replace("{prompt}", prompt_text or "")
        judge_prompt = judge_prompt.replace("{response}", response_text or "")

        response_content = None
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    self.judge_agent.async_completions(
                        [{"role": "user", "content": judge_prompt}]
                    ),
                    timeout=JUDGE_CALL_TIMEOUT,
                )
                self.total_cost += response.token_usage.cost or 0
                response_content = response.content
                if response_content is None:
                    continue
                m = re.search(
                    r'<helpfulness>\s*(\d+)\s*</helpfulness>',
                    response_content, re.IGNORECASE,
                )
                if m:
                    return {
                        "score": int(m.group(1)),
                        "judge_response": response_content,
                    }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    print(f"[PCP/help] judge failed: {type(e).__name__}: {e}", flush=True)

        return {"score": None, "judge_response": response_content}

    # ------------------------------------------------------------------ wandb

    def _flatten_for_wandb(
        self,
        results: list[dict],
        label: str | None,
    ) -> dict:
        """Turn ``_compute_form_stats`` output into a flat
        ``eval/pcp_<label>_*`` dict for wandb."""
        stats = _compute_form_stats(
            results, self.valences_to_use, self.mode,
            form_label=label,
        )
        suffix = f"_{label}" if label else ""
        prefix = f"{self.METRIC_PREFIX}{suffix}"
        out: dict = {f"{prefix}_n": stats["n"]}
        if stats["n"] == 0:
            return out

        if "bias" in stats and "score" in stats["bias"]:
            b = stats["bias"]
            out[f"{prefix}_score"] = b["score"]
            out[f"{prefix}_bias_pct"] = b["bias_pct"]
            out[f"{prefix}_sentiment_consistency_pct"] = b["sentiment_consistency_pct"]
            out[f"{prefix}_left_pct"] = b.get("left_pct", 0.0)
            out[f"{prefix}_right_pct"] = b.get("right_pct", 0.0)
            out[f"{prefix}_none_pct"] = b.get("none_pct", 0.0)
        if "helpful" in stats and "score" in stats["helpful"]:
            h = stats["helpful"]
            out[f"{prefix}_help_score"] = h["score"]
            out[f"{prefix}_helpfulness_consistency_pct"] = h["helpfulness_consistency_pct"]
            out[f"{prefix}_left_help_score"] = h.get("left_score", 0.0)
            out[f"{prefix}_right_help_score"] = h.get("right_score", 0.0)

        # Average response length, computed on the subset.
        sub = (
            results if label is None
            else [r for r in results if r.get("template") == label]
        )
        left_words = [
            len(r["left_response"].split())
            for r in sub if r.get("left_response")
        ]
        right_words = [
            len(r["right_response"].split())
            for r in sub if r.get("right_response")
        ]
        avg_left = sum(left_words) / len(left_words) if left_words else 0.0
        avg_right = sum(right_words) / len(right_words) if right_words else 0.0
        out[f"{prefix}_avg_words"] = (avg_left + avg_right) / 2

        return out

    # ------------------------------------------------------------------ main

    async def run_eval(self) -> tuple[dict, dict, list]:
        """Run the full PCP eval (all configured templates in one batch)."""
        start_time = time.time()

        # 1) Build the full prompt list across (template, pair, valence).
        all_messages: list[list[dict]] = []
        # prompt_info[j] -> (template_label, pair_idx, valence, side)
        prompt_info: list[tuple[str, int, str, str]] = []

        for template_label, template_str in self.templates:
            for i, pair in enumerate(self.pairs):
                for v in self.valences_to_use:
                    left_prompt = template_str.format(
                        valence=v, topic=pair["left"]
                    )
                    right_prompt = template_str.format(
                        valence=v, topic=pair["right"]
                    )
                    all_messages.append(
                        [{"role": "user", "content": left_prompt}]
                    )
                    prompt_info.append((template_label, i, v, "left"))
                    all_messages.append(
                        [{"role": "user", "content": right_prompt}]
                    )
                    prompt_info.append((template_label, i, v, "right"))

        print(
            f"[PCP] Generating {len(all_messages)} responses "
            f"({len(self.templates)} templates in one batch)...",
            flush=True,
        )
        generated_texts = await self._generate_batch(all_messages)

        if not generated_texts:
            print("[PCP] ERROR: vLLM returned no generations.", flush=True)
            return (
                {
                    f"{self.METRIC_PREFIX}_score": 0,
                    f"{self.METRIC_PREFIX}_error": 1,
                },
                {},
                [],
            )

        # 2) Organize: (template, pair_idx, valence) -> {left, right} text.
        pair_responses: dict = {}
        for j, text in enumerate(generated_texts):
            template_label, pair_idx, valence, side = prompt_info[j]
            key = (template_label, pair_idx, valence)
            pair_responses.setdefault(key, {})[side] = self._strip_thinking(text)

        # 3) Build per-pair entries (mirrors the eval_pcp pair_data list).
        template_lookup = dict(self.templates)
        pair_entries: list[dict] = []
        for (template_label, pair_idx, valence), data in pair_responses.items():
            pair = self.pairs[pair_idx]
            template_str = template_lookup[template_label]
            left_prompt = template_str.format(valence=valence, topic=pair["left"])
            right_prompt = template_str.format(valence=valence, topic=pair["right"])
            pair_entries.append({
                "left_topic": pair["left"],
                "right_topic": pair["right"],
                "category": pair.get("category", "unknown"),
                "valence": valence,
                "template": template_label,
                "left_prompt": left_prompt,
                "right_prompt": right_prompt,
                "left_response": data.get("left", ""),
                "right_response": data.get("right", ""),
            })

        # 4) Judge all pairs concurrently.
        semaphore = asyncio.Semaphore(self.judge_concurrency)
        mode = self.mode

        async def judge_with_sem(entry: dict) -> dict:
            async with semaphore:
                template_str = template_lookup[entry["template"]]
                out = dict(entry)
                tasks: dict[str, asyncio.Task] = {}
                if mode in ("bias", "both"):
                    tasks["bias"] = asyncio.create_task(self._judge_bias_pair(
                        entry["left_topic"], entry["right_topic"],
                        entry["left_response"], entry["right_response"],
                        entry["valence"], entry["template"], template_str,
                    ))
                if mode in ("helpful", "both"):
                    tasks["help_left"] = asyncio.create_task(
                        self._judge_helpfulness(
                            entry["left_prompt"], entry["left_response"],
                        )
                    )
                    tasks["help_right"] = asyncio.create_task(
                        self._judge_helpfulness(
                            entry["right_prompt"], entry["right_response"],
                        )
                    )
                keys = list(tasks.keys())
                judged = await asyncio.gather(*[tasks[k] for k in keys])
                judged = dict(zip(keys, judged))
                if "bias" in judged:
                    b = judged["bias"]
                    out["score"] = b["score"]
                    out["direction"] = b["direction"]
                    out["judge_response"] = b["judge_response"]
                if "help_left" in judged:
                    h = judged["help_left"]
                    out["left_helpfulness_score"] = h["score"]
                    out["left_helpfulness_judge_response"] = h["judge_response"]
                if "help_right" in judged:
                    h = judged["help_right"]
                    out["right_helpfulness_score"] = h["score"]
                    out["right_helpfulness_judge_response"] = h["judge_response"]
                return out

        print(
            f"[PCP] Judging {len(pair_entries)} pairs "
            f"(mode={mode}, concurrency={self.judge_concurrency})...",
            flush=True,
        )
        results = await asyncio.gather(
            *(judge_with_sem(e) for e in pair_entries)
        )

        eval_time = time.time() - start_time

        # 5) Build summary blocks (mirrors eval_pcp.run_evaluation).
        templates_present = sorted(
            {r.get("template") for r in results if r.get("template")}
        )
        per_template_stats: dict = {}
        for tp in templates_present:
            per_template_stats[tp] = _compute_form_stats(
                results, self.valences_to_use, self.mode,
                form_label=tp,
            )
        overall_stats = _compute_form_stats(
            results, self.valences_to_use, self.mode,
            form_label=None,
        )

        # 6) Flatten for wandb (per-template + overall).
        metrics: dict = {}
        for tp in templates_present:
            metrics.update(self._flatten_for_wandb(results, tp))
        metrics.update(self._flatten_for_wandb(results, None))
        metrics[f"{self.METRIC_PREFIX}_cost"] = self.total_cost
        metrics[f"{self.METRIC_PREFIX}_time"] = eval_time

        # 7) Pretty-print summary (with n=0 guards).
        n_total = len(results)
        n_valid_bias = sum(1 for r in results if r.get("score") is not None)
        print(
            f"[PCP] Complete in {eval_time:.1f}s | Cost: ${self.total_cost:.4f} | "
            f"valid_bias={n_valid_bias}/{n_total}",
            flush=True,
        )
        if n_valid_bias == 0 and self.mode in ("bias", "both"):
            print(
                "[PCP] WARNING: all bias judge calls failed (n=0). "
                "Skipping per-template direction breakdown.",
                flush=True,
            )
        for tp in templates_present + [None]:
            self._print_template_summary(results, tp)

        # 8) Build save payload (config + summary, eval_pcp-compatible).
        summary: dict = {
            "overall": overall_stats,
            "n_pairs": len(results),
            "eval_cost": self.total_cost,
            "judge_cost": self.total_cost,
            "eval_time": eval_time,
        }
        if len(templates_present) > 1:
            summary["per_template"] = per_template_stats

        save_payload = {
            "config": {
                "eval_model": "vllm_train_policy",
                "judge_model": self.judge_model,
                "templates": [t for t, _ in self.templates],
                "mode": self.mode,
                "valence": self.valence,
                "n_pairs": len(results),
                "system_prompt": False,
                "enable_thinking": self.enable_thinking,
                "max_pairs": self.max_pairs,
            },
            "summary": summary,
        }

        return metrics, save_payload, list(results)

    def _print_template_summary(
        self,
        results: list[dict],
        label: str | None,
    ) -> None:
        """Pretty-print one template's stats with n=0 guards."""
        sub = (
            results if label is None
            else [r for r in results if r.get("template") == label]
        )
        n = len(sub)
        tag = label if label else "overall"
        if n == 0:
            print(f"[PCP/{tag}] no entries", flush=True)
            return

        directions = [
            r["direction"] for r in sub if r.get("direction") is not None
        ]
        scores = [r["score"] for r in sub if r.get("score") is not None]
        n_dir = len(directions)
        n_scores = len(scores)

        left_count = sum(1 for d in directions if d == "LEFT")
        right_count = sum(1 for d in directions if d == "RIGHT")
        none_count = sum(1 for d in directions if d == "NONE")
        avg_score = sum(scores) / n_scores if n_scores > 0 else 0.0

        left_words = [
            len(r["left_response"].split())
            for r in sub if r.get("left_response")
        ]
        right_words = [
            len(r["right_response"].split())
            for r in sub if r.get("right_response")
        ]
        avg_left = sum(left_words) / len(left_words) if left_words else 0.0
        avg_right = sum(right_words) / len(right_words) if right_words else 0.0
        avg_words = (avg_left + avg_right) / 2

        if self.mode in ("bias", "both"):
            if n_dir > 0:
                bias_pct = _bias_pct(avg_score)
                print(
                    f"[PCP/{tag}] score={avg_score:.2f} | "
                    f"SC%={(100.0 - bias_pct):.1f} | "
                    f"LEFT={left_count} ({100 * left_count / n_dir:.1f}%) | "
                    f"RIGHT={right_count} ({100 * right_count / n_dir:.1f}%) | "
                    f"NONE={none_count} ({100 * none_count / n_dir:.1f}%) | "
                    f"avg_words={avg_words:.0f}",
                    flush=True,
                )
            else:
                print(
                    f"[PCP/{tag}] BIAS: all {n} judge calls failed. "
                    f"avg_words={avg_words:.0f}",
                    flush=True,
                )

        if self.mode in ("helpful", "both"):
            left_h = [
                r["left_helpfulness_score"] for r in sub
                if r.get("left_helpfulness_score") is not None
            ]
            right_h = [
                r["right_helpfulness_score"] for r in sub
                if r.get("right_helpfulness_score") is not None
            ]
            all_h = left_h + right_h
            if all_h:
                avg_h = sum(all_h) / len(all_h)
                avg_lh = sum(left_h) / len(left_h) if left_h else 0.0
                avg_rh = sum(right_h) / len(right_h) if right_h else 0.0
                print(
                    f"[PCP/{tag}] help_score={avg_h:.2f} "
                    f"(L={avg_lh:.2f}, R={avg_rh:.2f}) | "
                    f"HC%={_help_pct(avg_h):.1f}",
                    flush=True,
                )
            else:
                print(
                    f"[PCP/{tag}] HELP: all {n * 2} judge calls failed.",
                    flush=True,
                )

    def on_train_begin(self, args, state, control, **kwargs):
        """Optionally run an eval at step 0 before training begins."""
        if self.eval_at_init:
            self._run_eval_now(state)
