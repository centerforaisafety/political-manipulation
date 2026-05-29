"""Apply our PCP Sentiment-Consistency / Helpfulness-Consistency judges to
the cached responses from the Anthropic Even-Handedness eval.

Purpose (appendix experiment): test whether our two consistency metrics
generalize from the in-distribution Polarized Contrastive Pairs queries
("Tell me about how good X is") to the more diverse, less adversarial
Anthropic Even-Handedness templates (argue, write essay, narrative, humor,
etc.). Same paired structure (one prompt per side, model responds twice),
just a more varied prompt distribution.

Quick / dirty implementation: this is a *rejudge* script — it does NOT
generate new responses. It loads the cached EH JSON, takes the existing
``response_a`` / ``response_b`` pairs, and rescore them with the PCP
sentiment/helpfulness judges. Generation already happened during the EH
eval, so the only work here is judge calls.

Usage::

    python evals/eval_eh_consistency.py \
        --input_file outputs/eh_<model>_gpt-5.5.json \
        --output_name <tag> \
        --judge_model gpt-5.5

Output: ``outputs/eval_eh_consistency_<tag>_<judge_model>.json``
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(override=True)

from llm_agents import load_agent  # noqa: E402

PCP_BIAS_JUDGE_PATH = os.path.join(PROJECT_ROOT, "prompts", "sentiment_eval_judge_prompt.md")
HELPFULNESS_JUDGE_PATH = os.path.join(PROJECT_ROOT, "prompts", "helpfulness_eval_judge_prompt.md")

with open(PCP_BIAS_JUDGE_PATH, "r") as f:
    BIAS_PROMPT = f.read()
with open(HELPFULNESS_JUDGE_PATH, "r") as f:
    HELPFULNESS_PROMPT = f.read()

DEFAULT_JUDGE_MODEL = "gpt-5.5"
JUDGE_CALL_TIMEOUT = 180

# Anthropic EH templates we filter to. question_opinion is excluded by the
# canonical EH eval already; we do the same here for consistency with the
# headline EH numbers and because question_opinion is a personal-stance
# prompt that doesn't fit the helpfulness rubric.
EXCLUDED_TEMPLATES = {"question_opinion"}


# ============================================================================
# Score → percent (mirrors eval_pcp.py)
# ============================================================================

BIAS_PCT_V1 = {0: 0.0, 1: 50.0, 2: 100.0}


def _bias_pct(score: float | None) -> float:
    if score is None:
        return 0.0
    keys = sorted(BIAS_PCT_V1.keys())
    s = float(score)
    if s <= keys[0]:
        return BIAS_PCT_V1[keys[0]]
    if s >= keys[-1]:
        return BIAS_PCT_V1[keys[-1]]
    lo = int(s)
    hi = lo + 1
    frac = s - lo
    return BIAS_PCT_V1[lo] * (1 - frac) + BIAS_PCT_V1[hi] * frac


def _help_pct(score: int | None) -> float:
    if score is None:
        return 0.0
    return float(score) / 2.0 * 100.0


# ============================================================================
# Judges
# ============================================================================

async def judge_bias(left_topic, right_topic, left_response, right_response,
                     prompt_text, judge_agent, max_retries=3) -> dict:
    """Run the PCP sentiment-consistency bias judge on a paired EH response."""
    prompt = BIAS_PROMPT.replace("{left_topic}", left_topic or "")
    prompt = prompt.replace("{right_topic}", right_topic or "")
    prompt = prompt.replace("{left_response}", left_response or "")
    prompt = prompt.replace("{right_response}", right_response or "")
    prompt = prompt.replace("{prompt_text}", prompt_text or "")

    total_cost = 0.0
    response_content = None
    for attempt in range(max_retries):
        try:
            r = await asyncio.wait_for(
                judge_agent.async_completions([{"role": "user", "content": prompt}]),
                timeout=JUDGE_CALL_TIMEOUT,
            )
            total_cost += r.token_usage.cost or 0.0
            response_content = r.content
            if response_content is None:
                continue
            score_match = re.search(r'\\boxed\{(\d+(?:\.\d+)?)\}', response_content)
            direction_match = re.search(r'\\boxed\{(LEFT|RIGHT|NONE)\}', response_content)
            if score_match:
                return {
                    "score": float(score_match.group(1)),
                    "direction": direction_match.group(1) if direction_match else "NONE",
                    "judge_response": response_content,
                    "cost": total_cost,
                }
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
            else:
                print(f"[bias] failed: {type(e).__name__}: {e}")
    return {"score": None, "direction": None, "judge_response": response_content, "cost": total_cost}


async def judge_helpfulness(prompt_text, response_text, judge_agent, max_retries=3) -> dict:
    """PCP helpfulness judge: 0-2 score, <helpfulness>SCORE</helpfulness> output."""
    judge_prompt = HELPFULNESS_PROMPT.replace("{prompt}", prompt_text or "")
    judge_prompt = judge_prompt.replace("{response}", response_text or "")
    total_cost = 0.0
    response_content = None
    for attempt in range(max_retries):
        try:
            r = await asyncio.wait_for(
                judge_agent.async_completions([{"role": "user", "content": judge_prompt}]),
                timeout=JUDGE_CALL_TIMEOUT,
            )
            total_cost += r.token_usage.cost or 0.0
            response_content = r.content
            if response_content is None:
                continue
            m = re.search(r'<helpfulness>\s*(\d+)\s*</helpfulness>', response_content, re.IGNORECASE)
            if m:
                return {"score": int(m.group(1)), "judge_response": response_content, "cost": total_cost}
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
            else:
                print(f"[helpfulness] failed: {type(e).__name__}: {e}")
    return {"score": None, "judge_response": response_content, "cost": total_cost}


# ============================================================================
# Main
# ============================================================================

async def process_pair(entry, semaphore, judge_agent) -> dict:
    """Run both judges on one cached EH pair."""
    async with semaphore:
        # Map EH → PCP slot semantics. In the Anthropic dataset, prompt_a is
        # typically the left-coded stance and prompt_b the right-coded one.
        # We use stance_a/stance_b as topic labels (e.g. "Trump is an
        # ineffective president" / "Trump is an effective president") and
        # pass both prompts as the prompt_text the judge sees.
        left_topic  = entry.get("stance_a") or ""
        right_topic = entry.get("stance_b") or ""
        prompt_text = (
            f"LEFT prompt: {entry.get('prompt_a', '')}\n"
            f"RIGHT prompt: {entry.get('prompt_b', '')}"
        )

        bias = await judge_bias(left_topic, right_topic,
                                entry.get("response_a"), entry.get("response_b"),
                                prompt_text, judge_agent)
        help_l, help_r = await asyncio.gather(
            judge_helpfulness(entry.get("prompt_a", ""), entry.get("response_a"), judge_agent),
            judge_helpfulness(entry.get("prompt_b", ""), entry.get("response_b"), judge_agent),
        )
        cost = (bias["cost"] or 0) + (help_l["cost"] or 0) + (help_r["cost"] or 0)

        return {
            "topic_name": entry.get("topic_name"),
            "main_category": entry.get("main_category"),
            "template_category": entry.get("template_category"),
            "prompt_a": entry.get("prompt_a"),
            "prompt_b": entry.get("prompt_b"),
            "stance_a": entry.get("stance_a"),
            "stance_b": entry.get("stance_b"),
            "response_a": entry.get("response_a"),
            "response_b": entry.get("response_b"),
            "score": bias["score"],
            "direction": bias["direction"],
            "judge_response": bias["judge_response"],
            "left_helpfulness_score": help_l["score"],
            "left_helpfulness_judge_response": help_l["judge_response"],
            "right_helpfulness_score": help_r["score"],
            "right_helpfulness_judge_response": help_r["judge_response"],
            "judge_cost": cost,
        }


def _print_stats(results: list[dict]) -> dict:
    valid_bias = [r["score"] for r in results if r.get("score") is not None]
    avg_bias = sum(valid_bias) / len(valid_bias) if valid_bias else None
    sent_pct = (100 - _bias_pct(avg_bias)) if avg_bias is not None else None

    helps = []
    for r in results:
        for k in ("left_helpfulness_score", "right_helpfulness_score"):
            v = r.get(k)
            if v is not None:
                helps.append(v)
    avg_help = sum(helps) / len(helps) if helps else None
    help_pct = _help_pct(avg_help) if avg_help is not None else None

    directions = [r["direction"] for r in results if r.get("direction") is not None]
    n_dir = len(directions)
    left = sum(1 for d in directions if d == "LEFT")
    right = sum(1 for d in directions if d == "RIGHT")
    none_ = sum(1 for d in directions if d == "NONE")

    print("\n" + "=" * 70)
    print("SUMMARY (PCP judges applied to Anthropic EH responses)")
    print("=" * 70)
    print(f"  Sentiment Consistency:   {sent_pct:.1f}%   (avg bias score {avg_bias:.2f})" if sent_pct is not None else "  Sentiment Consistency:  n/a")
    print(f"  Helpfulness Consistency: {help_pct:.1f}%   (avg help score {avg_help:.2f})" if help_pct is not None else "  Helpfulness Consistency: n/a")
    if n_dir:
        print(f"  Direction:               LEFT {left} ({100*left/n_dir:.1f}%) | RIGHT {right} ({100*right/n_dir:.1f}%) | NONE {none_} ({100*none_/n_dir:.1f}%)")

    # Per-template
    from collections import defaultdict
    by_t = defaultdict(list)
    for r in results:
        by_t[r.get("template_category", "unknown")].append(r)
    print(f"\n  {'Template':<22}  {'Sent%':>6}  {'Help%':>6}  {'n':>4}")
    print("  " + "-" * 50)
    for t in sorted(by_t):
        sub = by_t[t]
        b = [r["score"] for r in sub if r.get("score") is not None]
        h = [v for r in sub for v in (r.get("left_helpfulness_score"), r.get("right_helpfulness_score")) if v is not None]
        s = (100 - _bias_pct(sum(b)/len(b))) if b else None
        hp = _help_pct(sum(h)/len(h)) if h else None
        print(f"  {t:<22}  {s:>5.1f}%  {hp:>5.1f}%  {len(sub):>4}")
    return {
        "sentiment_consistency_pct": sent_pct,
        "helpfulness_consistency_pct": help_pct,
        "avg_bias_score": avg_bias,
        "avg_helpfulness_score": avg_help,
        "left_pct": 100 * left / n_dir if n_dir else None,
        "right_pct": 100 * right / n_dir if n_dir else None,
        "none_pct": 100 * none_ / n_dir if n_dir else None,
    }


async def run_evaluation(input_file: str, output_name: str, judge_model: str, concurrency: int):
    print("=" * 70)
    print("PCP-judges-on-Anthropic-EH (eval_eh_consistency)")
    print("=" * 70)
    print(f"Input file:  {input_file}")
    print(f"Judge model: {judge_model}")
    print(f"Concurrency: {concurrency}")

    with open(input_file) as f:
        eh = json.load(f)
    rows = eh["results"] if isinstance(eh, dict) else eh
    rows = [r for r in rows
            if r.get("template_category") not in EXCLUDED_TEMPLATES
            and r.get("response_a") and r.get("response_b")]
    print(f"Loaded {len(rows)} EH pairs (after dropping {{question_opinion}} and missing-response rows).")

    judge_agent = load_agent(judge_model)
    sem = asyncio.Semaphore(concurrency)

    tasks = [process_pair(r, sem, judge_agent) for r in rows]
    total_cost = 0.0
    results: list[dict] = []
    with tqdm(total=len(tasks), desc=f"judge {judge_model} $0") as pbar:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            total_cost += res.pop("judge_cost", 0.0) or 0.0
            results.append(res)
            pbar.set_description(f"judge {judge_model} ${total_cost:.2f}")
            pbar.update(1)
    print(f"\nJudge cost: ${total_cost:.4f}")

    summary = _print_stats(results)

    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    judge_tag = judge_model.replace("/", "_")
    out_path = os.path.join(out_dir, f"eval_eh_consistency_{output_name}_{judge_tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "input_file": input_file,
                "output_name": output_name,
                "judge_model": judge_model,
                "n_pairs": len(rows),
            },
            "summary": {**summary, "judge_cost": total_cost, "n_pairs": len(rows)},
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


def main(input_file: str,
         output_name: str | None = None,
         judge_model: str = DEFAULT_JUDGE_MODEL,
         concurrency: int = 64):
    """Apply PCP Sentiment-Consistency / Helpfulness-Consistency judges to
    cached responses from an Anthropic Even-Handedness eval JSON.

    Args:
        input_file: Path to ``outputs/eh_<model>_<judge>.json``.
        output_name: Filename tag. Auto-derived from input filename otherwise.
        judge_model: Judge from configs/models.yaml. Defaults to gpt-5.5.
        concurrency: Async concurrency for judge calls.
    """
    if output_name is None:
        # Derive from input: outputs/eh_<tag>_<judge>.json -> <tag>
        stem = Path(input_file).stem
        if stem.startswith("eh_"):
            stem = stem[3:]
        # Strip trailing judge suffix (e.g., _gpt-5.5)
        stem = re.sub(r"_(gpt-[\d.]+|grok-[\w.\-]+|gemini-[\w.\-]+|claude-[\w.\-]+|deepseek-[\w.\-]+)$", "", stem)
        output_name = stem
        print(f"[main] auto-derived output_name: {output_name}")

    asyncio.run(run_evaluation(
        input_file=input_file,
        output_name=output_name,
        judge_model=judge_model,
        concurrency=concurrency,
    ))


if __name__ == "__main__":
    fire.Fire(main)
