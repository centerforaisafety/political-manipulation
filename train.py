"""
GRPO Training for Political Consistency.

Trains a model with two complementary judges, routed by the prompt's
``query_type`` field in the dataset:

- Helpfulness judge (no anchors): for "Tell me how good/bad" prompts —
  rewards strong, on-task argumentation. Punishes hedging / refusals.
- Sentiment judge (with left/right anchors + helpfulness sub-score): for
  "Tell me about" prompts — rewards balanced AND substantive responses.

The sentiment judge returns TWO scores in XML format::

    <bias_score>0-5</bias_score>             (3 = balanced)
    <helpfulness_score>0-2</helpfulness_score>

The final sentiment reward is multiplicative:
``reward = bias_factor * helpfulness_factor`` (inspired by GPT-5 safe
completions). A balanced-but-shallow response is punished, just like a
substantive-but-biased response.

Optionally, a markdown-header penalty scales rewards down on responses with
many ``#`` headers (set ``--markdown_penalty``).

Usage:
    python train.py \\
        --model_path Qwen/Qwen3-14B \\
        --data_path data/train_dataset_gemini_anchors.json \\
        --judge_model gemini-3.1-pro-preview \\
        --helpfulness_judge_prompt_path prompts/helpfulness_train_judge_prompt.md \\
        --sentiment_judge_prompt_path prompts/sentiment_train_judge_prompt.md
"""

import os
os.environ["NCCL_TIMEOUT"] = "7200"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "7200"
os.environ["NCCL_BLOCKING_WAIT"] = "0"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

# Override the default NCCL process group timeout (default is 600s = 10min,
# too short for eval API calls during GRPO training).
import datetime
import torch._C._distributed_c10d as _c10d
_c10d._DEFAULT_PG_NCCL_TIMEOUT = datetime.timedelta(seconds=7200)
_c10d._DEFAULT_PG_TIMEOUT = datetime.timedelta(seconds=7200)

import json
import re
import asyncio
import time

from dotenv import load_dotenv
load_dotenv(override=True)

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from peft import LoraConfig
from accelerate import PartialState
from accelerate.utils import gather_object
import fire

os.environ.setdefault("WEAVE_LOG_LEVEL", "WARNING")
os.environ.setdefault("WANDB_PROJECT", "grpo-political-consistency")
import wandb
import weave

ROLLOUT_LOG_FILE = None
if os.environ.get("ROLLOUT_LOGS"):
    ROLLOUT_LOG_FILE = open(os.environ["ROLLOUT_LOGS"], "a")

import transformers
import logging
import warnings

transformers.logging.set_verbosity_warning()
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
torch.set_float32_matmul_precision("high")

from llm_agents import get_llm_agent_class, get_agent_config
from eval_callbacks import PCPEvalCallback

VLLM_SERVER_PORT = int(os.environ.get("VLLM_PORT", 8000))
VLLM_GROUP_PORT = int(os.environ.get("VLLM_GROUP_PORT", 51216))


# =============================================================================
# REWARD MAPS
# =============================================================================

HELPFULNESS_REWARD_MAP = {
    0: -4.0,  # Broken response
    1: -2.0,  # Fully hedged / refused / argued opposite
    2: -1.0,  # Mostly hedged
    3: -0.5,  # Partially helpful (no specific evidence)
    4: 1.0,   # Genuinely helpful — evidence in measured tone
    5: 2.0,   # Thoughtfully helpful — engages counterargument, holds position
}

# =============================================================================
# Multiplicative sentiment reward: reward = bias_factor * helpfulness_factor.
#
# Inspired by the GPT-5 safe-completions paper (r = h * s). The model has to
# be good on BOTH dimensions to receive a positive reward:
#
#   bias=3, h=2  ->  4 * 3  = 12   (balanced + substantive — ideal)
#   bias=3, h=1  ->  4 * 1  =  4   (balanced but fence-sitting — discouraged)
#   bias=2, h=2  ->  2 * 3  =  6   (helpful but biased — discouraged)
#   bias=1, h=2  ->  1 * 3  =  3   (helpful but very biased — discouraged)
#   bias=0, h=*  ->  1 * *        (broken — capped low)
# =============================================================================

# Bias factor: peaks at 3 (balanced). Floor at 1 so a helpfulness gradient
# still exists when the model is strongly biased.
SENTIMENT_BIAS_FACTOR = {
    0: 1,    # Broken response
    1: 1,    # Strongly biased left
    2: 2,    # Somewhat biased left
    3: 4,    # Balanced (ideal)
    4: 2,    # Somewhat biased right
    5: 1,    # Strongly biased right
}

# Helpfulness factor: peaks at 2 (genuine evaluation).
SENTIMENT_HELPFULNESS_FACTOR = {
    0: 0,    # Wikipedia mode / broken
    1: 1,    # Fence-sitting / template mode
    2: 3,    # Genuine evaluation with verdicts
}


def get_reward_config() -> dict:
    return {
        "helpfulness_reward_map": HELPFULNESS_REWARD_MAP,
        "sentiment_bias_factor": SENTIMENT_BIAS_FACTOR,
        "sentiment_helpfulness_factor": SENTIMENT_HELPFULNESS_FACTOR,
        "description": {
            "helpfulness": (
                "For 'Tell me how good/bad' prompts: reward strong engagement "
                "(4-5), penalize hedging (1-2)."
            ),
            "sentiment": (
                "For 'Tell me about' prompts: MULTIPLICATIVE reward = "
                "bias_factor * helpfulness_factor. Must be good on BOTH "
                "dimensions to get positive reward. Inspired by GPT-5 safe "
                "completions paper (r = h * s)."
            ),
        },
    }


def extract_helpfulness_score(response: str) -> float | None:
    """Extract score from helpfulness judge (uses \\boxed{} format)."""
    if response is None:
        return None
    match = re.search(r"\\boxed\{(-?\d+\.?\d*)\}", response)
    if match:
        score = float(match.group(1))
        if -1 <= score <= 5:
            return score
    return None


def extract_sentiment_scores(response: str) -> tuple[int | None, int | None]:
    """Extract bias_score and helpfulness_score from sentiment judge (XML)."""
    if response is None:
        return None, None
    bias_match = re.search(r'<bias_score>(\d+)</bias_score>', response)
    help_match = re.search(r'<helpfulness_score>(\d+)</helpfulness_score>', response)
    bias_score = int(bias_match.group(1)) if bias_match else None
    help_score = int(help_match.group(1)) if help_match else None
    if bias_score is not None and not (0 <= bias_score <= 5):
        bias_score = None
    if help_score is not None and not (0 <= help_score <= 2):
        help_score = None
    return bias_score, help_score


def compute_md_factor(text: str) -> float:
    """Markdown-header penalty: more headers => smaller multiplier."""
    headers = len(re.findall(r'(?:^|\n)\s*#{1,4}\s+', text))
    if headers <= 1:
        return 1.0
    return 1.0 / (1.0 + (headers - 1) * 0.15)


def compute_helpfulness_reward(
    score: float,
    completion: str | None = None,
    markdown_penalty: bool = False,
) -> float:
    if score is None:
        return 0.0
    reward = HELPFULNESS_REWARD_MAP.get(score, 0.0)
    if markdown_penalty and completion is not None:
        reward *= compute_md_factor(completion)
    return reward


def compute_sentiment_reward(
    bias_score: int,
    helpfulness_score: int,
) -> float:
    """Multiplicative sentiment reward = bias_factor * helpfulness_factor.

    The markdown penalty is applied only to the helpfulness reward (where
    formatting can be reward-hacked); the sentiment reward depends on the
    judge's bias/helpfulness scores already, which the judge prompt itself
    accounts for, so no extra penalty is applied here.
    """
    if bias_score is None:
        return 0.0
    bias_factor = SENTIMENT_BIAS_FACTOR.get(bias_score, 0)
    help_factor = (
        SENTIMENT_HELPFULNESS_FACTOR.get(helpfulness_score, 0)
        if helpfulness_score is not None else 0
    )
    return float(bias_factor * help_factor)


class DualJudgeRewardFunction:
    __name__ = "dual_judge_pcp"

    def __init__(
        self,
        judge_model: str,
        helpfulness_judge_prompt: str,
        sentiment_judge_prompt: str,
        num_generations: int,
        markdown_penalty: bool = False,
        length_penalty: float = 0.0,
    ):
        self.judge_model_name = judge_model
        self.helpfulness_judge_prompt = helpfulness_judge_prompt
        self.sentiment_judge_prompt = sentiment_judge_prompt
        self.num_generations = num_generations
        self.markdown_penalty = markdown_penalty
        self.length_penalty = length_penalty
        config = get_agent_config(judge_model)
        self.total_cost = 0.0
        self.total_calls = 0
        self.agent = get_llm_agent_class(**config)
        self._state = PartialState()
        if self._state.is_main_process:
            print(f"Dual Judge: {judge_model} ({config['model']}) | "
                  f"markdown_penalty={markdown_penalty}")

    _TRAILING_SPECIAL_TOKENS_RE = re.compile(
        r"(?:<\|im_end\|>|<\|end\|>|<\|return\|>|<\|endoftext\|>)+\s*$"
    )

    def _strip_trailing_special(self, text: str) -> str:
        return self._TRAILING_SPECIAL_TOKENS_RE.sub("", text).strip()

    def _extract_final_response(self, raw_text: str) -> str:
        # gpt-oss harmony channel format
        if match := re.search(
            r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
            raw_text,
            re.DOTALL,
        ):
            return self._strip_trailing_special(match.group(1).strip())
        # Qwen3 thinking format
        if match := re.search(r"</think>(.*)", raw_text, re.DOTALL):
            return self._strip_trailing_special(match.group(1).strip())
        return self._strip_trailing_special(raw_text.strip())

    @weave.op
    async def _judge_single(
        self,
        prompt_text: str,
        completion_text: str,
        left: str,
        right: str,
        query_type: str | None = None,
    ) -> dict:
        completion_text = self._extract_final_response(completion_text)
        if query_type and query_type in ("helpfulness", "sentiment"):
            judge_type = query_type
        else:
            raise ValueError(
                f"query_type must be 'helpfulness' or 'sentiment', "
                f"got: {query_type!r}. Ensure your training data has a "
                f"'query_type' field for every entry."
            )

        if judge_type == "helpfulness":
            judge_prompt = self.helpfulness_judge_prompt.replace(
                "{model_prompt}", prompt_text
            )
            judge_prompt = judge_prompt.replace("{model_response}", completion_text)
        else:
            exemplars = (
                f"<left_wing_anchor>{left}</left_wing_anchor>\n\n"
                f"<right_wing_anchor>{right}</right_wing_anchor>"
            )
            judge_prompt = self.sentiment_judge_prompt.replace(
                "{model_prompt}", prompt_text
            )
            judge_prompt = judge_prompt.replace("{exemplars}", exemplars)
            judge_prompt = judge_prompt.replace(
                "{model_response}", completion_text
            )

        for _ in range(3):
            try:
                response = await self.agent.async_completions(
                    [{"role": "user", "content": judge_prompt}]
                )
                self.total_cost += response.token_usage.cost or 0.0

                if judge_type == "helpfulness":
                    score = extract_helpfulness_score(response.content)
                    if score is not None:
                        reward = compute_helpfulness_reward(
                            score,
                            completion=completion_text,
                            markdown_penalty=self.markdown_penalty,
                        )
                        if ROLLOUT_LOG_FILE:
                            ROLLOUT_LOG_FILE.write(
                                json.dumps({
                                    "prompt": prompt_text,
                                    "completion": completion_text,
                                    "score": score,
                                    "reward": reward,
                                    "judge_type": judge_type,
                                }) + "\n"
                            )
                            ROLLOUT_LOG_FILE.flush()
                        return {
                            "reward": reward,
                            "score": score,
                            "helpfulness_score": None,
                            "judge_type": judge_type,
                            "prompt": prompt_text,
                            "completion": completion_text,
                        }
                else:
                    bias_score, helpfulness_score = extract_sentiment_scores(
                        response.content
                    )
                    if bias_score is not None:
                        reward = compute_sentiment_reward(
                            bias_score,
                            helpfulness_score,
                        )
                        if self.length_penalty > 0:
                            word_count = len(completion_text.split())
                            if word_count > 1000:
                                reward -= self.length_penalty
                        if ROLLOUT_LOG_FILE:
                            ROLLOUT_LOG_FILE.write(
                                json.dumps({
                                    "prompt": prompt_text,
                                    "completion": completion_text,
                                    "score": bias_score,
                                    "helpfulness_score": helpfulness_score,
                                    "reward": reward,
                                    "judge_type": judge_type,
                                }) + "\n"
                            )
                            ROLLOUT_LOG_FILE.flush()
                        return {
                            "reward": reward,
                            "score": bias_score,
                            "helpfulness_score": helpfulness_score,
                            "judge_type": judge_type,
                            "prompt": prompt_text,
                            "completion": completion_text,
                        }
            except Exception:
                await asyncio.sleep(1.0)

        return {
            "reward": None,
            "score": None,
            "helpfulness_score": None,
            "judge_type": judge_type,
            "prompt": prompt_text,
            "completion": completion_text,
        }

    async def _judge_batch_async(
        self,
        prompts: list,
        completions: list,
        left_anchors: list[str],
        right_anchors: list[str],
        query_types: list[str] | None = None,
    ):
        semaphore = asyncio.Semaphore(256)
        if query_types is None:
            raise ValueError(
                "query_types must be provided. Ensure your training data has "
                "a 'query_type' field for every entry."
            )

        async def judge_with_semaphore(idx, p, c, l, r, qt):
            async with semaphore:
                prompt_text = p[0]["content"]
                completion_text = c[-1]["content"]
                result = await self._judge_single(
                    prompt_text, completion_text, l, r,
                    query_type=qt,
                )
                return (
                    idx,
                    result["reward"],
                    result["score"],
                    result["helpfulness_score"],
                    result["judge_type"],
                )

        tasks = [
            judge_with_semaphore(i, p, c, l, r, qt)
            for i, (p, c, l, r, qt) in enumerate(
                zip(prompts, completions, left_anchors, right_anchors, query_types)
            )
        ]

        results = await asyncio.gather(*tasks)

        rewards = []
        valid_indices = []
        scores = []
        helpfulness_scores = []
        judge_types = []
        failed_count = 0
        for idx, reward, score, help_score, jt in results:
            scores.append(score)
            helpfulness_scores.append(help_score)
            judge_types.append(jt)
            if reward is not None:
                rewards.append(reward)
                valid_indices.append(idx)
            else:
                failed_count += 1

        if failed_count > 0:
            print(
                f"Warning: {failed_count}/{len(results)} judgments failed "
                f"and will be skipped",
                flush=True,
            )

        return rewards, valid_indices, scores, helpfulness_scores, judge_types

    def __call__(
        self,
        prompts: list,
        completions: list,
        left_anchor: list[str],
        right_anchor: list[str],
        query_type: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        start_time = time.time()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            (rewards, valid_indices, scores,
             helpfulness_scores, judge_types) = loop.run_until_complete(
                self._judge_batch_async(
                    prompts, completions, left_anchor, right_anchor,
                    query_types=query_type,
                )
            )
        finally:
            loop.close()

        judge_time = time.time() - start_time
        self.total_calls += len(prompts)

        full_rewards: list = [None] * len(prompts)
        for idx, reward in zip(valid_indices, rewards):
            full_rewards[idx] = reward

        # For failed judgments, assign the group mean -> zero GRPO advantage.
        num_gens = self.num_generations
        for i in range(0, len(full_rewards), num_gens):
            group = full_rewards[i:i + num_gens]
            valid_in_group = [r for r in group if r is not None]
            group_mean = sum(valid_in_group) / len(valid_in_group) if valid_in_group else 0.0
            for j in range(i, min(i + num_gens, len(full_rewards))):
                if full_rewards[j] is None:
                    full_rewards[j] = group_mean

        valid_scores = [s for s in scores if s is not None]
        valid_help_scores = [
            s for s, jt in zip(scores, judge_types)
            if s is not None and jt == "helpfulness"
        ]
        valid_sent_scores = [
            s for s, jt in zip(scores, judge_types)
            if s is not None and jt == "sentiment"
        ]
        valid_helpfulness_subscores = [
            h for h, jt in zip(helpfulness_scores, judge_types)
            if h is not None and jt == "sentiment"
        ]
        local_stats = {
            "scores": valid_scores,
            "helpfulness_scores": valid_help_scores,
            "sentiment_scores": valid_sent_scores,
            "helpfulness_subscores": valid_helpfulness_subscores,
            "rewards": rewards,
            "cost": self.total_cost,
            "time": judge_time,
            "num_samples": len(prompts),
        }

        all_stats = gather_object([local_stats])

        if self._state.is_main_process:
            all_scores = []
            all_helpful = []
            all_sentiment = []
            all_help_sub = []
            all_rewards = []
            total_cost = 0.0
            max_time = 0.0
            total_samples = 0

            for stats in all_stats:
                all_scores.extend(stats["scores"])
                all_helpful.extend(stats["helpfulness_scores"])
                all_sentiment.extend(stats["sentiment_scores"])
                all_help_sub.extend(stats["helpfulness_subscores"])
                all_rewards.extend(stats["rewards"])
                total_cost += stats["cost"]
                max_time = max(max_time, stats["time"])
                total_samples += stats["num_samples"]

            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
            avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

            help_dist = {s: all_helpful.count(s) for s in set(all_helpful)}
            sent_dist = {s: all_sentiment.count(s) for s in set(all_sentiment)}
            sub_dist = {s: all_help_sub.count(s) for s in set(all_help_sub)}

            if wandb.run is not None:
                log_dict = {
                    "judge/score": avg_score,
                    "judge/reward": avg_reward,
                    "judge/cost": total_cost,
                }
                if all_helpful:
                    log_dict["judge/helpfulness_score"] = sum(all_helpful) / len(all_helpful)
                    log_dict["judge/helpfulness_hedged"] = (
                        all_helpful.count(1) + all_helpful.count(2)
                    )
                    log_dict["judge/helpfulness_mixed"] = all_helpful.count(3)
                    log_dict["judge/helpfulness_engaged"] = (
                        all_helpful.count(4) + all_helpful.count(5)
                    )
                if all_sentiment:
                    log_dict["judge/sentiment_score"] = sum(all_sentiment) / len(all_sentiment)
                    log_dict["judge/sentiment_balanced"] = all_sentiment.count(3)
                if all_help_sub:
                    log_dict["judge/sentiment_help_sub_score"] = (
                        sum(all_help_sub) / len(all_help_sub)
                    )
                    log_dict["judge/sentiment_help_sub_shallow"] = all_help_sub.count(0)
                    log_dict["judge/sentiment_help_sub_adequate"] = all_help_sub.count(1)
                    log_dict["judge/sentiment_help_sub_substantive"] = all_help_sub.count(2)
                wandb.log(log_dict)

            print(
                f"[Judge] Batch: {total_samples} | Time: {max_time:.1f}s | "
                f"Cost: ${total_cost:.4f} | Avg Score: {avg_score:.2f} | "
                f"Avg Reward: {avg_reward:.2f} | Helpfulness: {help_dist} | "
                f"Sentiment: {sent_dist} | Sentiment-help-sub: {sub_dist}",
                flush=True,
            )

        return full_rewards


def load_dataset_from_json(json_path: str) -> Dataset:
    print(f"Loading data from {json_path}...")
    data = json.load(open(json_path))

    missing = [i for i, item in enumerate(data) if "query_type" not in item]
    if missing:
        raise ValueError(
            f"Missing 'query_type' field in {len(missing)} entries "
            f"(first 5 indices: {missing[:5]}). Every entry must have "
            f"query_type='helpfulness' or 'sentiment'."
        )

    invalid = [
        (i, item["query_type"])
        for i, item in enumerate(data)
        if item["query_type"] not in ("helpfulness", "sentiment")
    ]
    if invalid:
        raise ValueError(
            f"Invalid query_type values in {len(invalid)} entries "
            f"(first 5: {invalid[:5]}). Must be 'helpfulness' or 'sentiment'."
        )

    dataset_dict = {
        "prompt": [[{"role": "user", "content": item["query"]}] for item in data],
        "left_anchor": [item.get("left", "") for item in data],
        "right_anchor": [item.get("right", "") for item in data],
        "query_type": [item["query_type"] for item in data],
    }

    dataset = Dataset.from_dict(dataset_dict)

    n_help = sum(1 for item in data if item["query_type"] == "helpfulness")
    n_sent = sum(1 for item in data if item["query_type"] == "sentiment")
    print(f"Loaded {len(dataset)} samples ({n_help} helpfulness, {n_sent} sentiment)")
    return dataset


def main(
    model_path: str = "Qwen/Qwen3-14B",
    judge_model: str = "gemini-3.1-pro-preview",
    eval_judge_model: str = "gpt-5.5",
    helpfulness_judge_prompt_path: str = "prompts/helpfulness_train_judge_prompt.md",
    sentiment_judge_prompt_path: str = "prompts/sentiment_train_judge_prompt.md",
    data_path: str = "data/train_dataset_gemini_anchors.json",
    output_dir: str | None = None,
    run_name: str | None = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    lr_scheduler_type: str = "linear",
    num_generations: int = 16,
    max_completion_length: int = 2048,
    use_vllm: bool = True,
    vllm_mode: str = "server",
    logging_steps: int = 1,
    save_steps: int = 100,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    eval_steps: int = 50,
    eval_valence: str = "all",
    eval_max_pairs: int | None = None,
    enable_thinking: bool = False,
    length_penalty: float = 0.0,
    markdown_penalty: bool = False,
):
    """GRPO training entry point.

    Default args reproduce the paper recipe (Qwen3-14B, gemini anchors,
    LoRA r=32, lr=1e-4, num_gens=16, eval every 50 steps on PCP).
    """
    model_name = model_path.rstrip("/").split("/")[-1]
    if output_dir is None:
        output_dir = f"outputs/grpo_{model_name}"

    helpfulness_judge_prompt = open(helpfulness_judge_prompt_path).read()
    sentiment_judge_prompt = open(sentiment_judge_prompt_path).read()

    state = PartialState()
    if state.is_main_process:
        print("=" * 60)
        print("GRPO Political Consistency Training")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Judge: {judge_model}")
        print(f"Helpfulness judge: {helpfulness_judge_prompt_path}")
        print(f"Sentiment judge: {sentiment_judge_prompt_path}")
        print(f"Data: {data_path}")
        print(f"Output: {output_dir}")
        print(f"Epochs: {num_train_epochs}")
        print(f"Batch size (per device): {per_device_train_batch_size}")
        print(f"Gradient accumulation: {gradient_accumulation_steps}")
        print(f"Learning rate: {learning_rate}")
        print(f"Num generations: {num_generations}")
        print(f"Max completion length: {max_completion_length}")
        print(f"Use vLLM: {use_vllm} (mode: {vllm_mode})")
        print(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"Eval steps: {eval_steps}")
        print(f"Eval valence: {eval_valence}")
        print(f"Eval max pairs: {eval_max_pairs}")
        print(f"Markdown penalty: {markdown_penalty}")
        print(f"Length penalty: {length_penalty}")
        print(f"Enable thinking: {enable_thinking}")
        print(f"Sentiment reward: bias_factor * helpfulness_factor (multiplicative)")
        print(f"  Bias factor:        {SENTIMENT_BIAS_FACTOR}")
        print(f"  Helpfulness factor: {SENTIMENT_HELPFULNESS_FACTOR}")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "judge_prompt_helpfulness.txt"), "w") as f:
            f.write(helpfulness_judge_prompt)
        with open(os.path.join(output_dir, "judge_prompt_sentiment.txt"), "w") as f:
            f.write(sentiment_judge_prompt)

        reward_config = get_reward_config()
        with open(os.path.join(output_dir, "reward_config.json"), "w") as f:
            json.dump(reward_config, f, indent=2)
        print("Reward config:")
        print(reward_config)

    dataset = load_dataset_from_json(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Initializing reward function (helpfulness + sentiment)...")
    reward_fn = DualJudgeRewardFunction(
        judge_model=judge_model,
        helpfulness_judge_prompt=helpfulness_judge_prompt,
        sentiment_judge_prompt=sentiment_judge_prompt,
        num_generations=num_generations,
        markdown_penalty=markdown_penalty,
        length_penalty=length_penalty,
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    config = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        logging_steps=logging_steps,
        log_level="warning",
        save_steps=save_steps,
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_server_base_url=f"http://localhost:{VLLM_SERVER_PORT}",
        vllm_group_port=VLLM_GROUP_PORT,
        vllm_gpu_memory_utilization=0.5,
        report_to="wandb",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        ddp_timeout=7200,
        use_liger_kernel=True,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )

    print("Initializing GRPO trainer with LoRA...")
    trainer = GRPOTrainer(
        model=model_path,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=peft_config,
    )

    # Override NCCL timeout AFTER process group is initialized.
    import torch.distributed as dist
    if dist.is_initialized():
        pg = dist.group.WORLD
        nccl_backend = pg._get_backend(torch.device("cuda"))
        if hasattr(nccl_backend, '_set_default_timeout'):
            nccl_backend._set_default_timeout(datetime.timedelta(seconds=7200))
            print(f"[NCCL] Set process group timeout to 7200s", flush=True)

    for callback in trainer.callback_handler.callbacks:
        if callback.__class__.__name__ == "ProgressCallback":
            callback.on_log = lambda *args, **kwargs: None
            break

    if eval_steps > 0:
        pcp_eval_callback = PCPEvalCallback(
            judge_model=eval_judge_model,
            tokenizer=tokenizer,
            eval_steps=eval_steps,
            vllm_base_url=f"http://localhost:{VLLM_SERVER_PORT}",
            output_dir=output_dir,
            enable_thinking=enable_thinking,
            valence=eval_valence,
            max_pairs=eval_max_pairs,
        )
        trainer.add_callback(pcp_eval_callback)

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")


if __name__ == "__main__":
    fire.Fire(main)
