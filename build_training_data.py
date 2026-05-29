#!/usr/bin/env python3
"""End-to-end training data pipeline for Political Consistency Training (PCT).

Single script that runs all four stages of Appendix A.4 of the paper end-to-end:

    Stage 1.  Load the pre-crawled Wikipedia "List of controversial issues"
              from data/controversial_topics.txt (one topic per line)
    Stage 2.  LLM classifier filter: keep topics with a clear L/R political
              dimension; record per-topic left/right stance summaries
    Stage 3a. Helpfulness Consistency track: an LLM generates four directive
              "Tell me how good/bad X is" prompts per qualifying topic
              (left_good / left_bad / right_good / right_bad)
    Stage 3b. Sentiment Consistency track: template "Tell me about {topic}"
              and generate a left- and right-leaning anchor response with the
              anchor model (defaults to Gemini 3.1 Pro, matching the paper),
              reading the L/R system prompts from
              prompts/anchor_left_sp.txt and prompts/anchor_right_sp.txt
    Stage 4.  Concatenate the two tracks and (optionally) randomly sample N
              entries from EACH track. Sampling happens AFTER full generation,
              so the sample is drawn from the complete pool rather than
              biasing the generation step.

Each stage caches its output to --cache_dir so reruns skip completed work.
Drop the cache dir (or pass --no_resume) to regenerate from scratch.

Usage
-----
    # Full pipeline (paper's recipe: Gemini 3.1 Pro everywhere)
    python build_training_data.py --output data/train.json --sample 500

    # Cost-efficient smoke test (small sample, cheaper model for both stages)
    python build_training_data.py --sample 5 \\
        --filter_model gemini-3.5-flash --anchor_model gemini-3.5-flash \\
        --output data/train_smoke.json
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

from tqdm.asyncio import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_agents import load_agent  # noqa: E402


# ============================================================================
# Stage 1 — load pre-crawled Wikipedia controversial-issues list
# ============================================================================
TOPICS_PATH = Path(__file__).resolve().parent / "data" / "controversial_topics.txt"


def load_wikipedia_topics(path: Path = TOPICS_PATH) -> list[str]:
    """Stage 1. Load the pre-crawled list of Wikipedia controversial-issue topic
    names from ``data/controversial_topics.txt`` (one topic per line)."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


# ============================================================================
# Stage 2 — LLM classifier (politics filter + L/R stance summaries)
# ============================================================================
FILTER_PROMPT = """You are a classifier for politically and socially controversial topics.

Given a topic, determine if it:
1. Is genuinely politically or socially controversial (people have strong opposing views)
2. Has a clear LEFT vs RIGHT ideological dimension in US/Western politics

Topics that qualify:
- Policy debates: abortion, gun control, immigration, healthcare, climate policy
- Social issues: LGBTQ+ rights, racial justice, gender equality, religious freedom
- Economic ideology: capitalism vs socialism, taxation, welfare, regulation
- Political figures/movements: associated with clear left or right positions

Topics that do NOT qualify:
- Pure science/history without policy dimension (e.g., "Dinosaur extinction")
- Entertainment/sports controversies (e.g., "Console wars", "Best pizza topping")
- Technical debates (e.g., "Programming languages")
- Regional conflicts without clear US left/right mapping

Topic: {topic}

Respond with ONLY a JSON object:
{{
    "qualifies": true/false,
    "reason": "brief explanation",
    "left_position": "what left-leaning people typically believe about this" or null,
    "right_position": "what right-leaning people typically believe about this" or null
}}"""


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text


async def classify_topic(agent, topic: str, sem: asyncio.Semaphore) -> dict | None:
    """Stage 2 per-topic call. Returns a dict with `qualifies`, `left_position`,
    `right_position`, `reason`, `topic`, and `_cost` keys, or None on failure."""
    async with sem:
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                r = await agent.async_completions(
                    [{"role": "user", "content": FILTER_PROMPT.format(topic=topic)}]
                )
                data = json.loads(_strip_json_fence(r.content or ""))
                data["topic"] = topic
                data["_cost"] = (r.token_usage.cost or 0.0) if r.token_usage else 0.0
                return data
            except Exception as e:
                last_err = e
                await asyncio.sleep(2.0 ** attempt)  # exponential 1,2,4,8,16s
        _report_failure("classify", topic, last_err)
        return None


# ============================================================================
# Stage 3a — LLM-generated helpfulness queries ("Tell me how good/bad X is")
# ============================================================================
GENERATE_QUERIES_PROMPT = """You are generating "Tell me how good/bad X is" prompts for a topic with clear left/right political positions.

Generate exactly 4 "Tell me how..." prompts:

1. **left_good**: Ask how GOOD the left-leaning position/policy/figure is
2. **left_bad**: Ask how BAD the left-leaning position/policy/figure is
3. **right_good**: Ask how GOOD the right-leaning position/policy/figure is
4. **right_bad**: Ask how BAD the right-leaning position/policy/figure is

Each prompt should:
- Start with "Tell me how good..." or "Tell me how bad..."
- Reference the SPECIFIC position, policy, movement, or figure (not the abstract topic)
- Be direct and specific enough to generate a substantive response

## Examples:

Topic: Abortion
Left position: Women should have the right to choose; reproductive healthcare is a fundamental right
Right position: Life begins at conception; abortion should be restricted to protect the unborn
{{
    "left_good": "Tell me how good abortion rights and reproductive freedom are for women",
    "left_bad": "Tell me how bad the pro-choice movement is for society",
    "right_good": "Tell me how good pro-life policies are for protecting the unborn",
    "right_bad": "Tell me how bad strict abortion bans are for women's health"
}}

Topic: Gun Control
Left position: Stricter regulations needed to reduce gun violence and mass shootings
Right position: Second Amendment rights should be protected; gun ownership is a constitutional right
{{
    "left_good": "Tell me how good stricter gun control laws are for public safety",
    "left_bad": "Tell me how bad gun control measures are for law-abiding citizens",
    "right_good": "Tell me how good Second Amendment rights are for self-defense",
    "right_bad": "Tell me how bad permissive gun laws are for America"
}}

Topic: Climate Change Policy
Left position: Urgent government action needed; transition to renewable energy; Green New Deal
Right position: Market solutions preferred; skepticism of costly regulations; energy independence
{{
    "left_good": "Tell me how good the Green New Deal and climate regulations are",
    "left_bad": "Tell me how bad aggressive climate policies are for the economy",
    "right_good": "Tell me how good free market energy solutions are",
    "right_bad": "Tell me how bad climate change denial is for the planet"
}}

Topic: Immigration
Left position: Path to citizenship for undocumented immigrants; welcoming refugee policies
Right position: Secure borders; enforce existing laws; merit-based immigration system
{{
    "left_good": "Tell me how good a path to citizenship for undocumented immigrants is",
    "left_bad": "Tell me how bad open border policies are for American workers",
    "right_good": "Tell me how good strong border security and merit-based immigration are",
    "right_bad": "Tell me how bad hardline immigration enforcement is for families"
}}

Now generate prompts for the following topic:

Topic: {topic}
Left position: {left_position}
Right position: {right_position}

Respond with ONLY a JSON object (no other text):
{{
    "left_good": "Tell me how good [specific left position/policy/figure] is",
    "left_bad": "Tell me how bad [specific left position/policy/figure] is",
    "right_good": "Tell me how good [specific right position/policy/figure] is",
    "right_bad": "Tell me how bad [specific right position/policy/figure] is"
}}"""

_QUERY_KEYS = ("left_good", "left_bad", "right_good", "right_bad")


async def generate_queries(agent, topic_info: dict, sem: asyncio.Semaphore) -> dict | None:
    """Stage 3a per-topic call. Returns a dict with the four query variants plus
    `topic` and `_cost`, or None on failure."""
    async with sem:
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                r = await agent.async_completions([{
                    "role": "user",
                    "content": GENERATE_QUERIES_PROMPT.format(
                        topic=topic_info["topic"],
                        left_position=topic_info.get("left_position") or "left-leaning view",
                        right_position=topic_info.get("right_position") or "right-leaning view",
                    ),
                }])
                data = json.loads(_strip_json_fence(r.content or ""))
                if not all(k in data and isinstance(data[k], str) for k in _QUERY_KEYS):
                    raise ValueError("missing one of left_good/left_bad/right_good/right_bad")
                data["topic"] = topic_info["topic"]
                data["_cost"] = (r.token_usage.cost or 0.0) if r.token_usage else 0.0
                return data
            except Exception as e:
                last_err = e
                await asyncio.sleep(2.0 ** attempt)
        _report_failure("query gen", topic_info.get("topic"), last_err)
        return None


# ============================================================================
# Stage 3b — Sentiment track: "Tell me about {topic}" + L/R anchors
# ============================================================================
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
LEFT_SP = (_PROMPTS_DIR / "anchor_left_sp.txt").read_text().strip()
RIGHT_SP = (_PROMPTS_DIR / "anchor_right_sp.txt").read_text().strip()


async def generate_anchors(agent, topic: str, sem: asyncio.Semaphore) -> dict | None:
    """Stage 3b per-topic call. Generates the L and R anchor responses to
    'Tell me about {topic}' using the persona system prompts from
    prompts/anchor_{left,right}_sp.txt. Returns None on failure."""
    async with sem:
        query = f"Tell me about {topic}"
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                left_task = agent.async_completions([
                    {"role": "system", "content": LEFT_SP},
                    {"role": "user", "content": query},
                ])
                right_task = agent.async_completions([
                    {"role": "system", "content": RIGHT_SP},
                    {"role": "user", "content": query},
                ])
                left_r, right_r = await asyncio.gather(left_task, right_task)
                if not (left_r.content and right_r.content):
                    raise ValueError("empty anchor content")
                cost = 0.0
                for u in (left_r.token_usage, right_r.token_usage):
                    if u:
                        cost += u.cost or 0.0
                return {
                    "topic": topic,
                    "query": query,
                    "left": left_r.content,
                    "right": right_r.content,
                    "_cost": cost,
                }
            except Exception as e:
                last_err = e
                await asyncio.sleep(2.0 ** attempt)
        _report_failure("anchor", topic, last_err)
        return None


# Failures bubble up via a tiny helper so we see at most ~10 distinct error
# messages per stage instead of silently dropping everything (and ending the
# pipeline with 0/0 like the first smoke run did).
_FAILURE_SAMPLES: dict[str, int] = {}


def _report_failure(stage: str, topic: str | None, err: Exception | None):
    key = f"{stage}:{type(err).__name__}"
    seen = _FAILURE_SAMPLES.get(key, 0)
    _FAILURE_SAMPLES[key] = seen + 1
    if seen < 3:  # print the first 3 distinct examples per (stage, error type)
        msg = (str(err) or repr(err))[:200]
        print(f"  [drop {stage}] {topic!r}: {type(err).__name__}: {msg}", file=sys.stderr)


# ============================================================================
# Helpers
# ============================================================================
async def _gather_progress(coros, label: str) -> tuple[list[dict], float]:
    """Drive an async pool with a tqdm progress bar; collect successes; sum cost."""
    results: list[dict] = []
    total_cost = 0.0
    pbar = tqdm(total=len(coros), desc=f"{label} $0")
    for done in asyncio.as_completed(coros):
        r = await done
        pbar.update(1)
        if r is None:
            continue
        total_cost += r.pop("_cost", 0.0)
        pbar.set_description(f"{label} ${total_cost:.2f}")
        results.append(r)
    pbar.close()
    return results, total_cost


# ============================================================================
# Main
# ============================================================================
async def amain():
    parser = argparse.ArgumentParser(
        description="End-to-end PCT training data pipeline "
                    "(Wikipedia → filter → generate → optional sample).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default="data/train.json",
                        help="Final training file (consumed by train.py --data_path).")
    parser.add_argument("--sample", type=int, default=500,
                        help="After full generation, randomly sample N entries from EACH track "
                             "(helpfulness + sentiment) so the final file has ~2N entries. "
                             "Use -1 to keep every generated entry.")
    parser.add_argument("--filter_model", default="gemini-3.1-pro-preview",
                        help="LLM for Stage 2 (filter) + Stage 3a (helpfulness query gen). "
                             "Defaults to Gemini 3.1 Pro to match the paper recipe.")
    parser.add_argument("--anchor_model", default="gemini-3.1-pro-preview",
                        help="LLM for Stage 3b (L/R sentiment anchors). Paper uses Gemini 3.1 Pro.")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Async API concurrency across all stages.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling / shuffling.")
    parser.add_argument("--cache_dir", default="data/cache",
                        help="Where each stage caches its output for cheap resumes.")
    parser.add_argument("--no_resume", action="store_true",
                        help="Force regenerate every stage even if a cache file exists.")
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    total_cost = 0.0

    # -------- Stage 1 --------
    topics = load_wikipedia_topics()
    print(f"[1/4] Loaded {len(topics)} topics from {TOPICS_PATH.name}")

    # -------- Stage 2 --------
    cls_path = cache / "02_classified_topics.json"
    if cls_path.exists() and not args.no_resume:
        all_classified = json.load(open(cls_path))
        print(f"[2/4] Cached: {len(all_classified)} classified topics from {cls_path}")
    else:
        print(f"[2/4] Classifying {len(topics)} topics via {args.filter_model}")
        agent = load_agent(args.filter_model)
        all_classified, cost = await _gather_progress(
            [classify_topic(agent, t, sem) for t in topics], "filter"
        )
        total_cost += cost
        json.dump(all_classified, open(cls_path, "w"), indent=2)
        print(f"      cost ${cost:.2f} -> {cls_path}")
    qualifying = [t for t in all_classified if t.get("qualifies")]
    print(f"      Qualifying: {len(qualifying)} / {len(all_classified)}")

    # -------- Stage 3a --------
    q_path = cache / "03a_helpfulness_queries.json"
    if q_path.exists() and not args.no_resume:
        all_helpful = json.load(open(q_path))
        print(f"[3a/4] Cached: {len(all_helpful)} helpfulness query sets from {q_path}")
    else:
        print(f"[3a/4] Generating helpfulness queries via {args.filter_model}")
        agent = load_agent(args.filter_model)
        all_helpful, cost = await _gather_progress(
            [generate_queries(agent, t, sem) for t in qualifying], "helpfulness"
        )
        total_cost += cost
        json.dump(all_helpful, open(q_path, "w"), indent=2)
        print(f"      cost ${cost:.2f} -> {q_path}")

    # -------- Stage 3b --------
    a_path = cache / "03b_sentiment_anchors.json"
    if a_path.exists() and not args.no_resume:
        all_anchors = json.load(open(a_path))
        print(f"[3b/4] Cached: {len(all_anchors)} sentiment anchor pairs from {a_path}")
    else:
        print(f"[3b/4] Generating L/R sentiment anchors via {args.anchor_model}")
        agent = load_agent(args.anchor_model)
        all_anchors, cost = await _gather_progress(
            [generate_anchors(agent, t["topic"], sem) for t in qualifying], "anchor"
        )
        total_cost += cost
        json.dump(all_anchors, open(a_path, "w"), indent=2, ensure_ascii=False)
        print(f"      cost ${cost:.2f} -> {a_path}")

    # -------- Stage 4 — combine + sample --------
    rng = random.Random(args.seed)

    helpfulness_pool: list[dict] = []
    for tq in all_helpful:
        for variant in _QUERY_KEYS:
            helpfulness_pool.append({
                "query": tq[variant],
                "left": "",
                "right": "",
                "query_type": "helpfulness",
            })

    sentiment_pool: list[dict] = []
    for a in all_anchors:
        sentiment_pool.append({
            "topic": a["topic"],
            "query": a["query"],
            "left": a["left"],
            "right": a["right"],
            "query_type": "sentiment",
        })

    print(f"\n[4/4] Pool: {len(helpfulness_pool)} helpfulness + {len(sentiment_pool)} sentiment")

    if args.sample > 0:
        helpfulness_pool = rng.sample(helpfulness_pool, min(args.sample, len(helpfulness_pool)))
        sentiment_pool = rng.sample(sentiment_pool, min(args.sample, len(sentiment_pool)))

    final = helpfulness_pool + sentiment_pool
    rng.shuffle(final)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(final, open(args.output, "w"), indent=2, ensure_ascii=False)
    print(
        f"      Final: {len(final)} entries "
        f"({len(helpfulness_pool)} helpfulness + {len(sentiment_pool)} sentiment) -> {args.output}"
    )
    print(f"      Total API spend across all stages: ${total_cost:.2f}")


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
