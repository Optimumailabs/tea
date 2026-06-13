#!/usr/bin/env python3
"""Token Efficiency Agent, Phase 0 scorer.

Computes the composite score S(P) defined in `product/token_efficiency_agent.md`
section 3.6, plus a token breakdown, cost estimate, and ranked optimisation
suggestions. Pure measurement: this script does NOT rewrite the prompt.

Usage:
    python score.py \
        --prompt-file /path/prompt.txt \
        [--context-file /path/context.txt] \
        [--completion-file /path/completion.txt] \
        [--quality 0.78] \
        [--model gpt-4o] \
        [--alpha 0.30 --beta 0.40 --gamma 0.20 --delta 0.10]

Emits JSON on stdout. Designed to be parsed by the calling skill, not read by
humans directly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str) -> int:
    """Try tiktoken for OpenAI-family models, else whitespace-approximate.

    The approximation is rough; it is here so the skill still works in
    environments without tiktoken, and the assumption flag in the output
    tells the caller what happened."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # ~1.3 tokens per whitespace-word is the rule of thumb for English.
        words = re.findall(r"\S+", text)
        return max(1, int(round(len(words) * 1.3)))


# ---------------------------------------------------------------------------
# Cost model: per-million-token rates, split into prefill and decode.
# ---------------------------------------------------------------------------
PRICING_USD_PER_M = {
    # Source: provider public pricing as of 2026-Q2. Update before shipping.
    "gpt-4o":          {"prefill":  2.50, "decode": 10.00},
    "gpt-4o-mini":     {"prefill":  0.15, "decode":  0.60},
    "claude-opus-4-7": {"prefill": 15.00, "decode": 75.00},
    "claude-sonnet-4-6":{"prefill":  3.00, "decode": 15.00},
    "claude-haiku-4-5":{"prefill":  1.00, "decode":  5.00},
    # Self-hosted approximation (vLLM/SGLang on an H100 at $2/hr).
    "self-hosted":     {"prefill":  0.05, "decode":  0.20},
}


def estimate_cost(model: str, n_prompt: int, n_completion: int) -> float:
    rates = PRICING_USD_PER_M.get(model, PRICING_USD_PER_M["gpt-4o"])
    return (n_prompt * rates["prefill"] + n_completion * rates["decode"]) / 1_000_000


# ---------------------------------------------------------------------------
# Section detection. Phase 0 is allowed to be heuristic.
# ---------------------------------------------------------------------------
SECTION_PATTERNS = {
    "system":   re.compile(r"^\s*(system|<system>|\[system\])", re.I | re.M),
    "examples": re.compile(r"^\s*(examples?:?|few[-_ ]shot|here are examples)", re.I | re.M),
    "context":  re.compile(r"^\s*(context:?|retrieved|knowledge|documents?:?|sources?:?)", re.I | re.M),
    "user":     re.compile(r"^\s*(user|<user>|\[user\]|question:?|query:?)", re.I | re.M),
}


def section_breakdown(prompt: str, model: str, explicit_context: str | None) -> tuple[dict[str, int], dict[str, str]]:
    """Best-effort split. Returns (token_counts, raw_text_per_section).

    If we can't detect section markers, everything lives in 'user'. When the
    caller passed an explicit `--context-file`, treat that as the authoritative
    context section."""
    sections = {"system": "", "examples": "", "context": "", "user": prompt}

    if explicit_context:
        sections["context"] = explicit_context

    lines = prompt.splitlines()
    if lines:
        current = "user"
        buckets = {k: [] for k in sections}
        for line in lines:
            matched = None
            for name, pat in SECTION_PATTERNS.items():
                if pat.match(line):
                    matched = name
                    break
            if matched is not None:
                current = matched
                continue
            buckets[current].append(line)
        if any(buckets[k] for k in ("system", "examples", "context")):
            for k in buckets:
                sections[k] = "\n".join(buckets[k]).strip()
            if explicit_context:
                sections["context"] = explicit_context

    counts = {k: count_tokens(v, model) for k, v in sections.items()}
    return counts, sections


# ---------------------------------------------------------------------------
# Used-context approximation
# ---------------------------------------------------------------------------
def approx_used_fraction(context: str, user_query: str, model: str) -> float:
    """Cheap stand-in for real attention. Splits context into chunks of ~3
    sentences each, scores each chunk by Jaccard similarity to the user query
    (after stopword removal), and reports the fraction of context-tokens in
    chunks above threshold 0.10.

    This is a Phase-0 placeholder. Production uses real attention or
    ablation-based KL (see brief §3.6). Do not present this as ground truth."""
    if not context.strip() or not user_query.strip():
        return 0.0

    stopwords = {
        "a","an","the","and","or","but","of","to","in","on","at","for","with",
        "is","are","was","were","be","been","being","have","has","had","do",
        "does","did","this","that","these","those","it","its","by","as","from",
        "i","you","he","she","we","they","me","him","her","us","them","my",
    }

    def toks(s):
        return {w.lower() for w in re.findall(r"\w+", s) if w.lower() not in stopwords}

    q_toks = toks(user_query)
    if not q_toks:
        return 0.0

    # Chunk into ~3-sentence windows.
    sentences = re.split(r"(?<=[\.\!\?])\s+", context)
    chunks = ["\n".join(sentences[i:i+3]) for i in range(0, len(sentences), 3) if sentences[i:i+3]]

    # Threshold scales down for short queries (Jaccard is sensitive to query size).
    threshold = 0.10 if len(q_toks) >= 10 else 0.04
    used_tokens = 0
    for chunk in chunks:
        c_toks = toks(chunk)
        if not c_toks:
            continue
        jaccard = len(q_toks & c_toks) / len(q_toks | c_toks)
        # Alternative score: overlap as fraction of query tokens (recall-weighted).
        q_overlap = len(q_toks & c_toks) / len(q_toks)
        if jaccard >= threshold or q_overlap >= 0.5:
            used_tokens += count_tokens(chunk, model)
    return min(1.0, used_tokens / max(1, count_tokens(context, model)))


# ---------------------------------------------------------------------------
# Optimization suggestions
# ---------------------------------------------------------------------------
@dataclass
class Suggestion:
    rule: str
    detail: str
    estimated_token_reduction: int


def build_suggestions(
    tokens: dict[str, int],
    util: float,
    prompt: str,
    context: str,
    user_query: str,
    model: str,
) -> list[Suggestion]:
    out: list[Suggestion] = []

    # Rule: drop low-utility context.
    if tokens["context"] > 0 and util < 0.5:
        drop = int(tokens["context"] * (1.0 - max(util, 0.05)))
        out.append(Suggestion(
            rule="drop",
            detail=f"Context utilization is about {util:.0%}. Roughly "
                   f"{drop:,} of {tokens['context']:,} context tokens "
                   f"look unused; drop the lowest-scoring chunks first.",
            estimated_token_reduction=drop,
        ))

    # Rule: dedupe near-duplicate paragraphs in context.
    if context:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", context) if p.strip()]
        seen = set()
        dup_tokens = 0
        for p in paragraphs:
            key = re.sub(r"\W+", "", p.lower())[:200]
            if key in seen:
                dup_tokens += count_tokens(p, model)
            else:
                seen.add(key)
        if dup_tokens > 0:
            out.append(Suggestion(
                rule="deduplicate",
                detail=f"{dup_tokens:,} tokens appear in duplicate paragraphs "
                       f"in the context. Drop one copy each.",
                estimated_token_reduction=dup_tokens,
            ))

    # Rule: reorder. Flag if the user query precedes most of the context.
    # Most LLMs over-weight recency; putting the query last is usually better.
    if user_query and context:
        q_pos = prompt.find(user_query.strip()[:80])
        c_pos = prompt.find(context.strip()[:80])
        if q_pos != -1 and c_pos != -1 and q_pos < c_pos:
            out.append(Suggestion(
                rule="reorder",
                detail="The user query appears before the retrieved context. "
                       "Most LLMs give the strongest attention to the end of "
                       "the prompt, so moving the query after the context, or "
                       "placing the highest-utility chunk next to the query, "
                       "tends to improve answer adherence at no token cost.",
                estimated_token_reduction=0,
            ))

    # Rule: compress. Fires when the few-shot examples are large relative
    # to the user query.
    if tokens["examples"] > 0 and tokens["examples"] > 3 * max(tokens["user"], 1):
        cut = int(tokens["examples"] * 0.5)
        out.append(Suggestion(
            rule="compress",
            detail=f"Few-shot examples take {tokens['examples']:,} tokens, "
                   f"which is roughly {tokens['examples'] // max(tokens['user'], 1)} "
                   f"times the size of the user query. Replace verbose examples "
                   f"with summarised forms, or drop to zero-shot if model "
                   f"confidence allows.",
            estimated_token_reduction=cut,
        ))

    # Sort by estimated savings, descending.
    out.sort(key=lambda s: s.estimated_token_reduction, reverse=True)
    return out[:5]


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------
def compute_score(
    tokens: dict[str, int],
    n_completion: int,
    quality: float,
    util: float,
    cost: float,
    cost_max_for_normalization: float,
    weights: dict[str, float],
) -> dict[str, float]:
    total_in = sum(tokens.values())
    denom = total_in + n_completion
    token_eff = (quality * n_completion) / denom if denom > 0 else 0.0
    cost_norm = min(1.0, cost / cost_max_for_normalization) if cost_max_for_normalization > 0 else 0.0
    util_penalty = max(0.0, 1.0 - util)

    contrib = {
        "TokenEff":     weights["alpha"] * token_eff,
        "Quality":      weights["beta"]  * quality,
        "CostPenalty": -weights["gamma"] * cost_norm,
        "UtilPenalty": -weights["delta"] * util_penalty,
    }
    s = sum(contrib.values())
    return {
        "token_eff":    round(token_eff, 4),
        "quality":      round(quality, 4),
        "cost_norm":    round(cost_norm, 4),
        "util_penalty": round(util_penalty, 4),
        "contributions": {k: round(v, 4) for k, v in contrib.items()},
        "S":            round(s, 4),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Token Efficiency Agent: measurement-only scorer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prompt-file",      required=True, type=Path)
    p.add_argument("--context-file",     type=Path, default=None,
                   help="If supplied, used as the authoritative context section.")
    p.add_argument("--user-query",       type=str, default=None,
                   help="The actual user query, for similarity-based 'used' approx. "
                        "If absent, derived from the last 200 chars of the prompt.")
    p.add_argument("--completion-file",  type=Path, default=None)
    p.add_argument("--completion-tokens",type=int, default=None,
                   help="Skip completion-file in favour of an explicit length.")
    p.add_argument("--quality",          type=float, default=0.75,
                   help="Quality proxy Q in [0,1]. Default 0.75 with a flag in the output.")
    p.add_argument("--quality-supplied", action="store_true",
                   help="Internal flag set by SKILL.md to suppress the placeholder warning.")
    p.add_argument("--model",            default="gpt-4o")
    p.add_argument("--alpha", type=float, default=0.30)
    p.add_argument("--beta",  type=float, default=0.40)
    p.add_argument("--gamma", type=float, default=0.20)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--cost-max", type=float, default=0.05,
                   help="Normalization ceiling for the cost penalty. Default $0.05/call.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.prompt_file.exists():
        print(json.dumps({"error": f"prompt file not found: {args.prompt_file}"}))
        return 2

    prompt = args.prompt_file.read_text(encoding="utf-8")
    context = args.context_file.read_text(encoding="utf-8") if args.context_file else ""
    completion = args.completion_file.read_text(encoding="utf-8") if args.completion_file else ""

    # Token counts and raw section texts.
    tokens, raw = section_breakdown(prompt, args.model, context if args.context_file else None)
    n_prompt = sum(tokens.values())
    n_completion = args.completion_tokens if args.completion_tokens is not None \
        else (count_tokens(completion, args.model) if completion else 200)
    completion_estimated = args.completion_tokens is None and not completion

    # User query for similarity scoring.
    user_query = args.user_query or raw.get("user") or prompt[-400:]
    context_text = raw.get("context", "") or context
    util = approx_used_fraction(context_text, user_query, args.model) \
        if (context_text or tokens["context"] > 0) else 1.0

    # Cost.
    cost = estimate_cost(args.model, n_prompt, n_completion)

    # Composite score.
    weights = {"alpha": args.alpha, "beta": args.beta,
               "gamma": args.gamma, "delta": args.delta}
    score = compute_score(tokens, n_completion, args.quality, util,
                          cost, args.cost_max, weights)

    # Suggestions.
    suggestions = build_suggestions(tokens, util, prompt, context_text, user_query, args.model)

    # Assumption flags so the caller sees which defaults kicked in.
    assumptions = []
    if args.quality == 0.75 and not args.quality_supplied:
        assumptions.append("quality not supplied; assumed Q = 0.75")
    if completion_estimated:
        assumptions.append("no completion supplied; assumed 200 output tokens")
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        assumptions.append("tiktoken not installed; token counts use a ~1.3x whitespace-word approximation")
    if context and not args.context_file:
        assumptions.append("context detected by heuristic; pass --context-file for precision")
    assumptions.append("'used context' is a Jaccard-similarity placeholder; see brief section 3.8 for the real attention path")

    out = {
        "model": args.model,
        "tokens": {
            **tokens,
            "total_prompt": n_prompt,
            "completion":   n_completion,
        },
        "score": {
            **score,
            "weights": weights,
        },
        "context": {
            "util": round(util, 4),
            "estimated_used_tokens": int(util * tokens["context"]) if tokens["context"] else 0,
        },
        "cost": {
            "per_request_usd": round(cost, 6),
            "rates_per_M":     PRICING_USD_PER_M.get(args.model, PRICING_USD_PER_M["gpt-4o"]),
        },
        "suggestions": [asdict(s) for s in suggestions],
        "assumptions": assumptions,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
