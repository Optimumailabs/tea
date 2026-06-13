"""Token Efficiency Agent (TEA).

A measurement and optimisation layer for LLM prompts. Cuts wasted tokens before
a request reaches the provider, with deterministic transforms by default and an
optional LLM compressor for deeper savings.

Quick start
-----------

Raw text::

    import tea
    result = tea.optimize("...long prompt...", query="the user question")
    print(result.optimized)
    print(result.summary())

Chat messages::

    messages = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ]
    result = tea.optimize_messages(messages, model="gpt-4o")
    cheaper = result.optimized

Scoring without rewriting::

    s = tea.score("...prompt...", query="...", quality=0.8)
    print(s["score"]["S"])

Framework integrations live in ``tea.integrations`` (OpenAI, Anthropic,
LangChain, CrewAI, AutoGen). Import the one you need; none are imported by
default so TEA has no hard dependency on any framework.
"""

from __future__ import annotations

from typing import Optional

from .optimizer import (
    Compressor,
    OptimizeResult,
    TransformResult,
    optimize_messages,
    optimize_text,
)
from .tokens import count_tokens, estimate_cost, tokenizer_is_exact

__version__ = "0.1.0"

# Default transform set: the safe, deterministic ones.
SAFE_TRANSFORMS = {"whitespace", "dedupe", "few_shot"}
# Everything, including opt-in context dropping. Compression still needs a
# compressor callable to actually run.
AGGRESSIVE_TRANSFORMS = {"whitespace", "dedupe", "few_shot", "drop_context", "compress"}


def optimize(prompt, *, query: Optional[str] = None, **kwargs) -> OptimizeResult:
    """Optimise a prompt. Accepts a string or a list of chat-message dicts.

    Keyword arguments are forwarded to ``optimize_text`` or
    ``optimize_messages`` depending on the input type. See those functions for
    the full parameter list.
    """
    if isinstance(prompt, str):
        return optimize_text(prompt, query=query, **kwargs)
    if isinstance(prompt, list):
        return optimize_messages(prompt, **kwargs)
    raise TypeError(
        f"optimize() expects a str or a list of message dicts, got {type(prompt).__name__}"
    )


def score(prompt: str, *, query: Optional[str] = None, quality: float = 0.75,
          model: str = "gpt-4o", context: Optional[str] = None) -> dict:
    """Score a prompt without rewriting it. Thin wrapper over the scorer in
    ``scripts/score.py`` logic, re-implemented here so the package is
    self-contained. Returns the same shape as the score.py JSON output's
    ``score`` block, plus token counts and cost."""
    from .tokens import count_tokens, estimate_cost

    n_prompt = count_tokens(prompt, model)
    n_completion = 200  # placeholder when no completion is known
    token_eff = (quality * n_completion) / (n_prompt + n_completion) if (n_prompt + n_completion) else 0.0

    # Reuse the optimiser's relevance scorer for utilisation.
    from .optimizer import _content_tokens, _jaccard
    util = 1.0
    if context and query:
        q = _content_tokens(query)
        paras = [p for p in context.split("\n\n") if p.strip()]
        used = 0
        total = 0
        for p in paras:
            c = _content_tokens(p)
            total += count_tokens(p, model)
            overlap = (len(q & c) / len(q)) if q else 0.0
            if max(overlap, _jaccard(q, c)) >= 0.06:
                used += count_tokens(p, model)
        util = (used / total) if total else 1.0

    cost = estimate_cost(model, n_prompt, n_completion)
    cost_max = 0.02
    weights = {"alpha": 0.30, "beta": 0.40, "gamma": 0.20, "delta": 0.10}
    cost_norm = min(1.0, cost / cost_max) if cost_max else 0.0
    util_pen = max(0.0, 1.0 - util)
    s = (weights["alpha"] * token_eff + weights["beta"] * quality
         - weights["gamma"] * cost_norm - weights["delta"] * util_pen)

    return {
        "model": model,
        "tokens": {"total_prompt": n_prompt, "completion": n_completion},
        "cost": {"per_request_usd": round(cost, 6)},
        "score": {
            "token_eff": round(token_eff, 4),
            "quality": round(quality, 4),
            "cost_norm": round(cost_norm, 4),
            "util": round(util, 4),
            "S": round(s, 4),
            "weights": weights,
        },
    }


__all__ = [
    "optimize",
    "optimize_text",
    "optimize_messages",
    "score",
    "OptimizeResult",
    "TransformResult",
    "Compressor",
    "count_tokens",
    "estimate_cost",
    "tokenizer_is_exact",
    "SAFE_TRANSFORMS",
    "AGGRESSIVE_TRANSFORMS",
    "__version__",
]
