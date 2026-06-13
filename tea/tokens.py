"""Token counting and cost estimation, shared across the TEA package.

Token counting uses tiktoken when available and falls back to a whitespace
approximation otherwise. The fallback is rough but keeps the package usable in
environments without tiktoken installed.
"""

from __future__ import annotations

import re
from functools import lru_cache


# Per-million-token rates, split into prefill (input) and decode (output).
# Update before any production billing claim. Values reflect public pricing
# at the time of writing and are intentionally easy to override per call.
PRICING_USD_PER_M = {
    "gpt-4o":            {"prefill":  2.50, "decode": 10.00},
    "gpt-4o-mini":       {"prefill":  0.15, "decode":  0.60},
    "gpt-4.1":           {"prefill":  2.00, "decode":  8.00},
    "claude-opus-4-8":   {"prefill": 15.00, "decode": 75.00},
    "claude-sonnet-4-6": {"prefill":  3.00, "decode": 15.00},
    "claude-haiku-4-5":  {"prefill":  1.00, "decode":  5.00},
    # Self-hosted approximation (vLLM / SGLang on an H100 at roughly $2/hr).
    "self-hosted":       {"prefill":  0.05, "decode":  0.20},
}

_DEFAULT_MODEL = "gpt-4o"


@lru_cache(maxsize=8)
def _encoding_for(model: str):
    """Return a tiktoken encoding, or None if tiktoken is unavailable."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except (KeyError, Exception):
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, model: str = _DEFAULT_MODEL) -> int:
    """Count tokens in `text` for `model`.

    Uses tiktoken if present. Falls back to a 1.3 tokens-per-word estimate,
    which is a reasonable rule of thumb for English. Anthropic models do not
    ship a public tokenizer, so they also use the cl100k fallback, which
    over- or under-counts by a few per cent. That is acceptable for relative
    comparisons (before vs after), which is what the optimizer cares about.
    """
    if not text:
        return 0
    enc = _encoding_for(model)
    if enc is not None:
        return len(enc.encode(text))
    words = re.findall(r"\S+", text)
    return max(1, int(round(len(words) * 1.3)))


def tokenizer_is_exact(model: str) -> bool:
    """True when token counts come from a real tokenizer, not the fallback."""
    return _encoding_for(model) is not None


def estimate_cost(model: str, n_prompt: int, n_completion: int) -> float:
    """Estimated dollar cost for a single call in USD."""
    rates = PRICING_USD_PER_M.get(model, PRICING_USD_PER_M[_DEFAULT_MODEL])
    return (n_prompt * rates["prefill"] + n_completion * rates["decode"]) / 1_000_000
