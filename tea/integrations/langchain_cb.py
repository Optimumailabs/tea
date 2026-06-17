"""LangChain adapter.

LangChain represents a prompt as a list of BaseMessage objects. This adapter
gives you two entry points.

1. A Runnable you can place in front of a chat model in LCEL::

       from langchain_openai import ChatOpenAI
       from tea.integrations.langchain_cb import TEAOptimizer

       model = ChatOpenAI(model="gpt-4o")
       chain = TEAOptimizer(model_name="gpt-4o") | model
       chain.invoke(messages)

2. A plain function to optimise a list of LangChain messages directly::

       from tea.integrations.langchain_cb import optimize_lc_messages
       optimized, report = optimize_lc_messages(messages, model_name="gpt-4o")

The adapter converts LangChain messages to the role/content dicts the TEA core
uses, optimises them, and converts back. It imports LangChain lazily so TEA
has no hard dependency on it.
"""

from __future__ import annotations

from typing import Any, Optional

from ..optimizer import Compressor, OptimizeResult, optimize_messages

# LangChain role names map to the simple roles the TEA core expects.
_LC_TYPE_TO_ROLE = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
    "function": "tool",
}


def _lc_to_dicts(messages: list) -> list[dict]:
    out = []
    for m in messages:
        # BaseMessage has .type and .content; fall back gracefully.
        role = _LC_TYPE_TO_ROLE.get(getattr(m, "type", None), "user")
        out.append({"role": role, "content": getattr(m, "content", str(m))})
    return out


def _apply_to_lc(messages: list, optimized: list[dict]) -> list:
    """Write optimised content back into copies of the original LangChain
    messages so message subtypes and metadata are preserved."""
    new = []
    for orig, opt in zip(messages, optimized):
        try:
            clone = orig.model_copy(update={"content": opt["content"]})
        except AttributeError:
            # Older LangChain: copy via constructor.
            clone = type(orig)(content=opt["content"])
        new.append(clone)
    return new


def optimize_lc_messages(
    messages: list,
    *,
    model_name: str = "gpt-4o",
    enable: Optional[set[str]] = None,
    compressor: Optional[Compressor] = None,
    log=None,
    **opt_kwargs: Any,
) -> tuple[list, OptimizeResult]:
    """Optimise a list of LangChain messages. Returns (new_messages, report).
    Passing ``log`` logs the call tagged with source "langchain"."""
    from .. import optimize as _optimize

    dicts = _lc_to_dicts(messages)
    result = _optimize(
        dicts, model=model_name, enable=enable, compressor=compressor,
        log=log, source="langchain", **opt_kwargs
    )
    new_messages = _apply_to_lc(messages, result.optimized)
    return new_messages, result


def TEAOptimizer(
    *,
    model_name: str = "gpt-4o",
    enable: Optional[set[str]] = None,
    compressor: Optional[Compressor] = None,
    log=None,
    on_report=None,
    **opt_kwargs: Any,
):
    """Build a LangChain Runnable that optimises messages as they pass through.

    Implemented with RunnableLambda so it slots into any LCEL chain with the
    ``|`` operator. Raises a clear error if LangChain is not installed.
    """
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError as e:
        raise ImportError(
            "TEAOptimizer needs langchain-core. Install it with "
            "`pip install langchain-core`."
        ) from e

    def _run(messages):
        new_messages, report = optimize_lc_messages(
            messages, model_name=model_name, enable=enable,
            compressor=compressor, log=log, **opt_kwargs
        )
        if on_report is not None:
            on_report(report)
        return new_messages

    return RunnableLambda(_run)
