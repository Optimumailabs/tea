"""OpenAI SDK adapter.

Two ways to use it.

1. Optimise the kwargs before you call the SDK yourself::

       from openai import OpenAI
       from tea.integrations.openai_wrap import optimize_openai_kwargs

       client = OpenAI()
       kwargs = {"model": "gpt-4o", "messages": [...]}
       kwargs, report = optimize_openai_kwargs(kwargs)
       resp = client.chat.completions.create(**kwargs)

2. Wrap the client so every ``chat.completions.create`` call is optimised
   transparently::

       from tea.integrations.openai_wrap import wrap_openai
       client = wrap_openai(OpenAI())
       resp = client.chat.completions.create(model="gpt-4o", messages=[...])

The adapter only rewrites the ``messages`` field. It never changes ``model``,
``tools``, ``temperature``, or any other parameter.
"""

from __future__ import annotations

from typing import Any, Optional

from ..optimizer import Compressor, OptimizeResult, optimize_messages


def optimize_openai_kwargs(
    kwargs: dict,
    *,
    enable: Optional[set[str]] = None,
    compressor: Optional[Compressor] = None,
    **opt_kwargs: Any,
) -> tuple[dict, Optional[OptimizeResult]]:
    """Return a copy of ``kwargs`` with optimised ``messages`` plus the report.

    If there are no messages, returns the kwargs unchanged and a None report.
    """
    messages = kwargs.get("messages")
    if not messages:
        return kwargs, None
    model = kwargs.get("model", "gpt-4o")
    result = optimize_messages(
        messages, model=model, enable=enable, compressor=compressor, **opt_kwargs
    )
    new_kwargs = dict(kwargs)
    new_kwargs["messages"] = result.optimized
    return new_kwargs, result


def wrap_openai(
    client: Any,
    *,
    enable: Optional[set[str]] = None,
    compressor: Optional[Compressor] = None,
    on_report=None,
    **opt_kwargs: Any,
) -> Any:
    """Monkey-patch ``client.chat.completions.create`` to optimise messages.

    ``on_report`` is an optional callable invoked with each OptimizeResult, so
    you can log savings. The original method is preserved and still called.
    Returns the same client object for chaining.
    """
    completions = client.chat.completions
    original_create = completions.create

    def patched_create(*args, **kwargs):
        kwargs, report = optimize_openai_kwargs(
            kwargs, enable=enable, compressor=compressor, **opt_kwargs
        )
        if report is not None and on_report is not None:
            on_report(report)
        return original_create(*args, **kwargs)

    completions.create = patched_create  # type: ignore[attr-defined]
    return client
