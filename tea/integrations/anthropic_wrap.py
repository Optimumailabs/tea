"""Anthropic SDK adapter.

The Anthropic Messages API separates the ``system`` prompt from the
``messages`` list, so this adapter optimises both.

Optimise kwargs before calling the SDK::

    from anthropic import Anthropic
    from tea.integrations.anthropic_wrap import optimize_anthropic_kwargs

    client = Anthropic()
    kwargs = {"model": "claude-sonnet-4-6", "max_tokens": 1024,
              "system": "...", "messages": [...]}
    kwargs, report = optimize_anthropic_kwargs(kwargs)
    resp = client.messages.create(**kwargs)

Or wrap the client::

    from tea.integrations.anthropic_wrap import wrap_anthropic
    client = wrap_anthropic(Anthropic())
"""

from __future__ import annotations

from typing import Any, Optional

from ..optimizer import Compressor, OptimizeResult, optimize_messages, optimize_text


def optimize_anthropic_kwargs(
    kwargs: dict,
    *,
    enable: Optional[set[str]] = None,
    compressor: Optional[Compressor] = None,
    log=None,
    **opt_kwargs: Any,
) -> tuple[dict, Optional[OptimizeResult]]:
    """Return a copy of ``kwargs`` with optimised ``system`` and ``messages``.

    The combined report's token counts cover both the system prompt and the
    messages. Returns the kwargs unchanged and None if there is nothing to do.
    Passing ``log`` logs the merged result tagged with source "anthropic".
    """
    model = kwargs.get("model", "claude-sonnet-4-6")
    messages = kwargs.get("messages") or []
    system = kwargs.get("system")

    new_kwargs = dict(kwargs)
    reports: list[OptimizeResult] = []

    # The last user message acts as the query for the system prompt too.
    query = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            query = c if isinstance(c, str) else _join_parts(c)
            break

    if isinstance(system, str) and system.strip():
        sys_res = optimize_text(
            system, query=query or None, context=system, model=model,
            enable=enable, compressor=compressor, **opt_kwargs
        )
        new_kwargs["system"] = sys_res.optimized
        reports.append(sys_res)

    if messages:
        msg_res = optimize_messages(
            messages, model=model, enable=enable, compressor=compressor, **opt_kwargs
        )
        new_kwargs["messages"] = msg_res.optimized
        reports.append(msg_res)

    if not reports:
        return kwargs, None

    # Merge the reports into one for the caller.
    merged = _merge_reports(reports, model)

    # Log the merged result if logging was requested.
    from ..logbook import resolve_logger
    logger = resolve_logger(log)
    if logger is not None:
        try:
            logger.record(merged, query=query or None, source="anthropic")
        except Exception:
            pass

    return new_kwargs, merged


def wrap_anthropic(
    client: Any,
    *,
    enable: Optional[set[str]] = None,
    compressor: Optional[Compressor] = None,
    log=None,
    on_report=None,
    **opt_kwargs: Any,
) -> Any:
    """Patch ``client.messages.create`` to optimise system + messages."""
    messages_api = client.messages
    original_create = messages_api.create

    def patched_create(*args, **kwargs):
        kwargs, report = optimize_anthropic_kwargs(
            kwargs, enable=enable, compressor=compressor, log=log, **opt_kwargs
        )
        if report is not None and on_report is not None:
            on_report(report)
        return original_create(*args, **kwargs)

    messages_api.create = patched_create  # type: ignore[attr-defined]
    return client


def _join_parts(content) -> str:
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                out.append(p.get("text", ""))
            elif isinstance(p, str):
                out.append(p)
        return "\n".join(o for o in out if o)
    return str(content)


def _merge_reports(reports: list[OptimizeResult], model: str) -> OptimizeResult:
    from ..optimizer import OptimizeResult as _R
    before = sum(r.tokens_before for r in reports)
    after = sum(r.tokens_after for r in reports)
    transforms = [t for r in reports for t in r.transforms]
    notes = sorted({n for r in reports for n in r.notes})
    # Carry combined text so a log record shows the real before/after content.
    orig = "\n\n".join(_as_text(r.original) for r in reports)
    opt = "\n\n".join(_as_text(r.optimized) for r in reports)
    return _R(
        original=orig, optimized=opt, model=model,
        tokens_before=before, tokens_after=after,
        transforms=transforms, notes=notes,
    )


def _as_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "\n".join(
            (m.get("content", "") if isinstance(m.get("content"), str)
             else _join_parts(m.get("content", "")))
            for m in obj if isinstance(m, dict)
        )
    return "" if obj is None else str(obj)
