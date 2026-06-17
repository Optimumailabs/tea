"""CrewAI adapter.

CrewAI builds prompts from Agent fields (role, goal, backstory) and Task fields
(description, expected_output). The biggest token sink in CrewAI runs is usually
a long, repeated backstory carried into every step. This adapter optimises those
text fields in place before the crew runs.

Usage::

    from crewai import Agent, Task, Crew
    from tea.integrations.crewai_hook import optimize_agents, optimize_tasks

    agents = [researcher, writer]
    tasks = [research_task, write_task]

    optimize_agents(agents, model_name="gpt-4o")
    optimize_tasks(tasks, model_name="gpt-4o")

    crew = Crew(agents=agents, tasks=tasks)
    crew.kickoff()

These functions mutate the objects in place and also return a combined report.
They touch only text attributes that exist; missing attributes are skipped.
"""

from __future__ import annotations

from typing import Any, Optional

from ..optimizer import Compressor, OptimizeResult, optimize_text

_AGENT_FIELDS = ("role", "goal", "backstory")
_TASK_FIELDS = ("description", "expected_output")


def _optimize_fields(
    obj: Any,
    fields: tuple[str, ...],
    *,
    query: str,
    model_name: str,
    enable: Optional[set[str]],
    compressor: Optional[Compressor],
    opt_kwargs: dict,
) -> list[OptimizeResult]:
    reports = []
    for f in fields:
        text = getattr(obj, f, None)
        if not isinstance(text, str) or not text.strip():
            continue
        res = optimize_text(
            text, query=query or None, context=text, model=model_name,
            enable=enable, compressor=compressor, **opt_kwargs
        )
        try:
            setattr(obj, f, res.optimized)
            reports.append(res)
        except Exception:
            # Some frameworks make fields read-only after construction; skip.
            pass
    return reports


def optimize_agents(
    agents: list,
    *,
    model_name: str = "gpt-4o",
    enable: Optional[set[str]] = None,
    compressor: Optional[Compressor] = None,
    log=None,
    **opt_kwargs: Any,
) -> OptimizeResult:
    """Optimise role/goal/backstory on each agent in place.
    Passing ``log`` logs the merged result tagged with source "crewai"."""
    reports: list[OptimizeResult] = []
    for a in agents:
        query = getattr(a, "goal", "") or getattr(a, "role", "")
        reports.extend(_optimize_fields(
            a, _AGENT_FIELDS, query=query, model_name=model_name,
            enable=enable, compressor=compressor, opt_kwargs=opt_kwargs,
        ))
    return _merge(reports, model_name, log=log)


def optimize_tasks(
    tasks: list,
    *,
    model_name: str = "gpt-4o",
    enable: Optional[set[str]] = None,
    compressor: Optional[Compressor] = None,
    log=None,
    **opt_kwargs: Any,
) -> OptimizeResult:
    """Optimise description/expected_output on each task in place.
    Passing ``log`` logs the merged result tagged with source "crewai"."""
    reports: list[OptimizeResult] = []
    for t in tasks:
        query = getattr(t, "expected_output", "") or getattr(t, "description", "")[:200]
        reports.extend(_optimize_fields(
            t, _TASK_FIELDS, query=query, model_name=model_name,
            enable=enable, compressor=compressor, opt_kwargs=opt_kwargs,
        ))
    return _merge(reports, model_name, log=log)


def _merge(reports: list[OptimizeResult], model: str, *, log=None) -> OptimizeResult:
    before = sum(r.tokens_before for r in reports)
    after = sum(r.tokens_after for r in reports)
    transforms = [t for r in reports for t in r.transforms]
    notes = sorted({n for r in reports for n in r.notes})
    orig = "\n\n".join(str(r.original) for r in reports if r.original)
    opt = "\n\n".join(str(r.optimized) for r in reports if r.optimized)
    merged = OptimizeResult(
        original=orig, optimized=opt, model=model,
        tokens_before=before, tokens_after=after,
        transforms=transforms, notes=notes,
    )
    if log is not None:
        from ..logbook import resolve_logger
        logger = resolve_logger(log)
        if logger is not None:
            try:
                logger.record(merged, source="crewai")
            except Exception:
                pass
    return merged
