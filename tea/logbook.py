"""Per-prompt logging for TEA.

Every optimise call can append a structured record to a log so a team can audit
what TEA did, how many tokens it saved, and how much memory the call used. Two
sinks are written when logging is enabled:

1. A JSONL file (one JSON object per line) for machine analysis.
2. A human-readable .log file with a formatted block per prompt.

The module also keeps a cumulative savings ledger (total calls, tokens before,
tokens after, tokens saved, estimated dollars saved) that is updated on every
record and written into each entry, so the latest line always shows the running
totals.

Logging is OFF by default. Turn it on by either:

    import tea
    tea.enable_logging("/path/to/tea_logs")      # directory; files created inside

or per call:

    tea.optimize(prompt, query=q, log=True)              # uses the default dir
    tea.optimize(prompt, query=q, log="/custom/dir")     # ad hoc dir

Environment override: set TEA_LOG_DIR to a directory and logging turns on for
the whole process without any code change.
"""

from __future__ import annotations

import json
import os
import threading
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .tokens import estimate_cost


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _process_rss_bytes() -> Optional[int]:
    """Resident set size of the current process in bytes, or None if it cannot
    be determined without extra dependencies.

    Tries psutil first (most portable), then the Unix resource module. On
    Windows without psutil this returns None, which the logger records honestly
    rather than guessing."""
    try:
        import psutil  # type: ignore
        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        import resource  # Unix only
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes. Normalise to bytes.
        # Heuristic: values under 10^9 are almost certainly KB on Linux.
        return int(ru * 1024) if ru < 10**9 else int(ru)
    except Exception:
        return None


@dataclass
class Ledger:
    """Running totals across all logged calls for one logger instance."""
    calls: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    usd_saved: float = 0.0

    def update(self, before: int, after: int, usd: float) -> None:
        self.calls += 1
        self.tokens_before += before
        self.tokens_after += after
        self.tokens_saved += max(0, before - after)
        self.usd_saved += max(0.0, usd)

    def as_dict(self) -> dict:
        reduction = (100.0 * self.tokens_saved / self.tokens_before) if self.tokens_before else 0.0
        return {
            "calls": self.calls,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "reduction_pct": round(reduction, 2),
            "usd_saved": round(self.usd_saved, 6),
        }


class TEALogger:
    """Writes one record per optimise call to a JSONL file and a text log, and
    maintains a cumulative ledger. Thread-safe via a simple lock so concurrent
    requests do not interleave lines."""

    def __init__(self, log_dir, *, full_text: bool = True, jsonl: bool = True,
                 human: bool = True):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dir / "tea_prompts.jsonl"
        self.human_path = self.dir / "tea_prompts.log"
        self.ledger_path = self.dir / "tea_ledger.json"
        self.full_text = full_text
        self.jsonl = jsonl
        self.human = human
        self._lock = threading.Lock()
        self._ledger = self._load_ledger()

    def _load_ledger(self) -> Ledger:
        if self.ledger_path.exists():
            try:
                d = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                return Ledger(
                    calls=d.get("calls", 0),
                    tokens_before=d.get("tokens_before", 0),
                    tokens_after=d.get("tokens_after", 0),
                    tokens_saved=d.get("tokens_saved", 0),
                    usd_saved=d.get("usd_saved", 0.0),
                )
            except Exception:
                pass
        return Ledger()

    @staticmethod
    def _as_text(obj) -> str:
        """Render a prompt (str or messages list) to plain text for logging."""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, list):
            out = []
            for m in obj:
                if isinstance(m, dict):
                    role = m.get("role", "?")
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    out.append(f"[{role}] {content}")
                else:
                    out.append(str(m))
            return "\n".join(out)
        return str(obj)

    def record(self, result, *, query: Optional[str] = None,
               source: str = "api", peak_kib: Optional[float] = None) -> dict:
        """Append a record for one OptimizeResult. Returns the record dict.

        ``source`` names where the call came from (api, openai, langchain, cli,
        and so on). ``peak_kib`` is the tracemalloc peak for the call if the
        caller measured it.
        """
        before, after = result.tokens_before, result.tokens_after
        usd_before = estimate_cost(result.model, before, 0)
        usd_after = estimate_cost(result.model, after, 0)
        usd_saved = max(0.0, usd_before - usd_after)

        with self._lock:
            self._ledger.update(before, after, usd_saved)
            ledger_snapshot = self._ledger.as_dict()

            orig_text = self._as_text(result.original)
            opt_text = self._as_text(result.optimized)

            record = {
                "ts": _utc_now_iso(),
                "source": source,
                "model": result.model,
                "tokens_before": before,
                "tokens_after": after,
                "tokens_saved": result.tokens_saved,
                "reduction_pct": round(result.reduction_pct, 2),
                "usd_saved": round(usd_saved, 6),
                "transforms": [
                    {"name": t.name, "saved": t.saved, "note": t.note}
                    for t in result.transforms
                ],
                "notes": list(result.notes),
                "query": query,
                "memory": {
                    "rss_bytes": _process_rss_bytes(),
                    "peak_kib": round(peak_kib, 1) if peak_kib is not None else None,
                },
                "ledger": ledger_snapshot,
            }
            if self.full_text:
                record["original_prompt"] = orig_text
                record["optimized_prompt"] = opt_text
            else:
                record["original_preview"] = orig_text[:200]
                record["optimized_preview"] = opt_text[:200]

            if self.jsonl:
                with self.jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if self.human:
                with self.human_path.open("a", encoding="utf-8") as f:
                    f.write(self._format_human(record, orig_text, opt_text))
            self._save_ledger()
        return record

    def _save_ledger(self) -> None:
        try:
            self.ledger_path.write_text(
                json.dumps(self._ledger.as_dict(), indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    @staticmethod
    def _format_human(record: dict, orig_text: str, opt_text: str) -> str:
        bar = "=" * 78
        mem = record["memory"]
        rss_mb = (mem["rss_bytes"] / (1024 * 1024)) if mem["rss_bytes"] else None
        lines = [
            bar,
            f"{record['ts']}  source={record['source']}  model={record['model']}",
            f"tokens: {record['tokens_before']:,} -> {record['tokens_after']:,} "
            f"(saved {record['tokens_saved']:,}, {record['reduction_pct']}%)  "
            f"usd_saved=${record['usd_saved']:.6f}",
        ]
        if record["transforms"]:
            applied = ", ".join(
                f"{t['name']}(+{t['saved']})" for t in record["transforms"] if t["saved"] > 0
            ) or "none with savings"
            lines.append(f"transforms: {applied}")
        if record["notes"]:
            lines.append("notes: " + "; ".join(record["notes"]))
        if rss_mb is not None:
            peak = mem["peak_kib"]
            peak_str = f", peak {peak:.1f} KiB" if peak is not None else ""
            lines.append(f"memory: rss {rss_mb:.1f} MiB{peak_str}")
        led = record["ledger"]
        lines.append(
            f"ledger: {led['calls']} calls, saved {led['tokens_saved']:,} tokens "
            f"({led['reduction_pct']}%), ${led['usd_saved']:.4f} total"
        )
        lines.append("--- original ---")
        lines.append(orig_text)
        lines.append("--- optimised ---")
        lines.append(opt_text)
        lines.append("")
        return "\n".join(lines) + "\n"

    @property
    def ledger(self) -> dict:
        return self._ledger.as_dict()


# ---------------------------------------------------------------------------
# Module-level default logger
# ---------------------------------------------------------------------------
_default_logger: Optional[TEALogger] = None
_default_lock = threading.Lock()


def enable_logging(log_dir=None, *, full_text: bool = True,
                   jsonl: bool = True, human: bool = True) -> TEALogger:
    """Turn on logging to ``log_dir`` (default: ./tea_logs or $TEA_LOG_DIR).
    Returns the logger so the caller can read its ledger."""
    global _default_logger
    if log_dir is None:
        log_dir = os.environ.get("TEA_LOG_DIR", "tea_logs")
    with _default_lock:
        _default_logger = TEALogger(log_dir, full_text=full_text, jsonl=jsonl, human=human)
    return _default_logger


def disable_logging() -> None:
    global _default_logger
    with _default_lock:
        _default_logger = None


def get_default_logger() -> Optional[TEALogger]:
    """Return the active default logger. If TEA_LOG_DIR is set but no logger
    has been created yet, create one lazily so env-only activation works."""
    global _default_logger
    if _default_logger is None:
        env_dir = os.environ.get("TEA_LOG_DIR")
        if env_dir:
            return enable_logging(env_dir)
    return _default_logger


def resolve_logger(log) -> Optional[TEALogger]:
    """Resolve the ``log`` argument passed to optimize().

    - None  -> use the default logger if one is active (or env-activated).
    - False -> no logging for this call.
    - True  -> use/lazily-create the default logger.
    - str/Path -> a one-off logger writing to that directory.
    - TEALogger -> use it directly.
    """
    if log is False:
        return None
    if log is None:
        return get_default_logger()
    if log is True:
        return get_default_logger() or enable_logging()
    if isinstance(log, TEALogger):
        return log
    # Treat anything else as a directory path for an ad hoc logger.
    return TEALogger(log)
