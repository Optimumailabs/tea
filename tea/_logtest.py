"""Logging tests for TEA. Run: python -m tea._logtest

Covers: logging off by default, enable/disable, JSONL + human + ledger files,
full-text capture, memory block, per-call source tag, ledger accumulation and
persistence across logger instances, ad hoc directory logging, env-var
activation, and that a logging failure never breaks optimisation.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import tea
from tea.logbook import TEALogger, resolve_logger


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return bool(cond)


def main() -> int:
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="tea_log_"))
    try:
        # 1. Logging is off by default: no files written.
        tea.disable_logging()
        d0 = tmp / "off"
        r = tea.optimize("a\n\na\n\nb", model="gpt-4o")  # no log arg
        ok &= check("logging off by default: no dir created", not d0.exists())

        # 2. enable_logging writes all three files.
        d1 = tmp / "on"
        tea.enable_logging(str(d1))
        tea.optimize("The cat sat. The cat sat.\n\nDogs run fast in the park.",
                     query="cats", enable=tea.AGGRESSIVE_TRANSFORMS)
        ok &= check("jsonl written", (d1 / "tea_prompts.jsonl").exists())
        ok &= check("human log written", (d1 / "tea_prompts.log").exists())
        ok &= check("ledger written", (d1 / "tea_ledger.json").exists())

        # 3. JSONL record has full prompt text and a memory block.
        line = (d1 / "tea_prompts.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
        rec = json.loads(line)
        ok &= check("record has full original_prompt", "original_prompt" in rec)
        ok &= check("record has optimized_prompt", "optimized_prompt" in rec)
        ok &= check("record has memory block", "memory" in rec and "rss_bytes" in rec["memory"])
        ok &= check("record has ledger snapshot", "ledger" in rec and rec["ledger"]["calls"] >= 1)
        ok &= check("record source tag is api", rec["source"] == "api")

        # 4. Ledger accumulates across calls.
        before_calls = tea.get_default_logger().ledger["calls"]
        tea.optimize("Repeated line. Repeated line.\n\nUnique.", query="unique",
                     enable=tea.AGGRESSIVE_TRANSFORMS)
        after_calls = tea.get_default_logger().ledger["calls"]
        ok &= check("ledger calls incremented", after_calls == before_calls + 1)

        # 5. Ledger persists across logger instances (same dir).
        led_a = tea.get_default_logger().ledger
        fresh = TEALogger(str(d1))
        ok &= check("ledger persists across instances",
                    fresh.ledger["calls"] == led_a["calls"])

        # 6. Ad hoc directory via log= argument.
        d2 = tmp / "adhoc"
        tea.disable_logging()
        tea.optimize("x\n\nx\n\ny", model="gpt-4o", log=str(d2))
        ok &= check("ad hoc log= dir writes files", (d2 / "tea_prompts.jsonl").exists())
        ok &= check("ad hoc does not touch default logger", tea.get_default_logger() is None)

        # 7. log=False never logs even when a default logger is active.
        d3 = tmp / "explicit_off"
        tea.enable_logging(str(d3))
        n_before = tea.get_default_logger().ledger["calls"]
        tea.optimize("a\n\na", model="gpt-4o", log=False)
        n_after = tea.get_default_logger().ledger["calls"]
        ok &= check("log=False suppresses logging", n_after == n_before)

        # 8. Source tag flows through from caller.
        d4 = tmp / "src"
        tea.disable_logging()
        tea.optimize("a\n\na\n\nb", model="gpt-4o", log=str(d4), source="openai")
        rec4 = json.loads((d4 / "tea_prompts.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
        ok &= check("custom source tag recorded", rec4["source"] == "openai")

        # 9. Env-var activation (TEA_LOG_DIR) creates a logger lazily.
        tea.disable_logging()
        d5 = tmp / "envdir"
        os.environ["TEA_LOG_DIR"] = str(d5)
        try:
            lg = resolve_logger(None)  # simulates a call with log=None
            ok &= check("env var activates logging lazily", lg is not None)
            ok &= check("env logger points at TEA_LOG_DIR", Path(lg.dir) == d5)
        finally:
            del os.environ["TEA_LOG_DIR"]
            tea.disable_logging()

        # 10. A broken logger never breaks optimisation.
        class BrokenLogger(TEALogger):
            def record(self, *a, **k):
                raise RuntimeError("disk full")
        d6 = tmp / "broken"
        broken = BrokenLogger(str(d6))
        r = tea.optimize("a\n\na\n\nb", model="gpt-4o", log=broken)
        ok &= check("broken logger does not break optimise", r.tokens_after <= r.tokens_before)

        # 11. messages input logs readable role-tagged text.
        d7 = tmp / "msgs"
        tea.optimize(
            [{"role": "system", "content": "Be concise. Be concise."},
             {"role": "user", "content": "Hello"}],
            model="gpt-4o", log=str(d7), source="api",
        )
        rec7 = json.loads((d7 / "tea_prompts.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
        ok &= check("messages logged with role tags", "[system]" in rec7["original_prompt"])

    finally:
        tea.disable_logging()
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("ALL LOGGING TESTS PASS" if ok else "SOME LOGGING TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
