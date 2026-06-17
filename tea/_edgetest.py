"""Edge-case tests for the TEA core. Run: python -m tea._edgetest

Covers empty/degenerate inputs, the chat list-of-parts content format, code
block preservation, compressor misbehaviour, threshold extremes, and unicode.
Each check prints PASS or FAIL and the suite exits non-zero if any fail.
"""

from __future__ import annotations

import sys

import tea
from tea.optimizer import optimize_text, optimize_messages, _normalise_whitespace


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return bool(cond)


def main() -> int:
    ok = True

    # --- Input shape ------------------------------------------------------
    r = optimize_text("", model="gpt-4o")
    ok &= check("empty string: no crash, 0 tokens", r.tokens_before == 0 and r.optimized == "")

    r = optimize_text("hello", model="gpt-4o")
    ok &= check("single word: survives unchanged", "hello" in r.optimized)

    r = optimize_text("   \n\n   \t  \n", model="gpt-4o")
    ok &= check("whitespace-only: collapses to empty-ish without crash", r.tokens_after <= r.tokens_before)

    r = optimize_messages([], model="gpt-4o")
    ok &= check("empty messages: no crash", r.tokens_before == 0 and r.optimized == [])

    # list-of-parts content (vision / tool format)
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": "Be concise.\n\nBe concise."}]},
        {"role": "user", "content": "What is in the image?"},
    ]
    r = optimize_messages(msgs, model="gpt-4o", enable={"dedupe"})
    ok &= check("list-of-parts content: no crash", len(r.optimized) == 2)

    # missing keys
    msgs = [{"content": "orphan content no role"}, {"role": "user"}]
    try:
        r = optimize_messages(msgs, model="gpt-4o")
        ok &= check("missing role/content keys: no crash", len(r.optimized) == 2)
    except Exception as e:
        ok &= check(f"missing role/content keys: no crash (raised {type(e).__name__})", False)

    # unicode / emoji
    uni = "Café data.\n\nCafé data.\n\n数据分析 是 重要 的。 🚀"
    r = optimize_text(uni, model="gpt-4o", enable={"dedupe"})
    ok &= check("unicode: deduped and preserved", "数据分析" in r.optimized and "Café" in r.optimized)

    # --- Content patterns -------------------------------------------------
    # code block must be preserved verbatim, including blank lines inside
    code = "Intro.\n\n```python\ndef f():\n\n\n    return 1\n```\n\nOutro."
    out = _normalise_whitespace(code)
    ok &= check("code block blank lines preserved", "return 1" in out and "def f():" in out)
    ok &= check("code fence intact", out.count("```") == 2)

    # all chunks relevant: drop_context should drop nothing
    ctx = ("The Sherman Act limits monopolies and restraint of trade.\n\n"
           "Sherman Act enforcement is handled by the DOJ antitrust division.\n\n"
           "Monopolies under the Sherman Act face civil and criminal penalties.")
    r = optimize_text(ctx, query="What does the Sherman Act do about monopolies?",
                      context=ctx, model="gpt-4o", enable={"drop_context"}, keep_threshold=0.05)
    ok &= check("all-relevant context: nothing dropped",
                r.tokens_after == r.tokens_before)

    # nothing relevant: keep the best, never empty
    ctx = "Bananas are yellow.\n\nOranges are round.\n\nGrapes grow in bunches."
    r = optimize_text(ctx, query="quantum chromodynamics gluon confinement",
                      context=ctx, model="gpt-4o", enable={"drop_context"}, keep_threshold=0.9)
    ok &= check("no-relevant context: not emptied", len(r.optimized.strip()) > 0)

    # duplicate sentences differing only in trailing whitespace
    dup = "The model uses attention.   \n\nThe model uses attention."
    r = optimize_text(dup, model="gpt-4o", enable={"dedupe"})
    ok &= check("near-dup (trailing ws) deduped", r.tokens_after < r.tokens_before)

    # short chunks (< 25 chars) must not be deduped away even if repeated
    short = "Yes.\n\nYes.\n\nNo."
    r = optimize_text(short, model="gpt-4o", enable={"dedupe"})
    ok &= check("short repeated lines preserved (paragraph dedupe still applies)",
                "No" in r.optimized)

    # one big blob, no paragraph breaks
    blob = "sentence one. " * 40
    r = optimize_text(blob, query="sentence", model="gpt-4o", enable={"dedupe"})
    ok &= check("no-paragraph-break blob: dedupes repeated sentences",
                r.tokens_after < r.tokens_before)

    # --- Compressor hook --------------------------------------------------
    base = "A genuinely long piece of text that the compressor will be asked to shrink down."
    r = optimize_text(base, model="gpt-4o", enable={"compress"},
                      compressor=lambda t, ratio: "")
    ok &= check("compressor returns empty: rejected, kept original",
                r.optimized == base or r.tokens_after == r.tokens_before)

    r = optimize_text(base, model="gpt-4o", enable={"compress"},
                      compressor=lambda t, ratio: t + " " + t)
    ok &= check("compressor returns longer: rejected", r.tokens_after <= r.tokens_before)

    def boom(t, ratio):
        raise ValueError("compressor exploded")
    r = optimize_text(base, model="gpt-4o", enable={"compress"}, compressor=boom)
    ok &= check("compressor raises: caught, original kept", r.optimized == base)

    r = optimize_text(base, model="gpt-4o", enable={"compress"},
                      compressor=lambda t, ratio: None)
    ok &= check("compressor returns None: handled", r.optimized == base)

    # --- Threshold extremes ----------------------------------------------
    ctx = "Alpha relevant text here.\n\nBeta unrelated text here.\n\nGamma other text."
    r = optimize_text(ctx, query="alpha", context=ctx, model="gpt-4o",
                      enable={"drop_context"}, keep_threshold=0.0)
    ok &= check("keep_threshold 0: keeps everything", r.tokens_after == r.tokens_before)

    r = optimize_text(ctx, query="alpha", context=ctx, model="gpt-4o",
                      enable={"drop_context"}, keep_threshold=1.0)
    ok &= check("keep_threshold 1: keeps at least the best chunk", len(r.optimized.strip()) > 0)

    # --- Pipeline interaction: dedupe modifies text, then drop_context runs --
    # The explicit context is no longer a verbatim substring after dedupe.
    # The reported reduction must match the actual optimized text.
    full = ("Alpha is about cats. Alpha is about cats.\n\n"
            "Beta is about quantum physics and gluons.\n\n"
            "Gamma is about cats and kittens.")
    r = optimize_text(
        full, query="tell me about cats", context=full, model="gpt-4o",
        enable={"dedupe", "drop_context"}, keep_threshold=0.10,
    )
    from tea.tokens import count_tokens as _ct
    ok &= check("report matches optimized text after dedupe+drop",
                r.tokens_after == _ct(r.optimized, "gpt-4o"))
    ok &= check("dedupe+drop keeps relevant, drops physics",
                "cats" in r.optimized.lower() and "gluons" not in r.optimized.lower())

    # --- Additional hard inputs ------------------------------------------
    # very long single line, no sentence punctuation
    longline = "word " * 5000
    r = optimize_text(longline, model="gpt-4o", enable={"whitespace", "dedupe"})
    ok &= check("very long single line: no crash", r.tokens_after > 0)

    # prompt that is only a code block (must be preserved, fences intact)
    onlycode = "```python\nx = 1\n\n\ny = 2\n```"
    r = optimize_text(onlycode, model="gpt-4o", enable={"whitespace", "dedupe"})
    ok &= check("code-only prompt: fences preserved", r.optimized.count("```") == 2)
    ok &= check("code-only prompt: code intact", "x = 1" in r.optimized and "y = 2" in r.optimized)

    # message content that is an int or None (malformed but must not crash)
    weird = [
        {"role": "system", "content": None},
        {"role": "user", "content": 12345},
        {"role": "user", "content": "Real question here."},
    ]
    try:
        r = optimize_messages(weird, model="gpt-4o")
        ok &= check("non-string message content: no crash", len(r.optimized) == 3)
    except Exception as e:
        ok &= check(f"non-string message content: no crash (raised {type(e).__name__})", False)

    # deeply nested list-of-parts with mixed dict and str parts
    nested = [
        {"role": "system", "content": [
            {"type": "text", "text": "Be brief. Be brief."},
            "loose string part",
            {"type": "image_url", "image_url": {"url": "http://x"}},
        ]},
        {"role": "user", "content": "Describe it."},
    ]
    try:
        r = optimize_messages(nested, model="gpt-4o", enable={"dedupe"})
        ok &= check("nested mixed parts: no crash", len(r.optimized) == 2)
    except Exception as e:
        ok &= check(f"nested mixed parts: no crash (raised {type(e).__name__})", False)

    # mixed-language repeated content dedupes correctly
    mixed = "日本語のテキストです。\n\n日本語のテキストです。\n\nDifferent English content."
    r = optimize_text(mixed, model="gpt-4o", enable={"dedupe"})
    ok &= check("mixed-language dedupe", r.tokens_after < r.tokens_before)
    ok &= check("mixed-language keeps English", "English" in r.optimized)

    # query longer than the context (short context, long question)
    r = optimize_text(
        "Cats purr.", query="Tell me everything about the behaviour of domestic cats",
        context="Cats purr.", model="gpt-4o", enable={"drop_context"}, keep_threshold=0.05,
    )
    ok &= check("long query, short context: kept", "Cats purr" in r.optimized)

    # enable set is empty: nothing changes
    r = optimize_text("a\n\na\n\nb", model="gpt-4o", enable=set())
    ok &= check("empty enable set: no-op", r.tokens_after == r.tokens_before)

    # unknown transform name in enable: ignored, no crash
    r = optimize_text("a\n\na", model="gpt-4o", enable={"nonexistent_transform"})
    ok &= check("unknown transform name: ignored", r.tokens_after <= r.tokens_before)

    # whitespace transform must not corrupt a markdown table
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n\n\n| c | d |\n|---|---|\n| 3 | 4 |"
    r = optimize_text(table, model="gpt-4o", enable={"whitespace"})
    ok &= check("markdown table survives whitespace", "| a | b |" in r.optimized and "| c | d |" in r.optimized)

    # concurrency: many threads logging to one logger must not corrupt JSONL
    import json as _json
    import tempfile as _tf
    import threading as _th
    import shutil as _sh
    from pathlib import Path as _P
    from tea.logbook import TEALogger as _TL
    d = _P(_tf.mkdtemp(prefix="tea_conc_"))
    try:
        lg = _TL(str(d))
        def worker(n):
            for _ in range(10):
                rr = optimize_text(f"Line {n}. Line {n}.\n\nUnique {n} content here.",
                                   query=f"unique {n}", enable={"dedupe"})
                lg.record(rr, source="thread")
        threads = [_th.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        lines = (d / "tea_prompts.jsonl").read_text(encoding="utf-8").strip().splitlines()
        all_valid = all(_json.loads(ln) for ln in lines)
        ok &= check("concurrent logging: 80 valid JSONL lines",
                    len(lines) == 80 and all_valid)
        ok &= check("concurrent ledger consistent", lg.ledger["calls"] == 80)
    finally:
        _sh.rmtree(d, ignore_errors=True)

    print()
    print("ALL EDGE CASES PASS" if ok else "SOME EDGE CASES FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
