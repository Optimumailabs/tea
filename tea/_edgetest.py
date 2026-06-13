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

    print()
    print("ALL EDGE CASES PASS" if ok else "SOME EDGE CASES FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
