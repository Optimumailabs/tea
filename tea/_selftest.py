"""Minimal self-test for the TEA core. Run: python -m tea._selftest

Not a full test suite; a fast sanity check that the deterministic transforms
shrink a known-bloated prompt without destroying it, that the safety guards
hold, and that the chat-message path works.
"""

from __future__ import annotations

import sys

import tea
from tea.optimizer import optimize_text, optimize_messages


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    return cond


def main() -> int:
    ok = True

    # 1. Dedupe removes a repeated paragraph.
    dup = "Alpha fact about cats.\n\nAlpha fact about cats.\n\nBeta fact about dogs."
    r = optimize_text(dup, model="gpt-4o", enable={"dedupe"})
    ok &= check("dedupe drops the repeat", r.tokens_after < r.tokens_before)
    ok &= check("dedupe keeps unique content", "dogs" in r.optimized and "cats" in r.optimized)

    # 2. Whitespace collapse.
    messy = "Line one.\n\n\n\n\nLine two.   \n\n\n"
    r = optimize_text(messy, model="gpt-4o", enable={"whitespace"})
    ok &= check("whitespace collapses blank lines", "\n\n\n" not in r.optimized)

    # 3. Drop low-utility context needs a query; drops the irrelevant chunk.
    ctx = ("The mitochondria is the powerhouse of the cell and produces ATP.\n\n"
           "Quarterly revenue rose on strong demand for cloud services.\n\n"
           "Photosynthesis converts light energy into chemical energy in plants.")
    r = optimize_text(
        ctx, query="What does the mitochondria do in a cell?",
        context=ctx, model="gpt-4o", enable={"drop_context"}, keep_threshold=0.10,
    )
    ok &= check("drop_context shrinks on an off-topic chunk", r.tokens_after < r.tokens_before)
    ok &= check("drop_context keeps the relevant chunk", "mitochondria" in r.optimized.lower())

    # 4. Safety: never drop everything.
    allbad = "Banana.\n\nOrange.\n\nGrape."
    r = optimize_text(
        allbad, query="quantum chromodynamics", context=allbad,
        model="gpt-4o", enable={"drop_context"}, keep_threshold=0.9,
    )
    ok &= check("drop_context never empties the context", len(r.optimized.strip()) > 0)

    # 5. Compressor guard rejects a degenerate compressor.
    r = optimize_text(
        "Some reasonably long text that should not be collapsed to nothing at all.",
        model="gpt-4o", enable={"compress"},
        compressor=lambda t, ratio: "x",  # absurdly short
    )
    ok &= check("compressor guard rejects near-empty output",
                any("guard" in n for n in r.notes))

    # 6. Compressor that returns a sane shorter string is accepted.
    long_text = " ".join(["sentence number {}".format(i) for i in range(60)])
    r = optimize_text(
        long_text, model="gpt-4o", enable={"compress"},
        compressor=lambda t, ratio: " ".join(t.split()[: int(len(t.split()) * 0.5)]),
    )
    ok &= check("valid compressor is applied", r.tokens_after < r.tokens_before)

    # 7. Chat messages path: optimises context, protects the live user query.
    messages = [
        {"role": "system", "content": "You are helpful.\n\nYou are helpful.\n\nBe concise."},
        {"role": "user", "content": "Summarise the attached note."},
    ]
    r = optimize_messages(messages, model="gpt-4o", enable={"dedupe", "whitespace"})
    ok &= check("messages: total tokens drop", r.tokens_after < r.tokens_before)
    ok &= check("messages: structure preserved", len(r.optimized) == 2)
    ok &= check("messages: last user query untouched",
                r.optimized[1]["content"] == "Summarise the attached note.")

    # 8. Public API dispatch.
    rs = tea.optimize("a\n\na\n\nb", model="gpt-4o")
    ok &= check("tea.optimize(str) returns OptimizeResult", hasattr(rs, "reduction_pct"))
    rm = tea.optimize([{"role": "user", "content": "hi"}], model="gpt-4o")
    ok &= check("tea.optimize(list) returns OptimizeResult", hasattr(rm, "reduction_pct"))

    # 9. score() returns a bounded S.
    sc = tea.score("Some prompt text here.", query="text", quality=0.8, model="gpt-4o")
    ok &= check("score S is a float", isinstance(sc["score"]["S"], float))

    print()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
