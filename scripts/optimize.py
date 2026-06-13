#!/usr/bin/env python3
"""Token Efficiency Agent, Phase 1 optimiser CLI.

Reads a prompt, applies the deterministic TEA transforms (and an optional LLM
compressor if the caller wires one in via the package API), and writes the
optimised prompt plus a JSON report.

This CLI is the entry point the SKILL.md uses. It depends only on the bundled
``tea`` package, which sits one directory up.

Usage:
    python optimize.py --prompt-file prompt.txt --query "the user question"
    python optimize.py --prompt-file prompt.txt --aggressive --json-only
    python optimize.py --messages-file chat.json --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the bundled tea package importable when run from anywhere.
_HERE = Path(__file__).resolve()
_SKILL_ROOT = _HERE.parent.parent           # .../product/skill
if str(_SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SKILL_ROOT))

import tea  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="TEA optimiser: shrink a prompt with safe deterministic transforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt-file", type=Path, help="A plain-text prompt to optimise.")
    src.add_argument("--messages-file", type=Path,
                     help="A JSON file holding a list of chat-message dicts.")
    p.add_argument("--query", default=None,
                   help="The user question, used to score context relevance. "
                        "Recommended when --aggressive is set.")
    p.add_argument("--model", default="gpt-4o", help="Model id, for token counting.")
    p.add_argument("--aggressive", action="store_true",
                   help="Also drop low-relevance context (needs --query). "
                        "Without this, only whitespace/dedupe/few-shot run.")
    p.add_argument("--out-file", type=Path, default=None,
                   help="Write the optimised prompt here. Defaults to stdout via the JSON.")
    p.add_argument("--json-only", action="store_true",
                   help="Print only the JSON report, no human summary.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    enable = tea.AGGRESSIVE_TRANSFORMS - {"compress"} if args.aggressive else tea.SAFE_TRANSFORMS
    # The CLI cannot supply an LLM compressor, so "compress" is always excluded
    # here. Callers who want compression use the package API and pass a
    # compressor callable; see the SKILL.md and README.

    if args.prompt_file is not None:
        if not args.prompt_file.exists():
            print(json.dumps({"error": f"prompt file not found: {args.prompt_file}"}))
            return 2
        prompt = args.prompt_file.read_text(encoding="utf-8")
        result = tea.optimize(prompt, query=args.query, model=args.model, enable=enable)
        optimized_text = result.optimized
    else:
        if not args.messages_file.exists():
            print(json.dumps({"error": f"messages file not found: {args.messages_file}"}))
            return 2
        try:
            messages = json.loads(args.messages_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"messages file is not valid JSON: {e}"}))
            return 2
        result = tea.optimize(messages, model=args.model, enable=enable)
        optimized_text = json.dumps(result.optimized, indent=2)

    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(optimized_text, encoding="utf-8")

    report = {
        "model": result.model,
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "tokens_saved": result.tokens_saved,
        "reduction_pct": round(result.reduction_pct, 2),
        "transforms": [
            {"name": t.name, "saved": t.saved, "note": t.note}
            for t in result.transforms
        ],
        "notes": result.notes,
        "exact_tokenizer": tea.tokenizer_is_exact(args.model),
        "out_file": str(args.out_file) if args.out_file else None,
    }
    print(json.dumps(report, indent=2))
    if not args.json_only:
        print("\n" + result.summary(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
