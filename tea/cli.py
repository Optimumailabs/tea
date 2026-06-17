"""Console-script entry points for the installed package.

After ``pip install token-efficiency-agent`` two commands are available:

    tea-optimize --prompt-file prompt.txt --query "..." --aggressive --log
    tea-score    --prompt-file prompt.txt --query "..." --model gpt-4o

These mirror the scripts in the repo's ``scripts/`` directory, but live inside
the package so they survive installation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import (
    optimize as _optimize,
    score as _score,
    tokenizer_is_exact,
    SAFE_TRANSFORMS,
    AGGRESSIVE_TRANSFORMS,
)


# ---------------------------------------------------------------------------
# tea-optimize
# ---------------------------------------------------------------------------
def _build_optimize_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tea-optimize",
        description="Shrink a prompt with safe deterministic transforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt-file", type=Path, help="A plain-text prompt to optimise.")
    src.add_argument("--messages-file", type=Path,
                     help="A JSON file holding a list of chat-message dicts.")
    p.add_argument("--query", default=None,
                   help="The user question, used to score context relevance.")
    p.add_argument("--model", default="gpt-4o", help="Model id, for token counting.")
    p.add_argument("--aggressive", action="store_true",
                   help="Also drop low-relevance context (needs --query).")
    p.add_argument("--out-file", type=Path, default=None,
                   help="Write the optimised prompt here.")
    p.add_argument("--json-only", action="store_true",
                   help="Print only the JSON report, no human summary.")
    p.add_argument("--log", nargs="?", const=True, default=None,
                   help="Log this prompt. No value uses ./tea_logs or $TEA_LOG_DIR; "
                        "a value logs to that directory.")
    return p


def optimize_main(argv=None) -> int:
    args = _build_optimize_parser().parse_args(argv)
    enable = (AGGRESSIVE_TRANSFORMS - {"compress"}) if args.aggressive else SAFE_TRANSFORMS

    if args.prompt_file is not None:
        if not args.prompt_file.exists():
            print(json.dumps({"error": f"prompt file not found: {args.prompt_file}"}))
            return 2
        prompt = args.prompt_file.read_text(encoding="utf-8")
        result = _optimize(prompt, query=args.query, model=args.model,
                           enable=enable, log=args.log, source="cli")
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
        result = _optimize(messages, model=args.model, enable=enable,
                           log=args.log, source="cli")
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
        "transforms": [{"name": t.name, "saved": t.saved, "note": t.note}
                       for t in result.transforms],
        "notes": result.notes,
        "exact_tokenizer": tokenizer_is_exact(args.model),
        "out_file": str(args.out_file) if args.out_file else None,
    }
    print(json.dumps(report, indent=2))
    if not args.json_only:
        print("\n" + result.summary(), file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# tea-score
# ---------------------------------------------------------------------------
def _build_score_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tea-score",
        description="Score a prompt's token efficiency without rewriting it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--query", default=None)
    p.add_argument("--context-file", type=Path, default=None)
    p.add_argument("--quality", type=float, default=0.75)
    p.add_argument("--model", default="gpt-4o")
    return p


def score_main(argv=None) -> int:
    args = _build_score_parser().parse_args(argv)
    if not args.prompt_file.exists():
        print(json.dumps({"error": f"prompt file not found: {args.prompt_file}"}))
        return 2
    prompt = args.prompt_file.read_text(encoding="utf-8")
    context = args.context_file.read_text(encoding="utf-8") if args.context_file else None
    out = _score(prompt, query=args.query, quality=args.quality,
                 model=args.model, context=context)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(optimize_main())
