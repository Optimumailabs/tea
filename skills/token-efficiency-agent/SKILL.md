---
name: token-efficiency-agent
description: Token Efficiency Agent (TEA). Measures and reduces wasted tokens in LLM prompts. Use it to "score this prompt", "optimise this prompt", "cut token waste", "shrink this context", or to wire token optimisation into LangChain, CrewAI, AutoGen, the OpenAI SDK, or the Anthropic SDK. Provides a measurement mode (score only) and an optimisation mode (rewrite the prompt with safe deterministic transforms plus an optional LLM compressor).
---

# Token Efficiency Agent (TEA)

TEA cuts the tokens an LLM prompt spends without changing what the model is
asked to do. It has two modes.

- **Measure**: score a prompt, do not change it.
- **Optimise**: rewrite the prompt to spend fewer tokens, and report what
  changed.

The bundled `tea` Python package also plugs into LangChain, CrewAI, AutoGen,
the OpenAI SDK, and the Anthropic SDK, so the same optimisation runs inside a
real application, not just from this skill.

## How to run the tools

Two ways, in priority order:

1. If the package is pip-installed, use the console commands `tea-optimize`
   and `tea-score`. These are on PATH after `pip install token-efficiency-agent`.
2. Otherwise run the bundled scripts with the plugin root, which Claude Code
   exposes as the `CLAUDE_PLUGIN_ROOT` environment variable:
   `python "$CLAUDE_PLUGIN_ROOT/scripts/optimize.py" ...`

Both accept the same flags. Examples below show the console form first.

## When to use this skill

- "Score this prompt" / "Is this prompt efficient?" -> measure mode.
- "Optimise this prompt" / "Cut the token waste" / "Shrink this context" ->
  optimise mode.
- "How do I add token optimisation to my LangChain / CrewAI / AutoGen app?" ->
  point the user at the matching integration (see the table below) and show the
  short snippet.

Do not invoke for general questions that merely mention prompts. This skill is
for explicit measure or optimise requests.

## Measure mode

Run the scorer and report the result. Always invoke the tool, never reimplement
the math.

```bash
tea-score \
    --prompt-file /tmp/prompt.txt \
    --query "the user question" \
    --quality 0.78 \
    --model gpt-4o
# or, without install:
# python "$CLAUDE_PLUGIN_ROOT/scripts/score.py" --prompt-file /tmp/prompt.txt --query "..." --quality 0.78 --quality-supplied --model gpt-4o
```

The tool prints JSON with `tokens`, `score`, `cost`, `suggestions`, and
`assumptions`. Render it as a short report: token breakdown, the composite
score S(P), the per-request cost, and the top suggestions. Surface the
`assumptions` so the reader knows which defaults were used.

## Optimise mode

```bash
# Safe transforms only (whitespace, dedupe, oversized few-shot pruning).
tea-optimize --prompt-file /tmp/prompt.txt --query "the user question" --model gpt-4o

# Add relevance-based context dropping (needs the query).
tea-optimize --prompt-file /tmp/prompt.txt --query "the user question" --aggressive

# Optimise a chat-messages JSON file instead of a raw prompt.
tea-optimize --messages-file /tmp/chat.json --model gpt-4o

# Log every optimised prompt to a directory (original, optimised, tokens
# saved, memory, running ledger). Add --log with no value for the default dir,
# or --log /path/to/dir for a specific one.
tea-optimize --prompt-file /tmp/prompt.txt --query "..." --log /tmp/tea_logs

# Without install, swap the command for:
# python "$CLAUDE_PLUGIN_ROOT/scripts/optimize.py" <same flags>
```

The tool prints a JSON report (`tokens_before`, `tokens_after`,
`tokens_saved`, `reduction_pct`, `transforms`, `notes`) and writes the
optimised prompt to `--out-file` if given. Report the reduction and which
transforms fired. If `exact_tokenizer` is false, mention that token counts are
approximate because tiktoken is not installed.

When `--log` is set, three files appear in the log directory:
`tea_prompts.jsonl` (one record per prompt), `tea_prompts.log` (human
readable), and `tea_ledger.json` (running totals). Each record holds the
original and optimised prompt, tokens saved, dollars saved, process memory,
and the cumulative ledger.

What each transform does:

- **whitespace**: collapse blank-line runs and trailing spaces. Preserves code
  fences. Tiny token savings, cleaner prompt.
- **dedupe**: drop duplicate paragraphs and repeated sentences. Targets the
  common RAG failure of retrieving the same passage twice.
- **few_shot**: prune the back half of an oversized few-shot block.
- **drop_context** (aggressive only): drop context chunks whose lexical overlap
  with the query is low. Never empties the context; always keeps the best chunk;
  removes at most 70 per cent in one pass.
- **compress** (package only): route the prompt through a caller-supplied LLM
  compressor. Guarded so a degenerate compressor output is rejected.

Expect 15 to 35 per cent reduction on bloated prompts from the deterministic
transforms alone. Deeper cuts need the optional LLM compressor.

## Using TEA inside an application

The `tea` package is what a developer imports into their own code. It has no
hard dependency on any framework; each adapter imports its framework lazily.

| Framework | Import | One-liner |
|---|---|---|
| OpenAI SDK | `from tea.integrations.openai_wrap import wrap_openai` | `client = wrap_openai(OpenAI())` |
| Anthropic SDK | `from tea.integrations.anthropic_wrap import wrap_anthropic` | `client = wrap_anthropic(Anthropic())` |
| LangChain | `from tea.integrations.langchain_cb import TEAOptimizer` | `chain = TEAOptimizer(model_name="gpt-4o") \| model` |
| CrewAI | `from tea.integrations.crewai_hook import optimize_agents, optimize_tasks` | `optimize_agents(agents); optimize_tasks(tasks)` |
| AutoGen | `from tea.integrations.autogen_hook import TEAMessageTransform` | add `TEAMessageTransform()` to a `TransformMessages` capability |

Direct API, no framework:

```python
import tea
result = tea.optimize(prompt, query="the user question", model="gpt-4o")
print(result.optimized)      # the shorter prompt
print(result.summary())      # what changed and how much was saved
```

Turn on logging for every call:

```python
tea.enable_logging("tea_logs")          # or set the TEA_LOG_DIR env var
```

## Output guidelines

- Numbers come from the tool or the package. Never invent them.
- If a transform saved 0 tokens, say so plainly; do not imply otherwise.
- When the optimiser drops context, it preserves the highest-overlap chunk. Tell
  the user the optimiser will never empty the context, so meaning is retained.
- If the tool errors, surface the error verbatim and stop.

## Honest limits

- Relevance scoring is lexical overlap, not real attention. It is a safe proxy
  that occasionally keeps a chunk a true attention signal would drop.
- Anthropic models have no public tokenizer, so their counts use the cl100k
  fallback and are approximate. Relative before/after comparisons stay valid.
- The deterministic transforms are quality-safe by design. The LLM compressor
  is opt-in and bounded, but any semantic compression carries some risk, so it
  is never on by default.

## Reference

- Package README: repository root `README.md`.
- Self-test: `python -m tea._selftest` from the repository root.
