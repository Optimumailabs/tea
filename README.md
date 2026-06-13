# Token Efficiency Agent (TEA)

> A measurement and optimisation layer for LLM prompts. TEA scores how much of
> a prompt is doing real work, then rewrites it to spend fewer tokens without
> changing the task. Deterministic by default, with an optional LLM compressor
> for deeper savings.

Works as a standalone Python package and as a Claude Code skill, with drop-in
adapters for **LangChain, CrewAI, AutoGen, the OpenAI SDK, and the Anthropic SDK**.

---

## Contents

- [Why TEA](#why-tea)
- [Install](#install)
- [Quick start](#quick-start)
- [Framework integrations](#framework-integrations)
- [Transforms](#transforms)
- [Command line](#command-line)
- [How it works](#how-it-works)
- [Safety](#safety)
- [Testing](#testing)
- [Project layout](#project-layout)

---

## Why TEA

A typical production prompt ships 40 to 65 per cent more tokens than the model
actually uses. The waste comes from duplicate retrieved passages, off-topic
context, oversized few-shot blocks, and boilerplate. That waste is pure cost
and added latency, and it is usually invisible because token counts and bills
both rise together without pointing at the cause.

TEA finds the waste and removes it, then reports exactly what it changed and
how many tokens it saved.

---

## Install

```bash
pip install tiktoken        # optional, for exact token counts
```

Then put this repository on your `PYTHONPATH`, or copy the `tea/` package into
your project. Importing `tea` pulls in no framework; each adapter imports its
own framework only when you use it.

---

## Quick start

```python
import tea

# Optimise a raw prompt
result = tea.optimize(long_prompt, query="the user question", model="gpt-4o")
print(result.optimized)        # the shorter prompt
print(result.summary())        # what changed, how many tokens saved

# Optimise chat messages
result = tea.optimize(messages, model="gpt-4o")
cheaper_messages = result.optimized

# Score a prompt without rewriting it
report = tea.score(long_prompt, query="...", quality=0.8)
print(report["score"]["S"])
```

Every call returns an `OptimizeResult` with `tokens_before`, `tokens_after`,
`tokens_saved`, `reduction_pct`, the list of `transforms` that fired, and any
`notes`.

---

## Framework integrations

| Framework | Import | One-liner |
|---|---|---|
| OpenAI SDK | `from tea.integrations.openai_wrap import wrap_openai` | `client = wrap_openai(OpenAI())` |
| Anthropic SDK | `from tea.integrations.anthropic_wrap import wrap_anthropic` | `client = wrap_anthropic(Anthropic())` |
| LangChain | `from tea.integrations.langchain_cb import TEAOptimizer` | `chain = TEAOptimizer(model_name="gpt-4o") \| model` |
| CrewAI | `from tea.integrations.crewai_hook import optimize_agents, optimize_tasks` | `optimize_agents(agents); optimize_tasks(tasks)` |
| AutoGen | `from tea.integrations.autogen_hook import TEAMessageTransform` | add `TEAMessageTransform()` to a `TransformMessages` capability |

**OpenAI example.** Wrap the client once and every call is optimised:

```python
from openai import OpenAI
from tea.integrations.openai_wrap import wrap_openai

client = wrap_openai(OpenAI())
client.chat.completions.create(model="gpt-4o", messages=[...])
```

**LangChain example.** Drop the optimiser into an LCEL chain:

```python
from langchain_openai import ChatOpenAI
from tea.integrations.langchain_cb import TEAOptimizer

model = ChatOpenAI(model="gpt-4o")
chain = TEAOptimizer(model_name="gpt-4o") | model
chain.invoke(messages)
```

---

## Transforms

| Transform | Default | What it does |
|---|---|---|
| `whitespace` | on | Collapse blank-line runs and trailing spaces. Preserves code fences. |
| `dedupe` | on | Drop duplicate paragraphs and repeated sentences. |
| `few_shot` | on | Prune the back half of an oversized few-shot block. |
| `drop_context` | opt-in | Drop context chunks with low overlap to the query. Capped and never empties context. |
| `compress` | opt-in | Route text through a caller-supplied LLM compressor, with a safety guard. |

The default set is `{whitespace, dedupe, few_shot}`. Pass
`enable=tea.AGGRESSIVE_TRANSFORMS` to add `drop_context` and `compress`.
Compression only runs if you also pass a `compressor` callable:

```python
def my_compressor(text: str, target_ratio: float) -> str:
    # call any model to shorten `text` to about target_ratio of its length
    ...

result = tea.optimize(
    prompt, query=q, model="gpt-4o",
    enable=tea.AGGRESSIVE_TRANSFORMS, compressor=my_compressor,
)
```

Expect 15 to 35 per cent reduction on bloated prompts from the deterministic
transforms alone. The LLM compressor goes further at the cost of one extra
model call.

---

## Command line

```bash
# Score a prompt
python scripts/score.py --prompt-file prompt.txt --query "..." --model gpt-4o

# Optimise a prompt (safe transforms)
python scripts/optimize.py --prompt-file prompt.txt --query "..."

# Optimise with relevance-based context dropping
python scripts/optimize.py --prompt-file prompt.txt --query "..." --aggressive

# Optimise a chat-messages JSON file
python scripts/optimize.py --messages-file chat.json --model gpt-4o
```

Both scripts print a JSON report. `optimize.py` also writes the optimised
prompt to `--out-file` when given.

---

## How it works

TEA assigns each request a composite score:

```
S(P) = a * TokenEff + b * Quality - c * (Cost / Cost_max) - d * (1 - Util)
```

where `TokenEff` is quality-weighted output per input token, `Util` is the
fraction of context the model actually used, and the weights `(a, b, c, d)`
sum to 1. The optimiser then searches a set of safe text transforms for the
variant that raises `S` without dropping quality below a floor. The full
derivation, including the closed-model attention path, lives in the product
brief that ships with the wider Optimum AI project.

The relevance signal in this open release is lexical overlap between each
context chunk and the query. It is a coarse but safe proxy for attention: it
errs toward keeping a chunk rather than dropping a useful one.

---

## Safety

- Deterministic transforms never change meaning. They remove repetition,
  boilerplate, and clearly off-topic context.
- `drop_context` keeps the highest-overlap chunk and removes at most 70 per
  cent of the context in a single pass, so a misjudgement by the lexical proxy
  cannot gut the prompt.
- The LLM compressor is opt-in and bounded. Its output is rejected if it
  collapses the text below a floor or fails to shrink it.
- If a compressor raises, TEA catches it and keeps the deterministic result.

---

## Testing

```bash
python -m tea._selftest      # 14 functional checks
python -m tea._edgetest      # 20 edge-case checks
```

---

## Project layout

```
.
├── SKILL.md                     Claude Code skill manifest
├── README.md                    this file
├── scripts/
│   ├── score.py                 measurement CLI
│   └── optimize.py              optimisation CLI
└── tea/                         the importable package
    ├── __init__.py              public API: optimize(), score()
    ├── optimizer.py             deterministic transforms + LLM hook
    ├── tokens.py                token counting and cost
    ├── _selftest.py             functional self-test
    ├── _edgetest.py             edge-case test
    └── integrations/            openai, anthropic, langchain, crewai, autogen
```

---

## License

MIT. See [LICENSE](LICENSE).

---

Built by [Optimum AI](https://www.optimumai.in).
