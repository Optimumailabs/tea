# Token Efficiency Agent - VS Code extension

Optimise the selected prompt text in any editor and see how many tokens were
saved, without leaving VS Code. The extension is a thin wrapper around the
[TEA Python package](../README.md); it shells out to it, so the package must be
installed in the interpreter you point the extension at.

## Status

This is a working stub written in plain JavaScript (no TypeScript build step).
It runs as-is when loaded as an extension, but packaging it into a `.vsix` for
the Marketplace needs Node tooling that is not bundled here.

## Prerequisites

```bash
pip install token-efficiency-agent
```

Set the interpreter in settings if `python` on PATH is not the one with the
package:

```json
{ "tea.pythonPath": "C:/path/to/python.exe" }
```

## Commands

- **TEA: Optimise selected prompt** - replaces the selection (or the whole
  file if nothing is selected) with the optimised text, then shows tokens
  saved in a notification.
- **TEA: Score selected prompt** - shows the efficiency score without editing.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `tea.pythonPath` | `python` | Interpreter with the `tea` package. |
| `tea.model` | `gpt-4o` | Model id for token counting. |
| `tea.aggressive` | `false` | Also drop low-relevance context. |
| `tea.logDir` | `""` | If set, every optimise call is logged here. |

## Build and run locally

```bash
cd vscode-extension
npm install          # pulls @types/vscode and @vscode/vsce
# Press F5 in VS Code with this folder open to launch an Extension Dev Host,
# or package a .vsix:
npx vsce package
```

The `npm install` and `vsce package` steps need Node and the VS Code extension
toolchain. They are not run as part of the Python package build.
