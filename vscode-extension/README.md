# Token Efficiency Agent - VS Code extension

Optimise the selected prompt text in any editor and see how many tokens were
saved, without leaving VS Code. The extension wraps the
[TEA Python package](https://github.com/Optimumailabs/Token-Efficiency-Agent);
it shells out to it, so the package must be installed in the interpreter you
point the extension at.

## Prerequisites

```bash
pip install token-efficiency-agent
```

If `python` on PATH is not the interpreter with the package, set it:

```json
{ "tea.pythonPath": "C:/path/to/python.exe" }
```

## Commands

- **TEA: Optimise selected prompt** - replaces the selection (or the whole
  file if nothing is selected) with the optimised text, then shows tokens
  saved. Also available on the editor right-click menu when text is selected.
- **TEA: Score selected prompt** - shows the efficiency score without editing.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `tea.pythonPath` | `python` | Interpreter with the `tea` package. |
| `tea.model` | `gpt-4o` | Model id for token counting. |
| `tea.aggressive` | `false` | Also drop low-relevance context. |
| `tea.logDir` | `""` | If set, every optimise call is logged here. |

## Run locally during development

```bash
cd vscode-extension
npm install            # pulls @types/vscode and @vscode/vsce
# Press F5 in VS Code with this folder open to launch an Extension Dev Host.
```

## Publish to the VS Code Marketplace

This follows the official guide:
https://code.visualstudio.com/api/working-with-extensions/publishing-extension

These steps need a Microsoft / Azure DevOps account and are done by a human,
once.

1. Install the packaging tool.

   ```bash
   npm install -g @vscode/vsce
   ```

2. Create a Marketplace publisher at
   https://marketplace.visualstudio.com/manage. The publisher ID must match the
   `publisher` field in `package.json` (currently `optimum-ai`). Change both if
   you use a different ID.

3. Create an Azure DevOps Personal Access Token at https://dev.azure.com.
   - Organization: **All accessible organizations**.
   - Scopes: **Custom defined**, then **Marketplace > Manage**.
   - Copy the token; you cannot see it again.

   Note: global PATs retire on 1 December 2026. For long-lived CI, Microsoft
   recommends Entra ID workload identity federation instead (see the CI section
   below).

4. Log in once with the token.

   ```bash
   vsce login optimum-ai
   ```

5. Package and publish.

   ```bash
   vsce package          # produces token-efficiency-agent-0.2.0.vsix
   vsce publish          # or: vsce publish minor  to bump the version
   ```

   To upload manually instead, run only `vsce package` and drag the `.vsix`
   into the publisher page at https://marketplace.visualstudio.com/manage.

## Continuous publishing (optional)

A GitHub Actions workflow that publishes on a tag needs the PAT stored as the
`VSCE_PAT` repository secret. A ready-to-use job is in
`.github/workflows/publish.yml` at the repo root under the `publish-vscode`
job; it runs `vsce publish -p $VSCE_PAT` and is skipped when the secret is
absent.

## Status

The extension is plain JavaScript with no build step. It runs as-is in an
Extension Dev Host and packages with `vsce package`. The only manual,
account-bound steps are the publisher and PAT creation above.
