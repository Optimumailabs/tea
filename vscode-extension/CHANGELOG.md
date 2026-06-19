# Changelog

All notable changes to the Token Efficiency Agent VS Code extension.

## 0.2.0

- Initial release.
- Command: TEA: Optimise selected prompt. Replaces the selection (or the whole
  file when nothing is selected) with the optimised text and reports tokens
  saved.
- Command: TEA: Score selected prompt. Shows the efficiency score without
  editing.
- Right-click context-menu entries when text is selected.
- Settings: `tea.pythonPath`, `tea.model`, `tea.aggressive`, `tea.logDir`.
- Wraps the `token-efficiency-agent` Python package, which must be installed in
  the configured interpreter.
