// Token Efficiency Agent - VS Code extension (plain JS, no build step).
//
// It shells out to the installed `tea` Python package. Install the package
// first in the interpreter configured by `tea.pythonPath`:
//
//     pip install token-efficiency-agent
//
// Two commands:
//   TEA: Optimise selected prompt  -> replaces the selection with the optimised
//                                     text and shows tokens saved.
//   TEA: Score selected prompt     -> shows the efficiency score without editing.

const vscode = require("vscode");
const { execFile } = require("child_process");
const os = require("os");
const path = require("path");
const fs = require("fs");

function runTea(pyPath, args, input) {
  // Write the selection to a temp file and pass it to the package CLIs.
  return new Promise((resolve, reject) => {
    const tmp = path.join(os.tmpdir(), `tea_sel_${Date.now()}.txt`);
    fs.writeFileSync(tmp, input, "utf8");
    const fullArgs = ["-m", "tea.cli", ...args, "--prompt-file", tmp];
    // tea.cli's __main__ runs optimize_main; for score we call the module fn.
    execFile(pyPath, fullArgs, { maxBuffer: 10 * 1024 * 1024 }, (err, stdout, stderr) => {
      try { fs.unlinkSync(tmp); } catch (e) { /* ignore */ }
      if (err) {
        reject(new Error(stderr || err.message));
        return;
      }
      resolve({ stdout, stderr, tmp });
    });
  });
}

// Because tea.cli's module entry runs optimize_main, we invoke the console
// scripts directly instead. Resolve them relative to the interpreter.
function runConsole(pyPath, script, args) {
  return new Promise((resolve, reject) => {
    // `python -c` wrapper keeps this independent of PATH for the console script.
    const code =
      script === "optimize"
        ? "import sys; from tea.cli import optimize_main; sys.exit(optimize_main())"
        : "import sys; from tea.cli import score_main; sys.exit(score_main())";
    execFile(pyPath, ["-c", code, ...args], { maxBuffer: 10 * 1024 * 1024 }, (err, stdout, stderr) => {
      if (err && !stdout) { reject(new Error(stderr || err.message)); return; }
      resolve({ stdout, stderr });
    });
  });
}

function activate(context) {
  const cfg = () => vscode.workspace.getConfiguration("tea");

  const optimise = vscode.commands.registerCommand("tea.optimizeSelection", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) { vscode.window.showWarningMessage("TEA: no active editor."); return; }
    const sel = editor.selection;
    const text = editor.document.getText(sel.isEmpty ? undefined : sel);
    if (!text.trim()) { vscode.window.showWarningMessage("TEA: nothing selected."); return; }

    const c = cfg();
    const tmp = path.join(os.tmpdir(), `tea_sel_${Date.now()}.txt`);
    fs.writeFileSync(tmp, text, "utf8");
    const args = ["--prompt-file", tmp, "--model", c.get("model") || "gpt-4o", "--json-only"];
    if (c.get("aggressive")) args.push("--aggressive");
    const logDir = (c.get("logDir") || "").trim();
    if (logDir) args.push("--log", logDir);
    const outFile = tmp + ".out";
    args.push("--out-file", outFile);

    try {
      const { stdout } = await runConsole(c.get("pythonPath") || "python", "optimize", args);
      const report = JSON.parse(stdout);
      const optimized = fs.readFileSync(outFile, "utf8");
      await editor.edit((eb) => {
        eb.replace(sel.isEmpty ? fullRange(editor.document) : sel, optimized);
      });
      vscode.window.showInformationMessage(
        `TEA: ${report.tokens_before} -> ${report.tokens_after} tokens ` +
        `(saved ${report.tokens_saved}, ${report.reduction_pct}%).`
      );
    } catch (e) {
      vscode.window.showErrorMessage("TEA optimise failed: " + e.message);
    } finally {
      try { fs.unlinkSync(tmp); } catch (e) {}
      try { fs.unlinkSync(outFile); } catch (e) {}
    }
  });

  const scoreCmd = vscode.commands.registerCommand("tea.scoreSelection", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) { vscode.window.showWarningMessage("TEA: no active editor."); return; }
    const sel = editor.selection;
    const text = editor.document.getText(sel.isEmpty ? undefined : sel);
    if (!text.trim()) { vscode.window.showWarningMessage("TEA: nothing selected."); return; }

    const c = cfg();
    const tmp = path.join(os.tmpdir(), `tea_score_${Date.now()}.txt`);
    fs.writeFileSync(tmp, text, "utf8");
    try {
      const { stdout } = await runConsole(c.get("pythonPath") || "python", "score",
        ["--prompt-file", tmp, "--model", c.get("model") || "gpt-4o"]);
      const report = JSON.parse(stdout);
      const s = report.score;
      vscode.window.showInformationMessage(
        `TEA score S=${s.S} (tokens ${report.tokens.total_prompt}, ` +
        `cost $${report.cost.per_request_usd}).`
      );
    } catch (e) {
      vscode.window.showErrorMessage("TEA score failed: " + e.message);
    } finally {
      try { fs.unlinkSync(tmp); } catch (e) {}
    }
  });

  context.subscriptions.push(optimise, scoreCmd);
}

function fullRange(document) {
  const last = document.lineCount - 1;
  return new vscode.Range(0, 0, last, document.lineAt(last).text.length);
}

function deactivate() {}

module.exports = { activate, deactivate };
