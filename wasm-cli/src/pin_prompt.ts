import type { BaseShellWorker } from '@jupyterlite/cockle';

// Cockle's own dark/light background detection (`ShellImpl._handleThemeChange`, run once inside
// `start()`) resets PS1 to its built-in colored "js-shell:" prompt, which would silently undo the
// `PS1` passed in via `IShell.IOptions.environment`. This captures our prompt right after cockle
// applies that environment override (`BaseShellWorker.initialize`, before `start()` runs) and
// freezes `environment.getPrompt()` to keep returning it. Cockle's dark-mode detection itself --
// `COCKLE_DARK_MODE` and error-text coloring -- is untouched; only the rendered prompt text is
// pinned.
export function pinPrompt(worker: BaseShellWorker): void {
  const shellImpl = (worker as unknown as { _shellImpl?: { environment: { getPrompt(): string } } })
    ._shellImpl;
  const environment = shellImpl?.environment;
  if (!environment) {
    return;
  }
  const prompt = environment.getPrompt();
  environment.getPrompt = () => prompt;
}
