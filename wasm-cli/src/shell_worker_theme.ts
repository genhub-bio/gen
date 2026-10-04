import type { BaseShellWorker } from '@jupyterlite/cockle';
import type { ThemeMode } from './theme';

// Cockle's `IShell` has no public API for updating an already-started shell's environment; the
// only theme-related hook (`themeChange`) updates cockle's own `COCKLE_DARK_MODE` var and
// error-text coloring, not arbitrary variables. This reaches into the same worker-internal
// `_shellImpl.environment` map that `pinPrompt` reads (see `pin_prompt.ts`) so the light/dark
// toggle can keep `GEN_THEME` in sync; `gen` processes started after a toggle pick up the new
// value, since environment is copied into each external command when it runs.
export function setThemeEnvironmentVariable(worker: BaseShellWorker, mode: ThemeMode): void {
  const shellImpl = (worker as unknown as { _shellImpl?: { environment: Map<string, string> } })
    ._shellImpl;
  shellImpl?.environment.set('GEN_THEME', mode);
}
