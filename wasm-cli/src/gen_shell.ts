import type { IShell } from '@jupyterlite/cockle';
import { BaseShell } from '@jupyterlite/cockle';
import type { ThemeMode } from './theme';

// Mirrors cockle's own Shell class (src/shell.ts), but loads our shell_worker/shell_worker_comlink
// bundles instead of cockle's stock ones, since those have a real (not no-op) initDriveFS.
export class GenShell extends BaseShell {
  protected override initWorker(options: IShell.IOptions): Worker {
    if (this.workerType === 'coincident') {
      return new Worker(new URL('./shell_worker.ts', import.meta.url), { type: 'module' });
    } else {
      return new Worker(new URL('./shell_worker_comlink.ts', import.meta.url), { type: 'module' });
    }
  }

  // Calls the custom `setThemeEnvironmentVariable` method added to the gen worker classes (see
  // `shell_worker_theme.ts`); `BaseShell` has no public accessor for `_remote`, so this reaches
  // into the same underlying property `themeChange`/`input`/etc. use internally.
  async setThemeEnvironmentVariable(mode: ThemeMode): Promise<void> {
    const remote = (
      this as unknown as {
        _remote?: { setThemeEnvironmentVariable(mode: ThemeMode): Promise<void> };
      }
    )._remote;
    await remote?.setThemeEnvironmentVariable(mode);
  }
}
