import type { IShell } from '@jupyterlite/cockle';
import { BaseShell } from '@jupyterlite/cockle';

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
}
