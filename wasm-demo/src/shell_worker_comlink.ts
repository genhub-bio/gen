import type { IDriveFSOptions } from '@jupyterlite/cockle';
import { ComlinkShellWorker } from '@jupyterlite/cockle';
import { expose } from 'comlink';
import { initDriveFS } from './init_drive_fs';

class GenComlinkShellWorker extends ComlinkShellWorker {
  protected override initDriveFS(options: IDriveFSOptions): void {
    initDriveFS(options);
  }
}

const worker = new GenComlinkShellWorker();
expose(worker);
