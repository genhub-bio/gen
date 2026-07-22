import type { IDriveFSOptions } from '@jupyterlite/cockle';
import { ComlinkShellWorker } from '@jupyterlite/cockle';
import { expose } from 'comlink';
import { initDriveFS } from './init_drive_fs';
import { pinPrompt } from './pin_prompt';

class GenComlinkShellWorker extends ComlinkShellWorker {
  protected override initDriveFS(options: IDriveFSOptions): void {
    initDriveFS(options);
  }

  override async initialize(
    ...args: Parameters<ComlinkShellWorker['initialize']>
  ): ReturnType<ComlinkShellWorker['initialize']> {
    await super.initialize(...args);
    pinPrompt(this);
  }
}

const worker = new GenComlinkShellWorker();
expose(worker);
