import type { IDriveFSOptions } from '@jupyterlite/cockle';
import { CoincidentShellWorker } from '@jupyterlite/cockle';
import coincident from 'coincident/worker';
import { initDriveFS } from './init_drive_fs';
import { pinPrompt } from './pin_prompt';

class GenCoincidentShellWorker extends CoincidentShellWorker {
  protected override initDriveFS(options: IDriveFSOptions): void {
    initDriveFS(options);
  }

  override async initialize(
    ...args: Parameters<CoincidentShellWorker['initialize']>
  ): ReturnType<CoincidentShellWorker['initialize']> {
    await super.initialize(...args);
    pinPrompt(this);
  }
}

const proxy = (await coincident()).proxy;
const worker = new GenCoincidentShellWorker();
worker.initProxy(proxy);
