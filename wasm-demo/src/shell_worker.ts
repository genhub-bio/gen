import type { IDriveFSOptions } from '@jupyterlite/cockle';
import { CoincidentShellWorker } from '@jupyterlite/cockle';
import coincident from 'coincident/worker';
import { initDriveFS } from './init_drive_fs';

class GenCoincidentShellWorker extends CoincidentShellWorker {
  protected override initDriveFS(options: IDriveFSOptions): void {
    initDriveFS(options);
  }
}

const proxy = (await coincident()).proxy;
const worker = new GenCoincidentShellWorker();
worker.initProxy(proxy);
