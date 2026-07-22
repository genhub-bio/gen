import type { ICoincidentShellWorker, IDriveFSOptions } from '@jupyterlite/cockle';
import { CoincidentShellWorker } from '@jupyterlite/cockle';
import coincident from 'coincident/worker';
import { initDriveFS } from './init_drive_fs';
import { pinPrompt } from './pin_prompt';
import { setThemeEnvironmentVariable } from './shell_worker_theme';
import type { ThemeMode } from './theme';

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

  // `initProxy` binds a fixed set of methods onto the shared coincident proxy (unlike comlink's
  // `expose`, which mirrors every method automatically); this custom method needs the same
  // explicit binding to be callable from the main thread.
  override initProxy(proxy: ICoincidentShellWorker): void {
    super.initProxy(proxy);
    (
      proxy as ICoincidentShellWorker & {
        setThemeEnvironmentVariable(mode: ThemeMode): void;
      }
    ).setThemeEnvironmentVariable = this.setThemeEnvironmentVariable.bind(this);
  }

  setThemeEnvironmentVariable(mode: ThemeMode): void {
    setThemeEnvironmentVariable(this, mode);
  }
}

const proxy = (await coincident()).proxy;
const worker = new GenCoincidentShellWorker();
worker.initProxy(proxy);
