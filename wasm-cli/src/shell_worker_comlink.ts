import type { IDriveFSOptions } from '@jupyterlite/cockle';
import { ComlinkShellWorker } from '@jupyterlite/cockle';
import { expose } from 'comlink';
import { initDriveFS } from './init_drive_fs';
import { pinPrompt } from './pin_prompt';
import { setThemeEnvironmentVariable } from './shell_worker_theme';
import type { ThemeMode } from './theme';

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

  // Exposed via comlink like every other public method on this class (see `expose(worker)`
  // below), unlike the coincident worker which needs explicit proxy binding.
  setThemeEnvironmentVariable(mode: ThemeMode): void {
    setThemeEnvironmentVariable(this, mode);
  }
}

const worker = new GenComlinkShellWorker();
expose(worker);
