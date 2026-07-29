import type { IDriveFSOptions } from '@jupyterlite/cockle';
import { DriveFS } from '@jupyterlite/services';

// Ported from @jupyterlite/terminal's worker.ts initDriveFS override, adapted to cockle's
// IDriveFSOptions/IFileSystem shape (structurally identical to @jupyterlite/services' own
// DriveFS.IOptions minus driveName, which is always '' here since there's only one drive).
export function initDriveFS(options: IDriveFSOptions): void {
  const { baseUrl, browsingContextId, fileSystem, mountpoint } = options;
  if (mountpoint === '' || baseUrl === undefined || browsingContextId === undefined) {
    console.warn('gen shell worker not connected to shared drive');
    return;
  }

  const { FS, ERRNO_CODES, PATH } = fileSystem;
  const driveFS = new DriveFS({
    FS,
    PATH,
    ERRNO_CODES,
    baseUrl,
    driveName: '',
    mountpoint,
    browsingContextId,
  });
  FS.mount(driveFS, {}, mountpoint);
}
