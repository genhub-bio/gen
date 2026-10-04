import type { IDriveFSOptions } from '@jupyterlite/cockle';
import type { IDriveStream, IEmscriptenStreamOps } from '@jupyterlite/services';
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
  // SQLite uses fsync before reopening a database to confirm a Dolt branch head.
  // DriveFS otherwise publishes only on close, so that reopen sees stale bytes.
  const streamOps = driveFS.stream_ops as IEmscriptenStreamOps & {
    fsync(stream: IDriveStream): number;
  };
  streamOps.fsync = (stream: IDriveStream): number => {
    const flags = stream.flags ?? stream.shared.flags;
    if (stream.file && FS.isFile(stream.node.mode) && (flags & 3) !== 0) {
      driveFS.API.put(driveFS.realPath(stream.node), stream.file);
    }
    return 0;
  };
  // Cockle mounts this drive through PROXYFS in each command module. Forward
  // sync there as well; Emscripten's stock PROXYFS silently drops it.
  const proxyFS = fileSystem.PROXYFS as {
    stream_ops: { fsync(stream: { nfd: IDriveStream }): number };
  };
  proxyFS.stream_ops.fsync = (stream: { nfd: IDriveStream }): number => {
    const operations = stream.nfd.node.stream_ops as IEmscriptenStreamOps & {
      fsync?(stream: IDriveStream): number;
    };
    return operations.fsync?.(stream.nfd) ?? 0;
  };
  FS.mount(driveFS, {}, mountpoint);
}
