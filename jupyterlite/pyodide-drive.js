/* Gen's Pyodide loader: publish DriveFS writes at SQLite's sync boundary. */
importScripts('https://cdn.jsdelivr.net/pyodide/v0.29.4/full/pyodide.js');

(() => {
  const originalLoadPyodide = self.loadPyodide;
  self.loadPyodide = async (options) => {
    const pyodide = await originalLoadPyodide({
      ...options,
      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.29.4/full/',
    });
    const filesystem = pyodide.FS;
    const originalMount = filesystem.mount;
    filesystem.mount = function (type, options, mountpoint) {
      if (mountpoint === '/drive') {
        if (!type.API?.put || !type.API?.mknod || !type.realPath || !type.stream_ops?.open || !type.node_ops?.setattr) {
          throw new Error('Gen requires the JupyterLite DriveFS sync adapter for /drive');
        }
        // JupyterLite 0.7's Contents processor recognizes only mode 040777
        // as a directory. Normalize the request, retaining local permissions;
        // otherwise mkdir(..., 0700), including mkdtemp, persists as a file.
        const originalMknod = type.API.mknod;
        type.API.mknod = function (path, mode) {
          return originalMknod.call(this, path, filesystem.isDir(mode) ? 0o40777 : mode);
        };
        // Plot controllers open additional connections in the same kernel.
        // Share file buffers so one connection sees another's writes, as on
        // MEMFS; publishing at fsync alone leaves existing readers stale.
        const openFiles = new Map();
        const operations = type.stream_ops;
        const originalOpen = operations.open;
        const originalClose = operations.close;
        operations.open = function (stream) {
          originalOpen.call(this, stream);
          if (stream.file && filesystem.isFile(stream.node.mode)) {
            const path = type.realPath(stream.node);
            const entry = openFiles.get(path) ?? { file: stream.file, count: 0 };
            stream.file = entry.file;
            entry.count += 1;
            openFiles.set(path, entry);
          }
        };
        operations.close = function (stream) {
          const path = type.realPath(stream.node);
          originalClose.call(this, stream);
          const entry = openFiles.get(path);
          if (entry && --entry.count === 0) {
            openFiles.delete(path);
          }
        };
        const originalSetattr = type.node_ops.setattr;
        type.node_ops.setattr = function (nodeOrStream, attributes) {
          originalSetattr.call(this, nodeOrStream, attributes);
          if (attributes.size !== undefined) {
            const node = nodeOrStream.node ?? nodeOrStream;
            const entry = openFiles.get(type.realPath(node));
            if (entry && attributes.size >= 0) {
              const data = new Uint8Array(attributes.size);
              data.set(entry.file.data.subarray(0, attributes.size));
              entry.file.data = data;
            }
          }
        };
        // Dolt reopens databases while their writing connection is still open.
        // Stock DriveFS publishes only on close, leaving that reopen stale.
        type.stream_ops.fsync = (stream) => {
          const flags = stream.flags ?? stream.shared.flags;
          if (stream.file && filesystem.isFile(stream.node.mode) && (flags & 3) !== 0) {
            type.API.put(type.realPath(stream.node), stream.file);
          }
          return 0;
        };
      }
      return originalMount.call(this, type, options, mountpoint);
    };
    return pyodide;
  };
})();
