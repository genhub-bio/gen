/* Exercise the loader's sync boundary without a browser or CDN download. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const loader = fs.readFileSync(path.join(__dirname, '../pyodide-drive.js'), 'utf8');

async function setup() {
  const published = new Map();
  const filesystem = {
    isFile: (mode) => mode === 'file',
    isDir: (mode) => (mode & 0o170000) === 0o40000,
    mount(type) { return type; },
  };
  const drive = {
    API: {
      put(name, file) { published.set(name, file.data.slice()); },
      mknod(name, mode) { return { name, mode }; },
    },
    realPath: (node) => node.name,
    node_ops: {
      setattr(node, attributes) {
        if (attributes.size !== undefined) {
          const data = new Uint8Array(attributes.size);
          data.set(published.get(node.name).subarray(0, attributes.size));
          published.set(node.name, data);
        }
      },
    },
    stream_ops: {
      open(stream) {
        stream.file = { data: published.get(stream.node.name).slice() };
      },
      close(stream) {
        if ((stream.flags & 3) !== 0) {
          published.set(stream.node.name, stream.file.data.slice());
        }
        stream.file = undefined;
      },
    },
  };
  const context = {
    Uint8Array,
    importScripts() {},
    self: { loadPyodide: async () => ({ FS: filesystem }) },
  };
  vm.runInNewContext(loader, context);
  await context.self.loadPyodide({});
  filesystem.mount(drive, {}, '/drive');
  return { drive, filesystem, published };
}

test('test_sync_publishes_writes_before_close_and_preserves_open_buffer', async () => {
  const { drive, published } = await setup();
  const stream = {
    flags: 2,
    node: { mode: 'file', name: 'database' },
    file: { data: new Uint8Array([1, 2]) },
  };
  assert.equal(drive.stream_ops.fsync(stream), 0);
  assert.deepEqual(published.get('database'), new Uint8Array([1, 2]));
  stream.file.data[1] = 3;
  drive.stream_ops.fsync(stream);
  assert.deepEqual(published.get('database'), new Uint8Array([1, 3]));
  assert.ok(stream.file);
});

test('test_sync_skips_readonly_and_directory_streams', async () => {
  const { drive, published } = await setup();
  drive.stream_ops.fsync({ shared: { flags: 0 }, node: { mode: 'file' }, file: {} });
  drive.stream_ops.fsync({ flags: 2, node: { mode: 'directory' }, file: {} });
  assert.equal(published.size, 0);
});

test('test_sync_propagates_persistence_failures', async () => {
  const { drive } = await setup();
  drive.API.put = () => { throw new Error('storage failed'); };
  assert.throws(() => drive.stream_ops.fsync({
    flags: 2, node: { mode: 'file' }, file: {},
  }), /storage failed/);
});

test('test_other_mounts_are_unchanged_and_unknown_drive_fails_explicitly', async () => {
  const { filesystem } = await setup();
  const memory = {};
  assert.equal(filesystem.mount(memory, {}, '/tmp'), memory);
  assert.throws(() => filesystem.mount(memory, {}, '/drive'), /sync adapter/);
});

test('test_private_directories_remain_directories_in_contents_storage', async () => {
  const { drive } = await setup();
  assert.equal(drive.API.mknod('private', 0o40700).mode, 0o40777);
  assert.equal(drive.API.mknod('file', 0o100600).mode, 0o100600);
});

test('test_open_connections_share_writes_and_truncation', async () => {
  const { drive, published } = await setup();
  published.set('database', new Uint8Array([1, 2, 3]));
  const node = { mode: 'file', name: 'database' };
  const writer = { flags: 2, node };
  const reader = { flags: 0, node };
  drive.stream_ops.open(writer);
  drive.stream_ops.open(reader);
  writer.file.data[1] = 9;
  assert.deepEqual(reader.file.data, new Uint8Array([1, 9, 3]));
  drive.node_ops.setattr(node, { size: 2 });
  assert.deepEqual(reader.file.data, new Uint8Array([1, 9]));
  drive.stream_ops.close(reader);
  assert.deepEqual(writer.file.data, new Uint8Array([1, 9]));
  drive.stream_ops.close(writer);
  // After all handles close, new opens must reload the Contents API bytes.
  published.set('database', new Uint8Array([4, 5]));
  drive.stream_ops.open(reader);
  assert.deepEqual(reader.file.data, new Uint8Array([4, 5]));
});
