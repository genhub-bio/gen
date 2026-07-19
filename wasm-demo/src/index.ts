import type { IShell } from '@jupyterlite/cockle';
import { ShellManager } from '@jupyterlite/cockle';
import { ContentsManager } from '@jupyterlab/services';
import { BrowserStorageDrive } from '@jupyterlite/services';
import { FitAddon } from '@xterm/addon-fit';
import { Terminal } from '@xterm/xterm';
import localforage from 'localforage';
import { GenShell } from './gen_shell';
import { ServiceWorkerManager } from './service_worker_manager';
import '../style/demo.css';

const DRIVE_MOUNTPOINT = '/drive';

async function runDemo(): Promise<void> {
  const baseUrl = window.location.href;

  const drive = new BrowserStorageDrive({ localforage, storageName: 'gen wasm demo' });
  const contentsManager = new ContentsManager({ defaultDrive: drive });

  const serviceWorkerManager = new ServiceWorkerManager({
    contents: contentsManager,
    workerUrl: new URL('service-worker.js', baseUrl).href,
  });

  const shellManager = new ShellManager();
  serviceWorkerManager.registerStdinHandler(
    'terminal',
    shellManager.handleStdin.bind(shellManager)
  );
  await serviceWorkerManager.ready;
  // Give the newly-activated service worker time to claim this page as a client (clients.claim()
  // in service-worker.ts is asynchronous relative to registration completing).
  await new Promise(resolve => setTimeout(resolve, 100));

  const targetDiv = document.getElementById('targetdiv')!;
  const term = new Terminal({ fontSize: 14, rows: 40 });
  const fitAddon = new FitAddon();
  term.loadAddon(fitAddon);

  const shellOptions: IShell.IOptions = {
    browsingContextId: serviceWorkerManager.browsingContextId,
    baseUrl,
    wasmBaseUrl: baseUrl,
    mountpoint: DRIVE_MOUNTPOINT,
    outputCallback: (text: string) => term.write(text),
    shellManager,
  };
  const shell = new GenShell(shellOptions);

  term.onResize(async arg => await shell.setSize({ rows: arg.rows, columns: arg.cols }));
  term.onData(async (data: string) => await shell.input(data));

  const resizeObserver = new ResizeObserver(() => fitAddon.fit());

  term.open(targetDiv);
  await shell.start();
  resizeObserver.observe(targetDiv);

  setupUpload(contentsManager);
}

function setupUpload(contentsManager: ContentsManager): void {
  const input = document.getElementById('upload-input') as HTMLInputElement;
  const status = document.getElementById('upload-status')!;

  input.addEventListener('change', () => {
    void (async () => {
      const file = input.files?.[0];
      input.value = '';
      if (!file) {
        return;
      }

      const content = await file.text();
      await contentsManager.save(file.name, { type: 'file', format: 'text', content });
      status.textContent = `Uploaded to ${DRIVE_MOUNTPOINT}/${file.name}`;
    })().catch(err => {
      console.error('Upload failed', err);
      status.textContent = `Upload failed: ${err}`;
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  void runDemo();
});
