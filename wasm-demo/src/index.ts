import type { IShell } from '@jupyterlite/cockle';
import { ShellManager } from '@jupyterlite/cockle';
import { ContentsManager } from '@jupyterlab/services';
import { BrowserStorageDrive } from '@jupyterlite/services';
import { FitAddon } from '@xterm/addon-fit';
import { WebLinksAddon } from '@xterm/addon-web-links';
import { WebglAddon } from '@xterm/addon-webgl';
import { Terminal } from '@xterm/xterm';
import localforage from 'localforage';
import { GenShell } from './gen_shell';
import { BrowserLoginBridge, splitOnBeginMessage } from './login_bridge';
import { ServiceWorkerManager } from './service_worker_manager';
import '../style/demo.css';

const DRIVE_MOUNTPOINT = '/drive';
const TERMINAL_FONT_FAMILY = '"JetBrains Mono", monospace';
const TERMINAL_FONT_SIZE = 14;
const FONT_LOAD_TIMEOUT_MS = 2000;

// document.fonts.load() should settle quickly, but the Font Loading API is not
// guaranteed to be present or well-behaved everywhere; race it against a timeout
// so a slow or missing implementation never blocks terminal startup.
async function waitForTerminalFont(): Promise<void> {
  if (!('fonts' in document)) {
    return;
  }
  try {
    await Promise.race([
      Promise.all([
        document.fonts.load(`400 ${TERMINAL_FONT_SIZE}px "JetBrains Mono"`),
        document.fonts.load(`600 ${TERMINAL_FONT_SIZE}px "JetBrains Mono"`),
      ]),
      new Promise<void>(resolve => setTimeout(resolve, FONT_LOAD_TIMEOUT_MS)),
    ]);
  } catch (error) {
    console.warn(
      'JetBrains Mono failed to load via the Font Loading API; falling back to the CSS fallback stack.',
      error
    );
  }
}

// Loads the WebGL addon after the terminal is open. Falls back silently to xterm.js's
// default renderer if WebGL is unavailable, and re-falls-back on context loss without
// recreating the terminal.
function initWebglAddon(term: Terminal): void {
  let webglAddon: WebglAddon | undefined;
  try {
    const addon = new WebglAddon();
    addon.onContextLoss(() => {
      console.warn('xterm.js WebGL context lost; using the fallback renderer.');
      addon.dispose();
      if (webglAddon === addon) {
        webglAddon = undefined;
      }
    });
    term.loadAddon(addon);
    webglAddon = addon;
    console.debug('[gen-wasm-demo] xterm.js WebGL renderer active.');
  } catch (error) {
    console.warn(
      'Unable to initialize the xterm.js WebGL renderer; using the fallback renderer.',
      error
    );
  }
}

async function runDemo(): Promise<void> {
  const baseUrl = window.location.href;

  const fontReady = waitForTerminalFont();

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
  const term = new Terminal({
    fontSize: TERMINAL_FONT_SIZE,
    fontFamily: TERMINAL_FONT_FAMILY,
    fontWeight: '400',
    fontWeightBold: '600',
    lineHeight: 1,
    letterSpacing: 0,
    customGlyphs: true,
    rows: 40,
  });
  const fitAddon = new FitAddon();
  term.loadAddon(fitAddon);

  // The `gen remote login` browser flow prints its login URL as ordinary terminal text (see
  // `outputCallback` below), so it already goes through xterm's web-links auto-detection. Route
  // clicks on that specific URL through `loginBridge.openLoginWindow()` (preserving the popup's
  // `window.opener`, required for the callback page's `postMessage` back); every other link keeps
  // exactly `WebLinksAddon`'s own default behavior (open a blank tab, clear its `opener`, then
  // navigate it -- see `@xterm/addon-web-links`'s default handler, reproduced here since the
  // addon only lets a custom handler *replace* the default, not wrap it).
  term.loadAddon(
    new WebLinksAddon((event, uri) => {
      if (loginBridge.isPendingLoginUrl(uri)) {
        loginBridge.openLoginWindow();
        return;
      }
      const blankTab = window.open();
      if (blankTab) {
        try {
          blankTab.opener = null;
        } catch {
          // Some browsers disallow reassigning `opener`; the tab still opened blank, which is
          // the security property we actually need here.
        }
        blankTab.location.href = uri;
      } else {
        console.warn('Opening link blocked as opener could not be cleared');
      }
    })
  );

  await fontReady;
  term.open(targetDiv);
  initWebglAddon(term);

  // Exposes the terminal instance for browser-level test automation (e.g. Playwright) to read
  // back rendered output via `term.buffer.active`; the WebGL renderer draws to a canvas, so the
  // rendered text is otherwise unavailable through the DOM.
  (window as unknown as { __genTerminal: Terminal }).__genTerminal = term;

  // Buffers output across chunks so a `gen-login` sentinel message split across two
  // `outputCallback` calls is never partially displayed as raw control text.
  let outputBuffer = '';
  function writeTerminalOutput(text: string): void {
    outputBuffer += text;
    for (;;) {
      const split = splitOnBeginMessage(outputBuffer);
      if (!split) {
        term.write(outputBuffer);
        outputBuffer = '';
        return;
      }
      term.write(split.before);
      if (split.begin) {
        loginBridge.begin(split.begin);
      }
      outputBuffer = split.after;
    }
  }

  const loginBridge = new BrowserLoginBridge(
    data => shell.input(data),
    loginUrl => {
      term.write(
        '\r\nThe login popup was blocked.\r\n' +
          `Open this link to sign in: ${loginUrl}\r\n` +
          'Waiting for authentication...\r\n'
      );
    }
  );

  const shellOptions: IShell.IOptions = {
    browsingContextId: serviceWorkerManager.browsingContextId,
    baseUrl,
    wasmBaseUrl: baseUrl,
    mountpoint: DRIVE_MOUNTPOINT,
    outputCallback: writeTerminalOutput,
    shellManager,
    environment: {
      GEN_TERMINAL_ORIGIN: window.location.origin,
      GEN_LOGIN_CALLBACK_URL: new URL('gen-login-callback.html', baseUrl).href,
    },
  };
  const shell = new GenShell(shellOptions);

  // The resize handler must be registered before the first fitAddon.fit() call below
  // (via resizeObserver.observe) so the shell receives the container's real dimensions;
  // xterm.js does not re-fire onResize for a fit() that lands on unchanged dimensions.
  term.onResize(async arg => await shell.setSize({ rows: arg.rows, columns: arg.cols }));
  term.onData(async (data: string) => await shell.input(data));

  const resizeObserver = new ResizeObserver(() => fitAddon.fit());

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
