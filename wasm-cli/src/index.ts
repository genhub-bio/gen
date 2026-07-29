import type { IShell } from '@jupyterlite/cockle';
import { ansi, ShellManager } from '@jupyterlite/cockle';
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
import { buildShellEnvironment, buildTerminalOptions, TERMINAL_FONT_SIZE } from './terminal_setup';
import type { ThemeMode } from './theme';
import { base16PaletteForMode, selectThemeMode, xtermThemeForMode } from './theme';
import '../style/demo.css';

const DRIVE_MOUNTPOINT = '/drive';
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
    console.debug('[gen-wasm-cli] xterm.js WebGL renderer active.');
  } catch (error) {
    console.warn(
      'Unable to initialize the xterm.js WebGL renderer; using the fallback renderer.',
      error
    );
  }
}

// Also called from `setupThemeToggle` whenever the reader flips the light/dark switch.
function applyThemeBackground(mode: ThemeMode): void {
  const background = base16PaletteForMode(mode).base00;
  document.documentElement.style.colorScheme = mode;
  document.body.style.backgroundColor = background;
  for (const id of ['terminal-wrap', 'targetdiv']) {
    const element = document.getElementById(id);
    if (element) {
      element.style.backgroundColor = background;
    }
  }
}

async function runDemo(): Promise<void> {
  const baseUrl = window.location.href;
  const themeMode = selectThemeMode();
  // Set before the terminal opens so there is no black/white flash while xterm initializes.
  applyThemeBackground(themeMode);

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
  const term = new Terminal(buildTerminalOptions(themeMode));
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
    environment: buildShellEnvironment(themeMode, {
      GEN_TERMINAL_ORIGIN: window.location.origin,
      GEN_LOGIN_CALLBACK_URL: new URL('gen-login-callback.html', baseUrl).href,
      // Cockle's default `js-shell:` PS1 is replaced with a plain prompt for this terminal;
      // the `>` is colored so it stands out against the rest of the prompt line.
      PS1: `${ansi.styleBrightBlue}>${ansi.styleReset} `,
    }),
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
  setupHelpLink(shell);
  setupThemeToggle(term, shell, themeMode);
}

// Wires the intro-bar light/dark switch to live-update the terminal's rendered theme, the page
// background, and the shell's `GEN_THEME` environment variable together, so all three always
// agree on which flavor is active.
function setupThemeToggle(term: Terminal, shell: GenShell, initialMode: ThemeMode): void {
  const input = document.getElementById('theme-toggle-input') as HTMLInputElement;
  input.checked = initialMode === 'light';
  input.addEventListener('change', () => {
    const mode: ThemeMode = input.checked ? 'light' : 'dark';
    applyThemeBackground(mode);
    term.options.theme = xtermThemeForMode(mode);
    void shell.themeChange(mode === 'dark');
    void shell.setThemeEnvironmentVariable(mode);
  });
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

      const fileName = file.name.split(/[\\/]/).pop() || 'upload';
      const content = await file.text();
      await contentsManager.save(fileName, { type: 'file', format: 'text', content });
      status.textContent = `Added ${fileName}`;
    })().catch(err => {
      console.error('Adding the file failed', err);
      status.textContent = `Adding the file failed: ${err}`;
    });
  });
}

const HELP_COMMAND = 'gen --help';
// Gives the reader a moment to read the typed-out command before it runs, rather than having it
// appear to execute instantly.
const HELP_COMMAND_RUN_DELAY_MS = 900;

// Types `gen --help` into the shell as though the reader had entered it themselves, pausing
// before submitting it so newcomers can see the command before its output appears.
function setupHelpLink(shell: GenShell): void {
  const link = document.getElementById('help-link') as HTMLAnchorElement;
  link.addEventListener('click', event => {
    event.preventDefault();
    void (async () => {
      await shell.input(HELP_COMMAND);
      await new Promise(resolve => setTimeout(resolve, HELP_COMMAND_RUN_DELAY_MS));
      await shell.input('\r');
    })();
  });
}

document.addEventListener('DOMContentLoaded', () => {
  void runDemo().catch(error => {
    console.error('Unable to start the Gen terminal', error);
    const targetDiv = document.getElementById('targetdiv');
    if (targetDiv) {
      targetDiv.textContent =
        'Unable to start the Gen terminal. This browser must support service workers, and the page must be served from a secure origin.';
      targetDiv.setAttribute('role', 'alert');
    }
  });
});
