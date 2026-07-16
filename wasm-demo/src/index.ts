import { Shell, ShellManager } from '@jupyterlite/cockle';
import type { IShell } from '@jupyterlite/cockle';
import { FitAddon } from '@xterm/addon-fit';
import { Terminal } from '@xterm/xterm';
import '../style/demo.css';

async function runDemo(): Promise<void> {
  const baseUrl = window.location.href;
  const shellManager = new ShellManager();
  const browsingContextId = await shellManager.installServiceWorker(baseUrl);

  const targetDiv = document.getElementById('targetdiv')!;
  const term = new Terminal({ fontSize: 14, rows: 40 });
  const fitAddon = new FitAddon();
  term.loadAddon(fitAddon);

  const shellOptions: IShell.IOptions = {
    browsingContextId,
    baseUrl,
    wasmBaseUrl: baseUrl,
    outputCallback: (text: string) => term.write(text),
    shellManager,
  };
  const shell = new Shell(shellOptions);

  term.onResize(async (arg) => await shell.setSize({ rows: arg.rows, columns: arg.cols }));
  term.onData(async (data: string) => await shell.input(data));

  const resizeObserver = new ResizeObserver(() => fitAddon.fit());

  term.open(targetDiv);
  await shell.start();
  resizeObserver.observe(targetDiv);
}

document.addEventListener('DOMContentLoaded', () => {
  void runDemo();
});
