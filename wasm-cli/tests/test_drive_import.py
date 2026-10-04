"""Regress fresh database writes through Cockle's DriveFS/PROXYFS worker.

Build with make wasm first. Requires Python Playwright and Chromium.
"""

import argparse
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
BUFFER = """() => {
    const buffer = window.__genTerminal.buffer.active;
    return Array.from({length: buffer.length}, (_, index) =>
        buffer.getLine(index)?.translateToString(true) ?? '').join('\\n').trimEnd();
}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4505)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(tempfile.mkdtemp(prefix="gen-drive-test-")),
    )
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    origin = f"http://127.0.0.1:{args.port}"
    server = subprocess.Popen(
        [
            str(ROOT / "wasm-cli/node_modules/.bin/static-handler"),
            "--coi",
            "--port",
            str(args.port),
            "dist/",
        ],
        cwd=ROOT / "wasm-cli",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(60):
            try:
                urlopen(origin, timeout=1).close()
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise AssertionError("local server should start")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(origin)
            page.wait_for_selector(".xterm", timeout=30000)
            page.wait_for_function(
                "window.__genTerminal && (" + BUFFER + ")().endsWith('>')"
            )
            page.click(".xterm")

            def command(text, expected=None):
                page.keyboard.type(text)
                page.keyboard.press("Enter")
                page.wait_for_function(
                    "([command, expected]) => { const text = (" + BUFFER + ")(); "
                    "const output = text.slice(text.lastIndexOf('> ' + command)); "
                    "return output.includes(command + '\\n') && output.trimEnd().endsWith('>') "
                    "&& (!expected || output.includes(expected)); }",
                    arg=[text, expected],
                    timeout=60000,
                )
                output = page.evaluate(BUFFER)
                assert "panicked" not in output and "unexpected error" not in output, (
                    output
                )
                print(text + ": passed", flush=True)
                return output

            name = f"regression{time.time_ns()}"
            command("mkdir " + name)
            command("cd " + name)
            command("gen init", "Gen repository initialized")
            page.set_input_files("#upload-input", str(ROOT / "fixtures/simple.fa"))
            page.wait_for_function(
                "document.getElementById('upload-status').textContent.includes('simple.fa')"
            )
            command(
                "gen import fasta /drive/simple.fa --sample testsample",
                "Fasta imported.",
            )
            command("gen list-samples", "testsample")
            output = command("gen operations", "Apply Gen schema migrations")
            assert re.search(r"m123: \d+ changes\.", output), output
            page.screenshot(path=str(args.artifacts / "import-history.png"))
            (args.artifacts / "terminal.txt").write_text(output)
            # A new page must read the repository flushed by previous commands.
            page.reload()
            page.wait_for_selector(".xterm", timeout=30000)
            page.wait_for_function("(" + BUFFER + ")().endsWith('>')")
            page.click(".xterm")
            command("cd " + name)
            command("gen list-samples", "testsample")
            page.screenshot(path=str(args.artifacts / "after-reload.png"))
            browser.close()
        print(
            f"PASS: fresh import, sample/history reads and reload; screenshots in {args.artifacts}"
        )
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
