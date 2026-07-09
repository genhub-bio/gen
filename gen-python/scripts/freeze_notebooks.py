"""Execute example notebooks in a real browser and freeze widget outputs to static PNGs.

``jupyter nbconvert --execute`` runs cells headlessly but never mounts the
anywidget frontend, so ``GraphWidget`` outputs stay as inert widget-view
placeholders that render as nothing once viewed outside a live kernel (e.g.
on GitHub). This script drives an actual Jupyter server through a headless
Playwright browser so cells execute normally and each widget mounts its real
canvas. A temporary cell calling ``gen.freeze_all_widgets()`` is appended
before running so freezing happens in-kernel, as part of the same "Run All
Cells" pass that runs the notebook's own cells — no DOM interaction or
guessed settle time needed for it, since the call blocks until the frontend
confirms each widget's canvas has been swapped for a static image. The
helper cell is stripped back out on disk once the notebook is saved.

Usage
-----
::

    .venv/bin/python gen-python/scripts/freeze_notebooks.py
    .venv/bin/python gen-python/scripts/freeze_notebooks.py introduction.ipynb

Run ``make notebook-clear-output`` afterwards before committing routine
changes; only commit the frozen outputs produced by this script when you
intend the rendered notebook to display on GitHub.
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import nbformat
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
TOKEN = "gen-notebook-freeze"
JUPYTER_BIN = str(Path(sys.executable).parent / "jupyter")

# Marks the temporary cell this script injects, so it can find and strip it
# back out after saving without touching any of the notebook's own cells.
FREEZE_CELL_TAG = "gen-freeze-helper"
FREEZE_CELL_SOURCE = "await gen.freeze_all_widgets()"

# These have no DOM/kernel signal to poll for (page load, save-to-disk flush),
# so they stay fixed waits. Widget mounting and the freeze round-trip used to
# need their own guessed waits here too, but now block for real inside the
# injected cell's `await gen.freeze_all_widgets()` instead.
NOTEBOOK_LOAD_SETTLE_SECONDS = 1
CELL_FOCUS_SETTLE_SECONDS = 0.3
SAVE_SETTLE_SECONDS = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/api/status?token={TOKEN}"
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    raise RuntimeError(f"Jupyter server did not start on port {port}")


@contextlib.contextmanager
def jupyter_server(notebook_dir: Path):
    port = _free_port()
    process = subprocess.Popen(
        [
            JUPYTER_BIN,
            "notebook",
            "--no-browser",
            f"--ServerApp.port={port}",
            f"--ServerApp.token={TOKEN}",
            "--ServerApp.ip=127.0.0.1",
            f"--ServerApp.root_dir={notebook_dir}",
            "--ServerApp.disable_check_xsrf=True",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_server(port)
        yield port
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


@contextlib.contextmanager
def freeze_helper_cell(notebook_path: Path):
    """Temporarily append the freeze-all cell to the notebook file on disk."""
    notebook = nbformat.read(notebook_path, as_version=4)
    helper = nbformat.v4.new_code_cell(
        source=FREEZE_CELL_SOURCE, metadata={"tags": [FREEZE_CELL_TAG]}
    )
    notebook.cells.append(helper)
    nbformat.write(notebook, notebook_path)
    try:
        yield
    finally:
        notebook = nbformat.read(notebook_path, as_version=4)
        notebook.cells = [
            cell
            for cell in notebook.cells
            if FREEZE_CELL_TAG not in cell.get("metadata", {}).get("tags", [])
        ]
        nbformat.write(notebook, notebook_path)


def _wait_for_cells_idle(page: Page, timeout: float = 300.0) -> None:
    """Block until cell prompts stop changing and none are mid-execution.

    "Run All Cells" halts on the first error, leaving later cells stuck at
    ``[ ]:`` forever, so waiting for every prompt to be numbered would hang.
    Instead wait for the prompt snapshot to be stable across two polls.
    """
    deadline = time.monotonic() + timeout
    with contextlib.suppress(PlaywrightTimeoutError):
        page.wait_for_selector('[title*="Kernel status: Busy"]', timeout=5000)
    previous: list[str] | None = None
    while time.monotonic() < deadline:
        prompts = [
            p for p in page.locator(".jp-InputArea-prompt").all_inner_texts() if p
        ]
        if prompts and not any("*" in p for p in prompts) and prompts == previous:
            return
        previous = prompts
        time.sleep(1)
    raise TimeoutError("Timed out waiting for notebook cells to finish executing")


def _click_menu_item(page: Page, menu: str, item: str) -> None:
    """Open a top menu bar entry (e.g. "Run") and click one of its items (e.g. "Run All Cells")."""
    page.locator(f".lm-MenuBar-itemLabel:text-is('{menu}')").click()
    page.locator(f".lm-Menu-itemLabel:text-is('{item}')").click()


def _check_for_errors(page: Page, notebook_name: str) -> None:
    text = page.inner_text(".jp-Notebook")
    marker = "Traceback (most recent call last)"
    index = text.find(marker)
    if index != -1:
        raise RuntimeError(
            f"{notebook_name} raised an error while executing:\n{text[index : index + 2000]}"
        )


def freeze_notebook(page: Page, port: int, notebook_name: str) -> None:
    url = f"http://127.0.0.1:{port}/notebooks/{notebook_name}?token={TOKEN}"
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector(".jp-Notebook", timeout=30000)
    time.sleep(NOTEBOOK_LOAD_SETTLE_SECONDS)

    first_cell = page.locator(".jp-CodeCell .jp-InputArea-editor").first
    if first_cell.count() == 0:
        # No code cells to run; nothing to freeze.
        return
    # Focus the notebook so the "Run" menu commands target it.
    first_cell.click()
    time.sleep(CELL_FOCUS_SETTLE_SECONDS)

    _click_menu_item(page, "Run", "Run All Cells")
    _wait_for_cells_idle(page)
    _check_for_errors(page, notebook_name)

    _click_menu_item(page, "File", "Save Notebook")
    time.sleep(SAVE_SETTLE_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Notebook filenames under examples/ to freeze (default: all *.ipynb)",
    )
    args = parser.parse_args()

    notebooks = args.notebooks or sorted(p.name for p in EXAMPLES_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found", file=sys.stderr)
        sys.exit(1)

    failures: list[str] = []
    with jupyter_server(EXAMPLES_DIR) as port:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            for name in notebooks:
                print(f"Freezing {name} ...")
                page = browser.new_page()
                try:
                    with freeze_helper_cell(EXAMPLES_DIR / name):
                        freeze_notebook(page, port, name)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    print(f"  FAILED: {exc}", file=sys.stderr)
                    failures.append(name)
                finally:
                    page.close()
            browser.close()

    if failures:
        print(f"\nFailed to freeze: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
