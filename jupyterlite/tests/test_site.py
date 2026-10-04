"""Run the bundled notebook in the actual JupyterLite UI after building the site."""

from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading

from playwright.sync_api import sync_playwright

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))
# Load the local server after adding the site's directory to the module path.
from serve import Handler  # noqa: E402


def run_cell(page, source, marker):
    """Append a cell through the notebook UI and wait for its output."""
    page.locator(".jp-CodeCell .cm-content").last.click()
    page.keyboard.press("Escape")
    page.keyboard.press("b")
    page.keyboard.press("Enter")
    page.keyboard.insert_text(source)
    page.keyboard.press("Shift+Enter")
    page.wait_for_function(
        "marker => Array.from(document.querySelectorAll('.jp-OutputArea'))"
        ".some(area => area.innerText.includes(marker) || area.innerText.includes('Traceback'))",
        arg=marker,
        timeout=180000,
    )
    output = page.locator(".jp-Notebook").inner_text()
    assert "Traceback" not in output, output
    assert marker in output, output


def reopen_code(marker, sample="edited", sequence="ACGTTTGT"):
    return f"""import piplite
await piplite.install('gen==0.3.1')
import gen
from pathlib import Path
root = Path(Path('/drive/gen-demo-path.txt').read_text())
assert str(root).startswith('/drive/')
repository = gen.Repository(str(root))
repository.export_fasta(str(root / 'reopened.fa'), sample='{sample}')
assert '{sequence}' in (root / 'reopened.fa').read_text()
assert len(repository.get_operations()) >= 2
print('{marker}')
"""


def main():
    artifacts = Path(tempfile.mkdtemp(prefix="gen-jupyterlite-site-"))
    # The notebook's disposable capability URL deliberately uses this fixed port.
    server = ThreadingHTTPServer(
        ("127.0.0.1", 4502), partial(Handler, directory=str(SITE / "_output"))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.on(
                "pageerror", lambda error: print(f"Browser error: {error}", flush=True)
            )
            page.on(
                "console",
                lambda message: print(message.text, flush=True)
                if message.type == "error"
                else None,
            )
            page.goto("http://127.0.0.1:4502/lab/index.html?path=Gen.ipynb")
            page.wait_for_selector(".jp-Notebook", timeout=90000)
            page.get_by_text("Run", exact=True).first.click()
            page.get_by_text("Run All Cells", exact=True).click()
            page.wait_for_function(
                "document.body.innerText.includes('Download gen-demo.zip') || "
                "document.body.innerText.includes('Traceback')",
                timeout=180000,
            )
            # Save and inspect all outputs, including cells outside the rendered viewport.
            page.keyboard.press("ControlOrMeta+s")
            saved_output = """async () => {
                const connection = await new Promise(resolve => {
                    const request = indexedDB.open('JupyterLite Storage - /');
                    request.onsuccess = () => resolve(request.result);
                });
                const notebook = await new Promise(resolve => {
                    const request = connection.transaction('files').objectStore('files').get('Gen.ipynb');
                    request.onsuccess = () => resolve(request.result);
                });
                connection.close();
                return notebook?.content?.cells?.flatMap(cell =>
                    (cell.outputs ?? []).map(output => output.text ?? output.data?.['text/plain'] ?? '')
                ).join('\\n') ?? '';
            }"""
            page.wait_for_function(
                "async () => (await ("
                + saved_output
                + ")()).includes('Download gen-demo.zip')",
                timeout=30000,
            )
            output = page.evaluate(saved_output)
            (artifacts / "notebook.txt").write_text(output)
            page.screenshot(path=str(artifacts / "notebook.png"), full_page=True)
            assert "Traceback" not in output, output
            for expected in [
                "0.3.1",
                "ACGTACGT",
                "ACGTTTGT",
                "Sequences inserted",
                "GenHub request failed with HTTP 422",
                "Download gen-demo.zip",
            ]:
                assert expected in output, expected
            assert page.get_by_text("gen-demo.zip", exact=True).count() > 0
            page.wait_for_function(
                "document.querySelectorAll('.jp-OutputArea canvas').length >= 2",
                timeout=60000,
            )
            plot = page.locator(".jp-OutputArea canvas").last
            assert plot.evaluate("canvas => canvas.width > 0 && canvas.height > 0")
            plot.screenshot(path=str(artifacts / "sequence-graph.png"))
            before_zoom = plot.evaluate("canvas => canvas.toDataURL()")
            # Short sequences look identical at full and truncated detail;
            # two steps reach the minimal view and visibly exercise comms.
            page.get_by_title("Zoom out (-)", exact=True).last.click()
            page.get_by_title("Zoom out (-)", exact=True).last.click()
            page.wait_for_function(
                "before => Array.from(document.querySelectorAll('.jp-OutputArea canvas'))"
                ".at(-1).toDataURL() !== before",
                arg=before_zoom,
                timeout=30000,
            )
            # A reload creates a fresh worker while retaining this origin's IndexedDB.
            print("PASS: initial /drive workflow", flush=True)
            page.reload()
            page.wait_for_selector(".jp-Notebook", timeout=90000)
            run_cell(
                page,
                reopen_code("PASS: persisted repository reopened"),
                "PASS: persisted repository reopened",
            )
            run_cell(
                page,
                """repository.update_with_sequence('TTTTACGT', 'edited', 'reloaded', 'tiny:0-8')
repository.export_fasta(str(root / 'reloaded.fa'), sample='reloaded')
assert 'TTTTACGT' in (root / 'reloaded.fa').read_text()
import gc
del repository
gc.collect()
print('PASS: persisted repository edited')
""",
                "PASS: persisted repository edited",
            )
            print("PASS: reload and subsequent edit", flush=True)
            page.get_by_text("Kernel", exact=True).first.click()
            page.get_by_text("Restart Kernel…", exact=True).click()
            page.get_by_role(
                "button", name="Confirm Kernel Restart", exact=True
            ).click()
            run_cell(
                page,
                reopen_code("PASS: kernel restart reopened", "reloaded", "TTTTACGT"),
                "PASS: kernel restart reopened",
            )
            page.screenshot(path=str(artifacts / "persistence.png"), full_page=True)
            browser.close()
        print(
            f"PASS: bundled wheel, local edits/history, interactive sequence plots, browser HTTP, archive export, reload and kernel restart; artifacts {artifacts}"
        )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
