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
            page.goto("http://127.0.0.1:4502/lab/index.html?path=Gen.ipynb")
            page.wait_for_selector(".jp-Notebook", timeout=90000)
            page.get_by_text("Run", exact=True).first.click()
            page.get_by_text("Run All Cells", exact=True).click()
            page.wait_for_function(
                "document.body.innerText.includes('Download gen-demo.zip') || "
                "document.body.innerText.includes('Traceback')",
                timeout=180000,
            )
            output = page.locator(".jp-Notebook").inner_text()
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
            browser.close()
        print(
            f"PASS: bundled wheel, local edits/history, browser HTTP and archive export; artifacts {artifacts}"
        )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
