"""Execute example notebooks with text graph output instead of embedded PNGs, and strip the
per-run details so re-running leaves them unchanged."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import nbformat

PYTHON_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PYTHON_DIR / "examples"


# Per-user temporary directories such as /var/folders/ab/xyz/T/tmpk3j1 differ on every run.
TEMPORARY_PATH = re.compile(
    r"(?:/private)?/var/folders/[^/\s\"']+/[^/\s\"']+/T/[^/\s\"']+"
)
TEMPORARY_PLACEHOLDER = "/tmp/gen-notebook"

# A notebook that runs longer than this end to end is committed without outputs, so slow
# notebooks do not hold up commits or the git hooks.
DEFAULT_NOTEBOOK_TIMEOUT_SECONDS = 300


def merge_adjacent_streams(cell: nbformat.NotebookNode) -> None:
    """Join consecutive outputs of one stream, which the kernel splits at timing-dependent points."""
    merged: list[nbformat.NotebookNode] = []
    for output in cell.get("outputs", []):
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and output.output_type == "stream"
            and previous.output_type == "stream"
            and previous.name == output.name
        ):
            previous.text += output.text
        else:
            merged.append(output)
    if "outputs" in cell:
        cell.outputs = merged


def normalize_notebook(notebook: nbformat.NotebookNode) -> None:
    """Drop what changes on every execution so a re-run leaves the notebook unchanged.

    Execution timestamps live in each cell's metadata and temporary directory names appear in
    printed paths. The execution counts are kept: they only change when the cells do.
    """
    for cell in notebook.cells:
        cell.get("metadata", {}).pop("execution", None)
        merge_adjacent_streams(cell)
        for output in cell.get("outputs", []):
            if "text" in output:
                output["text"] = TEMPORARY_PATH.sub(
                    TEMPORARY_PLACEHOLDER, output["text"]
                )
            data = output.get("data", {})
            if "text/plain" in data:
                data["text/plain"] = TEMPORARY_PATH.sub(
                    TEMPORARY_PLACEHOLDER, data["text/plain"]
                )


def clear_notebook(path: Path) -> None:
    """Remove every output, execution count and execution timestamp from a notebook."""
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs = []
        cell.execution_count = None
        cell.get("metadata", {}).pop("execution", None)
    nbformat.write(notebook, path)


def has_graphical_output(notebook: nbformat.NotebookNode) -> bool:
    return any(
        "image/png" in output.get("data", {})
        or "application/vnd.jupyter.widget-view+json" in output.get("data", {})
        for cell in notebook.cells
        for output in cell.get("outputs", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-graphical", action="store_true")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_NOTEBOOK_TIMEOUT_SECONDS,
        help="seconds a notebook may take end to end before its outputs are cleared instead",
    )
    parser.add_argument("notebooks", nargs="*")
    arguments = parser.parse_args()

    filenames = arguments.notebooks or sorted(EXAMPLES_DIR.glob("*.ipynb"))
    environment = os.environ.copy()
    environment["GEN_NOTEBOOK_TEXT_OUTPUT"] = "1"
    # matplotlib figures in a notebook would embed PNGs; only Gen's text plots belong here.
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(PYTHON_DIR / "python"), environment.get("PYTHONPATH")))
    )

    for filename in filenames:
        path = Path(filename)
        if not path.exists() and len(path.parts) == 1:
            path = EXAMPLES_DIR / path
        path = path.resolve()
        if path.parent != EXAMPLES_DIR or path.suffix != ".ipynb" or not path.is_file():
            parser.error(f"Expected an example notebook: {filename}")

        if arguments.if_graphical and not has_graphical_output(
            nbformat.read(path, as_version=4)
        ):
            print(f"Skipping {path.name}: no embedded graphical output", flush=True)
            continue

        print(f"Executing {path.name} with text graph output ...", flush=True)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nbconvert",
                    "--to=notebook",
                    "--execute",
                    "--inplace",
                    "--ExecutePreprocessor.timeout=120",
                    str(path),
                ],
                env=environment,
                check=False,
                timeout=arguments.timeout,
            )
        except subprocess.TimeoutExpired:
            clear_notebook(path)
            print(
                f"{path.name} took longer than {arguments.timeout} seconds; its outputs were cleared",
                file=sys.stderr,
                flush=True,
            )
            continue
        if result.returncode:
            return result.returncode
        executed = nbformat.read(path, as_version=4)
        if has_graphical_output(executed):
            print(f"Graphical output remains in {path.name}", file=sys.stderr)
            return 1
        normalize_notebook(executed)
        nbformat.write(executed, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
