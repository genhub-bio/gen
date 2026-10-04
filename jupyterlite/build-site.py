"""Bundle the actual Gen wheel and a demonstration notebook into JupyterLite."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
wheels = list((ROOT / "wheels").glob("gen-*.whl"))
if len(wheels) != 1:
    raise SystemExit("Build exactly one Gen wheel in jupyterlite/wheels first.")
wheel = wheels[0]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(True),
        "execution_count": None,
        "outputs": [],
    }


cells = [
    markdown(f"""# Gen in Pyodide

This site bundles `{wheel.name}` in its site-relative piplite index. Install it
using the kernel's supported wheel mechanism. Gen runs in the kernel worker.
"""),
    code(
        "import piplite\nawait piplite.install('gen==0.3.1')\nimport gen  # noqa: E402\nprint(gen.__version__)\n"
    ),
    markdown("""## Create a repository and read a sequence

Use `/tmp` (the kernel's MEMFS) for live databases. JupyterLite's `/drive` uses
buffered Contents API files, which are unsuitable for live SQLite connections.
This repository lasts for this kernel session. Export it before restarting.
"""),
    code("""from pathlib import Path
from tempfile import mkdtemp

root = Path(mkdtemp(prefix='gen-demo-', dir='/tmp'))
repository = gen.Repository(str(root))
fasta = root / 'small.fa'
fasta.write_text('>tiny\\nACGTACGT\\n')
repository.import_fasta(str(fasta), sample='initial')
repository.export_fasta(str(root / 'before.fa'), sample='initial')
print((root / 'before.fa').read_text())
"""),
    markdown("""## Edit and inspect committed history

Import and update methods commit their operation automatically. The region uses
zero-based half-open coordinates; replace the eight bases with a new sequence.
"""),
    code("""repository.update_with_sequence('ACGTTTGT', 'initial', 'edited', 'tiny:0-8')
repository.export_fasta(str(root / 'after.fa'), sample='edited')
print((root / 'after.fa').read_text())
for operation in repository.get_operations():
    print(operation.id, operation.message)
"""),
    markdown("""## Browser HTTP example

The local demonstration server deliberately rejects a clone with HTTP 422.
The call goes through Gen's Rust/C XHR transport and preserves the server's
error body. This is an error-path demonstration, not a successful remote clone.
For a real clone, supply a GenHub-compatible URL whose server permits CORS.
"""),
    code("""# Set this to the origin printed by serve.py if you use a different port.
remote_url = 'http://127.0.0.1:4502/api/repos/demo/rejected'
try:
    gen.clone(remote_url, str(root / 'remote-clone'))
except RuntimeError as error:
    print(error)
"""),
    markdown("""## Export the repository

Close all Gen objects before archiving, then copy the archive to `/drive` to
make it available in the file browser. Download it from there to keep it outside
browser storage. JupyterLite stores drive files in IndexedDB for this origin;
clearing site data deletes them. Installed wheels and `/tmp` are lost on restart.
"""),
    code("""import gc
import shutil

# Release the SQLite handles before packaging the complete repository.
del repository
gc.collect()
archive = shutil.make_archive('/tmp/gen-demo', 'zip', root_dir=root)
destination = Path('/drive/gen-demo.zip')
shutil.copyfile(archive, destination)
print(f'Download {destination.name} from the JupyterLite file browser.')
"""),
]
contents = ROOT / "content"
contents.mkdir(exist_ok=True)
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (Pyodide)",
            "language": "python",
            "name": "python",
        },
        "language_info": {"name": "python", "version": "3.13.2"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for index, cell in enumerate(cells):
    cell["id"] = f"gen-demo-{index}"
(contents / "Gen.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
subprocess.run(
    [
        "jupyter",
        "lite",
        "build",
        "--contents",
        "content",
        "--output-dir",
        "_output",
        "--piplite-wheels",
        str(wheel),
    ],
    cwd=ROOT,
    check=True,
)
