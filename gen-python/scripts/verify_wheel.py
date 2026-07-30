#!/usr/bin/env python3
"""Verify that a Gen wheel contains the Python package and bundled client."""

import sys
import zipfile
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(message)


def verify_wheel(wheel_path: Path) -> None:
    client_name = "gen.exe" if "-win" in wheel_path.name else "gen"

    if "-abi3-" not in wheel_path.name:
        fail(f"{wheel_path} should be tagged for the CPython stable ABI")

    with zipfile.ZipFile(wheel_path) as wheel:
        names = {entry.filename for entry in wheel.infolist()}

        if not any(name.endswith(f".data/scripts/{client_name}") for name in names):
            fail(f"{wheel_path} should contain the bundled {client_name}")

        extension_suffixes = (".pyd", ".so")
        if not any(
            name.startswith("gen/") and name.endswith(extension_suffixes)
            for name in names
        ):
            fail(f"{wheel_path} should contain the compiled gen extension")

        if "gen/static/jupyter_widget.js" not in names:
            fail(f"{wheel_path} should contain the Jupyter widget asset")

    print(f"Verified bundled client and Python package in {wheel_path}")


def main() -> None:
    if len(sys.argv) != 2:
        fail(f"usage: {Path(sys.argv[0]).name} WHEEL")

    wheel_path = Path(sys.argv[1])
    if not wheel_path.is_file():
        fail(f"wheel does not exist: {wheel_path}")

    verify_wheel(wheel_path)


if __name__ == "__main__":
    main()
