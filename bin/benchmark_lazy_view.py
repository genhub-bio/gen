#!/usr/bin/env python3
"""Time how long `gen view --full` takes to draw its first frame on large block groups.

Two cases exercise the lazily loaded viewer crawl:

  mhc  A GFA of many small segments (each node has few edges).
  hub  chr22 with a 100k-variant VCF, so one backing node carries ~390k edges.

Viewing is read-only, so each case's repository is built once under target/lazy-view-bench/ and
reused on every later run; pass --rebuild to recreate it. This never touches the repository's own
.gen. The viewer runs under a pseudo-terminal and the first frame is detected by its canvas
footer text. Build the release binary first (`cargo build --release`).

Example:
    bin/benchmark_lazy_view.py --mhc-gfa ../gen-large-files/MHC-57.gfa --timeout 120
"""

import argparse
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import shutil
import termios
import time
from fcntl import ioctl
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIRST_FRAME_MARKER = "drag pan"
ANSI_ESCAPE = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")


VIEWS = {
    "mhc": ["view", "", "-s", "mhc", "--full"],
    "hub": ["view", "chr22", "-s", "test", "-c", "test", "--full"],
}


BUILD_STEP_TIMEOUT_SECONDS = 900


def run(gen_bin: Path, workdir: Path, *args: str) -> None:
    print(f"  gen {' '.join(args)}", flush=True)
    subprocess.run(
        [str(gen_bin), *args],
        cwd=workdir,
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=BUILD_STEP_TIMEOUT_SECONDS,
    )


def build_mhc(gen_bin: Path, workdir: Path, gfa: Path) -> None:
    run(gen_bin, workdir, "init")
    run(gen_bin, workdir, "import", "gfa", str(gfa), "--sample", "mhc")


def build_hub(gen_bin: Path, workdir: Path, fixtures: Path) -> None:
    run(gen_bin, workdir, "init")
    run(
        gen_bin,
        workdir,
        "import",
        "fasta",
        "-n",
        "test",
        "--reference",
        "ref",
        str(fixtures / "chr22.fa.gz"),
    )
    run(
        gen_bin,
        workdir,
        "update",
        "vcf",
        "-n",
        "test",
        "-g",
        "0|1",
        "-s",
        "test",
        "--parent-samples",
        "ref",
        str(fixtures / "chr22_100k_no_samples.vcf.gz"),
    )


def time_first_frame(
    gen_bin: Path, workdir: Path, view_args: list[str], timeout: float
) -> float | None:
    """Seconds until the canvas footer is drawn, or None if it never appears within `timeout`."""
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(workdir)
        os.execv(str(gen_bin), ["gen", *view_args])
    ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 200, 0, 0))

    start = time.monotonic()
    seen = b""
    elapsed = None
    try:
        while (now := time.monotonic() - start) < timeout:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if not ready:
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            # Keep only a tail: the footer is redrawn whenever the frame is, and the marker is
            # matched after escape sequences are stripped since styling can split the text.
            seen = (seen + chunk)[-200_000:]
            visible = ANSI_ESCAPE.sub(b"", seen).decode("utf8", "replace")
            if FIRST_FRAME_MARKER in visible:
                elapsed = now
                break
    finally:
        # Close the master before waiting: a viewer with unread output cannot finish exiting
        # while the pseudo-terminal stays open, so waiting first would deadlock.
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.close(fd)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--gen-bin", type=Path, default=REPO_ROOT / "target/release/gen"
    )
    parser.add_argument(
        "--mhc-gfa", type=Path, default=REPO_ROOT.parent / "gen-large-files/MHC-57.gfa"
    )
    parser.add_argument("--fixtures", type=Path, default=REPO_ROOT / "fixtures")
    parser.add_argument("--cases", default="mhc,hub", help="comma-separated: mhc, hub")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per case")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=REPO_ROOT / "target/lazy-view-bench",
        help="where the per-case repositories are kept between runs",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="recreate the repositories"
    )
    args = parser.parse_args()

    builders = {
        "mhc": lambda gen, wd: build_mhc(gen, wd, args.mhc_gfa),
        "hub": lambda gen, wd: build_hub(gen, wd, args.fixtures),
    }
    results = []
    for name in args.cases.split(","):
        workdir = args.workdir / name
        ready_marker = workdir / ".benchmark-ready"
        build_seconds = 0.0
        if args.rebuild or not ready_marker.exists():
            shutil.rmtree(workdir, ignore_errors=True)
            workdir.mkdir(parents=True)
            print(f"[{name}] building repository", flush=True)
            built_at = time.monotonic()
            builders[name](args.gen_bin, workdir)
            build_seconds = time.monotonic() - built_at
            ready_marker.touch()
        print(f"[{name}] timing first frame", flush=True)
        results.append(
            (
                name,
                build_seconds,
                time_first_frame(args.gen_bin, workdir, VIEWS[name], args.timeout),
            )
        )

    failed = False
    print(f"{'case':<6} {'build (s)':>10} {'first frame (s)':>16}")
    for name, build_seconds, first_frame in results:
        if first_frame is None:
            failed = True
            print(
                f"{name:<6} {build_seconds:>10.1f} {'>' + str(int(args.timeout)):>16}"
            )
        else:
            print(f"{name:<6} {build_seconds:>10.1f} {first_frame:>16.2f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
