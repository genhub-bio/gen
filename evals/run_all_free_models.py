"""Run the full eval suite on all free OpenCode Zen models."""

import subprocess
import sys
import time

FREE_MODELS = [
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "hy3-free",
    "nemotron-3-ultra-free",
    "north-mini-code-free",
]

# deepseek-v4-flash-free doesn't support tool-calling (400s on cli arm)
# so skip cli for that one.
CLI_SKIP = {"deepseek-v4-flash-free"}

total_start = time.monotonic()
for model in FREE_MODELS:
    arms = ["python"]
    if model not in CLI_SKIP:
        arms.append("cli")
    for arm in arms:
        label = f"{arm:>8} / {model:<30}"
        start = time.monotonic()
        print(f"\n{'=' * 70}")
        print(f"START  {label}")
        print(f"{'=' * 70}")
        result = subprocess.run(
            [
                sys.executable,
                "run_eval.py",
                "--opencode-zen",
                "--model",
                model,
                "--arms",
                arm,
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        elapsed = time.monotonic() - start
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[:500])
        print(f"DONE   {label}  ({elapsed:.0f}s)")
        sys.stdout.flush()

total = time.monotonic() - total_start
print(f"\n{'=' * 70}")
print(f"ALL DONE  ({total:.0f}s total)")
