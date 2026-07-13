"""Run the CLI-first vs Python-first `gen` skill eval.

Usage:
    .venv/bin/python evals/run_eval.py --model anthropic/claude-haiku-4-5 --arms python cli
    .venv/bin/python evals/run_eval.py --tasks import-fasta region-replace --repeats 3

Requires ANTHROPIC_API_KEY (or another LiteLLM-supported provider's credentials,
with --model set accordingly).
"""

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import tempfile
import time

from smolagents import LiteLLMModel, OpenAIServerModel

from arms import build_cli_first_agent, build_python_first_agent
from tasks import TASKS
from transcript import save_markdown

# OpenCode Zen (https://opencode.ai/zen) exposes an OpenAI-compatible endpoint
# usable from any client, including a handful of genuinely free models. Get a
# key at https://opencode.ai/auth (a Zen account + API key is required even
# for the free models; no charge is made for them). Current free model ids —
# check https://opencode.ai/zen/v1/models for the live list, this changes:
#   deepseek-v4-flash-free, mimo-v2.5-free, hy3-free,
#   nemotron-3-ultra-free, north-mini-code-free
OPENCODE_ZEN_API_BASE = "https://opencode.ai/zen/v1"

RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"


@dataclasses.dataclass
class RunResult:
    arm: str
    task: str
    repeat: int
    passed: bool
    message: str
    input_tokens: int
    output_tokens: int
    step_count: int
    duration_seconds: float
    error: str | None = None
    transcript: list | None = None


def run_one(arm: str, task, model, repeat: int) -> RunResult:
    workdir = pathlib.Path(tempfile.mkdtemp(prefix=f"gen-eval-{arm}-{task.name}-"))
    task.setup(workdir)

    build_agent = build_python_first_agent if arm == "python" else build_cli_first_agent
    agent = build_agent(model, workdir)

    start = time.monotonic()
    error = None
    input_tokens = output_tokens = step_count = 0
    transcript = None
    # The CodeAgent's local Python executor runs in-process, so its os.getcwd()
    # is the real process cwd, not whatever we told the agent in its prompt —
    # actually chdir so "this working directory" is true for both arms (the
    # shell tool for the CLI arm gets cwd via subprocess's cwd= instead).
    previous_cwd = pathlib.Path.cwd()
    os.chdir(workdir)
    try:
        result = agent.run(task.intent, return_full_result=True)
        if result.token_usage is not None:
            input_tokens = result.token_usage.input_tokens
            output_tokens = result.token_usage.output_tokens
        step_count = len(result.steps)
        transcript = result.steps
    except Exception as exception:  # agent crashed / hit max steps / model error
        error = f"{type(exception).__name__}: {exception}"
    finally:
        os.chdir(previous_cwd)

    duration = time.monotonic() - start

    if error is None:
        try:
            passed, message = task.verify(workdir)
        except Exception as exception:
            passed, message = (
                False,
                f"verifier raised {type(exception).__name__}: {exception}",
            )
    else:
        passed, message = False, "agent run failed before verification"

    shutil.rmtree(workdir, ignore_errors=True)

    return RunResult(
        arm=arm,
        task=task.name,
        repeat=repeat,
        passed=passed,
        message=message,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        step_count=step_count,
        duration_seconds=duration,
        error=error,
        transcript=transcript,
    )


def summarize(results: list[RunResult]) -> None:
    by_arm: dict[str, list[RunResult]] = {}
    for result in results:
        by_arm.setdefault(result.arm, []).append(result)

    print("\n=== Summary ===")
    for arm, arm_results in by_arm.items():
        n = len(arm_results)
        passed = sum(r.passed for r in arm_results)
        avg_input = sum(r.input_tokens for r in arm_results) / n
        avg_output = sum(r.output_tokens for r in arm_results) / n
        avg_steps = sum(r.step_count for r in arm_results) / n
        print(
            f"{arm:>8}: {passed}/{n} passed | "
            f"avg tokens in/out = {avg_input:.0f}/{avg_output:.0f} | "
            f"avg steps = {avg_steps:.1f}"
        )

    print("\n=== Per-task ===")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"[{status}] {result.arm:>8} / {result.task:<22} repeat={result.repeat} "
            f"tokens={result.input_tokens}+{result.output_tokens} steps={result.step_count} "
            f"{result.duration_seconds:.1f}s"
        )
        if not result.passed:
            print(f"         {result.error or result.message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="anthropic/claude-haiku-4-5")
    parser.add_argument(
        "--opencode-zen",
        action="store_true",
        help=(
            "Route --model through OpenCode Zen's OpenAI-compatible API "
            f"({OPENCODE_ZEN_API_BASE}) instead of LiteLLM. Pass a Zen model id, "
            "e.g. --model deepseek-v4-flash-free. Reads the key from "
            "OPENCODE_ZEN_API_KEY (get one at https://opencode.ai/auth)."
        ),
    )
    parser.add_argument(
        "--arms", nargs="+", choices=["python", "cli"], default=["python", "cli"]
    )
    parser.add_argument("--tasks", nargs="+", default=[task.name for task in TASKS])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    selected_tasks = [task for task in TASKS if task.name in args.tasks]
    unknown = set(args.tasks) - {task.name for task in selected_tasks}
    if unknown:
        raise SystemExit(f"unknown task name(s): {sorted(unknown)}")

    if args.opencode_zen:
        # The free-tier models work anonymously — but only with *no*
        # Authorization header at all; a garbage/placeholder key is rejected
        # with 401, unlike a genuinely absent one. So when no real key is
        # configured, force the header blank rather than omitting api_key
        # (the openai client requires a non-empty api_key string to
        # construct). If OPENCODE_ZEN_API_KEY is set, send it normally for
        # higher rate limits / paid models.
        api_key = os.environ.get("OPENCODE_ZEN_API_KEY")
        client_kwargs = {} if api_key else {"default_headers": {"Authorization": ""}}
        model = OpenAIServerModel(
            model_id=args.model,
            api_base=OPENCODE_ZEN_API_BASE,
            api_key=api_key or "unauthenticated",
            client_kwargs=client_kwargs,
        )
    else:
        model = LiteLLMModel(model_id=args.model)

    results: list[RunResult] = []
    for arm in args.arms:
        for task in selected_tasks:
            for repeat in range(args.repeats):
                print(
                    f"running {arm} / {task.name} (repeat {repeat + 1}/{args.repeats})..."
                )
                results.append(run_one(arm, task, model, repeat))

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{int(time.time())}.json"
    out_path.write_text(
        json.dumps(
            [dataclasses.asdict(r) for r in results],
            indent=2,
            default=str,
        )
    )
    print(f"wrote {out_path}")

    md_path = save_markdown(str(out_path))
    print(f"wrote {md_path}")

    summarize(results)


if __name__ == "__main__":
    main()
