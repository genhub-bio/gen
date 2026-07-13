"""The two agent arms under comparison: Python-first (CodeAgent) vs CLI-first (ToolCallingAgent)."""

import pathlib
import subprocess

from smolagents import CodeAgent, Tool, ToolCallingAgent

EVALS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
GEN_BINARY = REPO_ROOT / "target" / "release" / "gen"

PYTHON_FIRST_SKILL_DIR = (
    REPO_ROOT
    / "gen-python"
    / "python"
    / "gen"
    / "_skills"
    / "gen-genetic-engineering-python"
)
CLI_FIRST_SKILL_DIR = EVALS_DIR / "arms" / "cli-first"


def _read_skill(skill_dir: pathlib.Path) -> str:
    parts = [skill_dir.joinpath("SKILL.md").read_text()]
    references_dir = skill_dir / "references"
    if references_dir.is_dir():
        for reference_file in sorted(references_dir.glob("*.md")):
            parts.append(
                f"\n\n--- {reference_file.name} ---\n\n{reference_file.read_text()}"
            )
    return "".join(parts)


class ShellTool(Tool):
    name = "shell"
    description = (
        "Run a shell command in the task's working directory and return its stdout+stderr. "
        "The `gen` CLI binary is on PATH. Use this for every gen operation."
    )
    inputs = {"command": {"type": "string", "description": "The shell command to run."}}
    output_type = "string"

    def __init__(self, workdir: pathlib.Path):
        super().__init__()
        self.workdir = workdir

    def forward(self, command: str) -> str:
        result = subprocess.run(
            command,
            shell=True,
            cwd=self.workdir,
            capture_output=True,
            text=True,
            timeout=60,
            env={"PATH": f"{GEN_BINARY.parent}:/usr/bin:/bin"},
        )
        return f"exit code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def build_python_first_agent(model, workdir: pathlib.Path) -> CodeAgent:
    skill_text = _read_skill(PYTHON_FIRST_SKILL_DIR)
    agent = CodeAgent(
        tools=[],
        model=model,
        additional_authorized_imports=["gen", "pathlib", "os", "networkx"],
        max_steps=12,
        # gen's PyO3 objects are unsendable (not thread-safe). smolagents' local
        # executor offloads each code step to a fresh worker thread *only* when a
        # timeout is set, so a `repo`/`sample` kept across steps would be touched
        # from a different thread and panic. Disabling the timeout runs every
        # step inline on the main thread, keeping those objects single-threaded.
        executor_kwargs={"timeout_seconds": None},
    )
    agent.prompt_templates["system_prompt"] += (
        "\n\nYou are working with the `gen` genetic-engineering version control system. "
        "Follow this skill exactly; do not shell out to the `gen` CLI.\n\n"
        + skill_text
        + f"\n\nYour working directory is {workdir}. Treat it as the repository root."
    )
    return agent


def build_cli_first_agent(model, workdir: pathlib.Path) -> ToolCallingAgent:
    skill_text = _read_skill(CLI_FIRST_SKILL_DIR)
    agent = ToolCallingAgent(
        tools=[ShellTool(workdir)],
        model=model,
        max_steps=12,
    )
    agent.prompt_templates["system_prompt"] += (
        "\n\nYou are working with the `gen` genetic-engineering version control system. "
        "Follow this skill exactly; use the `shell` tool for every gen operation, never "
        "write Python.\n\n"
        + skill_text
        + f"\n\nYour working directory is {workdir}. Treat it as the repository root."
    )
    return agent
