"""Generate a unified eval report with summary table + appendix of transcripts."""

import json
import glob
import pathlib
import re

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def _model_from_json(path: str) -> str:
    data = json.load(open(path))
    for r in data:
        for step in r.get("transcript") or []:
            if "model_output_message" not in step:
                continue
            msg = step["model_output_message"] or {}
            raw = msg.get("raw") or {}
            m = raw.get("model") or ""
            if m:
                return m.replace("tencent/", "")
    return "?"


def _bump_headings(text: str, by: int = 1) -> str:
    return re.sub(r"^#", "#" * (by + 1), text, flags=re.MULTILINE)


def build_report() -> str:
    json_files = sorted(glob.glob(str(RESULTS / "178384*.json")))

    lines = []
    lines.append("# Gen Agent Skill Eval — Full Report\n")

    total_runs = 0
    for jf in json_files:
        total_runs += len(json.load(open(jf)))
    lines.append(f"- {len(json_files)} result files, {total_runs} runs\n")

    # ── Technical introduction ──
    lines.append("## Introduction\n")
    lines.append(
        "This report compares two agent skill paradigms for the `gen` genetic-engineering "
        "version control system across five representative tasks. Each run isolates a fresh "
        "working directory, copies in the required fixture files, and hands a plain-English "
        "intent to a smolagents agent. The agent may take up to 12 steps to complete the task. "
        "After the agent finishes (or fails), a verifier reopens the repository and checks the "
        "result against ground truth computed independently via the `gen` Python API — the "
        "verifier never trusts the agent's own output.\n\n"
        "The two arms under comparison are:\n\n"
        "- **Python-first (`python`)** — a `CodeAgent` that writes and executes real Python "
        "code using the `gen` library directly. The agent is given the full `gen` skill prompt "
        "plus the working directory path. It runs each code step inline on the main thread "
        "(no timeout-based thread offloading) to avoid PyO3 unsendable-object panics.\n"
        "- **CLI-first (`cli`)** — a `ToolCallingAgent` restricted to a single `shell` tool "
        "that shells out to the `gen` binary. The agent is given the CLI-first skill prompt "
        "and must translate each step into a shell command. The shell tool captures stdout, "
        "stderr, and exit code and returns them as a string.\n\n"
        "All agents are tested against free-tier models served by the OpenCode Zen gateway "
        "(`https://opencode.ai/zen/v1`) with no API key required. Each model is evaluated on "
        "both arms (when its API capabilities support the arm's calling convention) and all "
        "five tasks. Results record token usage (input/output), step count, wall-clock "
        "duration, and pass/fail status per run.\n\n"
        "The five tasks are:\n\n"
        "1. **import-fasta** — Create a repository and import a single FASTA file as a named sample.\n"
        "2. **region-replace** — Import a FASTA as a reference sample, then derive a new "
        "sample with a specific nucleotide region replaced by a literal sequence. The "
        "reference must remain unmodified.\n"
        "3. **vcf-update** — Import a FASTA as a reference sample, then apply VCF variants "
        "to create branches corresponding to the VCF sample genotype columns.\n"
        "4. **combinatorial-library** — Build a combinatorial library from separate part "
        "sequences using a column layout CSV, producing a graph with all 9 possible combinations.\n"
        "5. **translate-annotation** — Import a GenBank plasmid, find the AmpR / bla "
        "gene annotation, and translate its CDS into a protein sequence graph. The beta-lactamase "
        "protein sequence is verified against a precomputed ground truth.\n\n"
        "Results are reported per (arm, model, task) combination in the summary table below, "
        "followed by full step-by-step transcripts for every run in the appendix.\n"
    )

    # ── Summary table ──
    lines.append("## Summary\n")

    tasks = [
        "import-fasta",
        "region-replace",
        "vcf-update",
        "combinatorial-library",
        "translate-annotation",
    ]

    matrix = {}
    for jf in json_files:
        data = json.load(open(jf))
        model = _model_from_json(jf)
        for r in data:
            key = f"{r['arm']}/{model}"
            matrix.setdefault(key, {})[r["task"]] = r["passed"]

    header = (
        "| arm/model | "
        + " | ".join(t + " " * max(0, 22 - len(t)) for t in tasks)
        + " |"
    )
    sep = "|" + "-" * 11 + "|" + "|".join("-" * 24 for _ in tasks) + "|"
    lines.append(header)
    lines.append(sep)

    for arm in ["python", "cli"]:
        for key in sorted(matrix, key=lambda k: (k.split("/")[1], k)):
            if not key.startswith(f"{arm}/"):
                continue
            row = f"| **{key}** |"
            for t in tasks:
                p = matrix[key].get(t)
                if p is None:
                    row += " — |"
                elif p:
                    row += " ✅ |"
                else:
                    row += " ❌ |"
            lines.append(row)

    lines.append("")

    # ── Performance overview ──
    lines.append("## Performance\n")
    for key in sorted(matrix, key=lambda x: -sum(1 for t in tasks if matrix[x].get(t))):
        passed = sum(1 for t in tasks if matrix[key].get(t))
        emoji = "🟢" if passed == 5 else "🟡" if passed >= 3 else "🔴"
        lines.append(f"- {emoji} **{key}**: {passed}/{len(tasks)} passed")

    lines.append("")

    # ── Appendix ──
    lines.append("---\n")
    lines.append("## Appendix: Transcripts\n")

    for jf in json_files:
        model = _model_from_json(jf)
        data = json.load(open(jf))
        md = pathlib.Path(jf).with_suffix(".md")
        appendix = md.read_text() if md.exists() else ""

        arm = data[0]["arm"] if data else "?"
        lines.append(f"### {arm} / {model}\n")

        if appendix.strip():
            lines.append(_bump_headings(appendix.strip(), by=2))
        else:
            lines.append("*(empty transcript)*")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    report = build_report()
    out = RESULTS / "full_report.md"
    out.write_text(report)
    print(f"wrote {out}")
