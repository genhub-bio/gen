"""Render eval result JSON(s) as markdown REPL transcripts."""

import json
import pathlib
import sys
import re


def _clean_observations(obs: str) -> str:
    obs = re.sub(r"^Execution logs:\n?", "", obs)
    obs = re.sub(r"\n?Last output from code snippet:\n?.*$", "", obs)
    obs = re.sub(r"<code>.*?</code>", "", obs, flags=re.DOTALL)
    obs = re.sub(r"Calling tools:\n\[.*?\]\n?", "", obs, flags=re.DOTALL)
    obs = re.sub(r"\nOut: None", "", obs)
    return obs.strip()


def _blockquote(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.split("\n"))


def render_from_data(data: list[dict]) -> str:
    blocks: list[str] = []
    step_counters: dict[str, int] = {}
    for run in data:
        task_name = ""
        transcript = run.get("transcript") or []
        for entry in transcript:
            if "task" in entry:
                task_name = entry["task"]
                blocks.append(f"# {task_name}\n")
                step_counters[task_name] = 0
                continue
            tc = entry.get("tool_calls") or []
            tool_name = tc[0]["function"]["name"] if tc else "?"
            code = (entry.get("code_action") or "").strip()
            obs = _clean_observations(entry.get("observations") or "")
            action = entry.get("action_output")
            if not code:
                continue
            step_counters[task_name] += 1
            n = step_counters[task_name]
            blocks.append(f"## Step {n}")
            blocks.append("")
            heading = f"> **{tool_name}**"
            fence = "> ```python"
            code_bq = _blockquote(code)
            obs_bq = _blockquote(obs) if obs else ""
            action_text = str(action).strip() if action else ""
            if action_text not in ("", "None", "__final_answer__"):
                action_bq = _blockquote(action_text)
            else:
                action_bq = ""
            parts = [
                p for p in [heading, fence, code_bq, "> ```", obs_bq, action_bq] if p
            ]
            blocks.append("\n".join(parts))
            blocks.append("")
    return "\n".join(blocks)


def render(json_path: str) -> str:
    return render_from_data(json.loads(pathlib.Path(json_path).read_text()))


def save_markdown(json_path: str) -> str:
    src = pathlib.Path(json_path)
    md = render_from_data(json.loads(src.read_text()))
    out = src.with_suffix(".md")
    out.write_text(md)
    return str(out)


if __name__ == "__main__":
    print(render(sys.argv[1]))
