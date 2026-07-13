# Skill eval: Python-first vs CLI-first

Head-to-head eval of two `gen` agent skills:

- **`python`** arm — `gen-python/python/gen/_skills/gen-genetic-engineering-python/`
  (the current skill, code-first, driven via a smolagents `CodeAgent` that
  executes real Python with `gen` importable).
- **`cli`** arm — `evals/arms/cli-first/` (a frozen snapshot of the original
  CLI-first skill from the unmerged `skill` branch / PR #181), driven via a
  smolagents `ToolCallingAgent` restricted to a single `shell` tool that runs
  the `gen` binary.

Each task copies fixture files from `fixtures/` into an isolated working
directory, gives the agent a plain-English intent, and verifies the resulting
`.gen` repository against ground truth computed independently with the `gen`
Python API — not by trusting whatever the agent produced. See `tasks.py`.

## Setup

```sh
.venv/bin/pip install -r evals/requirements.txt
cargo build --release --bin gen   # arms.py shells out to target/release/gen
```

Two model backends are supported:

- **Anthropic (or anything LiteLLM knows)** — needs `ANTHROPIC_API_KEY` set to a
  *billed Anthropic Console key*, not a Claude Code session credential (a Claude
  Code session key gets a 402 credit-balance error on direct API calls).
- **OpenCode Zen free tier** — `https://opencode.ai/zen` exposes an
  OpenAI-compatible endpoint with several genuinely free models, callable from
  any client, no OpenCode CLI required. The free models work **anonymously,
  with no API key** — pass `--opencode-zen`. (If you do have an
  `OPENCODE_ZEN_API_KEY`, it's picked up automatically for higher rate limits
  or paid models.) Check current free model ids at
  `https://opencode.ai/zen/v1/models`; as of writing:
  `deepseek-v4-flash-free`, `mimo-v2.5-free`, `hy3-free`,
  `nemotron-3-ultra-free`, `north-mini-code-free`.

## Running

```sh
cd evals
../.venv/bin/python run_eval.py                                   # both arms, all tasks, 1 repeat (Anthropic)
../.venv/bin/python run_eval.py --arms python                      # one arm
../.venv/bin/python run_eval.py --tasks import-fasta region-replace
../.venv/bin/python run_eval.py --model anthropic/claude-sonnet-5 --repeats 3

# free tier, no key needed:
../.venv/bin/python run_eval.py --opencode-zen --model deepseek-v4-flash-free --arms python
../.venv/bin/python run_eval.py --opencode-zen --model mimo-v2.5-free --arms cli
```

**Model notes from initial live testing (both $0, free tier):**
- `deepseek-v4-flash-free` worked well for the `python` arm (`CodeAgent`) but
  the OpenCode Zen gateway rejects it for the `cli` arm — `ToolCallingAgent`'s
  structured tool-calling request gets a `400 Upstream request failed` from
  the provider every time, not transient. It apparently doesn't support
  function-calling through this gateway.
- `mimo-v2.5-free` worked for the `cli` arm (passed `import-fasta` in 12
  steps, ~57k tokens — much more exploration than a paid frontier model would
  need, worth keeping in mind when comparing token-efficiency numbers across
  models).
- Haven't yet characterized `hy3-free`, `nemotron-3-ultra-free`,
  `north-mini-code-free`, or which free models are best for each arm — if you
  hit a similar 400 on one model, try another before assuming the skill/arm
  is broken.
- If you're comparing arms head-to-head, use the **same model** for both, since
  models vary a lot in tool-calling support and verbosity on this gateway.

Writes a timestamped JSON file to `evals/results/` (gitignored) with
per-run token usage, step counts, pass/fail, and failure messages, then
prints a summary table.

## Adding a task

Add a `Task(name=..., intent=..., fixtures=[...], verify=...)` to `TASKS` in
`tasks.py`. `verify(workdir)` reopens the repo the agent should have built at
`workdir` (`gen.Repository(str(workdir))`) and returns `(passed, message)`.
Compute expected values by running the equivalent `gen` Python calls
yourself first — don't hand-derive coordinates.

## Status

Harness runs for real, end to end, confirmed against OpenCode Zen's free
tier: agent construction, `chdir`'d sandboxed executor with `gen` importable,
shell-tool-restricted CLI arm, real ground-truth verification, token/step
capture, results JSON, summary — all validated live for $0 with
`deepseek-v4-flash-free` (python arm, 1 step, passed) and `mimo-v2.5-free`
(cli arm, 12 steps, passed) against the `import-fasta` task. No Anthropic key
with billing was available to validate against a frontier model.

Only 5 tasks exist so far (import, region edit, VCF update, combinatorial
library, GenBank annotation translation); the original eval plan calls for
15-30 across the same surface areas before treating results as conclusive.
No full matrix run (all tasks × both arms × multiple repeats) has happened
yet — only spot checks on `import-fasta` while building the harness.

**Known cosmetic noise:** after a `python`-arm run completes, you may see
`RuntimeError: ... is unsendable, but is being dropped on another thread`
printed to stderr for `PyRepository`/`PySample`/`PySequenceGraph`. This is a
PyO3 finalizer running on a GC thread other than the one that created the
object; it doesn't affect the run's result (verified: the task still passed
and results were written correctly) but it's noisy. Not investigated further.
