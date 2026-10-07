# Version control: workspace, branches, history, remotes

Signatures were checked against `gen-python/src/python_api/` and the stub
`gen-python/python/gen/gen/__init__.pyi`, which has every signature and docstring. Check
`help()` in the actual interpreter when using a different release.

Contents: workspace and handles; branches; history; merge, apply and reset; remotes; working
on an experiment; mistakes to avoid.

## Workspace and handles

```python
import gen

repo = gen.Repository("path/to/workspace", committer="Name", email="name@example.org")
repo.gen_dir
repo.samples
repo.get_sequence_graphs(name=None, sample=None, collection=None)   # sample: name or Sample
repo.get_sequence_graph(graph.id)
```

`Repository(path=None, committer=None, email=None)` opens or initializes a workspace; `path`
can be the workspace or its `.gen` directory. Omitted path discovers the workspace from the
current directory. Set `committer` and `email` so operations are attributed.

`repo.samples` is a list of `Sample` objects. A sample is iterable and indexable (including
negative indices), with `.name`, `.collection`, and `.sequence_graphs`. Graphs expose `.id`
(`HashId`), `.name`, `.sample` (the owning `Sample`), and `.collection`. Pass the typed id, or
its string, to `get_sequence_graph()`.

`HashId` is hashable and compares by value; it identifies graphs, annotations, assets,
operations and branch heads. `str(hash_id)` is the hex digest.

Mutating import/update/edit/copy APIs record operations automatically. The public API has no
transaction context manager. Raw SQL (`execute`/`query`) is low-level and hidden from the
stubs; use the operation-aware APIs instead.

## Branches

```python
repo.current_branch
repo.get_branches()
repo.create_branch(name, start=None)
repo.checkout(branch, create=False)      # exist_ok=True to reuse an existing branch
repo.delete_branch(branch)
```

Branch arguments accept names or `Branch` objects. `create_branch()` creates without
switching and accepts an `Operation`, its `HashId` or a string commit ref for `start`.
`checkout(branch, *, create=False)` switches branches, not arbitrary operation hashes, and
`create=True` creates the branch and switches to it in one call.

`Branch` fields: `name`, `head` (`HashId`), `remote`, `is_current`, `dirty`.

## History

```python
repo.get_operations(branch=None, limit=None)
repo.get_assets(branch=None)
```

`get_operations()` returns newest-first `Operation` objects, so `get_operations(limit=1)[0]`
is the latest. `Operation` fields: `id` (`HashId`), `parent_id`, `committer`, `email`, `date`,
`message`, `is_head`. Every mutating call records one operation; pass `message=` where an
edit method accepts it so the history reads as a design log. History is the audit trail:
there is nothing to commit.

## Merge, apply and reset

```python
repo.merge(branch)
repo.apply(operation)
repo.reset(operation)
```

`merge()`, `apply()`, and `reset()` return the resulting HEAD `Operation`; apply/reset accept
an `Operation`, its `HashId` or a string ref. `merge()` adopts another branch's operations
into the current one, `apply()` applies a given operation to the current branch, and `reset()` hard-resets the current
branch. History actions require a clean working set (`Branch.dirty` is false); do not discard
dirty state to bypass an error. Reset discards later operations on the branch, so confirm with
the user first. Re-query graphs and samples after history changes instead of reusing handles
from another state.

## Remotes

```python
cloned = gen.clone(url, path=None, committer=None, email=None)
repo.remotes
repo.default_remote
repo.add_remote(name, url)
repo.remove_remote(remote)
repo.set_default_remote(remote=None)
repo.set_branch_remote(remote=None)
repo.fetch(remote=None, branch=None)
repo.pull(remote=None, branch=None)
repo.push(remote=None, branch=None, force=False)
```

`clone()` returns an open `Repository`. Pass a destination that does not exist, is empty, or
only holds a freshly initialized workspace; do not copy files into `.gen` by hand. The clone
is already checked out on the remote's default branch (`checkout("main")` works afterwards).
`add_remote()` + `pull()` is for a repository that already has history. `add_remote()`
returns a `Remote` with `.name` and `.url`. Remote arguments accept a name or `Remote`; branch
arguments accept a name or `Branch`. Omitted remote/branch follows repository defaults and
branch tracking. Passing no remote to the setters clears that configuration.
`set_branch_remote()` applies to the current branch.

Fetch updates a remote-tracking ref without changing checkout; pull merges remote state and
transfers assets; push updates the remote. These methods return `None` and refresh the
repository connection. Re-query graphs and samples after sync instead of relying on
previously held handles from another state. Pushing is outward-facing and `force=True`
overwrites remote history: confirm both with the user.

## Working on an experiment

```python
original_branch = repo.current_branch
repo.checkout("design-experiment", create=True)
graph = repo.import_sequence("AAAACCCC", name="vector", sample="design")
graph.replace("vector:4-8", "GGGG", message="change downstream motif")
assert repo.get_operations(limit=1)[0].message == "change downstream motif"
repo.checkout(original_branch)
# Merge only when the requested workflow calls for adopting the experiment.
# repo.merge("design-experiment")
```

Use a branch when a change is exploratory, when the user wants to compare designs without
touching the baseline, or before a risky bulk update. Use `sample.copy()` (`design.md`) when
you only need an independent copy of a sample inside one branch. Operation diffs and patches
have no Python binding; use the Gen CLI for them.

## Mistakes to avoid

- Exporting a FASTA "as a backup" before editing. History already holds every prior state.
- Calling `checkout()` and then continuing with handles obtained on the other branch.
- Merging or resetting without being asked to adopt or discard work.
- Re-running a cell that creates a branch or copy without `exist_ok=True`.
- Switching to the CLI after a `TypeError`; check the signature in the stub instead.
