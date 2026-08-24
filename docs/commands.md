# Defaults

This controls default choices for the `gen` command line

- collection
  - This controls the default collection for `gen` to work on, it is what is passed to the `--name` argument.

# Apply

Apply operations to the current branch.

Operations for a given branch can be found via `gen operations -b branch_name`. To apply a commit
from a given branch, use `gen apply commit_id`. The changes will be applied to the current state of
the database and recorded as a new operation.

# Branches

Creating a new branch can be accomplished via `gen branch --create branch_name`.
Deleting a branch can be accomplished via `gen branch --delete branch_name`.
To list all available branches, `gen branch --list`. The current branch will be marked with a `>` before it.
To checkout a branch, `gen branch --checkout branch_name`. This will migrate the database to the last change
applied in a given branch.
To merge a branch, `gen branch --merge branch_name`, will merge a given branch into the current branch. If there
is no common point between the two branches, this will return an error.

# Reset

This will revert a branch to a given commit ID and detach operations made after this commit ID. This should be
done when work after a given point is no longer desired and you wish to start at a fresh point in the branch.

To reset the database to a given commit, run the command `gen reset commit_id`.

# Operations

Operations are changes that have been made to the database. Commands such as `import` and `update` create a new operation.
To see all operations, `gen operations` will list operations. The operation the database currently is on
will be prefixed with a `>`.

# Cache

Remote annotation indexes and unindexed remote annotation files are cached under
`.gen/cache`. Indexed annotation files are read remotely by range instead of
being downloaded in full when the server supports range requests. Otherwise,
the file is streamed into the cache and reused locally. Run `gen cache-clear` to
remove the cache. Local repository files and `.gen/assets` are not affected.

# View diff

Compare two commits or branches with:

```sh
gen view-diff <source> [target]
gen view-diff <source>..<target>
gen view-diff <source>...<target>
```

The two-ref and `source..target` forms compare the endpoint states. The
`source...target` form compares the source/target merge base with the target,
showing only changes introduced on the target side since the histories
diverged. A single source ref without a range compares it with `HEAD`.

# Patches

Like git, patches are the mechanism for bundling together pieces of work for distribution. Patches can be created via
the `patch-create` command and applied via `patch-apply`. Patch archives are versioned, bundle any tracked asset data
needed to replay the selected operations offline, and can be created from a non-checked-out branch with
`patch-create --branch <branch-name> HEAD` to export the commits reachable from that branch head but not from the
current head.

# Checkout

Checkouts allow a user to migrate the database to different states. To move the database to a given commit, use
`gen checkout -b branch_name commit_id`. If no branch name is specified, the current branch will be used.
The `commit_id` corresponds to a commit listed by `gen operations`.

# Clone

Clone a GenHub repository with `gen clone <repository-url>`. Gen creates a
directory named after the repository, downloads its files and history, and
records the canonical GenHub URL as the `origin` remote.

# Push

Push committed changes from one branch with:

```sh
gen push [--remote <name>] [--branch <name>] [--force]
```

The remote is selected from `--remote`, the branch's tracked remote, or the
repository default, in that order. A missing remote branch is created and
tracked. Non-fast-forward updates are rejected unless `--force` is supplied.
Uncommitted working-tree changes are not transferred.

After the push finishes, Gen uploads local assets referenced by the selected branch.
Files are validated against their recorded checksums before upload.

# Fetch

Fetch one branch without merging it into a local branch or changing the working
checkout:

```sh
gen fetch [--remote <name>] [--branch <name>]
```

The remote is selected from `--remote`, the branch's tracked remote, the
repository default, or the only configured remote, in that order. The branch
defaults to the current branch. Fetch runs Dolt's native fetch operation to
update the remote-tracking ref and download its graph history.

Gen also downloads every versioned asset reachable from the fetched branch into
the checksum-addressed `.gen/assets` store. It does not copy any fetched version
to its logical working-tree path; checkout and pull remain responsible for
materializing the selected branch state.

# Pull

Pull one branch with:

```sh
gen pull [--remote <name>] [--branch <name>]
```

Pull follows Dolt's merge behavior: it fast-forwards when possible and
automatically merges divergent history on the current branch. A missing local
branch is created and associated with the selected remote. Pulling the current
branch updates its working tree and rejects uncommitted changes; pulling a
non-current branch leaves the checkout unchanged and requires a fast-forward.
Pull uses the only configured remote when no explicit, tracked, or default
remote selects one.
Pull requires an existing repository and remote; use `gen clone` to initialize a
workspace from a remote repository.

After the pull finishes, Gen downloads missing local assets for the selected
branch. Downloads are accepted only after their recorded checksums validate.

# Remote authentication

Gen reuses credentials created by `gen remote login [<remote-name>]` and
refreshes expired login tokens when possible. For noninteractive use, set
`GENHUB_API_KEY`; it is sent as an `x-api-key` header. Public repositories can
be cloned and pulled anonymously.
