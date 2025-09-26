Gen offers commands similar to git for managing changes and branches.  Below are descriptions of each.

# Defaults

This controls default choices for the `gen` command line

- database
  - This controls the default database for `gen` to work on, it is what is passed to the `--db` argument. 
- collection
  - This controls the default collection for `gen` to work on, it is what is passed to the `--name` argument.

# Apply

Apply operations to the current branch.

Operations for a given branch can be found via `gen --db db_name.db operations -b branch_name`. To apply an operation
from a given branch, use `gen --db db_name.db apply operation_id`. The changes will be applied to the current state of
the database and recorded as a new operation.

# Branches

Creating a new branch can be accomplished via `gen --db db_name.db branch --create branch_name`.
Deleting a branch can be accomplished via `gen --db db_name.db branch --delete branch_name`.
To list all available branches, `gen --db db_name.db branch --list`. The current branch will be marked with a `>` before it.
To checkout a branch, `gen --db db_name.db branch --checkout branch_name`. This will migrate the database to the last change
applied in a given branch.
To merge a branch, `gen --db db_name.db branch --merge branch_name`, will merge a given branch into the current branch. If there
is no common point between the two branches, this will return an error.

# Reset

This will revert a branch to a given operation id and detach operations made beneath this operation id. This should be
done when work after a given point is no longer desired and you wish to start at a fresh point in the branch.

To reset the database to a given operation, run the command `gen --db db_name.db reset operation_id`.

# Patches

Like git, patches are the mechanism for bundling together pieces of work for distribution. Patches can be created via
the `patch-create` command and applied via `patch-apply`.

# Checkout

Checkouts allow a user to migrate the database to different states. To move the database to a given operation, the
command in `gen --db db_name.db checkout -b branch_name operation_id`. If no branch name is specified, the current
branch will be used. The operation_id corresponds to the operation in `operations`.
