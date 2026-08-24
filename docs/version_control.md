# Version Control

Gen enables version control similar to git with the sequence database. Many version control like commands in gen are
direct analogs of git commands. Slight differences are:

* Commits are called operations. This is because there is not the same kind of straightforward editing of text files 
that is common in git in biological engineering. Thus, many things we do to our sequences are operations -- such as
inserting a gene in a position.
* There is no concept of a staging area for changes. Whereas git will wait for the user to package together work 
into commits, operations are applied right away.

A summary of commands implemented in Gen and their counterparts in git are:

| Command             | Gen                             | Git                         |
|---------------------|---------------------------------|-----------------------------|
| Create a new branch | gen branch --create new_branch  | git branch new_branch       | 
| Delete a branch     | gen branch --delete branch_name | git branch -D branch_name   |
| List all branches   | gen branch --list               | git branch                  |
| Merge a branch      | gen branch --merge branch_name  | git merge branch_name       |
| Checkout a branch   | gen checkout -b branch_name     | git checkout branch_name    |
| Checkout a commit   | gen checkout <operation_id>     | git checkout commit         |
| Reset to a commit   | gen reset <operation_id>        | git reset commit            |
| List work history   | gen operations <branch_name>    | git log                     |
| Apply an operation  | gen apply <operation_id>        | git cherry-pick commit_hash |

# How are operations created

Operations are created when running commands such as `import` and `update`. Here's a simple example of managing
two projects in a single database.

```console
gen init
gen defaults -d example.db -c example
gen import test.fa
gen branch -c branch-1
gen checkout -b branch-1
gen`update --vcf some_changes.vcf
gen checkout main
gen branch -c branch-2
gen update --vcf other_changes.vcf
```

The outcome of this is 2 branches, `branch-1` and `branch-2`, each with a common operation, `test.fa`, and a different
vcf applied in each branch (which in turn made a new operation in each respective branch). These branches can be merged
back into the main branch when ready via `gen branch --merge branch-1` and similarly for branch-2. Cherry-picks 
(via `apply`) can also be carried out to selectively apply operations between branches.

# Missing git concepts

These are some missing git concepts which may make sense at some point to add, but currently do not:

* Rebase is not implemented and merge is used to apply changes from another branch.
* General file storage is not part of gen. You cannot add or keep track of arbitrary files.
* Many of the general file management commands are not present, such as `add`, `restore`, `rm`.

These are currently missing but planned to be added:

* Diffs
* Commit messages and packaging operations together into a unit
* Patch creation and import
* Tags
