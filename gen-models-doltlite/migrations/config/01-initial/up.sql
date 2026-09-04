CREATE TABLE defaults (
  id INTEGER PRIMARY KEY NOT NULL,
  collection_name TEXT,
  remote_name TEXT,
  current_branch_name TEXT,
  default_committer_name TEXT NOT NULL,
  default_committer_email TEXT NOT NULL
) STRICT;

CREATE TABLE remotes (
    name TEXT PRIMARY KEY NOT NULL,
    url TEXT NOT NULL
) STRICT;

CREATE TABLE remote_operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    remote_name TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    operation TEXT NOT NULL CHECK(operation IN ('clone', 'pull', 'push')),
    from_commit TEXT,
    assets_transfer_checkpoint TEXT,
    to_commit TEXT,
    transfer_id BLOB CHECK(transfer_id IS NULL OR length(transfer_id) = 16),
    transfer_expires_at INTEGER,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    failed_at TEXT,
    CHECK(completed_at IS NULL OR failed_at IS NULL),
    CHECK(completed_at IS NULL OR (
        to_commit IS NOT NULL
        AND assets_transfer_checkpoint IS NOT NULL
        AND assets_transfer_checkpoint = to_commit
    )),
    CHECK(operation = 'push' OR transfer_id IS NULL),
    CHECK((transfer_id IS NULL) = (transfer_expires_at IS NULL)),
    CHECK(operation != 'push' OR completed_at IS NULL OR transfer_id IS NOT NULL),
    FOREIGN KEY(remote_name) REFERENCES remotes(name) ON DELETE CASCADE
) STRICT;

CREATE UNIQUE INDEX remote_operations_pending
ON remote_operations(remote_name, branch_name)
WHERE completed_at IS NULL AND failed_at IS NULL;

CREATE TABLE remote_branch (
   remote_name TEXT,
   name TEXT,
   FOREIGN KEY(remote_name) REFERENCES remotes(name)
) STRICT;

INSERT INTO defaults (
    id,
    collection_name,
    remote_name,
    current_branch_name,
    default_committer_name,
    default_committer_email
) VALUES (
    1,
    NULL,
    NULL,
    NULL,
    'gen',
    'gen@genhub.bio'
);
