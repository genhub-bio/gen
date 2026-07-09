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
