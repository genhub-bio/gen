CREATE TABLE defaults (
  id INTEGER PRIMARY KEY NOT NULL,
  collection_name TEXT,
  remote_name TEXT,
  current_branch_name TEXT
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
INSERT INTO defaults values (1, NULL, NULL, NULL);
