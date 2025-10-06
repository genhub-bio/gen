CREATE TABLE defaults (
  id INTEGER PRIMARY KEY NOT NULL,
  db_name TEXT,
  collection_name TEXT,
  remote_name TEXT
) STRICT;

CREATE TABLE operation_state (
  id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
  operation_hash BLOB,
  branch_id INTEGER,
  FOREIGN KEY(operation_hash) REFERENCES operations(hash),
  FOREIGN KEY(branch_id) REFERENCES branch(id)
) STRICT;

CREATE TABLE operations (
  hash BLOB PRIMARY KEY NOT NULL,
  parent_hash BLOB,
  change_type TEXT NOT NULL,
  FOREIGN KEY(parent_hash) REFERENCES operations(hash)
) STRICT;

CREATE TABLE file_additions (
  id BLOB PRIMARY KEY NOT NULL,
  file_path TEXT NOT NULL,
  file_type TEXT NOT NULL,
  checksum BLOB NOT NULL
) STRICT;

CREATE TABLE operation_files (
  id INTEGER PRIMARY KEY NOT NULL,
  operation_hash BLOB NOT NULL,
  file_addition_id BLOB NOT NULL,
  FOREIGN KEY(operation_hash) REFERENCES operations(hash),
  FOREIGN KEY(file_addition_id) REFERENCES file_additions(id)
) STRICT;

CREATE TABLE operation_summary (
  id INTEGER PRIMARY KEY NOT NULL,
  operation_hash BLOB NOT NULL,
  summary TEXT NOT NULL,
  FOREIGN KEY(operation_hash) REFERENCES operations(hash)
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

CREATE TABLE branch (
  id INTEGER PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  current_operation_hash BLOB,
  remote_name TEXT,
  FOREIGN KEY(current_operation_hash) REFERENCES operations(hash),
  FOREIGN KEY(remote_name) REFERENCES remotes(name) ON DELETE SET NULL
) STRICT;
CREATE UNIQUE INDEX branch_uidx ON branch(name);

CREATE TABLE gen_databases (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    db_uuid TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    path TEXT NOT NULL
) STRICT;

INSERT INTO branch (id, name) values (1, 'main');
INSERT INTO defaults values (1, NULL, NULL, NULL);
INSERT INTO operation_state (id, branch_id) values (1, 1);
