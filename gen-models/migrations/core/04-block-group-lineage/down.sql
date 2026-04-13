PRAGMA foreign_keys = OFF;

DROP INDEX IF EXISTS block_group_lineage_uidx;
DROP INDEX IF EXISTS block_group_root_uidx;
DROP INDEX IF EXISTS block_groups_parent_idx;

CREATE TABLE block_groups_rollback (
  id BLOB PRIMARY KEY NOT NULL,
  collection_name TEXT NOT NULL,
  sample_name TEXT NOT NULL,
  name TEXT NOT NULL,
  created_on INTEGER NOT NULL,
  FOREIGN KEY(collection_name) REFERENCES collections(name),
  FOREIGN KEY(sample_name) REFERENCES samples(name)
) STRICT;

INSERT INTO block_groups_rollback (id, collection_name, sample_name, name, created_on)
SELECT id, collection_name, sample_name, name, created_on
FROM block_groups;

DROP TABLE block_groups;
ALTER TABLE block_groups_rollback RENAME TO block_groups;
CREATE UNIQUE INDEX block_group_uidx ON block_groups(collection_name, sample_name, name);

PRAGMA foreign_keys = ON;
