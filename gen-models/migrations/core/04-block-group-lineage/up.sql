ALTER TABLE block_groups
ADD COLUMN parent_block_group_id BLOB REFERENCES block_groups(id);

ALTER TABLE block_groups
ADD COLUMN is_default INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0, 1));

DROP INDEX IF EXISTS block_group_uidx;

CREATE UNIQUE INDEX block_group_uidx
ON block_groups(
  collection_name,
  sample_name,
  name,
  IFNULL(hex(parent_block_group_id), '')
);

CREATE INDEX block_group_parent_idx ON block_groups(parent_block_group_id);
