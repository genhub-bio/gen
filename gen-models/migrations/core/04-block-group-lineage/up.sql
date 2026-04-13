ALTER TABLE block_groups ADD COLUMN parent_block_group_id BLOB REFERENCES block_groups(id);

CREATE INDEX block_groups_parent_idx ON block_groups(parent_block_group_id);

DROP INDEX IF EXISTS block_group_uidx;

CREATE UNIQUE INDEX block_group_root_uidx
ON block_groups(collection_name, sample_name, name)
WHERE parent_block_group_id IS NULL;

CREATE UNIQUE INDEX block_group_lineage_uidx
ON block_groups(collection_name, sample_name, name, parent_block_group_id)
WHERE parent_block_group_id IS NOT NULL;
