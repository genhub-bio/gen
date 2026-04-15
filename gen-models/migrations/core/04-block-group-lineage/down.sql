DROP INDEX IF EXISTS block_group_parent_idx;
DROP INDEX IF EXISTS block_group_default_uidx;
DROP INDEX IF EXISTS block_group_uidx;

CREATE UNIQUE INDEX block_group_uidx ON block_groups(collection_name, sample_name, name);

ALTER TABLE block_groups DROP COLUMN is_default;
ALTER TABLE block_groups DROP COLUMN parent_block_group_id;
