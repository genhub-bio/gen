ALTER TABLE file_additions
RENAME COLUMN file_path TO asset_uri;

UPDATE file_additions
SET asset_uri = 'file://' || asset_uri
WHERE asset_uri NOT LIKE '%://%';
