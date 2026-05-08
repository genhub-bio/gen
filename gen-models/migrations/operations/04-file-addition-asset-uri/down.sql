UPDATE file_additions
SET asset_uri = substr(asset_uri, 8)
WHERE asset_uri LIKE 'file://%';

ALTER TABLE file_additions
RENAME COLUMN asset_uri TO file_path;
