UPDATE file_additions
SET asset_uri = COALESCE(
  (
    SELECT 'file://' || of.file_path
    FROM operation_files of
    WHERE of.file_addition_id = file_additions.id
      AND of.file_path NOT LIKE '%://%'
    ORDER BY of.id
    LIMIT 1
  ),
  asset_uri
)
WHERE asset_uri LIKE 'file://.gen/assets/%';

ALTER TABLE operation_files
DROP COLUMN file_path;
