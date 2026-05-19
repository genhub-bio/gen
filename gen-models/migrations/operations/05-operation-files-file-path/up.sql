ALTER TABLE operation_files
ADD COLUMN file_path TEXT NOT NULL DEFAULT '';

UPDATE operation_files
SET file_path = (
  SELECT CASE
    WHEN fa.asset_uri LIKE 'file://%' THEN substr(fa.asset_uri, 8)
    ELSE fa.asset_uri
  END
  FROM file_additions fa
  WHERE fa.id = operation_files.file_addition_id
);

UPDATE file_additions
SET asset_uri = 'file://.gen/assets/' || lower(hex(checksum)) || '.' ||
  CASE file_type
    WHEN 'gb' THEN 'gb'
    WHEN 'fasta' THEN 'fa'
    WHEN 'gfa' THEN 'gfa'
    WHEN 'gaf' THEN 'gaf'
    WHEN 'vcf' THEN 'vcf'
    WHEN 'changeset' THEN 'cs'
    WHEN 'csv' THEN 'csv'
    WHEN 'gff3' THEN 'gff3'
    WHEN 'bed' THEN 'bed'
    WHEN 'tabix' THEN 'tbi'
    ELSE 'none'
  END
WHERE asset_uri LIKE 'file://%';
