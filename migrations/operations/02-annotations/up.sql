CREATE TABLE annotation_files (
  id INTEGER PRIMARY KEY NOT NULL,
  operation_hash BLOB NOT NULL,
  file_addition_id BLOB NOT NULL,
  FOREIGN KEY(operation_hash) REFERENCES operations(hash),
  FOREIGN KEY(file_addition_id) REFERENCES file_additions(id)
) STRICT;
