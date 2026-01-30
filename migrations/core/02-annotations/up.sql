CREATE TABLE annotations (
  id BLOB PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  annotation_type TEXT NOT NULL,
  accession_id BLOB NOT NULL,
  FOREIGN KEY(accession_id) REFERENCES accessions(id)
) STRICT;
CREATE UNIQUE INDEX annotations_uidx ON annotations(accession_id, annotation_type, name);

CREATE TABLE annotations_sample (
  annotation_id BLOB NOT NULL,
  sample_name TEXT NOT NULL,
  PRIMARY KEY (annotation_id, sample_name),
  FOREIGN KEY(annotation_id) REFERENCES annotations(id),
  FOREIGN KEY(sample_name) REFERENCES samples(name)
) STRICT;
