CREATE TABLE annotation_groups (
  name TEXT PRIMARY KEY NOT NULL
) STRICT;

CREATE TABLE annotations (
  id BLOB PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  annotation_group TEXT NOT NULL,
  accession_id BLOB NOT NULL,
  FOREIGN KEY(accession_id) REFERENCES accessions(id),
  FOREIGN KEY(annotation_group) REFERENCES annotation_groups(name)
) STRICT;
CREATE UNIQUE INDEX annotations_uidx ON annotations(accession_id, annotation_group, name);

CREATE TABLE annotation_group_samples (
  annotation_group TEXT NOT NULL,
  sample_name TEXT NOT NULL,
  PRIMARY KEY (annotation_group, sample_name),
  FOREIGN KEY(annotation_group) REFERENCES annotation_groups(name),
  FOREIGN KEY(sample_name) REFERENCES samples(name)
) STRICT;
