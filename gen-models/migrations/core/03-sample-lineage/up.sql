CREATE TABLE sample_lineage (
  parent_sample_name TEXT NOT NULL,
  child_sample_name TEXT NOT NULL,
  PRIMARY KEY (parent_sample_name, child_sample_name),
  FOREIGN KEY(parent_sample_name) REFERENCES samples(name),
  FOREIGN KEY(child_sample_name) REFERENCES samples(name),
  CHECK (parent_sample_name != child_sample_name)
) STRICT;

CREATE INDEX sample_lineage_parent_idx ON sample_lineage(parent_sample_name);
CREATE INDEX sample_lineage_child_idx ON sample_lineage(child_sample_name);
