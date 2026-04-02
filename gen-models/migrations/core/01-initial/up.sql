CREATE TABLE gen_metadata (
  db_uuid TEXT PRIMARY KEY NOT NULL
) STRICT;

CREATE TABLE collections (
  name TEXT PRIMARY KEY NOT NULL
) STRICT;

CREATE TABLE samples (
  name TEXT PRIMARY KEY NOT NULL
) STRICT;

CREATE TABLE sequences (
  hash BLOB PRIMARY KEY NOT NULL,
  sequence_type TEXT NOT NULL,
  sequence TEXT NOT NULL,
  name TEXT NOT NULL,
  file_path TEXT NOT NULL,
  length INTEGER NOT NULL
) STRICT;

CREATE TABLE nodes (
  id BLOB PRIMARY KEY NOT NULL,
  sequence_hash BLOB NOT NULL,
  FOREIGN KEY(sequence_hash) REFERENCES sequences(hash)
) STRICT;

CREATE TABLE block_groups (
  id BLOB PRIMARY KEY NOT NULL,
  collection_name TEXT NOT NULL,
  sample_name TEXT NOT NULL,
  name TEXT NOT NULL,
  created_on INTEGER NOT NULL,
  FOREIGN KEY(collection_name) REFERENCES collections(name),
  FOREIGN KEY(sample_name) REFERENCES samples(name)
) STRICT;
CREATE UNIQUE INDEX block_group_uidx ON block_groups(collection_name, sample_name, name);

CREATE TABLE paths (
  id BLOB PRIMARY KEY NOT NULL,
  block_group_id BLOB NOT NULL,
  name TEXT NOT NULL,
  created_on INTEGER NOT NULL,
  FOREIGN KEY(block_group_id) REFERENCES block_groups(id)
) STRICT;
CREATE UNIQUE INDEX path_uidx ON paths(block_group_id, name);

CREATE TABLE accessions (
  id BLOB PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
--  path accessions can reference other path accessions
  path_id BLOB NOT NULL,
  parent_accession_id BLOB,
  FOREIGN KEY(path_id) REFERENCES paths(id),
  FOREIGN KEY(parent_accession_id) REFERENCES accessions(id)
) STRICT;
CREATE UNIQUE INDEX accession_uidx ON accessions(path_id, parent_accession_id, name) WHERE parent_accession_id is not null;
CREATE UNIQUE INDEX accession_null_aid_uidx ON accessions(path_id, name) WHERE parent_accession_id is null;

CREATE TABLE accession_edges (
  id BLOB PRIMARY KEY NOT NULL,
  source_node_id BLOB NOT NULL,
  source_coordinate INTEGER NOT NULL,
  source_strand TEXT NOT NULL,
  target_node_id BLOB NOT NULL,
  target_coordinate INTEGER NOT NULL,
  target_strand TEXT NOT NULL,
  chromosome_index INTEGER NOT NULL,
  FOREIGN KEY(source_node_id) REFERENCES nodes(id),
  FOREIGN KEY(target_node_id) REFERENCES nodes(id)
) STRICT;
CREATE UNIQUE INDEX accession_edge_uidx ON accession_edges(source_node_id, source_coordinate, source_strand, target_node_id, target_coordinate, target_strand, chromosome_index);

CREATE TABLE accession_paths (
  id BLOB PRIMARY KEY NOT NULL,
  accession_id BLOB NOT NULL,
  index_in_path INTEGER NOT NULL,
  edge_id BLOB NOT NULL,
  FOREIGN KEY(edge_id) REFERENCES accession_edges(id),
  FOREIGN KEY(accession_id) REFERENCES accessions(id)
) STRICT;
CREATE UNIQUE INDEX accession_path_uidx ON accession_paths(accession_id, edge_id, index_in_path);

CREATE TABLE edges (
  id BLOB PRIMARY KEY NOT NULL,
  source_node_id BLOB NOT NULL,
  source_coordinate INTEGER NOT NULL,
  source_strand TEXT NOT NULL,
  target_node_id BLOB NOT NULL,
  target_coordinate INTEGER NOT NULL,
  target_strand TEXT NOT NULL,
  FOREIGN KEY(source_node_id) REFERENCES nodes(id),
  FOREIGN KEY(target_node_id) REFERENCES nodes(id)
) STRICT;
CREATE UNIQUE INDEX edge_uidx ON edges(source_node_id, source_coordinate, source_strand, target_node_id, target_coordinate, target_strand);

CREATE TABLE path_edges (
  id BLOB PRIMARY KEY NOT NULL,
  path_id BLOB NOT NULL,
  edge_id BLOB NOT NULL,
  index_in_path INTEGER NOT NULL,
  FOREIGN KEY(edge_id) REFERENCES edges(id),
  FOREIGN KEY(path_id) REFERENCES paths(id)
) STRICT;
CREATE UNIQUE INDEX path_edges_uidx ON path_edges(path_id, edge_id, index_in_path);

CREATE TABLE block_group_edges (
  id BLOB PRIMARY KEY NOT NULL,
  block_group_id BLOB NOT NULL,
  edge_id BLOB NOT NULL,
  chromosome_index INTEGER,
  phased INTEGER NOT NULL,
  created_on INTEGER NOT NULL, 
  FOREIGN KEY(block_group_id) REFERENCES block_groups(id),
  FOREIGN KEY(edge_id) REFERENCES edges(id)
) STRICT;
CREATE UNIQUE INDEX block_group_edges_uidx ON block_group_edges(block_group_id, edge_id, chromosome_index, phased);

CREATE TABLE reference_aliases (
    reference_name TEXT NOT NULL,
    refseq_accession_id TEXT NOT NULL,
    genbank_accession_id TEXT NOT NULL
);

INSERT INTO gen_metadata (db_uuid) values (lower(
    hex(randomblob(4)) || '-' || hex(randomblob(2)) || '-' || '4' ||
    substr(hex( randomblob(2)), 2) || '-' ||
    substr('AB89', 1 + (abs(random()) % 4) , 1)  ||
    substr(hex(randomblob(2)), 2) || '-' ||
    hex(randomblob(6))
  ));
INSERT INTO sequences (hash, sequence_type, sequence, name, file_path, "length") values (X'84d6adbd5395281933fe41e877d3a7f02a3b1990a65be1901b2c91fc685e083b', "OTHER", "start-node-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy", "", "", 64), (X'1c7dfc64977b0838af0762d7333dcb64c175b15e65a70099ec38f46bf1a15ea3', "OTHER", "end-node-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", "", "", 64);
INSERT INTO nodes (id, sequence_hash) values (X'84d6adbd5395281933fe41e877d3a7f02a3b1990a65be1901b2c91fc685e083b', X'84d6adbd5395281933fe41e877d3a7f02a3b1990a65be1901b2c91fc685e083b');
INSERT INTO nodes (id, sequence_hash) values (X'1c7dfc64977b0838af0762d7333dcb64c175b15e65a70099ec38f46bf1a15ea3', X'1c7dfc64977b0838af0762d7333dcb64c175b15e65a70099ec38f46bf1a15ea3');

INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id) values ('E. coli K-12 MG1655', 'NC_000913.3', 'U00096.3');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id) values ('E. coli O157:H7 Sakai', 'NC_002695.2', 'BA000007.3');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id) values ('E. coli O157:H7 Sakai', 'NC_002127.1', 'AB011548.2');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id) values ('E. coli O157:H7 Sakai', 'NC_002128.1', 'AB011549.2');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id) values ('E. coli CFT073', 'NC_004431.1', 'AE014075.1');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id) values ('E. coli O157:H7 EDL933', 'NC_002655.2', 'AE005174.2');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id) values ('E. coli O157:H7 EDL933', 'NC_007414.1', 'AF074613.1');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id) values ('E. coli BL21(DE3)', 'NC_012892.2', 'AM946981.2');




