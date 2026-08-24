CREATE TABLE collections (
  name TEXT PRIMARY KEY NOT NULL
) STRICT;

CREATE TABLE samples (
  name TEXT PRIMARY KEY NOT NULL,
  is_reference INTEGER NOT NULL DEFAULT 0 CHECK (is_reference IN (0, 1))
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
  parent_block_group_id BLOB,
  is_default INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0, 1)),
  FOREIGN KEY(collection_name) REFERENCES collections(name),
  FOREIGN KEY(sample_name) REFERENCES samples(name),
  FOREIGN KEY(parent_block_group_id) REFERENCES block_groups(id)
) STRICT;
CREATE UNIQUE INDEX block_group_uidx
ON block_groups(
  collection_name,
  sample_name,
  name,
  IFNULL(hex(parent_block_group_id), '')
);
CREATE UNIQUE INDEX block_group_default_uidx
ON block_groups(collection_name, sample_name, name)
WHERE is_default = 1;
CREATE INDEX block_group_parent_idx ON block_groups(parent_block_group_id);

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
  block_group_id BLOB NOT NULL,
  parent_accession_id BLOB,
  FOREIGN KEY(block_group_id) REFERENCES block_groups(id),
  FOREIGN KEY(parent_accession_id) REFERENCES accessions(id)
) STRICT;
CREATE UNIQUE INDEX accession_uidx ON accessions(block_group_id, parent_accession_id, name) WHERE parent_accession_id is not null;
CREATE UNIQUE INDEX accession_null_aid_uidx ON accessions(block_group_id, name) WHERE parent_accession_id is null;

CREATE TABLE accession_nodes (
  id BLOB PRIMARY KEY NOT NULL,
  accession_id BLOB NOT NULL,
  node_id BLOB NOT NULL,
  sequence_start INTEGER NOT NULL,
  sequence_end INTEGER NOT NULL,
  strand TEXT NOT NULL,
  index_in_path INTEGER NOT NULL,
  FOREIGN KEY(node_id) REFERENCES nodes(id),
  FOREIGN KEY(accession_id) REFERENCES accessions(id)
) STRICT;
CREATE UNIQUE INDEX accession_nodes_path_uidx ON accession_nodes(accession_id, index_in_path);
CREATE UNIQUE INDEX accession_nodes_slice_uidx ON accession_nodes(accession_id, node_id, sequence_start, sequence_end, strand, index_in_path);

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
-- Edge IDs are deterministic hashes of the complete endpoint tuple, so the primary key is the
-- canonical identity constraint without a second copy of every endpoint in a unique index.
CREATE INDEX edge_source_idx ON edges(source_node_id, source_coordinate);
CREATE INDEX edge_target_node_idx ON edges(target_node_id);

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

CREATE TABLE annotation_groups (
  name TEXT PRIMARY KEY NOT NULL
) STRICT;

CREATE TABLE annotations (
  id BLOB PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  annotation_group TEXT NOT NULL,
  accession_id BLOB NOT NULL,
  extra TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(extra)),
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

CREATE TABLE reference_aliases (
  reference_name TEXT NOT NULL,
  refseq_accession_id TEXT,
  genbank_accession_id TEXT,
  ucsc_id TEXT,
  ensembl_id TEXT,
  custom_id TEXT,
  chromosome INTEGER
);
CREATE UNIQUE INDEX reference_alias_refseq_uidx ON reference_aliases(refseq_accession_id);

CREATE TABLE gen_asset_refs (
  id BLOB PRIMARY KEY NOT NULL,
  uri TEXT NOT NULL,
  file_type TEXT NOT NULL,
  checksum BLOB,
  size INTEGER,
  role TEXT NOT NULL,
  logical_path TEXT,
  name TEXT,
  created_on INTEGER NOT NULL,
  upstream_asset_ref_id BLOB,
  FOREIGN KEY(upstream_asset_ref_id) REFERENCES gen_asset_refs(id)
) STRICT;
CREATE INDEX gen_asset_refs_upstream_idx ON gen_asset_refs(upstream_asset_ref_id);

CREATE TABLE gen_operation_log (
  id BLOB PRIMARY KEY NOT NULL,
  operation_kind TEXT NOT NULL,
  command TEXT NOT NULL,
  created_on INTEGER NOT NULL
) STRICT;

CREATE TABLE gen_operation_assets (
  log_id BLOB NOT NULL,
  asset_ref_id BLOB NOT NULL,
  role TEXT NOT NULL,
  PRIMARY KEY (log_id, asset_ref_id, role),
  FOREIGN KEY(log_id) REFERENCES gen_operation_log(id),
  FOREIGN KEY(asset_ref_id) REFERENCES gen_asset_refs(id)
) STRICT;

INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('E. coli K-12 MG1655', 'NC_000913.3', 'U00096.3', 'U00096.3', '');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('E. coli O157:H7 Sakai', 'NC_002695.2', 'BA000007.3', 'BA000007.3', '');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('E. coli O157:H7 Sakai', 'NC_002127.1', 'AB011548.2', 'pOSAK1', '');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('E. coli O157:H7 Sakai', 'NC_002128.1', 'AB011549.2', 'pO157', '');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('E. coli CFT073', 'NC_004431.1', 'AE014075.1', 'AE014075.1', '');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('E. coli O157:H7 EDL933', 'NC_002655.2', 'AE005174.2', 'AE005174.2', '');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('E. coli O157:H7 EDL933', 'NC_007414.1', 'AF074613.1', 'pO157', '');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('E. coli BL21(DE3)', 'NC_012892.2', 'AM946981.2', 'AM946981.2', '');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000001.11', 'CM000663.2', 'chr1', '1');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000002.12', 'CM000664.2', 'chr2', '2');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000003.12', 'CM000665.2', 'chr3', '3');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000004.12', 'CM000666.2', 'chr4', '4');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000005.10', 'CM000667.2', 'chr5', '5');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000006.12', 'CM000668.2', 'chr6', '6');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000007.14', 'CM000669.2', 'chr7', '7');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000008.11', 'CM000670.2', 'chr8', '8');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000009.12', 'CM000671.2', 'chr9', '9');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000010.11', 'CM000672.2', 'chr10', '10');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000011.10', 'CM000673.2', 'chr11', '11');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000012.12', 'CM000674.2', 'chr12', '12');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000013.11', 'CM000675.2', 'chr13', '13');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000014.9', 'CM000676.2', 'chr14', '14');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000015.10', 'CM000677.2', 'chr15', '15');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000016.10', 'CM000678.2', 'chr16', '16');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000017.11', 'CM000679.2', 'chr17', '17');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000018.10', 'CM000680.2', 'chr18', '18');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000019.10', 'CM000681.2', 'chr19', '19');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000020.11', 'CM000682.2', 'chr20', '20');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000021.9', 'CM000683.2', 'chr21', '21');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000022.11', 'CM000684.2', 'chr22', '22');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000023.11', 'CM000685.2', 'chrX', 'X');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_000024.10', 'CM000686.2', 'chrY', 'Y');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Human', 'NC_012920.1', 'J01415.2', 'chrM', 'MT');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000067.7', 'CM000994.3', 'chr1', '1');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000068.8', 'CM000995.3', 'chr2', '2');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000069.7', 'CM000996.3', 'chr3', '3');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000070.7', 'CM000997.3', 'chr4', '4');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000071.7', 'CM000998.3', 'chr5', '5');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000072.7', 'CM000999.3', 'chr6', '6');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000073.7', 'CM001000.3', 'chr7', '7');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000074.7', 'CM001001.3', 'chr8', '8');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000075.7', 'CM001002.3', 'chr9', '9');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000076.7', 'CM001003.3', 'chr10', '10');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000077.7', 'CM001004.3', 'chr11', '11');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000078.7', 'CM001005.3', 'chr12', '12');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000079.7', 'CM001006.3', 'chr13', '13');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000080.7', 'CM001007.3', 'chr14', '14');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000081.7', 'CM001008.3', 'chr15', '15');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000082.7', 'CM001009.3', 'chr16', '16');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000083.7', 'CM001010.3', 'chr17', '17');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000084.7', 'CM001011.3', 'chr18', '18');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000085.7', 'CM001012.3', 'chr19', '19');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000086.8', 'CM001013.3', 'chrX', 'X');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_000087.8', 'CM001014.3', 'chrY', 'Y');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Mouse', 'NC_005089.1', 'AY172335.1', 'chrM', 'MT');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('C. elegans', 'NC_003279.8', 'BX284601.5', '', 'I');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('C. elegans', 'NC_003280.10', 'BX284602.5', '', 'II');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('C. elegans', 'NC_003281.10', 'BX284603.4', '', 'III');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('C. elegans', 'NC_003282.8', 'BX284604.4', '', 'IV');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('C. elegans', 'NC_003283.11', 'BX284605.5', '', 'V');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('C. elegans', 'NC_003284.9', 'BX284606.5', '', 'X');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('C. elegans', 'NC_001328.1', '', 'chrM', 'MtDNA');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Drosophila', 'NC_004354.4', 'AE014298.5', 'chrX', 'X');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Drosophila', 'NT_033779.5', 'AE014134.6', 'chr2L', '2L');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Drosophila', 'NT_033778.4', 'AE013599.5', 'chr2R', '2R');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Drosophila', 'NT_037436.4', 'AE014296.5', 'chr3L', '3L');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Drosophila', 'NT_033777.3', 'AE014297.3', 'chr3R', '3R');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Drosophila', 'NC_004353.4', 'AE014135.4', 'chr4', '4');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Drosophila', 'NC_024512.1', 'CP007106.1', 'chrY', 'Y');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Drosophila', 'NC_024511.2', 'KJ947872.2', 'chrM', 'mitochondrion_genome');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001133.9', 'BK006935.2', 'chrI', 'I', 1);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001134.8', 'BK006936.2', 'chrII', 'II', 2);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001135.5', 'BK006937.2', 'chrIII', 'III', 3);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001136.10', 'BK006938.2', 'chrIV', 'IV', 4);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001137.3', 'BK006939.2', 'chrV', 'V', 5);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001138.5', 'BK006940.2', 'chrVI', 'VI', 6);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001139.9', 'BK006941.2', 'chrVII', 'VII', 7);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001140.6', 'BK006934.2', 'chrVIII', 'VIII', 8);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001141.2', 'BK006942.2', 'chrIX', 'IX', 9);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001142.9', 'BK006943.2', 'chrX', 'X', 10);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001143.9', 'BK006944.2', 'chrXI', 'XI', 11);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001144.5', 'BK006945.2', 'chrXII', 'XII', 12);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001145.3', 'BK006946.2', 'chrXIII', 'XIII', 13);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001146.8', 'BK006947.3', 'chrXIV', 'XIV', 14);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001147.6', 'BK006948.2', 'chrXV', 'XV', 15);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, chromosome) values ('Yeast', 'NC_001148.4', 'BK006949.2', 'chrXVI', 'XVI', 16);
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Yeast', 'NC_001224.1', '', 'chrM', 'Mito');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018723.3', 'CM001378.3', 'chrA1', 'A1');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018724.3', 'CM001379.3', 'chrA2', 'A2');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018725.3', 'CM001380.3', 'chrA3', 'A3');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018726.3', 'CM001381.3', 'chrB1', 'B1');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018727.3', 'CM001382.3', 'chrB2', 'B2');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018728.3', 'CM001383.3', 'chrB3', 'B3');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018729.3', 'CM001384.3', 'chrB4', 'B4');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018730.3', 'CM001385.3', 'chrC1', 'C1');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018731.3', 'CM001386.3', 'chrC2', 'C2');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018732.3', 'CM001387.3', 'chrD1', 'D1');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018733.3', 'CM001388.3', 'chrD2', 'D2');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018734.3', 'CM001389.3', 'chrD3', 'D3');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018735.3', 'CM001390.3', 'chrD4', 'D4');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018736.3', 'CM001391.3', 'chrE1', 'E1');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018737.3', 'CM001392.3', 'chrE2', 'E2');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018738.3', 'CM001393.3', 'chrE3', 'E3');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018739.3', 'CM001394.3', 'chrF1', 'F1');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018740.3', 'CM001395.3', 'chrF2', 'F2');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_018741.3', 'CM001396.3', 'chrX', 'X');
INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id) values ('Domestic cat', 'NC_001700.1', '', 'chrM', 'MT');

INSERT INTO sequences (hash, sequence_type, sequence, name, file_path, "length") values (X'84d6adbd5395281933fe41e877d3a7f02a3b1990a65be1901b2c91fc685e083b', "OTHER", "start-node-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy", "", "", 64), (X'1c7dfc64977b0838af0762d7333dcb64c175b15e65a70099ec38f46bf1a15ea3', "OTHER", "end-node-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", "", "", 64);
INSERT INTO nodes (id, sequence_hash) values (X'84d6adbd5395281933fe41e877d3a7f0', X'84d6adbd5395281933fe41e877d3a7f02a3b1990a65be1901b2c91fc685e083b');
INSERT INTO nodes (id, sequence_hash) values (X'1c7dfc64977b0838af0762d7333dcb64', X'1c7dfc64977b0838af0762d7333dcb64c175b15e65a70099ec38f46bf1a15ea3');
