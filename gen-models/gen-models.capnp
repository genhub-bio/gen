@0xf7c8a1b2d3e4f5a6;

# Cap'n Proto schema for Gen genetic sequence version control system models
# This schema maps all the models defined in gen-models to Cap'n Proto format

using Core = import "/gen-core/gen-core.capnp";

# Core enums and types

enum FileType {
  genBank @0;
  fasta @1;
  gfa @2;
  gaf @3;
  vcf @4;
  changeset @5;
  csv @6;
  none @7;
}

# Core sequence and node models
struct Sequence {
  hash @0 :List(UInt8);
  sequenceType @1 :Text;
  sequence @2 :Text;
  name @3 :Text;
  filePath @4 :Text;
  length @5 :Int64;
  externalSequence @6 :Bool;
}

struct Node {
  id @0 :List(UInt8);
  sequenceHash @1 :List(UInt8);
}

# Graph edge models
struct Edge {
  id @0 :List(UInt8);
  sourceNodeId @1 :List(UInt8);
  sourceCoordinate @2 :Int64;
  sourceStrand @3 :Core.Strand;
  targetNodeId @4 :List(UInt8);
  targetCoordinate @5 :Int64;
  targetStrand @6 :Core.Strand;
}

# Collection and sample models
struct Collection {
  name @0 :Text;
}

struct Sample {
  name @0 :Text;
}

# Block group models
struct BlockGroup {
  id @0 :List(UInt8);
  collectionName @1 :Text;
  sampleName :union {
    none @2 :Void;
    some @3 :Text;
  }
  name @4 :Text;
  createdOn @5 :Int64;
}

struct BlockGroupEdge {
  id @0 :List(UInt8);
  blockGroupId @1 :List(UInt8);
  edgeId @2 :List(UInt8);
  chromosomeIndex @3 :Int64;
  phased @4 :Int64;
  createdOn @5 :Int64;
}

# Path models
struct Path {
  id @0 :List(UInt8);
  blockGroupId @1 :List(UInt8);
  name @2 :Text;
  createdOn @3 :Int64;
}

struct PathEdge {
  id @0 :List(UInt8);
  pathId @1 :List(UInt8);
  edgeId @2 :List(UInt8);
  indexInPath @3 :Int64;
}

# Accession models
struct Accession {
  id @0 :List(UInt8);
  name @1 :Text;
  pathId @2 :List(UInt8);
  parentAccessionId :union {
    none @3 :Void;
    some @4 :List(UInt8);
  }
}

struct AccessionEdge {
  id @0 :List(UInt8);
  sourceNodeId @1 :List(UInt8);
  sourceCoordinate @2 :Int64;
  sourceStrand @3 :Core.Strand;
  targetNodeId @4 :List(UInt8);
  targetCoordinate @5 :Int64;
  targetStrand @6 :Core.Strand;
  chromosomeIndex @7 :Int64;
}

struct AccessionPath {
  id @0 :List(UInt8);
  accessionId @1 :List(UInt8);
  indexInPath @2 :Int64;
  edgeId @3 :List(UInt8);
}

# Operation and version control models
struct Operation {
  hash @0 :List(UInt8);
  parentHash :union {
    none @1 :Void;
    some @2 :List(UInt8);
  }
  changeType @3 :Text;
}

struct FileAddition {
  id @0 :List(UInt8);
  filePath @1 :Text;
  fileType @2 :FileType;
  checksum @3 :List(UInt8);
}

struct OperationSummary {
  id @0 :Int64;
  operationHash @1 :List(UInt8);
  summary @2 :Text;
}

struct Branch {
  id @0 :Int64;
  name @1 :Text;
  currentOperationHash :union {
    none @2 :Void;
    some @3 :List(UInt8);
  }
  remoteName :union {
    none @4 :Void;
    some @5 :Text;
  }
}

struct Remote {
  name @0 :Text;
  url @1 :Text;
}

struct Defaults {
  id @0 :Int64;
  dbName :union {
    none @1 :Void;
    some @2 :Text;
  }
  collectionName :union {
    none @3 :Void;
    some @4 :Text;
  }
  remoteName :union {
    none @5 :Void;
    some @6 :Text;
  }
}

# Database and metadata models
struct GenDatabase {
  id @0 :Int64;
  dbUuid @1 :Text;
  name @2 :Text;
  path @3 :Text;
}

struct Metadata {
  dbUuid @0 :Text;
}

# Changeset and manifest models
struct DatabaseChangeset {
  dbPath @0 :Text;
  changes @1 :ChangesetModels;
}

struct ManifestOperation {
  operation @0 :Operation;
  changesetHash @1 :Text;
  dependenciesHash @2 :Text;
  fileAdditions @3 :List(FileAddition);
  operationSummary :union {
    none @4 :Void;
    some @5 :OperationSummary;
  }
}

struct Manifest {
  manifestVersion @0 :Text;
  branchName @1 :Text;
  endHash @2 :List(UInt8);
  operations @3 :List(ManifestOperation);
}

struct ManifestDiff {
  missingInManifest2 @0 :List(ManifestOperation);
  missingInManifest1 @1 :List(ManifestOperation);
}

struct Annotation {
  name @0 :Text;
  start @1 :Int64;
  end @2 :Int64;
}

struct ChangesetModels {
  collections @0 :List(Collection);
  samples @1 :List(Sample);
  sequences @2 :List(Sequence);
  blockGroups @3 :List(BlockGroup);
  nodes @4 :List(Node);
  edges @5 :List(Edge);
  blockGroupEdges @6 :List(BlockGroupEdge);
  paths @7 :List(Path);
  pathEdges @8 :List(PathEdge);
  accessions @9 :List(Accession);
  accessionEdges @10 :List(AccessionEdge);
  accessionPaths @11 :List(AccessionPath);
}

struct DependencyModels {
  collections @0 :List(Collection);
  samples @1 :List(Sample);
  sequences @2 :List(Sequence);
  blockGroup @3 :List(BlockGroup);
  nodes @4 :List(Node);
  edges @5 :List(Edge);
  paths @6 :List(Path);
  accessions @7 :List(Accession);
  accessionEdges @8 :List(AccessionEdge);
}
