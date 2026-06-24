@0xf7c8a1b2d3e4f5a6;

# Cap'n Proto schema for Gen genetic sequence version control system models
# This schema maps all the models defined in gen-models to Cap'n Proto format

using Core = import "gen-core.capnp";

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
  gff3 @8;
  bed @9;
  tabix @10;
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

struct EdgeChunk {
  values @0 :List(Edge);
}

# Collection and sample models
struct Collection {
  name @0 :Text;
}

struct Sample {
  name @0 :Text;
  isReference @1 :Bool;
}

struct SampleLineage {
  parentSampleName @0 :Text;
  childSampleName @1 :Text;
}

# Block group models
struct BlockGroup {
  id @0 :List(UInt8);
  collectionName @1 :Text;
  sampleName @2 :Text;
  name @3 :Text;
  createdOn @4 :Int64;
  parentBlockGroupId :union {
    none @5 :Void;
    some @6 :List(UInt8);
  }
  isDefault @7 :Bool;
}

struct BlockGroupEdge {
  id @0 :List(UInt8);
  blockGroupId @1 :List(UInt8);
  edgeId @2 :List(UInt8);
  chromosomeIndex @3 :Int64;
  phased @4 :Int64;
  createdOn @5 :Int64;
}

struct BlockGroupEdgeChunk {
  values @0 :List(BlockGroupEdge);
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
  blockGroupId @2 :List(UInt8);
  parentAccessionId :union {
    none @3 :Void;
    some @4 :List(UInt8);
  }
}

struct AccessionNode {
  id @0 :List(UInt8);
  accessionId @1 :List(UInt8);
  nodeId @2 :List(UInt8);
  sequenceStart @3 :Int64;
  sequenceEnd @4 :Int64;
  strand @5 :Core.Strand;
  indexInPath @6 :Int64;
}

# Operation and version control models
struct Operation {
  hash @0 :List(UInt8);
  parentHash :union {
    none @1 :Void;
    some @2 :List(UInt8);
  }
  changeType @3 :Text;
  createdOn @4 :Int64;
}

struct FileAddition {
  id @0 :List(UInt8);
  assetUri @1 :Text;
  fileType @2 :FileType;
  checksum @3 :List(UInt8);
}

struct OperationFile {
  filename @0 :Text;
  filePath @1 :Text;
  fileType @2 :FileType;
}

struct OperationInfo {
  files @0 :List(OperationFile);
  description @1 :Text;
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
  dbUuid @0 :Text;
  name @1 :Text;
  path @2 :Text;
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
  operationFileAdditions @1 :List(ManifestOperationFileAddition);
  annotationFileAdditions @4 :List(FileAddition);
  annotationFileDetails @5 :List(ManifestAnnotationFileAddition);
  operationSummary :union {
    none @2 :Void;
    some @3 :OperationSummary;
  }
}

struct ManifestOperationFileAddition {
  fileAddition @0 :FileAddition;
  filename @1 :Text;
  filePath @2 :Text;
}

struct ManifestAnnotationFileAddition {
  fileAddition @0 :FileAddition;
  name :union {
    none @1 :Void;
    some @2 :Text;
  }
  indexFileAddition :union {
    none @3 :Void;
    some @4 :FileAddition;
  }
}

struct Manifest {
  manifestVersion @0 :Text;
  branchName @1 :Text;
  endHash :union {
    none @2 :Void;
    some @3 :List(UInt8);
  }
  operations @4 :List(ManifestOperation);
}

struct ManifestDiff {
  missingInManifest2 @0 :List(ManifestOperation);
  missingInManifest1 @1 :List(ManifestOperation);
}

struct AnnotationInterval {
  name @0 :Text;
  start @1 :Int64;
  end @2 :Int64;
}

struct Annotation {
  id @0 :List(UInt8);
  name @1 :Text;
  annotationGroup @2 :Text;
  accessionId @3 :List(UInt8);
  extra @4 :Text;
}

struct AnnotationGroupSample {
  annotationGroup @0 :Text;
  sampleName @1 :Text;
}

struct AnnotationGroup {
  name @0 :Text;
}

struct ChangesetModels {
  collections @0 :List(Collection);
  samples @1 :List(Sample);
  sequences @2 :List(Sequence);
  blockGroups @3 :List(BlockGroup);
  nodes @4 :List(Node);
  edgeChunks @5 :List(EdgeChunk);
  blockGroupEdgeChunks @6 :List(BlockGroupEdgeChunk);
  paths @7 :List(Path);
  pathEdges @8 :List(PathEdge);
  accessions @9 :List(Accession);
  accessionNodes @10 :List(AccessionNode);
  annotationGroups @11 :List(AnnotationGroup);
  annotations @12 :List(Annotation);
  annotationGroupSamples @13 :List(AnnotationGroupSample);
  sampleLineages @14 :List(SampleLineage);
}

struct DependencyModels {
  collections @0 :List(Collection);
  samples @1 :List(Sample);
  sequences @2 :List(Sequence);
  blockGroup @3 :List(BlockGroup);
  nodes @4 :List(Node);
  edgeChunks @5 :List(EdgeChunk);
  paths @6 :List(Path);
  accessions @7 :List(Accession);
  accessionNodes @8 :List(AccessionNode);
}
