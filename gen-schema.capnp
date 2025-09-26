@0xa1b2c3d4e5f6a7b8;

# Cap'n Proto schema for Gen main crate
# This schema defines structures specific to the main gen crate

using GenModels = import "/gen-models/gen-models.capnp";

# Operation patch structure for serializing and deserializing patches
struct OperationPatch {
  operation @0 :GenModels.Operation;
  files @1 :List(GenModels.FileAddition);
  summary @2 :GenModels.OperationSummary;
  dependencies @3 :GenModels.DependencyModels;  # Serialized DependencyModels as bytes
  changeset @4 :GenModels.DatabaseChangeset;
}

# Collection of operation patches
struct OperationPatches {
  patches @0 :List(OperationPatch);
}