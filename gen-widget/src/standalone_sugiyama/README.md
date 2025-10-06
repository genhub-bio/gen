# Standalone Sugiyama Algorithm Module

This directory contains a self-contained implementation of the Sugiyama algorithm for hierarchical graph layout.

## Contents
- `mod.rs` - Main algorithm entry point with `custom_layout()` and `start()` functions
- `config.rs` - Configuration types (Config, RankingType, CrossingMinimization)
- `types.rs` - Type aliases (Layout, Layouts)
- `util.rs` - Utility functions (weakly_connected_components, radix_sort, iterate)
- `p0_cycle_removal/` - Phase 0: Cycle removal
- `p1_layering/` - Phase 1: Layering/ranking
- `p2_reduce_crossings/` - Phase 2: Crossing reduction
- `p3_calculate_coordinates/` - Phase 3: Coordinate calculation

## Public API
- `run_sugiyama_algorithm()` - Main function for the custom layout we need (modified to handle dummy nodes explicitly and return an intermediate checkpoint )
- `assign_coordinates()` - Complete the algorithm, the node sizes only come into play from this point.
- `Vertex` - Node type with layout information
- `Edge` - Edge type with layout information  
- `Config` - Configuration options
- `RankingType` - Vertical placement strategy
- `CrossingMinimization` - Crossing reduction heuristic 