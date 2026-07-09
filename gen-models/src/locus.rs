//! Nameless addresses in graph space.
//!
//! A specific location on the genome can be specified by a region on a path
//! (addressing in linear space), for example "chr1:123-456". If there is no
//! path defined that traverses the location you need to address in graph
//! space. There are two kinds of graphs to consider in our datamodel:
//!   - Graphs with Nodes as defined in the database and edges with start/end
//!     coordinates.
//!   - Graphs with Blocks aka GraphNodes that are carved out Nodes and which
//!     most closely resemble the pangenomics literature.
//!
//! The search algorithm, for example, is a lot easier to understand on Blocks
//! rather than Nodes. It's output can be used to address graph space and in
//! the form of a list of GraphNodeSlices: segments from blocks, in the
//! coordinate reference frame (left side = 0). But to store graph changes in
//! the additive model in the database we must convert back to the Node format.

use gen_core::{HashId, Strand};
pub use gen_graph::GraphNodeSlice;

use crate::{db::GraphConnection, node::Node, sequence::reverse_complement};

/// A region in graph space expressed as an ordered list of block slices.
/// Each slice carries its own strand, allowing trans-spliced loci where
/// individual exons may come from opposite strands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphLocus {
    pub slices: Vec<GraphNodeSlice>,
}

impl GraphLocus {
    /// Concatenate the sequence bytes covered by this locus.
    pub fn sequence(&self, conn: &GraphConnection) -> Vec<u8> {
        let node_ids: Vec<HashId> = self.slices.iter().map(|s| s.block.node_id).collect();
        let sequences = Node::get_sequences_by_node_ids(conn, &node_ids, None);

        let mut out = Vec::new();
        for s in &self.slices {
            let full = sequences[&s.block.node_id]
                .get_sequence(None, None)
                .expect("sequence data corrupt")
                .into_bytes();
            let block_start = s.block.sequence_start as usize;
            let text = &full[block_start..block_start + s.block.length() as usize];
            let slice_bytes = &text[s.start..s.end];
            if s.strand == Strand::Reverse {
                out.extend_from_slice(&reverse_complement(slice_bytes));
            } else {
                out.extend_from_slice(slice_bytes);
            }
        }

        out
    }
}
