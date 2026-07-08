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
        let sequences = Node::get_sequences_by_node_ids(conn, &node_ids);

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

    /// The slice and local offset (0-based, within that slice's block) at the
    /// midpoint of this locus's total sequence length. `None` for an empty locus.
    pub fn midpoint(&self) -> Option<(GraphNodeSlice, usize)> {
        let total: usize = self.slices.iter().map(|s| s.end - s.start).sum();
        if total == 0 {
            return None;
        }
        let half = total / 2;
        let mut consumed = 0;
        for (index, slice) in self.slices.iter().enumerate() {
            let len = slice.end - slice.start;
            let is_last = index == self.slices.len() - 1;
            if consumed + len > half || is_last {
                return Some((*slice, slice.start + (half - consumed)));
            }
            consumed += len;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use gen_core::HashId;
    use gen_graph::GraphNode;

    use super::*;

    fn node(id: u8, start: i64, end: i64) -> GraphNode {
        GraphNode {
            node_id: HashId([id; 32]),
            sequence_start: start,
            sequence_end: end,
        }
    }

    fn slice(block: GraphNode, start: usize, end: usize) -> GraphNodeSlice {
        GraphNodeSlice {
            block,
            start,
            end,
            strand: Strand::Forward,
        }
    }

    #[test]
    fn midpoint_single_slice_is_centered_within_it() {
        let block = node(1, 0, 10);
        let locus = GraphLocus {
            slices: vec![slice(block, 2, 8)],
        };
        // length 6, half = 3, so local offset = start(2) + 3 = 5
        assert_eq!(locus.midpoint(), Some((slice(block, 2, 8), 5)));
    }

    #[test]
    fn midpoint_spans_into_second_slice() {
        let first = node(1, 0, 10);
        let second = node(2, 0, 10);
        let locus = GraphLocus {
            slices: vec![slice(first, 8, 10), slice(second, 0, 10)],
        };
        // total length 12, half = 6; first slice covers 2 bases (consumed=2),
        // midpoint falls 4 bases into the second slice.
        assert_eq!(locus.midpoint(), Some((slice(second, 0, 10), 4)));
    }

    #[test]
    fn midpoint_of_empty_locus_is_none() {
        let locus = GraphLocus { slices: vec![] };
        assert_eq!(locus.midpoint(), None);
    }
}
