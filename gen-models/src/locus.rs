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

use gen_core::{HashId, Strand, Workspace};
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
    /// Number of bases the locus covers.
    pub fn len(&self) -> usize {
        self.slices
            .iter()
            .map(|slice| slice.end - slice.start)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The slice holding the base at `offset` in reading order, with that base's offset within
    /// the slice's block. A reverse-strand slice reads toward lower offsets, so its first base
    /// is its highest. Callers that centre a view on a match (Python, R and the terminal viewer)
    /// share this so they all land on the same base.
    pub fn slice_at(&self, mut offset: usize) -> Option<(GraphNodeSlice, usize)> {
        for slice in &self.slices {
            let length = slice.end - slice.start;
            if offset < length {
                let local_offset = if slice.strand == Strand::Reverse {
                    slice.end - 1 - offset
                } else {
                    slice.start + offset
                };
                return Some((*slice, local_offset));
            }
            offset -= length;
        }
        None
    }

    /// The base halfway through the locus in reading order, or `None` for an empty locus.
    pub fn midpoint(&self) -> Option<(GraphNodeSlice, usize)> {
        self.slice_at(self.len() / 2)
    }

    /// Concatenate the sequence bytes covered by this locus.
    pub fn sequence(&self, conn: &GraphConnection, workspace: &Workspace) -> Vec<u8> {
        let node_ids: Vec<HashId> = self.slices.iter().map(|s| s.block.node_id).collect();
        let sequences = Node::get_sequences_by_node_ids(conn, workspace, &node_ids, None);

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

#[cfg(test)]
mod tests {
    use gen_core::HashId;
    use gen_graph::GraphNode;

    use super::*;

    fn slice(label: &str, start: usize, end: usize, strand: Strand) -> GraphNodeSlice {
        GraphNodeSlice {
            block: GraphNode {
                node_id: HashId::convert_str(label),
                sequence_start: 0,
                sequence_end: 20,
            },
            start,
            end,
            strand,
        }
    }

    fn midpoint_of(slices: Vec<GraphNodeSlice>) -> (HashId, usize) {
        let (found, offset) = GraphLocus { slices }
            .midpoint()
            .expect("should have a midpoint");
        (found.block.node_id, offset)
    }

    #[test]
    fn test_midpoint_forward_single_slice() {
        let odd = midpoint_of(vec![slice("a", 4, 9, Strand::Forward)]);
        let even = midpoint_of(vec![slice("a", 4, 10, Strand::Forward)]);

        assert_eq!(odd, (HashId::convert_str("a"), 6));
        assert_eq!(even, (HashId::convert_str("a"), 7));
    }

    #[test]
    fn test_midpoint_reverse_single_slice_counts_down_from_the_end() {
        let odd = midpoint_of(vec![slice("a", 4, 9, Strand::Reverse)]);
        let even = midpoint_of(vec![slice("a", 4, 10, Strand::Reverse)]);

        assert_eq!(odd, (HashId::convert_str("a"), 6));
        assert_eq!(even, (HashId::convert_str("a"), 6));
    }

    #[test]
    fn test_midpoint_multi_slice_forward() {
        let midpoint = midpoint_of(vec![
            slice("a", 0, 3, Strand::Forward),
            slice("b", 5, 10, Strand::Forward),
        ]);

        assert_eq!(midpoint, (HashId::convert_str("b"), 6));
    }

    #[test]
    fn test_midpoint_multi_slice_reverse_reads_slices_in_order() {
        let midpoint = midpoint_of(vec![
            slice("a", 0, 3, Strand::Reverse),
            slice("b", 5, 10, Strand::Reverse),
        ]);

        assert_eq!(midpoint, (HashId::convert_str("b"), 8));
    }

    #[test]
    fn test_midpoint_mixed_strands() {
        let midpoint = midpoint_of(vec![
            slice("a", 0, 2, Strand::Forward),
            slice("b", 5, 9, Strand::Reverse),
        ]);

        assert_eq!(midpoint, (HashId::convert_str("b"), 7));
    }

    #[test]
    fn test_midpoint_of_empty_locus_is_none() {
        assert!(GraphLocus { slices: vec![] }.midpoint().is_none());
        assert!(
            GraphLocus {
                slices: vec![slice("a", 3, 3, Strand::Forward)]
            }
            .midpoint()
            .is_none()
        );
    }
}
