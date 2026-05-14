//! Nameless addresses in graph space.
//!
//! Two coordinate systems for the same underlying position:
//!
//! - **Block space** (`GraphLocus`): a sequence of `BlockSlice`s, each one a
//!   `GraphNode` with local start/end byte offsets.  This is what sequence
//!   search returns — natural when you're navigating the rendered graph or
//!   computing visual positions.

use gen_core::{HashId, Strand};
use gen_graph::GraphNode;
use serde::{Deserialize, Serialize};

use crate::{db::GraphConnection, node::Node};

// ---------------------------------------------------------------------------
// Block space
// ---------------------------------------------------------------------------

/// A slice of a single graph node: the node itself plus local start/end byte
/// offsets within that node's sequence.  Middle blocks in a multi-node locus
/// span the full node (`start = 0`, `end = node.length()`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockSlice {
    pub node: GraphNode,
    /// Local start offset within the node's sequence slice (`0..node.length()`).
    pub start: usize,
    /// Local end offset, exclusive (`start..=node.length()`).
    pub end: usize,
}

impl BlockSlice {
    pub fn full(node: GraphNode) -> Self {
        Self {
            node,
            start: 0,
            end: node.length() as usize,
        }
    }
}

/// A walk through graph space expressed as an ordered list of node slices.
///
/// Returned by sequence search.  Natural for visual navigation and rendering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphLocus {
    pub slices: Vec<BlockSlice>,
    pub strand: Strand,
}

impl GraphLocus {
    /// Concatenate the sequence bytes covered by this locus.
    pub fn sequence(&self, conn: &GraphConnection) -> Vec<u8> {
        let node_ids: Vec<HashId> = self.slices.iter().map(|s| s.node.node_id).collect();
        let sequences = Node::get_sequences_by_node_ids(conn, &node_ids);

        let mut out = Vec::new();
        for s in &self.slices {
            let full = sequences[&s.node.node_id]
                .get_sequence(None, None)
                .expect("sequence data corrupt")
                .into_bytes();
            let block_start = s.node.sequence_start as usize;
            let text = &full[block_start..block_start + s.node.length() as usize];
            out.extend_from_slice(&text[s.start..s.end]);
        }

        if self.strand == Strand::Reverse {
            reverse_complement(&out)
        } else {
            out
        }
    }
}

fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&base| match base.to_ascii_uppercase() {
            b'A' => b'T',
            b'T' => b'A',
            b'C' => b'G',
            b'G' => b'C',
            b'U' => b'A',
            b'N' => b'N',
            b'R' => b'Y',
            b'Y' => b'R',
            b'S' => b'S',
            b'W' => b'W',
            b'K' => b'M',
            b'M' => b'K',
            b'B' => b'V',
            b'V' => b'B',
            b'D' => b'H',
            b'H' => b'D',
            _ => base,
        })
        .collect()
}
