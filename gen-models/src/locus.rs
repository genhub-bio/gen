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

/// A slice of a single graph block: the block itself plus local start/end byte
/// offsets within that block's sequence.  Middle blocks in a multi-block locus
/// span the full block (`start = 0`, `end = block.length()`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockSlice {
    pub block: GraphNode,
    /// Local start offset within the block's sequence slice (`0..block.length()`).
    pub start: usize,
    /// Local end offset, exclusive (`start..=block.length()`).
    pub end: usize,
}

impl BlockSlice {
    pub fn full(block: GraphNode) -> Self {
        Self {
            block,
            start: 0,
            end: block.length() as usize,
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
