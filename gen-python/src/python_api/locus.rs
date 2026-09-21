//! Locus arithmetic used by the Python editing API.
//!
//! These helpers extend `GraphLocus` from gen-models without changing that crate:
//! canonicalizing a locus to node-absolute ranges gives edits a target address that
//! survives block re-carving, and slicing / reverse complementing let Python callers
//! derive new targets from search results and annotations.

use gen_core::Strand;
use gen_graph::{GraphNode, GraphNodeSlice};
use gen_models::locus::GraphLocus;

pub(crate) trait GraphLocusExt {
    /// Total length of the sequence covered by this locus.
    fn length(&self) -> usize;

    /// Rewrite this locus as node-absolute ranges, independent of block carving.
    ///
    /// Each slice becomes a full-width slice of a `GraphNode` whose `sequence_start`
    /// and `sequence_end` are the node coordinates of the covered positions, so the same
    /// positions produce the same canonical locus no matter how later edits split their
    /// blocks. Consecutive slices of one node that abut on the same strand are merged.
    fn canonical(&self) -> GraphLocus;

    /// The same positions read from the opposite strand.
    fn reverse_complement(&self) -> GraphLocus;

    /// Sub-locus covering positions `start..end`, counted in reading order.
    ///
    /// Offsets run along the slices in order; within a reverse-strand slice they run
    /// from the slice's end, matching how `GraphLocus::sequence` reads it. Returns `None`
    /// when the range is empty, inverted, or extends past the end of the locus.
    fn slice(&self, start: usize, end: usize) -> Option<GraphLocus>;
}

impl GraphLocusExt for GraphLocus {
    fn length(&self) -> usize {
        self.slices
            .iter()
            .map(|slice| slice.end - slice.start)
            .sum()
    }

    fn canonical(&self) -> GraphLocus {
        let mut slices: Vec<GraphNodeSlice> = Vec::with_capacity(self.slices.len());
        for slice in &self.slices {
            if slice.start == slice.end {
                continue;
            }
            let node_start = slice.block.sequence_start + slice.start as i64;
            let node_end = slice.block.sequence_start + slice.end as i64;
            if let Some(previous) = slices.last_mut()
                && previous.block.node_id == slice.block.node_id
                && previous.strand == slice.strand
            {
                if slice.strand != Strand::Reverse && previous.block.sequence_end == node_start {
                    previous.block.sequence_end = node_end;
                    previous.end = previous.block.length() as usize;
                    continue;
                }
                if slice.strand == Strand::Reverse && node_end == previous.block.sequence_start {
                    previous.block.sequence_start = node_start;
                    previous.end = previous.block.length() as usize;
                    continue;
                }
            }
            slices.push(GraphNodeSlice {
                block: GraphNode {
                    node_id: slice.block.node_id,
                    sequence_start: node_start,
                    sequence_end: node_end,
                },
                start: 0,
                end: (node_end - node_start) as usize,
                strand: slice.strand,
            });
        }
        GraphLocus { slices }
    }

    fn reverse_complement(&self) -> GraphLocus {
        GraphLocus {
            slices: self
                .slices
                .iter()
                .rev()
                .map(|slice| GraphNodeSlice {
                    strand: if slice.strand == Strand::Unknown {
                        Strand::Reverse
                    } else {
                        slice.strand.complement()
                    },
                    ..*slice
                })
                .collect(),
        }
    }

    fn slice(&self, start: usize, end: usize) -> Option<GraphLocus> {
        if start >= end || end > self.length() {
            return None;
        }
        let mut slices = vec![];
        let mut offset = 0;
        for slice in &self.slices {
            let length = slice.end - slice.start;
            let overlap_start = start.max(offset);
            let overlap_end = end.min(offset + length);
            if overlap_start < overlap_end {
                let (local_start, local_end) = (overlap_start - offset, overlap_end - offset);
                let (slice_start, slice_end) = if slice.strand == Strand::Reverse {
                    (slice.end - local_end, slice.end - local_start)
                } else {
                    (slice.start + local_start, slice.start + local_end)
                };
                slices.push(GraphNodeSlice {
                    start: slice_start,
                    end: slice_end,
                    ..*slice
                });
            }
            offset += length;
        }
        Some(GraphLocus { slices })
    }
}

#[cfg(test)]
mod tests {
    use gen_core::{HashId, Strand};
    use gen_graph::{GraphNode, GraphNodeSlice};
    use gen_models::locus::GraphLocus;

    use crate::python_api::locus::GraphLocusExt as _;

    fn slice(
        node: &str,
        block: (i64, i64),
        local: (usize, usize),
        strand: Strand,
    ) -> GraphNodeSlice {
        GraphNodeSlice {
            block: GraphNode {
                node_id: HashId::convert_str(node),
                sequence_start: block.0,
                sequence_end: block.1,
            },
            start: local.0,
            end: local.1,
            strand,
        }
    }

    #[test]
    fn test_canonical_merges_blocks_carved_from_one_node() {
        let carved = GraphLocus {
            slices: vec![
                slice("a", (0, 10), (4, 10), Strand::Forward),
                slice("a", (10, 20), (0, 3), Strand::Forward),
                slice("b", (5, 15), (0, 2), Strand::Forward),
            ],
        };
        let whole = GraphLocus {
            slices: vec![
                slice("a", (0, 20), (4, 13), Strand::Forward),
                slice("b", (0, 15), (5, 7), Strand::Forward),
            ],
        };

        assert_eq!(
            carved.canonical(),
            whole.canonical(),
            "carving should not change the canonical locus"
        );
        assert_eq!(
            carved.canonical().slices,
            vec![
                slice("a", (4, 13), (0, 9), Strand::Forward),
                slice("b", (5, 7), (0, 2), Strand::Forward),
            ],
            "canonical slices should be full-width node-absolute ranges"
        );
    }

    #[test]
    fn test_canonical_merges_reverse_slices_listed_in_reading_order() {
        let reading_order = GraphLocus {
            slices: vec![
                slice("a", (10, 20), (0, 10), Strand::Reverse),
                slice("a", (0, 10), (5, 10), Strand::Reverse),
            ],
        };
        assert_eq!(
            reading_order.canonical().slices,
            vec![slice("a", (5, 20), (0, 15), Strand::Reverse)],
            "abutting reverse-strand slices should merge regardless of listing order"
        );
    }

    #[test]
    fn test_reverse_complement_reverses_order_and_strand() {
        let locus = GraphLocus {
            slices: vec![
                slice("a", (0, 10), (2, 10), Strand::Forward),
                slice("b", (0, 10), (0, 4), Strand::Forward),
            ],
        };
        let reversed = locus.reverse_complement();
        assert_eq!(
            reversed.slices,
            vec![
                slice("b", (0, 10), (0, 4), Strand::Reverse),
                slice("a", (0, 10), (2, 10), Strand::Reverse),
            ],
            "reverse complement should flip slice order and strands"
        );
        assert_eq!(
            reversed.reverse_complement(),
            locus,
            "reverse complement should be an involution"
        );
    }

    #[test]
    fn test_canonical_preserves_reverse_reading_order_and_skips_empty_slices() {
        let locus = GraphLocus {
            slices: vec![
                slice("a", (0, 10), (0, 0), Strand::Forward),
                slice("a", (0, 10), (0, 5), Strand::Reverse),
                slice("a", (0, 10), (5, 10), Strand::Reverse),
            ],
        };
        assert_eq!(
            locus.canonical().slices.len(),
            2,
            "reverse slices visited in ascending coordinate order must not merge"
        );
        assert_eq!(locus.canonical().length(), 10);
    }

    #[test]
    fn test_slice_counts_offsets_in_reading_order() {
        let locus = GraphLocus {
            slices: vec![
                slice("a", (0, 10), (2, 10), Strand::Forward),
                slice("b", (0, 10), (0, 4), Strand::Forward),
            ],
        };
        assert_eq!(
            locus
                .slice(6, 10)
                .expect("should slice within bounds")
                .slices,
            vec![
                slice("a", (0, 10), (8, 10), Strand::Forward),
                slice("b", (0, 10), (0, 2), Strand::Forward),
            ],
            "a slice across a junction should cover both blocks"
        );
        assert!(locus.slice(8, 8).is_none(), "an empty slice should fail");
        assert!(
            locus.slice(3, 13).is_none(),
            "slices past the end should fail"
        );
    }

    #[test]
    fn test_slice_of_reverse_strand_reads_from_slice_end() {
        let locus = GraphLocus {
            slices: vec![slice("a", (0, 10), (2, 8), Strand::Reverse)],
        };
        assert_eq!(
            locus
                .slice(0, 2)
                .expect("should slice within bounds")
                .slices,
            vec![slice("a", (0, 10), (6, 8), Strand::Reverse)],
            "the first reverse-strand positions should come from the slice end"
        );
    }
}
