pub mod bed;
pub mod gff;

use std::ops::Range;

use gen_core::Strand;
use gen_graph::GraphNode;

pub(super) fn project_overlap_to_node(
    node: GraphNode,
    path_interval: &Range<i64>,
    overlap: &Range<i64>,
    traversal_strand: Strand,
) -> Range<i64> {
    let local_start = overlap.start - path_interval.start;
    let local_end = overlap.end - path_interval.start;

    if traversal_strand == Strand::Reverse {
        node.sequence_end - local_end..node.sequence_end - local_start
    } else {
        node.sequence_start + local_start..node.sequence_start + local_end
    }
}

pub(super) fn compose_strand(feature_strand: Strand, traversal_strand: Strand) -> Strand {
    match (feature_strand, traversal_strand) {
        (Strand::ImportantButUnknown, _) | (_, Strand::ImportantButUnknown) => {
            Strand::ImportantButUnknown
        }
        (Strand::Forward, Strand::Forward) | (Strand::Reverse, Strand::Reverse) => Strand::Forward,
        (Strand::Forward, Strand::Reverse) | (Strand::Reverse, Strand::Forward) => Strand::Reverse,
        _ => Strand::Unknown,
    }
}
