pub mod bed;
pub mod gff;

use gen_core::Strand;

// This is a helper to give an effective strand. For example, if we have a feature with a negative
// strand on a graph that is a negative traversal, it's just the positive strand.
pub(crate) fn compose_strand(feature_strand: Strand, traversal_strand: Strand) -> Strand {
    match (feature_strand, traversal_strand) {
        (Strand::ImportantButUnknown, _) | (_, Strand::ImportantButUnknown) => {
            Strand::ImportantButUnknown
        }
        (Strand::Forward, Strand::Forward) | (Strand::Reverse, Strand::Reverse) => Strand::Forward,
        (Strand::Forward, Strand::Reverse) | (Strand::Reverse, Strand::Forward) => Strand::Reverse,
        _ => Strand::Unknown,
    }
}
