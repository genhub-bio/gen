pub mod bed;
pub mod gff;

use gen_core::Strand;

// This helper gives the effective strand of a feature on an oriented path. For example, a
// negative feature on a negative traversal is forward in stored-node coordinates.
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
