use std::cmp::{max, min};

use gen_core::{
    HashId, Strand,
    path::PathBlock,
    range::{OrderedMerge, Range, merge_ordered_items},
};
use gen_models::{
    accession::{Accession, AccessionNode},
    annotations::Annotation,
    db::GraphConnection,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationSegment {
    pub node_id: HashId,
    pub range: Range,
    pub strand: Strand,
}

impl OrderedMerge for AnnotationSegment {
    fn should_merge_with(&self, next: &Self) -> bool {
        self.node_id == next.node_id
            && self.strand == next.strand
            && next.range.start >= self.range.start
            && next.range.start <= self.range.end
    }

    fn merge_with(&mut self, next: &Self) {
        self.range.end = max(self.range.end, next.range.end);
    }
}

impl From<&AccessionNode> for AnnotationSegment {
    fn from(node: &AccessionNode) -> Self {
        AnnotationSegment {
            node_id: node.node_id,
            range: Range {
                start: node.sequence_start,
                end: node.sequence_end,
            },
            strand: node.strand,
        }
    }
}

/// Compute the annotation segments from accession nodes.
pub fn annotation_segments(
    conn: &GraphConnection,
    annotation: &Annotation,
    history_ref: Option<&str>,
) -> Vec<AnnotationSegment> {
    Accession::get_nodes_by_id(conn, &annotation.accession_id, history_ref)
        .iter()
        .map(AnnotationSegment::from)
        .collect()
}

pub fn project_annotation_segments(
    accession_segments: &[AnnotationSegment],
    path_blocks: &[PathBlock],
    preserve_part_boundaries: bool,
) -> Vec<AnnotationSegment> {
    let projected = accession_segments
        .iter()
        .flat_map(|segment| {
            merge_ordered_items(
                path_blocks
                    .iter()
                    .filter_map(|block| {
                        if block.node_id != segment.node_id {
                            return None;
                        }
                        let overlap_start = max(segment.range.start, block.sequence_start);
                        let overlap_end = min(segment.range.end, block.sequence_end);
                        if overlap_end <= overlap_start {
                            return None;
                        }

                        let (start, end) = if block.strand == Strand::Reverse {
                            (
                                block.path_start + (block.sequence_end - overlap_end),
                                block.path_start + (block.sequence_end - overlap_start),
                            )
                        } else {
                            (
                                block.path_start + (overlap_start - block.sequence_start),
                                block.path_start + (overlap_end - block.sequence_start),
                            )
                        };
                        let strand = if block.strand == Strand::Reverse {
                            segment.strand.complement()
                        } else {
                            segment.strand
                        };

                        Some(AnnotationSegment {
                            node_id: block.node_id,
                            range: Range { start, end },
                            strand,
                        })
                    })
                    .collect(),
            )
        })
        .collect::<Vec<_>>();

    if preserve_part_boundaries {
        projected
    } else {
        merge_ordered_items(projected)
    }
}

#[cfg(test)]
mod tests {
    use gen_core::{HashId, path::PathBlock};

    use super::*;

    fn block(
        node_id: &str,
        sequence_start: i64,
        sequence_end: i64,
        path_start: i64,
        strand: Strand,
    ) -> PathBlock {
        PathBlock {
            node_id: HashId::convert_str(node_id),
            block_sequence: String::new(),
            sequence_start,
            sequence_end,
            path_start,
            path_end: path_start + (sequence_end - sequence_start),
            strand,
        }
    }

    #[test]
    fn test_merges_overlapping_ranges() {
        let node_id = HashId::convert_str("node");
        let merged = merge_ordered_items(vec![
            AnnotationSegment {
                node_id,
                range: Range { start: 2, end: 5 },
                strand: Strand::Forward,
            },
            AnnotationSegment {
                node_id,
                range: Range { start: 4, end: 8 },
                strand: Strand::Forward,
            },
        ]);

        assert_eq!(
            merged,
            vec![AnnotationSegment {
                node_id,
                range: Range { start: 2, end: 8 },
                strand: Strand::Forward,
            }]
        );
    }

    #[test]
    fn test_project_annotation_segments_complements_reverse_blocks() {
        let node_id = HashId::convert_str("node");
        let projected = project_annotation_segments(
            &[AnnotationSegment {
                node_id,
                range: Range { start: 10, end: 20 },
                strand: Strand::Forward,
            }],
            &[block("node", 0, 30, 100, Strand::Reverse)],
            false,
        );

        assert_eq!(
            projected,
            vec![AnnotationSegment {
                node_id,
                range: Range {
                    start: 110,
                    end: 120,
                },
                strand: Strand::Reverse,
            }]
        );
    }
}
