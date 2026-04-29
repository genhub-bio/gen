use std::{
    fmt,
    str::{self, FromStr},
};

use gb_io::seq::{Feature, Location, Seq};
use gen_core::Strand;
use gen_models::{
    annotations::{
        AnnotationError, AnnotationExtra, GenBankExtra, GenBankLocationOperator, GenBankQualifier,
    },
    errors::{OperationError, SequenceError},
};
use regex::{Error as RegexError, Regex};
use thiserror::Error;

use crate::normalize_string;

#[derive(Debug, Error, PartialEq)]
pub enum GenBankError {
    #[error("Feature Location Error: {0}")]
    LocationError(&'static str),
    #[error("Parse Error: {0}")]
    ParseError(String),
    #[error("Lookup Error: {0}")]
    LookupError(String),
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Annotation Error: {0}")]
    AnnotationError(#[from] AnnotationError),
    #[error("Database Error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("Regex Error: {0}")]
    Regex(#[from] RegexError),
    #[error("Sequence Error: {0}")]
    Sequence(#[from] SequenceError),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EditType {
    Deletion,
    Insertion,
    Replacement,
}

impl fmt::Display for EditType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EditType::Deletion => write!(f, "Deletion"),
            EditType::Insertion => write!(f, "Insertion"),
            EditType::Replacement => write!(f, "Replacement"),
        }
    }
}

impl FromStr for EditType {
    type Err = GenBankError;

    fn from_str(input: &str) -> Result<EditType, Self::Err> {
        match input {
            "Deletion" => Ok(EditType::Deletion),
            "Insertion" => Ok(EditType::Insertion),
            "Replacement" => Ok(EditType::Replacement),
            _ => Err(Self::Err::ParseError(
                format!("Unknown edit type: {input}").to_string(),
            )),
        }
    }
}

/// Represents how the wildtype sequence was changed.
///
/// The `start` and `end` coordinates refer to positions in the wildtype sequence, and have nothing
/// to do with the edit itself. `old_sequence` is the wildtype sequence that is replaced by `new_sequence`.
/// This is so there is an easy mapping to carry out an edit via old_sequence[start:end] = new_sequence
#[derive(Clone, Debug, PartialEq)]
pub struct GenBankEdit {
    pub start: i64,
    pub end: i64,
    pub old_sequence: String,
    pub new_sequence: String,
    pub edit_type: EditType,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenBankAnnotationSegment {
    pub start: i64,
    pub end: i64,
    pub strand: Strand,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenBankAnnotation {
    pub name: String,
    pub segments: Vec<GenBankAnnotationSegment>,
    pub extra: Option<AnnotationExtra>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenBankLocus {
    pub name: String,
    pub molecule_type: Option<String>,
    pub sequence: String,
    pub changes: Vec<GenBankEdit>,
    pub annotations: Vec<GenBankAnnotation>,
}

impl GenBankLocus {
    pub fn original_sequence(&self) -> String {
        let mut final_sequence = self.sequence.clone();
        let mut offset: i64 = 0;
        for edit in self.changes.iter() {
            let ustart = (edit.start + offset) as usize;
            let uend = (edit.end + offset) as usize;
            match edit.edit_type {
                EditType::Insertion => {
                    final_sequence =
                        format!("{}{}", &final_sequence[..ustart], &final_sequence[uend..]);
                }
                EditType::Deletion | EditType::Replacement => {
                    final_sequence = format!(
                        "{}{}{}",
                        &final_sequence[..ustart],
                        edit.old_sequence,
                        &final_sequence[uend..]
                    );
                }
            }
            offset += edit.old_sequence.len() as i64 - edit.new_sequence.len() as i64;
        }
        final_sequence
    }

    pub fn changes_to_wt(&self) -> Vec<GenBankEdit> {
        let mut wt_changes = vec![];
        let mut offset: i64 = 0;
        for edit in self.changes.iter() {
            let seq_diff = edit.old_sequence.len() as i64 - edit.new_sequence.len() as i64;
            wt_changes.push(GenBankEdit {
                start: edit.start + offset,
                end: edit.end + offset + seq_diff,
                old_sequence: edit.old_sequence.clone(),
                new_sequence: edit.new_sequence.clone(),
                edit_type: edit.edit_type,
            });
            offset += seq_diff;
        }
        wt_changes.sort_unstable_by(|a, b| Ord::cmp(&a.start, &b.start));
        wt_changes
    }
}

fn feature_qualifier_value(feature: &Feature, key: &str) -> Option<String> {
    feature
        .qualifiers
        .iter()
        .find_map(|(qualifier_key, value)| {
            let qualifier_name: &str = qualifier_key.as_ref();
            qualifier_name
                .eq_ignore_ascii_case(key)
                .then(|| value.as_deref().map(str::trim).map(str::to_string))
                .flatten()
        })
}

pub(crate) fn normalize_qualifier_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn annotation_segments_for_location_with_strand(
    location: &Location,
    strand: Strand,
) -> Vec<GenBankAnnotationSegment> {
    match location {
        Location::Range((start, _), (end, _)) => vec![GenBankAnnotationSegment {
            start: *start,
            end: *end,
            strand,
        }],
        Location::Between(start, end) => vec![GenBankAnnotationSegment {
            start: *start,
            end: end + 1,
            strand,
        }],
        Location::Complement(inner) => {
            annotation_segments_for_location_with_strand(inner, Strand::Reverse)
        }
        Location::Join(locations)
        | Location::Order(locations)
        | Location::Bond(locations)
        | Location::OneOf(locations) => locations
            .iter()
            .flat_map(|location| annotation_segments_for_location_with_strand(location, strand))
            .filter(|segment| segment.end > segment.start)
            .collect(),
        Location::External(_, maybe_location) => maybe_location
            .as_deref()
            .map(|location| annotation_segments_for_location_with_strand(location, strand))
            .unwrap_or_default(),
        Location::Gap(_) => vec![],
    }
}

fn annotation_segments_for_location(location: &Location) -> Vec<GenBankAnnotationSegment> {
    annotation_segments_for_location_with_strand(location, Strand::Forward)
}

fn genbank_location_operator(location: &Location) -> Option<GenBankLocationOperator> {
    match location {
        Location::Complement(inner) => genbank_location_operator(inner),
        Location::Join(_) => Some(GenBankLocationOperator::Join),
        Location::Order(_) => Some(GenBankLocationOperator::Order),
        Location::Bond(_) => Some(GenBankLocationOperator::Bond),
        Location::OneOf(_) => Some(GenBankLocationOperator::OneOf),
        Location::Range(..)
        | Location::Between(_, _)
        | Location::External(_, _)
        | Location::Gap(_) => None,
    }
}

fn genbank_extra_for_feature(feature: &Feature) -> AnnotationExtra {
    AnnotationExtra {
        genbank: Some(GenBankExtra {
            kind: feature.kind.as_ref().to_string(),
            qualifiers: feature
                .qualifiers
                .iter()
                .map(|(key, value)| GenBankQualifier {
                    key: key.as_ref().to_string(),
                    value: value.as_deref().map(normalize_qualifier_text),
                })
                .collect(),
            location_operator: genbank_location_operator(&feature.location),
        }),
        ..AnnotationExtra::default()
    }
}

fn annotation_for_feature(feature: &Feature) -> Option<GenBankAnnotation> {
    let segments = annotation_segments_for_location(&feature.location);
    if segments.is_empty() {
        return None;
    }

    // See https://www.insdc.org/submitting-standards/feature-table/#7.3.1 for feature qualifiers
    let name = feature_qualifier_value(feature, "label")
        .or_else(|| feature_qualifier_value(feature, "gene"))
        .or_else(|| feature_qualifier_value(feature, "protein_id"))
        .or_else(|| feature_qualifier_value(feature, "product"))
        .or_else(|| feature_qualifier_value(feature, "note"))
        .map(|value| normalize_qualifier_text(&value))
        .unwrap_or_else(|| feature.kind.as_ref().to_string());

    Some(GenBankAnnotation {
        name,
        segments,
        extra: Some(genbank_extra_for_feature(feature)),
    })
}

pub fn process_sequence(seq: Seq) -> Result<GenBankLocus, GenBankError> {
    let final_sequence = if let Ok(sequence) = str::from_utf8(&seq.seq) {
        sequence.to_string()
    } else {
        return Err(GenBankError::ParseError("No sequence present".to_string()));
    };

    let geneious_edit = Regex::new(r"Geneious type: Editing History (?P<edit_type>\w+)")?;
    let mut locus = GenBankLocus {
        name: seq.name.unwrap_or_default(),
        sequence: final_sequence.clone(),
        molecule_type: seq.molecule_type,
        changes: vec![],
        annotations: vec![],
    };

    for feature in seq.features.iter() {
        let edit_note = feature_qualifier_value(feature, "note")
            .map(|note| normalize_qualifier_text(&note))
            .and_then(|note| {
                geneious_edit
                    .captures(&note)
                    .map(|captures| captures["edit_type"].to_string())
            });
        if let Some(edit_type) = edit_note {
            let (mut start, mut end) = feature
                .location
                .find_bounds()
                .map_err(|_| GenBankError::LocationError("Ambiguous Bounds"))?;
            match edit_type.as_str() {
                "Insertion" => {
                    // If there is an insertion, it means that the WT is missing
                    // this sequence, so we actually treat it as a deletion
                    locus.changes.push(GenBankEdit {
                        start,
                        end,
                        old_sequence: "".to_string(),
                        new_sequence: final_sequence[start as usize..end as usize].to_string(),
                        edit_type: EditType::Insertion,
                    });
                }
                "Deletion" | "Replacement" => {
                    // If there is a deletion, it means that found sequence is missing
                    // this sequence, so we treat it as an insertion
                    let deleted_seq = normalize_string(
                        &feature_qualifier_value(feature, "Original_Bases")
                            .expect("Deleted sequence is not annotated."),
                    );
                    if matches!(feature.location, Location::Between(_, _)) {
                        start += 1;
                        end -= 1;
                    }
                    locus.changes.push(GenBankEdit {
                        start,
                        end,
                        old_sequence: deleted_seq,
                        new_sequence: final_sequence[start as usize..end as usize].to_string(),
                        edit_type: EditType::from_str(&edit_type)?,
                    });
                }
                t => {
                    println!("Unknown edit type {t}.")
                }
            }
            continue;
        }

        if let Some(annotation) = annotation_for_feature(feature) {
            locus.annotations.push(annotation);
        }
    }

    locus.changes.sort_unstable_by(|a, b| a.start.cmp(&b.start));
    Ok(locus)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gb_io::reader;
    use gen_core::Strand;
    use noodles::fasta;

    use super::*;

    fn get_unmodified_sequence() -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/geneious_genbank/unmodified.fa");
        let mut reader = fasta::io::reader::Builder.build_from_path(path).unwrap();
        let mut records = reader.records();
        let record = records.next().unwrap().unwrap();
        let seq = record.sequence();
        str::from_utf8(seq.as_ref()).unwrap().to_string()
    }

    #[test]
    fn test_restores_original_sequence() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/geneious_genbank/insertion.gb");
        let mut a = reader::parse_file(&path).unwrap();
        let seq = process_sequence(a.remove(0)).unwrap();
        assert_eq!(seq.original_sequence(), get_unmodified_sequence());
    }

    #[test]
    fn test_returns_changes_to_wt_sequence() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/geneious_genbank/multiple_insertions_deletions.gb");
        let mut a = reader::parse_file(&path).unwrap();
        let seq = process_sequence(a.remove(0)).unwrap();
        let changes = seq.changes_to_wt();
        assert_eq!(changes, vec![
            GenBankEdit {
                start: 119,
                end: 237,
                old_sequence: "TGCGTAAGGAGAAAATACCGCATCAGGCGCCATTCGCCATTCAGGCTGCGCAACTGTTGGGAAGGGCGATCGGTGCGGGCCTCTTCGCTATTACGCCAGCTGGCGAAAGGGGGATGTG".to_string(),
                new_sequence: "aact".to_string(),
                edit_type: EditType::Replacement
            }, GenBankEdit {
                start: 1425,
                end: 1425,
                old_sequence: "".to_string(),
                new_sequence: "tcagaagaactcgtcaagaaggcgatagaaggcgatgcgctgcgaatcgggagcggcgataccgtaaagcacgaggaagcggtcagcccattcgccgccaagctcttcagcaatatcacgggtagccaacgctatgtcctgatagcggtccgccacacccagccggccacagtcgatgaatccagaaaagcggccattttccaccatgatattcggcaagcaggcatcgccatgggtcacgacgagatcctcgccgtcgggcatgcgcgccttgagcctggcgaacagttcggctggcgcgagcccctgatgctcttcgtccagatcatcctgatcgacaagaccggcttccatccgagtacgtgctcgctcgatgcgatgtttcgcttggtggtcgaatgggcaggtagccggatcaagcgtatgcagccgccgcattgcatcagccatgatggatactttctcggcaggagcaaggtgagatgacaggagatcctgccccggcacttcgcccaatagcagccagtcccttcccgcttcagtgacaacgtcgagcacagctgcgcaaggaacgcccgtcgtggccagccacgatagccgcgctgcctcgtcctgcagttcattcagggcaccggacaggtcggtcttgacaaaaagaaccgggcgcccctgcgctgacagccggaacacggcggcatcagagcagccgattgtctgttgtgcccagtcatagccgaatagcctctccacccaagcggccggagaacctgcgtgcaatccatcttgttcaatcat".to_string(),
                edit_type: EditType::Insertion
            }, GenBankEdit {
                start: 3878,
                end: 4319,
                old_sequence: "TTCTTTGCTTCCTCGCCAGTTCGCTCGCTATGCTCGGTTACACGGCTGCGGCGAGCGCTAGTGATAATAAGTGACTGAGGTATGTGCTCTTCTTATCTCCTTTTGTAGTGTTGCTCTTATTTTAAACAACTTTGCGGTTTTTTGATGACTTTGCGATTTTGTTGTTGCTTTGCAGTAAATTGCAAGATTTAATAAAAAAACGCAAAGCAATGATTAAAGGATGTTCAGAATGAAACTCATGGAAACACTTAACCAGTGCATAAACGCTGGTCATGAAATGACGAAGGCTATCGCCATTGCACAGTTTAATGATGACAGCCCGGAAGCGAGGAAAATAACCCGGCGCTGGAGAATAGGTGAAGCAGCGGATTTAGTTGGGGTTTCTTCTCAGGCTATCAGAGATGCCGAGAAAGCAGGGCGACTACCGCACCCGGATATGGA".to_string(),
                new_sequence: "".to_string(),
                edit_type: EditType::Deletion
            }, GenBankEdit {
                start: 5750,
                end: 5908,
                old_sequence: "GCTTATGAACGTGGTCAGCGTTATGCAAGCCGATTGCAGAATGAATTTGCTGGAAATATTTCTGCGCTGGCTGATGCGGAAAATATTTCACGTAAGATTATTACCCGCTGTATCAACACCGCCAAATTGCCTAAATCAGTTGTTGCTCTTTTTTCTCA".to_string(),
                new_sequence: "aaattt".to_string(),
                edit_type: EditType::Replacement
            }, GenBankEdit {
                start: 5909,
                end: 5909,
                old_sequence: "".to_string(),
                new_sequence: "ccggg".to_string(),
                edit_type: EditType::Insertion
            }]);

        // apply all these changes to the WT sequence and ensure we get the final sequence out
        let mut wt_sequence = seq.original_sequence();
        assert_ne!(wt_sequence, seq.sequence);
        for change in changes.iter().rev() {
            wt_sequence = format!(
                "{}{}{}",
                &wt_sequence[..change.start as usize],
                change.new_sequence,
                &wt_sequence[change.end as usize..]
            );
        }
        assert_eq!(wt_sequence, seq.sequence);
    }

    #[test]
    fn test_preserves_reverse_strand_annotations() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/puc19.gb");
        let mut records = reader::parse_file(&path).unwrap();
        let seq = process_sequence(records.remove(0)).unwrap();

        let annotation = seq
            .annotations
            .iter()
            .find(|annotation| annotation.name == "M13 Forward")
            .unwrap();
        assert_eq!(
            annotation.segments,
            vec![GenBankAnnotationSegment {
                start: 688,
                end: 706,
                strand: Strand::Reverse,
            }]
        );
    }

    #[test]
    fn test_preserves_origin_spanning_annotations() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/puc19.gb");
        let mut records = reader::parse_file(&path).unwrap();
        let seq = process_sequence(records.remove(0)).unwrap();

        let annotation = seq
            .annotations
            .iter()
            .find(|annotation| annotation.name == "ori")
            .unwrap();
        assert_eq!(
            annotation.segments,
            vec![
                GenBankAnnotationSegment {
                    start: 2314,
                    end: 2686,
                    strand: Strand::Forward,
                },
                GenBankAnnotationSegment {
                    start: 0,
                    end: 217,
                    strand: Strand::Forward,
                },
            ]
        );
    }
}
