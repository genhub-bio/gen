use std::{
    fmt,
    str::{self, FromStr},
};

use gb_io::seq::{Location, Seq};
use gen_models::errors::OperationError;
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
    #[error("Regex Error: {0}")]
    Regex(#[from] RegexError),
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

#[derive(Clone, Debug, PartialEq)]
pub struct GenBankEdit {
    pub start: i64,
    pub end: i64,
    pub old_sequence: String,
    pub new_sequence: String,
    pub edit_type: EditType,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenBankLocus {
    pub name: String,
    pub molecule_type: Option<String>,
    pub sequence: String,
    pub changes: Vec<GenBankEdit>,
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
    };

    for feature in seq.features.iter() {
        for (key, value) in feature.qualifiers.iter() {
            if key == "note" {
                if let Some(v) = value {
                    let geneious_mod = geneious_edit.captures(v);
                    if let Some(edit) = geneious_mod {
                        let (mut start, mut end) = feature
                            .location
                            .find_bounds()
                            .map_err(|_| GenBankError::LocationError("Ambiguous Bounds"))?;
                        match &edit["edit_type"] {
                            "Insertion" => {
                                // If there is an insertion, it means that the WT is missing
                                // this sequence, so we actually treat it as a deletion
                                locus.changes.push(GenBankEdit {
                                    start,
                                    end,
                                    old_sequence: "".to_string(),
                                    new_sequence: final_sequence[start as usize..end as usize]
                                        .to_string(),
                                    edit_type: EditType::Insertion,
                                });
                            }
                            "Deletion" | "Replacement" => {
                                // If there is a deletion, it means that found sequence is missing
                                // this sequence, so we treat it as an insertion
                                let deleted_seq = normalize_string(
                                    &feature
                                        .qualifiers
                                        .iter()
                                        .filter(|(k, _v)| k == "Original_Bases")
                                        .map(|(_k, v)| v.clone())
                                        .collect::<Option<String>>()
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
                                    new_sequence: final_sequence[start as usize..end as usize]
                                        .to_string(),
                                    edit_type: EditType::from_str(&edit["edit_type"])?,
                                });
                            }
                            t => {
                                println!("Unknown edit type {t}.")
                            }
                        }
                    }
                }
            }
        }
    }

    locus.changes.sort_unstable_by(|a, b| a.start.cmp(&b.start));
    Ok(locus)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gb_io::reader;
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
}
