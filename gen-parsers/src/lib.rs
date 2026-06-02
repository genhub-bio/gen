use std::{collections::HashMap, io::BufRead};

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    #[error("failed to read alignment input: {0}")]
    Read(String),
    #[error("CLUSTAL alignment must start with CLUSTAL W or CLUSTALW")]
    MissingHeader,
    #[error("CLUSTAL alignment does not contain sequence rows")]
    NoSequences,
    #[error("alignment row {line_number} is missing a sequence fragment")]
    MissingSequenceFragment { line_number: usize },
    #[error("BLAST HSP is missing a Sbjct sequence row")]
    MissingBlastSubject,
    #[error("BLAST output does not contain pairwise HSP alignments")]
    NoBlastAlignments,
    #[error("PAF line {line_number} has {actual} fields but expected at least 12")]
    PafTooFewFields { line_number: usize, actual: usize },
    #[error("PAF line {line_number} is missing required cg:Z CIGAR tag")]
    MissingPafCigar { line_number: usize },
    #[error("PAF line {line_number} has invalid integer in field {field}")]
    InvalidPafInteger { line_number: usize, field: usize },
    #[error("PAF line {line_number} has invalid strand {strand}")]
    InvalidPafStrand { line_number: usize, strand: String },
    #[error("PAF line {line_number} has invalid CIGAR string {cigar}")]
    InvalidCigar { line_number: usize, cigar: String },
    #[error("PSL line {line_number} has {actual} fields but expected at least 21")]
    PslTooFewFields { line_number: usize, actual: usize },
    #[error("PSL line {line_number} has invalid integer in field {field}")]
    InvalidPslInteger { line_number: usize, field: usize },
    #[error("PSL line {line_number} has invalid strand {strand}")]
    InvalidPslStrand { line_number: usize, strand: String },
    #[error(
        "PSL line {line_number} blockCount is {block_count} but blockSizes, qStarts, and tStarts contain {block_sizes}, {query_starts}, and {target_starts} entries"
    )]
    MismatchedPslBlocks {
        line_number: usize,
        block_count: usize,
        block_sizes: usize,
        query_starts: usize,
        target_starts: usize,
    },
    #[error("MAF block ending at line {line_number} does not contain sequence rows")]
    MafBlockWithoutSequences { line_number: usize },
    #[error("MAF line {line_number} has {actual} fields but expected at least 7")]
    MafTooFewFields { line_number: usize, actual: usize },
    #[error("MAF line {line_number} has invalid integer in field {field}")]
    InvalidMafInteger { line_number: usize, field: usize },
    #[error("MAF line {line_number} has invalid strand {strand}")]
    InvalidMafStrand { line_number: usize, strand: String },
    #[error(
        "MAF line {line_number} sequence {name} declares size {declared} but has {actual} non-gap bases"
    )]
    MafSizeMismatch {
        line_number: usize,
        name: String,
        declared: i64,
        actual: usize,
    },
    #[error("aligned sequence {name} has length {actual} but expected {expected}")]
    MismatchedAlignedLength {
        name: String,
        actual: usize,
        expected: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CigarOp {
    Match(i64),
    Insertion(i64),
    Deletion(i64),
    ReferenceSkip(i64),
    SoftClip(i64),
    HardClip(i64),
    Equal(i64),
    Difference(i64),
}

#[derive(Debug)]
pub struct ParsedAlignment {
    pub base_name: String,
    pub sequence_order: Vec<String>,
    pub aligned_sequences: HashMap<String, String>,
    pub graph: GenGraph,
}

impl ParsedAlignment {
    pub fn ungapped_sequence(&self, name: &str) -> Option<String> {
        self.aligned_sequences
            .get(name)
            .map(|sequence| sequence.chars().filter(|base| *base != '-').collect())
    }
}

#[derive(Debug)]
pub struct ParsedMapping {
    pub query_name: String,
    pub query_len: i64,
    pub query_start: i64,
    pub query_end: i64,
    pub strand: Strand,
    pub target_name: String,
    pub target_len: i64,
    pub target_start: i64,
    pub target_end: i64,
    pub matching_bases: i64,
    pub block_len: i64,
    pub mapping_quality: i64,
    pub cigar: Vec<CigarOp>,
    pub tags: HashMap<String, String>,
    pub graph: GenGraph,
}

pub struct ClustalwParser<R> {
    reader: Option<R>,
}

impl<R> ClustalwParser<R> {
    pub const fn new(reader: R) -> Self {
        Self {
            reader: Some(reader),
        }
    }
}

pub struct MafParser<R> {
    alignments: std::vec::IntoIter<Result<ParsedAlignment, ParseError>>,
    _reader: std::marker::PhantomData<R>,
}

impl<R> MafParser<R>
where
    R: BufRead,
{
    pub fn new(reader: R) -> Self {
        let alignments = parse_maf_reader(reader);
        Self {
            alignments: alignments.into_iter(),
            _reader: std::marker::PhantomData,
        }
    }
}

impl<R> Iterator for MafParser<R> {
    type Item = Result<ParsedAlignment, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.alignments.next()
    }
}

pub struct PafParser<R> {
    mappings: std::vec::IntoIter<Result<ParsedMapping, ParseError>>,
    _reader: std::marker::PhantomData<R>,
}

impl<R> PafParser<R>
where
    R: BufRead,
{
    pub fn new(reader: R) -> Self {
        let mappings = parse_paf_reader(reader);
        Self {
            mappings: mappings.into_iter(),
            _reader: std::marker::PhantomData,
        }
    }
}

impl<R> Iterator for PafParser<R> {
    type Item = Result<ParsedMapping, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.mappings.next()
    }
}

pub struct PslParser<R> {
    mappings: std::vec::IntoIter<Result<ParsedMapping, ParseError>>,
    _reader: std::marker::PhantomData<R>,
}

impl<R> PslParser<R>
where
    R: BufRead,
{
    pub fn new(reader: R) -> Self {
        let mappings = parse_psl_reader(reader);
        Self {
            mappings: mappings.into_iter(),
            _reader: std::marker::PhantomData,
        }
    }
}

impl<R> Iterator for PslParser<R> {
    type Item = Result<ParsedMapping, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.mappings.next()
    }
}

pub struct BlastParser<R> {
    alignments: std::vec::IntoIter<Result<ParsedAlignment, ParseError>>,
    _reader: std::marker::PhantomData<R>,
}

impl<R> BlastParser<R>
where
    R: BufRead,
{
    pub fn new(reader: R) -> Self {
        let alignments = parse_blast_reader(reader);
        Self {
            alignments: alignments.into_iter(),
            _reader: std::marker::PhantomData,
        }
    }
}

impl<R> Iterator for BlastParser<R> {
    type Item = Result<ParsedAlignment, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.alignments.next()
    }
}

impl<R> Iterator for ClustalwParser<R>
where
    R: BufRead,
{
    type Item = Result<ParsedAlignment, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.take().map(parse_reader)
    }
}

fn parse_reader<R>(reader: R) -> Result<ParsedAlignment, ParseError>
where
    R: BufRead,
{
    let mut lines = reader.lines();
    let Some(header) = lines.next() else {
        return Err(ParseError::MissingHeader);
    };
    let header = header.map_err(|error| ParseError::Read(error.to_string()))?;
    if !is_clustal_header(&header) {
        return Err(ParseError::MissingHeader);
    }

    let mut sequence_order = Vec::new();
    let mut aligned_sequences = HashMap::new();
    let mut block = Vec::new();

    for (line_index, line) in lines.enumerate() {
        let line_number = line_index + 2;
        let line = line.map_err(|error| ParseError::Read(error.to_string()))?;
        if line.trim().is_empty() {
            flush_block(&mut block, &mut aligned_sequences)?;
            continue;
        }
        if is_conservation_line(&line) {
            continue;
        }

        let mut fields = line.split_whitespace();
        let name = fields
            .next()
            .ok_or(ParseError::MissingSequenceFragment { line_number })?;
        let fragment = fields
            .next()
            .ok_or(ParseError::MissingSequenceFragment { line_number })?;

        let name = name.to_string();
        if !aligned_sequences.contains_key(&name) {
            sequence_order.push(name.clone());
        }

        let count = fields.next().and_then(|field| field.parse::<usize>().ok());
        block.push(BlockRow {
            name: name.clone(),
            fragment: fragment.to_ascii_uppercase(),
            count,
        });
    }
    flush_block(&mut block, &mut aligned_sequences)?;

    if sequence_order.is_empty() {
        return Err(ParseError::NoSequences);
    }

    validate_lengths(&sequence_order, &aligned_sequences)?;
    let base_name = sequence_order[0].clone();
    let graph = build_graph(&base_name, &sequence_order, &aligned_sequences);

    Ok(ParsedAlignment {
        base_name,
        sequence_order,
        aligned_sequences,
        graph,
    })
}

fn parse_blast_reader<R>(reader: R) -> Vec<Result<ParsedAlignment, ParseError>>
where
    R: BufRead,
{
    let lines = match reader.lines().collect::<Result<Vec<_>, _>>() {
        Ok(lines) => lines,
        Err(error) => return vec![Err(ParseError::Read(error.to_string()))],
    };

    let mut query_name = String::from("Query");
    let mut subject_name = String::from("Sbjct");
    let mut query_parts = Vec::new();
    let mut subject_parts = Vec::new();
    let mut results = Vec::new();
    let mut saw_score = false;

    for line in lines {
        let trimmed = line.trim();
        if let Some(name) = trimmed.strip_prefix("Query=") {
            query_name = name.trim().to_string();
            continue;
        }
        if let Some(name) = trimmed.strip_prefix('>') {
            flush_blast_hsp(
                &mut results,
                &query_name,
                &subject_name,
                &mut query_parts,
                &mut subject_parts,
            );
            subject_name = name
                .split_whitespace()
                .next()
                .unwrap_or("Sbjct")
                .to_string();
            saw_score = false;
            continue;
        }
        if trimmed.starts_with("Score =") {
            flush_blast_hsp(
                &mut results,
                &query_name,
                &subject_name,
                &mut query_parts,
                &mut subject_parts,
            );
            saw_score = true;
            continue;
        }
        if let Some(fragment) = blast_alignment_fragment(trimmed, "Query") {
            query_parts.push(fragment.to_ascii_uppercase());
            continue;
        }
        if let Some(fragment) = blast_alignment_fragment(trimmed, "Sbjct") {
            subject_parts.push(fragment.to_ascii_uppercase());
        }
    }

    flush_blast_hsp(
        &mut results,
        &query_name,
        &subject_name,
        &mut query_parts,
        &mut subject_parts,
    );

    if results.is_empty() && saw_score {
        results.push(Err(ParseError::MissingBlastSubject));
    } else if results.is_empty() {
        results.push(Err(ParseError::NoBlastAlignments));
    }

    results
}

fn flush_blast_hsp(
    results: &mut Vec<Result<ParsedAlignment, ParseError>>,
    query_name: &str,
    subject_name: &str,
    query_parts: &mut Vec<String>,
    subject_parts: &mut Vec<String>,
) {
    if query_parts.is_empty() && subject_parts.is_empty() {
        return;
    }
    if subject_parts.is_empty() {
        results.push(Err(ParseError::MissingBlastSubject));
        query_parts.clear();
        return;
    }

    let query_sequence = query_parts.join("");
    let subject_sequence = subject_parts.join("");
    query_parts.clear();
    subject_parts.clear();

    let sequence_order = vec![query_name.to_string(), subject_name.to_string()];
    let mut aligned_sequences = HashMap::new();
    aligned_sequences.insert(query_name.to_string(), query_sequence);
    aligned_sequences.insert(subject_name.to_string(), subject_sequence);

    let result = validate_lengths(&sequence_order, &aligned_sequences).map(|()| ParsedAlignment {
        base_name: query_name.to_string(),
        sequence_order: sequence_order.clone(),
        graph: build_graph(query_name, &sequence_order, &aligned_sequences),
        aligned_sequences,
    });
    results.push(result);
}

fn blast_alignment_fragment<'a>(line: &'a str, label: &str) -> Option<&'a str> {
    let mut fields = line.split_whitespace();
    if fields.next()? != label {
        return None;
    }
    let _start = fields.next()?;
    fields.next()
}

fn parse_maf_reader<R>(reader: R) -> Vec<Result<ParsedAlignment, ParseError>>
where
    R: BufRead,
{
    let mut results = Vec::new();
    let mut rows = Vec::new();
    let mut in_alignment = false;

    for (line_index, line) in reader.lines().enumerate() {
        let line_number = line_index + 1;
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                results.push(Err(ParseError::Read(error.to_string())));
                continue;
            }
        };
        let trimmed = line.trim();

        if trimmed.is_empty() {
            flush_maf_block(&mut results, &mut rows, line_number, &mut in_alignment);
            continue;
        }
        if should_skip_maf_line(trimmed) {
            continue;
        }
        if trimmed.starts_with("a ") || trimmed == "a" {
            flush_maf_block(&mut results, &mut rows, line_number, &mut in_alignment);
            in_alignment = true;
            continue;
        }
        if trimmed.starts_with("s ") {
            match parse_maf_sequence_row(trimmed, line_number) {
                Ok(row) => rows.push(row),
                Err(error) => results.push(Err(error)),
            }
        }
    }

    flush_maf_block(&mut results, &mut rows, 0, &mut in_alignment);
    results
}

fn should_skip_maf_line(line: &str) -> bool {
    line.starts_with('#')
        || line.starts_with("track ")
        || line.starts_with("i ")
        || line.starts_with("e ")
        || line.starts_with("q ")
}

fn flush_maf_block(
    results: &mut Vec<Result<ParsedAlignment, ParseError>>,
    rows: &mut Vec<MafSequenceRow>,
    line_number: usize,
    in_alignment: &mut bool,
) {
    if !*in_alignment {
        return;
    }
    if rows.is_empty() {
        results.push(Err(ParseError::MafBlockWithoutSequences { line_number }));
        *in_alignment = false;
        return;
    }

    let result = build_maf_alignment(rows);
    results.push(result);
    rows.clear();
    *in_alignment = false;
}

fn parse_maf_sequence_row(line: &str, line_number: usize) -> Result<MafSequenceRow, ParseError> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 7 {
        return Err(ParseError::MafTooFewFields {
            line_number,
            actual: fields.len(),
        });
    }

    let name = fields[1].to_string();
    let _start = parse_maf_i64(fields[2], line_number, 3)?;
    let size = parse_maf_i64(fields[3], line_number, 4)?;
    let _strand = match fields[4] {
        "+" => Strand::Forward,
        "-" => Strand::Reverse,
        strand => {
            return Err(ParseError::InvalidMafStrand {
                line_number,
                strand: strand.to_string(),
            });
        }
    };
    let _source_size = parse_maf_i64(fields[5], line_number, 6)?;
    let text = fields[6].to_ascii_uppercase();
    let actual = text.chars().filter(|character| *character != '-').count();
    if actual != size as usize {
        return Err(ParseError::MafSizeMismatch {
            line_number,
            name,
            declared: size,
            actual,
        });
    }

    Ok(MafSequenceRow { name, text })
}

fn parse_maf_i64(field: &str, line_number: usize, field_number: usize) -> Result<i64, ParseError> {
    field.parse().map_err(|_| ParseError::InvalidMafInteger {
        line_number,
        field: field_number,
    })
}

fn build_maf_alignment(rows: &[MafSequenceRow]) -> Result<ParsedAlignment, ParseError> {
    let base = rows.first().expect("should have MAF sequence row");
    let mut sequence_order = Vec::new();
    let mut aligned_sequences = HashMap::new();

    for row in rows {
        sequence_order.push(row.name.clone());
        let text = if row.name == base.name {
            row.text.clone()
        } else {
            expand_maf_dots(&base.text, &row.text)
        };
        aligned_sequences.insert(row.name.clone(), text);
    }

    validate_lengths(&sequence_order, &aligned_sequences)?;
    let graph = build_graph(&base.name, &sequence_order, &aligned_sequences);
    Ok(ParsedAlignment {
        base_name: base.name.clone(),
        sequence_order,
        graph,
        aligned_sequences,
    })
}

fn expand_maf_dots(base: &str, text: &str) -> String {
    base.chars()
        .zip(text.chars())
        .map(|(base_character, character)| {
            if character == '.' {
                base_character
            } else {
                character
            }
        })
        .collect()
}

fn parse_paf_reader<R>(reader: R) -> Vec<Result<ParsedMapping, ParseError>>
where
    R: BufRead,
{
    reader
        .lines()
        .enumerate()
        .filter_map(|(line_index, line)| {
            let line_number = line_index + 1;
            let line = match line {
                Ok(line) => line,
                Err(error) => return Some(Err(ParseError::Read(error.to_string()))),
            };
            if line.trim().is_empty() {
                return None;
            }
            Some(parse_paf_line(&line, line_number))
        })
        .collect()
}

fn parse_paf_line(line: &str, line_number: usize) -> Result<ParsedMapping, ParseError> {
    let fields = line.split('\t').collect::<Vec<_>>();
    if fields.len() < 12 {
        return Err(ParseError::PafTooFewFields {
            line_number,
            actual: fields.len(),
        });
    }

    let query_name = fields[0].to_string();
    let query_len = parse_paf_i64(fields[1], line_number, 2)?;
    let query_start = parse_paf_i64(fields[2], line_number, 3)?;
    let query_end = parse_paf_i64(fields[3], line_number, 4)?;
    let strand = match fields[4] {
        "+" => Strand::Forward,
        "-" => Strand::Reverse,
        strand => {
            return Err(ParseError::InvalidPafStrand {
                line_number,
                strand: strand.to_string(),
            });
        }
    };
    let target_name = fields[5].to_string();
    let target_len = parse_paf_i64(fields[6], line_number, 7)?;
    let target_start = parse_paf_i64(fields[7], line_number, 8)?;
    let target_end = parse_paf_i64(fields[8], line_number, 9)?;
    let matching_bases = parse_paf_i64(fields[9], line_number, 10)?;
    let block_len = parse_paf_i64(fields[10], line_number, 11)?;
    let mapping_quality = parse_paf_i64(fields[11], line_number, 12)?;

    let mut tags = HashMap::new();
    let mut cigar = None;
    for field in fields.iter().skip(12) {
        let mut parts = field.splitn(3, ':');
        let Some(key) = parts.next() else {
            continue;
        };
        let Some(tag_type) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next() else {
            continue;
        };
        tags.insert(key.to_string(), format!("{tag_type}:{value}"));
        if key == "cg" && tag_type == "Z" {
            cigar = Some(parse_cigar(value, line_number)?);
        }
    }

    let Some(cigar) = cigar else {
        return Err(ParseError::MissingPafCigar { line_number });
    };

    let mut mapping = ParsedMapping {
        query_name,
        query_len,
        query_start,
        query_end,
        strand,
        target_name,
        target_len,
        target_start,
        target_end,
        matching_bases,
        block_len,
        mapping_quality,
        cigar,
        tags,
        graph: GenGraph::new(),
    };
    mapping.graph = build_paf_graph(&mapping);
    Ok(mapping)
}

fn parse_paf_i64(field: &str, line_number: usize, field_number: usize) -> Result<i64, ParseError> {
    field.parse().map_err(|_| ParseError::InvalidPafInteger {
        line_number,
        field: field_number,
    })
}

fn parse_cigar(cigar: &str, line_number: usize) -> Result<Vec<CigarOp>, ParseError> {
    let mut ops = Vec::new();
    let mut len = 0_i64;
    let mut has_digits = false;

    for character in cigar.chars() {
        if let Some(digit) = character.to_digit(10) {
            has_digits = true;
            len = len * 10 + i64::from(digit);
            continue;
        }

        if !has_digits || len == 0 {
            return Err(ParseError::InvalidCigar {
                line_number,
                cigar: cigar.to_string(),
            });
        }

        let op = match character {
            'M' => CigarOp::Match(len),
            'I' => CigarOp::Insertion(len),
            'D' => CigarOp::Deletion(len),
            'N' => CigarOp::ReferenceSkip(len),
            'S' => CigarOp::SoftClip(len),
            'H' => CigarOp::HardClip(len),
            '=' => CigarOp::Equal(len),
            'X' => CigarOp::Difference(len),
            _ => {
                return Err(ParseError::InvalidCigar {
                    line_number,
                    cigar: cigar.to_string(),
                });
            }
        };
        ops.push(op);
        len = 0;
        has_digits = false;
    }

    if has_digits || ops.is_empty() {
        return Err(ParseError::InvalidCigar {
            line_number,
            cigar: cigar.to_string(),
        });
    }

    Ok(ops)
}

fn parse_psl_reader<R>(reader: R) -> Vec<Result<ParsedMapping, ParseError>>
where
    R: BufRead,
{
    reader
        .lines()
        .enumerate()
        .filter_map(|(line_index, line)| {
            let line_number = line_index + 1;
            let line = match line {
                Ok(line) => line,
                Err(error) => return Some(Err(ParseError::Read(error.to_string()))),
            };
            if should_skip_psl_line(&line) {
                return None;
            }
            Some(parse_psl_line(&line, line_number))
        })
        .collect()
}

fn should_skip_psl_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty()
        || trimmed.starts_with("psLayout")
        || trimmed.starts_with("match ")
        || trimmed.starts_with("-----")
        || trimmed.starts_with("browser ")
        || trimmed.starts_with("track ")
}

fn parse_psl_line(line: &str, line_number: usize) -> Result<ParsedMapping, ParseError> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 21 {
        return Err(ParseError::PslTooFewFields {
            line_number,
            actual: fields.len(),
        });
    }

    let matches = parse_psl_i64(fields[0], line_number, 1)?;
    let mismatches = parse_psl_i64(fields[1], line_number, 2)?;
    let rep_matches = parse_psl_i64(fields[2], line_number, 3)?;
    let n_count = parse_psl_i64(fields[3], line_number, 4)?;
    let query_insert_count = parse_psl_i64(fields[4], line_number, 5)?;
    let query_insert_bases = parse_psl_i64(fields[5], line_number, 6)?;
    let target_insert_count = parse_psl_i64(fields[6], line_number, 7)?;
    let target_insert_bases = parse_psl_i64(fields[7], line_number, 8)?;
    let strand = parse_psl_strand(fields[8], line_number)?;
    let query_name = fields[9].to_string();
    let query_len = parse_psl_i64(fields[10], line_number, 11)?;
    let query_start = parse_psl_i64(fields[11], line_number, 12)?;
    let query_end = parse_psl_i64(fields[12], line_number, 13)?;
    let target_name = fields[13].to_string();
    let target_len = parse_psl_i64(fields[14], line_number, 15)?;
    let target_start = parse_psl_i64(fields[15], line_number, 16)?;
    let target_end = parse_psl_i64(fields[16], line_number, 17)?;
    let block_count = parse_psl_usize(fields[17], line_number, 18)?;
    let block_sizes = parse_psl_i64_list(fields[18], line_number, 19)?;
    let query_starts = parse_psl_i64_list(fields[19], line_number, 20)?;
    let target_starts = parse_psl_i64_list(fields[20], line_number, 21)?;

    if block_sizes.len() != block_count
        || query_starts.len() != block_count
        || target_starts.len() != block_count
    {
        return Err(ParseError::MismatchedPslBlocks {
            line_number,
            block_count,
            block_sizes: block_sizes.len(),
            query_starts: query_starts.len(),
            target_starts: target_starts.len(),
        });
    }

    let blocks = block_sizes
        .iter()
        .zip(query_starts.iter())
        .zip(target_starts.iter())
        .map(|((size, query_start), target_start)| PslBlock {
            size: *size,
            query_start: normalize_psl_query_start(*query_start, *size, query_len, strand),
            target_start: *target_start,
        })
        .collect::<Vec<_>>();
    let mut tags = HashMap::new();
    tags.insert("psl:misMatches".to_string(), mismatches.to_string());
    tags.insert("psl:repMatches".to_string(), rep_matches.to_string());
    tags.insert("psl:nCount".to_string(), n_count.to_string());
    tags.insert("psl:qNumInsert".to_string(), query_insert_count.to_string());
    tags.insert(
        "psl:qBaseInsert".to_string(),
        query_insert_bases.to_string(),
    );
    tags.insert(
        "psl:tNumInsert".to_string(),
        target_insert_count.to_string(),
    );
    tags.insert(
        "psl:tBaseInsert".to_string(),
        target_insert_bases.to_string(),
    );

    let mut mapping = ParsedMapping {
        query_name,
        query_len,
        query_start,
        query_end,
        strand,
        target_name,
        target_len,
        target_start,
        target_end,
        matching_bases: matches,
        block_len: block_sizes.iter().sum(),
        mapping_quality: 0,
        cigar: Vec::new(),
        tags,
        graph: GenGraph::new(),
    };
    mapping.graph = build_psl_graph(&mapping, &blocks);
    Ok(mapping)
}

fn parse_psl_i64(field: &str, line_number: usize, field_number: usize) -> Result<i64, ParseError> {
    field.parse().map_err(|_| ParseError::InvalidPslInteger {
        line_number,
        field: field_number,
    })
}

fn parse_psl_usize(
    field: &str,
    line_number: usize,
    field_number: usize,
) -> Result<usize, ParseError> {
    field.parse().map_err(|_| ParseError::InvalidPslInteger {
        line_number,
        field: field_number,
    })
}

fn parse_psl_i64_list(
    field: &str,
    line_number: usize,
    field_number: usize,
) -> Result<Vec<i64>, ParseError> {
    field
        .trim_end_matches(',')
        .split(',')
        .filter(|value| !value.is_empty())
        .map(|value| parse_psl_i64(value, line_number, field_number))
        .collect()
}

fn parse_psl_strand(field: &str, line_number: usize) -> Result<Strand, ParseError> {
    match field.chars().next() {
        Some('+') => Ok(Strand::Forward),
        Some('-') => Ok(Strand::Reverse),
        _ => Err(ParseError::InvalidPslStrand {
            line_number,
            strand: field.to_string(),
        }),
    }
}

fn normalize_psl_query_start(query_start: i64, size: i64, query_len: i64, strand: Strand) -> i64 {
    if strand == Strand::Reverse {
        query_len - query_start - size
    } else {
        query_start
    }
}

fn flush_block(
    block: &mut Vec<BlockRow>,
    aligned_sequences: &mut HashMap<String, String>,
) -> Result<(), ParseError> {
    if block.is_empty() {
        return Ok(());
    }

    let expected = block
        .iter()
        .map(|row| row.fragment.len())
        .max()
        .expect("should have block rows");
    let has_counts = block.iter().any(|row| row.count.is_some());
    for row in block.drain(..) {
        let actual = row.fragment.len();
        if actual != expected && !has_counts {
            return Err(ParseError::MismatchedAlignedLength {
                name: row.name,
                actual,
                expected,
            });
        }

        let sequence = aligned_sequences.entry(row.name).or_default();
        sequence.push_str(&row.fragment);
        for _ in actual..expected {
            sequence.push('-');
        }
    }

    Ok(())
}

fn is_clustal_header(header: &str) -> bool {
    let header = header.trim_start();
    header.starts_with("CLUSTAL")
}

fn is_conservation_line(line: &str) -> bool {
    line.starts_with(char::is_whitespace)
        && line
            .trim()
            .chars()
            .all(|character| matches!(character, '*' | ':' | '.' | ' '))
}

fn validate_lengths(
    sequence_order: &[String],
    aligned_sequences: &HashMap<String, String>,
) -> Result<(), ParseError> {
    let expected = aligned_sequences
        .get(&sequence_order[0])
        .expect("should have base sequence")
        .len();

    for name in sequence_order.iter().skip(1) {
        let actual = aligned_sequences
            .get(name)
            .expect("should have sequence")
            .len();
        if actual != expected {
            return Err(ParseError::MismatchedAlignedLength {
                name: name.clone(),
                actual,
                expected,
            });
        }
    }

    Ok(())
}

fn build_graph(
    base_name: &str,
    sequence_order: &[String],
    aligned_sequences: &HashMap<String, String>,
) -> GenGraph {
    let base_aligned = aligned_sequences
        .get(base_name)
        .expect("should have base sequence");
    let base_sequence = ungapped(base_aligned);
    let base_node_id = HashId::convert_str(&format!("clustalw:base:{base_name}:{base_sequence}"));
    let runs = alignment_runs(base_aligned, sequence_order, aligned_sequences);

    let mut graph = GenGraph::new();
    let mut previous_by_sequence = vec![terminal_node(PATH_START_NODE_ID); sequence_order.len()];
    let end_node = terminal_node(PATH_END_NODE_ID);
    let mut base_offset = 0_i64;

    for run in runs {
        let base_text = ungapped(&base_aligned[run.start..run.end]);
        let base_node = (!base_text.is_empty()).then(|| {
            let node = GraphNode {
                node_id: base_node_id,
                sequence_start: base_offset,
                sequence_end: base_offset + base_text.len() as i64,
            };
            base_offset += base_text.len() as i64;
            node
        });

        for (sequence_index, name) in sequence_order.iter().enumerate() {
            let aligned = aligned_sequences.get(name).expect("should have sequence");
            let text = ungapped(&aligned[run.start..run.end]);
            let next = if run.kind == RunKind::Invariant || sequence_index == 0 {
                base_node
            } else if text.is_empty() {
                None
            } else if text == base_text {
                base_node
            } else {
                Some(GraphNode {
                    node_id: HashId::convert_str(&format!(
                        "clustalw:variant:{name}:{}:{}:{text}",
                        run.start, run.end
                    )),
                    sequence_start: 0,
                    sequence_end: text.len() as i64,
                })
            };

            if let Some(next_node) = next {
                add_path_edge(
                    &mut graph,
                    previous_by_sequence[sequence_index],
                    next_node,
                    sequence_index as i64,
                );
                previous_by_sequence[sequence_index] = next_node;
            }
        }
    }

    for (sequence_index, previous) in previous_by_sequence.iter().enumerate() {
        add_path_edge(&mut graph, *previous, end_node, sequence_index as i64);
    }

    graph
}

fn build_paf_graph(mapping: &ParsedMapping) -> GenGraph {
    let mut graph = GenGraph::new();
    let mut target_previous = terminal_node(PATH_START_NODE_ID);
    let mut query_previous = terminal_node(PATH_START_NODE_ID);
    let end_node = terminal_node(PATH_END_NODE_ID);
    let mut target_offset = mapping.target_start;
    let mut query_offset = mapping.query_start;
    let target_node_id = HashId::convert_str(&format!(
        "paf:target:{}:{}",
        mapping.target_name, mapping.target_len
    ));
    let query_node_id = HashId::convert_str(&format!(
        "paf:query:{}:{}",
        mapping.query_name, mapping.query_len
    ));

    for op in &mapping.cigar {
        match *op {
            CigarOp::Match(len) | CigarOp::Equal(len) | CigarOp::Difference(len) => {
                let node = GraphNode {
                    node_id: target_node_id,
                    sequence_start: target_offset,
                    sequence_end: target_offset + len,
                };
                add_path_edge(&mut graph, target_previous, node, 0);
                add_path_edge(&mut graph, query_previous, node, 1);
                target_previous = node;
                query_previous = node;
                target_offset += len;
                query_offset += len;
            }
            CigarOp::Insertion(len) => {
                let node = GraphNode {
                    node_id: query_node_id,
                    sequence_start: query_offset,
                    sequence_end: query_offset + len,
                };
                add_path_edge(&mut graph, query_previous, node, 1);
                query_previous = node;
                query_offset += len;
            }
            CigarOp::Deletion(len) | CigarOp::ReferenceSkip(len) => {
                let node = GraphNode {
                    node_id: target_node_id,
                    sequence_start: target_offset,
                    sequence_end: target_offset + len,
                };
                add_path_edge(&mut graph, target_previous, node, 0);
                target_previous = node;
                target_offset += len;
            }
            CigarOp::SoftClip(len) => {
                query_offset += len;
            }
            CigarOp::HardClip(_) => {}
        }
    }

    add_path_edge(&mut graph, target_previous, end_node, 0);
    add_path_edge(&mut graph, query_previous, end_node, 1);
    graph
}

fn build_psl_graph(mapping: &ParsedMapping, blocks: &[PslBlock]) -> GenGraph {
    let mut graph = GenGraph::new();
    let mut target_previous = terminal_node(PATH_START_NODE_ID);
    let mut query_previous = terminal_node(PATH_START_NODE_ID);
    let end_node = terminal_node(PATH_END_NODE_ID);
    let target_node_id = HashId::convert_str(&format!(
        "psl:target:{}:{}",
        mapping.target_name, mapping.target_len
    ));
    let query_node_id = HashId::convert_str(&format!(
        "psl:query:{}:{}",
        mapping.query_name, mapping.query_len
    ));

    for (index, block) in blocks.iter().enumerate() {
        if let Some(previous_block) = index
            .checked_sub(1)
            .and_then(|previous| blocks.get(previous))
        {
            let target_gap_start = previous_block.target_start + previous_block.size;
            let target_gap_end = block.target_start;
            let (query_gap_start, query_gap_end) =
                psl_query_gap(previous_block, block, mapping.strand);

            if target_gap_end > target_gap_start {
                let node = GraphNode {
                    node_id: target_node_id,
                    sequence_start: target_gap_start,
                    sequence_end: target_gap_end,
                };
                add_path_edge(&mut graph, target_previous, node, 0);
                target_previous = node;
            }

            if query_gap_end > query_gap_start {
                let node = GraphNode {
                    node_id: query_node_id,
                    sequence_start: query_gap_start,
                    sequence_end: query_gap_end,
                };
                add_path_edge(&mut graph, query_previous, node, 1);
                query_previous = node;
            }
        }

        let node = GraphNode {
            node_id: target_node_id,
            sequence_start: block.target_start,
            sequence_end: block.target_start + block.size,
        };
        add_path_edge(&mut graph, target_previous, node, 0);
        add_path_edge(&mut graph, query_previous, node, 1);
        target_previous = node;
        query_previous = node;
    }

    add_path_edge(&mut graph, target_previous, end_node, 0);
    add_path_edge(&mut graph, query_previous, end_node, 1);
    graph
}

fn psl_query_gap(previous_block: &PslBlock, block: &PslBlock, strand: Strand) -> (i64, i64) {
    if strand == Strand::Reverse {
        (block.query_start + block.size, previous_block.query_start)
    } else {
        (
            previous_block.query_start + previous_block.size,
            block.query_start,
        )
    }
}

fn alignment_runs(
    base_aligned: &str,
    sequence_order: &[String],
    aligned_sequences: &HashMap<String, String>,
) -> Vec<AlignmentRun> {
    let mut runs = Vec::new();
    let mut start = 0;
    let mut current_kind = None;

    for column in 0..base_aligned.len() {
        let kind = column_kind(column, base_aligned, sequence_order, aligned_sequences);
        if current_kind.is_some_and(|current| current != kind) {
            runs.push(AlignmentRun {
                start,
                end: column,
                kind: current_kind.expect("should have run kind"),
            });
            start = column;
        }
        current_kind = Some(kind);
    }

    if let Some(kind) = current_kind {
        runs.push(AlignmentRun {
            start,
            end: base_aligned.len(),
            kind,
        });
    }

    runs
}

fn column_kind(
    column: usize,
    base_aligned: &str,
    sequence_order: &[String],
    aligned_sequences: &HashMap<String, String>,
) -> RunKind {
    let base = base_aligned.as_bytes()[column];
    if base == b'-' {
        return RunKind::Variant;
    }

    let invariant = sequence_order.iter().all(|name| {
        aligned_sequences
            .get(name)
            .expect("should have sequence")
            .as_bytes()[column]
            == base
    });

    if invariant {
        RunKind::Invariant
    } else {
        RunKind::Variant
    }
}

fn add_path_edge(graph: &mut GenGraph, source: GraphNode, target: GraphNode, sequence_index: i64) {
    let edge = GraphEdge {
        edge_id: HashId::convert_str(&format!(
            "clustalw:edge:{sequence_index}:{}:{}:{}:{}",
            source.node_id, source.sequence_start, target.node_id, target.sequence_start
        )),
        source_strand: Strand::Forward,
        target_strand: Strand::Forward,
        chromosome_index: sequence_index,
        phased: 0,
        created_on: 0,
    };

    if let Some(edges) = graph.edge_weight_mut(source, target) {
        edges.push(edge);
    } else {
        graph.add_edge(source, target, vec![edge]);
    }
}

fn terminal_node(node_id: HashId) -> GraphNode {
    GraphNode {
        node_id,
        sequence_start: 0,
        sequence_end: 0,
    }
}

fn ungapped(sequence: &str) -> String {
    sequence
        .chars()
        .filter(|character| *character != '-')
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AlignmentRun {
    start: usize,
    end: usize,
    kind: RunKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunKind {
    Invariant,
    Variant,
}

#[derive(Debug)]
struct BlockRow {
    name: String,
    fragment: String,
    count: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct PslBlock {
    size: i64,
    query_start: i64,
    target_start: i64,
}

#[derive(Clone, Debug)]
struct MafSequenceRow {
    name: String,
    text: String,
}
