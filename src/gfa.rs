use std::{
    fs::File,
    io::{BufWriter, Write},
};

use convert_case::{Case, Casing};
use gen_core::{HashId, Strand};

#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct Segment {
    pub sequence: String,
    pub node_id: HashId,
    pub sequence_start: i64,
    pub sequence_end: i64,
    pub strand: Strand,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct Link {
    pub source_segment_id: String,
    pub source_strand: Strand,
    pub target_segment_id: String,
    pub target_strand: Strand,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Path {
    pub name: String,
    pub segment_ids: Vec<String>,
    pub node_strands: Vec<Strand>,
}

impl Segment {
    pub fn segment_id(&self) -> String {
        format!(
            "{}.{}.{}",
            self.node_id, self.sequence_start, self.sequence_end
        )
    }
}

fn segment_line(segment: &Segment) -> String {
    // NOTE: We encode the node ID and start coordinate in the segment ID
    format!("S\t{}\t{}\n", segment.segment_id(), segment.sequence)
}

fn link_line(link: &Link) -> String {
    format!(
        "L\t{}\t{}\t{}\t{}\t0M\n",
        link.source_segment_id, link.source_strand, link.target_segment_id, link.target_strand
    )
}

pub fn path_line(path: &Path) -> String {
    let segments = path
        .segment_ids
        .iter()
        .zip(path.node_strands.iter())
        .map(|(segment_id, node_strand)| format!("{segment_id}{node_strand}"))
        .collect::<Vec<String>>()
        .join(",");
    format!("P\t{}\t{}\t*\n", path.name.to_case(Case::Train), segments)
}

pub fn write_segments(writer: &mut BufWriter<File>, segments: &[&Segment]) {
    for segment in segments {
        writer
            .write_all(&segment_line(segment).into_bytes())
            .unwrap_or_else(|_| {
                panic!(
                    "Error writing segment with sequence {} to GFA stream",
                    segment.sequence,
                )
            });
    }
}

pub fn write_links(writer: &mut BufWriter<File>, links: &[&Link]) {
    for link in links {
        writer
            .write_all(&link_line(link).into_bytes())
            .unwrap_or_else(|_| {
                panic!(
                    "Error writing link from segment {:?} to {:?} to GFA stream",
                    link.source_segment_id, link.target_segment_id,
                )
            });
    }
}

pub fn bool_to_strand(direction: bool) -> Strand {
    if direction {
        Strand::Forward
    } else {
        Strand::Reverse
    }
}
