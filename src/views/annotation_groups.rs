use gen_models::{annotations::AnnotationGroup, db::GraphConnection};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationGroupEntry {
    pub name: String,
}

pub fn load_annotation_group_entries(
    conn: &GraphConnection,
    sample_name: Option<&str>,
) -> Vec<AnnotationGroupEntry> {
    let Some(sample_name) = sample_name else {
        return Vec::new();
    };
    AnnotationGroup::query_by_sample(conn, sample_name)
        .into_iter()
        .map(|group| AnnotationGroupEntry { name: group.name })
        .collect()
}
