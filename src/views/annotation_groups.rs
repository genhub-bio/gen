use gen_models::db::GraphConnection;
use rusqlite::params;

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
    let mut stmt = conn
        .prepare(
            "SELECT annotation_group FROM annotation_group_samples WHERE sample_name = ?1 ORDER BY annotation_group;",
        )
        .unwrap();
    let rows = stmt
        .query_map(params![sample_name], |row| row.get::<_, String>(0))
        .unwrap();
    rows.filter_map(Result::ok)
        .map(|name| AnnotationGroupEntry { name })
        .collect()
}
