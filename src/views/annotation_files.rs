use std::path::Path as FsPath;

use gen_models::{db::OperationsConnection, operations::FileAddition, traits::Query};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationFileEntry {
    pub file_addition: FileAddition,
    pub name: Option<String>,
    pub display_name: String,
}

pub fn load_annotation_file_entries(conn: &OperationsConnection) -> Vec<AnnotationFileEntry> {
    let query = "\
        SELECT fa.*, af.name \
        FROM annotation_files af \
        JOIN file_additions fa ON fa.id = af.file_addition_id \
        ORDER BY COALESCE(af.name, fa.file_path);";
    let mut stmt = conn.prepare(query).unwrap();
    let rows = stmt
        .query_map([], |row| {
            let file_addition = FileAddition::process_row(row);
            let name: Option<String> = row.get(4)?;
            let display_name = name.clone().unwrap_or_else(|| {
                FsPath::new(&file_addition.file_path)
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_else(|| file_addition.file_path.clone())
            });
            Ok(AnnotationFileEntry {
                file_addition,
                name,
                display_name,
            })
        })
        .unwrap();
    rows.filter_map(Result::ok).collect()
}
