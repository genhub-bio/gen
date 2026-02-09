use std::path::Path as FsPath;

use gen_models::{annotations::AnnotationFile, db::OperationsConnection, operations::FileAddition};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationFileEntry {
    pub file_addition: FileAddition,
    pub index_file_addition: Option<FileAddition>,
    pub name: Option<String>,
    pub display_name: String,
}

pub fn load_annotation_file_entries(conn: &OperationsConnection) -> Vec<AnnotationFileEntry> {
    let mut entries = Vec::new();
    for info in AnnotationFile::get_all_files(conn) {
        let display_name = info.name.clone().unwrap_or_else(|| {
            FsPath::new(&info.file_addition.file_path)
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| info.file_addition.file_path.clone())
        });
        entries.push(AnnotationFileEntry {
            file_addition: info.file_addition,
            index_file_addition: info.index_file_addition,
            name: info.name,
            display_name,
        });
    }
    entries
}
