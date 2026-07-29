use std::path::Path as FsPath;

use gen_models::{
    assets::{AssetRef, AssetUri, Assets},
    db::GraphConnection,
    file_types::FileTypes,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationAssetEntry {
    pub id: gen_core::HashId,
    pub asset_uri: String,
    pub file_type: FileTypes,
    pub checksum: Option<gen_core::Sha256Hash>,
}

impl AnnotationAssetEntry {
    pub fn file_path(&self) -> &str {
        self.asset_uri
            .strip_prefix("file://")
            .unwrap_or(&self.asset_uri)
    }

    /// Returns the content-addressed local filename only when the asset has a checksum.
    ///
    /// Views keep checksumless remote annotations visible by using their URI directly instead of
    /// requiring a `.gen/assets` filename.
    pub fn hashed_filename(&self) -> Option<String> {
        self.checksum
            .as_ref()
            .map(|checksum| <dyn AssetUri>::from_uri(&self.asset_uri).hashed_filename(checksum))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationFileEntry {
    pub file_addition: AnnotationAssetEntry,
    pub index_file_addition: Option<AnnotationAssetEntry>,
    pub name: Option<String>,
    pub display_name: String,
}

// Preserve optional checksums while translating persistence records so checksumless remote
// annotations and indexes remain usable by collection and TUI views.
fn asset_entry_from_ref(asset_ref: &AssetRef) -> AnnotationAssetEntry {
    AnnotationAssetEntry {
        id: asset_ref.id,
        asset_uri: asset_ref.uri.clone(),
        file_type: FileTypes::from_storage_tag(&asset_ref.file_type),
        checksum: asset_ref.checksum,
    }
}

pub fn load_annotation_file_entries(
    conn: &GraphConnection,
    history_ref: Option<&str>,
) -> Vec<AnnotationFileEntry> {
    let annotation_files = Assets::get_annotation_files(conn, history_ref)
        .expect("should load annotation file assets");
    let mut entries = Vec::with_capacity(annotation_files.len());
    for annotation_file in annotation_files {
        let asset_ref = &annotation_file.annotation;
        let file_addition = asset_entry_from_ref(asset_ref);
        let index_file_addition = annotation_file.index.as_ref().map(asset_entry_from_ref);
        let name = asset_ref.name.clone();
        let display_name = name.clone().unwrap_or_else(|| {
            FsPath::new(file_addition.file_path())
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| file_addition.file_path().to_string())
        });
        entries.push(AnnotationFileEntry {
            file_addition,
            index_file_addition,
            name,
            display_name,
        });
    }
    entries
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_models::annotations::{AnnotationFileChecksumOverrides, add_annotation_file};

    use super::load_annotation_file_entries;
    use crate::test_helpers::setup_gen_on_disk;

    #[test]
    fn test_loads_annotation_file_entries_from_graph_asset_refs() {
        let context = setup_gen_on_disk();
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.gff");

        add_annotation_file(
            &context,
            fixture_path.to_str().expect("should encode fixture path"),
            None,
            None,
            Some("fixture-track"),
            Some("add-annotation"),
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should create annotation file operation");

        let entries = load_annotation_file_entries(context.graph().conn(), None);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name.as_deref(), Some("fixture-track"));
        assert_eq!(entries[0].display_name, "fixture-track");
        assert_eq!(
            entries[0].file_addition.file_type,
            gen_models::file_types::FileTypes::Gff3
        );
        assert!(entries[0].index_file_addition.is_none());
    }

    #[test]
    fn test_loads_checksumless_remote_annotation_and_index_entries() {
        let context = setup_gen_on_disk();
        let annotation_uri = "https://example.com/annotations/genes.gff3";
        let index_uri = "https://example.com/annotations/genes.gff3.tbi";

        add_annotation_file(
            &context,
            annotation_uri,
            None,
            Some(index_uri),
            Some("remote-track"),
            Some("add remote annotation"),
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should create remote annotation file operation");

        let entries = load_annotation_file_entries(context.graph().conn(), None);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_addition.asset_uri, annotation_uri);
        assert_eq!(entries[0].file_addition.checksum, None);
        let index = entries[0]
            .index_file_addition
            .as_ref()
            .expect("remote index should remain visible");
        assert_eq!(index.asset_uri, index_uri);
        assert_eq!(index.checksum, None);
    }
}
