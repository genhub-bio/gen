use std::path::Path as FsPath;

use gen_core::{HashId, Sha256Hash};
use gen_models::{
    assets::{AssetRef, AssetUri, Assets},
    db::GraphConnection,
    file_types::FileTypes,
};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationAssetEntry {
    pub id: HashId,
    pub asset_uri: String,
    pub file_type: FileTypes,
    pub checksum: Option<Sha256Hash>,
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
    pub attribute_index_file_addition: Option<AnnotationAssetEntry>,
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
        let attribute_index_file_addition = annotation_file
            .attribute_index
            .as_ref()
            .map(asset_entry_from_ref);
        let name = asset_ref.name.clone();
        let display_name = name.clone().unwrap_or_else(|| {
            FsPath::new(
                asset_ref
                    .logical_path
                    .as_deref()
                    .unwrap_or_else(|| file_addition.file_path()),
            )
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| file_addition.file_path().to_string())
        });
        entries.push(AnnotationFileEntry {
            file_addition,
            index_file_addition,
            attribute_index_file_addition,
            name,
            display_name,
        });
    }
    entries
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum AnnotationFileSelectionError {
    #[error("no imported annotation file is named {name:?}; imported files: {available}")]
    Unknown { name: String, available: String },
}

/// The imported annotation files the user named, in the order named. A name matches a file's
/// track name or, when it has none, the file name it is shown under. Viewers draw only these
/// files, so a typo has to stop the command rather than silently draw nothing.
pub fn select_annotation_file_entries(
    entries: Vec<AnnotationFileEntry>,
    names: &[String],
) -> Result<Vec<AnnotationFileEntry>, AnnotationFileSelectionError> {
    let mut selected: Vec<AnnotationFileEntry> = Vec::with_capacity(names.len());
    for name in names {
        let Some(entry) = entries.iter().find(|entry| &entry.display_name == name) else {
            let available = if entries.is_empty() {
                "none".to_string()
            } else {
                entries
                    .iter()
                    .map(|entry| entry.display_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            return Err(AnnotationFileSelectionError::Unknown {
                name: name.clone(),
                available,
            });
        };
        if !selected.contains(entry) {
            selected.push(entry.clone());
        }
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_core::HashId;
    use gen_models::{
        annotations::{AnnotationFileChecksumOverrides, add_annotation_file},
        assets::Assets,
        file_types::FileTypes,
    };

    use super::{
        AnnotationAssetEntry, AnnotationFileEntry, load_annotation_file_entries,
        select_annotation_file_entries,
    };
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
        let annotation_files = Assets::get_annotation_files(context.graph().conn(), None)
            .expect("should load the annotation and generated indexes");
        assert_eq!(annotation_files.len(), 1);
        let annotation_assets = &annotation_files[0];
        let coordinate_index = annotation_assets
            .index
            .as_ref()
            .expect("should generate a coordinate index for a locally imported GFF");
        assert_eq!(coordinate_index.file_type, FileTypes::Tabix.as_str());
        assert!(coordinate_index.uri.ends_with(".csi"));
        assert_eq!(
            coordinate_index.upstream_asset_ref_id,
            Some(annotation_assets.annotation.id),
            "generated CSI should be associated with its source annotation"
        );
        let attribute_index = annotation_assets
            .attribute_index
            .as_ref()
            .expect("should generate an attribute index for a locally imported GFF");
        assert_eq!(attribute_index.file_type, FileTypes::None.as_str());
        assert!(attribute_index.uri.ends_with(".gai"));
        assert_eq!(
            attribute_index.upstream_asset_ref_id,
            Some(annotation_assets.annotation.id),
            "generated GAI should be associated with its source annotation"
        );

        let loaded_coordinate_index = entries[0]
            .index_file_addition
            .as_ref()
            .expect("should retain the generated coordinate index in the viewer entry");
        assert_eq!(loaded_coordinate_index.id, coordinate_index.id);
        assert_eq!(loaded_coordinate_index.asset_uri, coordinate_index.uri);
        assert_eq!(loaded_coordinate_index.file_type, FileTypes::Tabix);
        let loaded_attribute_index = entries[0]
            .attribute_index_file_addition
            .as_ref()
            .expect("local GFF should have a generated attribute index");
        assert_eq!(
            loaded_attribute_index.id, attribute_index.id,
            "viewer entry should retain the generated GAI reference"
        );
        assert_eq!(loaded_attribute_index.asset_uri, attribute_index.uri);
        assert_eq!(loaded_attribute_index.file_type, FileTypes::None);
        assert!(loaded_attribute_index.asset_uri.ends_with(".gai"));
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

    #[test]
    fn test_load_annotation_file_entries_uses_remote_filename() {
        let context = setup_gen_on_disk();
        let annotation_uri = "https://example.com/annotations/genes.gff3";

        add_annotation_file(
            &context,
            annotation_uri,
            None,
            None,
            None,
            Some("add remote annotation"),
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should create remote annotation file operation");

        let entries = load_annotation_file_entries(context.graph().conn(), None);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].display_name, "genes.gff3");
        assert_eq!(entries[0].file_addition.asset_uri, annotation_uri);
        assert_eq!(entries[0].file_addition.checksum, None);
    }

    #[test]
    fn test_uses_logical_filename_for_unnamed_retained_annotation() {
        let context = setup_gen_on_disk();
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.gff");

        add_annotation_file(
            &context,
            fixture_path.to_str().expect("should encode fixture path"),
            None,
            None,
            None,
            Some("add unnamed annotation"),
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should create annotation file operation");

        let entries = load_annotation_file_entries(context.graph().conn(), None);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].display_name, "simple.gff");
        assert_ne!(
            entries[0].display_name,
            entries[0].file_addition.file_path()
        );

        let annotation_asset = Assets::get_annotation_files(context.graph().conn(), None)
            .expect("should load annotation asset refs")
            .into_iter()
            .next()
            .expect("should find annotation asset ref")
            .annotation;
        let versioned_store_path = annotation_asset
            .versioned_store_path(context.workspace())
            .expect("should locate annotation asset in versioned store");
        assert!(versioned_store_path.is_file());
    }

    fn entry_named(name: &str) -> AnnotationFileEntry {
        let asset = AnnotationAssetEntry {
            id: HashId::convert_str(name),
            asset_uri: format!("file:///{name}.gff"),
            file_type: FileTypes::from_storage_tag("gff3"),
            checksum: None,
        };
        AnnotationFileEntry {
            file_addition: asset,
            index_file_addition: None,
            attribute_index_file_addition: None,
            name: Some(name.to_string()),
            display_name: name.to_string(),
        }
    }

    #[test]
    fn test_selects_the_named_files_in_the_order_named() {
        let entries = vec![entry_named("genes"), entry_named("peaks")];

        let selected = select_annotation_file_entries(
            entries,
            &[
                "peaks".to_string(),
                "genes".to_string(),
                "peaks".to_string(),
            ],
        )
        .expect("should select the named files");

        let names: Vec<&str> = selected
            .iter()
            .map(|entry| entry.display_name.as_str())
            .collect();
        assert_eq!(names, ["peaks", "genes"]);
    }

    #[test]
    fn test_selecting_an_unknown_file_lists_the_imported_ones() {
        let entries = vec![entry_named("genes"), entry_named("peaks")];

        let error = select_annotation_file_entries(entries, &["gens".to_string()])
            .expect_err("should reject an unknown name");

        assert_eq!(
            error.to_string(),
            "no imported annotation file is named \"gens\"; imported files: genes, peaks"
        );
    }

    #[test]
    fn test_selecting_no_names_selects_no_files() {
        let selected = select_annotation_file_entries(vec![entry_named("genes")], &[])
            .expect("should select nothing");

        assert!(selected.is_empty());
    }
}
