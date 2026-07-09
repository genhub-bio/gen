use std::path::Path as FsPath;

use gen_models::{
    assets::{AssetUri, tables::AssetRef},
    db::GraphConnection,
    file_types::FileTypes,
    traits::Query,
};
use rusqlite::params;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationAssetEntry {
    pub id: gen_core::HashId,
    pub asset_uri: String,
    pub file_type: FileTypes,
    pub checksum: gen_core::HashId,
}

impl AnnotationAssetEntry {
    pub fn file_path(&self) -> &str {
        self.asset_uri
            .strip_prefix("file://")
            .unwrap_or(&self.asset_uri)
    }

    pub fn hashed_filename(&self) -> String {
        <dyn AssetUri>::from_uri(&self.asset_uri).hashed_filename(&self.checksum)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationFileEntry {
    pub file_addition: AnnotationAssetEntry,
    pub index_file_addition: Option<AnnotationAssetEntry>,
    pub name: Option<String>,
    pub display_name: String,
}

fn file_type_from_asset_ref(asset_ref: &AssetRef) -> Option<FileTypes> {
    match asset_ref.file_type.as_str() {
        "gb" => Some(FileTypes::GenBank),
        "fasta" => Some(FileTypes::Fasta),
        "gfa" => Some(FileTypes::GFA),
        "gaf" => Some(FileTypes::GAF),
        "vcf" => Some(FileTypes::VCF),
        "changeset" => Some(FileTypes::Changeset),
        "csv" => Some(FileTypes::CSV),
        "gff3" => Some(FileTypes::Gff3),
        "bed" => Some(FileTypes::Bed),
        "tabix" => Some(FileTypes::Tabix),
        "none" => Some(FileTypes::None),
        _ => None,
    }
}

fn asset_entry_from_ref(asset_ref: &AssetRef) -> Option<AnnotationAssetEntry> {
    Some(AnnotationAssetEntry {
        id: asset_ref.id,
        asset_uri: asset_ref.uri.clone(),
        file_type: file_type_from_asset_ref(asset_ref)?,
        checksum: asset_ref.checksum?,
    })
}

pub fn load_annotation_file_entries(conn: &GraphConnection) -> Vec<AnnotationFileEntry> {
    let asset_refs = AssetRef::query(
        conn,
        "SELECT asset_refs.* \
         FROM gen_operation_log operation_logs \
         JOIN gen_operation_assets operation_assets ON operation_assets.log_id = operation_logs.id \
         JOIN gen_asset_refs asset_refs ON asset_refs.id = operation_assets.asset_ref_id \
         WHERE operation_logs.operation_kind = ?1 \
         ORDER BY operation_logs.created_on, operation_assets.role, asset_refs.created_on, asset_refs.name",
        params!["annotation-file"],
    );
    let mut entries = Vec::new();
    let operation_assets = conn
        .prepare(
            "SELECT operation_logs.id, operation_assets.asset_ref_id, operation_assets.role \
             FROM gen_operation_log operation_logs \
             JOIN gen_operation_assets operation_assets ON operation_assets.log_id = operation_logs.id \
             WHERE operation_logs.operation_kind = ?1 \
             ORDER BY operation_logs.created_on, operation_assets.role, operation_assets.asset_ref_id",
        )
        .expect("should prepare annotation operation asset query")
        .query_map(params!["annotation-file"], |row| {
            Ok((
                row.get::<_, gen_core::HashId>(0)?,
                row.get::<_, gen_core::HashId>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .expect("should query annotation operation assets")
        .collect::<Result<Vec<_>, _>>()
        .expect("should decode annotation operation assets");
    let asset_refs_by_id = asset_refs
        .into_iter()
        .map(|asset_ref| (asset_ref.id, asset_ref))
        .collect::<std::collections::HashMap<_, _>>();
    let mut annotation_by_log = std::collections::HashMap::new();
    let mut index_by_log = std::collections::HashMap::new();
    for (log_id, asset_ref_id, role) in operation_assets {
        if role == "annotation" {
            annotation_by_log.insert(log_id, asset_ref_id);
        } else if role == "annotation-index" {
            index_by_log.insert(log_id, asset_ref_id);
        }
    }

    let mut log_ids = annotation_by_log.keys().copied().collect::<Vec<_>>();
    log_ids.sort();
    for log_id in log_ids {
        let Some(asset_ref) = annotation_by_log
            .get(&log_id)
            .and_then(|asset_ref_id| asset_refs_by_id.get(asset_ref_id))
        else {
            continue;
        };
        let Some(file_addition) = asset_entry_from_ref(asset_ref) else {
            continue;
        };
        let index_file_addition = index_by_log
            .get(&log_id)
            .and_then(|asset_ref_id| asset_refs_by_id.get(asset_ref_id))
            .and_then(asset_entry_from_ref);
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

    use gen_models::annotations::add_annotation_file;

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
        )
        .expect("should create annotation file operation");

        let entries = load_annotation_file_entries(context.graph().conn());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name.as_deref(), Some("fixture-track"));
        assert_eq!(entries[0].display_name, "fixture-track");
        assert_eq!(
            entries[0].file_addition.file_type,
            gen_models::file_types::FileTypes::Gff3
        );
        assert!(entries[0].index_file_addition.is_none());
    }
}
