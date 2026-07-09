use std::path::PathBuf;

use gen_core::{HashId, Workspace};
use rusqlite::{Row, params};

use crate::{
    assets::materialize_asset_uri_to_workspace, db::GraphConnection, errors::FileAdditionError,
    traits::Query,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetRef {
    pub id: HashId,
    pub uri: String,
    pub file_type: String,
    pub checksum: Option<HashId>,
    pub size: Option<i64>,
    pub role: String,
    pub logical_path: Option<String>,
    pub name: Option<String>,
    pub created_on: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationLog {
    pub id: HashId,
    pub operation_kind: String,
    pub command: String,
    pub created_on: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationAsset {
    pub log_id: HashId,
    pub asset_ref_id: HashId,
    pub role: String,
}

impl Query for AssetRef {
    type Model = AssetRef;

    const PRIMARY_KEY: &'static str = "id";
    const TABLE_NAME: &'static str = "gen_asset_refs";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            id: row.get(0).unwrap(),
            uri: row.get(1).unwrap(),
            file_type: row.get(2).unwrap(),
            checksum: row.get(3).unwrap(),
            size: row.get(4).unwrap(),
            role: row.get(5).unwrap(),
            logical_path: row.get(6).unwrap(),
            name: row.get(7).unwrap(),
            created_on: row.get(8).unwrap(),
        }
    }
}

impl Query for OperationLog {
    type Model = OperationLog;

    const PRIMARY_KEY: &'static str = "id";
    const TABLE_NAME: &'static str = "gen_operation_log";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            id: row.get(0).unwrap(),
            operation_kind: row.get(1).unwrap(),
            command: row.get(2).unwrap(),
            created_on: row.get(3).unwrap(),
        }
    }
}

impl Query for OperationAsset {
    type Model = OperationAsset;

    const PRIMARY_KEY: &'static str = "log_id";
    const TABLE_NAME: &'static str = "gen_operation_assets";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            log_id: row.get(0).unwrap(),
            asset_ref_id: row.get(1).unwrap(),
            role: row.get(2).unwrap(),
        }
    }
}

impl AssetRef {
    pub fn create(conn: &GraphConnection, asset_ref: &AssetRef) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT OR IGNORE INTO gen_asset_refs \
             (id, uri, file_type, checksum, size, role, logical_path, name, created_on) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                asset_ref.id,
                asset_ref.uri,
                asset_ref.file_type,
                asset_ref.checksum,
                asset_ref.size,
                asset_ref.role,
                asset_ref.logical_path,
                asset_ref.name,
                asset_ref.created_on
            ],
        )?;
        Ok(())
    }

    pub fn materialize(&self, workspace: &Workspace) -> Result<PathBuf, FileAdditionError> {
        materialize_asset_uri_to_workspace(
            workspace,
            &self.uri,
            self.checksum.as_ref(),
            self.logical_path.as_deref(),
        )
    }
}

impl OperationLog {
    pub fn create(conn: &GraphConnection, operation_log: &OperationLog) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT OR IGNORE INTO gen_operation_log (id, operation_kind, command, created_on) \
             VALUES (?1, ?2, ?3, ?4)",
            params![
                operation_log.id,
                operation_log.operation_kind,
                operation_log.command,
                operation_log.created_on
            ],
        )?;
        Ok(())
    }
}

impl OperationAsset {
    pub fn create(
        conn: &GraphConnection,
        operation_asset: &OperationAsset,
    ) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT OR IGNORE INTO gen_operation_assets (log_id, asset_ref_id, role) \
             VALUES (?1, ?2, ?3)",
            params![
                operation_asset.log_id,
                operation_asset.asset_ref_id,
                operation_asset.role
            ],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use gen_core::HashId;

    use super::{AssetRef, OperationAsset, OperationLog};
    use crate::{test_helpers::setup_gen, traits::Query};

    #[test]
    fn test_asset_reference_tables_round_trip() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let asset_ref = AssetRef {
            id: HashId::convert_str("asset-ref"),
            uri: "s3://bucket/reference.fa".to_string(),
            file_type: "fasta".to_string(),
            checksum: Some(HashId::convert_str("checksum")),
            size: Some(1024),
            role: "input".to_string(),
            logical_path: Some("refs/reference.fa".to_string()),
            name: Some("reference.fa".to_string()),
            created_on: 1,
        };
        let operation_log = OperationLog {
            id: HashId::convert_str("log"),
            operation_kind: "import".to_string(),
            command: "gen import fasta".to_string(),
            created_on: 1,
        };
        let operation_asset = OperationAsset {
            log_id: operation_log.id,
            asset_ref_id: asset_ref.id,
            role: "input".to_string(),
        };

        OperationLog::create(conn, &operation_log).expect("should insert operation log");
        AssetRef::create(conn, &asset_ref).expect("should insert asset ref");
        OperationAsset::create(conn, &operation_asset).expect("should insert operation asset");

        let asset_refs = AssetRef::query(conn, "SELECT * FROM gen_asset_refs", []);
        let operation_logs = OperationLog::query(conn, "SELECT * FROM gen_operation_log", []);
        let operation_assets =
            OperationAsset::query(conn, "SELECT * FROM gen_operation_assets", []);

        assert_eq!(asset_refs, vec![asset_ref]);
        assert_eq!(operation_logs, vec![operation_log]);
        assert_eq!(operation_assets, vec![operation_asset]);
    }

    #[test]
    fn test_asset_ref_materialize_restores_logical_file_path() {
        let context = setup_gen();
        let source_path = context
            .workspace()
            .repo_root()
            .expect("should resolve repo root")
            .join("source/remote.gff");
        fs::create_dir_all(
            source_path
                .parent()
                .expect("source fixture should have a parent directory"),
        )
        .expect("should create source directory");
        fs::write(&source_path, "remote-asset").expect("should write source asset");
        let asset_ref = AssetRef {
            id: HashId::convert_str("materialize-logical"),
            uri: format!(
                "file://{}",
                source_path
                    .strip_prefix(
                        context
                            .workspace()
                            .repo_root()
                            .expect("should resolve repo root")
                    )
                    .expect("source path should remain under repo root")
                    .to_string_lossy()
            ),
            file_type: "gff3".to_string(),
            checksum: Some(HashId::convert_str("logical-checksum")),
            size: Some(12),
            role: "annotation".to_string(),
            logical_path: Some("annotations/remote.gff".to_string()),
            name: Some("remote.gff".to_string()),
            created_on: 1,
        };

        let path = asset_ref
            .materialize(context.workspace())
            .expect("should materialize logical asset path");

        assert_eq!(
            path,
            context
                .workspace()
                .repo_root()
                .expect("should resolve repo root")
                .join("annotations/remote.gff")
        );
        assert_eq!(
            fs::read_to_string(&path).expect("should read materialized logical asset"),
            "remote-asset"
        );
    }

    #[test]
    fn test_asset_ref_materialize_falls_back_to_asset_cache() {
        let context = setup_gen();
        let source_path = context
            .workspace()
            .repo_root()
            .expect("should resolve repo root")
            .join("source/cached.gff");
        fs::create_dir_all(
            source_path
                .parent()
                .expect("source fixture should have a parent directory"),
        )
        .expect("should create source directory");
        fs::write(&source_path, "cached-asset").expect("should write source asset");
        let checksum = HashId::convert_str("cache-checksum");
        let asset_ref = AssetRef {
            id: HashId::convert_str("materialize-cache"),
            uri: format!(
                "file://{}",
                source_path
                    .strip_prefix(
                        context
                            .workspace()
                            .repo_root()
                            .expect("should resolve repo root")
                    )
                    .expect("source path should remain under repo root")
                    .to_string_lossy()
            ),
            file_type: "gff3".to_string(),
            checksum: Some(checksum),
            size: Some(12),
            role: "annotation".to_string(),
            logical_path: None,
            name: Some("remote.gff".to_string()),
            created_on: 1,
        };

        let path = asset_ref
            .materialize(context.workspace())
            .expect("should materialize cached asset path");

        let expected_name =
            <dyn crate::assets::AssetUri>::from_uri(&asset_ref.uri).hashed_filename(&checksum);
        assert_eq!(
            path,
            context
                .workspace()
                .asset_dir()
                .expect("should resolve asset dir")
                .join(expected_name)
        );
        assert_eq!(
            fs::read_to_string(&path).expect("should read materialized cached asset"),
            "cached-asset"
        );
    }
}
