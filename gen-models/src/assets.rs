use std::{
    collections::HashMap,
    fs,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    string::ToString,
    sync::{Arc, LazyLock, Mutex},
};

use gen_core::{DoltHashId, HashId, Sha256Hash, Workspace, calculate_hash};
use indexmap::IndexMap;
use opendal::{blocking, services};
use rusqlite::{
    Row, ToSql, named_params, params,
    types::{FromSql, FromSqlResult, ToSqlOutput, ValueRef},
};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;
use url::{Position, Url};

use crate::{
    db::GraphConnection,
    errors::{FileAdditionError, FileStoreError, QueryError},
    history::dolt::hash_of,
    operations::FileAddition,
    traits::Query,
};

static OPENDAL_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build OpenDAL runtime")
});

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum OperationKind {
    AddFile,
    AnnotationFile,
    HistoryCommit,
    Other(String),
}

impl OperationKind {
    pub fn as_str(&self) -> &str {
        match self {
            OperationKind::AddFile => "add-file",
            OperationKind::AnnotationFile => "annotation-file",
            OperationKind::HistoryCommit => "history-commit",
            OperationKind::Other(kind) => kind.as_str(),
        }
    }
}

impl From<&str> for OperationKind {
    fn from(kind: &str) -> Self {
        match kind {
            "add-file" => OperationKind::AddFile,
            "annotation-file" => OperationKind::AnnotationFile,
            "history-commit" => OperationKind::HistoryCommit,
            other => OperationKind::Other(other.to_string()),
        }
    }
}

impl From<String> for OperationKind {
    fn from(kind: String) -> Self {
        match kind.as_str() {
            "add-file" => OperationKind::AddFile,
            "annotation-file" => OperationKind::AnnotationFile,
            "history-commit" => OperationKind::HistoryCommit,
            _ => OperationKind::Other(kind),
        }
    }
}

impl core::fmt::Display for OperationKind {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl ToSql for OperationKind {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl FromSql for OperationKind {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        value.as_str().map(OperationKind::from)
    }
}

fn with_opendal_runtime<T>(f: impl FnOnce() -> T) -> T {
    let _guard = OPENDAL_RUNTIME.enter();
    f()
}

fn opendal_file_addition_error(err: opendal::Error) -> FileAdditionError {
    FileAdditionError::FileReadError(io::Error::other(err))
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum AssetRole {
    Input,
    SequenceIndex,
    Annotation,
    AnnotationIndex,
    Other(String),
}

impl AssetRole {
    pub fn as_str(&self) -> &str {
        match self {
            AssetRole::Input => "input",
            AssetRole::SequenceIndex => "sequence-index",
            AssetRole::Annotation => "annotation",
            AssetRole::AnnotationIndex => "annotation-index",
            AssetRole::Other(role) => role.as_str(),
        }
    }
}

impl From<&str> for AssetRole {
    fn from(role: &str) -> Self {
        match role {
            "input" => AssetRole::Input,
            "sequence-index" => AssetRole::SequenceIndex,
            "annotation" => AssetRole::Annotation,
            "annotation-index" => AssetRole::AnnotationIndex,
            other => AssetRole::Other(other.to_string()),
        }
    }
}

impl From<String> for AssetRole {
    fn from(role: String) -> Self {
        match role.as_str() {
            "input" => AssetRole::Input,
            "sequence-index" => AssetRole::SequenceIndex,
            "annotation" => AssetRole::Annotation,
            "annotation-index" => AssetRole::AnnotationIndex,
            _ => AssetRole::Other(role),
        }
    }
}

impl core::fmt::Display for AssetRole {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl ToSql for AssetRole {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl FromSql for AssetRole {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        value.as_str().map(AssetRole::from)
    }
}

/// An immutable reference to an asset recorded in repository history.
///
/// Several references can have the same logical path because Gen retains every committed version.
/// We have two ways to query for assets -- cumulative and materialized. The cumulative query
/// returns an array of every known version of the asset, including older versions whose local
/// path has been updated (for example a generic ./input.fa the user keeps updating for operations).
///
/// The materialized query returns the latest asset for a given logical path up to a provided ref.
/// For example if a user imports a genome with a generically named `reference.fa` and does it again,
/// the materialized query would return the last one.
///
/// Use [`Self::get_cumulative_assets_at`] when that provenance is needed and
/// [`Self::get_materialized_assets_at`] when constructing the one-file-per-path workspace view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetRef {
    pub id: HashId,
    pub uri: String,
    pub file_type: String,
    pub checksum: Option<Sha256Hash>,
    pub size: Option<i64>,
    pub role: AssetRole,
    // The local path to the asset. This is something like input.fa, data/reference.fa, etc.
    pub logical_path: Option<String>,
    pub name: Option<String>,
    pub created_on: i64,
    /// The source asset this asset indexes or otherwise works off.
    pub upstream_asset_ref_id: Option<HashId>,
}

pub struct Assets;

/// Selects whether an asset-history query preserves every version or the final workspace view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssetView {
    /// Retains every local asset version in the selected history.
    Cumulative,
    /// Retains only the asset selected for each logical path at the upper commit.
    Materialized,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationFileAssets {
    pub log_id: HashId,
    pub annotation: AssetRef,
    pub index: Option<AssetRef>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationLog {
    pub id: HashId,
    pub operation_kind: OperationKind,
    pub command: String,
    pub created_on: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationAsset {
    pub log_id: HashId,
    pub asset_ref_id: HashId,
    pub role: AssetRole,
}

impl Query for AssetRef {
    type Model = AssetRef;

    const PRIMARY_KEY: &'static str = "id";
    const TABLE_NAME: &'static str = "gen_asset_refs";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            id: row.get("id").unwrap(),
            uri: row.get("uri").unwrap(),
            file_type: row.get("file_type").unwrap(),
            checksum: row.get("checksum").unwrap(),
            size: row.get("size").unwrap(),
            role: row.get("role").unwrap(),
            logical_path: row.get("logical_path").unwrap(),
            name: row.get("name").unwrap(),
            created_on: row.get("created_on").unwrap(),
            upstream_asset_ref_id: row.get("upstream_asset_ref_id").unwrap(),
        }
    }
}

impl Query for OperationLog {
    type Model = OperationLog;

    const PRIMARY_KEY: &'static str = "id";
    const TABLE_NAME: &'static str = "gen_operation_log";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            id: row.get("id").unwrap(),
            operation_kind: row.get("operation_kind").unwrap(),
            command: row.get("command").unwrap(),
            created_on: row.get("created_on").unwrap(),
        }
    }
}

impl Query for OperationAsset {
    type Model = OperationAsset;

    const PRIMARY_KEY: &'static str = "log_id";
    const TABLE_NAME: &'static str = "gen_operation_assets";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            log_id: row.get("log_id").unwrap(),
            asset_ref_id: row.get("asset_ref_id").unwrap(),
            role: row.get("role").unwrap(),
        }
    }
}

impl AssetRef {
    /// Return the versioned store path for an asset.
    ///
    /// Logical paths describe the materialized workspace view and can point at a newer version.
    /// Sequence readers use this path so they cannot follow repository metadata outside the
    /// repository or silently switch to another asset version.
    pub fn versioned_store_path(
        &self,
        workspace: &Workspace,
    ) -> Result<PathBuf, FileAdditionError> {
        let checksum = self.checksum.ok_or_else(|| {
            FileAdditionError::ChecksumError(format!(
                "asset {} has no checksum for repository lookup",
                self.id
            ))
        })?;
        let repo_root = LocalAssetUri::canonicalize_or_normalize(&workspace.repo_root()?);
        let asset_path = workspace
            .asset_dir()?
            .join(<dyn AssetUri>::from_uri(&self.uri).hashed_filename(&checksum));
        if !asset_path.exists() {
            return Err(FileAdditionError::FileNotFound(
                asset_path.display().to_string(),
            ));
        }
        let resolved_path = LocalAssetUri::canonicalize_or_normalize(&asset_path);
        if !resolved_path.starts_with(&repo_root) {
            return Err(FileAdditionError::PathOutsideRepo {
                path: resolved_path,
                repo_root,
            });
        }
        Ok(resolved_path)
    }

    /// Opens the asset identified by this reference.
    ///
    /// Local references resolve through the repository's content-addressed asset store. Remote
    /// references retain their URI and are opened lazily through the configured storage backend.
    pub fn reader(&self, workspace: &Workspace) -> Result<blocking::StdReader, FileAdditionError> {
        if LocalAssetUri::is_local_path_or_file_uri(&self.uri) {
            let asset_path = self.versioned_store_path(workspace)?;
            return OpenDalLocation::from_absolute_path(&asset_path)
                .map_err(opendal_file_addition_error)?
                .reader();
        }

        <dyn AssetUri>::from_uri(&self.uri).reader(workspace)
    }

    pub fn id_hash(
        uri: &str,
        file_type: &str,
        checksum: Option<&Sha256Hash>,
        role: &AssetRole,
        logical_path: Option<&str>,
        name: Option<&str>,
        upstream_asset_ref_id: Option<&HashId>,
    ) -> HashId {
        let checksum = checksum
            .map(|checksum| checksum.to_string())
            .unwrap_or_default();
        let mut identity = format!(
            "{uri}:{file_type}:{checksum}:{role}:{logical_path}:{name}",
            role = role.as_str(),
            logical_path = logical_path.unwrap_or_default(),
            name = name.unwrap_or_default(),
        );
        if let Some(upstream_asset_ref_id) = upstream_asset_ref_id {
            identity.push(':');
            identity.push_str(&upstream_asset_ref_id.to_string());
        }
        HashId(calculate_hash(&identity))
    }

    pub(crate) fn from_file_addition(
        file_addition: &FileAddition,
        role: AssetRole,
        logical_path: Option<&str>,
        name: Option<&str>,
        upstream_asset_ref_id: Option<&HashId>,
        created_on: i64,
    ) -> Self {
        let file_type = file_addition.file_type.as_str();
        Self {
            id: Self::id_hash(
                &file_addition.asset_uri,
                file_type,
                file_addition.checksum.as_ref(),
                &role,
                logical_path,
                name,
                upstream_asset_ref_id,
            ),
            uri: file_addition.asset_uri.clone(),
            file_type: file_type.to_string(),
            checksum: file_addition.checksum,
            size: None,
            role,
            logical_path: logical_path.map(str::to_string),
            name: name.map(str::to_string),
            created_on,
            upstream_asset_ref_id: upstream_asset_ref_id.copied(),
        }
    }

    pub fn create(conn: &GraphConnection, asset_ref: &AssetRef) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT OR IGNORE INTO gen_asset_refs \
             (id, uri, file_type, checksum, size, role, logical_path, name, created_on, \
              upstream_asset_ref_id) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                asset_ref.id,
                asset_ref.uri,
                asset_ref.file_type,
                asset_ref.checksum,
                asset_ref.size,
                &asset_ref.role,
                asset_ref.logical_path,
                asset_ref.name,
                asset_ref.created_on,
                asset_ref.upstream_asset_ref_id
            ],
        )?;
        Ok(())
    }

    /// Returns every asset derived from a given asset.
    ///
    /// The relationship is format-independent so sequence, annotation, alignment, and other
    /// assets can discover any number of associated indices. There is no grouping mechanism,
    /// so all downstream assets will be returned and the caller needs to identify which ones
    /// go together.
    pub fn get_derived_assets(
        conn: &GraphConnection,
        upstream_asset_ref_id: &HashId,
        history_ref: Option<&str>,
    ) -> Vec<Self> {
        let table = Self::table_name_with_history_ref(history_ref);
        let query = format!(
            "SELECT * FROM {table} \
             WHERE upstream_asset_ref_id = :upstream_asset_ref_id \
             ORDER BY role, logical_path, id"
        );
        let mut query_params: Vec<(&str, &dyn ToSql)> =
            vec![(":upstream_asset_ref_id", upstream_asset_ref_id)];
        if let Some(history_ref) = history_ref.as_ref() {
            query_params.push((":history_ref", history_ref));
        }
        Self::query(conn, &query, &query_params[..])
    }

    /// Returns assets grouped by commit.
    ///
    /// Commits are in history order with commits without assets having an empty vector.
    pub fn get_assets_by_commit(
        conn: &GraphConnection,
        from_hash: Option<&DoltHashId>,
        to_hash: Option<&DoltHashId>,
        view: AssetView,
    ) -> Result<IndexMap<DoltHashId, Vec<Self>>, QueryError> {
        let (ancestry_boundary, view_query) = match view {
            AssetView::Cumulative => (
                "AND (:from_hash IS NULL OR ancestry.commit_hash <> :from_hash)",
                "SELECT bounded_ancestry.commit_hash, \
                        asset_versions.id, \
                        asset_versions.uri, \
                        asset_versions.file_type, \
                        asset_versions.checksum, \
                        asset_versions.size, \
                        asset_versions.role, \
                        asset_versions.logical_path, \
                        asset_versions.name, \
                        asset_versions.created_on, \
                        asset_versions.upstream_asset_ref_id \
                 FROM bounded_ancestry \
                 LEFT JOIN asset_versions \
                   ON asset_versions.introduction_commit = bounded_ancestry.commit_hash \
                 ORDER BY bounded_ancestry.depth DESC, \
                          asset_versions.id",
            ),
            AssetView::Materialized => (
                "",
                ", /* Keep only assets present at the upper bound, then rank each logical path by \
                      its introduction order in first-parent history. Null paths remain \
                      independent because they are partitioned by asset ID. */ \
                 materialized_assets AS ( \
                     SELECT asset_versions.*, \
                            ROW_NUMBER() OVER ( \
                                PARTITION BY asset_versions.logical_path, \
                                             CASE \
                                                 WHEN asset_versions.logical_path IS NULL \
                                                 THEN asset_versions.id \
                                                 END \
                                ORDER BY asset_versions.introduction_depth, \
                                         asset_versions.id \
                            ) AS materialized_rank \
                     FROM asset_versions \
                     JOIN dolt_at_gen_asset_refs( \
                         COALESCE(:to_hash, dolt_hashof('HEAD')) \
                     ) AS current_assets \
                       ON current_assets.id = asset_versions.id \
                 ) \
                 SELECT bounded_ancestry.commit_hash, \
                        materialized_assets.id, \
                        materialized_assets.uri, \
                        materialized_assets.file_type, \
                        materialized_assets.checksum, \
                        materialized_assets.size, \
                        materialized_assets.role, \
                        materialized_assets.logical_path, \
                        materialized_assets.name, \
                        materialized_assets.created_on, \
                        materialized_assets.upstream_asset_ref_id \
                 FROM bounded_ancestry \
                 LEFT JOIN materialized_assets \
                  ON materialized_assets.introduction_commit = bounded_ancestry.commit_hash \
                  AND materialized_assets.materialized_rank = 1 \
                 ORDER BY bounded_ancestry.depth DESC, \
                          materialized_assets.id",
            ),
        };
        let query = format!(
            "WITH RECURSIVE \
                     /* Walk first-parent history from the upper bound. Cumulative queries stop at \
                        the inclusive lower bound; materialized queries continue to the root so \
                        version precedence comes from commit order rather than wall-clock time. */ \
                     ancestry(commit_hash, depth) AS ( \
                         SELECT COALESCE(:to_hash, dolt_hashof('HEAD')), 0 \
                         UNION ALL \
                         SELECT parents.parent_hash, ancestry.depth + 1 \
                         FROM ancestry \
                         JOIN dolt_commit_ancestors AS parents \
                           ON parents.commit_hash = ancestry.commit_hash \
                          AND parents.parent_index = 0 \
                         WHERE parents.parent_hash IS NOT NULL \
                           {ancestry_boundary} \
                     ), \
                     /* If from_hash is not in the walk, reject the entire range instead of \
                        returning history from an unrelated commit. */ \
                     bounded_ancestry AS ( \
                         SELECT commit_hash, depth \
                         FROM ancestry \
                         WHERE :from_hash IS NULL \
                            OR EXISTS ( \
                                SELECT 1 FROM ancestry \
                                WHERE commit_hash = :from_hash \
                            ) \
                     ), \
                     /* Start Dolt's history walk at the upper bound, join the selected commits to \
                        its repeated table snapshots, keep local files, and collapse each immutable \
                        asset ID. SQLite associates the bare commit hash with the row supplying \
                        MAX(depth), identifying the version's introduction in first-parent history. */ \
                     asset_versions AS ( \
                         SELECT historical_assets.id, \
                                historical_assets.uri, \
                                historical_assets.file_type, \
                                historical_assets.checksum, \
                                historical_assets.size, \
                                historical_assets.role, \
                                historical_assets.logical_path, \
                                historical_assets.name, \
                                historical_assets.created_on, \
                                historical_assets.upstream_asset_ref_id, \
                                bounded_ancestry.commit_hash AS introduction_commit, \
                                MAX(bounded_ancestry.depth) AS introduction_depth \
                         FROM bounded_ancestry \
                         JOIN dolt_history_gen_asset_refs( \
                             COALESCE(:to_hash, dolt_hashof('HEAD')) \
                         ) AS historical_assets \
                           ON historical_assets.commit_hash = bounded_ancestry.commit_hash \
                         WHERE historical_assets.uri LIKE 'file://%' \
                         GROUP BY historical_assets.id \
                     ) \
                     {view_query}"
        );
        let mut statement = conn.prepare(&query)?;
        let rows = statement.query_map(
            named_params! {
                ":from_hash": from_hash,
                ":to_hash": to_hash,
            },
            |row| {
                let id = row.get::<_, Option<HashId>>("id")?;
                let asset = if let Some(id) = id {
                    Some(Self {
                        id,
                        uri: row.get("uri")?,
                        file_type: row.get("file_type")?,
                        checksum: row.get("checksum")?,
                        size: row.get("size")?,
                        role: row.get("role")?,
                        logical_path: row.get("logical_path")?,
                        name: row.get("name")?,
                        created_on: row.get("created_on")?,
                        upstream_asset_ref_id: row.get("upstream_asset_ref_id")?,
                    })
                } else {
                    None
                };
                Ok((row.get::<_, DoltHashId>("commit_hash")?, asset))
            },
        )?;
        let mut assets_by_commit = IndexMap::<DoltHashId, Vec<Self>>::new();
        for row in rows {
            let (commit_hash, asset) = row?;
            let assets = assets_by_commit.entry(commit_hash).or_default();
            if let Some(asset) = asset {
                assets.push(asset);
            }
        }
        Ok(assets_by_commit)
    }

    // Returns an array of assets between two commit ranges for a given view type.
    //
    // Cumulative views include assets that have been replaced by more recent versions.
    // Materialized view is the newest asset (as defined by closest to to_hash).
    fn get_assets_at(
        conn: &GraphConnection,
        from_hash: Option<&DoltHashId>,
        to_hash: Option<&DoltHashId>,
        view: AssetView,
    ) -> Result<Vec<Self>, QueryError> {
        let mut assets = Self::get_assets_by_commit(conn, from_hash, to_hash, view)?
            .into_values()
            .flatten()
            .collect::<Vec<_>>();
        assets.sort_by(|left, right| {
            left.logical_path
                .cmp(&right.logical_path)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(assets)
    }

    /// Returns every local asset reference present in an inclusive commit range.
    ///
    /// This cumulative view retains superseded or subsequently deleted references.
    /// Synchronization code needs those versions to recognize previously managed file contents,
    /// distinguish an intended update from a user edit, and report conflicts correctly. Omitting
    /// `to_hash` uses `HEAD`; omitting `from_hash` walks through the root commit, so omitting both
    /// bounds returns the complete history reachable from `HEAD`.
    pub fn get_cumulative_assets_at(
        conn: &GraphConnection,
        from_hash: Option<&DoltHashId>,
        to_hash: Option<&DoltHashId>,
    ) -> Result<Vec<Self>, QueryError> {
        Self::get_assets_at(conn, from_hash, to_hash, AssetView::Cumulative)
    }

    /// Returns the one-file-per-logical-path workspace view for an inclusive commit range.
    ///
    /// When multiple committed assets share a logical path, the asset introduced by the nearest
    /// first-parent commit to the upper bound represents that path. This collapsed view is for
    /// deciding which files should be materialized, not for history-aware conflict detection.
    /// Omitting `to_hash` uses `HEAD`; omitting `from_hash` walks through the root commit, so
    /// omitting both bounds uses the complete history reachable from `HEAD`.
    pub fn get_materialized_assets_at(
        conn: &GraphConnection,
        from_hash: Option<&DoltHashId>,
        to_hash: Option<&DoltHashId>,
    ) -> Result<Vec<Self>, QueryError> {
        Self::get_assets_at(conn, from_hash, to_hash, AssetView::Materialized)
    }
}

impl Assets {
    /// Returns the materialized asset view at the commit currently named by a branch or ref.
    ///
    /// Resolving the ref before querying gives [`AssetRef::get_materialized_assets_at`] a stable
    /// commit boundary even if the branch later advances. Call
    /// [`AssetRef::get_cumulative_assets_at`] directly when superseded versions are needed for
    /// conflict or update detection.
    pub fn get_branch_assets(
        conn: &GraphConnection,
        branch: &str,
    ) -> Result<HashMap<HashId, AssetRef>, QueryError> {
        let commit_hash = hash_of(conn, branch)?;
        Ok(
            AssetRef::get_materialized_assets_at(conn, None, Some(&commit_hash))?
                .into_iter()
                .map(|asset| (asset.id, asset))
                .collect(),
        )
    }

    pub fn get_annotation_files(
        conn: &GraphConnection,
        history_ref: Option<&str>,
    ) -> Result<Vec<AnnotationFileAssets>, QueryError> {
        let operation_logs_table = OperationLog::table_name_with_history_ref(history_ref);
        let operation_assets_table = OperationAsset::table_name_with_history_ref(history_ref);
        let asset_refs_table = AssetRef::table_name_with_history_ref(history_ref);
        let query = format!(
            "SELECT operation_logs.id AS log_id, \
                    annotation_assets.id AS annotation_id, \
                    annotation_assets.uri AS annotation_uri, \
                    annotation_assets.file_type AS annotation_file_type, \
                    annotation_assets.checksum AS annotation_checksum, \
                    annotation_assets.size AS annotation_size, \
                    annotation_assets.role AS annotation_role, \
                    annotation_assets.logical_path AS annotation_logical_path, \
                    annotation_assets.name AS annotation_name, \
                    annotation_assets.created_on AS annotation_created_on, \
                    annotation_assets.upstream_asset_ref_id AS annotation_upstream_asset_ref_id, \
                    index_assets.id AS index_id, \
                    index_assets.uri AS index_uri, \
                    index_assets.file_type AS index_file_type, \
                    index_assets.checksum AS index_checksum, \
                    index_assets.size AS index_size, \
                    index_assets.role AS index_role, \
                    index_assets.logical_path AS index_logical_path, \
                    index_assets.name AS index_name, \
                    index_assets.created_on AS index_created_on, \
                    index_assets.upstream_asset_ref_id AS index_upstream_asset_ref_id \
             FROM {operation_logs_table} operation_logs \
             JOIN {operation_assets_table} annotation_operation_assets \
               ON annotation_operation_assets.log_id = operation_logs.id \
              AND annotation_operation_assets.role = :annotation_role \
             JOIN {asset_refs_table} annotation_assets \
               ON annotation_assets.id = annotation_operation_assets.asset_ref_id \
             LEFT JOIN {operation_assets_table} index_operation_assets \
               ON index_operation_assets.log_id = operation_logs.id \
              AND index_operation_assets.role = :index_role \
             LEFT JOIN {asset_refs_table} index_assets \
               ON index_assets.id = index_operation_assets.asset_ref_id \
              AND index_assets.upstream_asset_ref_id = annotation_assets.id \
             WHERE operation_logs.operation_kind = :operation_kind \
             ORDER BY operation_logs.created_on, annotation_assets.created_on, \
                      annotation_assets.name"
        );
        let operation_kind = OperationKind::AnnotationFile;
        let annotation_role = AssetRole::Annotation;
        let index_role = AssetRole::AnnotationIndex;
        let mut query_params: Vec<(&str, &dyn ToSql)> = vec![
            (":operation_kind", &operation_kind),
            (":annotation_role", &annotation_role),
            (":index_role", &index_role),
        ];
        if let Some(history_ref) = history_ref.as_ref() {
            query_params.push((":history_ref", history_ref));
        }
        let mut statement = conn.prepare(&query)?;
        let rows = statement.query_map(&query_params[..], |row| {
            let index = match row.get::<_, Option<HashId>>("index_id")? {
                Some(id) => Some(AssetRef {
                    id,
                    uri: row.get("index_uri")?,
                    file_type: row.get("index_file_type")?,
                    checksum: row.get("index_checksum")?,
                    size: row.get("index_size")?,
                    role: row.get("index_role")?,
                    logical_path: row.get("index_logical_path")?,
                    name: row.get("index_name")?,
                    created_on: row.get("index_created_on")?,
                    upstream_asset_ref_id: row.get("index_upstream_asset_ref_id")?,
                }),
                None => None,
            };
            Ok(AnnotationFileAssets {
                log_id: row.get("log_id")?,
                annotation: AssetRef {
                    id: row.get("annotation_id")?,
                    uri: row.get("annotation_uri")?,
                    file_type: row.get("annotation_file_type")?,
                    checksum: row.get("annotation_checksum")?,
                    size: row.get("annotation_size")?,
                    role: row.get("annotation_role")?,
                    logical_path: row.get("annotation_logical_path")?,
                    name: row.get("annotation_name")?,
                    created_on: row.get("annotation_created_on")?,
                    upstream_asset_ref_id: row.get("annotation_upstream_asset_ref_id")?,
                },
                index,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(QueryError::from)
    }
}

impl OperationLog {
    pub fn id_hash(operation_kind: &OperationKind, command: &str, created_on: i64) -> HashId {
        HashId(calculate_hash(&format!(
            "{operation_kind}:{command}:{created_on}"
        )))
    }

    pub fn create(conn: &GraphConnection, operation_log: &OperationLog) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT OR IGNORE INTO gen_operation_log (id, operation_kind, command, created_on) \
             VALUES (?1, ?2, ?3, ?4)",
            params![
                operation_log.id,
                &operation_log.operation_kind,
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
                &operation_asset.role
            ],
        )?;
        Ok(())
    }

    pub fn by_log_id(conn: &GraphConnection, log_id: &HashId) -> Vec<Self> {
        Self::query(
            conn,
            "SELECT * FROM gen_operation_assets WHERE log_id = ?1",
            params![log_id],
        )
    }
}

#[doc(hidden)]
pub struct OpenDalLocation {
    operator: blocking::Operator,
    path: String,
}

struct ChecksumState {
    hasher: Sha256,
    checksum: Option<Sha256Hash>,
    complete: bool,
}

#[derive(Clone)]
pub struct ChecksumHandle {
    state: Arc<Mutex<ChecksumState>>,
}

impl ChecksumHandle {
    /// Returns the content checksum only after the associated reader reaches EOF.
    ///
    /// Consumers use this after useful streaming work, such as copying an asset into storage, so
    /// an interrupted read cannot publish the checksum of only a prefix.
    pub fn checksum(&self) -> Option<Sha256Hash> {
        let state = self.state.lock().unwrap();
        if state.complete { state.checksum } else { None }
    }
}

/// A reader that makes the checksum of a fully consumed stream available through a shared handle.
///
/// The handle lets callers hash bytes as they pass them to their real destination rather than
/// performing a checksum-only read of a potentially large asset.
pub struct ChecksummedReader {
    inner: Box<dyn Read>,
    state: Arc<Mutex<ChecksumState>>,
}

impl ChecksummedReader {
    /// Wraps a reader whose complete sequential contents need to be checksummed.
    pub fn new(inner: impl Read + 'static) -> Self {
        Self {
            inner: Box::new(inner),
            state: Arc::new(Mutex::new(ChecksumState {
                hasher: Sha256::new(),
                checksum: None,
                complete: false,
            })),
        }
    }

    pub fn checksum_handle(&self) -> ChecksumHandle {
        ChecksumHandle {
            state: Arc::clone(&self.state),
        }
    }

    // Publishing happens only at EOF so consumers never treat a partial stream hash as the asset's
    // content identity.
    fn finish(&self) {
        let mut state = self.state.lock().unwrap();
        if state.checksum.is_none() {
            let finalized = state.hasher.clone().finalize();
            state.checksum = Some(Sha256Hash(finalized.into()));
        }
        state.complete = true;
    }
}

impl Read for ChecksummedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes_read = self.inner.read(buf)?;
        if bytes_read == 0 {
            if !buf.is_empty() {
                self.finish();
            }
        } else {
            self.state.lock().unwrap().hasher.update(&buf[..bytes_read]);
        }
        Ok(bytes_read)
    }
}

impl OpenDalLocation {
    fn new_fs(root: &Path, path: &Path) -> Result<Self, opendal::Error> {
        let path = path
            .strip_prefix(Path::new("/"))
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let operator = with_opendal_runtime(|| {
            let builder = services::Fs::default().root(&root.to_string_lossy());
            let op = opendal::Operator::new(builder)?.finish();
            blocking::Operator::new(op)
        })?;

        Ok(Self { operator, path })
    }

    fn from_workspace_path(workspace: &Workspace, path: &Path) -> Result<Self, opendal::Error> {
        let repo_root = workspace.repo_root().map_err(|err| {
            opendal::Error::new(opendal::ErrorKind::Unexpected, "repo_root failed").set_source(err)
        })?;

        if path.is_absolute() {
            Self::new_fs(Path::new("/"), path)
        } else {
            Self::new_fs(&repo_root, path)
        }
    }

    fn from_absolute_path(path: &Path) -> Result<Self, opendal::Error> {
        if !path.is_absolute() {
            return Err(opendal::Error::new(
                opendal::ErrorKind::Unexpected,
                "local path must be absolute",
            ));
        }

        Self::new_fs(Path::new("/"), path)
    }

    fn from_remote_uri(asset_uri: &str) -> Result<blocking::StdReader, FileAdditionError> {
        let url = Url::parse(asset_uri)
            .map_err(|err| FileAdditionError::FileReadError(io::Error::other(err)))?;
        // An OpenDAL operator identifies the storage backend, while its reader identifies an
        // object within that backend. Keep the object path out of the operator configuration.
        let backend_uri = &url[..Position::BeforePath];
        let object_path = url.path().trim_start_matches('/').to_string();
        let anonymous_access_configured = url
            .query_pairs()
            .any(|(key, _)| key.eq_ignore_ascii_case("allow_anonymous"));
        opendal::init_default_registry();

        let open_reader = |allow_anonymous: Option<&str>| {
            // Query parameters are OpenDAL backend options, not part of the remote object key.
            let mut options = url
                .query_pairs()
                .map(|(key, value)| (key.into_owned(), value.into_owned()))
                .collect::<HashMap<_, _>>();
            if let Some(value) = allow_anonymous {
                options.insert("allow_anonymous".to_string(), value.to_string());
            }
            let operator =
                with_opendal_runtime(|| blocking::Operator::from_uri((backend_uri, options)))
                    .map_err(opendal_file_addition_error)?;
            Self {
                operator,
                path: object_path.clone(),
            }
            .reader()
        };

        // Credentials for another AWS account can reject an otherwise public object. Retry the
        // actual OpenDAL read anonymously instead of using request signing as a credential probe.
        match open_reader(None) {
            Ok(reader) => Ok(reader),
            Err(_) if url.scheme().eq_ignore_ascii_case("s3") && !anonymous_access_configured => {
                open_reader(Some("true"))
            }
            Err(error) => Err(error),
        }
    }

    fn reader(self) -> Result<blocking::StdReader, FileAdditionError> {
        self.operator
            .reader(&self.path)
            .map_err(opendal_file_addition_error)?
            .into_std_read(..)
            .map_err(opendal_file_addition_error)
    }
}

pub trait AssetUri {
    fn uri(&self) -> &str;

    fn reader(&self, workspace: &Workspace) -> Result<blocking::StdReader, FileAdditionError>;

    /// Performs the storage work required before recording an asset and returns a verified
    /// checksum when one is available.
    ///
    /// Local assets are retained and hashed during that copy. Remote assets remain lazy unless a
    /// caller already obtained a checksum while streaming them for another purpose.
    fn prepare_asset(
        &self,
        workspace: &Workspace,
        checksum_override: Option<Sha256Hash>,
    ) -> Result<Option<Sha256Hash>, FileAdditionError>;

    fn stored_asset_uri(
        &self,
        workspace: &Workspace,
        checksum: &Sha256Hash,
    ) -> Result<String, FileAdditionError>;

    fn store_file(
        &self,
        file_addition: &FileAddition,
        workspace: &Workspace,
    ) -> Result<(), FileStoreError>;

    fn hashed_filename(&self, checksum: &Sha256Hash) -> String {
        self.asset_filename(checksum)
    }

    fn suffix(&self) -> Option<String> {
        let path = if LocalAssetUri::is_file_uri(self.uri()) {
            LocalAssetUri::path_from_uri(self.uri())?
        } else if LocalAssetUri::has_uri_scheme(self.uri()) {
            Url::parse(self.uri()).ok()?.path().to_string()
        } else {
            self.uri().to_string()
        };

        Path::new(&path)
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|file_name| {
                file_name
                    .split_once('.')
                    .map(|(_, suffix)| suffix.to_string())
            })
    }

    fn generate_file_addition_id(checksum: Option<&Sha256Hash>, asset_uri: &str) -> HashId
    where
        Self: Sized,
    {
        // Checksumless remote assets still need stable database identity. The URI contributes only
        // to that record ID and is never represented as a content checksum.
        let checksum = checksum.map(Sha256Hash::to_string).unwrap_or_default();
        let combined = format!("{checksum};{asset_uri}");
        HashId(calculate_hash(&combined))
    }

    fn asset_filename(&self, checksum: &Sha256Hash) -> String {
        let suffix = self.suffix().unwrap_or_default();
        let suffix = suffix.strip_prefix('.').unwrap_or(&suffix);
        if suffix.is_empty() {
            return checksum.to_string();
        }
        format!("{checksum}.{suffix}")
    }

    fn asset_relative_path(
        workspace: &Workspace,
        asset_uri: &Self,
        checksum: &Sha256Hash,
    ) -> Result<String, FileAdditionError>
    where
        Self: Sized,
    {
        let repo_root = workspace.repo_root()?;
        let asset_path = workspace
            .asset_dir()?
            .join(asset_uri.asset_filename(checksum));

        Ok(asset_path
            .strip_prefix(&repo_root)
            .map_err(|_| FileAdditionError::PathOutsideRepo {
                path: asset_path.clone(),
                repo_root: repo_root.clone(),
            })?
            .to_string_lossy()
            .to_string())
    }
}

impl dyn AssetUri {
    pub fn new(workspace: &Workspace, uri: &str) -> Box<dyn AssetUri> {
        let (scheme, _) = uri.split_once("://").unwrap_or(("file", uri));
        match scheme.to_ascii_lowercase().as_str() {
            "file" => Box::new(
                LocalAssetUri::new_for_workspace(workspace, uri)
                    .expect("failed to construct local asset uri"),
            ),
            _ => Box::new(RemoteAssetUri::new(uri)),
        }
    }

    pub fn from_uri(uri: &str) -> Box<dyn AssetUri> {
        let (scheme, _) = uri.split_once("://").unwrap_or(("file", uri));
        match scheme.to_ascii_lowercase().as_str() {
            "file" => Box::new(LocalAssetUri::new(uri)),
            _ => Box::new(RemoteAssetUri::new(uri)),
        }
    }
}

/// Chooses where an asset should be materialized without inventing a content identity.
///
/// Explicit logical paths are usable without a checksum. Content-addressed materialization under
/// `.gen/assets` requires a verified checksum.
pub fn materialization_destination_path(
    workspace: &Workspace,
    asset_uri: &str,
    checksum: Option<&Sha256Hash>,
    logical_path: Option<&str>,
) -> Result<PathBuf, FileAdditionError> {
    if let Some(logical_path) = logical_path.filter(|path| !path.is_empty()) {
        return LocalAssetUri::repo_relative_destination_path(workspace, logical_path);
    }

    let checksum = checksum.ok_or_else(|| {
        FileAdditionError::FileReadError(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("cannot materialize asset without checksum: {asset_uri}"),
        ))
    })?;
    Ok(workspace
        .asset_dir()?
        .join(<dyn AssetUri>::from_uri(asset_uri).hashed_filename(checksum)))
}

pub struct LocalAssetUri {
    asset_uri: String,
    source_path: Option<PathBuf>,
    workspace_root: Option<PathBuf>,
    read_file: Option<blocking::StdReader>,
    write_file: Option<opendal::blocking::StdWriter>,
}

impl AssetUri for LocalAssetUri {
    fn uri(&self) -> &str {
        &self.asset_uri
    }

    fn reader(&self, workspace: &Workspace) -> Result<blocking::StdReader, FileAdditionError> {
        let source_file_path = self.resolved_source_file_path(workspace)?;
        OpenDalLocation::from_workspace_path(workspace, &source_file_path)
            .map_err(opendal_file_addition_error)?
            .reader()
    }

    fn prepare_asset(
        &self,
        workspace: &Workspace,
        checksum_override: Option<Sha256Hash>,
    ) -> Result<Option<Sha256Hash>, FileAdditionError> {
        // Operation creation must retain local bytes, so this copy is also the useful point at
        // which to verify or calculate their checksum.
        self.stage_asset_copy(workspace, checksum_override)
            .map(Some)
    }

    fn stored_asset_uri(
        &self,
        workspace: &Workspace,
        checksum: &Sha256Hash,
    ) -> Result<String, FileAdditionError> {
        let source_file_path = self.resolved_source_file_path(workspace)?;
        let repo_root = Self::canonicalize_or_normalize(&workspace.repo_root()?);
        if !source_file_path.starts_with(&repo_root) {
            return Ok(self.asset_uri.clone());
        }

        let source_asset_uri = Self::new(&source_file_path.to_string_lossy());
        let relative_file_path = Self::asset_relative_path(workspace, &source_asset_uri, checksum)?;
        Ok(Self::asset_uri(&relative_file_path))
    }

    fn store_file(
        &self,
        file_addition: &FileAddition,
        workspace: &Workspace,
    ) -> Result<(), FileStoreError> {
        Self::store_file(file_addition, workspace)
    }
}

impl LocalAssetUri {
    pub const OUTSIDE_ROOT_DIRECTORY: &'static str = ".gen/outside_root";
    pub const SCHEME: &'static str = "file://";

    pub fn new(path_or_uri: &str) -> Self {
        Self {
            asset_uri: Self::asset_uri(path_or_uri),
            source_path: None,
            workspace_root: None,
            read_file: None,
            write_file: None,
        }
    }

    pub fn new_for_workspace(
        workspace: &Workspace,
        path_or_uri: &str,
    ) -> Result<Self, FileAdditionError> {
        let source_path = if Self::is_file_uri(path_or_uri) {
            Self::resolve_source_path(workspace, path_or_uri)?
        } else {
            Self::resolve_input_source_path(workspace, path_or_uri)?
        };
        let asset_uri = Self::asset_uri(&Self::logical_file_path(workspace, &source_path)?);
        Ok(Self {
            asset_uri,
            source_path: Some(source_path),
            workspace_root: Some(workspace.repo_root()?),
            read_file: None,
            write_file: None,
        })
    }

    pub fn is_file_uri(asset_uri: &str) -> bool {
        asset_uri.starts_with(Self::SCHEME)
    }

    pub fn has_uri_scheme(path_or_uri: &str) -> bool {
        path_or_uri.contains("://")
    }

    pub fn is_local_path_or_file_uri(path_or_uri: &str) -> bool {
        Self::is_file_uri(path_or_uri) || !Self::has_uri_scheme(path_or_uri)
    }

    pub fn asset_uri(path: &str) -> String {
        let path = path.strip_prefix('/').unwrap_or(path);
        if Self::is_file_uri(path) {
            path.to_string()
        } else {
            format!("{}{path}", Self::SCHEME)
        }
    }

    pub fn path_from_uri(asset_uri: &str) -> Option<String> {
        asset_uri
            .strip_prefix(Self::SCHEME)
            .map(ToString::to_string)
    }

    pub fn operation_file_path(
        workspace: &Workspace,
        path_or_uri: &str,
        checksum: &Sha256Hash,
    ) -> Result<String, FileAdditionError> {
        let source_path = if Self::is_file_uri(path_or_uri) {
            Self::resolve_source_path(workspace, path_or_uri)?
        } else {
            Self::resolve_input_source_path(workspace, path_or_uri)?
        };
        Self::stored_relative_path(workspace, &source_path, checksum)
    }

    /// Checks whether a relative path is in the .gen/assets directory of a workspace.
    pub fn is_asset_relative_path(
        workspace: &Workspace,
        relative_path: &str,
    ) -> Result<bool, FileAdditionError> {
        let repo_root = workspace.repo_root()?;
        let asset_dir = workspace.asset_dir()?;
        let asset_relative_dir =
            asset_dir
                .strip_prefix(&repo_root)
                .map_err(|_| FileAdditionError::PathOutsideRepo {
                    path: asset_dir.clone(),
                    repo_root: repo_root.clone(),
                })?;

        let candidate = if relative_path.is_empty() {
            PathBuf::new()
        } else {
            Self::sanitize_relative_path(Path::new(relative_path))?
        };

        Ok(candidate.starts_with(asset_relative_dir))
    }

    /// Given a workspace and a relative path, try to safely give a destination path to copy
    /// the asset into
    pub fn repo_relative_destination_path(
        workspace: &Workspace,
        relative_path: &str,
    ) -> Result<PathBuf, FileAdditionError> {
        let sanitized = if relative_path.is_empty() {
            PathBuf::new()
        } else {
            Self::sanitize_relative_path(Path::new(relative_path))?
        };

        Ok(workspace.repo_root()?.join(sanitized))
    }

    fn file_path(&self) -> &str {
        self.asset_uri
            .strip_prefix(Self::SCHEME)
            .unwrap_or(&self.asset_uri)
    }

    fn io_path(&self) -> PathBuf {
        self.source_path.clone().unwrap_or_else(|| {
            let file_path = PathBuf::from(self.file_path());
            if file_path.is_absolute() {
                file_path
            } else if let Some(workspace_root) = &self.workspace_root {
                workspace_root.join(file_path)
            } else {
                file_path
            }
        })
    }

    pub fn close_write(&mut self) -> io::Result<()> {
        if let Some(mut file) = self.write_file.take() {
            file.close()?;
        }
        Ok(())
    }

    fn resolved_source_file_path(
        &self,
        workspace: &Workspace,
    ) -> Result<PathBuf, FileAdditionError> {
        self.source_path
            .clone()
            .map(Ok)
            .unwrap_or_else(|| Self::resolve_source_path(workspace, &self.asset_uri))
    }

    fn resolve_input_source_path(
        workspace: &Workspace,
        file_path_or_uri: &str,
    ) -> Result<PathBuf, FileAdditionError> {
        if !Self::is_local_path_or_file_uri(file_path_or_uri) {
            return Err(FileAdditionError::FileReadError(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported non-local uri: {file_path_or_uri}"),
            )));
        }

        let file_path =
            Self::path_from_uri(file_path_or_uri).unwrap_or_else(|| file_path_or_uri.to_string());
        if file_path.is_empty() {
            return Ok(PathBuf::new());
        }

        let provided_path = Path::new(&file_path);
        if provided_path.is_absolute() {
            Ok(Self::canonicalize_or_normalize(provided_path))
        } else {
            Ok(Self::canonicalize_or_normalize(
                &workspace.repo_root()?.join(provided_path),
            ))
        }
    }

    fn resolve_source_path(
        workspace: &Workspace,
        asset_uri: &str,
    ) -> Result<PathBuf, FileAdditionError> {
        let file_path = Self::path_from_uri(asset_uri).unwrap_or_else(|| asset_uri.to_string());
        if file_path.is_empty() {
            return Ok(PathBuf::new());
        }

        let provided_path = Path::new(&file_path);
        if provided_path.is_absolute() {
            return Err(FileAdditionError::PathOutsideRepo {
                path: provided_path.to_path_buf(),
                repo_root: workspace.repo_root()?,
            });
        }

        let repo_root = Self::canonicalize_or_normalize(&workspace.repo_root()?);
        let joined_path = workspace
            .repo_root()?
            .join(Self::sanitize_relative_path(provided_path)?);
        let normalized_path = Self::canonicalize_or_normalize(&joined_path);
        if !normalized_path.starts_with(&repo_root) {
            return Err(FileAdditionError::PathOutsideRepo {
                path: normalized_path,
                repo_root,
            });
        }
        Ok(normalized_path)
    }

    fn logical_file_path(
        workspace: &Workspace,
        source_path: &Path,
    ) -> Result<String, FileAdditionError> {
        if source_path.as_os_str().is_empty() {
            return Ok(String::new());
        }

        let repo_root = Self::canonicalize_or_normalize(&workspace.repo_root()?);
        let source_path = Self::canonicalize_or_normalize(source_path);
        if let Ok(relative_path) = source_path.strip_prefix(&repo_root) {
            return Ok(relative_path.to_string_lossy().to_string());
        }

        source_path
            .file_name()
            .map(|file_name| {
                format!(
                    "{}/{}",
                    Self::OUTSIDE_ROOT_DIRECTORY,
                    file_name.to_string_lossy()
                )
            })
            .ok_or_else(|| {
                FileAdditionError::FileReadError(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("asset path has no filename: {}", source_path.display()),
                ))
            })
    }

    fn stored_relative_path(
        workspace: &Workspace,
        source_path: &Path,
        checksum: &Sha256Hash,
    ) -> Result<String, FileAdditionError> {
        if source_path.as_os_str().is_empty() {
            return Ok(String::new());
        }
        let repo_root = Self::canonicalize_or_normalize(&workspace.repo_root()?);
        let source_path = Self::canonicalize_or_normalize(source_path);

        if source_path.starts_with(&repo_root) {
            return Ok(source_path
                .strip_prefix(&repo_root)
                .map_err(|_| FileAdditionError::PathOutsideRepo {
                    path: source_path.to_path_buf(),
                    repo_root: repo_root.clone(),
                })?
                .to_string_lossy()
                .to_string());
        }

        if source_path.is_absolute() {
            let source_asset_uri = Self::new(&source_path.to_string_lossy());
            return Self::asset_relative_path(workspace, &source_asset_uri, checksum);
        }

        Err(FileAdditionError::FileReadError(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unable to resolve local asset path: {}",
                source_path.display()
            ),
        )))
    }

    // Streams a local asset into content-addressed storage while computing its checksum. Combining
    // those jobs keeps large files to a single pass and leaves no checksum-only temporary output.
    fn stage_asset_copy(
        &self,
        workspace: &Workspace,
        checksum_override: Option<Sha256Hash>,
    ) -> Result<Sha256Hash, FileAdditionError> {
        let asset_dir = workspace.asset_dir()?;
        fs::create_dir_all(&asset_dir).map_err(FileAdditionError::FileReadError)?;
        if let Some(checksum) = checksum_override {
            let asset_path = asset_dir.join(self.asset_filename(&checksum));
            if asset_path.exists() {
                return Ok(checksum);
            }
        }

        let mut reader = ChecksummedReader::new(self.reader(workspace)?);
        let checksum_handle = reader.checksum_handle();
        let mut staged_file =
            NamedTempFile::new_in(&asset_dir).map_err(FileAdditionError::FileReadError)?;
        io::copy(&mut reader, &mut staged_file).map_err(FileAdditionError::FileReadError)?;
        staged_file
            .flush()
            .map_err(FileAdditionError::FileReadError)?;
        let checksum = checksum_handle.checksum().ok_or_else(|| {
            FileAdditionError::ChecksumError(format!(
                "local asset stream did not reach EOF: {}",
                self.uri()
            ))
        })?;
        if let Some(expected_checksum) = checksum_override
            && checksum != expected_checksum
        {
            return Err(FileAdditionError::ChecksumError(format!(
                "local asset checksum does not match the provided checksum: {}",
                self.uri()
            )));
        }

        let asset_path = asset_dir.join(self.asset_filename(&checksum));
        match staged_file.persist_noclobber(asset_path) {
            Ok(_) => {}
            Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(FileAdditionError::FileReadError(error.error)),
        }
        Ok(checksum)
    }

    fn sanitize_relative_path(path: &Path) -> Result<PathBuf, FileAdditionError> {
        let mut sanitized = PathBuf::new();
        for component in path.components() {
            match component {
                Component::Normal(part) => sanitized.push(part),
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(FileAdditionError::FileReadError(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        format!("path escapes workspace root: {}", path.display()),
                    )));
                }
            }
        }
        Ok(sanitized)
    }

    fn canonicalize_or_normalize(path: &Path) -> PathBuf {
        let normalized_path = fs::canonicalize(path).unwrap_or_else(|_| Self::normalize_path(path));
        // Unfortunately, the path we get back from Mac OS X sometimes starts
        // with /private even if the input path doesn't, which causes problems
        // for our path comparisons. Strip it out if it's there.
        let cleaned_path = normalized_path
            .as_path()
            .strip_prefix("/private")
            .unwrap_or(&normalized_path);
        Path::new("/").join(cleaned_path)
    }

    fn normalize_path(path: &Path) -> PathBuf {
        let mut normalized = PathBuf::new();
        for component in path.components() {
            match component {
                Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
                Component::RootDir => normalized.push(Path::new("/")),
                Component::CurDir => {}
                Component::ParentDir => {
                    normalized.pop();
                }
                Component::Normal(part) => normalized.push(part),
            }
        }
        normalized
    }

    pub fn store_file(
        file_addition: &FileAddition,
        workspace: &Workspace,
    ) -> Result<(), FileStoreError> {
        let checksum = file_addition.checksum.ok_or_else(|| {
            FileStoreError::IoError(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "cannot store local asset without checksum: {}",
                    file_addition.asset_uri
                ),
            ))
        })?;
        let asset_uri = Self::new(&file_addition.asset_uri);
        let asset_path = workspace
            .asset_dir()?
            .join(asset_uri.asset_filename(&checksum));
        if asset_path.exists() {
            return Ok(());
        }

        asset_uri
            .stage_asset_copy(workspace, Some(checksum))
            .map(|_| ())
            .map_err(|error| match error {
                FileAdditionError::ConfigError(error) => FileStoreError::ConfigError(error),
                FileAdditionError::FileReadError(error) => FileStoreError::IoError(error),
                other => FileStoreError::IoError(io::Error::other(other)),
            })
    }
}

impl Read for LocalAssetUri {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.read_file.is_none() {
            let io_path = self.io_path();
            self.read_file = Some(
                OpenDalLocation::from_absolute_path(&io_path)
                    .map_err(io::Error::other)?
                    .reader()
                    .map_err(io::Error::other)?,
            );
        }
        self.read_file.as_mut().unwrap().read(buf)
    }
}

impl Write for LocalAssetUri {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.write_file.is_none() {
            let io_path = self.io_path();
            let location =
                OpenDalLocation::from_absolute_path(&io_path).map_err(io::Error::other)?;
            self.write_file = Some(
                location
                    .operator
                    .writer(&location.path)
                    .map_err(io::Error::other)?
                    .into_std_write(),
            );
        }
        self.write_file.as_mut().unwrap().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(file) = self.write_file.as_mut() {
            file.flush()
        } else {
            Ok(())
        }
    }
}

impl Drop for LocalAssetUri {
    fn drop(&mut self) {
        let _ = self.close_write();
    }
}

pub struct RemoteAssetUri {
    asset_uri: String,
}

impl AssetUri for RemoteAssetUri {
    fn uri(&self) -> &str {
        &self.asset_uri
    }

    fn reader(&self, _workspace: &Workspace) -> Result<blocking::StdReader, FileAdditionError> {
        OpenDalLocation::from_remote_uri(&self.asset_uri)
    }

    fn prepare_asset(
        &self,
        _workspace: &Workspace,
        checksum_override: Option<Sha256Hash>,
    ) -> Result<Option<Sha256Hash>, FileAdditionError> {
        // Recording a remote URI must not require network access or credentials. A caller can pass
        // a checksum learned during useful streaming work; otherwise it remains unknown.
        Ok(checksum_override)
    }

    fn stored_asset_uri(
        &self,
        _workspace: &Workspace,
        _checksum: &Sha256Hash,
    ) -> Result<String, FileAdditionError> {
        Ok(self.asset_uri.clone())
    }

    fn store_file(
        &self,
        _file_addition: &FileAddition,
        _workspace: &Workspace,
    ) -> Result<(), FileStoreError> {
        Ok(())
    }
}

impl RemoteAssetUri {
    pub fn new(asset_uri: &str) -> Self {
        Self {
            asset_uri: asset_uri.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::{Read, Seek, SeekFrom, Write},
        net::TcpListener,
        path::PathBuf,
        thread,
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::{
        history::dolt::commit_all,
        operations::{calculate_file_checksum, calculate_reader_checksum},
        test_helpers::setup_gen,
        traits::Query,
    };

    #[test]
    fn test_asset_reference_tables_round_trip() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let asset_ref = AssetRef {
            id: HashId::convert_str("asset-ref"),
            uri: "s3://bucket/reference.fa".to_string(),
            file_type: "fasta".to_string(),
            checksum: Some(Sha256Hash::convert_str("checksum")),
            size: Some(1024),
            role: AssetRole::Input,
            logical_path: Some("refs/reference.fa".to_string()),
            name: Some("reference.fa".to_string()),
            created_on: 1,
            upstream_asset_ref_id: None,
        };
        let operation_log = OperationLog {
            id: HashId::convert_str("log"),
            operation_kind: OperationKind::Other("import".to_string()),
            command: "gen import fasta".to_string(),
            created_on: 1,
        };
        let operation_asset = OperationAsset {
            log_id: operation_log.id,
            asset_ref_id: asset_ref.id,
            role: AssetRole::Input,
        };

        OperationLog::create(conn, &operation_log).expect("should insert operation log");
        AssetRef::create(conn, &asset_ref).expect("should insert asset ref");
        OperationAsset::create(conn, &operation_asset).expect("should insert operation asset");

        let asset_refs = AssetRef::all(conn);
        let operation_logs = OperationLog::all(conn);
        let operation_assets = OperationAsset::all(conn);

        assert_eq!(asset_refs, vec![asset_ref]);
        assert_eq!(operation_logs, vec![operation_log]);
        assert_eq!(operation_assets, vec![operation_asset]);
    }

    #[test]
    fn test_asset_reference_supports_multiple_derived_assets() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let upstream = AssetRef {
            id: HashId::convert_str("sequence-asset"),
            uri: "file://reference.fa.bgz".to_string(),
            file_type: "fasta".to_string(),
            checksum: Some(Sha256Hash::convert_str("sequence-checksum")),
            size: None,
            role: AssetRole::Input,
            logical_path: Some("reference.fa.bgz".to_string()),
            name: Some("reference.fa.bgz".to_string()),
            created_on: 1,
            upstream_asset_ref_id: None,
        };
        let first_index = AssetRef {
            id: HashId::convert_str("first-index"),
            uri: "file://reference.fa.bgz.fai".to_string(),
            file_type: "none".to_string(),
            checksum: Some(Sha256Hash::convert_str("first-index-checksum")),
            size: None,
            role: AssetRole::Other("index".to_string()),
            logical_path: Some("reference.fa.bgz.fai".to_string()),
            name: Some("reference.fa.bgz.fai".to_string()),
            created_on: 1,
            upstream_asset_ref_id: Some(upstream.id),
        };
        let second_index = AssetRef {
            id: HashId::convert_str("second-index"),
            uri: "file://reference.fa.bgz.gzi".to_string(),
            checksum: Some(Sha256Hash::convert_str("second-index-checksum")),
            logical_path: Some("reference.fa.bgz.gzi".to_string()),
            name: Some("reference.fa.bgz.gzi".to_string()),
            ..first_index.clone()
        };
        AssetRef::create(conn, &upstream).expect("should insert upstream asset");
        AssetRef::create(conn, &first_index).expect("should insert first derived asset");
        AssetRef::create(conn, &second_index).expect("should insert second derived asset");
        let commit = commit_all(conn, "add derived assets").expect("should commit derived assets");

        assert_eq!(
            AssetRef::get_derived_assets(conn, &upstream.id, None),
            vec![first_index.clone(), second_index.clone()],
            "current lookup should return every derived asset"
        );
        assert_eq!(
            AssetRef::get_derived_assets(conn, &upstream.id, Some(&commit.to_string())),
            vec![first_index, second_index],
            "historical lookup should return every derived asset"
        );
    }

    #[test]
    fn test_get_annotation_files_pairs_annotation_and_index_in_one_query() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let log = OperationLog {
            id: HashId::convert_str("annotation-log"),
            operation_kind: OperationKind::AnnotationFile,
            command: "add annotation".to_string(),
            created_on: 1,
        };
        let annotation = AssetRef {
            id: HashId::convert_str("annotation-asset"),
            uri: "file://annotation.gff".to_string(),
            file_type: "gff3".to_string(),
            checksum: Some(Sha256Hash::convert_str("annotation-checksum")),
            size: None,
            role: AssetRole::Annotation,
            logical_path: Some("annotation.gff".to_string()),
            name: Some("genes".to_string()),
            created_on: 1,
            upstream_asset_ref_id: None,
        };
        let index = AssetRef {
            id: HashId::convert_str("annotation-index-asset"),
            uri: "file://annotation.gff.tbi".to_string(),
            file_type: "tabix".to_string(),
            checksum: Some(Sha256Hash::convert_str("annotation-index-checksum")),
            size: None,
            role: AssetRole::AnnotationIndex,
            logical_path: Some("annotation.gff.tbi".to_string()),
            name: Some("genes".to_string()),
            created_on: 1,
            upstream_asset_ref_id: Some(annotation.id),
        };
        OperationLog::create(conn, &log).expect("should insert annotation log");
        AssetRef::create(conn, &annotation).expect("should insert annotation asset");
        AssetRef::create(conn, &index).expect("should insert annotation index asset");
        OperationAsset::create(
            conn,
            &OperationAsset {
                log_id: log.id,
                asset_ref_id: annotation.id,
                role: AssetRole::Annotation,
            },
        )
        .expect("should associate annotation asset");
        OperationAsset::create(
            conn,
            &OperationAsset {
                log_id: log.id,
                asset_ref_id: index.id,
                role: AssetRole::AnnotationIndex,
            },
        )
        .expect("should associate annotation index asset");
        let commit = commit_all(conn, "add annotation assets").expect("should commit assets");

        let expected = vec![AnnotationFileAssets {
            log_id: log.id,
            annotation,
            index: Some(index),
        }];
        assert_eq!(
            Assets::get_annotation_files(conn, None).expect("should query current annotations"),
            expected
        );
        assert_eq!(
            Assets::get_annotation_files(conn, Some(&commit.to_string()))
                .expect("should query historical annotations"),
            expected
        );
    }

    mod asset_history {
        use gen_core::{DoltHashId, HashId, Sha256Hash};

        use super::{AssetRef, AssetRole, AssetView};
        use crate::{
            db::{DbContext, GraphConnection},
            history::dolt::{checkout, commit_all, create_branch},
            test_helpers::setup_gen,
        };

        /// Builds a linear history where `alpha.fa` gains a replacement, its superseded reference
        /// is later deleted, and `beta.fa` is added at `HEAD`. A persistent `zeta.fa` exercises
        /// unchanged assets, while an external URI verifies that only local files are returned. The
        /// replacement has an older `created_on` value so materialization must use commit order.
        /// The saved commit hashes let each test isolate materialized, cumulative, and bounded
        /// history behavior without repeating the setup.
        struct AssetHistoryFixture {
            context: DbContext,
            first_alpha: AssetRef,
            second_alpha: AssetRef,
            beta: AssetRef,
            zeta: AssetRef,
            first_commit: DoltHashId,
            second_commit: DoltHashId,
            deletion_commit: DoltHashId,
            latest_commit: DoltHashId,
        }

        impl AssetHistoryFixture {
            fn new() -> Self {
                let context = setup_gen();
                let conn = context.graph().conn();
                let first_alpha = AssetRef {
                    id: HashId::convert_str("first-alpha"),
                    uri: "file://assets/first-alpha.fa".to_string(),
                    file_type: "fasta".to_string(),
                    checksum: Some(Sha256Hash::convert_str("first-alpha-checksum")),
                    size: Some(4),
                    role: AssetRole::Input,
                    logical_path: Some("alpha.fa".to_string()),
                    name: Some("alpha.fa".to_string()),
                    created_on: 1,
                    upstream_asset_ref_id: None,
                };
                let zeta = AssetRef {
                    id: HashId::convert_str("zeta"),
                    uri: "file://assets/zeta.fa".to_string(),
                    checksum: Some(Sha256Hash::convert_str("zeta-checksum")),
                    logical_path: Some("zeta.fa".to_string()),
                    name: Some("zeta.fa".to_string()),
                    ..first_alpha.clone()
                };
                let remote_asset = AssetRef {
                    id: HashId::convert_str("remote-asset"),
                    uri: "https://example.com/reference.fa".to_string(),
                    logical_path: Some("remote.fa".to_string()),
                    name: Some("remote.fa".to_string()),
                    ..first_alpha.clone()
                };
                AssetRef::create(conn, &zeta).expect("should insert zeta asset");
                AssetRef::create(conn, &first_alpha).expect("should insert first alpha asset");
                AssetRef::create(conn, &remote_asset).expect("should insert remote asset");
                let first_commit =
                    commit_all(conn, "add first assets").expect("should commit first assets");

                let second_alpha = AssetRef {
                    id: HashId::convert_str("second-alpha"),
                    uri: "file://assets/second-alpha.fa".to_string(),
                    checksum: Some(Sha256Hash::convert_str("second-alpha-checksum")),
                    created_on: 0,
                    ..first_alpha.clone()
                };
                AssetRef::create(conn, &second_alpha).expect("should insert second alpha asset");
                let second_commit = commit_all(conn, "replace alpha asset")
                    .expect("should commit replacement asset");

                conn.execute("DELETE FROM gen_asset_refs WHERE id = ?1", [first_alpha.id])
                    .expect("should delete superseded alpha asset");
                let deletion_commit = commit_all(conn, "delete superseded asset")
                    .expect("should commit asset deletion");

                let beta = AssetRef {
                    id: HashId::convert_str("beta"),
                    uri: "file://assets/beta.fa".to_string(),
                    checksum: Some(Sha256Hash::convert_str("beta-checksum")),
                    logical_path: Some("beta.fa".to_string()),
                    name: Some("beta.fa".to_string()),
                    created_on: 3,
                    ..first_alpha.clone()
                };
                AssetRef::create(conn, &beta).expect("should insert beta asset");
                let latest_commit =
                    commit_all(conn, "add later beta asset").expect("should commit later asset");

                Self {
                    context,
                    first_alpha,
                    second_alpha,
                    beta,
                    zeta,
                    first_commit,
                    second_commit,
                    deletion_commit,
                    latest_commit,
                }
            }

            fn conn(&self) -> &GraphConnection {
                self.context.graph().conn()
            }
        }

        fn sorted_assets(mut assets: Vec<AssetRef>) -> Vec<AssetRef> {
            assets.sort_by(|left, right| {
                left.logical_path
                    .cmp(&right.logical_path)
                    .then_with(|| left.id.cmp(&right.id))
            });
            assets
        }

        #[test]
        fn test_materialized_assets_at_selects_latest_version_per_path() {
            let fixture = AssetHistoryFixture::new();

            assert_eq!(
                AssetRef::get_materialized_assets_at(
                    fixture.conn(),
                    None,
                    Some(&fixture.second_commit),
                )
                .expect("should materialize second commit"),
                vec![fixture.second_alpha.clone(), fixture.zeta.clone()]
            );
        }

        #[test]
        fn test_materialized_assets_at_uses_commit_order_in_a_bounded_range() {
            let fixture = AssetHistoryFixture::new();

            assert_eq!(
                AssetRef::get_materialized_assets_at(
                    fixture.conn(),
                    Some(&fixture.second_commit),
                    Some(&fixture.second_commit),
                )
                .expect("should materialize using commit order"),
                vec![fixture.second_alpha.clone(), fixture.zeta.clone()]
            );
        }

        #[test]
        fn test_materialized_assets_at_excludes_deleted_versions() {
            let fixture = AssetHistoryFixture::new();

            assert_eq!(
                AssetRef::get_materialized_assets_at(
                    fixture.conn(),
                    None,
                    Some(&fixture.deletion_commit),
                )
                .expect("should materialize deletion commit"),
                vec![fixture.second_alpha.clone(), fixture.zeta.clone()]
            );
        }

        #[test]
        fn test_materialized_assets_at_reads_non_current_branch_by_hash() {
            // Test that when querying assets, the join will work when querying for commits that are not
            // on the main branch.
            let context = setup_gen();
            let conn = context.graph().conn();
            let main_asset = AssetRef {
                id: HashId::convert_str("main-asset"),
                uri: "file://assets/main.fa".to_string(),
                file_type: "fasta".to_string(),
                checksum: Some(Sha256Hash::convert_str("main-checksum")),
                size: Some(4),
                role: AssetRole::Input,
                logical_path: Some("reference.fa".to_string()),
                name: Some("reference.fa".to_string()),
                created_on: 1,
                upstream_asset_ref_id: None,
            };
            AssetRef::create(conn, &main_asset).expect("should insert main asset");
            commit_all(conn, "add main asset").expect("should commit main asset");
            create_branch(conn, "feature").expect("should create feature branch");
            checkout(conn, "feature").expect("should checkout feature branch");

            let feature_asset = AssetRef {
                id: HashId::convert_str("feature-asset"),
                uri: "file://assets/feature.fa".to_string(),
                checksum: Some(Sha256Hash::convert_str("feature-checksum")),
                created_on: 2,
                ..main_asset
            };
            AssetRef::create(conn, &feature_asset).expect("should insert feature asset");
            let feature_commit =
                commit_all(conn, "add feature asset").expect("should commit feature asset");
            checkout(conn, "main").expect("should restore main branch");

            assert_eq!(
                AssetRef::get_materialized_assets_at(conn, None, Some(&feature_commit))
                    .expect("should read assets from the non-current feature commit"),
                vec![feature_asset],
                "the history range should start at the requested feature commit"
            );
        }

        #[test]
        fn test_cumulative_assets_at_retains_superseded_and_deleted_versions() {
            let fixture = AssetHistoryFixture::new();
            let expected = sorted_assets(vec![
                fixture.first_alpha.clone(),
                fixture.second_alpha.clone(),
                fixture.zeta.clone(),
            ]);

            assert_eq!(
                AssetRef::get_cumulative_assets_at(
                    fixture.conn(),
                    None,
                    Some(&fixture.deletion_commit),
                )
                .expect("should read cumulative assets at deletion commit"),
                expected
            );
        }

        #[test]
        fn test_asset_queries_respect_commit_range() {
            let fixture = AssetHistoryFixture::new();
            let expected = vec![
                fixture.second_alpha.clone(),
                fixture.beta.clone(),
                fixture.zeta.clone(),
            ];

            assert_eq!(
                AssetRef::get_materialized_assets_at(
                    fixture.conn(),
                    Some(&fixture.deletion_commit),
                    Some(&fixture.latest_commit),
                )
                .expect("should materialize bounded range"),
                expected
            );
            assert_eq!(
                AssetRef::get_cumulative_assets_at(
                    fixture.conn(),
                    Some(&fixture.deletion_commit),
                    Some(&fixture.latest_commit),
                )
                .expect("should accumulate bounded range"),
                sorted_assets(vec![
                    fixture.second_alpha.clone(),
                    fixture.beta.clone(),
                    fixture.zeta.clone(),
                ])
            );
        }

        #[test]
        fn test_asset_queries_default_missing_upper_bound_to_head() {
            let fixture = AssetHistoryFixture::new();

            assert_eq!(
                AssetRef::get_materialized_assets_at(
                    fixture.conn(),
                    Some(&fixture.deletion_commit),
                    None,
                )
                .expect("should materialize through HEAD"),
                vec![
                    fixture.second_alpha.clone(),
                    fixture.beta.clone(),
                    fixture.zeta.clone(),
                ]
            );
        }

        #[test]
        fn test_cumulative_assets_without_bounds_reads_full_head_history() {
            let fixture = AssetHistoryFixture::new();
            let expected = sorted_assets(vec![
                fixture.first_alpha.clone(),
                fixture.second_alpha.clone(),
                fixture.beta.clone(),
                fixture.zeta.clone(),
            ]);

            assert_eq!(
                AssetRef::get_cumulative_assets_at(fixture.conn(), None, None)
                    .expect("should read full history through HEAD"),
                expected
            );
        }

        #[test]
        fn test_assets_by_commit_orders_commits_and_preserves_empty_entries() {
            let fixture = AssetHistoryFixture::new();
            let assets_by_commit = AssetRef::get_assets_by_commit(
                fixture.conn(),
                Some(&fixture.first_commit),
                Some(&fixture.latest_commit),
                AssetView::Cumulative,
            )
            .expect("should group assets by introduction commit");
            let assets = sorted_assets(
                assets_by_commit
                    .values()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>(),
            );

            assert_eq!(
                assets,
                AssetRef::get_cumulative_assets_at(
                    fixture.conn(),
                    Some(&fixture.first_commit),
                    Some(&fixture.latest_commit),
                )
                .expect("should preserve the cumulative asset-only view")
            );
            assert_eq!(
                assets_by_commit.keys().copied().collect::<Vec<_>>(),
                vec![
                    fixture.first_commit,
                    fixture.second_commit,
                    fixture.deletion_commit,
                    fixture.latest_commit,
                ]
            );
            assert_eq!(
                assets_by_commit
                    .get(&fixture.second_commit)
                    .expect("should contain second commit")
                    .as_slice(),
                core::slice::from_ref(&fixture.second_alpha)
            );
            assert!(
                assets_by_commit
                    .get(&fixture.deletion_commit)
                    .expect("should contain deletion commit")
                    .is_empty(),
                "commits without introduced assets should retain empty entries"
            );
        }
    }

    #[test]
    fn test_get_branch_assets_uses_materialized_assets_at_branch_head() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let first_asset = AssetRef {
            id: HashId::convert_str("first-asset"),
            uri: "file://assets/first.fa".to_string(),
            file_type: "fasta".to_string(),
            checksum: Some(Sha256Hash::convert_str("first-checksum")),
            size: Some(4),
            role: AssetRole::Input,
            logical_path: Some("reference.fa".to_string()),
            name: Some("reference.fa".to_string()),
            created_on: 1,
            upstream_asset_ref_id: None,
        };
        AssetRef::create(conn, &first_asset).expect("should insert first asset");
        commit_all(conn, "add first asset").expect("should commit first asset");

        let replacement_asset = AssetRef {
            id: HashId::convert_str("replacement-asset"),
            uri: "file://assets/replacement.fa".to_string(),
            checksum: Some(Sha256Hash::convert_str("replacement-checksum")),
            created_on: 2,
            ..first_asset
        };
        AssetRef::create(conn, &replacement_asset).expect("should insert replacement asset");
        commit_all(conn, "replace asset").expect("should commit replacement asset");

        let assets =
            Assets::get_branch_assets(conn, "main").expect("should read assets from branch head");

        assert_eq!(
            assets,
            HashMap::from([(replacement_asset.id, replacement_asset)])
        );
    }

    #[test]
    fn test_generate_file_addition_id_consistency() {
        let checksum = Sha256Hash([1u8; 32]);
        let file_path = "/path/to/file.txt";

        let id1 = LocalAssetUri::generate_file_addition_id(Some(&checksum), file_path);
        let id2 = LocalAssetUri::generate_file_addition_id(Some(&checksum), file_path);

        assert_eq!(id1, id2);
    }

    #[test]
    fn test_generate_file_addition_id_uniqueness_different_paths() {
        let checksum = Sha256Hash([1u8; 32]);
        let file_path1 = "/path/to/file1.txt";
        let file_path2 = "/path/to/file2.txt";

        let id1 = LocalAssetUri::generate_file_addition_id(Some(&checksum), file_path1);
        let id2 = LocalAssetUri::generate_file_addition_id(Some(&checksum), file_path2);

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_generate_file_addition_id_uniqueness_different_checksums() {
        let checksum1 = Sha256Hash([1u8; 32]);
        let checksum2 = Sha256Hash([2u8; 32]);
        let file_path = "/path/to/file.txt";

        let id1 = LocalAssetUri::generate_file_addition_id(Some(&checksum1), file_path);
        let id2 = LocalAssetUri::generate_file_addition_id(Some(&checksum2), file_path);

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_resolve_source_path_rejects_absolute_path_from_file_in_repo() {
        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.base_dir();

        let absolute_path = repo_root.join("inputs").join("absolute.txt");
        fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        fs::write(&absolute_path, b"absolute").unwrap();
        let absolute_string = absolute_path.to_string_lossy().to_string();

        let err =
            LocalAssetUri::resolve_source_path(workspace, absolute_string.as_str()).unwrap_err();
        assert!(matches!(err, FileAdditionError::PathOutsideRepo { .. }));
    }

    #[test]
    fn test_stored_relative_path_returns_relative_path_from_file_in_repo() {
        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.base_dir();

        let absolute_path = repo_root.join("inputs").join("absolute.txt");
        fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        fs::write(&absolute_path, b"absolute").unwrap();
        let relative_string = absolute_path
            .strip_prefix(repo_root)
            .unwrap()
            .to_string_lossy()
            .to_string();

        let relative = LocalAssetUri::stored_relative_path(
            workspace,
            &absolute_path,
            &calculate_file_checksum(&absolute_path).unwrap(),
        )
        .unwrap();

        assert_eq!(relative, relative_string);
    }

    #[test]
    fn test_resolve_source_path_returns_absolute_path_from_relative_path_in_repo() {
        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.repo_root().unwrap();

        let relative_path = PathBuf::from("relative/path/file.txt");
        let absolute_path = repo_root.join(&relative_path);
        fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        fs::write(&absolute_path, b"relative").unwrap();
        let relative_string = relative_path.to_string_lossy().to_string();
        let absolute_string = absolute_path.to_string_lossy().to_string();

        let absolute =
            LocalAssetUri::resolve_source_path(workspace, relative_string.as_str()).unwrap();

        assert_eq!(absolute, PathBuf::from(absolute_string));
    }

    #[test]
    fn test_stored_relative_path_returns_relative_path_from_relative_path_in_repo() {
        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.repo_root().unwrap();

        let relative_path = PathBuf::from("relative/path/file.txt");
        let absolute_path = repo_root.join(&relative_path);
        fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        fs::write(&absolute_path, b"relative").unwrap();
        let relative_string = relative_path.to_string_lossy().to_string();

        let relative = LocalAssetUri::stored_relative_path(
            workspace,
            &absolute_path,
            &calculate_file_checksum(&absolute_path).unwrap(),
        )
        .unwrap();

        assert_eq!(relative, relative_string);
    }

    #[test]
    fn test_resolve_source_path_rejects_absolute_path_outside_repo() {
        let context = setup_gen();
        let workspace = context.workspace();

        let mut outside_file = tempfile::NamedTempFile::new().unwrap();
        outside_file.write_all(b"outside repo").unwrap();
        outside_file.flush().unwrap();
        let outside_string = outside_file.path().to_string_lossy().to_string();

        let err =
            LocalAssetUri::resolve_source_path(workspace, outside_string.as_str()).unwrap_err();
        assert!(matches!(err, FileAdditionError::PathOutsideRepo { .. }));
    }

    #[test]
    fn test_stored_relative_path_returns_asset_path_on_file_outside_repo() {
        let context = setup_gen();
        let workspace = context.workspace();

        let mut outside_file = tempfile::NamedTempFile::new().unwrap();
        outside_file.write_all(b"outside repo").unwrap();
        outside_file.flush().unwrap();
        let checksum = calculate_file_checksum(outside_file.path()).unwrap();
        let outside_suffix = LocalAssetUri::new(&outside_file.path().to_string_lossy())
            .suffix()
            .unwrap();

        let relative =
            LocalAssetUri::stored_relative_path(workspace, outside_file.path(), &checksum).unwrap();

        assert_eq!(relative, format!(".gen/assets/{checksum}.{outside_suffix}"));
    }

    #[test]
    fn test_new_for_workspace_namespaces_path_outside_workspace() {
        let context = setup_gen();
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("simple.fa");
        fs::write(&outside_path, b">seq\nACGT\n").unwrap();

        let asset_uri =
            LocalAssetUri::new_for_workspace(context.workspace(), &outside_path.to_string_lossy())
                .unwrap();

        assert_eq!(asset_uri.uri(), "file://.gen/outside_root/simple.fa");
        assert_eq!(asset_uri.suffix().as_deref(), Some("fa"));
    }

    #[test]
    fn test_new_for_workspace_distinguishes_external_file_from_workspace_root_file() {
        let context = setup_gen();
        let workspace_path = context.workspace().repo_root().unwrap().join("simple.fa");
        fs::write(&workspace_path, b"workspace file").unwrap();
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("simple.fa");
        fs::write(&outside_path, b"external file").unwrap();

        let workspace_asset = LocalAssetUri::new_for_workspace(
            context.workspace(),
            &workspace_path.to_string_lossy(),
        )
        .unwrap();
        let outside_asset =
            LocalAssetUri::new_for_workspace(context.workspace(), &outside_path.to_string_lossy())
                .unwrap();

        assert_eq!(workspace_asset.uri(), "file://simple.fa");
        assert_eq!(outside_asset.uri(), "file://.gen/outside_root/simple.fa");
        assert_ne!(workspace_asset.uri(), outside_asset.uri());
    }

    #[test]
    fn stored_relative_path_keeps_compression_suffix_for_asset_path() {
        let context = setup_gen();
        let workspace = context.workspace();
        let fixture_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/simple.fa.bgz");
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("simple.fa.bgz");
        fs::copy(&fixture_path, &outside_path).unwrap();
        let checksum = calculate_file_checksum(&outside_path).unwrap();

        let relative =
            LocalAssetUri::stored_relative_path(workspace, &outside_path, &checksum).unwrap();

        assert_eq!(relative, format!(".gen/assets/{checksum}.fa.bgz"));
    }

    #[test]
    fn stored_relative_path_keeps_unknown_file_type_suffix_for_asset_path() {
        let context = setup_gen();
        let workspace = context.workspace();
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("sample.custom.gz");
        fs::write(&outside_path, b"custom contents").unwrap();
        let checksum = calculate_file_checksum(&outside_path).unwrap();

        let relative =
            LocalAssetUri::stored_relative_path(workspace, &outside_path, &checksum).unwrap();

        assert_eq!(relative, format!(".gen/assets/{checksum}.custom.gz"));
    }

    #[test]
    fn stored_relative_path_keeps_full_source_suffix_for_asset_path() {
        let context = setup_gen();
        let workspace = context.workspace();
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("sample.fa.unhandled");
        fs::write(&outside_path, b"custom contents").unwrap();
        let checksum = calculate_file_checksum(&outside_path).unwrap();

        let relative =
            LocalAssetUri::stored_relative_path(workspace, &outside_path, &checksum).unwrap();

        assert_eq!(relative, format!(".gen/assets/{checksum}.fa.unhandled"));
    }

    #[test]
    fn stored_relative_path_omits_suffix_for_source_without_suffix() {
        let context = setup_gen();
        let workspace = context.workspace();
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("sample");
        fs::write(&outside_path, b"custom contents").unwrap();
        let checksum = calculate_file_checksum(&outside_path).unwrap();

        let relative =
            LocalAssetUri::stored_relative_path(workspace, &outside_path, &checksum).unwrap();

        assert_eq!(relative, format!(".gen/assets/{checksum}"));
    }

    #[test]
    fn hashed_filename_uses_http_uri_path_suffix() {
        let checksum = Sha256Hash::convert_str("remote");
        let asset_uri = RemoteAssetUri::new("http://example.com/assets/fasta.fa.gz?download=1");

        let filename = asset_uri.hashed_filename(&checksum);

        assert_eq!(filename, format!("{checksum}.fa.gz"));
    }

    #[test]
    fn test_resolve_source_path_anchors_relative_paths_to_workspace() {
        let context = setup_gen();
        let workspace = context.workspace();
        let expected = workspace.repo_root().unwrap().join("detached/file.txt");

        let absolute = LocalAssetUri::resolve_source_path(workspace, "detached/file.txt").unwrap();
        assert_eq!(absolute, expected);
    }

    #[test]
    fn test_resolve_source_path_rejects_parent_traversal() {
        let context = setup_gen();
        let workspace = context.workspace();

        let err = LocalAssetUri::resolve_source_path(workspace, "../detached/file.txt")
            .expect_err("expected traversal to be rejected");
        assert!(matches!(err, FileAdditionError::FileReadError(_)));
    }

    mod test_sanitize_relative_path {
        use super::*;

        #[test]
        fn keeps_normal_relative_path() {
            let sanitized = LocalAssetUri::sanitize_relative_path(Path::new("nested/file.txt"))
                .expect("expected relative path to be allowed");
            assert_eq!(sanitized, PathBuf::from("nested/file.txt"));
        }

        #[test]
        fn rejects_parent_dir() {
            let err = LocalAssetUri::sanitize_relative_path(Path::new("../nested/file.txt"))
                .expect_err("expected parent traversal to be rejected");
            assert!(matches!(err, FileAdditionError::FileReadError(_)));
        }
    }

    mod test_normalize_path {
        use super::*;

        #[test]
        fn collapses_dot_and_parent_components() {
            let normalized =
                LocalAssetUri::normalize_path(Path::new("nested/./deeper/../file.txt"));
            assert_eq!(normalized, PathBuf::from("nested/file.txt"));
        }
    }

    mod test_canonicalize_or_normalize {
        use super::*;

        #[test]
        fn normalizes_missing_path() {
            let normalized = LocalAssetUri::canonicalize_or_normalize(Path::new(
                "/tmp/gen-models-assets-tests/one/../two/file.txt",
            ));
            assert_eq!(
                normalized,
                PathBuf::from("/tmp/gen-models-assets-tests/two/file.txt")
            );
        }

        #[test]
        fn canonicalizes_existing_path() {
            let temp_dir = tempfile::tempdir().unwrap();
            let target_dir = temp_dir.path().join("target");
            fs::create_dir_all(&target_dir).unwrap();
            let file_path = target_dir.join("file.txt");
            fs::write(&file_path, b"data").unwrap();

            let canonicalized = LocalAssetUri::canonicalize_or_normalize(
                &temp_dir.path().join("target/../target/file.txt"),
            );

            // Unfortunately, the path we get back from Mac OS X sometimes starts
            // with /private even if the input path doesn't, which causes problems
            // for our path comparisons. Strip it out if it's there.
            let expected_canonical = fs::canonicalize(&file_path).unwrap();
            let cleaned_path = expected_canonical
                .strip_prefix("/private")
                .unwrap_or(&expected_canonical);
            let expected_path = Path::new("/").join(cleaned_path);
            assert_eq!(canonicalized, expected_path);
        }
    }

    #[test]
    fn test_new_for_workspace_rejects_parent_traversal_file_uri() {
        let context = setup_gen();
        let workspace = context.workspace();

        let err = LocalAssetUri::new_for_workspace(workspace, "file://../detached/file.txt")
            .err()
            .expect("expected traversal file uri to be rejected");
        assert!(matches!(err, FileAdditionError::FileReadError(_)));
    }

    #[test]
    fn test_resolve_source_path_rejects_absolute_asset_uri() {
        let context = setup_gen();
        let workspace = context.workspace();

        let absolute_path = workspace.repo_root().unwrap().join("detached/file.txt");
        let err = LocalAssetUri::resolve_source_path(
            workspace,
            &format!("file://{}", absolute_path.display()),
        )
        .expect_err("expected absolute asset uri to be rejected");
        assert!(matches!(err, FileAdditionError::PathOutsideRepo { .. }));
    }

    #[test]
    fn test_new_for_workspace_rejects_non_local_uri() {
        let context = setup_gen();
        let workspace = context.workspace();

        let err = LocalAssetUri::new_for_workspace(workspace, "https://example.com/reference.fa")
            .err()
            .expect("expected non-local uri to be rejected");
        assert!(matches!(err, FileAdditionError::FileReadError(_)));
        assert!(
            err.to_string()
                .contains("unsupported non-local uri: https://example.com/reference.fa")
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_resolve_source_path_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.repo_root().unwrap();
        let mut outside_file = tempfile::NamedTempFile::new().unwrap();
        outside_file.write_all(b"outside repo").unwrap();
        outside_file.flush().unwrap();

        let symlink_path = repo_root.join("escape-link.txt");
        if symlink_path.exists() || symlink_path.symlink_metadata().is_ok() {
            fs::remove_file(&symlink_path).unwrap();
        }
        symlink(outside_file.path(), &symlink_path).unwrap();

        let err = LocalAssetUri::resolve_source_path(workspace, "escape-link.txt")
            .expect_err("expected symlink escape to be rejected");
        assert!(matches!(err, FileAdditionError::PathOutsideRepo { .. }));
    }

    #[test]
    fn test_stored_relative_path_keeps_workspace_relative_paths() {
        let context = setup_gen();
        let workspace = context.workspace();

        let relative = LocalAssetUri::stored_relative_path(
            workspace,
            &workspace.repo_root().unwrap().join("detached/file.txt"),
            &Sha256Hash::convert_str("detached"),
        )
        .unwrap();
        assert_eq!(relative, "detached/file.txt");
    }

    #[test]
    fn test_resolve_source_path_empty() {
        let context = setup_gen();
        let workspace = context.workspace();

        let absolute_empty = LocalAssetUri::resolve_source_path(workspace, "").unwrap();
        assert_eq!(absolute_empty, PathBuf::new());
    }

    #[test]
    fn test_stored_relative_path_empty() {
        let context = setup_gen();
        let workspace = context.workspace();

        let relative_empty = LocalAssetUri::stored_relative_path(
            workspace,
            Path::new(""),
            &Sha256Hash::convert_str("empty"),
        )
        .unwrap();
        assert_eq!(relative_empty, "");
    }

    #[test]
    fn test_local_asset_uri_read_write() {
        let context = setup_gen();
        let repo_root = context.workspace().repo_root().unwrap();
        let path = repo_root.join("asset-uri-io.txt");
        fs::write(&path, "initial").unwrap();

        let mut reader =
            LocalAssetUri::new_for_workspace(context.workspace(), "asset-uri-io.txt").unwrap();
        let mut contents = String::new();
        reader.read_to_string(&mut contents).unwrap();
        assert_eq!(contents, "initial");

        let mut writer =
            LocalAssetUri::new_for_workspace(context.workspace(), "asset-uri-io.txt").unwrap();
        writer.write_all(b"updated").unwrap();
        writer.close_write().unwrap();

        assert_eq!(fs::read_to_string(path).unwrap(), "updated");
    }

    #[test]
    fn test_remote_asset_uri_checksums_streamed_http_content() {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as u16;
        let base_port = 20_000 + (seed % 20_000);
        let listener = (0..32)
            .find_map(|offset| {
                let port = base_port.saturating_add(offset);
                TcpListener::bind(("127.0.0.1", port)).ok()
            })
            .expect("failed to bind test HTTP listener");
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let started = Instant::now();
            let mut served_get_count = 0;
            while served_get_count < 1 && started.elapsed() < Duration::from_secs(5) {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                };
                let mut request = [0; 1024];
                let len = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..len]);
                if request.starts_with("GET ") {
                    served_get_count += 1;
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\ninitial",
                        )
                        .unwrap();
                } else {
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\n",
                        )
                        .unwrap();
                }
            }
            assert_eq!(served_get_count, 1);
        });

        let context = setup_gen();
        let uri = format!("http://{addr}/asset.fa");
        let asset_uri = RemoteAssetUri::new(&uri);
        assert_eq!(
            asset_uri.prepare_asset(context.workspace(), None).unwrap(),
            None
        );
        let mut reader = ChecksummedReader::new(asset_uri.reader(context.workspace()).unwrap());
        let checksum_handle = reader.checksum_handle();
        let mut contents = String::new();
        reader.read_to_string(&mut contents).unwrap();

        handle.join().unwrap();
        let expected_checksum = calculate_reader_checksum(contents.as_bytes()).unwrap();
        assert_eq!(checksum_handle.checksum(), Some(expected_checksum));
        assert_ne!(expected_checksum, Sha256Hash::convert_str(&uri));
    }

    #[test]
    fn test_remote_asset_uri_reader_uses_range_request() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("should bind test HTTP listener");
        listener
            .set_nonblocking(true)
            .expect("should configure test HTTP listener");
        let address = listener.local_addr().expect("should read listener address");
        let handle = thread::spawn(move || {
            let started = Instant::now();
            let mut served_range = false;
            while !served_range && started.elapsed() < Duration::from_secs(5) {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                };
                let mut request = [0; 1024];
                let length = stream.read(&mut request).expect("should read request");
                let request = String::from_utf8_lossy(&request[..length]).to_lowercase();
                if request.starts_with("get ") {
                    assert!(
                        request.contains("\r\nrange: bytes=2-6\r\n"),
                        "remote reader should resume at the seek position: {request}"
                    );
                    stream
                        .write_all(
                            b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 2-6/7\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\nitial",
                        )
                        .expect("should write ranged response");
                    served_range = true;
                } else {
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                        )
                        .expect("should write metadata response");
                }
            }
            assert!(served_range, "should serve a ranged GET request");
        });

        let mut reader = OpenDalLocation::from_remote_uri(&format!("http://{address}/asset.fa"))
            .expect("should open seekable remote reader");
        reader
            .seek(SeekFrom::Start(2))
            .expect("should seek within remote asset");
        let mut bytes = [0; 3];
        let bytes_read = reader
            .read(&mut bytes)
            .expect("should read requested remote range");
        handle.join().expect("should finish HTTP server");

        assert_eq!(bytes_read, bytes.len());
        assert_eq!(&bytes, b"iti");
    }

    #[test]
    fn test_remote_s3_uri_retries_anonymously_after_authenticated_read_fails() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("should bind test HTTP listener");
        listener
            .set_nonblocking(true)
            .expect("should configure test HTTP listener");
        let address = listener.local_addr().expect("should read listener address");
        let handle = thread::spawn(move || {
            let started = Instant::now();
            let mut saw_signed_request = false;
            let mut saw_anonymous_request = false;
            let mut served_content = false;
            while !served_content && started.elapsed() < Duration::from_secs(5) {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                };
                let mut request = [0; 4096];
                let length = stream.read(&mut request).expect("should read request");
                let request = String::from_utf8_lossy(&request[..length]).to_lowercase();
                if request.contains("\r\nauthorization:") {
                    saw_signed_request = true;
                    let body = "<Error><Code>AccessDenied</Code><Message>signed request rejected</Message></Error>";
                    write!(
                        stream,
                        "HTTP/1.1 403 Forbidden\r\nContent-Type: application/xml\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .expect("should reject signed request");
                } else if request.starts_with("head ") {
                    saw_anonymous_request = true;
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                        )
                        .expect("should accept anonymous metadata request");
                } else if request.starts_with("get ") {
                    saw_anonymous_request = true;
                    stream
                        .write_all(
                            b"HTTP/1.1 206 Partial Content\r\nContent-Length: 7\r\nContent-Range: bytes 0-6/7\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\ninitial",
                        )
                        .expect("should serve anonymous content request");
                    served_content = true;
                }
            }
            (saw_signed_request, saw_anonymous_request, served_content)
        });

        let mut url = Url::parse("s3://public-bucket/annotations/genes.gff3")
            .expect("should parse test S3 URI");
        url.query_pairs_mut()
            .append_pair("endpoint", &format!("http://{address}"))
            .append_pair("region", "us-east-1")
            .append_pair("access_key_id", "other-account-key")
            .append_pair("secret_access_key", "other-account-secret")
            .append_pair("disable_config_load", "true")
            .append_pair("disable_ec2_metadata", "true");

        let mut reader =
            OpenDalLocation::from_remote_uri(url.as_str()).expect("should retry anonymously");
        let mut contents = String::new();
        reader
            .read_to_string(&mut contents)
            .expect("should read public S3 content anonymously");
        let (saw_signed_request, saw_anonymous_request, served_content) =
            handle.join().expect("should finish HTTP server");

        assert!(
            saw_signed_request,
            "should try configured credentials first"
        );
        assert!(
            saw_anonymous_request,
            "should retry without credentials after the signed read fails"
        );
        assert!(
            served_content,
            "should serve the anonymously readable object"
        );
        assert_eq!(contents, "initial");
    }

    #[test]
    fn test_remote_s3_uri_respects_explicit_signed_only_access() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("should bind test HTTP listener");
        listener
            .set_nonblocking(true)
            .expect("should configure test HTTP listener");
        let address = listener.local_addr().expect("should read listener address");
        let handle = thread::spawn(move || {
            let started = Instant::now();
            let mut signed_request_at = None;
            let mut saw_signed_request = false;
            let mut saw_anonymous_request = false;
            while started.elapsed() < Duration::from_secs(5)
                && signed_request_at
                    .is_none_or(|time: Instant| time.elapsed() < Duration::from_millis(200))
            {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                };
                let mut request = [0; 4096];
                let length = stream.read(&mut request).expect("should read request");
                let request = String::from_utf8_lossy(&request[..length]).to_lowercase();
                if request.contains("\r\nauthorization:") {
                    saw_signed_request = true;
                    signed_request_at = Some(Instant::now());
                    let body = "<Error><Code>AccessDenied</Code><Message>signed request rejected</Message></Error>";
                    write!(
                        stream,
                        "HTTP/1.1 403 Forbidden\r\nContent-Type: application/xml\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .expect("should reject signed request");
                } else {
                    saw_anonymous_request = true;
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                        )
                        .expect("should accept anonymous metadata request");
                    break;
                }
            }
            (saw_signed_request, saw_anonymous_request)
        });

        let mut url = Url::parse("s3://public-bucket/annotations/genes.gff3")
            .expect("should parse test S3 URI");
        url.query_pairs_mut()
            .append_pair("endpoint", &format!("http://{address}"))
            .append_pair("region", "us-east-1")
            .append_pair("access_key_id", "other-account-key")
            .append_pair("secret_access_key", "other-account-secret")
            .append_pair("allow_anonymous", "false")
            .append_pair("disable_config_load", "true")
            .append_pair("disable_ec2_metadata", "true");

        let result = OpenDalLocation::from_remote_uri(url.as_str());
        let (saw_signed_request, saw_anonymous_request) =
            handle.join().expect("should finish HTTP server");

        assert!(
            result.is_err(),
            "should return the authenticated read error"
        );
        assert!(saw_signed_request, "should try configured credentials");
        assert!(
            !saw_anonymous_request,
            "should not override an explicit signed-only policy"
        );
    }
}
