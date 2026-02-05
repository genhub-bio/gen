pub mod gff;

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::anyhow;
use gen_core::{HashId, calculate_hash};
use gen_models::{
    annotations::{Annotation, AnnotationFile, AnnotationGroupSample, parse_annotation_file_type},
    block_group::{BlockGroup, PathCache},
    changesets::{ChangesetModels, DatabaseChangeset, write_changeset},
    db::DbContext,
    errors::OperationError,
    file_types::FileTypes,
    files::GenDatabase,
    metadata,
    operations::{FileAddition, Operation, OperationInfo, OperationSummary},
    sample::Sample,
    session_operations::{DependencyModels, end_operation, start_operation},
};
use noodles::core::Region;

fn detect_annotation_file_type(path: &str) -> Option<FileTypes> {
    let path = Path::new(path);
    let mut ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());
    if matches!(ext.as_deref(), Some("gz") | Some("bgz")) {
        ext = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| Path::new(stem).extension().and_then(|ext| ext.to_str()))
            .map(|ext| ext.to_ascii_lowercase());
    }
    match ext.as_deref() {
        Some("gff3") | Some("gff") => Some(FileTypes::Gff3),
        Some("bed") => Some(FileTypes::Bed),
        Some("genbank") | Some("gb") | Some("gbk") => Some(FileTypes::GenBank),
        _ => None,
    }
}

pub fn add_annotation(
    context: &DbContext,
    collection: &str,
    name: &str,
    group: Option<&str>,
    sample: Option<&str>,
    region: &str,
) -> Result<Operation, Box<dyn std::error::Error>> {
    let graph_conn = context.graph().conn();
    let operation_conn = context.operations().conn();
    let parsed_region = region.parse::<Region>()?;
    let interval = parsed_region.interval();
    let start = interval
        .start()
        .ok_or_else(|| anyhow!("Region missing start"))?
        .get() as i64;
    let end = interval
        .end()
        .ok_or_else(|| anyhow!("Region missing end"))?
        .get() as i64;

    let block_groups = Sample::get_block_groups(graph_conn, collection, sample);
    let block_group = block_groups
        .iter()
        .find(|bg| bg.name == parsed_region.name())
        .ok_or_else(|| {
            let sample_label = match sample {
                Some(name) => format!("sample {name}"),
                None => "default sample".to_string(),
            };
            anyhow!(
                "Graph {} not found for {sample_label}",
                parsed_region.name()
            )
        })?;
    let path = BlockGroup::get_current_path(graph_conn, &block_group.id);
    let path_length = path.length(graph_conn);
    if start < 0 || end < 0 || start > end || end > path_length {
        return Err(anyhow!("Region {region} is outside the path bounds (0-{path_length})").into());
    }

    let mut session = start_operation(graph_conn);
    graph_conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let mut cache = PathCache::new(graph_conn);
    let _ = PathCache::lookup(&mut cache, &block_group.id, path.name.clone());
    let accession = BlockGroup::add_accession(graph_conn, &path, name, start, end, &mut cache);

    let annotation_group = group.unwrap_or("default");
    let annotation = Annotation::get_or_create(graph_conn, name, annotation_group, &accession.id)?;
    if let Some(sample_name) = sample {
        AnnotationGroupSample::create(graph_conn, &annotation.group, sample_name)?;
    }

    let operation = end_operation(
        context,
        &mut session,
        &OperationInfo {
            files: vec![],
            description: format!("add annotation {name}"),
        },
        &format!("add annotation {name}"),
        None,
    )?;

    graph_conn.execute("END TRANSACTION", [])?;
    operation_conn.execute("END TRANSACTION", [])?;

    Ok(operation)
}

pub fn add_annotation_file(
    context: &DbContext,
    path: &str,
    format: Option<&str>,
    name: Option<&str>,
    message: Option<&str>,
) -> Result<Operation, Box<dyn std::error::Error>> {
    let workspace = context.workspace();
    let operation_conn = context.operations().conn();
    let graph_conn = context.graph().conn();
    let db_uuid = metadata::get_db_uuid(graph_conn);

    let file_type = match format {
        Some(format) => parse_annotation_file_type(format)?,
        None => detect_annotation_file_type(path).ok_or_else(|| {
            anyhow!(
                "Unable to detect annotation file format from the file extension. Use --format to specify it explicitly."
            )
        })?,
    };
    let file_addition =
        FileAddition::get_or_create(workspace, operation_conn, path, file_type, None)?;
    let name_value = name.unwrap_or_default();
    let operation_hash = HashId(calculate_hash(&format!(
        "{file_addition_id}:{name_value}",
        file_addition_id = file_addition.id
    )));
    let operation = match Operation::create(operation_conn, "annotation-file", &operation_hash) {
        Ok(operation) => operation,
        Err(rusqlite::Error::SqliteFailure(err, _details))
            if err.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            return Err(OperationError::NoChanges.into());
        }
        Err(err) => return Err(err.into()),
    };
    AnnotationFile::link_to_operation(operation_conn, &operation.hash, &file_addition.id, name)?;
    Operation::add_database(operation_conn, &operation.hash, &db_uuid)?;
    let summary = message
        .map(str::to_string)
        .unwrap_or_else(|| format!("Add annotation file {path}"));
    OperationSummary::create(operation_conn, &operation.hash, &summary);

    let gen_db = GenDatabase::get_by_uuid(operation_conn, &db_uuid)?;
    write_changeset(
        workspace,
        &operation,
        DatabaseChangeset {
            db_path: gen_db.path,
            changes: ChangesetModels::default(),
        },
        &DependencyModels::default(),
    );

    if file_type != FileTypes::Changeset && file_type != FileTypes::None {
        let gen_dir = workspace
            .find_gen_dir()
            .ok_or_else(|| anyhow!("No .gen directory found. Please run 'gen init' first."))?;
        let assets_dir = gen_dir.join("assets");
        fs::create_dir_all(&assets_dir)?;
        let asset_path = assets_dir.join(file_addition.hashed_filename());
        if !asset_path.exists() {
            let source_path = if Path::new(path).is_absolute() {
                PathBuf::from(path)
            } else {
                workspace.repo_root()?.join(path)
            };
            fs::copy(source_path, asset_path)?;
        }
    }

    Ok(operation)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use gen_models::{annotations::Annotation, errors::OperationError};

    use super::*;
    use crate::{
        test_helpers::{setup_block_group, setup_gen},
        track_database,
    };

    #[test]
    fn test_add_annotation() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let operation_conn = context.operations().conn();
        track_database(graph_conn, operation_conn).unwrap();
        let _ = setup_block_group(graph_conn);
        let operation = add_annotation(
            &context,
            "test",
            "gene-a",
            Some("track-1"),
            None,
            "chr1:1-5",
        )
        .unwrap();
        assert_eq!(operation.change_type, "add annotation gene-a");

        let annotations = Annotation::query_by_group(graph_conn, "track-1").unwrap();
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0].name, "gene-a");
    }

    #[test]
    fn test_add_annotation_file() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let operation_conn = context.operations().conn();
        track_database(graph_conn, operation_conn).unwrap();

        let repo_root = context.workspace().repo_root().unwrap();
        let annotation_path = repo_root.join("fixtures").join("annotation.gff3");
        fs::create_dir_all(annotation_path.parent().unwrap()).unwrap();
        fs::write(&annotation_path, "##gff-version 3\n").unwrap();
        let annotation_path_str = annotation_path.to_string_lossy().to_string();

        let operation =
            add_annotation_file(&context, &annotation_path_str, None, Some("track-1"), None)
                .unwrap();
        assert_eq!(operation.change_type, "annotation-file");

        let links = AnnotationFile::get_links_for_operation(operation_conn, &operation.hash);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].name.as_deref(), Some("track-1"));

        let err = add_annotation_file(&context, &annotation_path_str, None, Some("track-1"), None)
            .unwrap_err();
        let op_err = err
            .downcast_ref::<OperationError>()
            .expect("expected OperationError");
        assert_eq!(*op_err, OperationError::NoChanges);
    }
}
