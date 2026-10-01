use gen_core::Workspace;

use super::{AssetRef, LocalAssetUri, PreparedAssetSet};
use crate::{
    errors::FileAdditionError,
    operations::{FileAddition, OperationFile, PreparedFileAddition},
};

pub(crate) fn prepare_operation_file(
    operation_file: &OperationFile,
    workspace: &Workspace,
    created_on: i64,
) -> Result<PreparedAssetSet, FileAdditionError> {
    let prepared_file = FileAddition::prepare(
        workspace,
        &operation_file.file_path,
        operation_file.file_type,
        operation_file.role.clone(),
        operation_file.checksum_override,
        operation_file.materialized_checksum_override,
    )?;
    let logical_path = operation_file.logical_path(workspace, &prepared_file)?;
    let parent = AssetRef::from_file_addition(
        &prepared_file.addition,
        operation_file.role.clone(),
        Some(&logical_path),
        Some(&operation_file.filename),
        operation_file.upstream_asset_ref_id.as_ref(),
        created_on,
    );

    let related_files = operation_file.related_files.clone().unwrap_or_default();
    let mut derived = Vec::with_capacity(related_files.len());
    let mut prepared_related_files = Vec::with_capacity(related_files.len());
    for mut related_file in related_files {
        related_file.upstream_asset_ref_id = Some(parent.id);
        related_file.related_files = None;
        let prepared_related = FileAddition::prepare(
            workspace,
            &related_file.file_path,
            related_file.file_type,
            related_file.role.clone(),
            related_file.checksum_override,
            related_file.materialized_checksum_override,
        )?;
        let related_logical_path = related_file.logical_path(workspace, &prepared_related)?;
        let related_asset = AssetRef::from_file_addition(
            &prepared_related.addition,
            related_file.role.clone(),
            Some(&related_logical_path),
            Some(&related_file.filename),
            related_file.upstream_asset_ref_id.as_ref(),
            created_on,
        );
        prepared_related_files.push(normalized_operation_file(
            &related_file,
            &prepared_related,
            &related_asset,
            None,
        ));
        derived.push(related_asset);
    }

    let parent_operation_file = normalized_operation_file(
        operation_file,
        &prepared_file,
        &parent,
        operation_file
            .related_files
            .as_ref()
            .map(|_| prepared_related_files),
    );

    Ok(PreparedAssetSet {
        parent,
        derived,
        operation_files: vec![parent_operation_file],
    })
}

fn normalized_operation_file(
    source: &OperationFile,
    prepared: &PreparedFileAddition,
    asset_ref: &AssetRef,
    related_files: Option<Vec<OperationFile>>,
) -> OperationFile {
    let file_path = if LocalAssetUri::is_local_path_or_file_uri(&prepared.addition.asset_uri) {
        prepared.addition.file_path().to_string()
    } else {
        prepared.addition.asset_uri.clone()
    };
    let mut operation_file = OperationFile::new(file_path)
        .set_file_type(source.file_type)
        .set_role(source.role.clone());
    operation_file.filename = source.filename.clone();
    operation_file.checksum_override = prepared.addition.checksum;
    operation_file.materialized_checksum_override = prepared.addition.materialized_checksum;
    operation_file.logical_path_override = asset_ref.logical_path.clone();
    operation_file.upstream_asset_ref_id = asset_ref.upstream_asset_ref_id;
    operation_file.related_files = related_files;
    operation_file
}

#[cfg(test)]
mod tests {
    use std::fs;

    use gen_core::HashId;

    use crate::{
        assets::AssetRole, file_types::FileTypes, operations::OperationFile,
        test_helpers::setup_gen,
    };

    #[test]
    fn test_preparation_preserves_generic_related_asset_role() {
        let context = setup_gen();
        let repository_root = context.workspace().repo_root().unwrap();
        let parent_path = repository_root.join("parent.dat");
        let related_path = repository_root.join("related.idx");
        fs::write(&parent_path, b"parent bytes").unwrap();
        fs::write(&related_path, b"related bytes").unwrap();
        let related_role = AssetRole::AnnotationIndex;
        let stale_upstream_id = HashId::convert_str("stale-parent");
        let operation_file = OperationFile::new(parent_path.to_string_lossy())
            .set_file_type(FileTypes::None)
            .set_related_files(vec![
                OperationFile::new(related_path.to_string_lossy())
                    .set_file_type(FileTypes::None)
                    .set_role(related_role.clone())
                    .set_upstream_asset_ref_id(&stale_upstream_id),
            ]);

        let prepared = operation_file
            .prepare_assets(context.workspace(), 1)
            .expect("should prepare parent and related generic assets");

        assert_eq!(prepared.derived.len(), 1);
        assert_eq!(prepared.derived[0].role, related_role);
        assert_eq!(
            prepared.derived[0].upstream_asset_ref_id,
            Some(prepared.parent.id)
        );
        let normalized_related = &prepared.operation_files[0].related_files.as_ref().unwrap()[0];
        assert_eq!(normalized_related.role, AssetRole::AnnotationIndex);
        assert_eq!(
            normalized_related.upstream_asset_ref_id,
            Some(prepared.parent.id)
        );
    }
}
