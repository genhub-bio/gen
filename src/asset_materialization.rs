use std::{
    collections::HashMap,
    error::Error,
    fs,
    io::{self, Read as _, Write as _},
    path::{Path, PathBuf},
};

use flate2::read::MultiGzDecoder;
use gen_core::{HashId, Sha256Hash, config::Workspace};
use gen_models::{
    assets::{AssetRef, ChecksummedWriter, CompressionType, materialization_destination_path},
    operations::{calculate_file_checksum, calculate_reader_checksum},
};
use noodles::bgzf;
use sha2::{Digest as _, Sha256};

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum DownloadAssetOutcome {
    Unchanged,
    Downloaded,
    Conflict(PathBuf),
}

#[derive(Clone, Copy)]
pub(crate) enum AssetCopyStage {
    VersionedStore,
    Workspace,
}

pub(crate) trait AssetMaterializationProgress {
    fn checksum_started(&self, phase: &str, total_bytes: Option<u64>);
    fn checksum_bytes(&self, bytes: u64);
    fn copy_started(&self, stage: AssetCopyStage);
    fn copy_bytes(&self, stage: AssetCopyStage, bytes: u64);
}

/// Copies a checksum-addressed asset into its logical workspace path without losing local edits.
pub(crate) fn materialize_versioned_asset(
    workspace: &Workspace,
    asset: &AssetRef,
    previous_assets: &HashMap<HashId, AssetRef>,
    destination_logical_path: Option<&str>,
    versioned_path: &Path,
    versioned_file_created: bool,
) -> Result<DownloadAssetOutcome, Box<dyn Error>> {
    materialize_versioned_asset_with_progress(
        workspace,
        asset,
        previous_assets,
        destination_logical_path,
        versioned_path,
        versioned_file_created,
        None,
    )
}

pub(crate) fn materialize_versioned_asset_with_progress(
    workspace: &Workspace,
    asset: &AssetRef,
    previous_assets: &HashMap<HashId, AssetRef>,
    destination_logical_path: Option<&str>,
    versioned_path: &Path,
    versioned_file_created: bool,
    progress: Option<&dyn AssetMaterializationProgress>,
) -> Result<DownloadAssetOutcome, Box<dyn Error>> {
    let expected_checksum = asset_checksum(asset)?;
    let expected_materialized_checksum = materialized_asset_checksum(asset)?;
    if !versioned_path.is_file() {
        return Err(format!(
            "Cannot materialize asset {} because its versioned file is missing: {}",
            asset.id,
            versioned_path.display()
        )
        .into());
    }
    let versioned_checksum = calculate_file_checksum_with_progress(
        versioned_path,
        progress,
        "verifying the stored asset checksum",
    )?;
    if versioned_checksum != expected_checksum {
        return Err(format!(
            "Versioned asset {} at {} does not match its recorded checksum",
            asset.id,
            versioned_path.display()
        )
        .into());
    }
    let Some(destination_logical_path) = destination_logical_path else {
        return Ok(if versioned_file_created {
            DownloadAssetOutcome::Downloaded
        } else {
            DownloadAssetOutcome::Unchanged
        });
    };
    let destination_path = materialization_destination_path(
        workspace,
        &asset.uri,
        Some(&expected_checksum),
        Some(destination_logical_path),
    )?;
    if destination_path == versioned_path {
        return Ok(if versioned_file_created {
            DownloadAssetOutcome::Downloaded
        } else {
            DownloadAssetOutcome::Unchanged
        });
    }
    if destination_path.exists()
        && calculate_file_checksum_with_progress(
            &destination_path,
            progress,
            "checking the existing workspace asset",
        )
        .is_ok_and(|checksum| checksum == expected_materialized_checksum)
    {
        return Ok(DownloadAssetOutcome::Unchanged);
    }
    let existing_checksum = if destination_path.exists() {
        Some(calculate_file_checksum_with_progress(
            &destination_path,
            progress,
            "checking the existing workspace asset",
        )?)
    } else {
        None
    };
    let intended_change = if let Some(checksum) = existing_checksum.as_ref() {
        destination_matches_previous_asset(
            workspace,
            &destination_path,
            checksum,
            previous_assets,
            progress,
        )?
    } else {
        false
    };
    if existing_checksum.is_some() && !intended_change {
        let (conflict_path, already_materialized) =
            conflict_destination_path(&destination_path, &expected_materialized_checksum)?;
        if !already_materialized {
            copy_versioned_asset(
                versioned_path,
                &conflict_path,
                asset.materialized_checksum.as_ref(),
                &expected_materialized_checksum,
                progress,
                AssetCopyStage::Workspace,
            )?;
        }
        return Ok(DownloadAssetOutcome::Conflict(conflict_path));
    }

    copy_versioned_asset(
        versioned_path,
        &destination_path,
        asset.materialized_checksum.as_ref(),
        &expected_materialized_checksum,
        progress,
        AssetCopyStage::Workspace,
    )?;
    Ok(DownloadAssetOutcome::Downloaded)
}

/// Reports a preserved local file and the path containing the requested version.
pub(crate) fn warn_asset_conflict(
    workspace: &Workspace,
    asset: &AssetRef,
    destination_logical_path: Option<&str>,
    conflict_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let destination_path = materialization_destination_path(
        workspace,
        &asset.uri,
        asset.checksum.as_ref(),
        destination_logical_path,
    )?;
    eprintln!(
        "Warning: the requested asset version conflicts with the local file at {}. The local file was preserved and the requested version was written to {}. Choose the correct version before continuing.",
        destination_path.display(),
        conflict_path.display()
    );
    Ok(())
}

fn asset_checksum(asset: &AssetRef) -> Result<Sha256Hash, Box<dyn Error>> {
    asset
        .checksum
        .ok_or_else(|| format!("Local asset {} has no checksum", asset.id).into())
}

fn materialized_asset_checksum(asset: &AssetRef) -> Result<Sha256Hash, Box<dyn Error>> {
    match asset.materialized_checksum {
        Some(checksum) => Ok(checksum),
        None => asset_checksum(asset),
    }
}

fn destination_matches_previous_asset(
    workspace: &Workspace,
    destination_path: &Path,
    existing_checksum: &Sha256Hash,
    previous_assets: &HashMap<HashId, AssetRef>,
    progress: Option<&dyn AssetMaterializationProgress>,
) -> Result<bool, Box<dyn Error>> {
    let mut decoded_destination_checksum = None;
    let mut destination_decode_failed = false;
    for asset in previous_assets.values() {
        let archive_checksum = asset_checksum(asset)?;
        let logical_path = materialization_destination_path(
            workspace,
            &asset.uri,
            Some(&archive_checksum),
            asset.logical_path.as_deref(),
        )?;
        if logical_path != destination_path {
            continue;
        }
        if materialized_asset_checksum(asset)? == *existing_checksum {
            return Ok(true);
        }
        if !matches!(
            asset.materialized_compression_type(),
            CompressionType::Gzip | CompressionType::Bgzf
        ) {
            continue;
        }
        let archived_path =
            materialization_destination_path(workspace, &asset.uri, Some(&archive_checksum), None)?;
        if !archived_path.is_file() {
            continue;
        }
        if !calculate_file_checksum_with_progress(
            &archived_path,
            progress,
            "checking a previous asset archive",
        )
        .is_ok_and(|checksum| checksum == archive_checksum)
        {
            continue;
        }
        if destination_decode_failed {
            continue;
        }
        let existing_decoded_checksum = if let Some(checksum) = decoded_destination_checksum {
            checksum
        } else {
            match decoded_compressed_checksum(
                destination_path,
                progress,
                "checking the existing workspace asset",
            ) {
                Ok(checksum) => {
                    decoded_destination_checksum = Some(checksum);
                    checksum
                }
                Err(_) => {
                    destination_decode_failed = true;
                    continue;
                }
            }
        };
        let Ok(archived_decoded_checksum) = decoded_compressed_checksum(
            &archived_path,
            progress,
            "checking a previous asset archive",
        ) else {
            continue;
        };
        if existing_decoded_checksum == archived_decoded_checksum {
            return Ok(true);
        }
    }
    Ok(false)
}

fn decoded_compressed_checksum(
    path: &Path,
    progress: Option<&dyn AssetMaterializationProgress>,
    phase: &str,
) -> Result<Sha256Hash, Box<dyn Error>> {
    if let Some(progress) = progress {
        progress.checksum_started(phase, None);
    }
    let mut reader = MultiGzDecoder::new(fs::File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes_read = 0_u64;
    loop {
        let length = reader.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        bytes_read = bytes_read.saturating_add(length as u64);
        if let Some(progress) = progress {
            progress.checksum_bytes(bytes_read);
        }
        hasher.update(&buffer[..length]);
    }
    Ok(Sha256Hash(hasher.finalize().into()))
}

fn calculate_file_checksum_with_progress(
    file_path: &Path,
    progress: Option<&dyn AssetMaterializationProgress>,
    phase: &str,
) -> Result<Sha256Hash, Box<dyn Error>> {
    let Some(progress) = progress else {
        return calculate_file_checksum(file_path).map_err(Into::into);
    };
    let total_bytes = fs::metadata(file_path).ok().map(|metadata| metadata.len());
    progress.checksum_started(phase, total_bytes);
    let file = fs::File::open(file_path)?;
    let mut reader = CountingReader::new(io::BufReader::new(file), |bytes| {
        progress.checksum_bytes(bytes);
    });
    calculate_reader_checksum(&mut reader).map_err(Into::into)
}

fn conflict_destination_path(
    destination_path: &Path,
    expected_checksum: &Sha256Hash,
) -> Result<(PathBuf, bool), Box<dyn Error>> {
    let file_name = destination_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("asset");
    for index in 0_usize.. {
        let suffix = if index == 0 {
            ".conflict".to_string()
        } else {
            format!(".conflict.{index}")
        };
        let candidate = destination_path.with_file_name(format!("{file_name}{suffix}"));
        if !candidate.exists() {
            return Ok((candidate, false));
        }
        if calculate_file_checksum(&candidate).is_ok_and(|checksum| checksum == *expected_checksum)
        {
            return Ok((candidate, true));
        }
    }
    unreachable!("conflict suffix space should not be exhausted")
}

pub(crate) fn copy_versioned_asset(
    versioned_path: &Path,
    destination_path: &Path,
    materialized_checksum: Option<&Sha256Hash>,
    expected_output_checksum: &Sha256Hash,
    progress: Option<&dyn AssetMaterializationProgress>,
    copy_stage: AssetCopyStage,
) -> Result<(), Box<dyn Error>> {
    let parent = destination_path.parent().ok_or_else(|| {
        format!(
            "Asset destination has no parent: {}",
            destination_path.display()
        )
    })?;
    fs::create_dir_all(parent)?;
    let staged_path = temporary_path(destination_path)?;
    let result = (|| {
        let mut staged_file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&staged_path)?;
        let versioned_file = fs::File::open(versioned_path)?;
        if let Some(progress) = progress {
            progress.copy_started(copy_stage);
        }
        let staged_checksum = {
            let mut checksummed_writer = ChecksummedWriter::new(&mut staged_file);
            match materialized_checksum {
                Some(_) => {
                    let mut decoded_file = bgzf::io::Reader::new(versioned_file);
                    if let Some(progress) = progress {
                        let mut decoded_file = CountingReader::new(decoded_file, |bytes| {
                            progress.copy_bytes(copy_stage, bytes);
                        });
                        io::copy(&mut decoded_file, &mut checksummed_writer)?;
                    } else {
                        io::copy(&mut decoded_file, &mut checksummed_writer)?;
                    }
                }
                None => {
                    let mut versioned_file = versioned_file;
                    if let Some(progress) = progress {
                        let mut versioned_file = CountingReader::new(versioned_file, |bytes| {
                            progress.copy_bytes(copy_stage, bytes);
                        });
                        io::copy(&mut versioned_file, &mut checksummed_writer)?;
                    } else {
                        io::copy(&mut versioned_file, &mut checksummed_writer)?;
                    }
                }
            }
            checksummed_writer.flush()?;
            checksummed_writer.checksum()
        };
        staged_file.sync_all()?;
        drop(staged_file);
        if staged_checksum != *expected_output_checksum {
            return Err(format!(
                "Copied asset at {} failed checksum validation",
                destination_path.display()
            )
            .into());
        }
        fs::rename(&staged_path, destination_path)?;
        Ok::<(), Box<dyn Error>>(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&staged_path);
    }
    result
}

fn temporary_path(destination_path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let file_name = destination_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "Asset destination has no file name: {}",
                destination_path.display()
            )
        })?;
    Ok(destination_path.with_file_name(format!("{file_name}.tmp")))
}

struct CountingReader<R, F> {
    reader: R,
    bytes_read: u64,
    report_bytes: F,
}

impl<R, F> CountingReader<R, F> {
    fn new(reader: R, report_bytes: F) -> Self {
        Self {
            reader,
            bytes_read: 0,
            report_bytes,
        }
    }
}

impl<R, F> io::Read for CountingReader<R, F>
where
    R: io::Read,
    F: FnMut(u64),
{
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.reader.read(buffer)?;
        if read > 0 {
            self.bytes_read = self.bytes_read.saturating_add(read as u64);
            (self.report_bytes)(self.bytes_read);
        }
        Ok(read)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs,
        io::{Cursor, Write as _},
    };

    use flate2::{Compression, write::GzEncoder};
    use gen_core::{HashId, config::Workspace};
    use gen_models::{
        assets::{AssetRef, AssetRole, LocalAssetUri, materialization_destination_path},
        file_types::FileTypes,
        operations::{FileAddition, calculate_reader_checksum},
    };
    use noodles::bgzf;
    use tempfile::tempdir;

    use super::destination_matches_previous_asset;

    #[test]
    fn test_previous_compressed_asset_matches_by_decoded_content() {
        let temp = tempdir().expect("should create workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let logical_path = "reads.fa.gz";
        let source_contents = b">chr1\nAACCGG\n";
        let mut archive_contents = Vec::new();
        let mut archive_writer = bgzf::io::Writer::new(&mut archive_contents);
        archive_writer
            .write_all(source_contents)
            .expect("should write source bytes to BGZF");
        archive_writer
            .finish()
            .expect("should finish source BGZF archive");
        let archive_checksum = calculate_reader_checksum(Cursor::new(&archive_contents))
            .expect("should checksum source BGZF archive");
        let uri = LocalAssetUri::asset_uri(".gen/assets/reads.fa.bgz");
        let file_addition = FileAddition {
            id: HashId::convert_str("compressed-checkout-asset"),
            asset_uri: uri.clone(),
            file_type: FileTypes::Fasta,
            checksum: Some(archive_checksum),
            materialized_checksum: None,
        };
        let role = AssetRole::Input;
        let asset = AssetRef {
            id: AssetRef::id_hash(
                &file_addition,
                &role,
                Some(logical_path),
                Some("reads.fa.gz"),
                None,
            ),
            uri,
            file_type: FileTypes::Fasta.as_str().to_string(),
            checksum: Some(archive_checksum),
            materialized_checksum: None,
            size: Some(
                i64::try_from(archive_contents.len()).expect("should fit archive size in i64"),
            ),
            role,
            logical_path: Some(logical_path.to_string()),
            name: Some("reads.fa.gz".to_string()),
            created_on: 1,
            upstream_asset_ref_id: None,
        };
        let archive_path =
            materialization_destination_path(&workspace, &asset.uri, asset.checksum.as_ref(), None)
                .expect("should resolve source archive path");
        fs::create_dir_all(
            archive_path
                .parent()
                .expect("should have source archive parent"),
        )
        .expect("should create source archive directory");
        fs::write(&archive_path, archive_contents)
            .expect("should store checksum-addressed source archive");

        let mut gzip_writer = GzEncoder::new(Vec::new(), Compression::default());
        gzip_writer
            .write_all(source_contents)
            .expect("should write source bytes to gzip");
        let gzip_contents = gzip_writer.finish().expect("should finish gzip stream");
        let destination_path = temp.path().join(logical_path);
        fs::write(&destination_path, &gzip_contents).expect("should write gzip workspace file");
        let existing_checksum = calculate_reader_checksum(Cursor::new(gzip_contents))
            .expect("should checksum gzip workspace file");
        let previous_assets = HashMap::from([(asset.id, asset)]);

        assert!(
            destination_matches_previous_asset(
                &workspace,
                &destination_path,
                &existing_checksum,
                &previous_assets,
                None,
            )
            .expect("should compare compressed asset contents"),
            "different compression encodings of known contents should remain safe to replace"
        );
    }
}
