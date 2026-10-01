use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, BufReader, Cursor, Read, Seek, SeekFrom, Write},
    path::{Path as FsPath, PathBuf},
    str,
    time::{SystemTime, UNIX_EPOCH},
};

use flate2::read::MultiGzDecoder;
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Sha256Hash, Strand};
use gen_models::{
    assets::{AssetRef, AssetRole, AssetUri, ChecksummedReader, ChecksummedWriter, LocalAssetUri},
    block_group::{BlockGroup, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::DbContext,
    edge::Edge,
    errors::{CollectionError, SampleError},
    file_types::FileTypes,
    node::Node,
    operations::{OperationFile, OperationInfo, OperationSummary},
    path::Path,
    sample::Sample,
    sequence::Sequence,
};
use noodles::{
    bgzf::{self, gzi},
    fasta,
};
use tempfile::NamedTempFile;

use crate::{
    fasta::FastaError,
    progress_bar::{add_saving_operation_bar, get_handler, get_progress_bar},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FastaInputKind {
    Plain,
    Gzip,
    Bgzf,
}

struct ArchivedFasta {
    path: PathBuf,
    checksum: Sha256Hash,
    materialized_checksum: Option<Sha256Hash>,
}

type ReplayedFastaInput<R> = std::io::Chain<Cursor<Vec<u8>>, R>;
#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(context, fasta, collection_name, sample))
)]
pub fn import_fasta(
    context: &DbContext,
    fasta: &str,
    collection_name: &str,
    sample: &str,
    indexes: &[String],
) -> Result<OperationSummary, FastaError> {
    let conn = context.graph().conn();
    let progress_bar = get_handler();
    let source_reader =
        <dyn AssetUri>::new(context.workspace(), fasta).reader(context.workspace())?;
    let (input_kind, source_reader) = sniff_fasta_input(source_reader)?;
    let input_is_bgzf = input_kind == FastaInputKind::Bgzf;
    let input_is_remote = !LocalAssetUri::is_local_path_or_file_uri(fasta);
    let local_fasta_path = local_fasta_path(context, fasta)?;
    let explicit_fai = indexes
        .iter()
        .find(|index| index_extension(index).as_deref() == Some("fai"))
        .cloned();
    let explicit_gzi = indexes
        .iter()
        .find(|index| index_extension(index).as_deref() == Some("gzi"))
        .cloned();

    let remote_indexed_bgzf = input_is_remote
        && input_is_bgzf
        && explicit_fai
            .as_deref()
            .is_some_and(|index| !LocalAssetUri::is_local_path_or_file_uri(index))
        && explicit_gzi
            .as_deref()
            .is_some_and(|index| !LocalAssetUri::is_local_path_or_file_uri(index));

    let archived_fasta = if remote_indexed_bgzf {
        drop(source_reader);
        None
    } else {
        Some(archive_as_bgzf(
            context,
            source_reader,
            input_kind,
            !input_is_remote,
        )?)
    };
    let sequence_name = uri_basename(fasta);

    let sibling_fai = local_fasta_path
        .as_ref()
        .map(|path| PathBuf::from(format!("{}.fai", path.display())))
        .filter(|path| path.is_file())
        .map(|path| path.to_string_lossy().into_owned());
    let sibling_gzi = if input_is_bgzf {
        local_fasta_path
            .as_ref()
            .map(|path| PathBuf::from(format!("{}.gzi", path.display())))
            .filter(|path| path.is_file())
            .map(|path| path.to_string_lossy().into_owned())
    } else {
        None
    };
    let fai_source = explicit_fai.or(sibling_fai);
    let gzi_source = if input_is_bgzf {
        explicit_gzi.or(sibling_gzi)
    } else {
        None
    };

    let (sequence_path, sequence_asset_name, sequence_checksum) = match archived_fasta.as_ref() {
        Some(archive) => (
            archive.path.to_string_lossy().into_owned(),
            sequence_name.clone(),
            Some(archive.checksum),
        ),
        None => (fasta.to_owned(), sequence_name.clone(), None),
    };
    let mut sequence_operation_file =
        OperationFile::new(sequence_path).set_file_type(FileTypes::Fasta);
    sequence_operation_file.filename = sequence_asset_name.clone();
    if let Some(checksum) = sequence_checksum {
        sequence_operation_file = sequence_operation_file.set_checksum_override(checksum);
    }
    if let Some(materialized_checksum) = archived_fasta
        .as_ref()
        .and_then(|archive| archive.materialized_checksum)
    {
        sequence_operation_file =
            sequence_operation_file.set_materialized_checksum_override(materialized_checksum);
    }
    if local_fasta_path.is_some() {
        sequence_operation_file = sequence_operation_file.set_logical_path_override(
            LocalAssetUri::source_logical_path(context.workspace(), fasta)?,
        );
    }

    let (fai_index, fai_path) = match fai_source {
        Some(path) => (read_fai_index(context, &path)?, path),
        None => {
            let archive_path = archived_fasta
                .as_ref()
                .map(|archive| &archive.path)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "remote BGZF FASTA requires an explicit FASTA index",
                    )
                })?;
            let index = build_fai_index(archive_path)?;
            let path =
                stage_generated_index(context, "fa.bgz.fai", |file| write_fai_index(file, &index))?;
            (index, path.to_string_lossy().into_owned())
        }
    };

    let gzi_path = match gzi_source {
        Some(path) => path,
        None => {
            // Build missing GZI offsets against retained BGZF bytes; an ordinary-gzip index
            // cannot be reused after recompression.
            let archive_path = archived_fasta
                .as_ref()
                .map(|archive| &archive.path)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "remote BGZF FASTA requires an explicit gzip index",
                    )
                })?;
            let index = build_gzi_index(archive_path)?;
            let path = stage_generated_index(context, "fa.bgz.gzi", |file| {
                gzi::io::Writer::new(file).write_index(&index)
            })?;
            path.to_string_lossy().into_owned()
        }
    };

    let created_on = operation_created_on()?;
    let sequence_asset_ref =
        prepare_asset_ref(conn, context, &mut sequence_operation_file, created_on)?;

    let mut operation_files = vec![sequence_operation_file];
    let mut sequence_index_files = vec![
        (fai_path, format!("{sequence_name}.fai")),
        (gzi_path, format!("{sequence_name}.gzi")),
    ];
    for index in indexes {
        let extension = index_extension(index);
        if matches!(extension.as_deref(), Some("fai" | "gzi")) {
            continue;
        }
        sequence_index_files.push((index.clone(), uri_basename(index)));
    }
    let mut seen_index_paths = std::collections::HashSet::new();
    for (index_path, index_name) in sequence_index_files {
        if !seen_index_paths.insert(index_path.clone()) {
            continue;
        }
        let mut index_operation_file = OperationFile::new(index_path)
            .set_file_type(FileTypes::None)
            .set_role(AssetRole::SequenceIndex)
            .set_upstream_asset_ref_id(&sequence_asset_ref.id);
        index_operation_file.filename = index_name;
        prepare_asset_ref(conn, context, &mut index_operation_file, created_on)?;
        operation_files.push(index_operation_file);
    }

    let collection = match Collection::create(conn, collection_name) {
        Ok(collection) => collection,
        Err(CollectionError::Duplicate(collection)) => collection,
        Err(e) => return Err(FastaError::CollectionError(e)),
    };

    match Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: sample,
            ..Default::default()
        },
    ) {
        Ok(_) => {}
        Err(SampleError::Duplicate(_)) => {}
        Err(e) => {
            return Err(FastaError::SampleError(e));
        }
    }
    let mut summary: HashMap<String, i64> = HashMap::new();

    let _ = progress_bar.println("Parsing Fasta");
    let bar = progress_bar.add(get_progress_bar(None));
    bar.set_message("Entries Processed.");
    // FAI records provide names and lengths without allocating each large sequence as a String.
    for record in fai_index.as_ref() {
        let name = str::from_utf8(record.name())
            .map_err(|error| {
                FastaError::IOError(io::Error::new(io::ErrorKind::InvalidData, error))
            })?
            .to_string();
        let sequence_length = i64::try_from(record.length()).map_err(|error| {
            FastaError::IOError(io::Error::new(io::ErrorKind::InvalidData, error))
        })?;
        let seq = Sequence::new()
            .sequence_type("DNA")
            .name(&name)
            .asset_ref_id(Some(&sequence_asset_ref.id))
            .length(sequence_length)
            .save(conn)?;
        let node_id = Node::create(
            conn,
            &seq.hash,
            &HashId::convert_str(&format!(
                "{collection}.{name}:{hash}",
                collection = collection.name,
                hash = seq.hash
            )),
        )?;
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: &collection.name,
                sample_name: sample,
                name: &name,
                ..Default::default()
            },
        )?;
        let edge_into = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_id,
            0,
            Strand::Forward,
        )?;
        let edge_out_of = Edge::create(
            conn,
            node_id,
            sequence_length,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )?;

        let new_block_group_edges = vec![
            BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: edge_into.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: edge_out_of.id,
                chromosome_index: 0,
                phased: 0,
            },
        ];

        BlockGroupEdge::bulk_create(conn, &new_block_group_edges);
        let path = Path::create(
            conn,
            &name,
            &block_group.id,
            &[edge_into.id, edge_out_of.id],
        )?;
        summary.entry(path.name).or_insert(sequence_length);
        bar.inc(1);
    }
    bar.finish();
    let mut summary_str = "".to_string();
    for (path_name, change_count) in summary.iter() {
        summary_str.push_str(&format!(" {path_name}: {change_count} changes.\n"));
    }

    let bar = add_saving_operation_bar(&progress_bar);
    let operation_summary = OperationSummary::new(
        OperationInfo {
            files: operation_files,
            description: "fasta_addition".to_string(),
        },
        summary_str,
    );
    bar.finish();
    Ok(operation_summary)
}

fn operation_created_on() -> Result<i64, FastaError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?;
    i64::try_from(duration.as_nanos())
        .map_err(io::Error::other)
        .map_err(Into::into)
}

fn sniff_fasta_input<R: Read>(
    mut reader: R,
) -> io::Result<(FastaInputKind, ReplayedFastaInput<R>)> {
    let mut prefix = Vec::with_capacity(12);
    let kind = if !fill_prefix(&mut reader, &mut prefix, 2)? || prefix[..2] != [0x1f, 0x8b] {
        FastaInputKind::Plain
    } else if !fill_prefix(&mut reader, &mut prefix, 10)?
        || prefix[2] != 8
        || prefix[3] & 0x04 == 0
        || !fill_prefix(&mut reader, &mut prefix, 12)?
    {
        FastaInputKind::Gzip
    } else {
        let extra_length = usize::from(u16::from_le_bytes([prefix[10], prefix[11]]));
        if !fill_prefix(&mut reader, &mut prefix, 12 + extra_length)? {
            FastaInputKind::Gzip
        } else if has_bgzf_subfield(&prefix[12..]) {
            FastaInputKind::Bgzf
        } else {
            FastaInputKind::Gzip
        }
    };

    Ok((kind, Cursor::new(prefix).chain(reader)))
}

fn fill_prefix<R: Read>(reader: &mut R, prefix: &mut Vec<u8>, target: usize) -> io::Result<bool> {
    let mut buffer = [0; 8192];
    while prefix.len() < target {
        let remaining = target - prefix.len();
        let buffer_length = buffer.len();
        let read_length = remaining.min(buffer_length);
        let read = match reader.read(&mut buffer[..read_length]) {
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if read == 0 {
            return Ok(false);
        }
        prefix.extend_from_slice(&buffer[..read]);
    }
    Ok(true)
}

fn has_bgzf_subfield(extra: &[u8]) -> bool {
    let mut offset = 0;
    while offset + 4 <= extra.len() {
        let subfield_length =
            usize::from(u16::from_le_bytes([extra[offset + 2], extra[offset + 3]]));
        let end = offset + 4 + subfield_length;
        if end > extra.len() {
            return false;
        }
        if extra[offset] == b'B' && extra[offset + 1] == b'C' && subfield_length == 2 {
            return true;
        }
        offset = end;
    }
    false
}

fn archive_as_bgzf(
    context: &DbContext,
    input: impl Read + 'static,
    input_kind: FastaInputKind,
    retain_materialized_checksum: bool,
) -> Result<ArchivedFasta, FastaError> {
    let asset_dir = context
        .workspace()
        .asset_dir()
        .map_err(gen_models::errors::FileAdditionError::ConfigError)?;
    fs::create_dir_all(&asset_dir)?;
    let mut staged_file = NamedTempFile::new_in(&asset_dir)?;
    let (checksum, materialized_checksum) = match input_kind {
        FastaInputKind::Bgzf => (copy_with_checksum(input, staged_file.as_file_mut())?, None),
        FastaInputKind::Gzip => (
            write_bgzf_with_checksum(MultiGzDecoder::new(input), staged_file.as_file_mut())?,
            None,
        ),
        FastaInputKind::Plain if retain_materialized_checksum => {
            let input = ChecksummedReader::new(input);
            let source_checksum = input.checksum_handle();
            let archive_checksum = write_bgzf_with_checksum(input, staged_file.as_file_mut())?;
            let source_checksum = source_checksum
                .checksum()
                .ok_or_else(|| io::Error::other("plain FASTA source stream did not reach EOF"))?;
            (archive_checksum, Some(source_checksum))
        }
        FastaInputKind::Plain => (
            write_bgzf_with_checksum(input, staged_file.as_file_mut())?,
            None,
        ),
    };
    staged_file.flush()?;

    let archived_path = asset_dir.join(format!("{checksum}.fa.bgz"));
    match staged_file.persist_noclobber(&archived_path) {
        Ok(_) => {}
        Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.error.into()),
    }

    Ok(ArchivedFasta {
        path: archived_path,
        checksum,
        materialized_checksum,
    })
}

fn copy_with_checksum(input: impl Read, output: &mut File) -> io::Result<Sha256Hash> {
    let mut checksummed_writer = ChecksummedWriter::new(output);
    let mut input = input;
    io::copy(&mut input, &mut checksummed_writer)?;
    checksummed_writer.flush()?;
    Ok(checksummed_writer.checksum())
}

fn write_bgzf_with_checksum(input: impl Read, output: &mut File) -> io::Result<Sha256Hash> {
    let checksummed_writer = ChecksummedWriter::new(output);
    let mut writer = bgzf::io::Writer::new(checksummed_writer);
    let mut input = input;
    io::copy(&mut input, &mut writer)?;
    let mut checksummed_writer = writer.finish()?;
    checksummed_writer.flush()?;
    Ok(checksummed_writer.checksum())
}

fn stage_generated_index(
    context: &DbContext,
    suffix: &str,
    write_index: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> Result<PathBuf, FastaError> {
    let asset_dir = context
        .workspace()
        .asset_dir()
        .map_err(gen_models::errors::FileAdditionError::ConfigError)?;
    fs::create_dir_all(&asset_dir)?;
    let mut staged_file = NamedTempFile::new_in(&asset_dir)?;
    let checksum = {
        let mut checksummed_writer = ChecksummedWriter::new(staged_file.as_file_mut());
        write_index(&mut checksummed_writer)?;
        checksummed_writer.flush()?;
        checksummed_writer.checksum()
    };
    let index_path = asset_dir.join(format!("{checksum}.{suffix}"));
    match staged_file.persist_noclobber(&index_path) {
        Ok(_) => {}
        Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.error.into()),
    }
    Ok(index_path)
}

fn local_fasta_path(context: &DbContext, fasta: &str) -> Result<Option<PathBuf>, FastaError> {
    if !LocalAssetUri::is_local_path_or_file_uri(fasta) {
        return Ok(None);
    }

    let path = LocalAssetUri::path_from_uri(fasta).unwrap_or_else(|| fasta.to_string());
    let path = PathBuf::from(path);
    if path.is_absolute() {
        return Ok(Some(path));
    }

    let repo_root = context
        .workspace()
        .repo_root()
        .map_err(gen_models::errors::FileAdditionError::ConfigError)?;
    Ok(Some(repo_root.join(path)))
}

fn index_extension(index: &str) -> Option<String> {
    <dyn AssetUri>::from_uri(index)
        .suffix()
        .and_then(|suffix| suffix.rsplit('.').next().map(str::to_string))
}

fn uri_basename(path_or_uri: &str) -> String {
    let path = path_or_uri.split(['?', '#']).next().unwrap_or(path_or_uri);
    path.rsplit('/').next().unwrap_or(path).to_string()
}

fn build_fai_index(path: &FsPath) -> Result<fasta::fai::Index, FastaError> {
    let file = File::open(path)?;
    let reader = BufReader::new(bgzf::io::Reader::new(file));
    let mut indexer = fasta::io::Indexer::new(reader);
    let mut records = Vec::new();
    while let Some(record) = indexer
        .index_record()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?
    {
        records.push(record);
    }
    Ok(fasta::fai::Index::from(records))
}

fn read_fai_index(context: &DbContext, index_path: &str) -> Result<fasta::fai::Index, FastaError> {
    let reader =
        <dyn AssetUri>::new(context.workspace(), index_path).reader(context.workspace())?;
    Ok(fasta::fai::io::Reader::new(BufReader::new(reader)).read_index()?)
}

fn write_fai_index(file: &mut dyn Write, index: &fasta::fai::Index) -> io::Result<()> {
    fasta::fai::io::Writer::new(file).write_index(index)
}

fn build_gzi_index(path: &FsPath) -> Result<gzi::Index, FastaError> {
    let mut file = File::open(path)?;
    let file_length = file.metadata()?.len();
    let mut compressed_offset = 0_u64;
    let mut uncompressed_offset = 0_u64;
    let mut records = Vec::new();

    while compressed_offset < file_length {
        file.seek(SeekFrom::Start(compressed_offset))?;
        let mut header = [0; 12];
        file.read_exact(&mut header)?;
        if header[0] != 0x1f || header[1] != 0x8b || header[2] != 8 || header[3] & 0x04 == 0 {
            return Err(
                io::Error::new(io::ErrorKind::InvalidData, "invalid BGZF block header").into(),
            );
        }

        let extra_length = u64::from(u16::from_le_bytes([header[10], header[11]]));
        let mut extra = vec![0; usize::try_from(extra_length).map_err(io::Error::other)?];
        file.read_exact(&mut extra)?;
        let mut block_size = None;
        let mut offset = 0;
        while offset + 4 <= extra.len() {
            let subfield_length =
                usize::from(u16::from_le_bytes([extra[offset + 2], extra[offset + 3]]));
            let end = offset + 4 + subfield_length;
            if end > extra.len() {
                return Err(
                    io::Error::new(io::ErrorKind::InvalidData, "invalid BGZF extra field").into(),
                );
            }
            if extra[offset] == b'B' && extra[offset + 1] == b'C' && subfield_length == 2 {
                block_size =
                    Some(u64::from(u16::from_le_bytes([extra[offset + 4], extra[offset + 5]])) + 1);
                break;
            }
            offset = end;
        }
        let block_size = block_size.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "BGZF block has no BC field")
        })?;
        let header_size = 12 + extra_length;
        if block_size < header_size + 8 || compressed_offset + block_size > file_length {
            return Err(
                io::Error::new(io::ErrorKind::InvalidData, "invalid BGZF block size").into(),
            );
        }

        file.seek(SeekFrom::Start(compressed_offset + block_size - 4))?;
        let mut isize = [0; 4];
        file.read_exact(&mut isize)?;
        let block_uncompressed_size = u64::from(u32::from_le_bytes(isize));
        if compressed_offset > 0 && block_uncompressed_size > 0 {
            records.push((compressed_offset, uncompressed_offset));
        }
        compressed_offset += block_size;
        uncompressed_offset += block_uncompressed_size;
    }

    Ok(gzi::Index::from(records))
}

fn prepare_asset_ref(
    conn: &gen_models::db::GraphConnection,
    context: &DbContext,
    operation_file: &mut OperationFile,
    created_on: i64,
) -> Result<AssetRef, FastaError> {
    let asset_ref = operation_file.prepare_asset_ref(context.workspace(), created_on)?;
    if let Some(checksum) = asset_ref.checksum {
        *operation_file = operation_file.clone().set_checksum_override(checksum);
    }
    AssetRef::create(conn, &asset_ref)
        .map_err(gen_models::errors::FileAdditionError::DatabaseError)?;
    Ok(asset_ref)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        fs,
        io::{self, Cursor, Read, Write as _},
        net::TcpListener,
        path::PathBuf,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        thread,
        time::Duration,
    };

    use gen_models::{
        assets::{AssetRef, AssetRole, OperationAsset, OperationKind, OperationLog},
        block_group::BlockGroup,
        errors::OperationError,
        history::{HistoryStore, dolt::DoltHistoryStore},
        node::Node,
        operations::{calculate_reader_checksum, commit_operation_summary},
        path::Path,
        sample::Sample,
        sequence::Sequence,
    };
    use noodles::{bgzf, bgzf::gzi, fasta};

    use super::{FastaInputKind, import_fasta, sniff_fasta_input};
    use crate::test_helpers::{setup_gen, setup_gen_on_disk};

    struct TestHttpServer {
        address: String,
        requests: Arc<Mutex<Vec<String>>>,
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl TestHttpServer {
        fn new(files: HashMap<String, Vec<u8>>) -> Self {
            let listener =
                TcpListener::bind(("127.0.0.1", 0)).expect("should bind remote FASTA test server");
            listener
                .set_nonblocking(true)
                .expect("should configure remote FASTA test server");
            let address = listener
                .local_addr()
                .expect("should read remote FASTA server address")
                .to_string();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let server_requests = Arc::clone(&requests);
            let stop = Arc::new(AtomicBool::new(false));
            let server_stop = Arc::clone(&stop);
            let handle = thread::spawn(move || {
                while !server_stop.load(Ordering::Relaxed) {
                    let Ok((mut stream, _)) = listener.accept() else {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    };
                    let mut request_bytes = [0_u8; 8192];
                    let length = stream
                        .read(&mut request_bytes)
                        .expect("should read remote FASTA request");
                    let request = String::from_utf8_lossy(&request_bytes[..length]).to_string();
                    server_requests
                        .lock()
                        .expect("should lock remote FASTA request log")
                        .push(request.clone());
                    let mut request_lines = request.lines();
                    let request_line = request_lines.next().unwrap_or_default();
                    let mut request_parts = request_line.split_whitespace();
                    let method = request_parts.next().unwrap_or_default();
                    let path = request_parts
                        .next()
                        .unwrap_or_default()
                        .split('?')
                        .next()
                        .unwrap_or_default();
                    let Some(contents) = files.get(path) else {
                        stream
                            .write_all(
                                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                            )
                            .expect("should write missing remote FASTA response");
                        continue;
                    };
                    let range = request_lines.find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("range: bytes=")
                            .map(str::to_string)
                    });
                    let (status, start, end) = if let Some(range) = range {
                        let (start, end) = range
                            .split_once('-')
                            .expect("should parse remote FASTA byte range");
                        let start = start
                            .parse::<usize>()
                            .expect("should parse remote FASTA range start");
                        let end = if end.is_empty() {
                            contents.len().saturating_sub(1)
                        } else {
                            end.parse::<usize>()
                                .expect("should parse remote FASTA range end")
                                .min(contents.len().saturating_sub(1))
                        };
                        ("206 Partial Content", start, end)
                    } else {
                        ("200 OK", 0, contents.len().saturating_sub(1))
                    };
                    let body = if contents.is_empty() || start > end {
                        &[][..]
                    } else {
                        &contents[start..=end]
                    };
                    let content_range = if status.starts_with("206") {
                        format!("Content-Range: bytes {start}-{end}/{}\r\n", contents.len())
                    } else {
                        String::new()
                    };
                    write!(
                        stream,
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{content_range}Accept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .expect("should write remote FASTA response headers");
                    if method != "HEAD" {
                        stream
                            .write_all(body)
                            .expect("should write remote FASTA response body");
                    }
                }
            });
            Self {
                address,
                requests,
                stop,
                handle: Some(handle),
            }
        }

        fn url(&self, path: &str) -> String {
            format!("http://{}{path}", self.address)
        }

        fn clear_requests(&self) {
            self.requests
                .lock()
                .expect("should lock remote FASTA request log")
                .clear();
        }

        fn requests(&self) -> Vec<String> {
            self.requests
                .lock()
                .expect("should lock remote FASTA request log")
                .clone()
        }
    }

    impl Drop for TestHttpServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                handle.join().expect("should stop remote FASTA test server");
            }
        }
    }

    #[test]
    fn test_add_fasta() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let history_store = DoltHistoryStore::new(conn);

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");

        let operation_summary = import_fasta(
            &context,
            fasta_path.to_str().unwrap(),
            "test",
            Sample::DEFAULT_NAME,
            &[],
        )
        .unwrap();
        let commit_hash = commit_operation_summary(&context, &operation_summary).unwrap();
        assert_eq!(history_store.current_head().unwrap(), Some(commit_hash));
        let mut operation_logs = OperationLog::all(conn).expect("should load operation logs");
        operation_logs.sort_by_key(|operation_log| std::cmp::Reverse(operation_log.created_on));
        assert_eq!(
            operation_logs[0].operation_kind,
            OperationKind::Other("fasta_addition".to_string())
        );
        let asset_refs = AssetRef::all(conn).expect("should load asset references");
        assert_eq!(asset_refs.len(), 3);
        let sequence_asset = asset_refs
            .iter()
            .find(|asset_ref| asset_ref.role == AssetRole::Input)
            .expect("should retain one archived FASTA asset");
        assert!(sequence_asset.uri.starts_with("file://.gen/assets/"));
        assert!(sequence_asset.uri.ends_with(".fa.bgz"));
        assert_eq!(sequence_asset.name.as_deref(), Some("simple.fa"));
        assert_eq!(
            sequence_asset.logical_path.as_deref(),
            Some(".gen/outside_root/simple.fa")
        );
        assert!(sequence_asset.materialized_checksum.is_some());
        let index_assets = AssetRef::get_derived_assets(conn, &sequence_asset.id, None);
        assert_eq!(index_assets.len(), 2);
        assert!(
            index_assets
                .iter()
                .any(|asset_ref| { asset_ref.name.as_deref() == Some("simple.fa.fai") })
        );
        assert!(
            index_assets
                .iter()
                .any(|asset_ref| { asset_ref.name.as_deref() == Some("simple.fa.gzi") })
        );

        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, context.workspace(), &block_group_id, false)
                .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()]),
            "FASTA should read through retained sequence and index assets"
        );

        let path = Path::all(conn).expect("should load paths")[0].clone();
        assert_eq!(
            path.sequence(conn, context.workspace(), None).unwrap(),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()
        );
    }

    #[test]
    fn test_supports_normal_gz_fasta() {
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/gzipped.fa.gz");

        import_fasta(
            &context,
            fasta_path.to_str().unwrap(),
            "test",
            Sample::DEFAULT_NAME,
            &[],
        )
        .unwrap();
        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, context.workspace(), &block_group_id, false)
                .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
    }

    #[test]
    fn test_large_gz_fasta() {
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/chr22.fa.gz");

        import_fasta(
            &context,
            fasta_path.to_str().unwrap(),
            "test",
            Sample::DEFAULT_NAME,
            &[],
        )
        .unwrap();
        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "chr22", None);
        let sequences = Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id);
        let dna = sequences
            .iter()
            .filter(|s| s.sequence_type == "DNA")
            .collect::<Vec<_>>();
        assert_eq!(dna[0].length, 51304566);
    }

    #[test]
    fn test_supports_bgzip_fasta() {
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/bgzipped.fa.bgz");

        import_fasta(
            &context,
            fasta_path.to_str().unwrap(),
            "test",
            Sample::DEFAULT_NAME,
            &[],
        )
        .unwrap();
        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
        let sequence = Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
            .into_iter()
            .find(|sequence| sequence.external_sequence)
            .expect("should retain a BGZF sequence asset");
        let sequence_asset = AssetRef::select(conn)
            .get_by_id(sequence.asset_ref_id.unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            sequence_asset.checksum,
            Some(
                calculate_reader_checksum(sequence_asset.reader(context.workspace()).unwrap())
                    .unwrap()
            ),
            "native BGZF asset checksum should include every archived byte"
        );
        assert_eq!(
            BlockGroup::get_all_sequences(conn, context.workspace(), &block_group_id, false)
                .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
    }

    #[test]
    fn test_plain_and_gzip_inputs_are_archived_with_generated_indexes() {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let repo_root = context.workspace().repo_root().unwrap();
        let mut state = 0x1234_5678_u32;
        let expected_sequence = (0..200_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                b"ACGT"[(state & 0b11) as usize] as char
            })
            .collect::<String>();
        let fasta_contents = format!(">large\n{expected_sequence}\n").into_bytes();
        let mut gzip_encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gzip_encoder.write_all(&fasta_contents).unwrap();
        let gzip_contents = gzip_encoder.finish().unwrap();

        for (filename, sample_name, source_contents) in [
            ("plain.fa.bgz", "plain", fasta_contents.clone()),
            ("ordinary.fa.gz", "ordinary-gzip", gzip_contents),
        ] {
            let fasta_path = repo_root.join(filename);
            fs::write(&fasta_path, &source_contents).unwrap();
            let operation_summary = import_fasta(
                &context,
                fasta_path.to_string_lossy().as_ref(),
                "test",
                sample_name,
                &[],
            )
            .expect("should import the FASTA as an external BGZF asset");
            commit_operation_summary(&context, &operation_summary)
                .expect("should commit the archived FASTA and its indexes");
            fs::remove_file(&fasta_path).unwrap();

            let block_group_id = BlockGroup::get_id("test", sample_name, "large", None);
            let sequence =
                Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
                    .into_iter()
                    .find(|sequence| sequence.external_sequence)
                    .expect("should store the sequence by external asset reference");
            let inline_sequence = conn
                .query_row(
                    "SELECT sequence FROM sequences WHERE hash = ?1",
                    [sequence.hash],
                    |row| row.get::<_, String>(0),
                )
                .unwrap();
            assert!(
                inline_sequence.is_empty(),
                "external sequences should not be duplicated in SQLite"
            );

            let sequence_asset = AssetRef::select(conn)
                .get_by_id(sequence.asset_ref_id.unwrap())
                .unwrap()
                .unwrap();
            assert!(sequence_asset.uri.starts_with("file://.gen/assets/"));
            assert!(sequence_asset.uri.ends_with(".fa.bgz"));
            assert_eq!(sequence_asset.name.as_deref(), Some(filename));
            assert_eq!(
                sequence_asset.checksum,
                Some(
                    calculate_reader_checksum(sequence_asset.reader(context.workspace()).unwrap())
                        .expect("should checksum the complete archived BGZF bytes")
                ),
                "archived checksum should include BGZF final blocks"
            );
            assert_eq!(
                sequence_asset.materialized_checksum,
                (sample_name == "plain")
                    .then(|| calculate_reader_checksum(std::io::Cursor::new(&source_contents)))
                    .transpose()
                    .expect("should checksum original plain FASTA bytes")
            );
            assert_eq!(sequence_asset.logical_path.as_deref(), Some(filename));

            let mut archived_fasta =
                bgzf::io::Reader::new(sequence_asset.reader(context.workspace()).unwrap());
            let mut archived_contents = Vec::new();
            archived_fasta.read_to_end(&mut archived_contents).unwrap();
            assert_eq!(archived_contents, fasta_contents);

            let index_assets = AssetRef::get_derived_assets(conn, &sequence_asset.id, None);
            assert_eq!(
                index_assets.len(),
                2,
                "imports should retain one FAI and one GZI"
            );
            let fai_asset = index_assets
                .iter()
                .find(|asset_ref| {
                    asset_ref
                        .name
                        .as_deref()
                        .is_some_and(|name| name.ends_with(".fai"))
                })
                .expect("should retain a linked FASTA index");
            let gzi_asset = index_assets
                .iter()
                .find(|asset_ref| {
                    asset_ref
                        .name
                        .as_deref()
                        .is_some_and(|name| name.ends_with(".gzi"))
                })
                .expect("should retain a linked BGZF index");
            assert_eq!(
                fai_asset.checksum,
                Some(
                    calculate_reader_checksum(fai_asset.reader(context.workspace()).unwrap())
                        .expect("should checksum the complete FAI bytes")
                ),
                "FAI checksum should match the finalized index file"
            );
            assert_eq!(
                gzi_asset.checksum,
                Some(
                    calculate_reader_checksum(gzi_asset.reader(context.workspace()).unwrap())
                        .expect("should checksum the complete GZI bytes")
                ),
                "GZI checksum should match the finalized index file"
            );
            assert!(
                index_assets
                    .iter()
                    .all(|asset_ref| asset_ref.upstream_asset_ref_id == Some(sequence_asset.id))
            );
            assert!(
                index_assets
                    .iter()
                    .all(|asset_ref| asset_ref.role == AssetRole::SequenceIndex)
            );
            let fai_index =
                fasta::fai::io::Reader::new(fai_asset.reader(context.workspace()).unwrap())
                    .read_index()
                    .unwrap();
            assert_eq!(fai_index.as_ref()[0].name(), b"large");
            assert_eq!(
                fai_index.as_ref()[0].length(),
                expected_sequence.len() as u64
            );
            let gzi_index = gzi::io::Reader::new(gzi_asset.reader(context.workspace()).unwrap())
                .read_index()
                .unwrap();
            assert!(
                !gzi_index.as_ref().is_empty(),
                "multi-block BGZF needs GZI offsets"
            );
            let bgzf_reader = bgzf::io::indexed_reader::Builder::default()
                .set_index(gzi_index)
                .build_from_reader(sequence_asset.reader(context.workspace()).unwrap())
                .unwrap();
            let mut indexed_fasta_reader = fasta::io::indexed_reader::Builder::default()
                .set_index(fai_index)
                .build_from_reader(bgzf_reader)
                .unwrap();
            let region = "large:131001-131200"
                .parse::<noodles::core::Region>()
                .unwrap();
            let indexed_record = indexed_fasta_reader.query(&region).unwrap();
            assert_eq!(
                indexed_record.sequence().as_ref(),
                &expected_sequence.as_bytes()[131_000..131_200],
                "indexed reads beyond the first two BGZF blocks should use generated GZI offsets"
            );

            let export_path = repo_root.join(format!("{sample_name}.fa"));
            crate::exports::fasta::export_fasta(
                conn,
                context.workspace(),
                "test",
                Some(sample_name),
                &export_path,
                None,
            )
            .unwrap();
            let mut exported = fasta::io::reader::Builder
                .build_from_path(export_path)
                .unwrap();
            let exported_record = exported.records().next().unwrap().unwrap();
            assert_eq!(
                exported_record.sequence().as_ref(),
                expected_sequence.as_bytes(),
                "FASTA export should work after deleting the original input"
            );
        }

        assert_eq!(
            fs::read_dir(context.workspace().asset_dir().unwrap())
                .unwrap()
                .count(),
            3,
            "a new plain import should retain only the BGZF, FAI, and GZI files"
        );
    }

    #[test]
    fn test_add_fasta_creates_sample() {
        let context = setup_gen();
        let conn = context.graph().conn();

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");

        import_fasta(
            &context,
            fasta_path.to_str().unwrap(),
            "test",
            "new-sample",
            &[],
        )
        .unwrap();
        let block_group_id = BlockGroup::get_id("test", "new-sample", "m123", None);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, context.workspace(), &block_group_id, false)
                .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );

        let path = Path::all(conn).expect("should load paths")[0].clone();
        assert_eq!(
            path.sequence(conn, context.workspace(), None).unwrap(),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()
        );
        assert_eq!(
            Sample::get_by_name(conn, "new-sample").unwrap().name,
            "new-sample"
        );
    }

    #[test]
    fn test_add_fasta_uses_retained_bgzf_and_reused_indexes_by_default() {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();

        let fasta_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("indexed.fa.bgz");
        let index_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("indexed.fa.bgz.fai");
        let gzip_index_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("indexed.fa.bgz.gzi");
        fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/bgzipped.fa.bgz"),
            &fasta_path,
        )
        .unwrap();
        fs::write(&index_path, "m123\t34\t6\t34\t35\n").unwrap();
        gzi::fs::write(&gzip_index_path, &gzi::Index::default()).unwrap();

        let operation_summary = import_fasta(
            &context,
            fasta_path.to_str().unwrap(),
            "test",
            Sample::DEFAULT_NAME,
            &[],
        )
        .unwrap();
        commit_operation_summary(&context, &operation_summary).unwrap();
        fs::remove_file(&fasta_path).unwrap();
        fs::write(&index_path, "invalid logical-path index\n").unwrap();
        fs::write(&gzip_index_path, "invalid logical-path gzip index\n").unwrap();
        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, context.workspace(), &block_group_id, false)
                .expect("should load sequence from retained sequence and index assets"),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()]),
            "FASTA should read through retained sequence and index assets"
        );
        let sequence = Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
            .into_iter()
            .find(|sequence| sequence.asset_ref_id.is_some())
            .expect("should persist the external sequence AssetRef pointer");
        let asset_ref = AssetRef::select(conn)
            .get_by_id(
                sequence
                    .asset_ref_id
                    .expect("should have sequence AssetRef"),
            )
            .expect("should query sequence AssetRef")
            .expect("should persist sequence AssetRef");
        assert!(
            asset_ref
                .versioned_store_path(context.workspace())
                .expect("should resolve immutable sequence asset")
                .is_file(),
            "immutable sequence asset should remain after logical file removal"
        );

        let path = Path::all(conn).expect("should load paths")[0].clone();
        assert_eq!(
            path.sequence(conn, context.workspace(), None).unwrap(),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "path sequence should use the immutable FASTA asset"
        );
        let sequence = Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
            .into_iter()
            .find(|sequence| sequence.asset_ref_id.is_some())
            .expect("should store the shallow sequence asset pointer");
        let asset_refs = AssetRef::all(conn).expect("should load asset references");
        assert!(
            asset_refs.iter().any(|asset_ref| {
                Some(asset_ref.id) == sequence.asset_ref_id && asset_ref.role == AssetRole::Input
            }),
            "external sequence should point to its archived FASTA AssetRef"
        );
        let mut index_assets = asset_refs
            .iter()
            .filter(|asset_ref| {
                asset_ref.upstream_asset_ref_id == sequence.asset_ref_id
                    && asset_ref.role == AssetRole::SequenceIndex
            })
            .collect::<Vec<_>>();
        index_assets.sort_by_key(|asset_ref| asset_ref.name.as_deref());
        assert_eq!(
            index_assets.len(),
            2,
            "BGZF FASTA should retain both discovered indexes"
        );
        assert_eq!(
            index_assets[0].name.as_deref(),
            Some("indexed.fa.bgz.fai"),
            "first retained index should be the FASTA index"
        );
        assert_eq!(
            index_assets[1].name.as_deref(),
            Some("indexed.fa.bgz.gzi"),
            "second retained index should be the gzip index"
        );
        let operation_assets = OperationAsset::all(conn).expect("should load operation assets");
        assert!(
            operation_assets.iter().any(|operation_asset| {
                Some(operation_asset.asset_ref_id) == sequence.asset_ref_id
            }),
            "operation should track the archived sequence asset"
        );
        assert!(
            index_assets.iter().all(|index_asset| {
                operation_assets
                    .iter()
                    .any(|operation_asset| operation_asset.asset_ref_id == index_asset.id)
            }),
            "operation should track every retained sequence index"
        );
    }

    #[test]
    fn test_add_fasta_generates_each_missing_sibling_index() {
        for (sample_name, missing_index) in [("fai-only", "gzi"), ("gzi-only", "fai")] {
            let context = setup_gen_on_disk();
            let conn = context.graph().conn();
            let fasta_path = context
                .workspace()
                .repo_root()
                .unwrap()
                .join(format!("{sample_name}.fa.bgz"));
            fs::copy(
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/bgzipped.fa.bgz"),
                &fasta_path,
            )
            .unwrap();

            let fai_path = PathBuf::from(format!("{}.fai", fasta_path.display()));
            let gzi_path = PathBuf::from(format!("{}.gzi", fasta_path.display()));
            let reused_index_bytes = if missing_index == "gzi" {
                let bytes = b"m123\t34\t6\t34\t35\n".to_vec();
                fs::write(&fai_path, &bytes).unwrap();
                bytes
            } else {
                let path = tempfile::NamedTempFile::new().unwrap();
                gzi::fs::write(path.path(), &gzi::Index::default()).unwrap();
                let bytes = fs::read(path.path()).unwrap();
                fs::write(&gzi_path, &bytes).unwrap();
                bytes
            };

            let operation_summary = import_fasta(
                &context,
                fasta_path.to_string_lossy().as_ref(),
                "test",
                sample_name,
                &[],
            )
            .expect("should import BGZF while generating its missing sibling index");
            commit_operation_summary(&context, &operation_summary)
                .expect("should commit reused and generated indexes");

            let block_group_id = BlockGroup::get_id("test", sample_name, "m123", None);
            let sequence =
                Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
                    .into_iter()
                    .find(|sequence| sequence.asset_ref_id.is_some())
                    .expect("should store the sequence by external asset reference");
            let sequence_asset = AssetRef::select(conn)
                .get_by_id(sequence.asset_ref_id.unwrap())
                .unwrap()
                .unwrap();
            let index_assets = AssetRef::get_derived_assets(conn, &sequence_asset.id, None);
            assert_eq!(index_assets.len(), 2, "both indexes should be retained");
            assert!(index_assets.iter().all(|asset_ref| {
                asset_ref.role == AssetRole::SequenceIndex
                    && asset_ref.upstream_asset_ref_id == Some(sequence_asset.id)
            }));

            let fai_asset = index_assets
                .iter()
                .find(|asset_ref| {
                    asset_ref
                        .name
                        .as_deref()
                        .is_some_and(|name| name.ends_with(".fai"))
                })
                .expect("should retain a FASTA index");
            let gzi_asset = index_assets
                .iter()
                .find(|asset_ref| {
                    asset_ref
                        .name
                        .as_deref()
                        .is_some_and(|name| name.ends_with(".gzi"))
                })
                .expect("should retain a BGZF index");
            if missing_index == "gzi" {
                let mut retained_fai_bytes = Vec::new();
                fai_asset
                    .reader(context.workspace())
                    .unwrap()
                    .read_to_end(&mut retained_fai_bytes)
                    .unwrap();
                assert_eq!(
                    retained_fai_bytes, reused_index_bytes,
                    "the existing sibling FAI should be reused byte for byte"
                );
            } else {
                let fai_index =
                    fasta::fai::io::Reader::new(fai_asset.reader(context.workspace()).unwrap())
                        .read_index()
                        .unwrap();
                assert_eq!(fai_index.as_ref()[0].name(), b"m123");
                assert_eq!(fai_index.as_ref()[0].length(), 34);
            }
            if missing_index == "fai" {
                let mut retained_gzi_bytes = Vec::new();
                gzi_asset
                    .reader(context.workspace())
                    .unwrap()
                    .read_to_end(&mut retained_gzi_bytes)
                    .unwrap();
                assert_eq!(
                    retained_gzi_bytes, reused_index_bytes,
                    "the existing sibling GZI should be reused byte for byte"
                );
            } else {
                gzi::io::Reader::new(gzi_asset.reader(context.workspace()).unwrap())
                    .read_index()
                    .unwrap();
            }
        }
    }

    #[test]
    fn test_add_remote_bgzf_fasta_with_multiple_remote_indexes() {
        let fasta_contents = fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/bgzipped.fa.bgz"),
        )
        .expect("should read BGZF FASTA fixture");
        let server = TestHttpServer::new(HashMap::from([
            ("/reference.fa.bgz".to_string(), fasta_contents),
            (
                "/reference.fa.bgz.fai".to_string(),
                b"m123\t34\t6\t34\t35\n".to_vec(),
            ),
            (
                "/reference.fa.bgz.gzi".to_string(),
                0_u64.to_le_bytes().to_vec(),
            ),
        ]));
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let fasta = server.url("/reference.fa.bgz");
        let indexes = [
            server.url("/reference.fa.bgz.fai"),
            server.url("/reference.fa.bgz.gzi"),
        ];

        let operation_summary =
            import_fasta(&context, &fasta, "test", Sample::DEFAULT_NAME, &indexes)
                .expect("should import a remote BGZF with remote indexes");
        let import_requests = server.requests();
        assert_eq!(
            import_requests
                .iter()
                .filter(|request| {
                    let mut request_parts = request
                        .lines()
                        .next()
                        .unwrap_or_default()
                        .split_whitespace();
                    request_parts.next() == Some("GET")
                        && request_parts.next() == Some("/reference.fa.bgz")
                })
                .count(),
            1,
            "classification should read the remote FASTA source with one GET"
        );
        commit_operation_summary(&context, &operation_summary)
            .expect("should commit remote FASTA assets");

        let sequence_asset = AssetRef::all(conn)
            .expect("should load asset references")
            .into_iter()
            .find(|asset_ref| asset_ref.uri == fasta)
            .expect("should store the remote sequence AssetRef");
        assert_eq!(
            sequence_asset.checksum, None,
            "remote sequence AssetRef should remain checksumless"
        );
        let index_assets = AssetRef::get_derived_assets(conn, &sequence_asset.id, None);
        assert_eq!(
            index_assets.len(),
            2,
            "remote BGZF sequence should retain both index AssetRefs"
        );
        assert!(
            index_assets.iter().all(|asset_ref| {
                asset_ref.role == AssetRole::SequenceIndex && asset_ref.checksum.is_none()
            }),
            "remote indexes should remain checksumless sequence-index AssetRefs"
        );
        assert!(
            context
                .workspace()
                .asset_dir()
                .unwrap()
                .read_dir()
                .unwrap()
                .next()
                .is_none(),
            "remote indexed import should not retain local sequence or index assets"
        );

        server.clear_requests();
        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
        let sequence = Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
            .into_iter()
            .find(|sequence| sequence.asset_ref_id == Some(sequence_asset.id))
            .expect("should resolve the remote shallow sequence AssetRef");
        assert_eq!(
            sequence.get_sequence(2, 8).unwrap(),
            "CGATCG",
            "indexed remote lookup should return the requested slice"
        );

        let requests = server.requests();
        assert!(
            requests.iter().any(|request| {
                request.starts_with("GET /reference.fa.bgz.fai ")
                    || request.starts_with("HEAD /reference.fa.bgz.fai ")
            }),
            "indexed lookup should request the remote FASTA index"
        );
        assert!(
            requests.iter().any(|request| {
                request.starts_with("GET /reference.fa.bgz.gzi ")
                    || request.starts_with("HEAD /reference.fa.bgz.gzi ")
            }),
            "indexed lookup should request the remote gzip index"
        );
        assert!(
            requests.iter().any(|request| {
                request.starts_with("GET /reference.fa.bgz ")
                    && request.to_ascii_lowercase().contains("\r\nrange: bytes=")
            }),
            "indexed remote lookup should use a byte-range request"
        );
    }

    #[test]
    fn test_deduplicates_nodes() {
        let context = setup_gen();
        let conn = context.graph().conn();

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let collection = "test".to_string();

        let operation_summary = import_fasta(
            &context,
            fasta_path.to_str().unwrap(),
            &collection,
            Sample::DEFAULT_NAME,
            &[],
        )
        .unwrap();
        commit_operation_summary(&context, &operation_summary).unwrap();
        assert_eq!(
            Node::select(conn)
                .load()
                .expect("should load imported nodes")
                .len(),
            3
        );

        let operation_summary = import_fasta(
            &context,
            fasta_path.to_str().unwrap(),
            &collection,
            Sample::DEFAULT_NAME,
            &[],
        )
        .unwrap();
        let result_error = commit_operation_summary(&context, &operation_summary).unwrap_err();

        assert!(matches!(result_error, OperationError::NoChanges));
    }

    struct ShortReadReader<R> {
        reader: R,
        maximum_read: usize,
    }

    impl<R: Read> Read for ShortReadReader<R> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let buffer_length = buffer.len();
            let read_length = buffer_length.min(self.maximum_read);
            self.reader.read(&mut buffer[..read_length])
        }
    }

    #[test]
    fn test_sniff_replays_short_read_plain_gzip_and_bgzf_inputs() {
        let mut gzip_encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gzip_encoder
            .write_all(b">gzip\nACGT\n")
            .expect("should write gzip FASTA bytes");
        let gzip_contents = gzip_encoder
            .finish()
            .expect("should finish gzip FASTA bytes");
        let bgzf_contents = fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/bgzipped.fa.bgz"),
        )
        .expect("should read BGZF fixture");
        let cases = [
            (b">x\nA\n".to_vec(), FastaInputKind::Plain),
            (gzip_contents, FastaInputKind::Gzip),
            (bgzf_contents, FastaInputKind::Bgzf),
        ];

        for (source_contents, expected_kind) in cases {
            let reader = ShortReadReader {
                reader: Cursor::new(source_contents.clone()),
                maximum_read: 1,
            };
            let (actual_kind, mut replayed_reader) =
                sniff_fasta_input(reader).expect("should sniff the input stream");
            let mut replayed_contents = Vec::new();
            replayed_reader
                .read_to_end(&mut replayed_contents)
                .expect("should read the sniffed stream");

            assert_eq!(actual_kind, expected_kind);
            assert_eq!(replayed_contents, source_contents);
        }
    }

    #[test]
    fn test_sniff_reads_full_gzip_extra_field_before_classifying_bgzf() {
        let mut source_contents = vec![0x1f, 0x8b, 8, 0x04, 0, 0, 0, 0, 0, 255];
        source_contents.extend_from_slice(&u16::MAX.to_le_bytes());
        let unknown_subfield_length = usize::from(u16::MAX) - 6;
        let unknown_payload_length = unknown_subfield_length - 4;
        source_contents.extend_from_slice(&[
            b'X',
            b'Y',
            unknown_payload_length as u8,
            (unknown_payload_length >> 8) as u8,
        ]);
        source_contents.resize(12 + unknown_subfield_length, 0);
        source_contents.extend_from_slice(&[b'B', b'C', 2, 0, 0, 0]);
        source_contents.extend_from_slice(b"compressed payload");

        let reader = ShortReadReader {
            reader: Cursor::new(source_contents.clone()),
            maximum_read: 7,
        };
        let (kind, mut replayed_reader) =
            sniff_fasta_input(reader).expect("should parse the complete gzip extra field");
        let mut replayed_contents = Vec::new();
        replayed_reader
            .read_to_end(&mut replayed_contents)
            .expect("should read all replayed bytes");

        assert_eq!(kind, FastaInputKind::Bgzf);
        assert_eq!(replayed_contents, source_contents);
    }

    #[test]
    fn test_archived_remote_fasta_inputs_use_one_source_get() {
        let bgzf_contents = fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/bgzipped.fa.bgz"),
        )
        .expect("should read BGZF fixture");
        let mut decoded_bgzf = bgzf::io::Reader::new(Cursor::new(bgzf_contents.clone()));
        let mut fasta_contents = Vec::new();
        decoded_bgzf
            .read_to_end(&mut fasta_contents)
            .expect("should decode BGZF fixture");

        let mut gzip_encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gzip_encoder
            .write_all(&fasta_contents)
            .expect("should write ordinary-gzip FASTA");
        let gzip_contents = gzip_encoder
            .finish()
            .expect("should finish ordinary-gzip FASTA");
        let cases = vec![
            (
                "/remote-plain.fa".to_string(),
                "remote-plain".to_string(),
                fasta_contents.clone(),
            ),
            (
                "/remote-gzip.fa.gz".to_string(),
                "remote-gzip".to_string(),
                gzip_contents,
            ),
            (
                "/remote-bgzf.fa.bgz".to_string(),
                "remote-bgzf".to_string(),
                bgzf_contents,
            ),
        ];
        let files = cases
            .iter()
            .map(|(path, _, contents)| (path.clone(), contents.clone()))
            .collect();
        let server = TestHttpServer::new(files);
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();

        for (path, sample_name, _) in cases {
            server.clear_requests();
            let fasta = server.url(&path);
            let operation_summary = import_fasta(&context, &fasta, "test", &sample_name, &[])
                .expect("should archive remote FASTA input");
            let source_get_count = server
                .requests()
                .iter()
                .filter(|request| {
                    let mut request_parts = request
                        .lines()
                        .next()
                        .unwrap_or_default()
                        .split_whitespace();
                    request_parts.next() == Some("GET")
                        && request_parts.next() == Some(path.as_str())
                })
                .count();
            assert_eq!(
                source_get_count, 1,
                "each source stream should be consumed once for {path}"
            );
            commit_operation_summary(&context, &operation_summary)
                .expect("should commit archived remote FASTA");

            let block_group_id = BlockGroup::get_id("test", &sample_name, "m123", None);
            let sequence =
                Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
                    .into_iter()
                    .find(|sequence| sequence.external_sequence)
                    .expect("should retain the remote sequence as an external asset");
            let sequence_asset = AssetRef::select(conn)
                .get_by_id(
                    sequence
                        .asset_ref_id
                        .expect("should have a sequence AssetRef"),
                )
                .expect("should query sequence AssetRef")
                .expect("should find sequence AssetRef");
            let mut archived_fasta = bgzf::io::Reader::new(
                sequence_asset
                    .reader(context.workspace())
                    .expect("should open retained sequence asset"),
            );
            let mut archived_contents = Vec::new();
            archived_fasta
                .read_to_end(&mut archived_contents)
                .expect("should decode retained sequence asset");

            assert!(sequence_asset.uri.starts_with("file://.gen/assets/"));
            assert_eq!(sequence_asset.materialized_checksum, None);
            assert_eq!(archived_contents, fasta_contents);
        }
    }
}
