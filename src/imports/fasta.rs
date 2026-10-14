use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufRead, Cursor, Read, Write},
    path::Path as FsPath,
    str,
    time::{SystemTime, UNIX_EPOCH},
};

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_models::{
    assets::{AssetRef, AssetRole, AssetUri, CompressionType, LocalAssetUri},
    block_group::{BlockGroup, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::DbContext,
    edge::Edge,
    errors::{CollectionError, SampleError},
    file_types::FileTypes,
    node::Node,
    operations::{FileAddition, OperationFile, OperationInfo, OperationSummary},
    path::Path,
    sample::Sample,
    sequence::Sequence,
};
use noodles::{
    bgzf::{self, gzi},
    fasta::{self, fai},
};
use tempfile::Builder as TempFileBuilder;

use crate::{
    fasta::FastaError,
    progress_bar::{add_saving_operation_bar, get_handler, get_progress_bar},
};
#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(context, fasta, collection_name, sample, fai, gzi))
)]
pub fn import_fasta(
    context: &DbContext,
    fasta: &String,
    collection_name: &str,
    sample: &str,
    fai: Option<&str>,
    gzi: Option<&str>,
) -> Result<OperationSummary, FastaError> {
    let conn = context.graph().conn();
    let progress_bar = get_handler();

    // We have a created_on field passed through to file creation for indices, etc. We pregenerate
    // the creation date so indices can be shown as being made at the same time.
    let created_on = i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("should create sequence asset timestamp")
            .as_nanos(),
    )
    .expect("should fit sequence asset timestamp in i64");
    let workspace = context.workspace();
    let is_local = LocalAssetUri::is_local_path_or_file_uri(fasta);
    let source_uri = <dyn AssetUri>::new(workspace, fasta);
    let (compression_type, replayed_reader) =
        CompressionType::sniff(source_uri.reader(workspace)?).map_err(std::io::Error::other)?;
    if gzi.is_some() && compression_type != CompressionType::Bgzf {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "a GZI index can only be used with a BGZF-compressed FASTA",
        )
        .into());
    }

    let supplied_fai = if let Some(location) = fai {
        let bytes = read_supplied_index_bytes(workspace, location, "FAI")?;
        let index = parse_fai_index(&bytes).ok_or_else(|| invalid_index_error("FAI", location))?;
        Some((location.to_string(), bytes, index))
    } else {
        None
    };
    let supplied_gzi = if let Some(location) = gzi {
        let bytes = read_supplied_index_bytes(workspace, location, "GZI")?;
        parse_gzi_index(&bytes).ok_or_else(|| invalid_index_error("GZI", location))?;
        Some((location.to_string(), bytes))
    } else {
        None
    };

    // We check is the remote is indexed because if it is not, sequence access would require a full download
    // for any lookup. So if it is not indexed, we download it locally so we can index it properly for sequence
    // access.
    let indexed_remote_parent = !is_local
        && compression_type == CompressionType::Bgzf
        && supplied_fai.is_some()
        && supplied_gzi.is_some();
    let parent_operation_file =
        OperationFile::new(fasta.to_string()).set_file_type(FileTypes::Fasta);
    let sequence_asset = if indexed_remote_parent {
        parent_operation_file.prepare_asset_ref(workspace, created_on)?
    } else {
        let file_addition = FileAddition::prepare_from_reader(
            workspace,
            fasta,
            FileTypes::Fasta,
            replayed_reader,
            compression_type,
            None,
        )?;
        let logical_path = if is_local {
            Some(OperationFile::storage_file_path(
                workspace,
                fasta,
                file_addition.checksum.as_ref(),
            )?)
        } else {
            None
        };
        AssetRef::from_file_addition(
            &file_addition,
            AssetRole::Input,
            logical_path.as_deref(),
            Some(&parent_operation_file.filename),
            None,
            created_on,
        )
    };

    let sequence_path = if indexed_remote_parent {
        None
    } else {
        Some(sequence_asset.versioned_store_path(workspace)?)
    };

    // If there is no index present for the fasta, generate it
    let (fai_location, fai_bytes, fasta_index) = match supplied_fai {
        Some((location, bytes, index)) => (location, bytes, index),
        None => {
            let sequence_path = sequence_path.as_ref().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "remote FASTA index was not available",
                )
            })?;
            let (index, bytes) = build_fai_index(sequence_path)?;
            let index_location = sibling_index_location(fasta, "fai");
            (index_location, bytes, index)
        }
    };

    // if there is no index present for the bgz compressed fasta, generate it
    let (gzi_location, gzi_bytes) = match supplied_gzi {
        Some((location, bytes)) => (location, bytes),
        None => {
            let sequence_path = sequence_path.as_ref().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "remote BGZF index was not available",
                )
            })?;
            let bytes = build_gzi_index(sequence_path)?;
            (sibling_index_location(fasta, "gzi"), bytes)
        }
    };

    let mut prepared_asset_refs = vec![sequence_asset.clone()];
    AssetRef::create(conn, &sequence_asset)
        .map_err(gen_models::errors::FileAdditionError::DatabaseError)?;
    let mut operation_files = vec![parent_operation_file];
    for (index_location, bytes, file_type) in [
        (fai_location, fai_bytes, FileTypes::FastaIndex),
        (gzi_location, gzi_bytes, FileTypes::BgzfIndex),
    ] {
        let operation_file = OperationFile::new(index_location.clone())
            .set_file_type(file_type)
            .set_role(AssetRole::SequenceIndex)
            .set_upstream_asset_ref_id(&sequence_asset.id);

        // These are the asset refs for the indices we generated or passed in
        let index_asset_ref = if indexed_remote_parent
            && !LocalAssetUri::is_local_path_or_file_uri(&index_location)
        {
            operation_file.prepare_asset_ref(workspace, created_on)?
        } else {
            prepare_stored_index_asset_ref(
                workspace,
                &bytes,
                file_type,
                &index_location,
                &sequence_asset,
                created_on,
            )?
        };
        AssetRef::create(conn, &index_asset_ref)
            .map_err(gen_models::errors::FileAdditionError::DatabaseError)?;
        prepared_asset_refs.push(index_asset_ref.clone());
        operation_files.push(operation_file);
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
    for record in fasta_index.as_ref() {
        let name = str::from_utf8(record.name())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?
            .to_string();
        let sequence_length = i64::try_from(record.length())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        let seq = Sequence::new()
            .sequence_type("DNA")
            .name(&name)
            .asset_ref_id(Some(&sequence_asset.id))
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
    )
    .with_prepared_asset_refs(prepared_asset_refs);
    bar.finish();
    Ok(operation_summary)
}

fn sibling_index_location(fasta: &str, extension: &str) -> String {
    let source_path = fasta.split(['?', '#']).next().unwrap_or(fasta);
    format!("{source_path}.{extension}")
}

fn read_index_bytes(
    workspace: &gen_core::Workspace,
    path_or_uri: &str,
) -> Result<Vec<u8>, FastaError> {
    if LocalAssetUri::is_local_path_or_file_uri(path_or_uri) {
        let path = LocalAssetUri::resolve_input_source_path(workspace, path_or_uri)?;
        return Ok(fs::read(path)?);
    }

    let mut reader = <dyn AssetUri>::new(workspace, path_or_uri).reader(workspace)?;
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn read_supplied_index_bytes(
    workspace: &gen_core::Workspace,
    path_or_uri: &str,
    extension: &str,
) -> Result<Vec<u8>, FastaError> {
    read_index_bytes(workspace, path_or_uri).map_err(|error| {
        std::io::Error::other(format!(
            "could not read supplied {extension} index '{path_or_uri}': {error}"
        ))
        .into()
    })
}

fn invalid_index_error(extension: &str, path_or_uri: &str) -> FastaError {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("supplied {extension} index is invalid: {path_or_uri}"),
    )
    .into()
}

fn parse_fai_index(bytes: &[u8]) -> Option<fai::Index> {
    let index = fai::io::Reader::new(Cursor::new(bytes)).read_index().ok()?;
    let mut names = HashSet::new();
    let records = index.as_ref();
    let is_usable = !records.is_empty()
        && records.iter().all(|record| {
            record.length() > 0
                && record.line_bases() > 0
                && record.line_width() >= record.line_bases()
                && str::from_utf8(record.name()).is_ok_and(|name| names.insert(name.to_string()))
        });
    is_usable.then_some(index)
}

fn parse_gzi_index(bytes: &[u8]) -> Option<gzi::Index> {
    let index = gzi::io::Reader::new(Cursor::new(bytes)).read_index().ok()?;
    let mut previous = (0, 0);
    for &(compressed_offset, uncompressed_offset) in index.as_ref() {
        if compressed_offset <= previous.0 || uncompressed_offset <= previous.1 {
            return None;
        }
        previous = (compressed_offset, uncompressed_offset);
    }
    Some(index)
}

fn build_fai_index(path: &FsPath) -> Result<(fai::Index, Vec<u8>), FastaError> {
    let file = File::open(path)?;
    let mut indexer = fasta::io::Indexer::new(bgzf::io::Reader::new(file));
    let mut records = Vec::new();
    while let Some(record) = indexer.index_record().map_err(std::io::Error::from)? {
        records.push(record);
    }
    if records.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "FASTA contains no indexed records",
        )
        .into());
    }
    let index = fai::Index::from(records);
    let mut bytes = Vec::new();
    fai::io::Writer::new(&mut bytes).write_index(&index)?;
    Ok((index, bytes))
}

fn build_gzi_index(path: &FsPath) -> Result<Vec<u8>, FastaError> {
    let file = File::open(path)?;
    let mut reader = bgzf::io::Reader::new(file);
    let mut records = Vec::new();
    let mut uncompressed_offset = 0_u64;
    loop {
        let block_length = {
            let block = reader.fill_buf()?;
            if block.is_empty() {
                break;
            }
            block.len()
        };
        let compressed_offset = reader.virtual_position().compressed();
        if uncompressed_offset > 0 {
            records.push((compressed_offset, uncompressed_offset));
        }
        reader.consume(block_length);
        uncompressed_offset = uncompressed_offset
            .checked_add(u64::try_from(block_length).map_err(std::io::Error::other)?)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "BGZF uncompressed offset overflows",
                )
            })?;
    }

    let index = gzi::Index::from(records);
    let mut bytes = Vec::new();
    gzi::io::Writer::new(&mut bytes).write_index(&index)?;
    Ok(bytes)
}

fn prepare_stored_index_asset_ref(
    workspace: &gen_core::Workspace,
    bytes: &[u8],
    file_type: FileTypes,
    source_path_or_uri: &str,
    sequence_asset: &AssetRef,
    created_on: i64,
) -> Result<AssetRef, FastaError> {
    let asset_directory = workspace
        .asset_dir()
        .map_err(gen_models::errors::FileAdditionError::ConfigError)?;
    fs::create_dir_all(&asset_directory)?;
    let extension = file_type.as_str();
    let suffix = format!(".{extension}");
    // A dotless prefix keeps AssetUri suffix extraction from including the random tempfile basename.
    let mut temporary_file = TempFileBuilder::new()
        .prefix("index-")
        .suffix(&suffix)
        .tempfile_in(&asset_directory)?;
    temporary_file.write_all(bytes)?;
    temporary_file.flush()?;
    let file_addition = FileAddition::prepare(
        workspace,
        &temporary_file.path().to_string_lossy(),
        file_type,
        None,
    )?;
    Ok(AssetRef::from_file_addition(
        &file_addition,
        AssetRole::SequenceIndex,
        None,
        Some(&OperationFile::new(source_path_or_uri.to_string()).filename),
        Some(&sequence_asset.id),
        created_on,
    ))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        fs,
        io::{BufRead as _, BufReader, Read as _, Write as _},
        net::{TcpListener, TcpStream},
        panic::catch_unwind,
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
        operations::{calculate_file_checksum, commit_operation_summary},
        path::Path,
        sample::Sample,
        sequence::Sequence,
    };
    use noodles::{
        bgzf::{self, gzi},
        core::Region,
        fasta::{self, fai},
    };

    use super::import_fasta;
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
                    // macOS inherits the listener's nonblocking mode on accepted sockets.
                    // Only accepting connections should poll; request I/O must wait for bytes.
                    stream
                        .set_nonblocking(false)
                        .expect("should configure blocking remote FASTA request I/O");
                    // TCP can split headers across reads, and sniffing clients can disconnect
                    // before consuming a response. Neither should terminate the fixture server.
                    let mut reader = BufReader::new(&mut stream);
                    let mut request = String::new();
                    loop {
                        let mut line = String::new();
                        match reader.read_line(&mut line) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {
                                request.push_str(&line);
                                if line == "\r\n" {
                                    break;
                                }
                            }
                        }
                    }
                    if !request.ends_with("\r\n\r\n") {
                        continue;
                    }
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
                        let _ = stream.write_all(
                                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                            );
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
                    if write!(
                        stream,
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{content_range}Accept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                        body.len()
                    ).is_err() {
                        continue;
                    }
                    if method != "HEAD" {
                        let _ = stream.write_all(body);
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

        fn stop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                handle.join().expect("should stop remote FASTA test server");
            }
        }
    }

    impl Drop for TestHttpServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                // Preserve the original test failure if the server also panicked.
                let _ = handle.join();
            }
        }
    }

    #[test]
    fn test_remote_fasta_server_reads_fragmented_headers() {
        let mut server = TestHttpServer::new(HashMap::from([(
            "/reference.fa".to_string(),
            b">sequence\nACGT\n".to_vec(),
        )]));
        let mut stream =
            TcpStream::connect(&server.address).expect("should connect to test server");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("should limit response wait");
        stream
            .write_all(b"GET /reference.fa HTTP/1.1\r\nHost: ")
            .expect("should send partial headers");
        thread::sleep(Duration::from_millis(20));
        stream
            .write_all(b"localhost\r\n\r\n")
            .expect("should finish request headers");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("should read complete response");
        assert!(response.ends_with(">sequence\nACGT\n"));
        assert!(server.requests()[0].ends_with("Host: localhost\r\n\r\n"));
        server.stop();
    }

    #[test]
    fn test_remote_fasta_server_drop_preserves_original_panic() {
        let failure = catch_unwind(|| {
            let _server = TestHttpServer {
                address: String::new(),
                requests: Arc::new(Mutex::new(Vec::new())),
                stop: Arc::new(AtomicBool::new(false)),
                handle: Some(thread::spawn(|| panic!("server failure"))),
            };
            panic!("original test failure");
        })
        .expect_err("should preserve the test panic");
        assert_eq!(
            failure.downcast_ref::<&str>(),
            Some(&"original test failure")
        );
    }

    #[test]
    fn test_add_fasta() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let history_store = DoltHistoryStore::new(conn);

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let source_checksum = calculate_file_checksum(&fasta_path).unwrap();

        let operation_summary = import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            None,
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
        assert_eq!(
            asset_refs.len(),
            3,
            "FASTA and both indexes should be retained"
        );
        let sequence_asset = asset_refs
            .iter()
            .find(|asset_ref| asset_ref.role == AssetRole::Input)
            .expect("should retain the sequence asset");
        assert!(sequence_asset.uri.ends_with(".fa.bgz"));
        let archived_path = sequence_asset
            .versioned_store_path(context.workspace())
            .expect("should resolve retained FASTA archive");
        assert_eq!(
            sequence_asset.checksum,
            Some(calculate_file_checksum(archived_path).unwrap())
        );
        assert_eq!(sequence_asset.materialized_checksum, Some(source_checksum));
        assert!(
            sequence_asset
                .logical_path
                .as_deref()
                .is_some_and(|path| path == ".gen/outside_root/simple.fa")
        );
        assert_eq!(sequence_asset.name.as_deref(), Some("simple.fa"));

        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
        let sequence = Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
            .into_iter()
            .find(|sequence| sequence.asset_ref_id == Some(sequence_asset.id))
            .expect("should store the sequence as an external asset");
        assert!(sequence.external_sequence);
        assert_eq!(sequence.length, 34);
        let stored_sequence = conn
            .query_row(
                "SELECT sequence FROM sequences WHERE hash = ?1",
                [sequence.hash],
                |row| row.get::<_, String>(0),
            )
            .expect("should read the sequence row payload");
        assert!(stored_sequence.is_empty(), "FASTA bases stay out of SQLite");
        assert_eq!(
            BlockGroup::get_all_sequences(conn, context.workspace(), &block_group_id, false)
                .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()]),
            "shallow FASTA should read through retained sequence and index assets"
        );

        let path = Path::all(conn).expect("should load paths")[0].clone();
        assert_eq!(
            path.sequence(conn, context.workspace(), None).unwrap(),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()
        );
    }

    #[test]
    fn test_explicit_fai_is_reused_for_plain_fasta() {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let fai_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("arbitrary-fasta-offsets");
        fs::write(&fai_path, "m123\t34\t6\t34\t35\n").expect("should write the explicit FAI index");

        let operation_summary = import_fasta(
            &context,
            &fasta_path.to_string_lossy().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            fai_path.to_str(),
            None,
        )
        .expect("should reuse an FAI for the same uncompressed FASTA payload");
        commit_operation_summary(&context, &operation_summary)
            .expect("should commit the FASTA and retained indexes");

        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
        let sequence = Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
            .into_iter()
            .find(|sequence| sequence.asset_ref_id.is_some())
            .expect("should retain the imported sequence asset");
        assert_eq!(
            sequence
                .get_sequence(2, 8)
                .expect("should read through the supplied FAI"),
            "CGATCG"
        );
    }

    #[test]
    fn test_explicit_missing_and_invalid_indexes_fail() {
        let context = setup_gen_on_disk();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let index_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("custom-fasta-index");
        let fasta = fasta_path.to_string_lossy().to_string();
        let index = index_path.to_string_lossy().to_string();

        let missing_error = import_fasta(
            &context,
            &fasta,
            "test",
            Sample::DEFAULT_NAME,
            Some(&index),
            None,
        )
        .expect_err("should reject a missing explicitly supplied FAI");
        assert!(
            missing_error
                .to_string()
                .contains("could not read supplied FAI index"),
            "missing explicit FAI should report its type and path: {missing_error}"
        );

        fs::write(&index_path, "invalid index\n").expect("should write an invalid FAI");
        let invalid_error = import_fasta(
            &context,
            &fasta,
            "test",
            Sample::DEFAULT_NAME,
            Some(&index),
            None,
        )
        .expect_err("should reject an invalid explicitly supplied FAI");
        assert!(
            invalid_error
                .to_string()
                .contains("supplied FAI index is invalid"),
            "invalid explicit FAI should report its type and path: {invalid_error}"
        );

        let incompatible_error = import_fasta(
            &context,
            &fasta,
            "test",
            Sample::DEFAULT_NAME,
            None,
            Some(&index),
        )
        .expect_err("should reject a GZI for a plain FASTA");
        assert!(
            incompatible_error
                .to_string()
                .contains("GZI index can only be used with a BGZF-compressed FASTA"),
            "incompatible GZI should report why it cannot be reused: {incompatible_error}"
        );

        let bgzf_fasta =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/bgzipped.fa.bgz");
        fs::write(&index_path, "invalid GZI\n").expect("should write an invalid GZI");
        let invalid_gzi_error = import_fasta(
            &context,
            &bgzf_fasta.to_string_lossy().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Some(&index),
        )
        .expect_err("should reject an invalid explicitly supplied GZI");
        assert!(
            invalid_gzi_error
                .to_string()
                .contains("supplied GZI index is invalid"),
            "invalid explicit GZI should report its type and path: {invalid_gzi_error}"
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
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            None,
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
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            None,
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
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            None,
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
    fn test_ignores_neighbor_indexes_without_explicit_options() {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let fasta_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("source.fa.bgz");
        let fai_path = PathBuf::from(format!("{}.fai", fasta_path.display()));
        let gzi_path = PathBuf::from(format!("{}.gzi", fasta_path.display()));
        fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/bgzipped.fa.bgz"),
            &fasta_path,
        )
        .expect("should copy the BGZF FASTA fixture");
        fs::write(&fai_path, "other-record\t1\t6\t1\t2\n")
            .expect("should write a stale neighboring FAI");
        gzi::fs::write(&gzi_path, &gzi::Index::from(vec![(1, 1)]))
            .expect("should write a stale neighboring GZI");

        let summary = import_fasta(
            &context,
            &fasta_path.to_string_lossy().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            None,
        )
        .expect("should generate indexes when no index paths are supplied");
        commit_operation_summary(&context, &summary).expect("should commit generated indexes");

        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
        let sequence = Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
            .into_iter()
            .find(|sequence| sequence.asset_ref_id.is_some())
            .expect("should import the source FASTA record, not the neighboring FAI record");
        assert_eq!(
            sequence
                .get_sequence(0, 34)
                .expect("should read through generated indexes"),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
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
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            "new-sample",
            None,
            None,
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
    fn test_add_fasta_shallow() {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();

        let fasta_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("shallow.fa.bgz");
        let index_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("fasta-offsets");
        let gzip_index_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("bgzf-offsets");
        fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/bgzipped.fa.bgz"),
            &fasta_path,
        )
        .unwrap();
        fs::write(&index_path, "m123\t34\t6\t34\t35\n").unwrap();
        gzi::fs::write(&gzip_index_path, &gzi::Index::default()).unwrap();

        let operation_summary = import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            index_path.to_str(),
            gzip_index_path.to_str(),
        )
        .unwrap();
        commit_operation_summary(&context, &operation_summary).unwrap();
        fs::remove_file(&fasta_path).unwrap();
        fs::write(&index_path, "invalid logical-path index\n").unwrap();
        fs::write(&gzip_index_path, "invalid logical-path gzip index\n").unwrap();
        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, context.workspace(), &block_group_id, false)
                .expect("should load shallow sequence from retained sequence and index assets"),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()]),
            "shallow FASTA should read through retained sequence and index assets"
        );
        let sequence = Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
            .into_iter()
            .find(|sequence| sequence.asset_ref_id.is_some())
            .expect("should persist the shallow sequence AssetRef pointer");
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
            "shallow sequence should point to its input AssetRef"
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
            "BGZF FASTA should retain both supplied indexes"
        );
        assert_eq!(
            index_assets[0].name.as_deref(),
            Some("bgzf-offsets"),
            "GZI should keep the supplied filename"
        );
        assert_eq!(
            index_assets[1].name.as_deref(),
            Some("fasta-offsets"),
            "FAI should keep the supplied filename"
        );
        assert!(
            index_assets
                .iter()
                .any(|asset_ref| asset_ref.file_type == "fai"),
            "FAI type should identify an arbitrary-named retained index"
        );
        assert!(
            index_assets
                .iter()
                .any(|asset_ref| asset_ref.file_type == "gzi"),
            "GZI type should identify an arbitrary-named retained index"
        );
        let operation_assets = OperationAsset::all(conn).expect("should load operation assets");
        assert!(
            operation_assets.iter().any(|operation_asset| {
                Some(operation_asset.asset_ref_id) == sequence.asset_ref_id
            }),
            "operation should track the shallow sequence asset"
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
    fn test_add_plain_and_gzip_fasta_shallow_commits_retained_assets() {
        let fixture_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let test_inputs = [
            ("simple.fa", "plain-shallow.fa", true),
            ("fastas/gzipped.fa.gz", "gzip-shallow.fa.gz", false),
        ];

        for (fixture, source_name, has_materialized_checksum) in test_inputs {
            let context = setup_gen_on_disk();
            let conn = context.graph().conn();
            let fasta_path = context.workspace().repo_root().unwrap().join(source_name);
            fs::copy(fixture_directory.join(fixture), &fasta_path)
                .expect("should copy the FASTA fixture into the repository");
            let fasta_path_string = fasta_path.to_string_lossy().to_string();

            let operation_summary = import_fasta(
                &context,
                &fasta_path_string,
                "test",
                Sample::DEFAULT_NAME,
                None,
                None,
            )
            .expect("should import a FASTA through the retained BGZF asset");
            commit_operation_summary(&context, &operation_summary)
                .expect("should commit the retained FASTA asset");
            fs::remove_file(&fasta_path).expect("should remove the original FASTA");

            let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
            assert_eq!(
                BlockGroup::get_all_sequences(conn, context.workspace(), &block_group_id, false)
                    .expect("should load the shallow sequence from the retained asset"),
                HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()]),
                "shallow import should remain readable after the source is removed"
            );
            let sequence =
                Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
                    .into_iter()
                    .find(|sequence| sequence.asset_ref_id.is_some())
                    .expect("should store the external sequence asset pointer");
            let asset_ref = AssetRef::select(conn)
                .get_by_id(
                    sequence
                        .asset_ref_id
                        .expect("should have the external sequence asset pointer"),
                )
                .expect("should query the external sequence asset")
                .expect("should retain the external sequence asset");
            assert!(
                asset_ref.uri.ends_with(".fa.bgz"),
                "FASTA should be read from its retained BGZF URI"
            );
            assert_eq!(
                asset_ref.materialized_checksum.is_some(),
                has_materialized_checksum,
                "only the original plain input should record its materialized checksum"
            );
        }
    }

    #[test]
    fn test_add_remote_shallow_fasta_with_multiple_remote_indexes() {
        // Put the indexed sequence in a later BGZF block so lookup requires a nonzero seek.
        // Readers can stream a block at byte zero without sending a Range header.
        let padding = b">padding\nA\n";
        let mut writer = bgzf::io::Writer::new(Vec::new());
        writer
            .write_all(padding)
            .expect("should write padding block");
        writer.flush().expect("should flush padding block");
        let mut fasta_contents = writer.into_inner();
        let compressed_offset = fasta_contents.len() as u64;
        fasta_contents.extend(
            fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/bgzipped.fa.bgz"),
            )
            .expect("should read BGZF FASTA fixture"),
        );
        let mut gzip_index = Vec::new();
        gzi::io::Writer::new(&mut gzip_index)
            .write_index(&gzi::Index::from(vec![(
                compressed_offset,
                padding.len() as u64,
            )]))
            .expect("should write gzip index for the second block");
        let server = TestHttpServer::new(HashMap::from([
            ("/reference.fa.gz".to_string(), fasta_contents),
            (
                "/indexes/fasta-index".to_string(),
                format!("m123\t34\t{}\t34\t35\n", padding.len() + 6).into_bytes(),
            ),
            ("/indexes/bgzf-index".to_string(), gzip_index),
        ]));
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let fasta = server.url("/reference.fa.gz");
        let fai = server.url("/indexes/fasta-index?token=fai");
        let gzi = server.url("/indexes/bgzf-index?token=gzi");

        let operation_summary = import_fasta(
            &context,
            &fasta,
            "test",
            Sample::DEFAULT_NAME,
            Some(&fai),
            Some(&gzi),
        )
        .expect("should import a remote BGZF with explicitly supplied indexes");
        commit_operation_summary(&context, &operation_summary)
            .expect("should commit remote shallow FASTA assets");

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
        assert_eq!(
            index_assets
                .iter()
                .map(|asset_ref| asset_ref.uri.clone())
                .collect::<HashSet<_>>(),
            HashSet::from([fai.clone(), gzi.clone()]),
            "explicit remote index URIs, including query strings, should be retained"
        );
        assert!(
            index_assets.iter().all(|asset_ref| {
                asset_ref.role == AssetRole::SequenceIndex
                    && asset_ref.checksum.is_none()
                    && matches!(asset_ref.file_type.as_str(), "fai" | "gzi")
            }),
            "remote indexes should retain their explicit FAI and GZI types"
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
            "remote shallow import should not retain local sequence or index assets"
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
                (request.starts_with("GET /indexes/fasta-index ")
                    || request.starts_with("HEAD /indexes/fasta-index "))
                    && request.contains("authorization: Bearer fai")
            }),
            "indexed lookup should request the remote FASTA index; requests: {requests:?}"
        );
        assert!(
            requests.iter().any(|request| {
                (request.starts_with("GET /indexes/bgzf-index ")
                    || request.starts_with("HEAD /indexes/bgzf-index "))
                    && request.contains("authorization: Bearer gzi")
            }),
            "indexed lookup should request the remote gzip index"
        );
        assert!(
            requests.iter().any(|request| {
                request.starts_with("GET /reference.fa.gz ")
                    && request.to_ascii_lowercase().contains("\r\nrange: bytes=")
            }),
            "indexed remote lookup should use a byte-range request; requests: {requests:?}"
        );
    }

    #[test]
    fn test_remote_fasta_without_both_usable_indexes_is_archived_locally() {
        let fixture_directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let test_cases = [
            (
                "/plain.fa",
                fs::read(fixture_directory.join("simple.fa"))
                    .expect("should read plain FASTA fixture"),
                None,
                true,
            ),
            (
                "/ordinary.fa.gz",
                fs::read(fixture_directory.join("fastas/gzipped.fa.gz"))
                    .expect("should read gzip FASTA fixture"),
                None,
                false,
            ),
            (
                "/bgzf.fa.gz",
                fs::read(fixture_directory.join("fastas/bgzipped.fa.bgz"))
                    .expect("should read BGZF FASTA fixture"),
                None,
                false,
            ),
            (
                "/partial.fa.bgz",
                fs::read(fixture_directory.join("fastas/bgzipped.fa.bgz"))
                    .expect("should read BGZF FASTA fixture"),
                Some(("/indexes/custom-fai", b"m123\t34\t6\t34\t35\n".to_vec())),
                false,
            ),
        ];

        for (fasta_path, fasta_contents, partial_fai, has_materialized_checksum) in test_cases {
            let mut files = HashMap::from([(fasta_path.to_string(), fasta_contents)]);
            if let Some((path, bytes)) = &partial_fai {
                files.insert(path.to_string(), bytes.clone());
            }
            let mut server = TestHttpServer::new(files);
            let context = setup_gen_on_disk();
            let conn = context.graph().conn();
            let fasta = server.url(fasta_path);
            let fai = partial_fai.as_ref().map(|(path, _)| server.url(path));

            let summary = import_fasta(
                &context,
                &fasta,
                "test",
                Sample::DEFAULT_NAME,
                fai.as_deref(),
                None,
            )
            .expect("should download and index a remote FASTA locally");
            commit_operation_summary(&context, &summary)
                .expect("should commit the local FASTA archive and indexes");

            let source_gets = server
                .requests()
                .iter()
                .filter(|request| request.starts_with(&format!("GET {fasta_path} ")))
                .count();
            assert_eq!(source_gets, 1, "source bytes should be fetched once");
            server.stop();

            let sequence_asset = AssetRef::all(conn)
                .expect("should load asset references")
                .into_iter()
                .find(|asset_ref| asset_ref.role == AssetRole::Input)
                .expect("should retain the sequence asset");
            assert!(
                sequence_asset.uri.starts_with("file://.gen/assets/"),
                "fallback sequence should use the retained local archive"
            );
            assert!(sequence_asset.uri.ends_with(".fa.bgz"));
            assert_eq!(
                sequence_asset.materialized_checksum.is_some(),
                has_materialized_checksum
            );
            let index_assets = AssetRef::get_derived_assets(conn, &sequence_asset.id, None);
            assert_eq!(index_assets.len(), 2, "fallback should retain FAI and GZI");
            assert!(index_assets.iter().all(|asset_ref| {
                asset_ref.role == AssetRole::SequenceIndex
                    && asset_ref.upstream_asset_ref_id == Some(sequence_asset.id)
                    && asset_ref.uri.starts_with("file://.gen/assets/")
            }));
            assert!(
                index_assets
                    .iter()
                    .any(|asset_ref| asset_ref.file_type == "fai")
                    && index_assets
                        .iter()
                        .any(|asset_ref| asset_ref.file_type == "gzi"),
                "fallback indexes should retain explicit type tags"
            );

            let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
            let sequence =
                Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
                    .into_iter()
                    .find(|sequence| sequence.asset_ref_id == Some(sequence_asset.id))
                    .expect("should retain an external sequence row");
            assert_eq!(
                sequence
                    .get_sequence(0, 34)
                    .expect("should read local archive after server stop"),
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
            );
        }
    }

    #[test]
    fn test_generated_gzi_supports_indexed_reads_beyond_first_bgzf_block() {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let fasta_path = context
            .workspace()
            .repo_root()
            .expect("should resolve the repository root")
            .join("multi-block.fa");
        let expected_sequence = (0..200_000)
            .map(|index| b"ACGT"[index % 4])
            .collect::<Vec<_>>();
        let mut fasta_contents = b">large\n".to_vec();
        fasta_contents.extend_from_slice(&expected_sequence);
        fasta_contents.push(b'\n');
        fs::write(&fasta_path, fasta_contents).expect("should write the multi-block FASTA");

        let summary = import_fasta(
            &context,
            &fasta_path.to_string_lossy().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            None,
        )
        .expect("should import and index the multi-block FASTA");
        commit_operation_summary(&context, &summary).expect("should commit the FASTA assets");

        let sequence_asset = AssetRef::all(conn)
            .expect("should load asset references")
            .into_iter()
            .find(|asset_ref| asset_ref.role == AssetRole::Input)
            .expect("should retain the sequence asset");
        let index_assets = AssetRef::get_derived_assets(conn, &sequence_asset.id, None);
        let fai_asset = index_assets
            .iter()
            .find(|asset_ref| asset_ref.uri.ends_with(".fai"))
            .expect("should retain the FAI index");
        let gzi_asset = index_assets
            .iter()
            .find(|asset_ref| asset_ref.uri.ends_with(".gzi"))
            .expect("should retain the GZI index");
        let fai_index = fai::io::Reader::new(
            fai_asset
                .reader(context.workspace())
                .expect("should open the retained FAI index"),
        )
        .read_index()
        .expect("should parse the retained FAI index");
        let gzi_index = gzi::io::Reader::new(
            gzi_asset
                .reader(context.workspace())
                .expect("should open the retained GZI index"),
        )
        .read_index()
        .expect("should parse the retained GZI index");
        assert!(
            !gzi_index.as_ref().is_empty(),
            "multi-block BGZF should need nonzero GZI offsets"
        );
        let bgzf_reader = bgzf::io::indexed_reader::Builder::default()
            .set_index(gzi_index)
            .build_from_reader(
                sequence_asset
                    .reader(context.workspace())
                    .expect("should open the retained BGZF sequence"),
            )
            .expect("should build a GZI-indexed BGZF reader");
        let mut reader = fasta::io::indexed_reader::Builder::default()
            .set_index(fai_index)
            .build_from_reader(bgzf_reader)
            .expect("should build a FAI-indexed FASTA reader");
        let region = "large:130001-130200"
            .parse::<Region>()
            .expect("should parse the FASTA region");
        let record = reader
            .query(&region)
            .expect("should seek to a FASTA region beyond the first BGZF block");
        assert_eq!(
            record.sequence().as_ref(),
            &expected_sequence[130_000..130_200]
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
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            None,
            None,
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
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            None,
            None,
        )
        .unwrap();
        let result_error = commit_operation_summary(&context, &operation_summary).unwrap_err();

        assert!(matches!(result_error, OperationError::NoChanges));
    }
}
