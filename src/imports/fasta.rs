use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read},
    path::Path as FsPath,
    str,
    time::{SystemTime, UNIX_EPOCH},
};

use flate2::read::MultiGzDecoder;
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_models::{
    assets::{AssetRef, AssetRole, AssetUri, ChecksummedReader},
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
use noodles::{bgzf, fasta};

use crate::{
    fasta::FastaError,
    progress_bar::{add_saving_operation_bar, get_handler, get_progress_bar},
};
#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(context, fasta, collection_name, sample))
)]
pub fn import_fasta(
    context: &DbContext,
    fasta: &String,
    collection_name: &str,
    sample: &str,
    shallow: bool,
    indexes: &[String],
) -> Result<OperationSummary, FastaError> {
    let conn = context.graph().conn();
    let progress_bar = get_handler();
    let mut operation_files =
        vec![OperationFile::new(fasta.to_string()).set_file_type(FileTypes::Fasta)];
    let mut sequence_asset_ref_id = None;
    let mut checksum_handle = None;
    let input: Box<dyn Read> = if shallow {
        let created_on = i64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("should create sequence asset timestamp")
                .as_nanos(),
        )
        .expect("should fit sequence asset timestamp in i64");
        let sequence_asset_ref =
            operation_files[0].prepare_asset_ref(context.workspace(), created_on)?;
        if let Some(checksum) = sequence_asset_ref.checksum {
            operation_files[0] = operation_files[0].clone().set_checksum_override(checksum);
        }
        AssetRef::create(conn, &sequence_asset_ref)
            .map_err(gen_models::errors::FileAdditionError::DatabaseError)?;
        sequence_asset_ref_id = Some(sequence_asset_ref.id);

        let mut index_locations = indexes.to_vec();
        for index_extension in ["fai", "gzi"] {
            let index_location = format!("{fasta}.{index_extension}");
            if !index_locations.contains(&index_location) && FsPath::new(&index_location).is_file()
            {
                index_locations.push(index_location);
            }
        }
        for index_location in index_locations {
            let mut index_operation_file = OperationFile::new(index_location)
                .set_file_type(FileTypes::None)
                .set_role(AssetRole::SequenceIndex)
                .set_upstream_asset_ref_id(&sequence_asset_ref.id);
            let index_asset_ref =
                index_operation_file.prepare_asset_ref(context.workspace(), created_on)?;
            if let Some(checksum) = index_asset_ref.checksum {
                index_operation_file = index_operation_file.set_checksum_override(checksum);
            }
            AssetRef::create(conn, &index_asset_ref)
                .map_err(gen_models::errors::FileAdditionError::DatabaseError)?;
            operation_files.push(index_operation_file);
        }
        Box::new(sequence_asset_ref.reader(context.workspace())?)
    } else {
        let asset_uri = <dyn AssetUri>::new(context.workspace(), fasta);
        let file = ChecksummedReader::new(asset_uri.reader(context.workspace())?);
        checksum_handle = Some(file.checksum_handle());
        Box::new(file)
    };

    let fasta_extension = <dyn AssetUri>::from_uri(fasta)
        .suffix()
        .and_then(|suffix| suffix.rsplit('.').next().map(str::to_string));
    let reader_stream: Box<dyn BufRead> = match fasta_extension.as_deref() {
        Some("gz") => Box::new(BufReader::new(MultiGzDecoder::new(input))),
        Some("bgz") => Box::new(bgzf::io::Reader::new(input)),
        _ => Box::new(BufReader::new(input)),
    };
    let mut reader = fasta::io::reader::Builder.build_from_reader(reader_stream)?;

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
    for result in reader.records() {
        let record = result.expect("Error during fasta record parsing");
        let sequence = str::from_utf8(record.sequence().as_ref())
            .unwrap()
            .to_string();
        let name = String::from_utf8(record.name().to_vec()).unwrap();
        let sequence_length = record.sequence().len() as i64;
        let seq = if shallow {
            Sequence::new()
                .sequence_type("DNA")
                .name(&name)
                .asset_ref_id(sequence_asset_ref_id.as_ref())
                .length(sequence_length)
                .save(conn)?
        } else {
            Sequence::new()
                .sequence_type("DNA")
                .sequence(&sequence)
                .save(conn)?
        };
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

    if let Some(checksum_handle) = checksum_handle {
        let checksum_override = checksum_handle.checksum().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "FASTA reader did not reach EOF before checksum was requested",
            )
        })?;
        operation_files[0] = operation_files[0]
            .clone()
            .set_checksum_override(checksum_override);
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

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        fs,
        io::{Read as _, Write as _},
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
        operations::commit_operation_summary,
        path::Path,
        sample::Sample,
        sequence::Sequence,
        traits::Query,
    };
    use noodles::bgzf::gzi;

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
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            false,
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
        assert_eq!(asset_refs.len(), 1);
        assert_eq!(asset_refs[0].uri, "file://.gen/outside_root/simple.fa");
        assert!(
            asset_refs[0]
                .logical_path
                .as_deref()
                .is_some_and(|path| path.starts_with(".gen/assets/"))
        );
        assert_eq!(asset_refs[0].name.as_deref(), Some("simple.fa"));

        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
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
            false,
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
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            false,
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
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            false,
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
            false,
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
            .join("shallow.fa.bgz.fai");
        let gzip_index_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("shallow.fa.bgz.gzi");
        fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/bgzipped.fa.bgz"),
            &fasta_path,
        )
        .unwrap();
        fs::write(&index_path, "m123\t37\t6\t37\t38\n").unwrap();
        gzi::fs::write(&gzip_index_path, &gzi::Index::default()).unwrap();

        let operation_summary = import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            true,
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
            "BGZF FASTA should retain both discovered indexes"
        );
        assert_eq!(
            index_assets[0].name.as_deref(),
            Some("shallow.fa.bgz.fai"),
            "first retained index should be the FASTA index"
        );
        assert_eq!(
            index_assets[1].name.as_deref(),
            Some("shallow.fa.bgz.gzi"),
            "second retained index should be the gzip index"
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
    fn test_add_remote_shallow_fasta_with_multiple_remote_indexes() {
        let fasta_contents = fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/bgzipped.fa.bgz"),
        )
        .expect("should read BGZF FASTA fixture");
        let server = TestHttpServer::new(HashMap::from([
            ("/reference.fa.bgz".to_string(), fasta_contents),
            (
                "/reference.fa.bgz.fai".to_string(),
                b"m123\t37\t6\t37\t38\n".to_vec(),
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

        let operation_summary = import_fasta(
            &context,
            &fasta,
            "test",
            Sample::DEFAULT_NAME,
            true,
            &indexes,
        )
        .expect("should import a shallow remote BGZF with remote indexes");
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
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        commit_operation_summary(&context, &operation_summary).unwrap();
        assert_eq!(
            Node::query(conn, "select * from nodes;", rusqlite::params!()).len(),
            3
        );

        let operation_summary = import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        let result_error = commit_operation_summary(&context, &operation_summary).unwrap_err();

        assert!(matches!(result_error, OperationError::NoChanges));
    }
}
