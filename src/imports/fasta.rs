use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read},
    str,
    time::{SystemTime, UNIX_EPOCH},
};

use flate2::read::MultiGzDecoder;
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_models::{
    assets::{AssetRef, AssetUri, ChecksummedReader},
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
) -> Result<OperationSummary, FastaError> {
    let conn = context.graph().conn();
    let progress_bar = get_handler();
    let mut operation_file = OperationFile::new(fasta.to_string()).set_file_type(FileTypes::Fasta);
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
            operation_file.prepare_asset_ref(context.workspace(), created_on)?;
        if let Some(checksum) = sequence_asset_ref.checksum {
            operation_file = operation_file.set_checksum_override(checksum);
        }
        AssetRef::create(conn, &sequence_asset_ref)
            .map_err(gen_models::errors::FileAdditionError::DatabaseError)?;
        sequence_asset_ref_id = Some(sequence_asset_ref.id);
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
        operation_file = operation_file.set_checksum_override(checksum_override);
    }

    let bar = add_saving_operation_bar(&progress_bar);
    let operation_summary = OperationSummary::new(
        OperationInfo {
            files: vec![operation_file],
            description: "fasta_addition".to_string(),
        },
        summary_str,
    );
    bar.finish();
    Ok(operation_summary)
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use std::{
        collections::HashSet,
        fs,
        io::Write as _,
        net::TcpListener,
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread,
        time::Duration,
    };

    use gen_models::{
        assets::{AssetRef, OperationKind, OperationLog},
        errors::OperationError,
        history::{HistoryStore, dolt::DoltHistoryStore},
        operations::commit_operation_summary,
        traits::*,
    };

    use super::*;
    use crate::test_helpers::{setup_gen, setup_gen_on_disk};

    struct TestHttpServer {
        address: String,
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl TestHttpServer {
        fn new(contents: Vec<u8>) -> Self {
            let listener =
                TcpListener::bind(("127.0.0.1", 0)).expect("should bind remote FASTA server");
            listener
                .set_nonblocking(true)
                .expect("should configure remote FASTA server");
            let address = listener
                .local_addr()
                .expect("should read remote FASTA server address")
                .to_string();
            let stop = Arc::new(AtomicBool::new(false));
            let server_stop = Arc::clone(&stop);
            let handle = thread::spawn(move || {
                while !server_stop.load(Ordering::Relaxed) {
                    let Ok((mut stream, _)) = listener.accept() else {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    };
                    let mut request_bytes = [0_u8; 4096];
                    let length = stream
                        .read(&mut request_bytes)
                        .expect("should read remote FASTA request");
                    let request = String::from_utf8_lossy(&request_bytes[..length]);
                    let method = request
                        .lines()
                        .next()
                        .and_then(|line| line.split_whitespace().next())
                        .unwrap_or_default();
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        contents.len()
                    )
                    .expect("should write remote FASTA response headers");
                    if method != "HEAD" {
                        stream
                            .write_all(&contents)
                            .expect("should write remote FASTA response body");
                    }
                }
            });
            Self {
                address,
                stop,
                handle: Some(handle),
            }
        }

        fn url(&self) -> String {
            format!("http://{}/reference.fa", self.address)
        }
    }

    impl Drop for TestHttpServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                handle.join().expect("should stop remote FASTA server");
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
        )
        .unwrap();
        let commit_hash = commit_operation_summary(&context, &operation_summary).unwrap();
        assert_eq!(history_store.current_head().unwrap(), Some(commit_hash));
        let mut operation_logs = OperationLog::all(conn);
        operation_logs.sort_by_key(|operation_log| std::cmp::Reverse(operation_log.created_on));
        assert_eq!(
            operation_logs[0].operation_kind,
            OperationKind::Other("fasta_addition".to_string())
        );
        let asset_refs = AssetRef::all(conn);
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
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );

        let path = Path::all(conn)[0].clone();
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
        )
        .unwrap();
        let block_group_id = BlockGroup::get_id("test", "new-sample", "m123", None);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, context.workspace(), &block_group_id, false)
                .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );

        let path = Path::all(conn)[0].clone();
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
    fn test_add_fasta_shallow_uses_immutable_asset() {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let fasta_path = context
            .workspace()
            .repo_root()
            .expect("should have repository root")
            .join("shallow.fa");
        fs::copy(&fixture_path, &fasta_path).expect("should copy shallow FASTA fixture");

        let operation_summary = import_fasta(
            &context,
            &fasta_path.to_string_lossy().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            true,
        )
        .expect("should import shallow FASTA");
        commit_operation_summary(&context, &operation_summary)
            .expect("should commit shallow FASTA import");
        fs::remove_file(&fasta_path).expect("should remove logical FASTA path");

        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, context.workspace(), &block_group_id, false)
                .expect("should load shallow sequence from immutable asset"),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        let sequence = Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
            .into_iter()
            .find(|sequence| sequence.asset_ref_id.is_some())
            .expect("should persist the shallow sequence AssetRef pointer");
        let asset_ref = AssetRef::get_by_id(
            conn,
            &sequence
                .asset_ref_id
                .expect("should have sequence AssetRef"),
            None,
        )
        .expect("should persist sequence AssetRef");
        assert!(
            asset_ref
                .versioned_store_path(context.workspace())
                .expect("should resolve immutable sequence asset")
                .is_file(),
            "immutable sequence asset should remain after logical file removal"
        );

        let path = Path::all(conn)[0].clone();
        assert_eq!(
            path.sequence(conn, context.workspace(), None)
                .expect("should read path through immutable sequence asset"),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    #[test]
    fn test_add_remote_fasta_shallow_uses_remote_asset() {
        let server = TestHttpServer::new(b">m123\nATCGATCGATCGATCGATCGGGAACACACAGAGA\n".to_vec());
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let fasta = server.url();

        let operation_summary = import_fasta(&context, &fasta, "test", Sample::DEFAULT_NAME, true)
            .expect("should import shallow remote FASTA");
        commit_operation_summary(&context, &operation_summary)
            .expect("should commit shallow remote FASTA import");

        let sequence_asset = AssetRef::all(conn)
            .into_iter()
            .find(|asset_ref| asset_ref.uri == fasta)
            .expect("should persist remote sequence AssetRef");
        assert_eq!(
            sequence_asset.checksum, None,
            "remote shallow asset should not require a retained local checksum"
        );
        assert!(
            context
                .workspace()
                .asset_dir()
                .expect("should have asset directory")
                .read_dir()
                .expect("should read asset directory")
                .next()
                .is_none(),
            "remote shallow import should not retain a local asset"
        );

        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "m123", None);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, context.workspace(), &block_group_id, false)
                .expect("should stream shallow sequence from remote AssetRef"),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        let sequence = Sequence::query_by_blockgroup(conn, context.workspace(), &block_group_id)
            .into_iter()
            .find(|sequence| sequence.asset_ref_id == Some(sequence_asset.id))
            .expect("should resolve shallow sequence through remote AssetRef");
        assert_eq!(
            sequence
                .get_sequence(2, 8)
                .expect("should read remote sequence slice"),
            "CGATCG"
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
        )
        .unwrap();
        let result_error = commit_operation_summary(&context, &operation_summary).unwrap_err();

        assert!(matches!(result_error, OperationError::NoChanges));
    }
}
