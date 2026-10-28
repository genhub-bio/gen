use std::{
    io::{Read, Write as _},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use fake_gcs::FakeGcsServer;
use r#gen::{core::Workspace, get_connection};
use gen_models::{assets::AssetRef, history::dolt::commit_all};
use tempfile::{TempDir, tempdir};
use url::Url;

#[path = "support/fake_gcs.rs"]
mod fake_gcs;

const TEST_TRANSFER_ID: &str = "00000000-0000-0000-0000-000000000001";
const STREAM_PAUSE: Duration = Duration::from_millis(1_150);
const CHILD_TIMEOUT: Duration = Duration::from_secs(20);

fn gen_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_gen"))
}

fn run_gen(repo_root: &Path, args: &[&str]) -> std::process::Output {
    Command::new(gen_binary())
        .current_dir(repo_root)
        .args(args)
        .output()
        .expect("should run gen command")
}

fn assert_success(output: &std::process::Output, context: &str) {
    assert!(
        output.status.success(),
        "{context}: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

struct SourceFixture {
    _temporary_directory: TempDir,
    _fake_gcs: FakeGcsServer,
    graph_url: String,
    local_graph_url: String,
    asset_id: String,
    stored_filename: String,
    stored_bytes: Vec<u8>,
    materialized_bytes: Vec<u8>,
}

fn source_fixture() -> SourceFixture {
    let temporary_directory = tempdir().expect("should create source fixture directory");
    let source_root = temporary_directory.path().join("source");
    std::fs::create_dir_all(&source_root).expect("should create source Gen workspace");
    assert_success(
        &run_gen(&source_root, &["init"]),
        "source Gen init should succeed",
    );

    let input_path = source_root.join("small.txt");
    let materialized_bytes = b"a small asset for clone progress\n".to_vec();
    std::fs::write(&input_path, &materialized_bytes).expect("should write source asset");
    let input_path_text = input_path
        .to_str()
        .expect("should encode source asset path");
    assert_success(
        &run_gen(
            &source_root,
            &["add-file", input_path_text, "--message", "clone fixture"],
        ),
        "source asset should be committed",
    );

    let graph_path = source_root.join(".gen/default.db");
    let graph = get_connection(&graph_path).expect("should open source graph");
    let size_rows_updated = graph
        .execute(
            "UPDATE gen_asset_refs SET size = NULL WHERE size IS NOT NULL",
            [],
        )
        .expect("should clear optional asset size for Content-Length coverage");
    if size_rows_updated > 0 {
        commit_all(&graph, "omit optional asset size")
            .expect("should commit source asset without a size hint");
    }
    let assets = AssetRef::all(&graph).expect("should read source assets");
    assert_eq!(assets.len(), 1, "source fixture should have one asset");
    let asset = assets
        .into_iter()
        .next()
        .expect("should contain the source asset");
    assert_eq!(asset.size, None, "asset metadata should omit its size");
    let asset_id = asset.id.to_string();
    let stored_path = asset
        .versioned_store_path(&Workspace::new(&source_root))
        .expect("should resolve source asset bytes");
    let stored_filename = stored_path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("should identify versioned source asset")
        .to_string();
    let stored_bytes = std::fs::read(stored_path).expect("should read versioned source asset");
    drop(graph);
    let fake_gcs = FakeGcsServer::start();
    let gcs_database = fake_gcs.create_database(&graph_path);
    let graph_url = gcs_database.uri();
    let local_graph_url = Url::from_file_path(&graph_path)
        .expect("should create source graph file URL")
        .to_string();

    SourceFixture {
        _temporary_directory: temporary_directory,
        _fake_gcs: fake_gcs,
        graph_url,
        local_graph_url,
        asset_id,
        stored_filename,
        stored_bytes,
        materialized_bytes,
    }
}

enum AssetResponse {
    Stream(Vec<u8>),
    Truncated {
        prefix: Vec<u8>,
        url_secret: &'static str,
    },
    Failure {
        status: &'static str,
        body: &'static [u8],
        url_secret: &'static str,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ServerEvent {
    CapabilityReceived,
    AssetListRequested,
    AssetBodyPaused,
}

struct CloneServer {
    repository_url: String,
    events: Receiver<ServerEvent>,
    release_capability: Sender<()>,
    release_asset_body: Sender<()>,
    stopped: Arc<AtomicBool>,
    server_thread: Option<JoinHandle<Vec<String>>>,
}

impl CloneServer {
    fn start(graph_url: &str, asset_id: &str, asset_response: AssetResponse) -> Self {
        Self::start_with_capability_gate(graph_url, asset_id, asset_response, false)
    }

    fn start_with_capability_gate(
        graph_url: &str,
        asset_id: &str,
        asset_response: AssetResponse,
        hold_capability: bool,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind clone test server");
        listener
            .set_nonblocking(true)
            .expect("should make clone test listener nonblocking");
        let address = listener
            .local_addr()
            .expect("should read clone test server address");
        let origin = format!("http://{address}");
        let repository_url = format!("{origin}/api/repos/alice/trepo");
        let graph_url = graph_url.to_string();
        let asset_id = asset_id.to_string();
        let stopped = Arc::new(AtomicBool::new(false));
        let server_stopped = Arc::clone(&stopped);
        let (event_sender, events) = mpsc::channel();
        let (release_capability, capability_gate) = mpsc::channel();
        let (release_asset_body, asset_body_gate) = mpsc::channel();
        let server_thread = thread::spawn(move || {
            let mut requests = Vec::new();
            while requests.len() < 3 && !server_stopped.load(Ordering::Acquire) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                        continue;
                    }
                    Err(error) => panic!("should accept clone request: {error}"),
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .expect("should bound clone test request reads");
                stream
                    .set_write_timeout(Some(Duration::from_secs(5)))
                    .expect("should bound clone test response writes");
                let request = read_http_request(&mut stream);
                let request_line = request.lines().next().unwrap_or_default().to_string();
                requests.push(request);

                if request_line.contains("/remote-capability ") {
                    let _ = event_sender.send(ServerEvent::CapabilityReceived);
                    if hold_capability
                        && capability_gate
                            .recv_timeout(Duration::from_secs(10))
                            .is_err()
                    {
                        break;
                    }
                    let body = serde_json::json!({
                        "remote_url": graph_url,
                        "expires_at": "2035-01-01T00:00:00Z",
                        "default_branch": "main",
                        "transfer_id": TEST_TRANSFER_ID
                    })
                    .to_string();
                    if write_http_response(&mut stream, "200 OK", body.as_bytes()).is_err() {
                        break;
                    }
                } else if request_line.contains("/asset-transfers ") {
                    let _ = event_sender.send(ServerEvent::AssetListRequested);
                    let asset_url = match &asset_response {
                        AssetResponse::Stream(_) => format!("{origin}/assets/{asset_id}"),
                        AssetResponse::Truncated { url_secret, .. } => {
                            format!("{origin}/assets/{asset_id}?X-Goog-Signature={url_secret}")
                        }
                        AssetResponse::Failure { url_secret, .. } => {
                            format!("{origin}/assets/{asset_id}?X-Goog-Signature={url_secret}")
                        }
                    };
                    let body = serde_json::json!({
                        "assets": [{ "id": asset_id, "url": asset_url }]
                    })
                    .to_string();
                    if write_http_response(&mut stream, "200 OK", body.as_bytes()).is_err() {
                        break;
                    }
                } else if request_line.starts_with("GET /assets/") {
                    match &asset_response {
                        AssetResponse::Stream(bytes) => {
                            let pause_at = (bytes.len() * 2 / 3).clamp(1, bytes.len() - 1);
                            let first_pause_at = (pause_at / 2).max(1);
                            if write_response_prefix(
                                &mut stream,
                                "200 OK",
                                bytes.len(),
                                &bytes[..first_pause_at],
                            )
                            .is_err()
                            {
                                break;
                            }
                            thread::sleep(STREAM_PAUSE);
                            if stream.write_all(&bytes[first_pause_at..pause_at]).is_err() {
                                break;
                            }
                            if event_sender.send(ServerEvent::AssetBodyPaused).is_err() {
                                break;
                            }
                            if asset_body_gate
                                .recv_timeout(Duration::from_secs(10))
                                .is_err()
                            {
                                break;
                            }
                            if stream.write_all(&bytes[pause_at..]).is_err() {
                                break;
                            }
                        }
                        AssetResponse::Truncated { prefix, .. } => {
                            let declared_length = prefix.len() + 16;
                            if write_response_prefix(&mut stream, "200 OK", declared_length, prefix)
                                .is_err()
                            {
                                break;
                            }
                        }
                        AssetResponse::Failure { status, body, .. } => {
                            if write_http_response(&mut stream, status, body).is_err() {
                                break;
                            }
                        }
                    }
                } else {
                    panic!("unexpected clone test request: {request_line}");
                }
            }
            requests
        });

        Self {
            repository_url,
            events,
            release_capability,
            release_asset_body,
            stopped,
            server_thread: Some(server_thread),
        }
    }

    fn next_event(&self, timeout: Duration) -> ServerEvent {
        self.events
            .recv_timeout(timeout)
            .expect("should receive the expected clone server event")
    }

    fn release_capability(&self) {
        self.release_capability
            .send(())
            .expect("should release held clone capability response");
    }

    fn release_asset_body(&self) {
        self.release_asset_body
            .send(())
            .expect("should release held clone asset response");
    }

    fn join(mut self) -> Vec<String> {
        self.stopped.store(true, Ordering::Release);
        self.server_thread
            .take()
            .expect("should own clone server thread")
            .join()
            .expect("should finish clone test server")
    }
}

impl Drop for CloneServer {
    fn drop(&mut self) {
        let _ = self.release_capability.send(());
        let _ = self.release_asset_body.send(());
        self.stopped.store(true, Ordering::Release);
        if let Some(server_thread) = self.server_thread.take() {
            let _ = server_thread.join();
        }
    }
}

fn read_http_request(stream: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let read = stream
            .read(&mut buffer)
            .expect("should read clone HTTP request");
        if read == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..read]);
        let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") else {
            continue;
        };
        let body_start = header_end + 4;
        let headers = String::from_utf8_lossy(&request[..body_start]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        if request.len() >= body_start + content_length {
            break;
        }
    }
    String::from_utf8_lossy(&request).into_owned()
}

fn write_http_response(stream: &mut TcpStream, status: &str, body: &[u8]) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

fn write_response_prefix(
    stream: &mut TcpStream,
    status: &str,
    content_length: usize,
    prefix: &[u8],
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
    )?;
    stream.write_all(prefix)
}

#[derive(Clone, Copy)]
enum OutputStream {
    Stdout,
    Stderr,
}

struct RunningClone {
    child: Option<Child>,
    output_chunks: Receiver<(OutputStream, Vec<u8>)>,
    reader_threads: Vec<JoinHandle<()>>,
    stdout: String,
    stderr: String,
}

impl RunningClone {
    fn spawn(parent: &Path, repository_url: &str) -> Self {
        let mut child = Command::new(gen_binary())
            .current_dir(parent)
            .arg("clone")
            .arg(repository_url)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_remove("GENHUB_API_KEY")
            .spawn()
            .expect("should start gen clone process");
        let (chunk_sender, output_chunks) = mpsc::channel();
        let stdout = child.stdout.take().expect("should capture clone stdout");
        let stderr = child.stderr.take().expect("should capture clone stderr");
        let reader_threads = vec![
            spawn_output_reader(stdout, OutputStream::Stdout, chunk_sender.clone()),
            spawn_output_reader(stderr, OutputStream::Stderr, chunk_sender),
        ];
        Self {
            child: Some(child),
            output_chunks,
            reader_threads,
            stdout: String::new(),
            stderr: String::new(),
        }
    }

    fn wait_for_stderr(&mut self, fragment: &str, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while !self.stderr.contains(fragment) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return false;
            }
            match self.output_chunks.recv_timeout(remaining) {
                Ok((stream, bytes)) => self.append_output(stream, bytes),
                Err(_) => return self.stderr.contains(fragment),
            }
        }
        true
    }

    fn try_wait(&mut self) -> Option<ExitStatus> {
        self.child
            .as_mut()
            .and_then(|child| child.try_wait().expect("should check clone process status"))
    }

    fn finish(mut self) -> (ExitStatus, String, String) {
        let deadline = Instant::now() + CHILD_TIMEOUT;
        let status = loop {
            if let Some(status) = self.try_wait() {
                break status;
            }
            assert!(
                Instant::now() < deadline,
                "clone process should finish promptly; stderr={}",
                self.stderr
            );
            self.drain_output(Duration::from_millis(20));
        };
        self.child.take();
        for reader_thread in self.reader_threads.drain(..) {
            reader_thread
                .join()
                .expect("should finish clone output reader");
        }
        self.drain_available_output();
        (
            status,
            std::mem::take(&mut self.stdout),
            std::mem::take(&mut self.stderr),
        )
    }

    fn drain_output(&mut self, timeout: Duration) {
        if let Ok((stream, bytes)) = self.output_chunks.recv_timeout(timeout) {
            self.append_output(stream, bytes);
        }
    }

    fn drain_available_output(&mut self) {
        while let Ok((stream, bytes)) = self.output_chunks.try_recv() {
            self.append_output(stream, bytes);
        }
    }

    fn append_output(&mut self, stream: OutputStream, bytes: Vec<u8>) {
        let text = String::from_utf8_lossy(&bytes);
        match stream {
            OutputStream::Stdout => self.stdout.push_str(&text),
            OutputStream::Stderr => self.stderr.push_str(&text),
        }
    }
}

fn spawn_output_reader<R>(
    mut reader: R,
    stream: OutputStream,
    sender: Sender<(OutputStream, Vec<u8>)>,
) -> JoinHandle<()>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut buffer = [0_u8; 2048];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) | Err(_) => return,
                Ok(read) => {
                    if sender.send((stream, buffer[..read].to_vec())).is_err() {
                        return;
                    }
                }
            }
        }
    })
}

impl Drop for RunningClone {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        for reader_thread in self.reader_threads.drain(..) {
            let _ = reader_thread.join();
        }
    }
}

#[test]
fn test_clone_cli_reports_wait_and_download_progress_with_content_length_totals() {
    let source = source_fixture();
    assert!(
        source.stored_bytes.len() > 1,
        "asset body should be chunkable"
    );
    let server = CloneServer::start_with_capability_gate(
        &source.graph_url,
        &source.asset_id,
        AssetResponse::Stream(source.stored_bytes.clone()),
        true,
    );
    let clone_parent = tempdir().expect("should create clone destination parent");
    let mut process = RunningClone::spawn(clone_parent.path(), &server.repository_url);

    assert_eq!(
        server.next_event(Duration::from_secs(5)),
        ServerEvent::CapabilityReceived
    );
    assert!(
        process.wait_for_stderr(
            "Requesting graph access (attempt 1/2)...",
            Duration::from_secs(3)
        ),
        "CLI should report its remote capability wait before the server responds; stderr={}",
        process.stderr
    );
    assert!(
        process.try_wait().is_none(),
        "clone should remain active while the capability response is held"
    );
    server.release_capability();

    assert_eq!(
        server.next_event(Duration::from_secs(10)),
        ServerEvent::AssetListRequested
    );
    assert!(
        process.wait_for_stderr(
            "Cloning graph history (attempt 1/2)...",
            Duration::from_secs(3)
        ),
        "CLI should identify graph history work; stderr={}",
        process.stderr
    );
    assert_eq!(
        server.next_event(Duration::from_secs(10)),
        ServerEvent::AssetBodyPaused
    );
    let bytes_staged = source.stored_bytes.len() * 2 / 3;
    let byte_progress = format!(
        "{} B of {} B staged (",
        bytes_staged,
        source.stored_bytes.len()
    );
    assert!(
        process.wait_for_stderr(&byte_progress, Duration::from_secs(5)),
        "CLI should report downloaded bytes while the response body remains incomplete; stderr={}",
        process.stderr
    );
    assert!(
        process.stderr.contains("Downloading 1 asset (")
            && process.stderr.contains("size unknown for 1 asset"),
        "CLI should report an asset count and identify metadata sizes that were unknown; stderr={}",
        process.stderr
    );
    assert!(
        process.try_wait().is_none(),
        "clone should remain active while the asset response body is held"
    );
    server.release_asset_body();

    let (status, stdout, stderr) = process.finish();
    assert!(
        status.success(),
        "clone should succeed: stdout={stdout} stderr={stderr}"
    );
    assert!(
        stderr.contains("Asset 1/1 ready; checksum verified"),
        "CLI should report checksum-verified asset completion: {stderr}"
    );
    assert!(
        stderr.contains("Clone of branch 'main' is complete."),
        "CLI should report completed clone: {stderr}"
    );
    let destination = clone_parent.path().join("trepo");
    assert_eq!(
        std::fs::read(destination.join("small.txt")).expect("should read cloned asset"),
        source.materialized_bytes,
        "clone should materialize the verified source bytes"
    );
    assert_eq!(
        server.join().len(),
        3,
        "clone should use the expected three HTTP requests"
    );
}

#[test]
fn test_clone_cli_rejects_non_cloud_network_capabilities() {
    let source = source_fixture();
    let server = CloneServer::start(
        &source.local_graph_url,
        &source.asset_id,
        AssetResponse::Stream(source.stored_bytes.clone()),
    );
    let clone_parent = tempdir().expect("should create clone destination parent");
    let process = RunningClone::spawn(clone_parent.path(), &server.repository_url);

    let (status, stdout, stderr) = process.finish();
    assert!(
        !status.success(),
        "clone should reject a non-cloud graph capability"
    );
    let combined_output = format!("{stdout}\n{stderr}");
    assert!(
        !combined_output.contains(&source.local_graph_url),
        "CLI should not print the rejected capability URL: {combined_output}"
    );
    let requests = server.join();
    assert_eq!(
        requests.len(),
        1,
        "clone should reject before asset transfer"
    );
    assert!(
        requests[0].contains("/remote-capability"),
        "clone should only request the graph capability"
    );
}

#[test]
fn test_clone_cli_reports_http_failure_without_leaking_signed_url_or_provider_body() {
    let source = source_fixture();
    const URL_SECRET: &str = "clone-signed-url-secret";
    const PROVIDER_SECRET: &[u8] = b"provider response includes clone-provider-secret";
    let server = CloneServer::start(
        &source.graph_url,
        &source.asset_id,
        AssetResponse::Failure {
            status: "503 Service Unavailable",
            body: PROVIDER_SECRET,
            url_secret: URL_SECRET,
        },
    );
    let clone_parent = tempdir().expect("should create clone destination parent");
    let process = RunningClone::spawn(clone_parent.path(), &server.repository_url);

    let (status, stdout, stderr) = process.finish();
    assert!(!status.success(), "asset HTTP failure should fail clone");
    assert!(
        stderr.contains("Clone failed after ")
            && stderr.contains("during downloading branch assets")
            && stderr.contains("HTTP 503"),
        "CLI should report elapsed time, transfer stage, and HTTP status: {stderr}"
    );
    assert!(
        stderr.contains("asset download 1/1") && stderr.contains("failed after "),
        "CLI should include the failed asset phase and elapsed time: {stderr}"
    );
    let combined_output = format!("{stdout}\n{stderr}");
    assert!(
        !combined_output.contains(URL_SECRET),
        "CLI should not reveal the signed asset URL: {combined_output}"
    );
    assert!(
        !combined_output.contains("clone-provider-secret"),
        "CLI should not reveal provider response bodies: {combined_output}"
    );
    assert_eq!(
        server.join().len(),
        3,
        "failed clone should use the expected three HTTP requests"
    );
}

#[test]
fn test_clone_cli_sanitizes_truncated_asset_body_and_keeps_resume_bytes_staged() {
    let source = source_fixture();
    const URL_SECRET: &str = "truncated-clone-signed-url-secret";
    const PROVIDER_SECRET: &[u8] = b"partial bytes include truncated-provider-secret";
    let server = CloneServer::start(
        &source.graph_url,
        &source.asset_id,
        AssetResponse::Truncated {
            prefix: PROVIDER_SECRET.to_vec(),
            url_secret: URL_SECRET,
        },
    );
    let clone_parent = tempdir().expect("should create clone destination parent");
    let process = RunningClone::spawn(clone_parent.path(), &server.repository_url);

    let (status, stdout, stderr) = process.finish();
    assert!(
        !status.success(),
        "truncated asset response should fail clone"
    );
    assert!(
        stderr.contains("Clone failed after ")
            && stderr.contains("during downloading branch assets")
            && stderr.contains("asset download 1/1")
            && stderr.contains("failed after ")
            && stderr.contains("during downloading asset bytes")
            && stderr.contains("asset response body read failed"),
        "CLI should report elapsed time, asset phase, and a safe body-read error: {stderr}"
    );
    assert!(
        !stderr.contains("checksum verified") && !stderr.contains("Asset 1/1 ready"),
        "CLI should not report a truncated asset as complete: {stderr}"
    );
    let combined_output = format!("{stdout}\n{stderr}");
    assert!(
        !combined_output.contains(URL_SECRET),
        "CLI should not reveal the signed asset URL: {combined_output}"
    );
    assert!(
        !combined_output.contains("truncated-provider-secret"),
        "CLI should not reveal bytes received from the incomplete provider response: {combined_output}"
    );

    let destination = clone_parent.path().join("trepo");
    assert!(
        !destination.join("small.txt").exists(),
        "an incomplete response should not create the materialized asset"
    );
    let staged_path = destination
        .join(".gen/assets")
        .join(format!("{}.tmp", source.stored_filename));
    assert_eq!(
        std::fs::read(staged_path).expect("should retain downloaded resume prefix"),
        PROVIDER_SECRET,
        "incomplete response bytes should remain staged for a future resume"
    );
    assert_eq!(
        server.join().len(),
        3,
        "truncated clone should use the expected three HTTP requests"
    );
}
