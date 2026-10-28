use std::{
    collections::{HashMap, HashSet},
    io::{self, Read as _, Write as _},
    net::{TcpListener, TcpStream},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use rusqlite::blockcachevfs::{BlockCacheVfs, CloudConnectionUri, Storage};
use serde_json::{Value, json};
use tempfile::tempdir;

static NEXT_PREFIX: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct StoredObject {
    body: Vec<u8>,
    generation: String,
}

#[derive(Default)]
struct ObjectStore {
    buckets: HashSet<String>,
    objects: HashMap<String, StoredObject>,
    next_generation: u64,
    write_request_count: u64,
    writes_denied: bool,
}

pub struct FakeGcsServer {
    endpoint: String,
    stopped: Arc<AtomicBool>,
    store: Arc<Mutex<ObjectStore>>,
    worker: Option<JoinHandle<()>>,
}

impl FakeGcsServer {
    pub fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind fake GCS server");
        listener
            .set_nonblocking(true)
            .expect("should make fake GCS listener nonblocking");
        let address = listener
            .local_addr()
            .expect("should read fake GCS server address");
        let stopped = Arc::new(AtomicBool::new(false));
        let server_stopped = Arc::clone(&stopped);
        let store = Arc::new(Mutex::new(ObjectStore::default()));
        let server_store = Arc::clone(&store);
        let worker = thread::spawn(move || {
            while !server_stopped.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => match read_request(&mut stream) {
                        Ok(request) => {
                            let response = handle_request(&server_store, &request);
                            let _ = stream.write_all(&response);
                        }
                        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {}
                        Err(error) => panic!("should read fake GCS request: {error}"),
                    },
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("should accept fake GCS request: {error}"),
                }
            }
        });

        Self {
            endpoint: format!("http://{address}"),
            stopped,
            store,
            worker: Some(worker),
        }
    }

    pub fn create_database(&self, local_database: &Path) -> FakeGcsDatabase {
        let suffix = NEXT_PREFIX.fetch_add(1, Ordering::Relaxed);
        let bucket = format!("gen-test-{}-{suffix}", std::process::id());
        let prefix = format!("test-repositories/{suffix}/.gen/graph_db/");
        let project = "gen-test-project";
        let database = FakeGcsDatabase {
            bucket,
            prefix,
            project: project.to_string(),
            endpoint: self.endpoint.clone(),
            store: Arc::clone(&self.store),
        };
        database.initialize_container();
        database.upload(local_database);
        database
    }
}

pub struct FakeGcsDatabase {
    bucket: String,
    prefix: String,
    project: String,
    endpoint: String,
    store: Arc<Mutex<ObjectStore>>,
}

impl FakeGcsDatabase {
    pub fn write_request_count(&self) -> u64 {
        self.store
            .lock()
            .expect("should lock fake GCS object store")
            .write_request_count
    }

    pub fn set_writes_denied(&self, denied: bool) {
        self.store
            .lock()
            .expect("should lock fake GCS object store")
            .writes_denied = denied;
    }

    pub fn storage(&self) -> Storage {
        Storage::google_json_with_endpoint(
            &self.project,
            format!("{}/{}/", self.bucket, self.prefix.trim_matches('/')),
            &self.endpoint,
        )
    }

    pub fn uri(&self) -> String {
        CloudConnectionUri::gcs(
            &self.bucket,
            &self.prefix,
            &self.project,
            "fake-gcs-token",
            "default.db",
            Some(&self.endpoint),
        )
        .expect("should build fake GCS read capability")
        .as_str()
        .to_string()
    }

    fn upload(&self, local_database: &Path) {
        let storage = Storage::google_json_with_endpoint(
            &self.project,
            format!("{}/{}/", self.bucket, self.prefix.trim_matches('/')),
            &self.endpoint,
        );
        let cache_directory = tempdir().expect("should create fake GCS seed cache");
        let block_cache = BlockCacheVfs::builder(cache_directory.path())
            .expect("should create fake GCS seed VFS")
            .auth_callback(|_provider, _account, _container| Ok("fake-gcs-token".to_string()))
            .init_owned()
            .expect("should initialize fake GCS seed VFS");
        block_cache
            .create_database(&storage, local_database, "default.db")
            .expect("should upload source graph to fake GCS");
    }

    fn initialize_container(&self) {
        let storage = Storage::google_json_with_endpoint(
            &self.project,
            format!("{}/{}/", self.bucket, self.prefix.trim_matches('/')),
            &self.endpoint,
        );
        let cache_directory = tempdir().expect("should create fake GCS seed cache");
        let block_cache = BlockCacheVfs::builder(cache_directory.path())
            .expect("should create fake GCS seed VFS")
            .auth_callback(|_provider, _account, _container| Ok("fake-gcs-token".to_string()))
            .init_owned()
            .expect("should initialize fake GCS seed VFS");
        block_cache
            .initialize_container(&storage, None)
            .expect("should initialize fake GCS container");
    }
}

impl Drop for FakeGcsServer {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        let wake_address = self
            .endpoint
            .strip_prefix("http://")
            .expect("fake GCS endpoint should be HTTP");
        let _ = TcpStream::connect(wake_address);
        if let Some(worker) = self.worker.take() {
            worker.join().expect("should stop fake GCS server");
        }
    }
}

struct HttpRequest {
    method: String,
    target: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> io::Result<HttpRequest> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "fake GCS request closed before its headers completed",
            ));
        }
        bytes.extend_from_slice(&buffer[..read]);
        let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let content_length = String::from_utf8_lossy(&bytes[..header_end])
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        let body_start = header_end + 4;
        if bytes.len() >= body_start + content_length {
            let request_line = String::from_utf8_lossy(&bytes[..header_end]);
            let request_line = request_line
                .lines()
                .next()
                .ok_or_else(|| io::Error::other("fake GCS request has no request line"))?;
            let mut request_parts = request_line.split_whitespace();
            let method = request_parts.next().unwrap_or_default().to_string();
            let target = request_parts.next().unwrap_or_default().to_string();
            return Ok(HttpRequest {
                method,
                target,
                body: bytes[body_start..body_start + content_length].to_vec(),
            });
        }
    }
}

fn handle_request(store: &Arc<Mutex<ObjectStore>>, request: &HttpRequest) -> Vec<u8> {
    if request.method == "POST" && request.target.starts_with("/storage/v1/b?") {
        let bucket = serde_json::from_slice::<Value>(&request.body)
            .ok()
            .and_then(|body| body.get("name").and_then(Value::as_str).map(str::to_string));
        let Some(bucket) = bucket else {
            return json_response("400 Bad Request", json!({"error":"missing bucket name"}));
        };
        let mut store = store.lock().expect("should lock fake GCS object store");
        store.write_request_count += 1;
        if store.writes_denied {
            return json_response("403 Forbidden", json!({"error":"writes are disabled"}));
        }
        let status = if store.buckets.insert(bucket.clone()) {
            "200 OK"
        } else {
            "409 Conflict"
        };
        return json_response(status, json!({"name":bucket}));
    }

    if request.method == "POST" && request.target.starts_with("/upload/storage/v1/b/") {
        let Some((bucket, object_name)) = upload_object_name(&request.target) else {
            return json_response("400 Bad Request", json!({"error":"missing object name"}));
        };
        let mut store = store.lock().expect("should lock fake GCS object store");
        store.write_request_count += 1;
        if store.writes_denied {
            return json_response("403 Forbidden", json!({"error":"writes are disabled"}));
        }
        store.next_generation += 1;
        let generation = store.next_generation.to_string();
        store.objects.insert(
            format!("{bucket}/{object_name}"),
            StoredObject {
                body: request.body.clone(),
                generation: generation.clone(),
            },
        );
        return json_response("200 OK", json!({"generation":generation}));
    }

    if request.method == "GET" && request.target.starts_with("/storage/v1/b/") {
        let Some((bucket, prefix)) = list_prefix(&request.target) else {
            return json_response("400 Bad Request", json!({"error":"invalid object listing"}));
        };
        let store = store.lock().expect("should lock fake GCS object store");
        let items = store
            .objects
            .iter()
            .filter_map(|(key, object)| {
                let object_name = key.strip_prefix(&format!("{bucket}/"))?;
                object_name
                    .starts_with(&prefix)
                    .then(|| json!({"name":object_name,"generation":object.generation}))
            })
            .collect::<Vec<_>>();
        return json_response("200 OK", json!({"items":items}));
    }

    if request.method == "GET" && request.target.starts_with("/download/storage/v1/b/") {
        let Some((bucket, object_name)) = download_object_name(&request.target) else {
            return json_response("400 Bad Request", json!({"error":"invalid object path"}));
        };
        let store = store.lock().expect("should lock fake GCS object store");
        let Some(object) = store.objects.get(&format!("{bucket}/{object_name}")) else {
            return json_response("404 Not Found", json!({"error":"object not found"}));
        };
        let checksum = STANDARD.encode(crc32c::crc32c(&object.body).to_be_bytes());
        let mut response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nx-goog-generation: {}\r\nx-goog-hash: crc32c={}\r\nConnection: close\r\n\r\n",
            object.body.len(),
            object.generation,
            checksum
        )
        .into_bytes();
        response.extend_from_slice(&object.body);
        return response;
    }

    json_response("404 Not Found", json!({"error":"unknown GCS endpoint"}))
}

fn upload_object_name(target: &str) -> Option<(String, String)> {
    let (path, query) = target.split_once('?')?;
    let bucket = path
        .strip_prefix("/upload/storage/v1/b/")?
        .strip_suffix("/o")?
        .to_string();
    let object_name = query_parameter(query, "name")?;
    Some((bucket, object_name))
}

fn list_prefix(target: &str) -> Option<(String, String)> {
    let (path, query) = target.split_once('?')?;
    let bucket = path.strip_prefix("/storage/v1/b/")?.strip_suffix("/o")?;
    Some((
        bucket.to_string(),
        query_parameter(query, "prefix").unwrap_or_default(),
    ))
}

fn download_object_name(target: &str) -> Option<(String, String)> {
    let path = target.split_once('?').map_or(target, |(path, _)| path);
    let rest = path.strip_prefix("/download/storage/v1/b/")?;
    let (bucket, encoded_object) = rest.split_once("/o/")?;
    Some((bucket.to_string(), percent_decode(encoded_object)?))
}

fn query_parameter(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (key, value) = pair.split_once('=')?;
        (percent_decode(key).as_deref() == Some(name))
            .then(|| percent_decode(value))
            .flatten()
    })
}

fn percent_decode(encoded: &str) -> Option<String> {
    let mut decoded = Vec::with_capacity(encoded.len());
    let mut bytes = encoded.bytes();
    while let Some(byte) = bytes.next() {
        match byte {
            b'%' => {
                let high = bytes.next()?;
                let low = bytes.next()?;
                decoded.push((hex_value(high)? << 4) | hex_value(low)?);
            }
            b'+' => decoded.push(b' '),
            byte => decoded.push(byte),
        }
    }
    String::from_utf8(decoded).ok()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn json_response(status: &str, body: Value) -> Vec<u8> {
    let body = body.to_string();
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}
