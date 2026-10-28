//! Lazy, read-only asset access for a direct browser graph connection.
//!
//! The graph remote remains responsible for authorizing a repository and immutable snapshot.
//! This source obtains the existing asset-transfer manifest only when a graph query opens a
//! missing local asset, then gives the range reader a short-lived URL for that exact asset.

use std::{
    collections::HashMap,
    io,
    sync::{Arc, Mutex},
};

use chrono::{NaiveDateTime, Utc};
use gen_core::{AssetReadRequest, AssetReadSeek, DoltHashId, HashId, config::AssetReaderSource};
use gen_models::assets::{AssetUri, LocalAssetUri};
use url::Url;

use super::{browser_range::RangeAssetReader, safe_transfer_error};
use crate::commands::remote::{
    client::{AssetTransferRequest, RemoteOperation, RepositoryRemote, acquire_asset_transfers},
    login_origin,
};

struct SourceRef {
    branch: String,
    snapshot_hash: DoltHashId,
    generation: u64,
}

struct SourceState {
    current_ref: SourceRef,
    manifest_loaded: bool,
    asset_urls: HashMap<HashId, String>,
    closed: bool,
}

struct BrowserAssetSourceInner {
    repository: RepositoryRemote,
    bucket: String,
    storage_origin: String,
    namespace: String,
    slug: String,
    state: Mutex<SourceState>,
}

/// Serves immutable branch assets from signed object URLs without downloading complete files.
///
/// The reader resolves a GenHub transfer manifest for a published commit and keeps the returned
/// URLs in memory. It never sends the CAB to object storage, writes to the graph VFS, or logs a
/// signed URL.
pub(super) struct BrowserAssetSource {
    inner: Arc<BrowserAssetSourceInner>,
}

impl BrowserAssetSource {
    /// Creates the source from the repository's CAB bucket and the graph's selected published ref.
    ///
    /// # Errors
    ///
    /// Returns an error if the repository URL, branch, storage endpoint, or published identity is
    /// invalid.
    pub(super) fn new(
        repository: RepositoryRemote,
        bucket: String,
        storage_origin: &str,
        branch: &str,
        snapshot_hash: DoltHashId,
    ) -> io::Result<Self> {
        validate_branch(branch)?;
        let storage_origin = canonical_storage_origin(storage_origin)?;
        let repository_url = Url::parse(repository.canonical_url())
            .map_err(|_| invalid_source("invalid repository identity"))?;
        let segments = repository_url
            .path_segments()
            .ok_or_else(|| invalid_source("invalid repository identity"))?
            .collect::<Vec<_>>();
        let ["api", "repos", namespace, slug] = segments.as_slice() else {
            return Err(invalid_source("invalid repository identity"));
        };

        Ok(Self {
            inner: Arc::new(BrowserAssetSourceInner {
                repository,
                bucket,
                storage_origin,
                namespace: (*namespace).to_string(),
                slug: (*slug).to_string(),
                state: Mutex::new(SourceState {
                    current_ref: SourceRef {
                        branch: branch.to_string(),
                        snapshot_hash,
                        generation: 0,
                    },
                    manifest_loaded: false,
                    asset_urls: HashMap::new(),
                    closed: false,
                }),
            }),
        })
    }
}

/// Extracts the GCS bucket and exact object-storage origin from a validated CAB URI.
///
/// The default endpoint is Google's HTTPS storage host. An explicit HTTP endpoint is accepted
/// only for loopback development and local fake-GCS tests.
///
/// # Errors
///
/// Returns an error if the CAB URI is malformed, has duplicate endpoint parameters, or names an
/// unsupported storage endpoint.
pub(super) fn cloud_storage_scope(uri: &str) -> io::Result<(String, String)> {
    let (scheme, rest) = uri
        .split_once("://")
        .ok_or_else(|| invalid_source("invalid CAB storage URI"))?;
    let (location, query) = rest
        .split_once('?')
        .ok_or_else(|| invalid_source("invalid CAB storage URI"))?;
    let (bucket, _) = location
        .split_once('/')
        .ok_or_else(|| invalid_source("invalid CAB storage URI"))?;
    if scheme != "gcs" || bucket.is_empty() {
        return Err(invalid_source("invalid CAB storage URI"));
    }
    let mut endpoints = url::form_urlencoded::parse(query.as_bytes())
        .filter(|(name, _)| name == "endpoint")
        .map(|(_, value)| value.into_owned());
    let endpoint = endpoints
        .next()
        .unwrap_or_else(|| "https://storage.googleapis.com".to_string());
    if endpoints.next().is_some() {
        return Err(invalid_source("duplicate CAB storage endpoint"));
    }
    let origin = canonical_storage_origin(&endpoint)?;
    Ok((bucket.to_string(), origin))
}

impl AssetReaderSource for BrowserAssetSource {
    fn set_ref(&self, branch: &str, snapshot_hash: &str) -> io::Result<()> {
        validate_branch(branch)?;
        let snapshot_hash = DoltHashId::try_from(snapshot_hash)
            .map_err(|_| invalid_source("invalid published graph hash"))?;
        let mut state = lock_state(&self.inner)?;
        if state.closed {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "browser asset source is closed",
            ));
        }
        if state.current_ref.branch == branch && state.current_ref.snapshot_hash == snapshot_hash {
            return Ok(());
        }
        let generation =
            state.current_ref.generation.checked_add(1).ok_or_else(|| {
                io::Error::other("browser asset source ref changed too many times")
            })?;
        state.current_ref = SourceRef {
            branch: branch.to_string(),
            snapshot_hash,
            generation,
        };
        state.manifest_loaded = false;
        state.asset_urls.clear();
        Ok(())
    }

    fn close(&self) {
        if let Ok(mut state) = self.inner.state.lock() {
            state.closed = true;
            state.manifest_loaded = false;
            state.asset_urls.clear();
        }
    }

    fn published_base_hash_for_remote(&self, remote_url: &str) -> Option<DoltHashId> {
        if self.inner.repository.canonical_url() != remote_url {
            return None;
        }
        self.inner
            .state
            .lock()
            .ok()
            .filter(|state| !state.closed)
            .map(|state| state.current_ref.snapshot_hash)
    }

    fn open(&self, request: &AssetReadRequest<'_>) -> io::Result<Box<dyn AssetReadSeek>> {
        if !LocalAssetUri::is_local_path_or_file_uri(request.uri) {
            return Err(invalid_source(
                "only repository file assets can be read remotely",
            ));
        }
        let expected_filename =
            <dyn AssetUri>::from_uri(request.uri).hashed_filename(&request.checksum);
        validate_archive_filename(&expected_filename)?;
        let pin = {
            let state = lock_state(&self.inner)?;
            if state.closed {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "browser asset source is closed",
                ));
            }
            SourceRef {
                branch: state.current_ref.branch.clone(),
                snapshot_hash: state.current_ref.snapshot_hash,
                generation: state.current_ref.generation,
            }
        };

        let inner = Arc::clone(&self.inner);
        let asset_id = request.id;
        let url_provider = Arc::new(move |force_renew: bool| {
            resolve_asset_url(&inner, &pin, asset_id, &expected_filename, force_renew)
        });
        let reader = RangeAssetReader::new(url_provider)?;
        Ok(Box::new(reader))
    }
}

/// Finds the signed object URL for one asset pinned to a published graph snapshot.
///
/// Cached URLs are reused while valid. A refresh requests the existing transfer manifest for the
/// same immutable commit, then rechecks the source generation before installing its URLs.
///
/// # Errors
///
/// Returns an error if the source closed or changed refs during lookup, the asset is absent from
/// the published history, the manifest request fails, or the URL fails scope validation.
fn resolve_asset_url(
    inner: &BrowserAssetSourceInner,
    pin: &SourceRef,
    asset_id: HashId,
    expected_filename: &str,
    force_renew: bool,
) -> io::Result<String> {
    let cached_url = {
        let state = lock_state(inner)?;
        ensure_current_ref(&state, pin)?;
        if !force_renew && state.manifest_loaded {
            Some(state.asset_urls.get(&asset_id).cloned().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "asset is not in the published graph history",
                )
            })?)
        } else {
            None
        }
    };
    if let Some(url) = cached_url
        && !signed_url_near_expiry(&url)
    {
        validate_signed_asset_url(inner, &url, expected_filename)?;
        return Ok(url);
    }

    let manifest = fetch_manifest(inner, pin)?;
    let mut state = lock_state(inner)?;
    ensure_current_ref(&state, pin)?;
    state.asset_urls = manifest;
    state.manifest_loaded = true;
    let url = state.asset_urls.get(&asset_id).cloned().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "asset is not in the published graph history",
        )
    })?;
    validate_signed_asset_url(inner, &url, expected_filename)?;
    Ok(url)
}

/// Requests signed URLs for assets reachable from the pinned graph commit without fetching bodies.
///
/// # Errors
///
/// Returns an error if the existing GenHub transfer-manifest request fails or returns duplicate
/// asset identities.
fn fetch_manifest(
    inner: &BrowserAssetSourceInner,
    pin: &SourceRef,
) -> io::Result<HashMap<HashId, String>> {
    let response = acquire_asset_transfers(
        &inner.repository,
        &AssetTransferRequest {
            operation: RemoteOperation::Pull,
            branch: &pin.branch,
            from_commit: None,
            to_commit: Some(&pin.snapshot_hash),
        },
        login_origin,
    )
    .map_err(|error| io::Error::other(safe_transfer_error(&error)))?;

    let mut asset_urls = HashMap::with_capacity(response.assets.len());
    for transfer in response.assets {
        if asset_urls.insert(transfer.id, transfer.url).is_some() {
            return Err(invalid_source("GenHub returned duplicate asset references"));
        }
    }
    Ok(asset_urls)
}

/// Restricts a returned capability URL to the CAB bucket, repository, and expected archive name.
///
/// # Errors
///
/// Returns an error if the URL uses another origin or points outside the authorized content path.
fn validate_signed_asset_url(
    inner: &BrowserAssetSourceInner,
    url: &str,
    expected_filename: &str,
) -> io::Result<()> {
    let parsed =
        Url::parse(url).map_err(|_| invalid_source("GenHub returned an invalid asset URL"))?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.origin().ascii_serialization() != inner.storage_origin
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.fragment().is_some()
        || parsed.query().is_none_or(str::is_empty)
    {
        return Err(invalid_source("GenHub returned an invalid asset URL"));
    }
    let segments = parsed
        .path_segments()
        .ok_or_else(|| invalid_source("GenHub returned an invalid asset URL"))?
        .collect::<Vec<_>>();
    if segments.len() != 7
        || segments[0] != inner.bucket
        || segments[1] != "repos"
        || segments[2] != inner.namespace
        || segments[3] != inner.slug
        || segments[4] != ".gen"
        || segments[5] != "assets"
        || segments[6] != expected_filename
    {
        return Err(invalid_source(
            "GenHub returned an asset URL outside the authorized repository",
        ));
    }
    Ok(())
}

/// Canonicalizes the exact storage origin allowed by this graph capability.
///
/// # Errors
///
/// Returns an error for URLs containing credentials, paths, query data, or non-loopback HTTP.
fn canonical_storage_origin(origin: &str) -> io::Result<String> {
    let parsed = Url::parse(origin).map_err(|_| invalid_source("invalid CAB storage endpoint"))?;
    let Some(host) = parsed.host_str() else {
        return Err(invalid_source("invalid CAB storage endpoint"));
    };
    if !matches!(parsed.scheme(), "http" | "https")
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.path() != "/"
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(invalid_source("invalid CAB storage endpoint"));
    }
    if parsed.scheme() == "http" && !is_loopback_host(host) {
        return Err(invalid_source(
            "plain HTTP CAB storage is limited to local development",
        ));
    }
    Ok(parsed.origin().ascii_serialization())
}

/// Recognizes host forms suitable for an explicitly configured local development endpoint.
fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

/// Refreshes Google signed URLs before expiry and treats malformed expiry data as stale.
fn signed_url_near_expiry(url: &str) -> bool {
    let Ok(parsed) = Url::parse(url) else {
        return true;
    };
    let query = parsed.query_pairs().collect::<HashMap<_, _>>();
    let (Some(date), Some(seconds)) = (query.get("X-Goog-Date"), query.get("X-Goog-Expires"))
    else {
        return false;
    };
    let Ok(date) = NaiveDateTime::parse_from_str(date, "%Y%m%dT%H%M%SZ") else {
        return true;
    };
    let Ok(seconds) = seconds.parse::<i64>() else {
        return true;
    };
    let Some(lifetime) = chrono::Duration::try_seconds(seconds).filter(|_| seconds > 0) else {
        return true;
    };
    let Some(expires_at) = date.and_utc().checked_add_signed(lifetime) else {
        return true;
    };
    expires_at <= Utc::now() + chrono::Duration::seconds(30)
}

/// Rejects a lookup if its captured ref or generation no longer matches the live source.
///
/// # Errors
///
/// Returns `PermissionDenied` after close or when another published snapshot has been selected.
fn ensure_current_ref(state: &SourceState, pin: &SourceRef) -> io::Result<()> {
    if state.closed {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "browser asset source is closed",
        ));
    }
    if state.current_ref.generation != pin.generation
        || state.current_ref.branch != pin.branch
        || state.current_ref.snapshot_hash != pin.snapshot_hash
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "browser asset source changed its published snapshot",
        ));
    }
    Ok(())
}

/// Rejects empty or control-character branch names before they reach the transfer API.
///
/// # Errors
///
/// Returns an error if the branch name is empty or contains a control character.
fn validate_branch(branch: &str) -> io::Result<()> {
    if branch.trim().is_empty() || branch.chars().any(char::is_control) {
        return Err(invalid_source("invalid published branch"));
    }
    Ok(())
}

/// Ensures a content-addressed object name cannot select a different path in the bucket.
///
/// # Errors
///
/// Returns an error for empty names, path separators, control characters, or dot segments.
fn validate_archive_filename(filename: &str) -> io::Result<()> {
    if filename.is_empty()
        || filename == "."
        || filename == ".."
        || filename
            .chars()
            .any(|character| character == '/' || character == '\\' || character.is_control())
    {
        return Err(invalid_source("invalid content-addressed archive name"));
    }
    Ok(())
}

/// Acquires source metadata without exposing the signed URLs stored in its state.
///
/// # Errors
///
/// Returns an error if another thread panicked while holding the metadata lock.
fn lock_state(
    inner: &BrowserAssetSourceInner,
) -> io::Result<std::sync::MutexGuard<'_, SourceState>> {
    inner
        .state
        .lock()
        .map_err(|_| io::Error::other("browser asset source state is unavailable"))
}

/// Builds sanitized source errors that never include the signed URL or CAB token.
fn invalid_source(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read as _, Write as _},
        net::{TcpListener, TcpStream},
        thread::{self, JoinHandle},
    };

    use gen_core::{AssetReadRequest, DoltHashId, HashId, Sha256Hash, config::AssetReaderSource};
    use gen_models::assets::AssetUri;
    use serde_json::Value;
    use url::Url;

    use super::{
        BrowserAssetSource, SourceRef, resolve_asset_url, signed_url_near_expiry,
        validate_signed_asset_url,
    };
    use crate::commands::remote::client::RepositoryRemote;

    struct CapturedManifestRequest {
        path: String,
        authorization: bool,
        payload: Value,
    }

    fn spawn_manifest_server(
        responses: Vec<Vec<u8>>,
    ) -> (String, JoinHandle<Vec<CapturedManifestRequest>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("should bind fake GenHub");
        let address = listener
            .local_addr()
            .expect("should get fake GenHub address");
        let count = responses.len();
        let server = thread::spawn(move || {
            let mut requests = Vec::with_capacity(count);
            for response_body in responses {
                let (mut stream, _) = listener.accept().expect("should accept manifest request");
                let (request, body) = read_http_request(&mut stream);
                let request_line = request.lines().next().expect("should have request line");
                let path = request_line
                    .split_whitespace()
                    .nth(1)
                    .expect("request line should have a path")
                    .to_string();
                let authorization = request.lines().any(|line| {
                    line.split_once(':')
                        .is_some_and(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                });
                let payload = serde_json::from_slice(&body).expect("should parse request body");
                requests.push(CapturedManifestRequest {
                    path,
                    authorization,
                    payload,
                });
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response_body.len()
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("should write manifest response headers");
                stream
                    .write_all(&response_body)
                    .expect("should write manifest response body");
            }
            requests
        });
        (format!("http://{address}"), server)
    }

    fn read_http_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 1024];
        let header_end = loop {
            let count = stream.read(&mut buffer).expect("should read request");
            assert_ne!(count, 0, "request should contain headers");
            bytes.extend_from_slice(&buffer[..count]);
            if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                break index + 4;
            }
            assert!(
                bytes.len() <= 16 * 1024,
                "request headers should be bounded"
            );
        };
        let headers = String::from_utf8(bytes[..header_end].to_vec())
            .expect("request headers should be valid text");
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().expect("valid content length"))
            })
            .unwrap_or_default();
        while bytes.len() < header_end + content_length {
            let count = stream.read(&mut buffer).expect("should read request body");
            assert_ne!(count, 0, "request body should be complete");
            bytes.extend_from_slice(&buffer[..count]);
        }
        (
            headers,
            bytes[header_end..header_end + content_length].to_vec(),
        )
    }

    fn transfer_response(asset_id: HashId, url: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "assets": [{"id": asset_id, "url": url}],
        }))
        .expect("should encode transfer response")
    }

    fn make_source(repository_url: &str) -> BrowserAssetSource {
        BrowserAssetSource::new(
            RepositoryRemote::parse(repository_url).expect("should parse repository URL"),
            "cab-bucket".to_string(),
            "https://storage.googleapis.com",
            "main",
            DoltHashId([1; 20]),
        )
        .expect("should create an asset source")
    }

    #[test]
    fn signed_asset_url_is_bound_to_cab_bucket_repository_and_archive_name() {
        let source = BrowserAssetSource::new(
            RepositoryRemote::parse("https://genhub.example/api/repos/alice/graph")
                .expect("should parse repository URL"),
            "cab-bucket".to_string(),
            "https://storage.googleapis.com",
            "main",
            DoltHashId([1; 20]),
        )
        .expect("should create an asset source");
        let valid_url = "https://storage.googleapis.com/cab-bucket/repos/alice/graph/.gen/assets/abc.fa.bgz?X-Goog-Signature=short";
        validate_signed_asset_url(&source.inner, valid_url, "abc.fa.bgz")
            .expect("should accept the scoped object URL");

        for invalid_url in [
            "https://storage.googleapis.com/other-bucket/repos/alice/graph/.gen/assets/abc.fa.bgz?sig=x",
            "https://storage.googleapis.com/cab-bucket/repos/alice/other/.gen/assets/abc.fa.bgz?sig=x",
            "https://storage.googleapis.com/cab-bucket/repos/alice/graph/.gen/assets/other.fa.bgz?sig=x",
            "https://storage.googleapis.com/cab-bucket/repos/alice/graph/.gen/assets/abc.fa.bgz/%2e%2e?sig=x",
            "https://storage.googleapis.com/cab-bucket/repos/alice/graph/.gen/assets/abc.fa.bgz",
            "http://storage.googleapis.com/cab-bucket/repos/alice/graph/.gen/assets/abc.fa.bgz?sig=x",
        ] {
            assert!(
                validate_signed_asset_url(&source.inner, invalid_url, "abc.fa.bgz").is_err(),
                "accepted unsafe URL: {invalid_url}"
            );
        }
    }

    #[test]
    fn local_gcs_urls_are_bound_to_the_explicit_loopback_cab_endpoint() {
        let source = BrowserAssetSource::new(
            RepositoryRemote::parse("https://genhub.example/api/repos/alice/graph")
                .expect("should parse repository URL"),
            "cab-bucket".to_string(),
            "http://127.0.0.1:4443",
            "main",
            DoltHashId([1; 20]),
        )
        .expect("should accept the explicitly configured local emulator");
        validate_signed_asset_url(
            &source.inner,
            "http://127.0.0.1:4443/cab-bucket/repos/alice/graph/.gen/assets/abc.fa.bgz?sig=local",
            "abc.fa.bgz",
        )
        .expect("should accept a URL at the exact CAB endpoint");
        assert!(
            validate_signed_asset_url(
                &source.inner,
                "http://127.0.0.1:4444/cab-bucket/repos/alice/graph/.gen/assets/abc.fa.bgz?sig=local",
                "abc.fa.bgz",
            )
            .is_err(),
            "should reject a different local storage origin"
        );
        assert!(
            BrowserAssetSource::new(
                RepositoryRemote::parse("https://genhub.example/api/repos/alice/graph")
                    .expect("should parse repository URL"),
                "cab-bucket".to_string(),
                "http://storage.example.test:4443",
                "main",
                DoltHashId([1; 20]),
            )
            .is_err(),
            "plain HTTP must stay limited to loopback development endpoints"
        );
    }

    #[test]
    fn changing_or_closing_the_source_invalidates_existing_readers() {
        let source = BrowserAssetSource::new(
            RepositoryRemote::parse("https://genhub.example/repos/alice/graph")
                .expect("should parse repository URL"),
            "cab-bucket".to_string(),
            "https://storage.googleapis.com",
            "main",
            DoltHashId([1; 20]),
        )
        .expect("should create an asset source");
        source
            .set_ref("main", &DoltHashId([2; 20]).to_string())
            .expect("should update the published source ref");
        source.close();
        assert!(
            source
                .set_ref("main", &DoltHashId([2; 20]).to_string())
                .is_err()
        );
    }

    #[test]
    fn published_base_is_available_only_for_its_open_repository_source() {
        let source = make_source("https://genhub.example/api/repos/alice/graph");
        assert_eq!(
            source.published_base_hash_for_remote("https://genhub.example/api/repos/alice/graph"),
            Some(DoltHashId([1; 20]))
        );
        assert_eq!(
            source.published_base_hash_for_remote("https://genhub.example/api/repos/alice/other"),
            None
        );
        source.close();
        assert_eq!(
            source.published_base_hash_for_remote("https://genhub.example/api/repos/alice/graph"),
            None
        );
    }

    #[test]
    fn signed_url_expiry_is_parsed_without_exposing_url_contents() {
        assert!(signed_url_near_expiry(
            "https://storage.googleapis.com/bucket/object?X-Goog-Date=20200101T000000Z&X-Goog-Expires=60"
        ));
        assert!(!signed_url_near_expiry(
            "https://storage.googleapis.com/bucket/object?X-Goog-Date=29990101T000000Z&X-Goog-Expires=60"
        ));
        assert!(signed_url_near_expiry(
            "https://storage.googleapis.com/bucket/object?X-Goog-Date=29990101T000000Z&X-Goog-Expires=9223372036854775807"
        ));
        assert!(!signed_url_near_expiry(
            "https://storage.googleapis.com/bucket/object?sig=x"
        ));
    }

    #[test]
    fn cloud_storage_scope_defaults_to_google_and_accepts_only_loopback_http() {
        assert_eq!(
            super::cloud_storage_scope("gcs://bucket/.gen/graph_db/?vfs=blockcachevfs")
                .expect("should use the production storage default"),
            (
                "bucket".to_string(),
                "https://storage.googleapis.com".to_string()
            )
        );
        assert_eq!(
            super::cloud_storage_scope(
                "gcs://bucket/.gen/graph_db/?vfs=blockcachevfs&endpoint=http%3A%2F%2F127.0.0.1%3A4443"
            )
            .expect("should accept the local fake GCS endpoint"),
            ("bucket".to_string(), "http://127.0.0.1:4443".to_string())
        );
        assert!(super::cloud_storage_scope(
            "gcs://bucket/.gen/graph_db/?vfs=blockcachevfs&endpoint=http%3A%2F%2Fstorage.example.test%3A4443"
        )
        .is_err());
    }

    #[test]
    fn source_opens_lazily_and_pins_manifest_refresh_to_the_same_published_hash() {
        let asset_id = HashId([4; 16]);
        let checksum = Sha256Hash([5; 32]);
        let uri = "file://samples/one.fa.bgz";
        let filename = <dyn AssetUri>::from_uri(uri).hashed_filename(&checksum);
        let snapshot_hash = DoltHashId([1; 20]);
        let url1 = format!(
            "https://storage.googleapis.com/cab-bucket/repos/alice/graph/.gen/assets/{filename}?X-Goog-Date=20200101T000000Z&X-Goog-Expires=3600&X-Goog-Signature=expired"
        );
        let url2 = format!(
            "https://storage.googleapis.com/cab-bucket/repos/alice/graph/.gen/assets/{filename}?X-Goog-Date=29990101T000000Z&X-Goog-Expires=3600&X-Goog-Signature=second"
        );
        let url3 = format!(
            "https://storage.googleapis.com/cab-bucket/repos/alice/graph/.gen/assets/{filename}?X-Goog-Date=29990101T000000Z&X-Goog-Expires=3600&X-Goog-Signature=third"
        );
        let responses = vec![
            transfer_response(asset_id, &url1),
            transfer_response(asset_id, &url2),
            transfer_response(asset_id, &url3),
        ];
        let (origin, server) = spawn_manifest_server(responses);
        let source = make_source(&format!("{origin}/api/repos/alice/graph"));
        let request = AssetReadRequest {
            id: asset_id,
            checksum,
            uri,
        };

        // Opening the file reader must not request either the manifest or object bytes.
        let _reader = source
            .open(&request)
            .expect("should create a lazy asset reader");
        let pin = SourceRef {
            branch: "main".to_string(),
            snapshot_hash,
            generation: 0,
        };
        let first = resolve_asset_url(&source.inner, &pin, asset_id, &filename, false)
            .expect("should fetch the published asset manifest");
        let proactive_refresh = resolve_asset_url(&source.inner, &pin, asset_id, &filename, false)
            .expect("should refresh the expired cached URL");
        let forced_refresh = resolve_asset_url(&source.inner, &pin, asset_id, &filename, true)
            .expect("should renew the signed URL at the same published ref");

        let first_url = Url::parse(&first).expect("should parse initial signed URL");
        let proactive_url = Url::parse(&proactive_refresh).expect("should parse refreshed URL");
        let forced_url = Url::parse(&forced_refresh).expect("should parse renewed URL");
        assert_eq!(first_url.path(), proactive_url.path());
        assert_eq!(first_url.path(), forced_url.path());
        assert_ne!(first_url.query(), proactive_url.query());
        assert_ne!(proactive_url.query(), forced_url.query());

        let requests = server.join().expect("should finish fake GenHub");
        assert_eq!(requests.len(), 3);
        for request in requests {
            assert_eq!(request.path, "/api/repos/alice/graph/asset-transfers");
            assert!(
                !request.authorization,
                "public pull should not attach a bearer token"
            );
            assert_eq!(request.payload["operation"], "pull");
            assert_eq!(request.payload["branch"], "main");
            assert!(request.payload["from_commit"].is_null());
            assert_eq!(request.payload["to_commit"], snapshot_hash.to_string());
        }
    }

    #[test]
    fn source_pin_change_and_close_reject_existing_readers_before_network_access() {
        let source = make_source("http://127.0.0.1:1/api/repos/alice/graph");
        let request = AssetReadRequest {
            id: HashId([7; 16]),
            checksum: Sha256Hash([8; 32]),
            uri: "file://samples/two.fa",
        };
        let mut reader = source.open(&request).expect("should create a lazy reader");
        source
            .set_ref("main", &DoltHashId([2; 20]).to_string())
            .expect("should change the published ref");
        let mut byte = [0_u8; 1];
        assert_eq!(
            reader
                .read(&mut byte)
                .expect_err("should reject a stale reader")
                .kind(),
            std::io::ErrorKind::PermissionDenied
        );

        let mut reader = source
            .open(&request)
            .expect("should create a reader for the new ref");
        source.close();
        assert_eq!(
            reader
                .read(&mut byte)
                .expect_err("should reject a closed source")
                .kind(),
            std::io::ErrorKind::PermissionDenied
        );
    }
}
