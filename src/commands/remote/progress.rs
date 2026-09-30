use std::{
    io::{self, Write as _},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use rusqlite::blockcachevfs::UploadProgress;

const PROGRESS_INTERVAL: Duration = Duration::from_secs(1);

type ProgressSink = dyn Fn(&str) + Send + Sync;

#[derive(Clone)]
struct ProgressOutput {
    sink: Arc<ProgressSink>,
    interval: Duration,
}

impl ProgressOutput {
    fn stderr() -> Self {
        Self {
            sink: Arc::new(|line| {
                let _ = writeln!(io::stderr().lock(), "{line}");
            }),
            interval: PROGRESS_INTERVAL,
        }
    }

    fn reporter(&self) -> ThrottledOutput {
        ThrottledOutput {
            output: self.clone(),
            state: Mutex::new(ThrottleState {
                last_output: None,
                last_message: None,
            }),
        }
    }
}

struct ThrottleState {
    last_output: Option<Instant>,
    last_message: Option<String>,
}

struct ThrottledOutput {
    output: ProgressOutput,
    state: Mutex<ThrottleState>,
}

impl ThrottledOutput {
    fn report(&self, message: &str, force: bool) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Instant::now();
        let already_reported = state.last_message.as_deref() == Some(message);
        let interval_elapsed = state
            .last_output
            .is_none_or(|last_output| now.duration_since(last_output) >= self.output.interval);
        if (force && !already_reported) || (!force && interval_elapsed) {
            (self.output.sink)(message);
            state.last_output = Some(now);
            state.last_message = Some(message.to_string());
        }
    }
}

#[derive(Clone)]
pub(crate) struct GraphUploadProgressReporter {
    output: Arc<ThrottledOutput>,
    attempt: usize,
    attempts: usize,
    latest: Arc<Mutex<Option<UploadProgress>>>,
}

impl GraphUploadProgressReporter {
    pub(crate) fn new(attempt: usize, attempts: usize) -> Self {
        Self::with_output(ProgressOutput::stderr(), attempt, attempts)
    }

    fn with_output(output: ProgressOutput, attempt: usize, attempts: usize) -> Self {
        Self {
            output: Arc::new(output.reporter()),
            attempt,
            attempts,
            latest: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn report(&self, progress: UploadProgress) {
        *self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(progress);
        self.output.report(&self.format(progress), false);
    }

    pub(crate) fn finish(&self) {
        let latest = *self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(latest) = latest {
            self.output.report(&self.format(latest), true);
        }
    }

    fn format(&self, progress: UploadProgress) -> String {
        format!(
            "Graph upload attempt {}/{}: {} blocks uploaded ({}); {} blocks reused ({})",
            self.attempt,
            self.attempts,
            progress.uploaded_blocks,
            format_bytes(progress.uploaded_bytes),
            progress.reused_blocks,
            format_bytes(progress.reused_bytes),
        )
    }
}

#[derive(Clone)]
pub(crate) struct AssetUploadProgressReporter {
    output: Arc<ThrottledOutput>,
    index: usize,
    total: usize,
    name: Arc<str>,
    total_bytes: u64,
    request_body_bytes: Arc<AtomicU64>,
}

impl AssetUploadProgressReporter {
    pub(crate) fn new(index: usize, total: usize, name: String, total_bytes: u64) -> Self {
        Self::with_output(ProgressOutput::stderr(), index, total, name, total_bytes)
    }

    fn with_output(
        output: ProgressOutput,
        index: usize,
        total: usize,
        name: String,
        total_bytes: u64,
    ) -> Self {
        Self {
            output: Arc::new(output.reporter()),
            index,
            total,
            name: Arc::from(name),
            total_bytes,
            request_body_bytes: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn checksum_started(&self) {
        self.output.report(
            &format!(
                "Checking asset {}/{}: {} ({})",
                self.index,
                self.total,
                self.name,
                format_bytes(self.total_bytes),
            ),
            true,
        );
    }

    pub(crate) fn checksum_bytes(&self, bytes: u64) {
        self.output
            .report(&self.format_bytes("Checksum scan", bytes), false);
    }

    pub(crate) fn checksum_verified(&self) {
        self.output.report(
            &format!(
                "Checksum verified for asset {}/{}: {}; starting upload",
                self.index, self.total, self.name,
            ),
            true,
        );
    }

    pub(crate) fn upload_started(&self) {
        self.request_body_bytes.store(0, Ordering::Relaxed);
        self.output.report(&self.format_upload(0), true);
    }

    pub(crate) fn upload_bytes(&self, bytes: u64) {
        self.request_body_bytes.store(bytes, Ordering::Relaxed);
        self.output.report(&self.format_upload(bytes), false);
    }

    pub(crate) fn upload_accepted(&self) {
        let request_body_bytes = self.request_body_bytes.load(Ordering::Relaxed);
        self.output
            .report(&self.format_upload(request_body_bytes), true);
        self.output.report(
            &format!(
                "Asset {}/{} uploaded to storage: {}",
                self.index, self.total, self.name,
            ),
            true,
        );
    }

    pub(crate) fn upload_already_present(&self) {
        self.output.report(
            &format!(
                "Asset {}/{} already exists (HTTP 412); awaiting server verification: {}",
                self.index, self.total, self.name,
            ),
            true,
        );
    }

    fn format_bytes(&self, phase: &str, bytes: u64) -> String {
        format!(
            "{phase} asset {}/{}: {} — {} of {} ({}%)",
            self.index,
            self.total,
            self.name,
            format_bytes(bytes),
            format_bytes(self.total_bytes),
            percentage(bytes, self.total_bytes),
        )
    }

    fn format_upload(&self, bytes: u64) -> String {
        format!(
            "Uploading asset {}/{}: {} — request body read {} of {} ({}%)",
            self.index,
            self.total,
            self.name,
            format_bytes(bytes),
            format_bytes(self.total_bytes),
            percentage(bytes, self.total_bytes),
        )
    }
}

pub(crate) struct UploadBodyReader<R> {
    reader: R,
    progress: AssetUploadProgressReporter,
    bytes_read: u64,
}

impl<R> UploadBodyReader<R> {
    pub(crate) fn new(reader: R, progress: AssetUploadProgressReporter) -> Self {
        Self {
            reader,
            progress,
            bytes_read: 0,
        }
    }
}

impl<R: io::Read> io::Read for UploadBodyReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let bytes_read = self.reader.read(buffer)?;
        if bytes_read > 0 {
            self.bytes_read += bytes_read as u64;
            self.progress.upload_bytes(self.bytes_read);
        }
        Ok(bytes_read)
    }
}

pub(crate) fn write_progress_line(message: &str) {
    let _ = writeln!(io::stderr().lock(), "{message}");
}

fn percentage(bytes: u64, total: u64) -> u64 {
    if total == 0 {
        return 100;
    }
    ((bytes.min(total) as u128 * 100) / total as u128) as u64
}

fn format_bytes(bytes: u64) -> String {
    const KIBIBYTE: u64 = 1024;
    const MEBIBYTE: u64 = KIBIBYTE * 1024;
    const GIBIBYTE: u64 = MEBIBYTE * 1024;
    if bytes >= GIBIBYTE {
        format!("{:.1} GiB", bytes as f64 / GIBIBYTE as f64)
    } else if bytes >= MEBIBYTE {
        format!("{:.1} MiB", bytes as f64 / MEBIBYTE as f64)
    } else if bytes >= KIBIBYTE {
        format!("{:.1} KiB", bytes as f64 / KIBIBYTE as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Cursor, Read as _},
        sync::{Arc, Mutex},
        time::Duration,
    };

    use rusqlite::blockcachevfs::UploadProgress;

    use super::{
        AssetUploadProgressReporter, GraphUploadProgressReporter, ProgressOutput, UploadBodyReader,
    };

    fn captured_output(interval: Duration) -> (ProgressOutput, Arc<Mutex<Vec<String>>>) {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let captured_lines = Arc::clone(&lines);
        let output = ProgressOutput {
            sink: Arc::new(move |line| {
                captured_lines
                    .lock()
                    .expect("should capture progress output")
                    .push(line.to_string());
            }),
            interval,
        };
        (output, lines)
    }

    #[test]
    fn test_graph_progress_reports_native_uploaded_and_reused_counts() {
        let (output, lines) = captured_output(Duration::ZERO);
        let progress = GraphUploadProgressReporter::with_output(output, 1, 2);
        progress.report(UploadProgress {
            uploaded_blocks: 3,
            uploaded_bytes: 12 * 1024 * 1024,
            reused_blocks: 2,
            reused_bytes: 8 * 1024 * 1024,
        });
        progress.finish();

        let lines = lines.lock().expect("should read captured output");
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("Graph upload attempt 1/2"));
        assert!(lines[0].contains("3 blocks uploaded (12.0 MiB)"));
        assert!(lines[0].contains("2 blocks reused (8.0 MiB)"));
    }

    #[test]
    fn test_upload_body_reader_reports_bytes_consumed_by_request() {
        let (output, lines) = captured_output(Duration::from_secs(60));
        let reporter =
            AssetUploadProgressReporter::with_output(output, 1, 1, "simple.fa".into(), 10);
        reporter.upload_started();
        let mut body = UploadBodyReader::new(Cursor::new(b"0123456789"), reporter);
        let mut buffer = [0; 4];

        assert_eq!(body.read(&mut buffer).expect("should read first chunk"), 4);
        assert_eq!(body.read(&mut buffer).expect("should read second chunk"), 4);
        assert_eq!(body.read(&mut buffer).expect("should read final chunk"), 2);
        assert_eq!(body.read(&mut buffer).expect("should reach end"), 0);

        let lines_before_acceptance = lines.lock().expect("should read captured output");
        assert_eq!(lines_before_acceptance.len(), 1);
        assert!(lines_before_acceptance[0].contains("0 B of 10 B (0%)"));
        assert!(
            !lines_before_acceptance
                .iter()
                .any(|line| line.contains("uploaded to storage"))
        );
        drop(lines_before_acceptance);

        body.progress.upload_accepted();
        let lines_after_acceptance = lines.lock().expect("should read accepted output");
        assert!(lines_after_acceptance.iter().any(|line| {
            line.contains("Uploading asset 1/1: simple.fa") && line.contains("10 B of 10 B (100%)")
        }));
        assert!(
            lines_after_acceptance
                .iter()
                .any(|line| line.contains("uploaded to storage"))
        );
    }

    #[test]
    fn test_existing_asset_waits_for_server_verification() {
        let (output, lines) = captured_output(Duration::ZERO);
        let reporter =
            AssetUploadProgressReporter::with_output(output, 1, 1, "simple.fa".into(), 512 * 1024);
        reporter.upload_already_present();

        let lines = lines.lock().expect("should read captured output");
        assert!(lines[0].contains("already exists (HTTP 412)"));
        assert!(lines[0].contains("awaiting server verification"));
        assert!(!lines[0].contains("verified"));
    }
}
