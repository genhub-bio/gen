use std::{
    io::{self, Write as _},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use rusqlite::blockcachevfs::UploadProgress;

const PROGRESS_INTERVAL: Duration = Duration::from_secs(1);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);

type ProgressSink = dyn Fn(&str) + Send + Sync;
type SharedActivity = Arc<Mutex<Instant>>;

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

/// Periodically reports that a long upload phase is still waiting for real progress.
///
/// The worker thread is always joined when this guard is dropped. If the operating system cannot
/// create a thread, progress remains best-effort and the transfer continues without heartbeats.
pub(crate) struct ProgressHeartbeat {
    stop: Option<Sender<()>>,
    worker: Option<JoinHandle<()>>,
}

impl ProgressHeartbeat {
    pub(crate) fn waiting(phase: &str) -> Self {
        let now = Instant::now();
        Self::start(
            ProgressOutput::stderr(),
            Arc::new(Mutex::new(now)),
            phase.to_string(),
            HEARTBEAT_INTERVAL,
        )
    }

    fn start(
        output: ProgressOutput,
        activity: Arc<Mutex<Instant>>,
        phase: String,
        interval: Duration,
    ) -> Self {
        let (stop, receiver) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("gen-upload-progress".to_string())
            .spawn(move || {
                let started = Instant::now();
                let mut last_heartbeat = started;
                loop {
                    let now = Instant::now();
                    let last_activity = *activity
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let idle = now.duration_since(last_activity);
                    let wait = if idle >= interval {
                        interval.saturating_sub(now.duration_since(last_heartbeat))
                    } else {
                        interval - idle
                    };
                    match receiver.recv_timeout(wait) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => return,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            let now = Instant::now();
                            let idle = now.duration_since(
                                *activity
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner),
                            );
                            if idle >= interval
                                && now.duration_since(last_heartbeat) >= interval
                            {
                                (output.sink)(&format!(
                                    "Still waiting during {phase}; no new transfer progress (elapsed {}).",
                                    format_elapsed(now.duration_since(started)),
                                ));
                                last_heartbeat = now;
                            }
                        }
                    }
                }
            })
            .ok();
        Self {
            stop: Some(stop),
            worker,
        }
    }
}

impl Drop for ProgressHeartbeat {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
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
    started: Instant,
    phase: Arc<Mutex<String>>,
    activity: SharedActivity,
    latest: Arc<Mutex<Option<UploadProgress>>>,
}

impl GraphUploadProgressReporter {
    pub(crate) fn new(attempt: usize, attempts: usize) -> Self {
        Self::with_output(ProgressOutput::stderr(), attempt, attempts)
    }

    fn with_output(output: ProgressOutput, attempt: usize, attempts: usize) -> Self {
        let now = Instant::now();
        Self {
            output: Arc::new(output.reporter()),
            attempt,
            attempts,
            started: now,
            phase: Arc::new(Mutex::new(
                "preparing direct GCS graph transfer".to_string(),
            )),
            activity: Arc::new(Mutex::new(now)),
            latest: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn report(&self, progress: UploadProgress) {
        let mut latest = self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *latest != Some(progress) {
            *self
                .activity
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Instant::now();
        }
        *latest = Some(progress);
        drop(latest);
        self.output.report(&self.format(progress), false);
    }

    pub(crate) fn heartbeat(&self, phase: &str) -> ProgressHeartbeat {
        self.heartbeat_with_interval(phase, HEARTBEAT_INTERVAL)
    }

    pub(crate) fn heartbeat_preserving_phase(&self, phase: &str) -> ProgressHeartbeat {
        self.begin_activity_window(phase, HEARTBEAT_INTERVAL)
    }

    fn heartbeat_with_interval(&self, phase: &str, interval: Duration) -> ProgressHeartbeat {
        *self
            .phase
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = phase.to_string();
        self.begin_activity_window(phase, interval)
    }

    fn begin_activity_window(&self, phase: &str, interval: Duration) -> ProgressHeartbeat {
        *self
            .activity
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Instant::now();
        ProgressHeartbeat::start(
            self.output.output.clone(),
            Arc::clone(&self.activity),
            phase.to_string(),
            interval,
        )
    }

    pub(crate) fn failed(&self) {
        let phase = self
            .phase
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let latest = *self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let summary = latest.map_or_else(
            || "No completed block counters were reported.".to_string(),
            |progress| format!("Last graph progress: {}", self.format_details(progress)),
        );
        (self.output.output.sink)(&format!(
            "Graph upload attempt {}/{} failed after {} during {phase}. {summary}",
            self.attempt,
            self.attempts,
            format_elapsed(self.started.elapsed()),
        ));
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
            "Graph upload attempt {}/{}: {}",
            self.attempt,
            self.attempts,
            self.format_details(progress),
        )
    }

    fn format_details(&self, progress: UploadProgress) -> String {
        let counters = format!(
            "{} blocks uploaded ({}); {} blocks reused ({})",
            progress.uploaded_blocks,
            format_bytes(progress.uploaded_bytes),
            progress.reused_blocks,
            format_bytes(progress.reused_bytes),
        );
        let Some(expected) = progress.expected else {
            return format!("Expected block total and byte size pending; {counters}");
        };
        let handled_blocks = progress
            .uploaded_blocks
            .saturating_add(progress.reused_blocks);
        let handled_bytes = progress
            .uploaded_bytes
            .saturating_add(progress.reused_bytes);
        format!(
            "{handled_blocks} of {} blocks handled ({}); {} of {} block-data bytes handled ({}); {counters}",
            expected.blocks,
            completion_percentage(handled_blocks, expected.blocks),
            format_bytes(handled_bytes),
            format_bytes(expected.bytes),
            completion_percentage(handled_bytes, expected.bytes),
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
    started: Instant,
    phase: Arc<Mutex<String>>,
    activity: SharedActivity,
    checksum_bytes: Arc<AtomicU64>,
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
        let now = Instant::now();
        Self {
            output: Arc::new(output.reporter()),
            index,
            total,
            name: Arc::from(name),
            total_bytes,
            started: now,
            phase: Arc::new(Mutex::new("preparing asset upload".to_string())),
            activity: Arc::new(Mutex::new(now)),
            checksum_bytes: Arc::new(AtomicU64::new(0)),
            request_body_bytes: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn checksum_started(&self) {
        self.set_phase("asset checksum scan");
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
        let previous = self.checksum_bytes.swap(bytes, Ordering::Relaxed);
        if bytes > previous {
            self.mark_activity();
        }
        self.output
            .report(&self.format_bytes("Checksum scan", bytes), false);
    }

    pub(crate) fn checksum_verified(&self) {
        self.set_phase("opening asset upload stream");
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
        self.set_phase("sending asset request body and waiting for storage acceptance");
        self.output.report(&self.format_upload(0), true);
    }

    pub(crate) fn upload_bytes(&self, bytes: u64) {
        let previous = self.request_body_bytes.swap(bytes, Ordering::Relaxed);
        if bytes > previous {
            self.mark_activity();
        }
        self.output.report(&self.format_upload(bytes), false);
    }

    pub(crate) fn upload_accepted(&self) {
        let request_body_bytes = self.request_body_bytes.load(Ordering::Relaxed);
        self.output
            .report(&self.format_upload(request_body_bytes), true);
        self.set_phase("asset upload accepted by storage");
        self.output.report(
            &format!(
                "Asset {}/{} uploaded to storage: {}",
                self.index, self.total, self.name,
            ),
            true,
        );
    }

    pub(crate) fn upload_already_present(&self) {
        self.set_phase("waiting for GenHub to verify the existing asset");
        self.output.report(
            &format!(
                "Asset {}/{} already exists (HTTP 412); awaiting server verification: {}",
                self.index, self.total, self.name,
            ),
            true,
        );
    }

    pub(crate) fn heartbeat(&self, phase: &str) -> ProgressHeartbeat {
        self.set_phase(phase);
        *self
            .activity
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Instant::now();
        ProgressHeartbeat::start(
            self.output.output.clone(),
            Arc::clone(&self.activity),
            phase.to_string(),
            HEARTBEAT_INTERVAL,
        )
    }

    pub(crate) fn failed(&self) {
        let phase = self
            .phase
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let checksum_bytes = self.checksum_bytes.load(Ordering::Relaxed);
        let request_body_bytes = self.request_body_bytes.load(Ordering::Relaxed);
        (self.output.output.sink)(&format!(
            "Asset upload {}/{} ({}) failed after {} during {phase}; checksum bytes read: {}; request body bytes consumed: {} of {}.",
            self.index,
            self.total,
            self.name,
            format_elapsed(self.started.elapsed()),
            format_bytes(checksum_bytes),
            format_bytes(request_body_bytes),
            format_bytes(self.total_bytes),
        ));
    }

    fn set_phase(&self, phase: &str) {
        *self
            .phase
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = phase.to_string();
    }

    fn mark_activity(&self) {
        *self
            .activity
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Instant::now();
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

fn completion_percentage(completed: u64, expected: u64) -> String {
    if expected == 0 {
        "not applicable (no block work planned)".to_string()
    } else {
        format!("{}%", percentage(completed, expected))
    }
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

pub(crate) fn format_elapsed(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    if seconds >= 60 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Cursor, Read as _},
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    use rusqlite::blockcachevfs::{UploadPlan, UploadProgress};

    use super::{
        AssetUploadProgressReporter, GraphUploadProgressReporter, ProgressHeartbeat,
        ProgressOutput, UploadBodyReader,
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
            ..UploadProgress::default()
        });
        progress.finish();

        let lines = lines.lock().expect("should read captured output");
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("Graph upload attempt 1/2"));
        assert!(lines[0].contains("Expected block total and byte size pending"));
        assert!(!lines[0].contains('%'));
        assert!(lines[0].contains("3 blocks uploaded (12.0 MiB)"));
        assert!(lines[0].contains("2 blocks reused (8.0 MiB)"));
    }

    #[test]
    fn test_graph_progress_reports_planned_block_and_byte_completion() {
        let (output, lines) = captured_output(Duration::ZERO);
        let progress = GraphUploadProgressReporter::with_output(output, 1, 2);
        progress.report(UploadProgress {
            uploaded_blocks: 2,
            uploaded_bytes: 2 * 1024,
            reused_blocks: 1,
            reused_bytes: 1024,
            expected: Some(UploadPlan {
                blocks: 8,
                bytes: 16 * 1024,
            }),
        });

        let lines = lines.lock().expect("should read planned progress");
        assert!(lines[0].contains("3 of 8 blocks handled (37%)"));
        assert!(lines[0].contains("3.0 KiB of 16.0 KiB block-data bytes handled (18%)"));
        assert!(lines[0].contains("2 blocks uploaded (2.0 KiB)"));
        assert!(lines[0].contains("1 blocks reused (1.0 KiB)"));
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

    #[test]
    fn test_heartbeat_reports_idle_phase_and_stops_after_guard_drop() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let (heartbeat_tx, heartbeat_rx) = std::sync::mpsc::channel();
        let captured_lines = Arc::clone(&lines);
        let output = ProgressOutput {
            sink: Arc::new(move |line| {
                captured_lines
                    .lock()
                    .expect("should capture heartbeat output")
                    .push(line.to_string());
                if line.starts_with("Still waiting") {
                    let _ = heartbeat_tx.send(line.to_string());
                }
            }),
            interval: Duration::ZERO,
        };
        let activity = Arc::new(Mutex::new(Instant::now()));
        let heartbeat = ProgressHeartbeat::start(
            output,
            activity,
            "staging graph blocks".to_string(),
            Duration::from_millis(200),
        );
        let message = heartbeat_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("should report a stalled phase");
        assert!(message.contains("staging graph blocks") && message.contains("elapsed"));
        drop(heartbeat);

        let lines_after_drop = lines.lock().expect("should read heartbeat output").len();
        assert!(lines_after_drop > 0);
        let _queued_before_join = heartbeat_rx.try_iter().count();
        assert!(matches!(
            heartbeat_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Disconnected)
        ));
        assert_eq!(
            lines.lock().expect("should verify heartbeat stopped").len(),
            lines_after_drop,
            "joined heartbeat should emit no lines after its guard is dropped"
        );
    }

    #[test]
    fn test_injected_graph_failure_keeps_error_phase_and_last_real_counters() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let (heartbeat_tx, heartbeat_rx) = std::sync::mpsc::channel();
        let captured_lines = Arc::clone(&lines);
        let output = ProgressOutput {
            sink: Arc::new(move |line| {
                captured_lines
                    .lock()
                    .expect("should capture failure output")
                    .push(line.to_string());
                if line.starts_with("Still waiting") {
                    let _ = heartbeat_tx.send(line.to_string());
                }
            }),
            interval: Duration::ZERO,
        };
        let progress = GraphUploadProgressReporter::with_output(output, 1, 2);
        progress.report(UploadProgress {
            uploaded_blocks: 2,
            uploaded_bytes: 8 * 1024 * 1024,
            reused_blocks: 1,
            reused_bytes: 4 * 1024 * 1024,
            expected: Some(UploadPlan {
                blocks: 4,
                bytes: 24 * 1024 * 1024,
            }),
        });
        let result = (|| -> Result<(), &str> {
            let _heartbeat = progress.heartbeat_with_interval(
                "Dolt push through loopback RemoteServer",
                Duration::from_millis(200),
            );
            heartbeat_rx
                .recv_timeout(Duration::from_secs(2))
                .map_err(|_| "heartbeat did not fire")?;
            Err("injected disk I/O error")
        })();
        let cleanup_heartbeat =
            progress.heartbeat_preserving_phase("closing the direct GCS graph session");
        drop(cleanup_heartbeat);
        progress.failed();

        assert_eq!(result, Err("injected disk I/O error"));
        let _queued_before_join = heartbeat_rx.try_iter().count();
        assert!(matches!(
            heartbeat_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        let lines = lines.lock().expect("should read failure output");
        assert!(lines.iter().any(|line| {
            line.contains("Graph upload attempt 1/2 failed")
                && line.contains("during Dolt push through loopback RemoteServer")
                && line.contains("3 of 4 blocks handled (75%)")
                && line.contains("12.0 MiB of 24.0 MiB block-data bytes handled (50%)")
                && line.contains("2 blocks uploaded (8.0 MiB)")
                && line.contains("1 blocks reused (4.0 MiB)")
        }));
        assert!(lines.iter().any(|line| {
            line.contains("Still waiting during Dolt push through loopback RemoteServer")
        }));
        assert!(
            lines
                .iter()
                .all(|line| { !line.contains("access_token") && !line.contains("signed_url") })
        );
    }

    #[test]
    fn test_graph_failure_without_callbacks_does_not_report_fake_zero_counts() {
        let (output, lines) = captured_output(Duration::ZERO);
        let progress = GraphUploadProgressReporter::with_output(output, 1, 2);
        progress.failed();

        let lines = lines
            .lock()
            .expect("should read no-progress failure output");
        assert!(lines[0].contains("No completed block counters were reported"));
        assert!(!lines[0].contains("0 blocks uploaded"));
        assert!(!lines[0].contains("0 blocks reused"));
    }

    #[test]
    fn test_asset_failure_reports_only_bytes_consumed_by_request() {
        let (output, lines) = captured_output(Duration::ZERO);
        let reporter =
            AssetUploadProgressReporter::with_output(output, 1, 1, "simple.fa".into(), 10);
        reporter.upload_started();
        reporter.upload_bytes(4);
        reporter.failed();

        let lines = lines.lock().expect("should read asset failure output");
        assert!(lines.iter().any(|line| {
            line.contains("request body bytes consumed: 4 B of 10 B")
                && line.contains("during sending asset request body")
        }));
    }
}
