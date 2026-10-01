use std::{
    io::{self, IsTerminal as _, Write as _},
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use indicatif::{MultiProgress, ProgressBar};
use rusqlite::{DoltPushProgressEvent, blockcachevfs::UploadProgress};

use crate::progress_bar::{get_handler, get_message_bar, get_progress_bar, get_time_elapsed_bar};

const PROGRESS_INTERVAL: Duration = Duration::from_secs(1);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);

type ProgressSink = dyn Fn(&str) + Send + Sync;
type SharedActivity = Arc<Mutex<Instant>>;
static ACTIVE_INDICATIF: OnceLock<Mutex<Weak<IndicatifOutput>>> = OnceLock::new();

#[derive(Clone)]
struct ProgressOutput {
    sink: Arc<ProgressSink>,
    interval: Duration,
    indicatif: Option<Arc<IndicatifOutput>>,
}

impl ProgressOutput {
    fn stderr() -> Self {
        Self::stderr_with_active_handler(false)
    }

    fn active_or_stderr() -> Self {
        Self::stderr_with_active_handler(true)
    }

    fn stderr_with_active_handler(use_active: bool) -> Self {
        let active = use_active.then(|| {
            ACTIVE_INDICATIF.get().and_then(|active| {
                active
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .upgrade()
            })
        });
        let indicatif =
            Self::select_indicatif(active.flatten(), io::stderr().is_terminal(), || {
                Arc::new(IndicatifOutput::new())
            });
        Self {
            sink: Arc::new(|line| {
                let _ = writeln!(io::stderr().lock(), "{line}");
            }),
            interval: PROGRESS_INTERVAL,
            indicatif,
        }
    }

    fn select_indicatif(
        active: Option<Arc<IndicatifOutput>>,
        stderr_is_terminal: bool,
        make_terminal: impl FnOnce() -> Arc<IndicatifOutput>,
    ) -> Option<Arc<IndicatifOutput>> {
        active
            .filter(|indicatif| !indicatif.handler.is_hidden())
            .or_else(|| {
                stderr_is_terminal
                    .then(make_terminal)
                    .filter(|indicatif| !indicatif.handler.is_hidden())
            })
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

    fn println(&self, line: &str) {
        if let Some(indicatif) = &self.indicatif {
            let _ = indicatif.handler.println(line);
        } else {
            (self.sink)(line);
        }
    }

    fn register_active(&self) {
        let Some(indicatif) = &self.indicatif else {
            return;
        };
        if indicatif.handler.is_hidden() {
            return;
        }
        let active = ACTIVE_INDICATIF.get_or_init(|| Mutex::new(Weak::new()));
        *active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::downgrade(indicatif);
    }

    fn start_phase(&self, phase: &str) -> Option<ProgressBar> {
        self.indicatif
            .as_ref()
            .map(|indicatif| indicatif.start_phase(phase))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BarKind {
    Spinner,
    Determinate(u64),
    Message,
}

#[derive(Default)]
struct BarState {
    kind: Option<BarKind>,
    bar: Option<ProgressBar>,
}

struct IndicatifOutput {
    handler: MultiProgress,
    bar: Mutex<BarState>,
    details: Mutex<Vec<ProgressBar>>,
}

impl IndicatifOutput {
    fn new() -> Self {
        Self::with_handler(get_handler())
    }

    fn with_handler(handler: MultiProgress) -> Self {
        Self {
            handler,
            bar: Mutex::new(BarState::default()),
            details: Mutex::new(Vec::new()),
        }
    }

    fn start_phase(&self, phase: &str) -> ProgressBar {
        let bar = self.handler.add(get_time_elapsed_bar());
        bar.set_message(phase.to_string());
        bar
    }

    fn set_bar(&self, kind: BarKind, position: u64, message: &str) {
        let mut state = self
            .bar
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.kind != Some(kind) {
            if let Some(bar) = state.bar.take() {
                bar.finish_and_clear();
                self.handler.remove(&bar);
            }
            let bar = match kind {
                BarKind::Spinner => get_progress_bar(None),
                BarKind::Determinate(length) => get_progress_bar(Some(length)),
                BarKind::Message => get_message_bar(),
            };
            state.bar = Some(self.handler.add(bar));
            state.kind = Some(kind);
        }
        let Some(bar) = state.bar.as_ref() else {
            return;
        };
        if let BarKind::Determinate(length) = kind {
            bar.set_position(position.min(length));
        } else if kind == BarKind::Spinner {
            bar.set_position(position);
        }
        bar.set_message(message.to_string());
    }

    fn graph_progress(&self, logical: Option<LogicalChunkProgress>, gcs: Option<UploadProgress>) {
        let mut details = Vec::new();
        if let Some(progress) = logical {
            details.push(format!(
                "Dolt payload bytes: {} of {} ({} of {})",
                progress.acknowledged_bytes,
                progress.planned_bytes,
                format_bytes(progress.acknowledged_bytes),
                format_bytes(progress.planned_bytes),
            ));
        } else {
            details.push("Dolt transfer plan: waiting for destination counts".to_string());
        }
        if let Some(progress) = gcs {
            let handled_blocks = progress
                .uploaded_blocks
                .saturating_add(progress.reused_blocks);
            let handled_bytes = progress
                .uploaded_bytes
                .saturating_add(progress.reused_bytes);
            match progress.expected {
                Some(expected) => {
                    details.push(format!(
                        "GCS blocks: {handled_blocks} of {}; {} uploaded, {} reused",
                        expected.blocks, progress.uploaded_blocks, progress.reused_blocks,
                    ));
                    details.push(format!(
                        "GCS block-data bytes: {handled_bytes} of {} ({}, {})",
                        expected.bytes,
                        format_bytes(handled_bytes),
                        format_bytes(expected.bytes),
                    ));
                }
                None => {
                    details.push(format!(
                        "GCS blocks handled: {handled_blocks}; {} uploaded, {} reused; total pending",
                        progress.uploaded_blocks,
                        progress.reused_blocks,
                    ));
                    details.push(format!(
                        "GCS block-data bytes handled: {handled_bytes} ({}); total pending",
                        format_bytes(handled_bytes),
                    ));
                }
            }
        } else if logical.is_some() {
            details.push("GCS block work: waiting for storage counters".to_string());
        }
        self.set_details(&details);
        match logical {
            Some(progress) if progress.planned_chunks > 0 => self.set_bar(
                BarKind::Determinate(progress.planned_chunks),
                progress.acknowledged_chunks,
                "Dolt logical chunks acknowledged",
            ),
            Some(progress) => self.set_bar(
                BarKind::Message,
                0,
                &format!(
                    "Dolt push has no destination-missing chunks ({} payload bytes)",
                    progress.planned_bytes,
                ),
            ),
            None => self.set_bar(BarKind::Spinner, 0, "Preparing Dolt chunk transfer"),
        }
    }

    fn set_details(&self, messages: &[String]) {
        let mut bars = self
            .details
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while bars.len() < messages.len() {
            bars.push(self.handler.add(get_message_bar()));
        }
        while bars.len() > messages.len() {
            if let Some(bar) = bars.pop() {
                self.handler.remove(&bar);
            }
        }
        for (bar, message) in bars.iter().zip(messages) {
            bar.set_message(message.clone());
        }
    }

    fn asset_progress(&self, total_bytes: u64, bytes: u64, message: &str) {
        if total_bytes == 0 {
            self.set_bar(BarKind::Message, 0, message);
        } else {
            self.set_bar(BarKind::Determinate(total_bytes), bytes, message);
        }
    }

    fn asset_waiting(&self, message: &str) {
        self.set_bar(BarKind::Spinner, 0, message);
    }

    fn finish_bar(&self, message: &str, accepted: bool) {
        let mut state = self
            .bar
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(bar) = state.bar.take() {
            if accepted {
                bar.finish_with_message(message.to_string());
            } else {
                bar.abandon_with_message(message.to_string());
            }
        }
        state.kind = None;
        let mut details = self
            .details
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for bar in details.drain(..) {
            bar.finish_and_clear();
            self.handler.remove(&bar);
        }
    }

    fn clear_graph_progress(&self) {
        let mut state = self
            .bar
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(bar) = state.bar.take() {
            bar.finish_and_clear();
            self.handler.remove(&bar);
        }
        state.kind = None;
        drop(state);
        self.set_details(&[]);
    }

    fn println(&self, message: &str) {
        let _ = self.handler.println(message);
    }
}

/// Periodically reports that a long upload phase is still waiting for real progress.
///
/// The worker thread is always joined when this guard is dropped. If the operating system cannot
/// create a thread, progress remains best-effort and the transfer continues without heartbeats.
pub(crate) struct ProgressHeartbeat {
    stop: Option<Sender<()>>,
    worker: Option<JoinHandle<()>>,
    bar: Option<ProgressBar>,
}

impl ProgressHeartbeat {
    pub(crate) fn waiting(phase: &str) -> Self {
        let now = Instant::now();
        Self::start(
            ProgressOutput::active_or_stderr(),
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
        if let Some(bar) = output.start_phase(&phase) {
            return Self {
                stop: None,
                worker: None,
                bar: Some(bar),
            };
        }
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
            bar: None,
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
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
    }
}

struct ThrottleState {
    last_output: Option<Instant>,
    last_message: Option<String>,
}

#[derive(Clone, Copy)]
struct LogicalChunkProgress {
    plan_number: u64,
    planned_chunks: u64,
    planned_bytes: u64,
    acknowledged_chunks: u64,
    acknowledged_bytes: u64,
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
            self.output.println(message);
            state.last_output = Some(now);
            state.last_message = Some(message.to_string());
        }
    }

    fn println(&self, message: &str) {
        self.output.println(message);
    }
}

#[derive(Clone)]
pub(crate) struct GraphUploadProgressReporter {
    output: Arc<ThrottledOutput>,
    logical_output: Arc<ThrottledOutput>,
    attempt: usize,
    attempts: usize,
    started: Instant,
    phase: Arc<Mutex<String>>,
    activity: SharedActivity,
    latest: Arc<Mutex<Option<UploadProgress>>>,
    logical_chunks: Arc<Mutex<Option<LogicalChunkProgress>>>,
}

impl GraphUploadProgressReporter {
    pub(crate) fn new(attempt: usize, attempts: usize) -> Self {
        let output = ProgressOutput::stderr();
        output.register_active();
        Self::with_output(output, attempt, attempts)
    }

    fn with_output(output: ProgressOutput, attempt: usize, attempts: usize) -> Self {
        let now = Instant::now();
        Self {
            output: Arc::new(output.reporter()),
            logical_output: Arc::new(output.reporter()),
            attempt,
            attempts,
            started: now,
            phase: Arc::new(Mutex::new(
                "preparing direct GCS graph transfer".to_string(),
            )),
            activity: Arc::new(Mutex::new(now)),
            latest: Arc::new(Mutex::new(None)),
            logical_chunks: Arc::new(Mutex::new(None)),
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
        if let Some(indicatif) = &self.output.output.indicatif {
            let logical = *self
                .logical_chunks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            indicatif.graph_progress(logical, Some(progress));
        } else {
            self.output.report(&self.format(progress), false);
        }
    }

    pub(crate) fn report_dolt_progress(&self, event: DoltPushProgressEvent) {
        let is_plan = matches!(&event, DoltPushProgressEvent::Plan { .. });
        let now = Instant::now();
        let (message, force) = match event {
            DoltPushProgressEvent::Plan {
                missing_chunk_count,
                missing_payload_bytes,
            } => {
                let (plan_number, previous) = {
                    let mut logical_chunks = self
                        .logical_chunks
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let plan_number = (*logical_chunks)
                        .map_or(1, |progress| progress.plan_number.saturating_add(1));
                    let previous = *logical_chunks;
                    *logical_chunks = Some(LogicalChunkProgress {
                        plan_number,
                        planned_chunks: missing_chunk_count,
                        planned_bytes: missing_payload_bytes,
                        acknowledged_chunks: 0,
                        acknowledged_bytes: 0,
                    });
                    (plan_number, previous)
                };
                *self
                    .activity
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = now;
                if let Some(previous) = previous {
                    self.logical_output.report(
                        &format!(
                            "Dolt logical chunk plan {} was replanned after: {}",
                            previous.plan_number,
                            format_logical_chunk_progress(previous),
                        ),
                        true,
                    );
                }
                (
                    format!(
                        "Dolt logical chunk transfer plan {}: {} destination-missing chunks ({} logical payload bytes); {} bytes.",
                        plan_number,
                        missing_chunk_count,
                        format_bytes(missing_payload_bytes),
                        missing_payload_bytes,
                    ),
                    true,
                )
            }
            DoltPushProgressEvent::Uploaded {
                acknowledged_chunk_count,
                acknowledged_payload_bytes,
            } => {
                let message = {
                    let mut logical_chunks = self
                        .logical_chunks
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let Some(progress) = logical_chunks.as_mut() else {
                        return;
                    };
                    if progress.acknowledged_chunks != acknowledged_chunk_count
                        || progress.acknowledged_bytes != acknowledged_payload_bytes
                    {
                        *self
                            .activity
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = now;
                    }
                    progress.acknowledged_chunks = acknowledged_chunk_count;
                    progress.acknowledged_bytes = acknowledged_payload_bytes;
                    format!(
                        "Dolt logical chunks acknowledged for plan {}: {}",
                        progress.plan_number,
                        format_logical_chunk_progress(*progress),
                    )
                };
                (message, false)
            }
        };
        if let Some(indicatif) = &self.logical_output.output.indicatif {
            if is_plan {
                if force {
                    self.logical_output.println(&message);
                }
                let logical = *self
                    .logical_chunks
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let gcs = *self
                    .latest
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                indicatif.graph_progress(logical, gcs);
            } else {
                let logical = *self
                    .logical_chunks
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let gcs = *self
                    .latest
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                indicatif.graph_progress(logical, gcs);
            }
        } else {
            self.logical_output.report(&message, force);
        }
    }

    pub(crate) fn finish_dolt_progress(&self, push_succeeded: bool) {
        let Some(progress) = *self
            .logical_chunks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        else {
            return;
        };
        let outcome = if push_succeeded {
            "after Dolt push returned successfully"
        } else {
            "after Dolt push returned an error"
        };
        let summary = format!(
            "Final Dolt logical chunk transfer counters for plan {} {outcome}: {}",
            progress.plan_number,
            format_logical_chunk_progress(progress),
        );
        if let Some(indicatif) = &self.logical_output.output.indicatif {
            let gcs = *self
                .latest
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            indicatif.graph_progress(Some(progress), gcs);
            self.logical_output.println(&summary);
        } else {
            self.logical_output.report(&summary, true);
        }
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

    pub(crate) fn waiting_for_local_server_cleanup(&self) {
        self.output.println("Waiting for local server cleanup...");
    }

    pub(crate) fn failed_silently(&self) {
        if let Some(indicatif) = &self.output.output.indicatif {
            indicatif.clear_graph_progress();
        }
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
        let summary = match latest {
            Some(progress) => format!("Last graph progress: {}", self.format_details(progress)),
            None => {
                let logical_chunks = *self
                    .logical_chunks
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let mut summary = "No completed GCS block counters were reported.".to_string();
                if let Some(logical_chunks) = logical_chunks {
                    summary.push_str(&format!(
                        " Last Dolt logical chunk progress: {}.",
                        format_logical_chunk_progress(logical_chunks),
                    ));
                }
                summary
            }
        };
        self.output.println(&format!(
            "Graph upload attempt {}/{} failed after {} during {phase}. {summary}",
            self.attempt,
            self.attempts,
            format_elapsed(self.started.elapsed()),
        ));
        if let Some(indicatif) = &self.output.output.indicatif {
            indicatif.finish_bar("Graph upload failed", false);
        }
    }

    pub(crate) fn finish(&self) {
        let latest = *self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(latest) = latest {
            if let Some(indicatif) = &self.output.output.indicatif {
                let logical = *self
                    .logical_chunks
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                indicatif.graph_progress(logical, Some(latest));
                indicatif.finish_bar("Graph block counters captured", false);
                self.output.println(&self.format(latest));
            } else {
                self.output.report(&self.format(latest), true);
            }
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
            "{} GCS blocks uploaded ({}); {} GCS blocks reused ({})",
            progress.uploaded_blocks,
            format_bytes(progress.uploaded_bytes),
            progress.reused_blocks,
            format_bytes(progress.reused_bytes),
        );
        let logical_chunks = *self
            .logical_chunks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let block_progress = match progress.expected {
            None if logical_chunks.is_some() => format!("GCS block counters: {counters}"),
            None => format!("Expected GCS block total and byte size pending; {counters}"),
            Some(expected) => {
                let handled_blocks = progress
                    .uploaded_blocks
                    .saturating_add(progress.reused_blocks);
                let handled_bytes = progress
                    .uploaded_bytes
                    .saturating_add(progress.reused_bytes);
                format!(
                    "GCS blocks handled: {handled_blocks} of {} ({}); {} of {} GCS block-data bytes handled ({}); {counters}",
                    expected.blocks,
                    completion_percentage(handled_blocks, expected.blocks),
                    format_bytes(handled_bytes),
                    format_bytes(expected.bytes),
                    completion_percentage(handled_bytes, expected.bytes),
                )
            }
        };
        let Some(logical_chunks) = logical_chunks else {
            return block_progress;
        };
        format!(
            "Dolt logical chunk progress: {}; {block_progress}",
            format_logical_chunk_progress(logical_chunks),
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
        let output = ProgressOutput::stderr();
        output.register_active();
        Self::with_output(output, index, total, name, total_bytes)
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
        let message = format!(
            "Checking asset {}/{}: {}",
            self.index, self.total, self.name,
        );
        if let Some(indicatif) = &self.output.output.indicatif {
            indicatif.asset_progress(self.total_bytes, 0, &message);
        } else {
            self.output.report(
                &format!("{message} ({})", format_bytes(self.total_bytes)),
                true,
            );
        }
    }

    pub(crate) fn checksum_bytes(&self, bytes: u64) {
        let previous = self.checksum_bytes.swap(bytes, Ordering::Relaxed);
        if bytes > previous {
            self.mark_activity();
        }
        if let Some(indicatif) = &self.output.output.indicatif {
            indicatif.asset_progress(
                self.total_bytes,
                bytes,
                &self.format_bytes("Checksum scan", bytes),
            );
        } else {
            self.output
                .report(&self.format_bytes("Checksum scan", bytes), false);
        }
    }

    pub(crate) fn checksum_verified(&self) {
        self.set_phase("opening asset upload stream");
        let message = format!(
            "Checksum verified for asset {}/{}: {}",
            self.index, self.total, self.name,
        );
        if let Some(indicatif) = &self.output.output.indicatif {
            indicatif.finish_bar(&message, true);
        } else {
            self.output
                .report(&format!("{message}; starting upload"), true);
        }
    }

    pub(crate) fn upload_started(&self) {
        self.request_body_bytes.store(0, Ordering::Relaxed);
        self.set_phase("sending asset request body and waiting for storage acceptance");
        if let Some(indicatif) = &self.output.output.indicatif {
            indicatif.asset_progress(
                self.total_bytes,
                0,
                &format!(
                    "Uploading asset {}/{}: {}",
                    self.index, self.total, self.name
                ),
            );
        } else {
            self.output.report(&self.format_upload(0), true);
        }
    }

    pub(crate) fn upload_bytes(&self, bytes: u64) {
        let previous = self.request_body_bytes.swap(bytes, Ordering::Relaxed);
        if bytes > previous {
            self.mark_activity();
        }
        if let Some(indicatif) = &self.output.output.indicatif {
            indicatif.asset_progress(self.total_bytes, bytes, &self.format_upload(bytes));
        } else {
            self.output.report(&self.format_upload(bytes), false);
        }
    }

    pub(crate) fn upload_accepted(&self) {
        let request_body_bytes = self.request_body_bytes.load(Ordering::Relaxed);
        self.set_phase("asset upload accepted by storage");
        let message = format!(
            "Asset {}/{} uploaded to storage: {}",
            self.index, self.total, self.name,
        );
        if let Some(indicatif) = &self.output.output.indicatif {
            indicatif.finish_bar(&message, true);
        } else {
            self.output
                .report(&self.format_upload(request_body_bytes), true);
            self.output.report(&message, true);
        }
    }

    pub(crate) fn upload_already_present(&self) {
        self.set_phase("waiting for GenHub to verify the existing asset");
        let message = format!(
            "Asset {}/{} already exists (HTTP 412); awaiting server verification: {}",
            self.index, self.total, self.name,
        );
        if let Some(indicatif) = &self.output.output.indicatif {
            indicatif.asset_waiting(&message);
            indicatif.finish_bar(&message, false);
        } else {
            self.output.report(&message, true);
        }
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
        if let Some(indicatif) = &self.output.output.indicatif {
            indicatif.finish_bar(&format!("Asset upload failed: {}", self.name), false);
        }
        self.output.println(&format!(
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
    if let Some(indicatif) = ACTIVE_INDICATIF.get().and_then(|active| {
        active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .upgrade()
            .filter(|indicatif| !indicatif.handler.is_hidden())
    }) {
        indicatif.println(message);
    } else {
        let _ = writeln!(io::stderr().lock(), "{message}");
    }
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

fn format_logical_chunk_progress(progress: LogicalChunkProgress) -> String {
    format!(
        "{} of {} destination-missing chunks acknowledged ({}); {} of {} logical payload bytes acknowledged ({})",
        progress.acknowledged_chunks,
        progress.planned_chunks,
        logical_percentage(progress.acknowledged_chunks, progress.planned_chunks),
        format_bytes(progress.acknowledged_bytes),
        format_bytes(progress.planned_bytes),
        logical_percentage(progress.acknowledged_bytes, progress.planned_bytes),
    )
}

fn logical_percentage(completed: u64, expected: u64) -> String {
    if expected == 0 {
        "not applicable (no logical chunk work planned)".to_string()
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
        sync::{Arc, Mutex, Weak},
        time::{Duration, Instant},
    };

    use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, TermLike};
    use rusqlite::{
        DoltPushProgressEvent,
        blockcachevfs::{UploadPlan, UploadProgress},
    };

    use super::{
        ACTIVE_INDICATIF, AssetUploadProgressReporter, BarKind, GraphUploadProgressReporter,
        IndicatifOutput, LogicalChunkProgress, ProgressHeartbeat, ProgressOutput, UploadBodyReader,
    };

    #[derive(Debug)]
    struct RecordingTerm {
        output: Arc<Mutex<Vec<u8>>>,
    }

    impl TermLike for RecordingTerm {
        fn width(&self) -> u16 {
            120
        }

        fn move_cursor_up(&self, _n: usize) -> std::io::Result<()> {
            Ok(())
        }

        fn move_cursor_down(&self, _n: usize) -> std::io::Result<()> {
            Ok(())
        }

        fn move_cursor_right(&self, _n: usize) -> std::io::Result<()> {
            Ok(())
        }

        fn move_cursor_left(&self, _n: usize) -> std::io::Result<()> {
            Ok(())
        }

        fn write_line(&self, line: &str) -> std::io::Result<()> {
            let mut output = self.output.lock().expect("should lock recording terminal");
            output.extend_from_slice(line.as_bytes());
            output.push(b'\n');
            Ok(())
        }

        fn write_str(&self, text: &str) -> std::io::Result<()> {
            self.output
                .lock()
                .expect("should lock recording terminal")
                .extend_from_slice(text.as_bytes());
            Ok(())
        }

        fn clear_line(&self) -> std::io::Result<()> {
            Ok(())
        }

        fn flush(&self) -> std::io::Result<()> {
            Ok(())
        }
    }

    struct ActiveIndicatifGuard {
        previous: Weak<IndicatifOutput>,
    }

    impl ActiveIndicatifGuard {
        fn install(indicatif: &Arc<IndicatifOutput>) -> Self {
            let active = ACTIVE_INDICATIF.get_or_init(|| Mutex::new(Weak::new()));
            let previous = std::mem::replace(
                &mut *active.lock().expect("should lock active indicatif handler"),
                Arc::downgrade(indicatif),
            );
            Self { previous }
        }
    }

    impl Drop for ActiveIndicatifGuard {
        fn drop(&mut self) {
            if let Some(active) = ACTIVE_INDICATIF.get() {
                *active
                    .lock()
                    .expect("should restore active indicatif handler") = self.previous.clone();
            }
        }
    }

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
            indicatif: None,
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
        assert!(lines[0].contains("Expected GCS block total and byte size pending"));
        assert!(!lines[0].contains('%'));
        assert!(lines[0].contains("3 GCS blocks uploaded (12.0 MiB)"));
        assert!(lines[0].contains("2 GCS blocks reused (8.0 MiB)"));
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
        assert!(lines[0].contains("GCS blocks handled: 3 of 8 (37%)"));
        assert!(lines[0].contains("3.0 KiB of 16.0 KiB GCS block-data bytes handled (18%)"));
        assert!(lines[0].contains("2 GCS blocks uploaded (2.0 KiB)"));
        assert!(lines[0].contains("1 GCS blocks reused (1.0 KiB)"));
    }

    #[test]
    fn test_indicatif_graph_plan_uses_acknowledged_position_and_failure_keeps_it_incomplete() {
        let indicatif = Arc::new(IndicatifOutput::new());
        let output = ProgressOutput {
            sink: Arc::new(|_| {}),
            interval: Duration::ZERO,
            indicatif: Some(Arc::clone(&indicatif)),
        };
        let progress = GraphUploadProgressReporter::with_output(output, 1, 1);
        progress.report_dolt_progress(DoltPushProgressEvent::Plan {
            missing_chunk_count: 10,
            missing_payload_bytes: 1024,
        });
        progress.report_dolt_progress(DoltPushProgressEvent::Uploaded {
            acknowledged_chunk_count: 3,
            acknowledged_payload_bytes: 256,
        });
        progress.report(UploadProgress {
            uploaded_blocks: 1,
            uploaded_bytes: 128,
            reused_blocks: 1,
            reused_bytes: 128,
            expected: Some(UploadPlan {
                blocks: 4,
                bytes: 512,
            }),
        });

        let bar = indicatif
            .bar
            .lock()
            .expect("should inspect indicatif bar")
            .bar
            .as_ref()
            .expect("should create the planned chunk bar")
            .clone();
        assert_eq!(bar.length(), Some(10));
        assert_eq!(bar.position(), 3);
        assert!(bar.message().contains("Dolt logical chunks acknowledged"));

        let detail_messages = indicatif
            .details
            .lock()
            .expect("should inspect indicatif counter lines")
            .iter()
            .map(ProgressBar::message)
            .collect::<Vec<_>>();
        assert!(detail_messages[0].contains("Dolt payload bytes: 256 of 1024"));
        assert!(detail_messages[1].contains("GCS blocks: 2 of 4; 1 uploaded, 1 reused"));
        assert!(detail_messages[2].contains("GCS block-data bytes: 256 of 512"));

        progress.failed();
        assert_eq!(bar.length(), Some(10));
        assert_eq!(bar.position(), 3);
        assert_eq!(bar.message(), "Graph upload failed");
    }

    #[test]
    fn test_indicatif_graph_finish_captures_counters_without_completing_failed_upload() {
        let indicatif = Arc::new(IndicatifOutput::new());
        let output = ProgressOutput {
            sink: Arc::new(|_| {}),
            interval: Duration::ZERO,
            indicatif: Some(Arc::clone(&indicatif)),
        };
        let progress = GraphUploadProgressReporter::with_output(output, 1, 1);
        progress.report_dolt_progress(DoltPushProgressEvent::Plan {
            missing_chunk_count: 10,
            missing_payload_bytes: 1024,
        });
        progress.report_dolt_progress(DoltPushProgressEvent::Uploaded {
            acknowledged_chunk_count: 3,
            acknowledged_payload_bytes: 256,
        });
        progress.report(UploadProgress {
            uploaded_blocks: 1,
            uploaded_bytes: 128,
            reused_blocks: 1,
            reused_bytes: 128,
            expected: Some(UploadPlan {
                blocks: 4,
                bytes: 512,
            }),
        });
        let bar = indicatif
            .bar
            .lock()
            .expect("should inspect indicatif bar before counter flush")
            .bar
            .as_ref()
            .expect("should create the planned chunk bar")
            .clone();

        // Operations call finish to flush the last counters even if a later publication stage
        // fails. That call must preserve the acknowledged position instead of filling the bar.
        progress.finish();

        assert_eq!(bar.length(), Some(10));
        assert_eq!(bar.position(), 3);
        assert_eq!(bar.message(), "Graph block counters captured");
        assert_ne!(
            bar.position(),
            bar.length().expect("should have a fixed chunk plan")
        );
    }

    #[test]
    fn test_indicatif_zero_chunk_plan_uses_message_without_a_fake_total() {
        let indicatif = IndicatifOutput::new();
        indicatif.graph_progress(
            Some(LogicalChunkProgress {
                plan_number: 1,
                planned_chunks: 0,
                planned_bytes: 0,
                acknowledged_chunks: 0,
                acknowledged_bytes: 0,
            }),
            None,
        );

        let state = indicatif
            .bar
            .lock()
            .expect("should inspect no-work bar state");
        assert_eq!(state.kind, Some(BarKind::Message));
        let bar = state
            .bar
            .as_ref()
            .expect("should show that no chunks are missing");
        assert_eq!(bar.length(), None);
        assert!(bar.message().contains("no destination-missing chunks"));
        assert!(!bar.message().contains("100%"));
    }

    #[test]
    fn test_terminal_with_hidden_indicatif_target_uses_bounded_line_output() {
        let hidden = Arc::new(IndicatifOutput::with_handler(
            MultiProgress::with_draw_target(ProgressDrawTarget::hidden()),
        ));
        let (mut output, lines) = captured_output(Duration::ZERO);

        output.indicatif = ProgressOutput::select_indicatif(None, true, || Arc::clone(&hidden));
        let status_output = output.clone();
        let progress = GraphUploadProgressReporter::with_output(output, 1, 1);
        progress.report_dolt_progress(DoltPushProgressEvent::Plan {
            missing_chunk_count: 2,
            missing_payload_bytes: 1024,
        });
        progress.report_dolt_progress(DoltPushProgressEvent::Uploaded {
            acknowledged_chunk_count: 1,
            acknowledged_payload_bytes: 512,
        });
        progress.report(UploadProgress {
            uploaded_blocks: 1,
            uploaded_bytes: 128,
            reused_blocks: 1,
            reused_bytes: 128,
            expected: Some(UploadPlan {
                blocks: 4,
                bytes: 512,
            }),
        });
        progress.finish();
        status_output.println("Graph staging complete.");

        let lines = lines.lock().expect("should read fallback progress output");
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Dolt logical chunk transfer plan 1")),
            "fallback output should include the native plan: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("GCS blocks handled: 2 of 4")),
            "fallback output should include current GCS counters: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Graph upload attempt 1/1:")),
            "fallback output should include the flushed transfer summary: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| line == "Graph staging complete."),
            "fallback output should retain the staging status line: {lines:?}"
        );
    }

    #[test]
    fn test_waiting_phase_and_operation_lines_reuse_the_active_indicatif_handler() {
        let terminal_output = Arc::new(Mutex::new(Vec::new()));
        let handler = MultiProgress::with_draw_target(ProgressDrawTarget::term_like(Box::new(
            RecordingTerm {
                output: Arc::clone(&terminal_output),
            },
        )));
        let indicatif = Arc::new(IndicatifOutput::with_handler(handler));
        let _active_handler = ActiveIndicatifGuard::install(&indicatif);

        let heartbeat = ProgressHeartbeat::waiting("publishing graph manifest");
        assert_eq!(
            heartbeat
                .bar
                .as_ref()
                .expect("should create a phase bar on the visible handler")
                .message(),
            "publishing graph manifest"
        );

        super::write_progress_line("Graph manifest published.");
        let rendered = String::from_utf8_lossy(
            &terminal_output
                .lock()
                .expect("should read recorded terminal output"),
        )
        .into_owned();
        assert!(
            rendered.contains("publishing graph manifest"),
            "waiting phase should render through the active MultiProgress handler: {rendered:?}"
        );
        assert!(
            rendered.contains("Graph manifest published."),
            "operation status should use the active MultiProgress handler: {rendered:?}"
        );
    }

    #[test]
    fn test_logical_chunk_progress_survives_interleaved_gcs_updates_and_throttling() {
        let (output, lines) = captured_output(Duration::from_secs(60));
        let progress = GraphUploadProgressReporter::with_output(output, 1, 2);
        progress.report_dolt_progress(DoltPushProgressEvent::Plan {
            missing_chunk_count: 8,
            missing_payload_bytes: 12 * 1024,
        });
        progress.report(UploadProgress {
            uploaded_blocks: 1,
            uploaded_bytes: 4 * 1024,
            ..UploadProgress::default()
        });
        progress.report(UploadProgress {
            uploaded_blocks: 2,
            uploaded_bytes: 8 * 1024,
            ..UploadProgress::default()
        });
        {
            let mut throttle = progress
                .logical_output
                .state
                .lock()
                .expect("should lock logical progress throttle");
            throttle.last_output = Some(Instant::now() - Duration::from_secs(61));
        }
        progress.report_dolt_progress(DoltPushProgressEvent::Uploaded {
            acknowledged_chunk_count: 5,
            acknowledged_payload_bytes: 9 * 1024,
        });
        progress.finish_dolt_progress(true);

        let lines = lines.lock().expect("should read interleaved progress");
        assert_eq!(lines.len(), 4);
        assert!(lines[0].contains("Dolt logical chunk transfer plan 1"));
        assert!(lines[0].contains("8 destination-missing chunks (12.0 KiB logical payload bytes)"));
        assert!(lines[1].contains(
            "Dolt logical chunk progress: 0 of 8 destination-missing chunks acknowledged (0%)"
        ));
        assert!(lines[1].contains("GCS block counters: 1 GCS blocks uploaded (4.0 KiB)"));
        assert!(lines[2].contains("Dolt logical chunks acknowledged for plan 1"));
        assert!(lines[2].contains("5 of 8 destination-missing chunks acknowledged (62%)"));
        assert!(lines[2].contains("9.0 KiB of 12.0 KiB logical payload bytes acknowledged (75%)"));
        assert!(lines[3].contains("Final Dolt logical chunk transfer counters"));
        assert!(
            lines
                .iter()
                .all(|line| !line.contains("GCS block total and byte size pending"))
        );
    }

    #[test]
    fn test_graph_failure_without_gcs_callbacks_reports_logical_chunk_progress() {
        let (output, lines) = captured_output(Duration::ZERO);
        let progress = GraphUploadProgressReporter::with_output(output, 1, 1);
        progress.report_dolt_progress(DoltPushProgressEvent::Plan {
            missing_chunk_count: 4,
            missing_payload_bytes: 16 * 1024,
        });
        progress.report_dolt_progress(DoltPushProgressEvent::Uploaded {
            acknowledged_chunk_count: 1,
            acknowledged_payload_bytes: 4 * 1024,
        });
        progress.failed();

        let lines = lines.lock().expect("should read failed graph progress");
        let failure = lines.last().expect("should report the graph failure");
        assert!(failure.contains("No completed GCS block counters were reported."));
        assert!(failure.contains("Last Dolt logical chunk progress:"));
        assert!(failure.contains("1 of 4 destination-missing chunks acknowledged (25%)"));
        assert!(failure.contains("4.0 KiB of 16.0 KiB logical payload bytes acknowledged (25%)"));
        assert!(!failure.contains("0 GCS blocks"));
    }

    #[test]
    fn test_dolt_replanned_chunk_counts_reset_before_final_summary() {
        let (output, lines) = captured_output(Duration::ZERO);
        let progress = GraphUploadProgressReporter::with_output(output, 1, 2);
        progress.report_dolt_progress(DoltPushProgressEvent::Plan {
            missing_chunk_count: 5,
            missing_payload_bytes: 10 * 1024,
        });
        progress.report_dolt_progress(DoltPushProgressEvent::Uploaded {
            acknowledged_chunk_count: 3,
            acknowledged_payload_bytes: 6 * 1024,
        });
        progress.report_dolt_progress(DoltPushProgressEvent::Plan {
            missing_chunk_count: 2,
            missing_payload_bytes: 4 * 1024,
        });
        progress.finish_dolt_progress(false);

        let lines = lines.lock().expect("should read replanned progress");
        assert!(lines.iter().any(|line| {
            line.contains("Dolt logical chunk plan 1 was replanned after")
                && line.contains("3 of 5 destination-missing chunks acknowledged (60%)")
        }));
        assert!(lines.iter().any(|line| {
            line.contains("Dolt logical chunk transfer plan 2")
                && line.contains("2 destination-missing chunks (4.0 KiB logical payload bytes)")
        }));
        assert!(lines.iter().any(|line| {
            line.contains("Final Dolt logical chunk transfer counters for plan 2")
                && line.contains("0 of 2 destination-missing chunks acknowledged (0%)")
                && line.contains("0 B of 4.0 KiB logical payload bytes acknowledged (0%)")
                && line.contains("after Dolt push returned an error")
        }));
    }

    #[test]
    fn test_dolt_chunk_plan_with_no_missing_chunks_has_no_percentage() {
        let (output, lines) = captured_output(Duration::ZERO);
        let progress = GraphUploadProgressReporter::with_output(output, 1, 2);
        progress.report_dolt_progress(DoltPushProgressEvent::Plan {
            missing_chunk_count: 0,
            missing_payload_bytes: 0,
        });
        progress.finish_dolt_progress(true);

        let lines = lines.lock().expect("should read empty chunk plan progress");
        assert!(lines[0].contains("0 destination-missing chunks (0 B logical payload bytes)"));
        assert!(lines[1].contains("not applicable (no logical chunk work planned)"));
        assert!(lines.iter().all(|line| !line.contains("100%")));
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
            indicatif: None,
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
            indicatif: None,
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
                && line.contains("GCS blocks handled: 3 of 4 (75%)")
                && line.contains("12.0 MiB of 24.0 MiB GCS block-data bytes handled (50%)")
                && line.contains("2 GCS blocks uploaded (8.0 MiB)")
                && line.contains("1 GCS blocks reused (4.0 MiB)")
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
        assert!(lines[0].contains("No completed GCS block counters were reported"));
        assert!(!lines[0].contains("0 GCS blocks uploaded"));
        assert!(!lines[0].contains("0 GCS blocks reused"));
    }

    #[test]
    fn test_expected_branch_conflict_reports_cleanup_without_transfer_failure_summary() {
        let (output, lines) = captured_output(Duration::ZERO);
        let progress = GraphUploadProgressReporter::with_output(output, 1, 2);
        progress.waiting_for_local_server_cleanup();
        progress.failed_silently();

        assert_eq!(
            *lines.lock().expect("should read branch conflict progress"),
            ["Waiting for local server cleanup..."]
        );
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
