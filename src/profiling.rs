use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[cfg(feature = "profiling")]
use pprof::{ProfilerGuardBuilder, Report};
use tracing::{Id, Subscriber, span::Attributes};
use tracing_subscriber::{
    layer::{Context, Layer, SubscriberExt as _},
    registry::{LookupSpan, Registry},
};

/// Collects tracing span durations and prints an exact report for every recorded call stack.
///
/// `Profiler` measures spans directly through a tracing subscriber, so it is useful when you
/// want per-stack call counts, total wall-clock time, and average time from span enter/exit
/// events. This will only measure code that has been instrumented with
/// ```
/// #[cfg_attr(
///    feature = "profiling",
///    tracing::instrument(...)
/// )]
/// ```
///
/// which means profiling is enabled as a feature.
#[derive(Clone, Default)]
pub struct Profiler {
    stats: Arc<Mutex<HashMap<String, Stat>>>,
}

#[derive(Default)]
struct Stat {
    calls: u64,
    total: Duration,
}

struct SpanState {
    stack: String,
    starts: Vec<Instant>,
}

impl Profiler {
    pub fn run<T>(self, f: impl FnOnce() -> T) -> T {
        let subscriber = Registry::default().with(ProfilerLayer {
            profiler: self.clone(),
        });
        tracing::subscriber::with_default(subscriber, || {
            let result = f();
            self.print_report();
            result
        })
    }

    fn record(&self, key: String, duration: Duration) {
        let mut stats = self.stats.lock().unwrap_or_else(|err| err.into_inner());
        let entry = stats.entry(key).or_default();
        entry.calls = entry.calls.saturating_add(1);
        entry.total += duration;
    }

    fn print_report(&self) {
        let stats = self.stats.lock().unwrap_or_else(|err| err.into_inner());
        let mut rows = stats
            .iter()
            .map(|(stack, stat)| (stack_frames(stack), stat.calls, stat.total))
            .collect::<Vec<_>>();
        // .2 is total duration of code, .0 is the stack frame names (method calls)
        rows.sort_by(|left, right| right.2.cmp(&left.2).then_with(|| left.0.cmp(&right.0)));
        let stack_width = rows
            .iter()
            .map(|(frames, _, _)| stack_label_width(frames))
            .max()
            .unwrap_or(0)
            .max("Stack".len());

        println!("Profile results");
        println!(
            "{:<stack_width$} {:>10} {:>14} {:>14}",
            "Stack", "Calls", "Total (ms)", "Avg (us)"
        );
        println!(
            "{:<stack_width$} {:>10} {:>14} {:>14}",
            "-----", "-----", "----------", "--------"
        );
        for (frames, calls, total) in rows {
            let total_ms = total.as_secs_f64() * 1000.0;
            let avg_us = if calls == 0 {
                0.0
            } else {
                total.as_secs_f64() * 1_000_000.0 / calls as f64
            };
            print_stacked_row(
                &frames,
                stack_width,
                format_args!("{:>10} {:>14.3} {:>14.3}", calls, total_ms, avg_us),
            );
        }
    }
}

struct ProfilerLayer {
    profiler: Profiler,
}

impl<S> Layer<S> for ProfilerLayer
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, context: Context<'_, S>) {
        if let Some(span) = context.span(id) {
            let metadata = attrs.metadata();
            let current = format!("{}::{}", metadata.target(), metadata.name());
            let stack = match span.parent() {
                Some(parent) => parent
                    .extensions()
                    .get::<SpanState>()
                    .map(|state| format!("{} -> {}", state.stack, current))
                    .unwrap_or(current),
                None => current,
            };
            let mut extensions = span.extensions_mut();
            extensions.insert(SpanState {
                stack,
                starts: Vec::new(),
            });
        }
    }

    fn on_enter(&self, id: &Id, context: Context<'_, S>) {
        if let Some(span) = context.span(id) {
            let mut extensions = span.extensions_mut();
            if let Some(state) = extensions.get_mut::<SpanState>() {
                state.starts.push(Instant::now());
            }
        }
    }

    fn on_exit(&self, id: &Id, context: Context<'_, S>) {
        if let Some(span) = context.span(id) {
            let mut extensions = span.extensions_mut();
            let snapshot = extensions.get_mut::<SpanState>().and_then(|state| {
                state
                    .starts
                    .pop()
                    .map(|started| (state.stack.clone(), started.elapsed()))
            });
            drop(extensions);
            if let Some((stack, duration)) = snapshot {
                self.profiler.record(stack, duration);
            }
        }
    }
}

#[cfg(feature = "profiling")]
/// Collects periodic samples from a `pprof` guard and prints an approximate hot-path report.
///
/// `SamplingProfiler` better suited for broad performance inspection, but it reports sampled
/// stacks. This means a snapshot of the code is taken every x intervals, and wherever the
/// code is, is recorded. This means some code that is run may never show up if it is never
/// sampled during its execution. But this likely means that code is fast and not a bottleneck.
#[derive(Clone, Default)]
pub struct SamplingProfiler;

#[cfg(feature = "profiling")]
impl SamplingProfiler {
    pub fn run<T>(self, f: impl FnOnce() -> T) -> T {
        let guard = ProfilerGuardBuilder::default()
            .frequency(100)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .ok();

        let result = f();
        if let Some(guard) = guard
            && let Ok(report) = guard.report().build()
        {
            self.print_report(&report);
        }

        result
    }

    fn print_report(&self, report: &Report) {
        let mut stats: HashMap<String, u64> = HashMap::new();
        let mut total_samples: u64 = 0;
        let sample_frequency = report.timing.frequency.max(1) as f64;

        for (frames, count) in &report.data {
            if *count <= 0 {
                continue;
            }

            let sample_count = *count as u64;
            total_samples = total_samples.saturating_add(sample_count);

            if let Some(stack) = sample_stack(frames) {
                let entry = stats.entry(stack).or_default();
                *entry = entry.saturating_add(sample_count);
            }
        }

        let mut rows = stats
            .into_iter()
            .map(|(stack, samples)| (stack_frames(&stack), samples))
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        let stack_width = rows
            .iter()
            .map(|(frames, _)| stack_label_width(frames))
            .max()
            .unwrap_or(0)
            .max("Stack".len());

        println!("Sampling profile results");
        println!(
            "{:<stack_width$} {:>10} {:>14} {:>10}",
            "Stack", "Samples", "Time (ms)", "Pct"
        );
        println!(
            "{:<stack_width$} {:>10} {:>14} {:>10}",
            "-----", "-------", "---------", "---"
        );
        for (frames, sample_count) in rows {
            let sample_time = duration_for_samples(sample_count, sample_frequency);
            let pct = if total_samples == 0 {
                0.0
            } else {
                (sample_count as f64 * 100.0) / total_samples as f64
            };
            print_stacked_row(
                &frames,
                stack_width,
                format_args!(
                    "{:>10} {:>14.3} {:>9.2}%",
                    sample_count,
                    sample_time.as_secs_f64() * 1000.0,
                    pct
                ),
            );
        }
    }
}

#[cfg(feature = "profiling")]
fn duration_for_samples(samples: u64, frequency: f64) -> Duration {
    Duration::from_secs_f64(samples as f64 / frequency)
}

#[cfg(feature = "profiling")]
fn stack_frames(stack: &str) -> Vec<String> {
    stack
        .split(" -> ")
        .map(ToString::to_string)
        .collect::<Vec<_>>()
}

#[cfg(feature = "profiling")]
fn trim_sample_prefix(frames: &mut Vec<String>) {
    let prefix = [
        "gen::main",
        "gen::profiling::SamplingProfiler::run",
        "gen::call_cli",
    ];

    if frames.len() >= prefix.len()
        && frames
            .iter()
            .take(prefix.len())
            .map(String::as_str)
            .eq(prefix)
    {
        frames.drain(..prefix.len());
    }
}

#[cfg(feature = "profiling")]
fn stack_label_width(frames: &[String]) -> usize {
    match frames.split_last() {
        Some((leaf, parents)) => parents.len() + leaf.len(),
        None => "<empty>".len(),
    }
}

#[cfg(feature = "profiling")]
fn print_stacked_row(frames: &[String], stack_width: usize, tail: std::fmt::Arguments<'_>) {
    match frames.split_last() {
        Some((leaf, parents)) => {
            for (depth, frame) in parents.iter().enumerate() {
                println!("{}{}", " ".repeat(depth), frame);
            }
            let label = format!("{}{}", " ".repeat(parents.len()), leaf);
            println!("{:<stack_width$} {}", label, tail);
        }
        None => println!("{:<stack_width$} {}", "<empty>", tail),
    }
}

#[cfg(feature = "profiling")]
fn sample_stack(frames: &pprof::Frames) -> Option<String> {
    let mut stack = Vec::new();
    for frame in frames.frames.iter().rev() {
        if let Some(name) = frame.iter().find_map(owned_frame_name) {
            stack.push(name);
        }
    }

    if stack.is_empty() {
        None
    } else {
        trim_sample_prefix(&mut stack);
        if stack.is_empty() {
            None
        } else {
            Some(stack.join(" -> "))
        }
    }
}

#[cfg(feature = "profiling")]
fn owned_frame_name(frame: &pprof::Symbol) -> Option<String> {
    let name = frame.name();
    if is_owned_code(&name) {
        Some(normalize_frame_name(&name))
    } else {
        None
    }
}

#[cfg(feature = "profiling")]
fn is_owned_code(name: &str) -> bool {
    matches!(
        name,
        n if n.starts_with("gen::")
            || n.starts_with("r#gen::")
            || n.starts_with("gen_models::")
            || n.starts_with("gen_core::")
            || n.starts_with("gen_graph::")
            || n.starts_with("gen_diff::")
            || n.starts_with("gen_annotations::")
            || n.starts_with("gen_tui::")
            || n.starts_with("gen_sugiyama::")
            || n.starts_with("gen_capnp_schemas::")
    )
}

#[cfg(feature = "profiling")]
fn normalize_frame_name(name: &str) -> String {
    name.strip_prefix("r#").unwrap_or(name).to_string()
}
