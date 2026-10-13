use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
    sync::mpsc::{self, Receiver, SyncSender, TrySendError},
    thread,
    time::{Duration, Instant},
};

use gen_core::{HashId, Workspace};

use crate::views::{
    annotation_files::AnnotationFileEntry,
    annotations::{
        AnnotationFileTrackLoadResult, AnnotationFileTrackRequest, load_annotation_file_track,
    },
    helpers::open_worker_connection,
};

pub(super) struct AnnotationLoadSpec {
    pub collection_name: String,
    pub sample_name: String,
    pub block_group_id: HashId,
    pub block_group_name: String,
    pub history_ref: Option<String>,
    pub query_window: Option<(i64, i64)>,
    pub node_filter: HashSet<HashId>,
    pub entry: AnnotationFileEntry,
}

#[derive(Clone, Debug)]
pub(super) struct AnnotationLoadIdentity {
    pub generation: u64,
    pub activation: u64,
    pub request_id: u64,
    pub collection_name: String,
    pub sample_name: String,
    pub block_group_id: HashId,
    pub block_group_name: String,
    pub history_ref: Option<String>,
    pub query_window: Option<(i64, i64)>,
    pub entry: AnnotationFileEntry,
}

struct AnnotationLoadJob {
    identity: AnnotationLoadIdentity,
    node_filter: HashSet<HashId>,
}

pub(super) struct AnnotationLoadResult {
    pub identity: AnnotationLoadIdentity,
    pub result: Result<AnnotationFileTrackLoadResult, String>,
}

struct AnnotationLoadWorker {
    jobs: Option<SyncSender<Box<AnnotationLoadJob>>>,
    results: Receiver<AnnotationLoadResult>,
}

impl AnnotationLoadWorker {
    fn spawn(
        db_path: PathBuf,
        workspace: Workspace,
        branch: Option<String>,
        history_ref: Option<String>,
    ) -> std::io::Result<Self> {
        let (job_sender, job_receiver) = mpsc::sync_channel::<Box<AnnotationLoadJob>>(1);
        let (result_sender, result_receiver) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name("annotation-track-loader".to_string())
            .spawn(move || {
                let connection =
                    open_worker_connection(&db_path, branch.as_deref(), history_ref.as_deref());
                while let Ok(job) = job_receiver.recv() {
                    let job = *job;
                    let result = match connection.as_ref() {
                        Ok(connection) => {
                            let query = AnnotationFileTrackRequest {
                                conn: connection,
                                history_ref: job.identity.history_ref.as_deref(),
                                workspace: &workspace,
                                collection_name: &job.identity.collection_name,
                                sample_name: &job.identity.sample_name,
                                block_group_name: Some(&job.identity.block_group_name),
                                query_window: job.identity.query_window,
                                node_filter: &job.node_filter,
                                entry: &job.identity.entry,
                            };
                            load_annotation_file_track(&query).map_err(|error| error.to_string())
                        }
                        Err(error) => Err(error.clone()),
                    };
                    if result_sender
                        .send(AnnotationLoadResult {
                            identity: job.identity,
                            result,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })?;
        Ok(Self {
            jobs: Some(job_sender),
            results: result_receiver,
        })
    }

    fn try_send(&self, job: AnnotationLoadJob) -> Result<(), TrySendError<Box<AnnotationLoadJob>>> {
        let job = Box::new(job);
        match self.jobs.as_ref() {
            Some(jobs) => jobs.try_send(job),
            None => Err(TrySendError::Disconnected(job)),
        }
    }
}

struct InFlightAnnotationLoad {
    identity: AnnotationLoadIdentity,
    started_at: Instant,
}

pub(super) struct AnnotationLoadStatus {
    pub display_name: String,
    pub elapsed: Duration,
    pub waiting: bool,
}

/// Owns the bounded job queue and rejects results from an old view or activation.
pub(super) struct AnnotationLoadCoordinator {
    worker: AnnotationLoadWorker,
    generation: u64,
    activations: HashMap<HashId, u64>,
    next_request_id: u64,
    pending: VecDeque<AnnotationLoadJob>,
    ready_results: VecDeque<AnnotationLoadResult>,
    in_flight: Option<InFlightAnnotationLoad>,
}

impl AnnotationLoadCoordinator {
    pub(super) fn new(
        db_path: PathBuf,
        workspace: Workspace,
        branch: Option<String>,
        history_ref: Option<String>,
    ) -> std::io::Result<Self> {
        Ok(Self {
            worker: AnnotationLoadWorker::spawn(db_path, workspace, branch, history_ref)?,
            generation: 0,
            activations: HashMap::new(),
            next_request_id: 0,
            pending: VecDeque::new(),
            ready_results: VecDeque::new(),
            in_flight: None,
        })
    }

    pub(super) fn invalidate_view(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.activations.clear();
        self.pending.clear();
    }

    pub(super) fn queue_activation(&mut self, spec: AnnotationLoadSpec) {
        let file_id = spec.entry.file_addition.id;
        let activation = self
            .activations
            .get(&file_id)
            .copied()
            .unwrap_or_default()
            .wrapping_add(1);
        self.activations.insert(file_id, activation);
        self.queue(spec, activation);
    }

    pub(super) fn queue_reload(&mut self, spec: AnnotationLoadSpec) {
        let file_id = spec.entry.file_addition.id;
        if let Some(activation) = self.activations.get(&file_id).copied() {
            self.queue(spec, activation);
        }
    }

    fn queue(&mut self, spec: AnnotationLoadSpec, activation: u64) {
        self.next_request_id = self.next_request_id.wrapping_add(1);
        let file_id = spec.entry.file_addition.id;
        let identity = AnnotationLoadIdentity {
            generation: self.generation,
            activation,
            request_id: self.next_request_id,
            collection_name: spec.collection_name,
            sample_name: spec.sample_name,
            block_group_id: spec.block_group_id,
            block_group_name: spec.block_group_name,
            history_ref: spec.history_ref,
            query_window: spec.query_window,
            entry: spec.entry,
        };
        let job = AnnotationLoadJob {
            identity,
            node_filter: spec.node_filter,
        };
        self.pending
            .retain(|pending| pending.identity.entry.file_addition.id != file_id);
        self.pending.push_back(job);
    }

    pub(super) fn cancel_file(&mut self, file_id: HashId) {
        let activation = self
            .activations
            .get(&file_id)
            .copied()
            .unwrap_or_default()
            .wrapping_add(1);
        self.activations.insert(file_id, activation);
        self.pending
            .retain(|pending| pending.identity.entry.file_addition.id != file_id);
    }

    pub(super) fn is_current(&self, identity: &AnnotationLoadIdentity) -> bool {
        self.generation == identity.generation
            && self.activations.get(&identity.entry.file_addition.id) == Some(&identity.activation)
    }

    pub(super) fn try_receive(&mut self) -> Option<AnnotationLoadResult> {
        let result = if let Some(result) = self.ready_results.pop_front() {
            result
        } else {
            match self.worker.results.try_recv() {
                Ok(result) => result,
                Err(mpsc::TryRecvError::Empty) => return None,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if let Some(in_flight) = self.in_flight.take() {
                        self.ready_results.push_back(AnnotationLoadResult {
                            identity: in_flight.identity,
                            result: Err("annotation loading worker stopped".to_string()),
                        });
                    }
                    self.ready_results.extend(self.pending.drain(..).map(|job| {
                        AnnotationLoadResult {
                            identity: job.identity,
                            result: Err("annotation loading worker stopped".to_string()),
                        }
                    }));
                    self.ready_results.pop_front()?
                }
            }
        };
        if self
            .in_flight
            .as_ref()
            .is_some_and(|in_flight| in_flight.identity.request_id == result.identity.request_id)
        {
            self.in_flight = None;
        }
        Some(result)
    }

    pub(super) fn dispatch_pending(&mut self) {
        if self.in_flight.is_some() {
            return;
        }
        while let Some(job) = self.pending.pop_front() {
            if !self.is_current(&job.identity) {
                continue;
            }
            let identity = job.identity.clone();
            match self.worker.try_send(job) {
                Ok(()) => {
                    self.in_flight = Some(InFlightAnnotationLoad {
                        identity,
                        started_at: Instant::now(),
                    });
                    return;
                }
                Err(TrySendError::Full(job)) => {
                    self.pending.push_front(*job);
                    return;
                }
                Err(TrySendError::Disconnected(job)) => {
                    self.ready_results.push_back(AnnotationLoadResult {
                        identity: job.identity,
                        result: Err("annotation loading worker stopped".to_string()),
                    });
                }
            }
        }
    }

    pub(super) fn status(&self) -> Option<AnnotationLoadStatus> {
        if let Some(in_flight) = self.in_flight.as_ref()
            && self.is_current(&in_flight.identity)
        {
            return Some(AnnotationLoadStatus {
                display_name: in_flight.identity.entry.display_name.clone(),
                elapsed: in_flight.started_at.elapsed(),
                waiting: false,
            });
        }
        self.pending
            .iter()
            .find(|pending| self.is_current(&pending.identity))
            .map(|pending| AnnotationLoadStatus {
                display_name: pending.identity.entry.display_name.clone(),
                elapsed: Duration::ZERO,
                waiting: true,
            })
    }

    pub(super) fn is_busy(&self) -> bool {
        self.in_flight.is_some() || !self.pending.is_empty() || !self.ready_results.is_empty()
    }

    pub(super) fn has_current_work_for_file(&self, file_id: HashId) -> bool {
        self.pending.iter().any(|pending| {
            pending.identity.entry.file_addition.id == file_id && self.is_current(&pending.identity)
        }) || self.in_flight.as_ref().is_some_and(|in_flight| {
            in_flight.identity.entry.file_addition.id == file_id
                && self.is_current(&in_flight.identity)
        })
    }

    pub(super) fn has_current_request_covering(
        &self,
        file_id: HashId,
        visible_window: (i64, i64),
    ) -> bool {
        let covers = |identity: &AnnotationLoadIdentity| {
            identity.entry.file_addition.id == file_id
                && self.is_current(identity)
                && identity.query_window.is_some_and(|(start, end)| {
                    start <= visible_window.0 && end >= visible_window.1
                })
        };
        self.pending.iter().any(|pending| covers(&pending.identity))
            || self
                .in_flight
                .as_ref()
                .is_some_and(|in_flight| covers(&in_flight.identity))
    }

    pub(super) fn cancel_pending_reloads(&mut self, file_id: HashId) {
        self.pending
            .retain(|pending| pending.identity.entry.file_addition.id != file_id);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        path::PathBuf,
        thread,
        time::{Duration, Instant},
    };

    use gen_core::{HashId, Strand};
    use gen_models::{
        annotations::{AnnotationFileChecksumOverrides, add_annotation_file},
        block_group::BlockGroup,
        history::dolt::{active_branch, connect_branch, create_branch},
        operations::commit_operation_summary,
        sample::Sample,
    };

    use super::{AnnotationLoadCoordinator, AnnotationLoadSpec};
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::setup_gen_on_disk,
        views::{
            annotation_files::{AnnotationFileEntry, load_annotation_file_entries},
            annotations::{AnnotationFileTrackRequest, load_annotation_file_track},
        },
    };

    type AnnotationSegmentSignature = (HashId, i64, i64, Strand);
    type AnnotationSignature = (String, Vec<AnnotationSegmentSignature>);

    fn setup_annotation_file() -> (
        crate::test_helpers::TestContext,
        BlockGroup,
        AnnotationFileEntry,
        String,
    ) {
        let context = setup_gen_on_disk();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let fasta_path_string = fasta_path
            .to_str()
            .expect("should encode FASTA fixture path")
            .to_string();
        let import = import_fasta(
            &context,
            &fasta_path_string,
            "test",
            Sample::DEFAULT_NAME,
            None,
            None,
        )
        .expect("should import the fixture FASTA");
        commit_operation_summary(&context, &import).expect("should commit the fixture FASTA");

        let conn = context.graph().conn();
        create_branch(conn, "annotation-loader-feature").expect("should create the feature branch");
        connect_branch(conn, "annotation-loader-feature")
            .expect("should connect the feature branch");

        let annotation_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.gff");
        let annotation_path_string = annotation_path
            .to_str()
            .expect("should encode GFF fixture path")
            .to_string();
        let history_ref = add_annotation_file(
            &context,
            &annotation_path_string,
            Some("gff3"),
            None,
            Some("feature-annotations"),
            Some("add feature annotations"),
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should add the feature annotation file")
        .to_string();
        let block_group = BlockGroup::get_by_name(conn, "test", Sample::DEFAULT_NAME, "m123", None)
            .expect("should find the imported block group");
        let entry = load_annotation_file_entries(conn, None)
            .into_iter()
            .next()
            .expect("should load the feature annotation entry");
        (context, block_group, entry, history_ref)
    }

    fn annotation_spec(
        block_group: &BlockGroup,
        entry: AnnotationFileEntry,
        history_ref: Option<String>,
        node_filter: HashSet<HashId>,
    ) -> AnnotationLoadSpec {
        AnnotationLoadSpec {
            collection_name: "test".to_string(),
            sample_name: block_group.sample_name.clone(),
            block_group_id: block_group.id,
            block_group_name: block_group.name.clone(),
            history_ref,
            query_window: Some((0, 34)),
            node_filter,
            entry,
        }
    }

    fn wait_for_result(coordinator: &mut AnnotationLoadCoordinator) -> super::AnnotationLoadResult {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            coordinator.dispatch_pending();
            if let Some(result) = coordinator.try_receive() {
                return result;
            }
            assert!(
                Instant::now() < deadline,
                "annotation load should finish promptly"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn annotation_signature(
        load: &crate::views::annotations::AnnotationFileTrackLoadResult,
    ) -> Vec<AnnotationSignature> {
        let mut signature = load
            .track
            .annotations
            .iter()
            .map(|annotation| {
                (
                    annotation.name.clone(),
                    annotation
                        .segments
                        .iter()
                        .map(|segment| {
                            (segment.node_id, segment.start, segment.end, segment.strand)
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        signature.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        signature
    }

    fn load_in_background(
        context: &gen_models::db::DbContext,
        spec: AnnotationLoadSpec,
        branch: Option<String>,
    ) -> super::AnnotationLoadResult {
        let mut coordinator = AnnotationLoadCoordinator::new(
            context
                .workspace()
                .graph_db_path()
                .expect("should resolve graph database path"),
            context.workspace().clone(),
            branch,
            spec.history_ref.clone(),
        )
        .expect("should start annotation loading worker");
        coordinator.queue_activation(spec);
        assert!(
            coordinator.is_busy(),
            "queued work should keep the UI polling"
        );
        coordinator.dispatch_pending();
        assert!(
            coordinator.status().is_some(),
            "active work should provide loading status"
        );
        wait_for_result(&mut coordinator)
    }

    #[test]
    fn test_worker_matches_active_feature_branch_and_read_only_history() {
        let (context, block_group, current_entry, history_ref) = setup_annotation_file();
        let conn = context.graph().conn();
        let branch = active_branch(conn).expect("should read the active feature branch");
        assert_eq!(branch, "annotation-loader-feature");

        let graph = BlockGroup::get_graph(conn, context.workspace(), &block_group.id, None)
            .expect("should load the current feature graph");
        let node_filter = graph
            .nodes()
            .map(|node| node.node_id)
            .collect::<HashSet<_>>();
        let expected_current = load_annotation_file_track(&AnnotationFileTrackRequest {
            conn,
            history_ref: None,
            workspace: context.workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_name: Some("m123"),
            query_window: Some((0, 34)),
            node_filter: &node_filter,
            entry: &current_entry,
        })
        .expect("should load annotations from the current branch");
        let current_result = load_in_background(
            &context,
            annotation_spec(
                &block_group,
                current_entry.clone(),
                None,
                node_filter.clone(),
            ),
            Some(branch),
        );
        let current_load = current_result
            .result
            .expect("should load annotations on the active branch");
        assert_eq!(
            annotation_signature(&current_load),
            annotation_signature(&expected_current),
            "worker should reconnect to the selected branch's working set"
        );

        let history_entry = load_annotation_file_entries(conn, Some(&history_ref))
            .into_iter()
            .next()
            .expect("should load the annotation entry at the commit");
        let history_group = BlockGroup::get_by_id(conn, &block_group.id, Some(&history_ref))
            .expect("should load the block group at the commit");
        let history_graph = BlockGroup::get_graph(
            conn,
            context.workspace(),
            &history_group.id,
            Some(&history_ref),
        )
        .expect("should load the historical block graph");
        let history_node_filter = history_graph
            .nodes()
            .map(|node| node.node_id)
            .collect::<HashSet<_>>();
        let expected_history = load_annotation_file_track(&AnnotationFileTrackRequest {
            conn,
            history_ref: Some(&history_ref),
            workspace: context.workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_name: Some("m123"),
            query_window: Some((0, 34)),
            node_filter: &history_node_filter,
            entry: &history_entry,
        })
        .expect("should load annotations at the historical commit");
        let history_result = load_in_background(
            &context,
            annotation_spec(
                &history_group,
                history_entry,
                Some(history_ref),
                history_node_filter,
            ),
            None,
        );
        let history_load = history_result
            .result
            .expect("should load annotations through the read-only connection");
        assert_eq!(
            annotation_signature(&history_load),
            annotation_signature(&expected_history),
            "worker history results should match the foreground history view"
        );
    }

    #[test]
    fn test_file_toggle_and_view_change_invalidate_completed_loads() {
        let (context, block_group, entry, _history_ref) = setup_annotation_file();
        let graph = BlockGroup::get_graph(
            context.graph().conn(),
            context.workspace(),
            &block_group.id,
            None,
        )
        .expect("should load the feature graph");
        let node_filter = graph
            .nodes()
            .map(|node| node.node_id)
            .collect::<HashSet<_>>();
        let branch =
            active_branch(context.graph().conn()).expect("should read the active feature branch");
        let mut coordinator = AnnotationLoadCoordinator::new(
            context
                .workspace()
                .graph_db_path()
                .expect("should resolve graph database path"),
            context.workspace().clone(),
            Some(branch),
            None,
        )
        .expect("should start annotation loading worker");
        let file_id = entry.file_addition.id;
        coordinator.queue_activation(annotation_spec(
            &block_group,
            entry.clone(),
            None,
            node_filter.clone(),
        ));
        let queued_identity = coordinator
            .pending
            .front()
            .expect("should queue a file load")
            .identity
            .clone();
        assert!(coordinator.is_current(&queued_identity));
        coordinator.cancel_file(file_id);
        assert!(
            !coordinator.is_current(&queued_identity),
            "turning a file off should invalidate its queued activation"
        );
        assert!(!coordinator.has_current_work_for_file(file_id));

        coordinator.queue_activation(annotation_spec(
            &block_group,
            entry.clone(),
            None,
            node_filter.clone(),
        ));
        coordinator.dispatch_pending();
        let completed_result = wait_for_result(&mut coordinator);
        coordinator.cancel_file(file_id);
        assert!(
            !coordinator.is_current(&completed_result.identity),
            "a result arriving after a file toggle should be ignored"
        );

        coordinator.queue_activation(annotation_spec(&block_group, entry, None, node_filter));
        coordinator.dispatch_pending();
        let completed_result = wait_for_result(&mut coordinator);
        coordinator.invalidate_view();
        assert!(
            !coordinator.is_current(&completed_result.identity),
            "switching block group or collection should invalidate previous-view results"
        );
    }

    #[test]
    fn test_pan_back_to_loaded_window_discards_queued_future_window() {
        let (context, block_group, entry, _history_ref) = setup_annotation_file();
        let graph = BlockGroup::get_graph(
            context.graph().conn(),
            context.workspace(),
            &block_group.id,
            None,
        )
        .expect("should load the feature graph");
        let node_filter = graph
            .nodes()
            .map(|node| node.node_id)
            .collect::<HashSet<_>>();
        let branch =
            active_branch(context.graph().conn()).expect("should read the active feature branch");
        let mut coordinator = AnnotationLoadCoordinator::new(
            context
                .workspace()
                .graph_db_path()
                .expect("should resolve graph database path"),
            context.workspace().clone(),
            Some(branch),
            None,
        )
        .expect("should start annotation loading worker");
        let file_id = entry.file_addition.id;
        coordinator.queue_activation(annotation_spec(
            &block_group,
            entry.clone(),
            None,
            node_filter.clone(),
        ));
        coordinator.dispatch_pending();

        let mut future_window = annotation_spec(&block_group, entry, None, node_filter);
        future_window.query_window = Some((100, 200));
        coordinator.queue_reload(future_window);
        assert_eq!(coordinator.pending.len(), 1);
        assert_eq!(
            coordinator
                .pending
                .front()
                .map(|job| job.identity.query_window),
            Some(Some((100, 200))),
            "the pan to B should wait behind the in-flight A query"
        );
        assert!(
            coordinator.has_current_request_covering(file_id, (5, 20)),
            "panning back to A should reuse the still-running A query"
        );

        let result = wait_for_result(&mut coordinator);
        let loaded_window = result
            .result
            .as_ref()
            .expect("should finish the in-flight A query")
            .loaded_window;
        assert!(
            loaded_window.is_some_and(|(start, end)| start <= 5 && end >= 20),
            "the in-flight A query should cover the viewport after the pan back"
        );
        coordinator.cancel_pending_reloads(file_id);
        assert!(
            !coordinator.has_current_work_for_file(file_id),
            "an A result that covers the current viewport should discard obsolete B work"
        );
    }

    #[test]
    fn test_disconnected_worker_returns_error_without_staying_busy() {
        let (context, block_group, entry, _history_ref) = setup_annotation_file();
        let graph = BlockGroup::get_graph(
            context.graph().conn(),
            context.workspace(),
            &block_group.id,
            None,
        )
        .expect("should load the feature graph");
        let branch =
            active_branch(context.graph().conn()).expect("should read the active feature branch");
        let mut coordinator = AnnotationLoadCoordinator::new(
            context
                .workspace()
                .graph_db_path()
                .expect("should resolve graph database path"),
            context.workspace().clone(),
            Some(branch),
            None,
        )
        .expect("should start annotation loading worker");
        let file_id = entry.file_addition.id;
        coordinator.queue_activation(annotation_spec(
            &block_group,
            entry,
            None,
            graph.nodes().map(|node| node.node_id).collect(),
        ));
        drop(coordinator.worker.jobs.take());
        coordinator.dispatch_pending();
        assert!(coordinator.is_busy());
        let result = coordinator
            .try_receive()
            .expect("should surface the disconnected worker as a result");
        assert_eq!(result.identity.entry.file_addition.id, file_id);
        assert!(result.result.is_err());
        assert!(
            !coordinator.is_busy(),
            "a failed worker should stop the loading indicator"
        );
    }

    #[test]
    fn test_worker_reports_missing_annotation_asset_and_stops_loading() {
        let (context, block_group, mut entry, _history_ref) = setup_annotation_file();
        entry.file_addition.asset_uri = "/missing/annotation-file.gff3".to_string();
        entry.file_addition.checksum = None;
        entry.index_file_addition = None;
        entry.attribute_index_file_addition = None;
        let graph = BlockGroup::get_graph(
            context.graph().conn(),
            context.workspace(),
            &block_group.id,
            None,
        )
        .expect("should load the feature graph");
        let branch =
            active_branch(context.graph().conn()).expect("should read the active feature branch");
        let result = load_in_background(
            &context,
            annotation_spec(
                &block_group,
                entry,
                None,
                graph.nodes().map(|node| node.node_id).collect(),
            ),
            Some(branch),
        );
        assert!(
            result.result.is_err(),
            "missing sources should reach the UI as a load error"
        );
    }
}
