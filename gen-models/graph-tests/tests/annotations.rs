use gen_models::{
    annotations::Annotation,
    assets::{OperationKind, OperationLog},
    errors::OperationError,
    history::{HistoryStore, dolt::DoltHistoryStore},
    operations::commit_operation_summary,
    traits::Query,
};
use gen_models_graph_tests::{setup_block_group, setup_gen};

#[test]
fn test_add_annotation_creates_annotation() {
    let context = setup_gen();
    let conn = context.graph().conn();
    let history_store = DoltHistoryStore::new(conn);
    setup_block_group(conn);

    let summary = gen_graph::models::add_annotation(
        &context,
        "test",
        "gene-a",
        Some("track-1"),
        "test",
        "chr1:1-5",
    )
    .expect("should add the annotation");
    let commit_hash =
        commit_operation_summary(&context, &summary).expect("should commit the annotation");
    assert_eq!(
        history_store.current_head().expect("should read history"),
        Some(commit_hash)
    );
    let mut operation_logs = OperationLog::all(conn);
    operation_logs.sort_by_key(|operation_log| core::cmp::Reverse(operation_log.created_on));
    assert_eq!(
        operation_logs[0].operation_kind,
        OperationKind::Other("add annotation gene-a".to_string())
    );
    let annotations =
        Annotation::query_by_group(conn, "track-1", None).expect("should query annotations");
    assert_eq!(annotations.len(), 1);
    assert_eq!(annotations[0].name, "gene-a");
}

#[test]
fn test_add_annotation_detects_no_changes() {
    let context = setup_gen();
    let conn = context.graph().conn();
    setup_block_group(conn);

    let add = || {
        gen_graph::models::add_annotation(
            &context,
            "test",
            "gene-a",
            Some("track-1"),
            "test",
            "chr1:1-5",
        )
        .expect("should add the annotation")
    };
    commit_operation_summary(&context, &add()).expect("should commit the first annotation");
    let error = commit_operation_summary(&context, &add())
        .expect_err("should detect that the annotation already exists");
    assert_eq!(error, OperationError::NoChanges);
}
