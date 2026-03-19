#![cfg(test)]

use std::path::PathBuf;

use gen_models::{db::DbContext, sample::Sample};
use gen_tui::testing::create_test_terminal;
use ratatui::layout::Rect;

use crate::{
    imports::{fasta::import_fasta, gfa::import_gfa, library::import_library},
    test_helpers::setup_gen,
    track_database,
    updates::{
        fasta::update_with_fasta, gfa::update_with_gfa, library::update_with_library,
        sequence::update_with_sequence, vcf::update_with_vcf,
    },
    views::gen_graph_widget::{create_gen_graph_controller, create_gen_graph_widget},
};

fn fixture(relative_path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative_path)
}

/// Render the graph widget for (collection, sample) after the given setup closure runs,
/// and assert the result matches the stored snapshot for the calling test.
fn make_snapshot(collection: &str, sample: Option<&str>, setup: impl FnOnce(&DbContext)) {
    let context = setup_gen();
    track_database(context.graph().conn(), context.operations().conn())
        .expect("track_database failed");
    setup(&context);

    let conn = context.graph();
    let graph = Sample::get_graph(conn.conn(), collection, sample);
    let mut controller = create_gen_graph_controller(&graph);

    let area = Rect::new(0, 0, 80, 25);
    controller.viewport_state.viewport_bounds = area;
    controller.viewport_state.focus();

    let mut terminal = create_test_terminal(area.width, area.height);
    terminal
        .draw(|f| {
            let widget = create_gen_graph_widget(conn.conn());
            f.render_stateful_widget(widget, f.area(), &mut controller);
        })
        .expect("render failed");
    // Derive the snapshot name from the test thread name so each calling test
    // gets its own snapshot file even though assert_snapshot! is called here.
    let test_name = std::thread::current()
        .name()
        .and_then(|n| n.rsplit("::").next())
        .unwrap_or("snapshot")
        .to_owned();
    insta::assert_snapshot!(test_name, format!("{}", terminal.backend()));
}

// --- GFA import snapshots ---

#[test]
fn import_simple_gfa() {
    make_snapshot("test", None, |ctx| {
        import_gfa(ctx, &fixture("fixtures/simple.gfa"), "test", None).expect("import failed");
    });
}

#[test]
fn import_no_path_gfa() {
    make_snapshot("no path", None, |ctx| {
        import_gfa(ctx, &fixture("fixtures/no_path.gfa"), "no path", None).expect("import failed");
    });
}

#[test]
fn import_walk_gfa() {
    make_snapshot("walk", None, |ctx| {
        import_gfa(ctx, &fixture("fixtures/walk.gfa"), "walk", None).expect("import failed");
    });
}

#[test]
fn import_reverse_strand_gfa() {
    make_snapshot("test", None, |ctx| {
        import_gfa(ctx, &fixture("fixtures/reverse_strand.gfa"), "test", None)
            .expect("import failed");
    });
}

#[test]
fn import_cycle_no_path() {
    make_snapshot("/", None, |ctx| {
        import_gfa(ctx, &fixture("fixtures/gfa/cycle_no_path.gfa"), "/", None)
            .expect("import failed");
    });
}

#[test]
fn import_cycle_with_path() {
    make_snapshot("/", None, |ctx| {
        import_gfa(ctx, &fixture("fixtures/gfa/cycle_with_path.gfa"), "/", None)
            .expect("import failed");
    });
}

// --- GFA update snapshots ---

#[test]
fn update_with_gfa_path_diff() {
    make_snapshot("test", Some("applied diff"), |ctx| {
        import_fasta(
            ctx,
            &fixture("fixtures/simple.fa").to_string_lossy().into_owned(),
            "test",
            None,
            false,
        )
        .expect("fasta import failed");
        update_with_gfa(
            ctx,
            "test",
            None,
            "applied diff",
            fixture("fixtures/path-diff.gfa").to_str().unwrap(),
        )
        .expect("gfa update failed");
    });
}

#[test]
fn update_with_gfa_walk_diff() {
    make_snapshot("test", Some("applied diff"), |ctx| {
        import_fasta(
            ctx,
            &fixture("fixtures/simple.fa").to_string_lossy().into_owned(),
            "test",
            None,
            false,
        )
        .expect("fasta import failed");
        update_with_gfa(
            ctx,
            "test",
            None,
            "applied diff",
            fixture("fixtures/walk-diff.gfa").to_str().unwrap(),
        )
        .expect("gfa update failed");
    });
}

// --- FASTA update snapshots ---

#[test]
fn update_fasta_with_fasta() {
    // Graph after update: AT -> CGA -> TCGATCGATCGATCGGGAACACACAGAGA
    //                        \-> AAAAAAAA ->/
    make_snapshot("test", Some("child sample"), |ctx| {
        import_fasta(
            ctx,
            &fixture("fixtures/simple.fa").to_string_lossy().into_owned(),
            "test",
            None,
            false,
        )
        .expect("fasta import failed");
        update_with_fasta(
            ctx,
            "test",
            None,
            "child sample",
            "m123",
            2,
            5,
            fixture("fixtures/aaaaaaaa.fa").to_str().unwrap(),
            false,
        )
        .expect("fasta update failed");
    });
}

// --- VCF update snapshots ---

#[test]
fn update_fasta_with_vcf() {
    make_snapshot("test", None, |ctx| {
        import_fasta(
            ctx,
            &fixture("fixtures/simple.fa").to_string_lossy().into_owned(),
            "test",
            None,
            false,
        )
        .expect("fasta import failed");
        update_with_vcf(
            ctx,
            &fixture("fixtures/simple.vcf")
                .to_string_lossy()
                .into_owned(),
            "test",
            "".to_string(),
            "".to_string(),
            None,
        )
        .expect("vcf update failed");
    });
}

// --- Sequence update snapshots ---

#[test]
fn update_fasta_with_sequence() {
    // Graph after update: AT -> CGA -> TCGATCGATCGATCGGGAACACACAGAGA
    //                        \-> AAAAAAAA ->/
    make_snapshot("test", Some("child sample"), |ctx| {
        import_fasta(
            ctx,
            &fixture("fixtures/simple.fa").to_string_lossy().into_owned(),
            "test",
            None,
            false,
        )
        .expect("fasta import failed");
        update_with_sequence(
            ctx,
            "test",
            None,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        )
        .expect("sequence update failed");
    });
}

// --- Library import snapshots ---

#[test]
fn import_library_affix() {
    make_snapshot("test", None, |ctx| {
        import_library(
            ctx,
            "test",
            None,
            fixture("fixtures/affix_parts.fa").to_str().unwrap(),
            fixture("fixtures/affix_layout.csv").to_str().unwrap(),
            "library graph",
        )
        .expect("library import failed");
    });
}

#[test]
fn import_library_single_column() {
    make_snapshot("test", None, |ctx| {
        import_library(
            ctx,
            "test",
            None,
            fixture("fixtures/parts.fa").to_str().unwrap(),
            fixture("fixtures/single_column_design.csv")
                .to_str()
                .unwrap(),
            "m123",
        )
        .expect("library import failed");
    });
}

#[test]
fn import_library_two_columns() {
    make_snapshot("test", None, |ctx| {
        import_library(
            ctx,
            "test",
            None,
            fixture("fixtures/parts.fa").to_str().unwrap(),
            fixture("fixtures/design_reusing_parts.csv")
                .to_str()
                .unwrap(),
            "m123",
        )
        .expect("library import failed");
    });
}

// --- Library update snapshots ---

#[test]
fn update_with_library_pool() {
    make_snapshot("test", Some("new sample"), |ctx| {
        import_fasta(
            ctx,
            &fixture("fixtures/simple.fa").to_string_lossy().into_owned(),
            "test",
            None,
            false,
        )
        .expect("fasta import failed");
        update_with_library(
            ctx,
            "test",
            None,
            "new sample",
            "m123",
            7,
            20,
            fixture("fixtures/parts.fa").to_str().unwrap(),
            fixture("fixtures/combinatorial_design.csv")
                .to_str()
                .unwrap(),
        )
        .expect("library update failed");
    });
}

#[test]
fn update_with_library_single_column() {
    make_snapshot("test", Some("new sample"), |ctx| {
        import_fasta(
            ctx,
            &fixture("fixtures/simple.fa").to_string_lossy().into_owned(),
            "test",
            None,
            false,
        )
        .expect("fasta import failed");
        update_with_library(
            ctx,
            "test",
            None,
            "new sample",
            "m123",
            7,
            20,
            fixture("fixtures/parts.fa").to_str().unwrap(),
            fixture("fixtures/single_column_design.csv")
                .to_str()
                .unwrap(),
        )
        .expect("library update failed");
    });
}

#[test]
fn update_with_library_two_columns() {
    make_snapshot("test", Some("new sample"), |ctx| {
        import_fasta(
            ctx,
            &fixture("fixtures/simple.fa").to_string_lossy().into_owned(),
            "test",
            None,
            false,
        )
        .expect("fasta import failed");
        update_with_library(
            ctx,
            "test",
            None,
            "new sample",
            "m123",
            7,
            20,
            fixture("fixtures/parts.fa").to_str().unwrap(),
            fixture("fixtures/design_reusing_parts.csv")
                .to_str()
                .unwrap(),
        )
        .expect("library update failed");
    });
}
