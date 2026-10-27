pub mod annotation_files;
pub mod annotation_groups;
mod annotation_loading;
pub mod annotation_track;
pub mod annotations;
#[cfg(all(feature = "cli", not(target_os = "emscripten")))]
pub mod block_group;
#[cfg(all(feature = "cli", not(target_os = "emscripten")))]
pub mod block_group_inline;
#[cfg(all(feature = "cli", not(target_os = "emscripten")))]
pub mod collection;
#[cfg(all(feature = "cli", not(target_os = "emscripten")))]
pub mod diff;
pub mod diff_graph;
pub mod dot_export;
pub mod gen_graph_widget;
pub mod graph_overlay;
pub mod helpers;
pub mod inline_label_placement;
pub mod messages;
#[cfg(all(feature = "cli", not(target_os = "emscripten")))]
pub mod operations;
pub mod panels;
#[cfg(all(feature = "cli", not(target_os = "emscripten")))]
pub mod patch;
#[cfg(all(feature = "cli", not(target_os = "emscripten")))]
pub mod region_search;
#[cfg(all(feature = "cli", not(target_os = "emscripten")))]
pub mod samples;
#[cfg(all(feature = "cli", not(target_os = "emscripten")))]
pub mod tui_runtime;

#[cfg(all(test, feature = "cli"))]
pub mod testing;
