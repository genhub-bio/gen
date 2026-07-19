pub mod annotation_files;
pub mod annotation_groups;
pub mod annotation_track;
pub mod annotations;
pub mod block_group;
pub mod block_group_inline;
pub mod collection;
#[cfg(not(target_os = "emscripten"))]
pub mod diff;
pub mod diff_graph;
pub mod dot_export;
pub mod emscripten_backend;
pub mod emscripten_input;
pub mod gen_graph_widget;
pub mod graph_overlay;
pub mod helpers;
pub mod inline_label_placement;
pub mod messages;
#[cfg(not(target_os = "emscripten"))]
pub mod operations;
pub mod panels;
#[cfg(not(target_os = "emscripten"))]
pub mod patch;
pub mod samples;
pub mod tui_runtime;

#[cfg(test)]
pub mod testing;
