pub mod block_group;
pub mod collection;
pub mod diff;
pub mod dot_export;
pub mod gen_graph_widget;
pub mod inline_gen_graph_widget;
pub mod patch;

// Deprecated legacy modules - kept as dangling files for reference
// pub mod block_group_viewer_deprecated;
// pub mod block_layout_deprecated;
// pub mod operations; // depends on deprecated block_group_viewer

#[cfg(test)]
pub mod testing;
