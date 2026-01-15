pub mod block_group;
pub mod collection;
// pub mod diff; // temporarily disabled due to API conflicts
pub mod dot_export;
pub mod gen_graph_widget;
pub mod inline_gen_graph_widget;
pub mod patch;

// Deprecated legacy modules - kept as dangling files for reference
// Temporarily disabled due to API conflicts after merge - need to update deprecated code
// pub mod block_group_viewer_deprecated;
// pub mod block_layout_deprecated;
// pub mod operations_deprecated;

#[cfg(test)]
pub mod testing;
