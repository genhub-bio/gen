pub mod annotation_files;
pub mod annotation_groups;
pub mod annotation_track;
pub mod annotations;
pub mod block_group;
pub mod block_group_inline;
pub mod collection;
pub mod diff;
pub mod diff_graph;
pub mod dot_export;
pub mod gen_graph_widget;
pub mod helpers;
pub mod inline_label_placement;
pub mod messages;
#[cfg(feature = "native-tui")]
pub mod operations;
#[cfg(all(feature = "browser-tui", not(feature = "native-tui")))]
pub mod operations {
    use std::io;

    use gen_models::{db::DbContext, operations::Operation};

    pub fn view_operations(
        _context: &DbContext,
        _operations: &[Operation],
    ) -> Result<(), io::Error> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "operations --interactive is not available in browser-tui builds yet",
        ))
    }
}
pub mod panels;
pub mod patch;
pub mod samples;
pub mod tui_runtime;

#[cfg(test)]
pub mod testing;
