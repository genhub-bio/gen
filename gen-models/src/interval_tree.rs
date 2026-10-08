use std::sync::Arc;

use gen_core::NodeIntervalBlock;
use intervaltree::IntervalTree;

use crate::db::GraphConnection;

/// Provides runtime interval-tree caching and model-specific database loading.
pub trait IntervalTreeSource {
    /// The error returned while loading or building the interval tree.
    type Error;

    /// Returns the model's attached runtime tree, if present.
    fn cached_interval_tree(&self) -> Option<&Arc<IntervalTree<i64, NodeIntervalBlock>>>;

    /// Replaces the model's attached runtime tree.
    fn set_cached_interval_tree(
        &mut self,
        interval_tree: Option<Arc<IntervalTree<i64, NodeIntervalBlock>>>,
    );

    /// Loads the tree from the connected graph database without consulting the runtime cache.
    fn load_interval_tree(
        &self,
        conn: &GraphConnection,
    ) -> Result<Arc<IntervalTree<i64, NodeIntervalBlock>>, Self::Error>;

    /// Returns the attached tree or loads it from the connected graph database.
    fn intervaltree(
        &self,
        conn: &GraphConnection,
    ) -> Result<Arc<IntervalTree<i64, NodeIntervalBlock>>, Self::Error> {
        if let Some(interval_tree) = self.cached_interval_tree() {
            return Ok(Arc::clone(interval_tree));
        }

        self.load_interval_tree(conn)
    }
}
