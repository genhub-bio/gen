use gen_core::HashId;
use rusqlite::Row;

use crate::{
    Direction,
    block_group::{BlockGroup, BlockGroupSelect},
    db::GraphConnection,
    lineage::SqlLineage,
    traits::Query,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockGroupLineage {
    pub parent_block_group_id: HashId,
    pub child_block_group_id: HashId,
}

impl Query for BlockGroupLineage {
    type Model = BlockGroupLineage;

    const PRIMARY_KEY: &'static str = "id";
    const TABLE_NAME: &'static str = "block_groups";

    fn process_row(row: &Row) -> Self::Model {
        BlockGroupLineage {
            parent_block_group_id: row.get(0).unwrap(),
            child_block_group_id: row.get(1).unwrap(),
        }
    }
}

impl SqlLineage for BlockGroupLineage {
    type Id = HashId;

    const CHILD_COLUMN: &'static str = "id";
    const CHILD_ID_COLUMN: &'static str = "id";
    const CHILD_TABLE_NAME: &'static str = "block_groups";
    const PARENT_COLUMN: &'static str = "parent_block_group_id";
    const PARENT_ID_COLUMN: &'static str = "id";
    const PARENT_TABLE_NAME: &'static str = "block_groups";

    fn parent_id(&self) -> &Self::Id {
        &self.parent_block_group_id
    }

    fn child_id(&self) -> &Self::Id {
        &self.child_block_group_id
    }
}

impl BlockGroupLineage {
    pub fn get_parents(conn: &GraphConnection, child_block_group_id: &HashId) -> Vec<HashId> {
        BlockGroup::select(conn)
            .id(*child_block_group_id)
            .only(BlockGroupSelect::ParentBlockGroupId)
            .load()
            .expect("should load parent block group ids")
            .into_iter()
            .flatten()
            .collect()
    }

    pub fn get_children(conn: &GraphConnection, parent_block_group_id: &HashId) -> Vec<HashId> {
        BlockGroup::select(conn)
            .parent_block_group_id(*parent_block_group_id)
            .order_by(BlockGroupSelect::CreatedOn, Direction::Asc)
            .order_by(BlockGroupSelect::Id, Direction::Asc)
            .only(BlockGroupSelect::Id)
            .load()
            .expect("should load child block group ids")
    }

    pub fn get_parent_block_groups(
        conn: &GraphConnection,
        child_block_group_id: &HashId,
    ) -> Vec<BlockGroup> {
        let parent_ids = BlockGroupLineage::get_parents(conn, child_block_group_id);
        BlockGroup::query_by_ids(conn, &parent_ids, None)
    }

    pub fn get_ancestor_block_groups(
        conn: &GraphConnection,
        child_block_group_id: &HashId,
        max_depth: Option<usize>,
    ) -> Vec<BlockGroup> {
        let ancestor_ids =
            BlockGroupLineage::get_ancestors(conn, child_block_group_id, max_depth, None);
        BlockGroup::query_by_ids(conn, &ancestor_ids, None)
    }

    pub fn get_descendant_block_groups(
        conn: &GraphConnection,
        parent_block_group_id: &HashId,
        max_depth: Option<usize>,
    ) -> Vec<BlockGroup> {
        let descendant_ids =
            BlockGroupLineage::get_descendants(conn, parent_block_group_id, max_depth);
        BlockGroup::query_by_ids(conn, &descendant_ids, None)
    }
}
