use gen_core::HashId;
use rusqlite::{Row, params};

use crate::{block_group::BlockGroup, db::GraphConnection, lineage::SqlLineage, traits::Query};

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
        BlockGroupLineage::query(
            conn,
            "SELECT parent_block_group_id, id
             FROM block_groups
             WHERE id = ?1 AND parent_block_group_id IS NOT NULL;",
            params![child_block_group_id],
        )
        .into_iter()
        .map(|lineage| lineage.parent_block_group_id)
        .collect()
    }

    pub fn get_children(conn: &GraphConnection, parent_block_group_id: &HashId) -> Vec<HashId> {
        BlockGroupLineage::query(
            conn,
            "SELECT parent_block_group_id, id
             FROM block_groups
             WHERE parent_block_group_id = ?1
             ORDER BY created_on, id;",
            params![parent_block_group_id],
        )
        .into_iter()
        .map(|lineage| lineage.child_block_group_id)
        .collect()
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
        let ancestor_ids = BlockGroupLineage::get_ancestors(conn, child_block_group_id, max_depth);
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
