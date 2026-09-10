use std::str::FromStr;

use gen_core::HashId;
use rusqlite::{
    Connection, Row,
    types::{FromSql, ToSql},
};

use crate::select::sql_table_name_with_history_ref;

// This looks a bit redundant with the HashId sql parsing, but is not because the id column can be any type. The macro below
// and these traits provide a generic way to go from hex in sql -> rust type. The traversal code encodes everything as hex
// so ints/strings/blobs/etc. can all be treated identically in traversal.
pub trait LineageId: Clone + Eq + FromSql + ToSql {
    fn decode_hex_token(token: &str) -> Self;
}

/// A stable page of lineage IDs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LineagePage<Id> {
    /// IDs returned in the requested order.
    pub ids: Vec<Id>,
    /// Whether another page can be requested with the next offset.
    pub has_more: bool,
}

/// A stable page of descendants ordered by depth and ID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DescendantPage<Id> {
    /// Descendants paired with their minimum depth from the requested parent.
    pub ids: Vec<(usize, Id)>,
    /// Whether another page can be requested with the next offset.
    pub has_more: bool,
}

fn decode_hex_bytes(token: &str) -> Vec<u8> {
    assert_eq!(token.len() % 2, 0, "hex tokens must have an even length");

    token
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let pair = std::str::from_utf8(pair).expect("hex token must be valid ascii");
            u8::from_str_radix(pair, 16).expect("hex token must be valid")
        })
        .collect()
}

impl LineageId for String {
    fn decode_hex_token(token: &str) -> Self {
        String::from_utf8(decode_hex_bytes(token)).expect("lineage token should decode to utf-8")
    }
}

impl LineageId for HashId {
    fn decode_hex_token(token: &str) -> Self {
        let bytes = decode_hex_bytes(token);
        HashId::try_from(bytes.as_slice()).expect("lineage token should decode to a hash id")
    }
}

macro_rules! impl_numeric_lineage_id {
    ($($ty:ty),* $(,)?) => {
        $(
            impl LineageId for $ty {
                fn decode_hex_token(token: &str) -> Self {
                    let value =
                        String::from_utf8(decode_hex_bytes(token)).expect("lineage token should decode to utf-8");
                    <$ty>::from_str(&value).expect("lineage token should decode to a number")
                }
            }
        )*
    };
}

impl_numeric_lineage_id!(i32, i64, u32, u64, usize);

pub trait SqlLineage: Sized {
    type Id: LineageId;

    const TABLE_NAME: &'static str;
    const PARENT_TABLE_NAME: &'static str;
    const PARENT_ID_COLUMN: &'static str;
    const CHILD_TABLE_NAME: &'static str;
    const CHILD_ID_COLUMN: &'static str;
    const PARENT_COLUMN: &'static str;
    const CHILD_COLUMN: &'static str;

    fn parent_id(&self) -> &Self::Id;
    fn child_id(&self) -> &Self::Id;
    fn process_row(row: &Row) -> rusqlite::Result<Self>;

    /// Return root entities that have no incoming lineage edge.
    fn get_roots_page(
        conn: &Connection,
        page_size: usize,
        offset: u32,
        history_ref: Option<&str>,
    ) -> LineagePage<Self::Id> {
        let lineage_table_name =
            sql_table_name_with_history_ref(Self::TABLE_NAME, Some(Self::TABLE_NAME), history_ref);
        let parent_table_name = sql_table_name_with_history_ref(
            Self::PARENT_TABLE_NAME,
            Some(Self::PARENT_TABLE_NAME),
            history_ref,
        );
        let query = format!(
            "SELECT parent.{parent_id_column}
             FROM {parent_table_name} parent
             WHERE NOT EXISTS (
                 SELECT 1
                 FROM {lineage_table_name} lineage
                 WHERE lineage.{child_column} = parent.{parent_id_column}
             )
             ORDER BY parent.{parent_id_column}
             LIMIT :limit OFFSET :offset;",
            parent_id_column = Self::PARENT_ID_COLUMN,
            parent_table_name = parent_table_name,
            lineage_table_name = lineage_table_name,
            child_column = Self::CHILD_COLUMN,
        );
        let page_size = page_size.max(1);
        let fetch_limit = i64::try_from(page_size.saturating_add(1)).unwrap_or(i64::MAX);
        let offset = i64::from(offset);
        let history_ref_param = history_ref.map(str::to_owned);
        let mut query_params: Vec<(&str, &dyn ToSql)> =
            vec![(":limit", &fetch_limit), (":offset", &offset)];
        if let Some(history_ref) = history_ref_param.as_ref() {
            query_params.push((":history_ref", history_ref));
        }
        let mut ids = conn
            .prepare(&query)
            .unwrap()
            .query_map(&query_params[..], |row| row.get(0))
            .unwrap()
            .map(|value| value.unwrap())
            .collect::<Vec<Self::Id>>();
        let has_more = ids.len() > page_size;
        if has_more {
            ids.truncate(page_size);
        }
        LineagePage { ids, has_more }
    }

    /// Return direct children in a stable, offset-paginated order.
    fn get_children_page(
        conn: &Connection,
        parent_id: &Self::Id,
        page_size: usize,
        offset: u32,
        history_ref: Option<&str>,
    ) -> LineagePage<Self::Id> {
        let lineage_table_name =
            sql_table_name_with_history_ref(Self::TABLE_NAME, Some(Self::TABLE_NAME), history_ref);
        let child_table_name = sql_table_name_with_history_ref(
            Self::CHILD_TABLE_NAME,
            Some(Self::CHILD_TABLE_NAME),
            history_ref,
        );
        let query = format!(
            "SELECT child.{child_id_column}
             FROM {child_table_name} child
             WHERE EXISTS (
                 SELECT 1
                 FROM {lineage_table_name} lineage
                 WHERE lineage.{parent_column} = :parent_id
                   AND lineage.{child_column} != :parent_id
                   AND lineage.{child_column} = child.{child_id_column}
             )
             ORDER BY child.{child_id_column}
             LIMIT :limit OFFSET :offset;",
            child_id_column = Self::CHILD_ID_COLUMN,
            child_table_name = child_table_name,
            lineage_table_name = lineage_table_name,
            parent_column = Self::PARENT_COLUMN,
            child_column = Self::CHILD_COLUMN,
        );
        let page_size = page_size.max(1);
        let fetch_limit = i64::try_from(page_size.saturating_add(1)).unwrap_or(i64::MAX);
        let offset = i64::from(offset);
        let history_ref_param = history_ref.map(str::to_owned);
        let mut query_params: Vec<(&str, &dyn ToSql)> = vec![
            (":parent_id", parent_id),
            (":limit", &fetch_limit),
            (":offset", &offset),
        ];
        if let Some(history_ref) = history_ref_param.as_ref() {
            query_params.push((":history_ref", history_ref));
        }
        let mut ids = conn
            .prepare(&query)
            .unwrap()
            .query_map(&query_params[..], |row| row.get(0))
            .unwrap()
            .map(|value| value.unwrap())
            .collect::<Vec<Self::Id>>();
        let has_more = ids.len() > page_size;
        if has_more {
            ids.truncate(page_size);
        }
        LineagePage { ids, has_more }
    }

    /// Return descendants ordered by minimum depth and ID with a global page bound.
    ///
    /// The page limits rows returned to the caller. The recursive traversal still visits the
    /// bounded-depth subgraph so convergent paths can be deduplicated before the offset is
    /// applied; callers that need very wide levels should use `get_children_page`.
    fn get_descendants_page(
        conn: &Connection,
        parent_id: &Self::Id,
        max_depth: Option<usize>,
        page_size: usize,
        offset: u32,
        history_ref: Option<&str>,
    ) -> DescendantPage<Self::Id> {
        let max_depth = max_depth.map(|depth| depth as i64);
        let lineage_table_name =
            sql_table_name_with_history_ref(Self::TABLE_NAME, Some(Self::TABLE_NAME), history_ref);
        let child_table_name = sql_table_name_with_history_ref(
            Self::CHILD_TABLE_NAME,
            Some(Self::CHILD_TABLE_NAME),
            history_ref,
        );
        let query = format!(
            "WITH RECURSIVE descendants(id, depth, visited) AS (
                 SELECT lineage.{child_column}, 1,
                        printf('|%s|%s|', hex(:parent_id), hex(lineage.{child_column}))
                 FROM {lineage_table_name} lineage
                 WHERE lineage.{parent_column} = :parent_id
                   AND lineage.{child_column} != :parent_id
                 UNION ALL
                 SELECT lineage.{child_column}, descendants.depth + 1,
                        descendants.visited || hex(lineage.{child_column}) || '|'
                 FROM {lineage_table_name} lineage
                 JOIN descendants ON lineage.{parent_column} = descendants.id
                 WHERE instr(descendants.visited, printf('|%s|', hex(lineage.{child_column}))) = 0
                   AND (:max_depth IS NULL OR descendants.depth < :max_depth)
             ), ranked_descendants(id, depth) AS (
                 SELECT id, MIN(depth)
                 FROM descendants
                 GROUP BY id
             )
             SELECT ranked_descendants.depth, child.{child_id_column}
             FROM {child_table_name} child
             JOIN ranked_descendants ON child.{child_id_column} = ranked_descendants.id
             WHERE (:max_depth IS NULL OR ranked_descendants.depth <= :max_depth)
             ORDER BY ranked_descendants.depth, child.{child_id_column}
             LIMIT :limit OFFSET :offset;",
            lineage_table_name = lineage_table_name,
            parent_column = Self::PARENT_COLUMN,
            child_column = Self::CHILD_COLUMN,
            child_table_name = child_table_name,
            child_id_column = Self::CHILD_ID_COLUMN,
        );
        let page_size = page_size.max(1);
        let fetch_limit = i64::try_from(page_size.saturating_add(1)).unwrap_or(i64::MAX);
        let offset = i64::from(offset);
        let history_ref_param = history_ref.map(str::to_owned);
        let mut query_params: Vec<(&str, &dyn ToSql)> = vec![
            (":parent_id", parent_id),
            (":max_depth", &max_depth),
            (":limit", &fetch_limit),
            (":offset", &offset),
        ];
        if let Some(history_ref) = history_ref_param.as_ref() {
            query_params.push((":history_ref", history_ref));
        }
        let mut ids = conn
            .prepare(&query)
            .unwrap()
            .query_map(&query_params[..], |row| {
                Ok((row.get::<_, i64>(0)? as usize, row.get(1)?))
            })
            .unwrap()
            .map(|value| value.unwrap())
            .collect::<Vec<(usize, Self::Id)>>();
        let has_more = ids.len() > page_size;
        if has_more {
            ids.truncate(page_size);
        }
        DescendantPage { ids, has_more }
    }

    fn get_ancestors(
        conn: &Connection,
        child_id: &Self::Id,
        max_depth: Option<usize>,
        history_ref: Option<&str>,
    ) -> Vec<Self::Id> {
        let max_depth = max_depth.map(|depth| depth as i64);
        let lineage_table_name =
            sql_table_name_with_history_ref(Self::TABLE_NAME, Some(Self::TABLE_NAME), history_ref);
        let parent_table_name = history_ref.map_or_else(
            || Self::PARENT_TABLE_NAME.to_string(),
            |_| format!("dolt_at_{}(:history_ref)", Self::PARENT_TABLE_NAME),
        );
        let query = format!(
            "WITH RECURSIVE ancestors(id, depth, visited) AS (
                SELECT
                    lineage.{parent_column},
                    1,
                    printf('|%s|', hex(lineage.{parent_column}))
                FROM {table_name} lineage
                WHERE lineage.{child_column} = :child_id
                UNION ALL
                SELECT
                    lineage.{parent_column},
                    ancestors.depth + 1,
                    ancestors.visited || hex(lineage.{parent_column}) || '|'
                FROM {table_name} lineage
                JOIN ancestors ON lineage.{child_column} = ancestors.id
                WHERE instr(
                    ancestors.visited,
                    printf('|%s|', hex(lineage.{parent_column}))
                ) = 0
                AND (:max_depth IS NULL OR ancestors.depth < :max_depth)
            ),
            ranked_ancestors(id, depth) AS (
                SELECT id, MIN(depth)
                FROM ancestors
                GROUP BY id
            )
            SELECT parent.{parent_id_column}
            FROM {parent_table_name} parent
            JOIN ranked_ancestors ancestors ON parent.{parent_id_column} = ancestors.id
            WHERE :max_depth IS NULL OR ancestors.depth <= :max_depth
            ORDER BY ancestors.depth, parent.{parent_id_column};",
            table_name = lineage_table_name,
            parent_column = Self::PARENT_COLUMN,
            child_column = Self::CHILD_COLUMN,
            parent_table_name = parent_table_name,
            parent_id_column = Self::PARENT_ID_COLUMN,
        );

        let mut stmt = conn.prepare(&query).unwrap();
        let mut query_params: Vec<(&str, &dyn ToSql)> =
            vec![(":child_id", child_id), (":max_depth", &max_depth)];
        if let Some(history_ref) = history_ref.as_ref() {
            query_params.push((":history_ref", history_ref));
        }
        stmt.query_map(&query_params[..], |row| row.get(0))
            .unwrap()
            .map(|value| value.unwrap())
            .collect()
    }

    /// Return unique descendants ordered by minimum depth, then child ID.
    /// Both lineage edges and child entities are read at `history_ref` when supplied.
    fn get_descendants(
        conn: &Connection,
        parent_id: &Self::Id,
        max_depth: Option<usize>,
        history_ref: Option<&str>,
    ) -> Vec<Self::Id> {
        let max_depth = max_depth.map(|depth| depth as i64);
        let lineage_table_name =
            sql_table_name_with_history_ref(Self::TABLE_NAME, Some(Self::TABLE_NAME), history_ref);
        let child_table_name = sql_table_name_with_history_ref(
            Self::CHILD_TABLE_NAME,
            Some(Self::CHILD_TABLE_NAME),
            history_ref,
        );
        let query = format!(
            "WITH RECURSIVE descendants(id, depth, visited) AS (
                SELECT
                    lineage.{child_column},
                    1,
                    printf('|%s|%s|', hex(:parent_id), hex(lineage.{child_column}))
                FROM {table_name} lineage
                WHERE lineage.{parent_column} = :parent_id
                  AND lineage.{child_column} != :parent_id
                UNION ALL
                SELECT
                    lineage.{child_column},
                    descendants.depth + 1,
                    descendants.visited || hex(lineage.{child_column}) || '|'
                FROM {table_name} lineage
                JOIN descendants ON lineage.{parent_column} = descendants.id
                WHERE instr(
                    descendants.visited,
                    printf('|%s|', hex(lineage.{child_column}))
                ) = 0
                AND (:max_depth IS NULL OR descendants.depth < :max_depth)
            ),
            ranked_descendants(id, depth) AS (
                SELECT id, MIN(depth)
                FROM descendants
                GROUP BY id
            )
            SELECT child.{child_id_column}
            FROM {child_table_name} child
            JOIN ranked_descendants descendants ON child.{child_id_column} = descendants.id
            WHERE :max_depth IS NULL OR descendants.depth <= :max_depth
            ORDER BY descendants.depth, child.{child_id_column};",
            table_name = lineage_table_name,
            parent_column = Self::PARENT_COLUMN,
            child_column = Self::CHILD_COLUMN,
            child_table_name = child_table_name,
            child_id_column = Self::CHILD_ID_COLUMN,
        );

        let mut stmt = conn.prepare(&query).unwrap();
        let mut query_params: Vec<(&str, &dyn ToSql)> =
            vec![(":parent_id", parent_id), (":max_depth", &max_depth)];
        if let Some(history_ref) = history_ref.as_ref() {
            query_params.push((":history_ref", history_ref));
        }
        stmt.query_map(&query_params[..], |row| row.get(0))
            .unwrap()
            .map(|value| value.unwrap())
            .collect()
    }

    fn get_graph(conn: &Connection, history_ref: Option<&str>) -> Vec<Self> {
        let lineage_table_name =
            sql_table_name_with_history_ref(Self::TABLE_NAME, Some(Self::TABLE_NAME), history_ref);
        let query = format!(
            "WITH RECURSIVE lineage_graph({parent_column}, {child_column}) AS (
                SELECT {parent_column}, {child_column}
                FROM {table_name}
                UNION
                SELECT lineage.{parent_column}, lineage.{child_column}
                FROM {table_name} lineage
                JOIN lineage_graph graph ON lineage.{parent_column} = graph.{child_column}
            )
            SELECT {parent_column}, {child_column}
            FROM lineage_graph;",
            table_name = lineage_table_name,
            parent_column = Self::PARENT_COLUMN,
            child_column = Self::CHILD_COLUMN,
        );

        let mut statement = conn.prepare(&query).unwrap();
        let mut query_params: Vec<(&str, &dyn ToSql)> = Vec::new();
        if let Some(history_ref) = history_ref.as_ref() {
            query_params.push((":history_ref", history_ref));
        }
        statement
            .query_map(&query_params[..], Self::process_row)
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }

    fn get_path_between(
        conn: &Connection,
        source_id: &Self::Id,
        target_id: &Self::Id,
        history_ref: Option<&str>,
    ) -> Vec<Self::Id> {
        if source_id == target_id {
            return vec![source_id.clone()];
        }

        let lineage_table_name =
            sql_table_name_with_history_ref(Self::TABLE_NAME, Some(Self::TABLE_NAME), history_ref);
        let query = format!(
            "WITH RECURSIVE traversal(current_id, visited, node_path, depth) AS (
                SELECT
                    :source_id,
                    printf('|%s|', hex(:source_id)),
                    printf('%s', hex(:source_id)),
                    0
                UNION ALL
                SELECT
                    CASE
                        WHEN lineage.{parent_column} = traversal.current_id THEN lineage.{child_column}
                        ELSE lineage.{parent_column}
                    END,
                    traversal.visited || hex(
                        CASE
                            WHEN lineage.{parent_column} = traversal.current_id THEN lineage.{child_column}
                            ELSE lineage.{parent_column}
                        END
                    ) || '|',
                    traversal.node_path || ',' || hex(
                        CASE
                            WHEN lineage.{parent_column} = traversal.current_id THEN lineage.{child_column}
                            ELSE lineage.{parent_column}
                        END
                    ),
                    traversal.depth + 1
                FROM traversal
                JOIN {lineage_table_name} lineage
                    ON lineage.{parent_column} = traversal.current_id
                    OR lineage.{child_column} = traversal.current_id
                WHERE instr(
                    traversal.visited,
                    printf(
                        '|%s|',
                        hex(
                            CASE
                                WHEN lineage.{parent_column} = traversal.current_id THEN lineage.{child_column}
                                ELSE lineage.{parent_column}
                            END
                        )
                    )
                ) = 0
            )
            SELECT node_path
            FROM traversal
            WHERE current_id = :target_id
            ORDER BY depth
            LIMIT 1;",
            lineage_table_name = lineage_table_name,
            parent_column = Self::PARENT_COLUMN,
            child_column = Self::CHILD_COLUMN,
        );

        let mut stmt = conn.prepare(&query).unwrap();
        let mut query_params: Vec<(&str, &dyn ToSql)> =
            vec![(":source_id", source_id), (":target_id", target_id)];
        if let Some(history_ref) = history_ref.as_ref() {
            query_params.push((":history_ref", history_ref));
        }
        let encoded_path = stmt
            .query_row(&query_params[..], |row| row.get::<_, String>(0))
            .ok();

        encoded_path
            .map(|path| {
                path.split(',')
                    .filter(|token| !token.is_empty())
                    .map(Self::Id::decode_hex_token)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn get_path_edges_between(
        conn: &Connection,
        source_id: &Self::Id,
        target_id: &Self::Id,
        history_ref: Option<&str>,
    ) -> Vec<Self> {
        let path = Self::get_path_between(conn, source_id, target_id, history_ref);
        let lineage_table_name =
            sql_table_name_with_history_ref(Self::TABLE_NAME, Some(Self::TABLE_NAME), history_ref);
        let mut edges = Vec::new();
        for pair in path.windows(2) {
            let query = format!(
                "SELECT {parent_column}, {child_column}
                FROM {lineage_table_name}
                WHERE ({parent_column} = :source_id AND {child_column} = :target_id)
                   OR ({parent_column} = :target_id AND {child_column} = :source_id)
                LIMIT 1;",
                lineage_table_name = lineage_table_name,
                parent_column = Self::PARENT_COLUMN,
                child_column = Self::CHILD_COLUMN,
            );

            let mut query_params: Vec<(&str, &dyn ToSql)> =
                vec![(":source_id", &pair[0]), (":target_id", &pair[1])];
            if let Some(history_ref) = history_ref.as_ref() {
                query_params.push((":history_ref", history_ref));
            }
            let edge = conn.query_row(&query, &query_params[..], Self::process_row);
            if let Ok(edge) = edge {
                edges.push(edge);
            }
        }
        edges
    }
}

#[cfg(test)]
mod tests {
    use rusqlite::{Connection, Row, params};

    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct NumericLineage {
        parent_id: i64,
        child_id: i64,
    }

    impl SqlLineage for NumericLineage {
        type Id = i64;

        const CHILD_COLUMN: &'static str = "child_id";
        const CHILD_ID_COLUMN: &'static str = "id";
        const CHILD_TABLE_NAME: &'static str = "numeric_nodes";
        const PARENT_COLUMN: &'static str = "parent_id";
        const PARENT_ID_COLUMN: &'static str = "id";
        const PARENT_TABLE_NAME: &'static str = "numeric_nodes";
        const TABLE_NAME: &'static str = "numeric_lineage";

        fn process_row(row: &Row) -> rusqlite::Result<Self> {
            Ok(NumericLineage {
                parent_id: row.get(0)?,
                child_id: row.get(1)?,
            })
        }

        fn parent_id(&self) -> &Self::Id {
            &self.parent_id
        }

        fn child_id(&self) -> &Self::Id {
            &self.child_id
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct HashLineage {
        parent_id: HashId,
        child_id: HashId,
    }

    impl SqlLineage for HashLineage {
        type Id = HashId;

        const CHILD_COLUMN: &'static str = "child_id";
        const CHILD_ID_COLUMN: &'static str = "id";
        const CHILD_TABLE_NAME: &'static str = "hash_nodes";
        const PARENT_COLUMN: &'static str = "parent_id";
        const PARENT_ID_COLUMN: &'static str = "id";
        const PARENT_TABLE_NAME: &'static str = "hash_nodes";
        const TABLE_NAME: &'static str = "hash_lineage";

        fn process_row(row: &Row) -> rusqlite::Result<Self> {
            Ok(HashLineage {
                parent_id: row.get(0)?,
                child_id: row.get(1)?,
            })
        }

        fn parent_id(&self) -> &Self::Id {
            &self.parent_id
        }

        fn child_id(&self) -> &Self::Id {
            &self.child_id
        }
    }

    fn setup_numeric_lineage_connection() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "
            CREATE TABLE numeric_nodes (id INTEGER PRIMARY KEY);
            CREATE TABLE numeric_lineage (
                parent_id INTEGER NOT NULL,
                child_id INTEGER NOT NULL
            );
            ",
        )
        .unwrap();

        for id in [1_i64, 2, 3, 4, 5, 6] {
            conn.execute("INSERT INTO numeric_nodes (id) VALUES (?1);", params![id])
                .unwrap();
        }

        for (parent_id, child_id) in [(1_i64, 2_i64), (2, 3), (3, 4), (2, 5)] {
            conn.execute(
                "INSERT INTO numeric_lineage (parent_id, child_id) VALUES (?1, ?2);",
                params![parent_id, child_id],
            )
            .unwrap();
        }

        conn
    }

    fn setup_hash_lineage_connection() -> (Connection, HashId, HashId, HashId, HashId) {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "
            CREATE TABLE hash_nodes (id BLOB PRIMARY KEY);
            CREATE TABLE hash_lineage (
                parent_id BLOB NOT NULL,
                child_id BLOB NOT NULL
            );
            ",
        )
        .unwrap();

        let root = HashId::convert_str("lineage-root");
        let middle = HashId::convert_str("lineage-middle");
        let leaf = HashId::convert_str("lineage-leaf");
        let other = HashId::convert_str("lineage-other");

        for id in [root, middle, leaf, other] {
            conn.execute("INSERT INTO hash_nodes (id) VALUES (?1);", params![id])
                .unwrap();
        }

        for (parent_id, child_id) in [(root, middle), (middle, leaf)] {
            conn.execute(
                "INSERT INTO hash_lineage (parent_id, child_id) VALUES (?1, ?2);",
                params![parent_id, child_id],
            )
            .unwrap();
        }

        (conn, root, middle, leaf, other)
    }

    #[test]
    fn test_descendants_deduplicate_shared_paths_and_terminate_cycles() {
        // The fixture starts with 1 -> 2 -> 3 -> 4 and 2 -> 5. Add 5 -> 3
        // as a shared route to 3 and 4 -> 2 as the cycle 2 -> 3 -> 4 -> 2.

        let conn = setup_numeric_lineage_connection();
        // This adds a cycle between 4->2 and a shared path
        conn.execute_batch(
            "INSERT INTO numeric_lineage (parent_id, child_id) VALUES (5, 3), (4, 2);",
        )
        .expect("should add shared descendant and cycle");
        assert_eq!(
            NumericLineage::get_descendants(&conn, &1, None, None),
            vec![2, 3, 5, 4]
        );
        // A depth-two limit keeps direct children and grandchildren, excluding node 4 at depth three.
        assert_eq!(
            NumericLineage::get_descendants(&conn, &1, Some(2), None),
            vec![2, 3, 5]
        );
    }

    #[test]
    fn test_sql_lineage_queries_with_numeric_ids() {
        let conn = setup_numeric_lineage_connection();

        assert_eq!(
            NumericLineage::get_ancestors(&conn, &4, None, None),
            vec![3, 2, 1]
        );
        assert_eq!(
            NumericLineage::get_ancestors(&conn, &4, Some(2), None),
            vec![3, 2]
        );
        assert_eq!(
            NumericLineage::get_descendants(&conn, &1, None, None),
            vec![2, 3, 5, 4]
        );
        assert_eq!(
            NumericLineage::get_descendants(&conn, &1, Some(2), None),
            vec![2, 3, 5]
        );
        assert_eq!(
            NumericLineage::get_path_between(&conn, &1, &4, None),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            NumericLineage::get_path_edges_between(&conn, &1, &4, None),
            vec![
                NumericLineage {
                    parent_id: 1,
                    child_id: 2,
                },
                NumericLineage {
                    parent_id: 2,
                    child_id: 3,
                },
                NumericLineage {
                    parent_id: 3,
                    child_id: 4,
                },
            ]
        );

        let mut graph = NumericLineage::get_graph(&conn, None);
        graph.sort_by(|left, right| {
            left.parent_id
                .cmp(&right.parent_id)
                .then(left.child_id.cmp(&right.child_id))
        });
        assert_eq!(
            graph,
            vec![
                NumericLineage {
                    parent_id: 1,
                    child_id: 2,
                },
                NumericLineage {
                    parent_id: 2,
                    child_id: 3,
                },
                NumericLineage {
                    parent_id: 2,
                    child_id: 5,
                },
                NumericLineage {
                    parent_id: 3,
                    child_id: 4,
                },
            ]
        );
    }

    #[test]
    fn test_paged_roots_children_and_descendants_are_stable() {
        let conn = setup_numeric_lineage_connection();
        conn.execute_batch(
            "INSERT INTO numeric_nodes (id) VALUES (7), (8), (9), (10), (11), (12);
             INSERT INTO numeric_lineage (parent_id, child_id) VALUES
                 (1, 7), (1, 8), (1, 9);",
        )
        .expect("should add paginated lineage fixtures");

        let first_roots = NumericLineage::get_roots_page(&conn, 2, 0, None);
        assert_eq!(first_roots.ids, vec![1, 6]);
        assert!(first_roots.has_more);
        let second_roots = NumericLineage::get_roots_page(&conn, 2, 2, None);
        assert_eq!(second_roots.ids, vec![10, 11]);
        assert!(second_roots.has_more);
        let third_roots = NumericLineage::get_roots_page(&conn, 2, 4, None);
        assert_eq!(third_roots.ids, vec![12]);
        assert!(!third_roots.has_more);

        let first_children = NumericLineage::get_children_page(&conn, &1, 2, 0, None);
        assert_eq!(first_children.ids, vec![2, 7]);
        assert!(first_children.has_more);

        let second_children = NumericLineage::get_children_page(&conn, &1, 2, 2, None);
        assert_eq!(second_children.ids, vec![8, 9]);
        assert!(!second_children.has_more);

        let first_descendants =
            NumericLineage::get_descendants_page(&conn, &1, Some(2), 2, 0, None);
        assert_eq!(first_descendants.ids, vec![(1, 2), (1, 7)]);
        assert!(first_descendants.has_more);
        let second_descendants =
            NumericLineage::get_descendants_page(&conn, &1, Some(2), 2, 2, None);
        assert_eq!(second_descendants.ids, vec![(1, 8), (1, 9)]);
        assert!(second_descendants.has_more);
        let third_descendants =
            NumericLineage::get_descendants_page(&conn, &1, Some(2), 2, 4, None);
        assert_eq!(third_descendants.ids, vec![(2, 3), (2, 5)]);
        assert!(!third_descendants.has_more);

        conn.execute_batch(
            "INSERT INTO numeric_lineage (parent_id, child_id) VALUES (5, 3), (4, 2), (5, 1);",
        )
        .expect("should add convergent and cyclic lineage fixtures");
        let page = NumericLineage::get_descendants_page(&conn, &1, Some(5), 20, 0, None);
        assert_eq!(
            page.ids,
            vec![(1, 2), (1, 7), (1, 8), (1, 9), (2, 3), (2, 5), (3, 4)]
        );
        assert!(!page.has_more);
        assert!(!page.ids.iter().any(|(_, id)| *id == 1));
        assert_eq!(
            NumericLineage::get_descendants(&conn, &1, Some(5), None),
            vec![2, 7, 8, 9, 3, 5, 4]
        );
    }

    #[test]
    fn test_sql_lineage_path_between_handles_same_and_disconnected_numeric_ids() {
        let conn = setup_numeric_lineage_connection();

        assert_eq!(
            NumericLineage::get_path_between(&conn, &3, &3, None),
            vec![3]
        );
        assert_eq!(
            NumericLineage::get_path_between(&conn, &1, &6, None),
            Vec::<i64>::new()
        );
        assert_eq!(
            NumericLineage::get_path_edges_between(&conn, &1, &6, None),
            Vec::<NumericLineage>::new()
        );
    }

    #[test]
    fn test_sql_lineage_hash_id_paths_decode_hex_tokens() {
        let (conn, root, middle, leaf, other) = setup_hash_lineage_connection();

        assert_eq!(
            HashLineage::get_path_between(&conn, &root, &leaf, None),
            vec![root, middle, leaf]
        );
        assert_eq!(
            HashLineage::get_path_edges_between(&conn, &root, &leaf, None),
            vec![
                HashLineage {
                    parent_id: root,
                    child_id: middle,
                },
                HashLineage {
                    parent_id: middle,
                    child_id: leaf,
                },
            ]
        );
        assert_eq!(
            HashLineage::get_path_between(&conn, &root, &other, None),
            Vec::<HashId>::new()
        );
    }
}
