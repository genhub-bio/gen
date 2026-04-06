use std::str::FromStr;

use gen_core::HashId;
use rusqlite::{
    Connection, params,
    types::{FromSql, ToSql},
};

use crate::traits::Query;

// This looks a bit redundant with the HashId sql parsing, but is not because the id column can be any type. The macro below
// and these traits provide a generic way to go from hex in sql -> rust type. The traversal code encodes everything as hex
// so ints/strings/blobs/etc. can all be treated identically in traversal.
pub trait LineageId: Clone + Eq + FromSql + ToSql {
    fn decode_hex_token(token: &str) -> Self;
}

fn decode_hex_bytes(token: &str) -> Vec<u8> {
    assert_eq!(token.len() % 2, 0, "hex tokens must have an even length");

    token
        .as_bytes()
        .chunks_exact(2)
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

pub trait SqlLineage: Query<Model = Self> + Sized {
    type Id: LineageId;

    const PARENT_TABLE_NAME: &'static str;
    const PARENT_ID_COLUMN: &'static str;
    const CHILD_TABLE_NAME: &'static str;
    const CHILD_ID_COLUMN: &'static str;
    const PARENT_COLUMN: &'static str;
    const CHILD_COLUMN: &'static str;

    fn parent_id(&self) -> &Self::Id;
    fn child_id(&self) -> &Self::Id;

    fn get_ancestors(
        conn: &Connection,
        child_id: &Self::Id,
        max_depth: Option<usize>,
    ) -> Vec<Self::Id> {
        let max_depth = max_depth.map(|depth| depth as i64);
        let query = format!(
            "WITH RECURSIVE ancestors(id, depth, visited) AS (
                SELECT
                    lineage.{parent_column},
                    1,
                    printf('|%s|', hex(lineage.{parent_column}))
                FROM {table_name} lineage
                WHERE lineage.{child_column} = ?1
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
                AND (?2 IS NULL OR ancestors.depth < ?2)
            ),
            ranked_ancestors(id, depth) AS (
                SELECT id, MIN(depth)
                FROM ancestors
                GROUP BY id
            )
            SELECT parent.{parent_id_column}
            FROM {parent_table_name} parent
            JOIN ranked_ancestors ancestors ON parent.{parent_id_column} = ancestors.id
            WHERE ?2 IS NULL OR ancestors.depth <= ?2
            ORDER BY ancestors.depth, parent.{parent_id_column};",
            table_name = Self::TABLE_NAME,
            parent_column = Self::PARENT_COLUMN,
            child_column = Self::CHILD_COLUMN,
            parent_table_name = Self::PARENT_TABLE_NAME,
            parent_id_column = Self::PARENT_ID_COLUMN,
        );

        let mut stmt = conn.prepare(&query).unwrap();
        stmt.query_map(params![child_id, max_depth], |row| row.get(0))
            .unwrap()
            .map(|value| value.unwrap())
            .collect()
    }

    fn get_descendants(
        conn: &Connection,
        parent_id: &Self::Id,
        max_depth: Option<usize>,
    ) -> Vec<Self::Id> {
        let max_depth = max_depth.map(|depth| depth as i64);
        let query = format!(
            "WITH RECURSIVE descendants(id, depth, visited) AS (
                SELECT
                    lineage.{child_column},
                    1,
                    printf('|%s|', hex(lineage.{child_column}))
                FROM {table_name} lineage
                WHERE lineage.{parent_column} = ?1
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
                AND (?2 IS NULL OR descendants.depth < ?2)
            ),
            ranked_descendants(id, depth) AS (
                SELECT id, MIN(depth)
                FROM descendants
                GROUP BY id
            )
            SELECT child.{child_id_column}
            FROM {child_table_name} child
            JOIN ranked_descendants descendants ON child.{child_id_column} = descendants.id
            WHERE ?2 IS NULL OR descendants.depth <= ?2
            ORDER BY descendants.depth, child.{child_id_column};",
            table_name = Self::TABLE_NAME,
            parent_column = Self::PARENT_COLUMN,
            child_column = Self::CHILD_COLUMN,
            child_table_name = Self::CHILD_TABLE_NAME,
            child_id_column = Self::CHILD_ID_COLUMN,
        );

        let mut stmt = conn.prepare(&query).unwrap();
        stmt.query_map(params![parent_id, max_depth], |row| row.get(0))
            .unwrap()
            .map(|value| value.unwrap())
            .collect()
    }

    fn get_graph(conn: &Connection) -> Vec<Self> {
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
            table_name = Self::TABLE_NAME,
            parent_column = Self::PARENT_COLUMN,
            child_column = Self::CHILD_COLUMN,
        );

        Self::query(conn, &query, [])
    }

    fn get_path_between(
        conn: &Connection,
        source_id: &Self::Id,
        target_id: &Self::Id,
    ) -> Vec<Self::Id> {
        if source_id == target_id {
            return vec![source_id.clone()];
        }

        let query = format!(
            "WITH RECURSIVE traversal(current_id, visited, node_path, depth) AS (
                SELECT
                    ?1,
                    printf('|%s|', hex(?1)),
                    printf('%s', hex(?1)),
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
                JOIN {table_name} lineage
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
            WHERE current_id = ?2
            ORDER BY depth
            LIMIT 1;",
            table_name = Self::TABLE_NAME,
            parent_column = Self::PARENT_COLUMN,
            child_column = Self::CHILD_COLUMN,
        );

        let mut stmt = conn.prepare(&query).unwrap();
        let encoded_path = stmt
            .query_row(params![source_id, target_id], |row| row.get::<_, String>(0))
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
    ) -> Vec<Self> {
        let path = Self::get_path_between(conn, source_id, target_id);
        let mut edges = Vec::new();
        for pair in path.windows(2) {
            let query = format!(
                "SELECT {parent_column}, {child_column}
                FROM {table_name}
                WHERE ({parent_column} = ?1 AND {child_column} = ?2)
                   OR ({parent_column} = ?2 AND {child_column} = ?1)
                LIMIT 1;",
                table_name = Self::TABLE_NAME,
                parent_column = Self::PARENT_COLUMN,
                child_column = Self::CHILD_COLUMN,
            );

            if let Ok(edge) = Self::get(conn, &query, params![&pair[0], &pair[1]]) {
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

    impl Query for NumericLineage {
        type Model = NumericLineage;

        const PRIMARY_KEY: &'static str = "parent_id";
        const TABLE_NAME: &'static str = "numeric_lineage";

        fn process_row(row: &Row) -> Self::Model {
            NumericLineage {
                parent_id: row.get(0).unwrap(),
                child_id: row.get(1).unwrap(),
            }
        }
    }

    impl SqlLineage for NumericLineage {
        type Id = i64;

        const CHILD_COLUMN: &'static str = "child_id";
        const CHILD_ID_COLUMN: &'static str = "id";
        const CHILD_TABLE_NAME: &'static str = "numeric_nodes";
        const PARENT_COLUMN: &'static str = "parent_id";
        const PARENT_ID_COLUMN: &'static str = "id";
        const PARENT_TABLE_NAME: &'static str = "numeric_nodes";

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

    impl Query for HashLineage {
        type Model = HashLineage;

        const PRIMARY_KEY: &'static str = "parent_id";
        const TABLE_NAME: &'static str = "hash_lineage";

        fn process_row(row: &Row) -> Self::Model {
            HashLineage {
                parent_id: row.get(0).unwrap(),
                child_id: row.get(1).unwrap(),
            }
        }
    }

    impl SqlLineage for HashLineage {
        type Id = HashId;

        const CHILD_COLUMN: &'static str = "child_id";
        const CHILD_ID_COLUMN: &'static str = "id";
        const CHILD_TABLE_NAME: &'static str = "hash_nodes";
        const PARENT_COLUMN: &'static str = "parent_id";
        const PARENT_ID_COLUMN: &'static str = "id";
        const PARENT_TABLE_NAME: &'static str = "hash_nodes";

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
    fn test_sql_lineage_queries_with_numeric_ids() {
        let conn = setup_numeric_lineage_connection();

        assert_eq!(
            NumericLineage::get_ancestors(&conn, &4, None),
            vec![3, 2, 1]
        );
        assert_eq!(
            NumericLineage::get_ancestors(&conn, &4, Some(2)),
            vec![3, 2]
        );
        assert_eq!(
            NumericLineage::get_descendants(&conn, &1, None),
            vec![2, 3, 5, 4]
        );
        assert_eq!(
            NumericLineage::get_descendants(&conn, &1, Some(2)),
            vec![2, 3, 5]
        );
        assert_eq!(
            NumericLineage::get_path_between(&conn, &1, &4),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            NumericLineage::get_path_edges_between(&conn, &1, &4),
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

        let mut graph = NumericLineage::get_graph(&conn);
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
    fn test_sql_lineage_path_between_handles_same_and_disconnected_numeric_ids() {
        let conn = setup_numeric_lineage_connection();

        assert_eq!(NumericLineage::get_path_between(&conn, &3, &3), vec![3]);
        assert_eq!(
            NumericLineage::get_path_between(&conn, &1, &6),
            Vec::<i64>::new()
        );
        assert_eq!(
            NumericLineage::get_path_edges_between(&conn, &1, &6),
            Vec::<NumericLineage>::new()
        );
    }

    #[test]
    fn test_sql_lineage_hash_id_paths_decode_hex_tokens() {
        let (conn, root, middle, leaf, other) = setup_hash_lineage_connection();

        assert_eq!(
            HashLineage::get_path_between(&conn, &root, &leaf),
            vec![root, middle, leaf]
        );
        assert_eq!(
            HashLineage::get_path_edges_between(&conn, &root, &leaf),
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
            HashLineage::get_path_between(&conn, &root, &other),
            Vec::<HashId>::new()
        );
    }
}
