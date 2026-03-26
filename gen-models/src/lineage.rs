use std::str::FromStr;

use gen_core::HashId;
use rusqlite::{
    Connection, params,
    types::{FromSql, ToSql},
};

use crate::traits::Query;

pub trait LineageId: Clone + Eq + FromSql + ToSql {
    fn decode_hex_token(token: &str) -> Self;
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
