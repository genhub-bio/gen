use gen_core::traits::Capnp;
use rusqlite::{Result as SQLResult, Row, params, types::Value as SQLValue};
use serde::{Deserialize, Serialize};

use crate::{
    Direction, ModelSelect, db::GraphConnection, gen_models_capnp::sample_lineage,
    lineage::SqlLineage, select::SqlFilter, traits::Query,
};

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize, ModelSelect)]
pub struct SampleLineage {
    pub parent_sample_name: String,
    pub child_sample_name: String,
}

impl SampleLineageSelect<'_> {
    pub fn name_contains(self, name: impl Into<String>) -> Self {
        let name = name.into();
        let parent_column = self.column("parent_sample_name");
        let child_column = self.column("child_sample_name");
        self.push_filter(SqlFilter::new(
            format!(
                "(instr(lower({parent_column}), lower(?)) > 0 OR \
                 instr(lower({child_column}), lower(?)) > 0)"
            ),
            vec![SQLValue::from(name.clone()), SQLValue::from(name)],
        ))
    }
}

impl<'a> Capnp<'a> for SampleLineage {
    type Builder = sample_lineage::Builder<'a>;
    type Reader = sample_lineage::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_parent_sample_name(&self.parent_sample_name);
        builder.set_child_sample_name(&self.child_sample_name);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let parent_sample_name = reader
            .get_parent_sample_name()
            .unwrap()
            .to_string()
            .unwrap();
        let child_sample_name = reader.get_child_sample_name().unwrap().to_string().unwrap();

        SampleLineage {
            parent_sample_name,
            child_sample_name,
        }
    }
}

impl Query for SampleLineage {
    type Model = SampleLineage;

    const PRIMARY_KEY: &'static str = "parent_sample_name";
    const TABLE_NAME: &'static str = "sample_lineage";

    fn process_row(row: &Row) -> Self::Model {
        SampleLineage {
            parent_sample_name: row.get(0).unwrap(),
            child_sample_name: row.get(1).unwrap(),
        }
    }
}

impl SqlLineage for SampleLineage {
    type Id = String;

    const CHILD_COLUMN: &'static str = "child_sample_name";
    const CHILD_ID_COLUMN: &'static str = "name";
    const CHILD_TABLE_NAME: &'static str = "samples";
    const PARENT_COLUMN: &'static str = "parent_sample_name";
    const PARENT_ID_COLUMN: &'static str = "name";
    const PARENT_TABLE_NAME: &'static str = "samples";

    fn parent_id(&self) -> &Self::Id {
        &self.parent_sample_name
    }

    fn child_id(&self) -> &Self::Id {
        &self.child_sample_name
    }
}

impl SampleLineage {
    pub fn get_parents(
        conn: &GraphConnection,
        child_sample_name: &str,
        history_ref: Option<&str>,
    ) -> Vec<String> {
        let mut select = SampleLineage::select(conn)
            .child_sample_name(child_sample_name)
            .order_by(SampleLineageSelect::ParentSampleName, Direction::Asc);
        if let Some(history_ref) = history_ref {
            select = select.with_ref(history_ref);
        }
        select
            .load()
            .into_iter()
            .map(|lineage| lineage.parent_sample_name)
            .collect()
    }

    pub fn get_children(
        conn: &GraphConnection,
        parent_sample_name: &str,
        history_ref: Option<&str>,
    ) -> Vec<String> {
        let mut select = SampleLineage::select(conn)
            .parent_sample_name(parent_sample_name)
            .order_by(SampleLineageSelect::ChildSampleName, Direction::Asc);
        if let Some(history_ref) = history_ref {
            select = select.with_ref(history_ref);
        }
        select
            .load()
            .into_iter()
            .map(|lineage| lineage.child_sample_name)
            .collect()
    }

    pub fn create(
        conn: &GraphConnection,
        parent_sample_name: &str,
        child_sample_name: &str,
    ) -> SQLResult<Self> {
        let query = "INSERT INTO sample_lineage (parent_sample_name, child_sample_name)
            VALUES (?1, ?2)
            ON CONFLICT(parent_sample_name, child_sample_name) DO NOTHING;";
        let mut stmt = conn.prepare(query).unwrap();
        stmt.execute(params![parent_sample_name, child_sample_name])?;

        Ok(SampleLineage {
            parent_sample_name: parent_sample_name.to_string(),
            child_sample_name: child_sample_name.to_string(),
        })
    }

    pub fn delete(
        conn: &GraphConnection,
        parent_sample_name: &str,
        child_sample_name: &str,
    ) -> SQLResult<()> {
        let query =
            "DELETE FROM sample_lineage WHERE parent_sample_name = ?1 AND child_sample_name = ?2;";
        let mut stmt = conn.prepare(query).unwrap();
        stmt.execute(params![parent_sample_name, child_sample_name])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;

    use super::*;
    use crate::{
        history::dolt::commit_all,
        lineage::SqlLineage,
        sample::{NewSample, Sample},
        test_helpers::get_connection,
    };

    #[test]
    fn test_capnp_serialization() {
        let lineage = SampleLineage {
            parent_sample_name: "parent".to_string(),
            child_sample_name: "child".to_string(),
        };

        let mut message = TypedBuilder::<sample_lineage::Owned>::new_default();
        let mut root = message.init_root();
        lineage.write_capnp(&mut root);

        let deserialized = SampleLineage::read_capnp(root.into_reader());
        assert_eq!(lineage, deserialized);
    }

    #[test]
    fn test_lineage_queries() {
        let conn = get_connection(None).unwrap();

        for sample in ["root", "left", "right", "leaf", "sibling"] {
            Sample::get_or_create(
                &conn,
                NewSample {
                    name: sample,
                    ..Default::default()
                },
            )
            .unwrap();
        }

        SampleLineage::create(&conn, "root", "left").unwrap();
        SampleLineage::create(&conn, "root", "right").unwrap();
        SampleLineage::create(&conn, "left", "leaf").unwrap();
        SampleLineage::create(&conn, "right", "leaf").unwrap();
        SampleLineage::create(&conn, "right", "sibling").unwrap();

        let ancestors = SampleLineage::get_ancestors(&conn, &"leaf".to_string(), None, None);
        assert_eq!(ancestors, vec!["left", "right", "root"]);
        assert_eq!(
            SampleLineage::get_ancestors(&conn, &"leaf".to_string(), Some(1), None),
            vec!["left", "right"]
        );
        assert_eq!(
            SampleLineage::get_ancestors(&conn, &"leaf".to_string(), Some(0), None),
            Vec::<String>::new()
        );

        assert_eq!(
            SampleLineage::get_parents(&conn, "leaf", None),
            vec!["left".to_string(), "right".to_string()]
        );
        assert_eq!(
            SampleLineage::get_children(&conn, "right", None),
            vec!["leaf".to_string(), "sibling".to_string()]
        );

        let descendants = SampleLineage::get_descendants(&conn, &"root".to_string(), None);
        assert_eq!(descendants, vec!["left", "right", "leaf", "sibling"]);
        assert_eq!(
            SampleLineage::get_descendants(&conn, &"root".to_string(), Some(1)),
            vec!["left", "right"]
        );
        assert_eq!(
            SampleLineage::get_descendants(&conn, &"root".to_string(), Some(0)),
            Vec::<String>::new()
        );

        let mut graph = SampleLineage::get_graph(&conn);
        graph.sort_by(|left, right| {
            left.parent_sample_name
                .cmp(&right.parent_sample_name)
                .then(left.child_sample_name.cmp(&right.child_sample_name))
        });
        assert_eq!(
            graph,
            vec![
                SampleLineage {
                    parent_sample_name: "left".to_string(),
                    child_sample_name: "leaf".to_string(),
                },
                SampleLineage {
                    parent_sample_name: "right".to_string(),
                    child_sample_name: "leaf".to_string(),
                },
                SampleLineage {
                    parent_sample_name: "right".to_string(),
                    child_sample_name: "sibling".to_string(),
                },
                SampleLineage {
                    parent_sample_name: "root".to_string(),
                    child_sample_name: "left".to_string(),
                },
                SampleLineage {
                    parent_sample_name: "root".to_string(),
                    child_sample_name: "right".to_string(),
                },
            ]
        );

        let path =
            SampleLineage::get_path_between(&conn, &"leaf".to_string(), &"sibling".to_string());
        assert_eq!(path, vec!["leaf", "right", "sibling"]);

        let edges = SampleLineage::get_path_edges_between(
            &conn,
            &"leaf".to_string(),
            &"sibling".to_string(),
        );
        assert_eq!(
            edges,
            vec![
                SampleLineage {
                    parent_sample_name: "right".to_string(),
                    child_sample_name: "leaf".to_string(),
                },
                SampleLineage {
                    parent_sample_name: "right".to_string(),
                    child_sample_name: "sibling".to_string(),
                },
            ]
        );
    }

    #[test]
    fn test_lineage_depth_limit() {
        let conn = get_connection(None).unwrap();

        for sample in ["root", "left", "right", "leaf", "sibling"] {
            Sample::get_or_create(
                &conn,
                NewSample {
                    name: sample,
                    ..Default::default()
                },
            )
            .unwrap();
        }

        SampleLineage::create(&conn, "root", "left").unwrap();
        SampleLineage::create(&conn, "root", "right").unwrap();
        SampleLineage::create(&conn, "left", "leaf").unwrap();
        SampleLineage::create(&conn, "right", "leaf").unwrap();
        SampleLineage::create(&conn, "right", "sibling").unwrap();

        assert_eq!(
            SampleLineage::get_ancestors(&conn, &"leaf".to_string(), Some(1), None),
            vec!["left", "right"]
        );
        assert_eq!(
            SampleLineage::get_ancestors(&conn, &"leaf".to_string(), Some(2), None),
            vec!["left", "right", "root"]
        );
        assert_eq!(
            SampleLineage::get_descendants(&conn, &"root".to_string(), Some(1)),
            vec!["left", "right"]
        );
        assert_eq!(
            SampleLineage::get_descendants(&conn, &"root".to_string(), Some(2)),
            vec!["left", "right", "leaf", "sibling"]
        );
    }

    #[test]
    fn test_get_ancestors_respects_history_ref() {
        let conn = get_connection(None).unwrap();
        for sample in ["grand", "parent", "child"] {
            Sample::get_or_create(
                &conn,
                NewSample {
                    name: sample,
                    ..Default::default()
                },
            )
            .unwrap();
        }
        SampleLineage::create(&conn, "parent", "child").unwrap();
        let base = commit_all(&conn, "parent lineage").unwrap();
        SampleLineage::create(&conn, "grand", "parent").unwrap();

        assert_eq!(
            SampleLineage::get_ancestors(&conn, &"child".to_string(), None, None),
            vec!["parent", "grand"]
        );
        assert_eq!(
            SampleLineage::get_ancestors(
                &conn,
                &"child".to_string(),
                None,
                Some(&base.to_string()),
            ),
            vec!["parent"]
        );
        assert_eq!(
            SampleLineage::get_parents(&conn, "parent", None),
            vec!["grand"]
        );
        assert!(SampleLineage::get_parents(&conn, "parent", Some(&base.to_string())).is_empty());
    }

    #[test]
    fn test_self_references_are_rejected() {
        let conn = get_connection(None).unwrap();
        Sample::get_or_create(
            &conn,
            NewSample {
                name: "sample",
                ..Default::default()
            },
        )
        .unwrap();

        let err = SampleLineage::create(&conn, "sample", "sample").unwrap_err();
        assert!(matches!(
            err,
            rusqlite::Error::SqliteFailure(code, _)
                if code.code == rusqlite::ErrorCode::ConstraintViolation
        ));
    }

    #[test]
    fn test_search_name_returns_partial_matches() {
        let conn = get_connection(None).unwrap();

        for sample in [
            "alpha",
            "BarFooBaz",
            "child",
            "foo",
            "plain-parent",
            "plain-child",
            "QuxFood",
            "zzz",
        ] {
            Sample::get_or_create(
                &conn,
                NewSample {
                    name: sample,
                    ..Default::default()
                },
            )
            .unwrap();
        }

        SampleLineage::create(&conn, "alpha", "BarFooBaz").unwrap();
        SampleLineage::create(&conn, "foo", "child").unwrap();
        SampleLineage::create(&conn, "plain-parent", "plain-child").unwrap();
        SampleLineage::create(&conn, "zzz", "QuxFood").unwrap();

        let matches = SampleLineage::select(&conn)
            .name_contains("FoO")
            .order_by(SampleLineageSelect::ParentSampleName, Direction::Asc)
            .order_by(SampleLineageSelect::ChildSampleName, Direction::Asc)
            .load();

        assert_eq!(
            matches,
            vec![
                SampleLineage {
                    parent_sample_name: "alpha".to_string(),
                    child_sample_name: "BarFooBaz".to_string(),
                },
                SampleLineage {
                    parent_sample_name: "foo".to_string(),
                    child_sample_name: "child".to_string(),
                },
                SampleLineage {
                    parent_sample_name: "zzz".to_string(),
                    child_sample_name: "QuxFood".to_string(),
                },
            ]
        );

        let limited_matches = SampleLineage::select(&conn)
            .name_contains("FoO")
            .order_by(SampleLineageSelect::ChildSampleName, Direction::Desc)
            .order_by(SampleLineageSelect::ParentSampleName, Direction::Desc)
            .limit(2)
            .load();

        assert_eq!(
            limited_matches,
            vec![
                SampleLineage {
                    parent_sample_name: "foo".to_string(),
                    child_sample_name: "child".to_string(),
                },
                SampleLineage {
                    parent_sample_name: "zzz".to_string(),
                    child_sample_name: "QuxFood".to_string(),
                },
            ]
        );
    }

    #[test]
    fn test_search_supports_sort_and_pagination() {
        let conn = get_connection(None).unwrap();

        for sample in ["alpha", "beta", "child-a", "child-b", "foo", "zzz"] {
            Sample::get_or_create(
                &conn,
                NewSample {
                    name: sample,
                    ..Default::default()
                },
            )
            .unwrap();
        }

        SampleLineage::create(&conn, "alpha", "child-a").unwrap();
        SampleLineage::create(&conn, "foo", "child-b").unwrap();
        SampleLineage::create(&conn, "zzz", "beta").unwrap();

        let matches = SampleLineage::select(&conn)
            .name_contains("a")
            .order_by(SampleLineageSelect::ChildSampleName, Direction::Desc)
            .order_by(SampleLineageSelect::ParentSampleName, Direction::Desc)
            .limit(2)
            .offset(1)
            .load();

        assert_eq!(
            matches,
            vec![SampleLineage {
                parent_sample_name: "zzz".to_string(),
                child_sample_name: "beta".to_string(),
            }]
        );
    }
}
