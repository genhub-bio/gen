use gen_core::traits::Capnp;
use rusqlite::{Result as SQLResult, Row, params, types::Value as SQLValue};
use serde::{Deserialize, Serialize};

use crate::{
    Direction, ModelSelect,
    db::GraphConnection,
    gen_models_capnp::sample_lineage,
    lineage::SqlLineage,
    select::{ModelSelectRow, SqlFilter},
};

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize, ModelSelect)]
#[model_select(
    table = "sample_lineage",
    default_sort(parent_sample_name = "asc", child_sample_name = "asc")
)]
pub struct SampleLineage {
    #[model_select(primary_key)]
    pub parent_sample_name: String,
    #[model_select(primary_key)]
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

impl SqlLineage for SampleLineage {
    type Id = String;

    const CHILD_COLUMN: &'static str = "child_sample_name";
    const CHILD_ID_COLUMN: &'static str = "name";
    const CHILD_TABLE_NAME: &'static str = "samples";
    const PARENT_COLUMN: &'static str = "parent_sample_name";
    const PARENT_ID_COLUMN: &'static str = "name";
    const PARENT_TABLE_NAME: &'static str = "samples";
    const TABLE_NAME: &'static str = "sample_lineage";

    fn parent_id(&self) -> &Self::Id {
        &self.parent_sample_name
    }

    fn child_id(&self) -> &Self::Id {
        &self.child_sample_name
    }

    fn process_row(row: &Row) -> rusqlite::Result<Self> {
        <Self as ModelSelectRow>::process_row(row)
    }
}

impl SampleLineage {
    pub fn get_parents(
        conn: &GraphConnection,
        child_sample_name: &str,
        history_ref: Option<&str>,
    ) -> Vec<String> {
        let select = SampleLineage::select(conn)
            .child_sample_name(child_sample_name)
            .order_by(SampleLineageSelect::ParentSampleName, Direction::Asc)
            .with_ref(history_ref);
        select
            .load()
            .expect("should load parent sample lineage")
            .into_iter()
            .map(|lineage| lineage.parent_sample_name)
            .collect()
    }

    pub fn get_children(
        conn: &GraphConnection,
        parent_sample_name: &str,
        history_ref: Option<&str>,
    ) -> Vec<String> {
        let select = SampleLineage::select(conn)
            .parent_sample_name(parent_sample_name)
            .order_by(SampleLineageSelect::ChildSampleName, Direction::Asc)
            .with_ref(history_ref);
        select
            .load()
            .expect("should load child sample lineage")
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
        history::dolt::{checkout, commit_all, create_branch},
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

        let descendants = SampleLineage::get_descendants(&conn, &"root".to_string(), None, None);
        assert_eq!(descendants, vec!["left", "right", "leaf", "sibling"]);
        assert_eq!(
            SampleLineage::get_descendants(&conn, &"root".to_string(), Some(1), None),
            vec!["left", "right"]
        );
        assert_eq!(
            SampleLineage::get_descendants(&conn, &"root".to_string(), Some(0), None),
            Vec::<String>::new()
        );

        let mut graph = SampleLineage::get_graph(&conn, None);
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

        let path = SampleLineage::get_path_between(
            &conn,
            &"leaf".to_string(),
            &"sibling".to_string(),
            None,
        );
        assert_eq!(path, vec!["leaf", "right", "sibling"]);

        let edges = SampleLineage::get_path_edges_between(
            &conn,
            &"leaf".to_string(),
            &"sibling".to_string(),
            None,
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
            SampleLineage::get_descendants(&conn, &"root".to_string(), Some(1), None),
            vec!["left", "right"]
        );
        assert_eq!(
            SampleLineage::get_descendants(&conn, &"root".to_string(), Some(2), None),
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
    fn test_lineage_queries_respect_history_and_branches() {
        // The base snapshot is root -> child -> leaf.
        // The alternate branch deletes child -> leaf and leaf, then adds root -> later.
        // The queries compare current state with commit and branch refs before and after checkout.
        let conn = get_connection(None).unwrap();
        for name in ["root", "child", "leaf"] {
            Sample::get_or_create(
                &conn,
                NewSample {
                    name,
                    ..Default::default()
                },
            )
            .expect("should create sample");
        }
        SampleLineage::create(&conn, "root", "child").expect("should create lineage");
        SampleLineage::create(&conn, "child", "leaf").expect("should create lineage");
        let base = commit_all(&conn, "base lineage")
            .expect("should commit lineage")
            .to_string();
        create_branch(&conn, "lineage-base").expect("should create branch");
        create_branch(&conn, "lineage-alternate").expect("should create branch");
        checkout(&conn, "lineage-alternate").expect("should checkout branch");
        SampleLineage::delete(&conn, "child", "leaf").expect("should delete lineage");
        Sample::delete_by_name(&conn, "leaf");
        Sample::get_or_create(
            &conn,
            NewSample {
                name: "later",
                ..Default::default()
            },
        )
        .expect("should create later sample");
        SampleLineage::create(&conn, "root", "later").expect("should create later lineage");
        commit_all(&conn, "alternate lineage").expect("should commit alternate lineage");

        let root = "root".to_string();
        assert_eq!(
            SampleLineage::get_descendants(&conn, &root, None, None),
            vec!["child", "later"]
        );
        assert_eq!(
            SampleLineage::get_descendants(&conn, &root, None, Some(&base)),
            vec!["child", "leaf"]
        );
        let historical_roots = SampleLineage::get_roots_page(&conn, 10, 0, Some(&base));
        assert_eq!(historical_roots.ids, vec!["root"]);
        assert!(!historical_roots.has_more);
        let historical_children =
            SampleLineage::get_children_page(&conn, &root, 10, 0, Some(&base));
        assert_eq!(historical_children.ids, vec!["child"]);
        assert!(!historical_children.has_more);
        let historical_descendants =
            SampleLineage::get_descendants_page(&conn, &root, None, 1, 0, Some(&base));
        assert_eq!(historical_descendants.ids, vec!["child".to_string()]);
        assert!(historical_descendants.has_more);
        let historical_descendants =
            SampleLineage::get_descendants_page(&conn, &root, None, 1, 1, Some(&base));
        assert_eq!(historical_descendants.ids, vec!["leaf".to_string()]);
        assert!(!historical_descendants.has_more);
        assert_eq!(
            SampleLineage::get_descendants(&conn, &root, None, Some("lineage-base")),
            vec!["child", "leaf"]
        );
        assert_eq!(
            SampleLineage::get_descendants(&conn, &root, Some(1), Some(&base)),
            vec!["child"]
        );
        assert!(SampleLineage::get_descendants(&conn, &root, Some(0), Some(&base)).is_empty());

        let sort_graph = |graph: &mut Vec<SampleLineage>| {
            graph.sort_by(|left, right| {
                left.parent_sample_name
                    .cmp(&right.parent_sample_name)
                    .then(left.child_sample_name.cmp(&right.child_sample_name))
            });
        };
        let mut current_graph = SampleLineage::get_graph(&conn, None);
        sort_graph(&mut current_graph);
        assert_eq!(
            current_graph,
            vec![
                SampleLineage {
                    parent_sample_name: "root".to_string(),
                    child_sample_name: "child".to_string(),
                },
                SampleLineage {
                    parent_sample_name: "root".to_string(),
                    child_sample_name: "later".to_string(),
                },
            ]
        );
        let mut base_graph = SampleLineage::get_graph(&conn, Some(&base));
        sort_graph(&mut base_graph);
        assert_eq!(
            base_graph,
            vec![
                SampleLineage {
                    parent_sample_name: "child".to_string(),
                    child_sample_name: "leaf".to_string(),
                },
                SampleLineage {
                    parent_sample_name: "root".to_string(),
                    child_sample_name: "child".to_string(),
                },
            ]
        );
        assert_eq!(
            SampleLineage::get_path_between(&conn, &root, &"leaf".to_string(), None),
            Vec::<String>::new()
        );
        assert_eq!(
            SampleLineage::get_path_between(&conn, &root, &"leaf".to_string(), Some(&base)),
            vec!["root", "child", "leaf"]
        );
        assert_eq!(
            SampleLineage::get_path_edges_between(&conn, &root, &"leaf".to_string(), Some(&base)),
            vec![
                SampleLineage {
                    parent_sample_name: "root".to_string(),
                    child_sample_name: "child".to_string(),
                },
                SampleLineage {
                    parent_sample_name: "child".to_string(),
                    child_sample_name: "leaf".to_string(),
                },
            ]
        );
        checkout(&conn, "lineage-base").expect("should checkout base branch");
        assert_eq!(
            SampleLineage::get_descendants(&conn, &root, None, Some("lineage-alternate")),
            vec!["child", "later"]
        );
        let mut alternate_graph = SampleLineage::get_graph(&conn, Some("lineage-alternate"));
        sort_graph(&mut alternate_graph);
        assert_eq!(alternate_graph, current_graph);
        assert_eq!(
            SampleLineage::get_path_between(
                &conn,
                &root,
                &"later".to_string(),
                Some("lineage-alternate"),
            ),
            vec!["root", "later"]
        );
        assert_eq!(
            SampleLineage::get_path_edges_between(
                &conn,
                &root,
                &"later".to_string(),
                Some("lineage-alternate"),
            ),
            vec![SampleLineage {
                parent_sample_name: "root".to_string(),
                child_sample_name: "later".to_string(),
            }]
        );
    }

    #[test]
    fn test_get_descendants_fetches_five_levels_at_history_ref() {
        // The committed fixture is the chain root -> one -> two -> three -> four -> five -> six.
        // The historical reads below check the five-edge cutoff and unlimited traversal.
        let conn = get_connection(None).unwrap();
        let names = ["root", "one", "two", "three", "four", "five", "six"];
        for name in names {
            Sample::get_or_create(
                &conn,
                NewSample {
                    name,
                    ..Default::default()
                },
            )
            .expect("should create sample");
        }
        for pair in names.windows(2) {
            SampleLineage::create(&conn, pair[0], pair[1]).expect("should create lineage");
        }
        let history = commit_all(&conn, "deep lineage")
            .expect("should commit lineage")
            .to_string();
        let root = "root".to_string();
        assert_eq!(
            SampleLineage::get_descendants(&conn, &root, Some(5), Some(&history)),
            vec!["one", "two", "three", "four", "five"]
        );
        assert_eq!(
            SampleLineage::get_descendants(&conn, &root, None, Some(&history)),
            vec!["one", "two", "three", "four", "five", "six"]
        );
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
            .load()
            .expect("should load matching sample lineage");

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
            .load()
            .expect("should load limited sample lineage");

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
            .load()
            .expect("should load paginated sample lineage");

        assert_eq!(
            matches,
            vec![SampleLineage {
                parent_sample_name: "zzz".to_string(),
                child_sample_name: "beta".to_string(),
            }]
        );
    }
}
