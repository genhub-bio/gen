use gen_core::HashId;
use gen_models::{
    annotations::AnnotationGroup, block_group::BlockGroup, db::GraphConnection,
    lineage::SqlLineage, sample_lineage::SampleLineage, traits::Query,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationGroupEntry {
    pub id: String,
    pub name: String,
    pub sample_name: String,
    pub source_block_group_id: HashId,
    pub origin: AnnotationGroupOrigin,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnnotationGroupOrigin {
    CurrentSample,
    ParentSample,
    AncestorSample,
}

pub fn load_annotation_group_entries(
    conn: &GraphConnection,
    block_group: &BlockGroup,
) -> Vec<AnnotationGroupEntry> {
    let mut entries = Vec::new();
    let parent_sample_name = block_group
        .parent_block_group_id
        .map(|parent_id| BlockGroup::get_by_id(conn, &parent_id).unwrap().sample_name);
    let ancestor_samples = SampleLineage::get_ancestors(conn, &block_group.sample_name, None);
    let sample_order = std::iter::once(block_group.sample_name.clone())
        .chain(ancestor_samples.iter().cloned())
        .enumerate()
        .map(|(idx, sample_name)| (sample_name, idx))
        .collect::<std::collections::HashMap<_, _>>();

    entries.extend(
        AnnotationGroup::query_by_sample(conn, &block_group.sample_name)
            .into_iter()
            .map(|group| AnnotationGroupEntry {
                id: format!("{}::{}", block_group.sample_name, group.name),
                name: group.name,
                sample_name: block_group.sample_name.clone(),
                source_block_group_id: block_group.id,
                origin: AnnotationGroupOrigin::CurrentSample,
            }),
    );

    for ancestor_sample in ancestor_samples {
        let origin = if parent_sample_name.as_deref() == Some(ancestor_sample.as_str()) {
            AnnotationGroupOrigin::ParentSample
        } else {
            AnnotationGroupOrigin::AncestorSample
        };
        let ancestor_block_groups = BlockGroup::query(
            conn,
            "select * from block_groups
             where collection_name = ?1 AND sample_name = ?2 AND name = ?3
             order by created_on, id",
            rusqlite::params![
                block_group.collection_name,
                ancestor_sample,
                block_group.name
            ],
        );

        for ancestor_block_group in ancestor_block_groups {
            entries.extend(
                AnnotationGroup::query_by_sample(conn, &ancestor_block_group.sample_name)
                    .into_iter()
                    .map(|group| AnnotationGroupEntry {
                        id: format!("{}::{}", ancestor_block_group.sample_name, group.name),
                        name: group.name,
                        sample_name: ancestor_block_group.sample_name.clone(),
                        source_block_group_id: ancestor_block_group.id,
                        origin,
                    }),
            );
        }
    }

    entries.sort_by(|left, right| {
        origin_rank(left.origin)
            .cmp(&origin_rank(right.origin))
            .then(
                sample_order
                    .get(&left.sample_name)
                    .copied()
                    .unwrap_or(usize::MAX)
                    .cmp(
                        &sample_order
                            .get(&right.sample_name)
                            .copied()
                            .unwrap_or(usize::MAX),
                    ),
            )
            .then(left.name.cmp(&right.name))
    });
    entries.dedup_by(|left, right| left.id == right.id);
    entries
}

fn origin_rank(origin: AnnotationGroupOrigin) -> usize {
    match origin {
        AnnotationGroupOrigin::CurrentSample => 0,
        AnnotationGroupOrigin::ParentSample => 1,
        AnnotationGroupOrigin::AncestorSample => 2,
    }
}

#[cfg(test)]
mod tests {
    use gen_models::{
        annotations::AnnotationGroupSample,
        block_group::{BlockGroup, NewBlockGroup},
        collection::Collection,
        sample::Sample,
        sample_lineage::SampleLineage,
    };

    use super::*;
    use crate::test_helpers::setup_gen;

    #[test]
    fn loads_current_parent_and_ancestor_annotation_groups() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let _ = Collection::create(conn, "/test");

        for sample in ["grand", "parent", "child"] {
            let _ = Sample::get_or_create(conn, sample);
        }
        SampleLineage::create(conn, "grand", "parent").unwrap();
        SampleLineage::create(conn, "parent", "child").unwrap();

        let grand = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "/test",
                sample_name: "grand",
                name: "region",
                ..Default::default()
            },
        )
        .unwrap();
        let parent = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "/test",
                sample_name: "parent",
                name: "region",
                parent_block_group_id: Some(&grand.id),
                ..Default::default()
            },
        )
        .unwrap();
        let child = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "/test",
                sample_name: "child",
                name: "region",
                parent_block_group_id: Some(&parent.id),
                ..Default::default()
            },
        )
        .unwrap();

        AnnotationGroupSample::create(conn, "child-group", "child").unwrap();
        AnnotationGroupSample::create(conn, "parent-group", "parent").unwrap();
        AnnotationGroupSample::create(conn, "grand-group", "grand").unwrap();

        let child = BlockGroup::get_by_id(conn, &child.id).unwrap();
        let entries = load_annotation_group_entries(conn, &child);

        assert_eq!(
            entries
                .iter()
                .map(|entry| (&entry.name, &entry.sample_name, entry.origin))
                .collect::<Vec<_>>(),
            vec![
                (
                    &"child-group".to_string(),
                    &"child".to_string(),
                    AnnotationGroupOrigin::CurrentSample,
                ),
                (
                    &"parent-group".to_string(),
                    &"parent".to_string(),
                    AnnotationGroupOrigin::ParentSample,
                ),
                (
                    &"grand-group".to_string(),
                    &"grand".to_string(),
                    AnnotationGroupOrigin::AncestorSample,
                ),
            ]
        );
    }
}
