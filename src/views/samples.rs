use std::collections::HashSet;

use crate::views::collection::{CollectionExplorerData, CollectionExplorerState};

const AUTO_EXPAND_ROOTS: usize = 5;
const AUTO_EXPAND_DEPTH: usize = 3;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SampleTreeEntry {
    Sample {
        name: String,
        expanded: bool,
        depth: usize,
        has_children: bool,
    },
    BlockGroup {
        id: gen_core::HashId,
        name: String,
        depth: usize,
    },
}

impl SampleTreeEntry {
    pub fn depth(&self) -> usize {
        match self {
            SampleTreeEntry::Sample { depth, .. } => *depth,
            SampleTreeEntry::BlockGroup { depth, .. } => *depth,
        }
    }
}

pub struct SampleTree<'a> {
    data: &'a CollectionExplorerData,
}

impl<'a> SampleTree<'a> {
    pub fn new(data: &'a CollectionExplorerData) -> Self {
        Self { data }
    }

    pub fn build_entries(&self, state: &CollectionExplorerState) -> Vec<SampleTreeEntry> {
        let mut entries = Vec::new();
        let mut visited = HashSet::new();

        let mut roots = self.data.sample_roots.clone();
        if roots.is_empty() {
            roots = self.data.collection_samples.clone();
        }

        for (index, root) in roots.iter().enumerate() {
            self.push_sample(
                root,
                0,
                index < AUTO_EXPAND_ROOTS,
                &mut visited,
                state,
                &mut entries,
            );
        }

        entries
    }

    pub fn selected_scroll(
        &self,
        state: &CollectionExplorerState,
        entries: &[SampleTreeEntry],
    ) -> u16 {
        state
            .list_state
            .selected
            .and_then(|idx| entries.get(idx))
            .map(|entry| (entry.depth() as u16).saturating_mul(2))
            .unwrap_or(0)
    }

    pub fn selected_sample_name(
        &self,
        state: &CollectionExplorerState,
        entries: &[SampleTreeEntry],
    ) -> Option<String> {
        state
            .list_state
            .selected
            .and_then(|idx| entries.get(idx))
            .and_then(|entry| match entry {
                SampleTreeEntry::Sample { name, .. } => Some(name.clone()),
                _ => None,
            })
    }

    fn push_sample(
        &self,
        sample_name: &str,
        depth: usize,
        auto_expand: bool,
        visited: &mut HashSet<String>,
        state: &CollectionExplorerState,
        entries: &mut Vec<SampleTreeEntry>,
    ) {
        if !visited.insert(sample_name.to_string()) {
            return;
        }

        let children = self
            .data
            .sample_children
            .get(sample_name)
            .cloned()
            .unwrap_or_default();
        let block_groups = self
            .data
            .sample_block_groups
            .get(sample_name)
            .cloned()
            .unwrap_or_default();
        let expanded =
            state.is_sample_expanded(sample_name, auto_expand && depth < AUTO_EXPAND_DEPTH);

        entries.push(SampleTreeEntry::Sample {
            name: sample_name.to_string(),
            expanded,
            depth,
            has_children: !children.is_empty() || !block_groups.is_empty(),
        });

        if !expanded {
            return;
        }

        for (id, name) in block_groups {
            entries.push(SampleTreeEntry::BlockGroup {
                id,
                name,
                depth: depth + 1,
            });
        }

        for child in children {
            self.push_sample(&child, depth + 1, auto_expand, visited, state, entries);
        }
    }
}
