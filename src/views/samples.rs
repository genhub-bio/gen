use std::collections::HashSet;

use crate::views::collection::{CollectionExplorerData, CollectionExplorerState};

const AUTO_UNFURL_LINES: usize = 10;

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
        let mut remaining = AUTO_UNFURL_LINES;

        let mut roots = self.data.sample_roots.clone();
        if roots.is_empty() {
            roots = self.data.collection_samples.clone();
        }

        for root in roots {
            self.push_sample(&root, 0, &mut visited, state, &mut entries, &mut remaining);
            if remaining == 0 {
                break;
            }
        }

        for sample_name in &self.data.collection_samples {
            if !visited.contains(sample_name) {
                self.push_sample(
                    sample_name,
                    0,
                    &mut visited,
                    state,
                    &mut entries,
                    &mut remaining,
                );
                if remaining == 0 {
                    break;
                }
            }
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
        visited: &mut HashSet<String>,
        state: &CollectionExplorerState,
        entries: &mut Vec<SampleTreeEntry>,
        remaining: &mut usize,
    ) {
        if *remaining == 0 {
            return;
        }
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
        let expanded = state.is_sample_expanded(sample_name) || *remaining > 0;

        entries.push(SampleTreeEntry::Sample {
            name: sample_name.to_string(),
            expanded,
            depth,
            has_children: !children.is_empty() || !block_groups.is_empty(),
        });
        *remaining = remaining.saturating_sub(1);

        if !expanded || *remaining == 0 {
            return;
        }

        for (id, name) in block_groups {
            if *remaining == 0 {
                return;
            }
            entries.push(SampleTreeEntry::BlockGroup {
                id,
                name,
                depth: depth + 1,
            });
            *remaining = remaining.saturating_sub(1);
        }

        for child in children {
            if *remaining == 0 {
                return;
            }
            self.push_sample(&child, depth + 1, visited, state, entries, remaining);
        }
    }
}
