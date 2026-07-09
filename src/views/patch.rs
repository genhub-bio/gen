use std::collections::HashMap;

use gen_core::{HashId, config::Workspace};
use gen_models::sequence::SequenceError;

use crate::patch::OperationPatch;

fn placeholder_diagram(patch: &OperationPatch) -> String {
    let summary = patch.commit_message.replace('"', "\\\"");
    let commit_hash = patch.commit.commit_hash.0.replace('"', "\\\"");
    let change_type = patch.commit.change_type.replace('"', "\\\"");

    format!(
        "digraph {{\n    rankdir=LR\n    node [shape=box]\n    patch [label=\"commit: {commit_hash}\\nkind: {change_type}\\nsummary: {summary}\"]\n}}\n"
    )
}

pub fn view_patches(
    _workspace: &Workspace,
    patches: &[OperationPatch],
) -> Result<HashMap<HashId, HashMap<HashId, String>>, SequenceError> {
    let mut diagrams = HashMap::new();
    for patch in patches {
        let mut patch_diagrams = HashMap::new();
        patch_diagrams.insert(HashId::pad_str(0), placeholder_diagram(patch));
        diagrams.insert(patch.commit.hash, patch_diagrams);
    }
    Ok(diagrams)
}
