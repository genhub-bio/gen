//! An owned handle on one graph database, for viewers that outlive any borrowed connection.
//!
//! The TUI borrows its connection from `main`, but the Jupyter widget is a `#[pyclass]` that
//! Python moves between threads and the R widget sits behind an `ExternalPtr` with no
//! lifetime, so neither can hold a `&GraphConnection`. A `GraphDatabase` carries what it takes
//! to open a connection instead (the file, the branch the viewed graph lives on, and the
//! workspace), opens it on first use, and hands the same file and branch to the lazy graph and
//! sequence sources, which open their own.

use std::{error::Error, path::PathBuf, sync::Mutex};

use gen_core::{HashId, Workspace, errors::ConnectionError};
use gen_models::{db::GraphConnection, history::dolt::active_branch};

use crate::{
    get_connection_for_branch,
    views::{gen_graph_widget::PathSequenceSource, lazy_graph_source::SqlGraphSource},
};

#[derive(Debug)]
pub struct GraphDatabase {
    db_path: PathBuf,
    /// A fresh connection starts on the repository's default branch, so every connection
    /// opened for this viewer is pinned to the branch the viewed graph lives on.
    branch: Option<String>,
    workspace: Workspace,
    /// Opened on first use and kept. The `Mutex` only makes this type `Sync`
    /// (`rusqlite::Connection` is `Send` but not `Sync`); access goes through `&mut self`, so it
    /// is never contended.
    connection: Mutex<Option<GraphConnection>>,
}

/// A clone opens its own connection on first use, since a live connection can't be shared.
impl Clone for GraphDatabase {
    fn clone(&self) -> Self {
        Self::new(
            self.db_path.clone(),
            self.branch.clone(),
            self.workspace.clone(),
        )
    }
}

impl GraphDatabase {
    pub fn new(db_path: PathBuf, branch: Option<String>, workspace: Workspace) -> Self {
        Self {
            db_path,
            branch,
            workspace,
            connection: Mutex::new(None),
        }
    }

    /// A handle on the database `conn` is open on, pinned to the branch it has checked out.
    pub fn for_connection(
        conn: &GraphConnection,
        workspace: &Workspace,
    ) -> Result<Self, Box<dyn Error>> {
        let db_path = conn
            .path()
            .map(PathBuf::from)
            .ok_or("graph database has no file path")?;
        Ok(Self::new(
            db_path,
            Some(active_branch(conn)?),
            workspace.clone(),
        ))
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// The connection, opened on the pinned branch the first time it's asked for.
    pub fn connection(&mut self) -> Result<&GraphConnection, ConnectionError> {
        let slot = self
            .connection
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot.is_none() {
            *slot = Some(get_connection_for_branch(
                &self.db_path,
                self.branch.as_deref(),
            )?);
        }
        Ok(slot
            .as_ref()
            .expect("should have just opened the connection"))
    }

    /// A source that crawls `block_group_id` from this database; `prune` leaves out the edges
    /// `BlockGroup::prune_graph` would remove (see `SqlGraphSource::new_pruned`).
    pub fn graph_source(&self, block_group_id: HashId, prune: bool) -> SqlGraphSource {
        let source = if prune {
            SqlGraphSource::new_pruned(self.db_path.clone(), block_group_id)
        } else {
            SqlGraphSource::new(self.db_path.clone(), block_group_id)
        };
        source.with_branch(self.branch.clone())
    }

    /// A source the renderers read node sequences from.
    pub fn sequence_source(&self) -> PathSequenceSource {
        PathSequenceSource::new(self.db_path.clone()).with_branch(self.branch.clone())
    }
}
