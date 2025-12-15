use gen_core::config::{DbContext, RepoHandle, RepoKind};
use rusqlite::Connection;

pub trait GraphDb {
    fn graph_conn(&self) -> &Connection;
}

pub trait OperationsDb {
    fn operations_conn(&self) -> &Connection;
}

impl GraphDb for Connection {
    fn graph_conn(&self) -> &Connection {
        self
    }
}

impl GraphDb for DbContext {
    fn graph_conn(&self) -> &Connection {
        self.graph().conn()
    }
}

impl GraphDb for RepoHandle {
    fn graph_conn(&self) -> &Connection {
        debug_assert!(
            self.kind() == RepoKind::Graph,
            "GraphDb used with non-graph handle"
        );
        self.conn()
    }
}

impl OperationsDb for Connection {
    fn operations_conn(&self) -> &Connection {
        self
    }
}

impl OperationsDb for DbContext {
    fn operations_conn(&self) -> &Connection {
        self.operations().conn()
    }
}

impl OperationsDb for RepoHandle {
    fn operations_conn(&self) -> &Connection {
        debug_assert!(
            self.kind() == RepoKind::Operations,
            "OperationsDb used with non-operations handle"
        );
        self.conn()
    }
}
