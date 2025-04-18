use crate::NewCli;

#[derive(Debug)]
pub struct CliContext {
    pub db: Option<String>,
}

impl<'a> From<&'a NewCli> for CliContext {
    fn from(cli: &'a NewCli) -> Self {
        CliContext { db: cli.db.clone() }
    }
}
