use crate::commands::Cli;

#[derive(Debug)]
pub struct CliContext {
    pub db: Option<String>,
}

impl<'a> From<&'a Cli> for CliContext {
    fn from(cli: &'a Cli) -> Self {
        CliContext { db: cli.db.clone() }
    }
}
