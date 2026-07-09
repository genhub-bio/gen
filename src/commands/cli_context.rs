use gen_models::db::DbContext;

pub struct CliContext<'a> {
    pub context: &'a DbContext,
    pub history_ref: Option<&'a str>,
}
