use gen_models_doltlite::db::DbContext;

pub struct CliContext<'a> {
    pub context: &'a DbContext,
    pub history_ref: Option<&'a str>,
}
