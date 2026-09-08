use gen_models::ModelSelect;

#[derive(ModelSelect)]
#[model_select(table = "skipped_models")]
struct SkippedWithoutFromRow {
    value: i64,
    #[model_select(skip)]
    derived: String,
}

fn main() {}
