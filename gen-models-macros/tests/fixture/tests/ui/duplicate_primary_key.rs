use gen_models::ModelSelect;

#[derive(ModelSelect)]
#[model_select(table = "duplicate_primary_keys")]
struct DuplicatePrimaryKey {
    #[model_select(primary_key)]
    first: i64,
    #[model_select(primary_key)]
    second: i64,
}

fn main() {}
