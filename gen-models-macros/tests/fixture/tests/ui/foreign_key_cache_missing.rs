use gen_models::ModelSelect;

#[derive(ModelSelect)]
#[model_select(table = "foreign_targets")]
struct ForeignTarget {
    id: i64,
}

#[derive(ModelSelect)]
#[model_select(table = "foreign_relations")]
struct MissingCacheField {
    id: i64,
    #[model_select(foreign_key(ForeignTarget, cache = cached_target))]
    target_id: i64,
}

fn main() {}
