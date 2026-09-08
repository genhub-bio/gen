use gen_models::ModelSelect;

#[derive(ModelSelect)]
#[model_select(table = "unknown_default_sort_field", default_sort(missing = "asc"))]
struct UnknownDefaultSortField {
    id: i64,
}

fn main() {}
