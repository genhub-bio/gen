use gen_models::ModelSelect;

#[derive(ModelSelect)]
#[model_select(table = "invalid_default_sort", default_sort(id = "sideways"))]
struct InvalidDefaultSort {
    id: i64,
}

fn main() {}
