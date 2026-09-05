use gen_models::ModelSelect;

#[derive(ModelSelect)]
#[model_select(table = "invalid_default_sort")]
struct InvalidDefaultSort {
    #[model_select(default_sort = "sideways")]
    id: i64,
}

fn main() {}
