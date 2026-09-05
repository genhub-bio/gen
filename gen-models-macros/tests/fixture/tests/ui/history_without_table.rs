use gen_models::ModelSelect;

#[derive(ModelSelect)]
#[model_select(history = false)]
struct HistoryWithoutTable {
    value: i64,
}

fn main() {}
