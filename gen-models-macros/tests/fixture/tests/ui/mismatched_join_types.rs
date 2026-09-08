use gen_models::{ModelSelect, select::Connection};

#[derive(ModelSelect)]
#[model_select(table = "left_models")]
struct LeftModel {
    id: i64,
    name: String,
}

#[derive(ModelSelect)]
#[model_select(table = "right_models")]
struct RightModel {
    id: i64,
}

fn main() {
    let conn = Connection::open_in_memory().expect("should open test database");
    let _selector = LeftModel::select(&conn).join_on(LeftModelSelect::Name, RightModelSelect::Id);
}
