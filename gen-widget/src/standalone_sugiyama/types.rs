/// Represents a layout of a single graph component:
/// - A list of vertex IDs and their (x,y) coordinates
/// - The width of the layout
/// - The height of the layout
pub type Layout = (Vec<(usize, (f64, f64))>, f64, f64);

/// Represents layouts for multiple graph components, with a generic type T
/// for the vertex identifier
pub type Layouts<T> = Vec<(Vec<(T, (f64, f64))>, f64, f64)>;
