use std::{path::PathBuf, sync::Mutex};

use r#gen::{
    get_connection,
    views::gen_graph_widget::{GenGraphNodeRenderer, GenGraphNodeSizer},
};
use gen_graph::GenGraph;
use gen_models::{block_group::BlockGroup, db::GraphConnection};
use gen_tui::{graph_controller::GraphController, graph_widget::GraphWidget, layout::VisualDetail};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyDict, PyType},
};
use ratatui::{buffer::Buffer, layout::Rect, style::Modifier, widgets::StatefulWidget};
use serde::Serialize;

use crate::python_api::{block_group::PyBlockGroup, repository::PyRepository};

/// Convert a ratatui `Color` to a CSS hex string.
fn color_to_hex(color: Option<ratatui::style::Color>, default_hex: &str) -> String {
    use ratatui::style::Color;
    match color {
        None | Some(Color::Reset) => default_hex.to_string(),
        Some(Color::Rgb(r, g, b)) => format!("#{r:02x}{g:02x}{b:02x}"),
        Some(Color::Black) => "#000000".to_string(),
        Some(Color::Red) => "#cc0000".to_string(),
        Some(Color::Green) => "#00cc00".to_string(),
        Some(Color::Yellow) => "#cccc00".to_string(),
        Some(Color::Blue) => "#0000cc".to_string(),
        Some(Color::Magenta) => "#cc00cc".to_string(),
        Some(Color::Cyan) => "#00cccc".to_string(),
        Some(Color::Gray) => "#888888".to_string(),
        Some(Color::DarkGray) => "#444444".to_string(),
        Some(Color::LightRed) => "#ff5555".to_string(),
        Some(Color::LightGreen) => "#55ff55".to_string(),
        Some(Color::LightYellow) => "#ffff55".to_string(),
        Some(Color::LightBlue) => "#5555ff".to_string(),
        Some(Color::LightMagenta) => "#ff55ff".to_string(),
        Some(Color::LightCyan) => "#55ffff".to_string(),
        Some(Color::White) => "#ffffff".to_string(),
        Some(Color::Indexed(i)) => indexed_to_hex(i),
    }
}

/// Map an ANSI 256-colour index to a hex string.
fn indexed_to_hex(i: u8) -> String {
    // Indices 0-15: standard + bright ANSI colours.
    const STANDARD: [&str; 16] = [
        "#000000", "#800000", "#008000", "#808000", "#000080", "#800080", "#008080", "#c0c0c0",
        "#808080", "#ff0000", "#00ff00", "#ffff00", "#0000ff", "#ff00ff", "#00ffff", "#ffffff",
    ];
    if (i as usize) < STANDARD.len() {
        return STANDARD[i as usize].to_string();
    }
    // Indices 232-255: grayscale ramp (8, 18, 28, … 238).
    if i >= 232 {
        let v = 8u8.saturating_add((i - 232) * 10);
        return format!("#{v:02x}{v:02x}{v:02x}");
    }
    // Indices 16-231: 6×6×6 colour cube.
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let n = i - 16;
    let r = LEVELS[(n / 36) as usize];
    let g = LEVELS[((n / 6) % 6) as usize];
    let b = LEVELS[(n % 6) as usize];
    format!("#{r:02x}{g:02x}{b:02x}")
}

/// Format by which the buffer is to be serialized
#[derive(Serialize)]
struct RenderedCell {
    text: String,
    fg: String,
    bg: String,
    bold: bool,
    italic: bool,
    underline: bool,
}

#[derive(Serialize)]
struct RenderedFrame {
    cols: u16,
    rows: u16,
    cells: Vec<RenderedCell>,
}

fn serialize_buffer(buf: &Buffer, cols: u16, rows: u16) -> RenderedFrame {
    let mut cells = Vec::with_capacity(cols as usize * rows as usize);
    for row in 0..rows {
        for col in 0..cols {
            let cell = buf.cell((col, row)).expect("cell index in bounds");
            let style = cell.style();
            cells.push(RenderedCell {
                text: cell.symbol().to_string(),
                fg: color_to_hex(style.fg, "#cdd6f4"), // Catppuccin text default
                bg: color_to_hex(style.bg, "#1e1e2e"), // Catppuccin base default
                bold: style.add_modifier.contains(Modifier::BOLD),
                italic: style.add_modifier.contains(Modifier::ITALIC),
                underline: style.add_modifier.contains(Modifier::UNDERLINED),
            });
        }
    }
    RenderedFrame { cols, rows, cells }
}

/// Internal graph controller for the Jupyter notebook widget.
///
/// Not intended for direct use from Python — users should call
/// `repo.plot(bg)` or `bg.plot()` which return a `GenGraphWidget`.
#[pyclass]
pub struct PyGraphController {
    db_path: PathBuf,
    controller: GraphController<GenGraph, GenGraphNodeSizer>,
    conn: Mutex<Option<GraphConnection>>,
}

impl PyGraphController {
    pub fn new(db_path: PathBuf, graph: GenGraph) -> Self {
        let node_sizer = GenGraphNodeSizer;
        let mut controller = GraphController::new(graph, node_sizer);
        controller.set_detail_level(VisualDetail::Truncated);
        controller.hide_cursor();
        Self {
            db_path,
            controller,
            conn: Mutex::new(None),
        }
    }

    fn ensure_connection(&self) -> PyResult<()> {
        let mut guard = self.conn.lock().unwrap();
        if guard.is_none() {
            *guard = Some(
                get_connection(&self.db_path)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            );
        }
        Ok(())
    }
}

#[pymethods]
impl PyGraphController {
    /// Create a `GraphController` from a `Repository` and a `PyBlockGroup`.
    #[classmethod]
    fn from_block_group(
        _cls: &Bound<'_, PyType>,
        repo: &PyRepository,
        block_group: &PyBlockGroup,
    ) -> PyResult<Self> {
        let bg_id = block_group.id;
        let graph = repo.with_connection(|conn| BlockGroup::get_graph(conn, &bg_id));
        Ok(Self::new(repo.db_path.clone(), graph))
    }

    /// Set the level of node detail.
    ///
    /// Parameters
    /// ----------
    /// detail : {"normal", "full", "minimal"}
    ///     ``"normal"`` shows truncated labels (default); ``"full"`` shows
    ///     complete labels; ``"minimal"`` shows the smallest representation.
    pub fn set_detail(&mut self, detail: &str) -> PyResult<()> {
        let level = match detail {
            "normal" => VisualDetail::Truncated,
            "full" => VisualDetail::Full,
            "minimal" => VisualDetail::Minimal,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "detail must be \"normal\", \"full\", or \"minimal\"; got {other:?}"
                )));
            }
        };
        self.controller.set_detail_level(level);
        Ok(())
    }

    fn render_frame(&mut self, cols: u16, rows: u16) -> PyResult<String> {
        self.ensure_connection()?;

        let area = Rect::new(0, 0, cols, rows);
        let mut buf = Buffer::empty(area);

        {
            let guard = self.conn.lock().unwrap();
            let conn = guard.as_ref().unwrap();
            let renderer = GenGraphNodeRenderer::new(conn);
            GraphWidget::with_renderer(renderer).render(area, &mut buf, &mut self.controller);
        }

        let frame = serialize_buffer(&buf, cols, rows);
        serde_json::to_string(&frame).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn zoom_in(&mut self) {
        self.controller.zoom_in();
    }

    fn zoom_out(&mut self) {
        self.controller.zoom_out();
    }

    fn handle_click(&mut self, col: u16, row: u16) -> bool {
        self.controller.handle_click(col, row)
    }

    fn move_by(&mut self, dx: i16, dy: i16) {
        self.controller.move_by_terminal(dx, dy);
        self.controller.sync_cursor_to_closest_node();
    }
}

#[cfg(test)]
mod tests {
    use r#gen::test_helpers::{setup_block_group, setup_gen_on_disk};
    use gen_models::block_group::BlockGroup;
    use pyo3::{exceptions::PyValueError, prelude::*};
    use serde_json::Value;

    use super::PyGraphController;

    /// Shared setup: on-disk DB + a minimal block group, returning the
    /// controller and the db path.
    fn make_controller(detail: Option<&str>) -> PyResult<(std::path::PathBuf, PyGraphController)> {
        let ctx = setup_gen_on_disk();
        let (bg_id, _) = setup_block_group(ctx.graph().conn());
        let graph = BlockGroup::get_graph(ctx.graph().conn(), &bg_id);
        let db_path = ctx.workspace().ensure_gen_dir().join("default.db");
        let mut ctrl = PyGraphController::new(db_path.clone(), graph);
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        Ok((db_path, ctrl))
    }

    #[test]
    fn test_detail_invalid_raises_value_error() {
        pyo3::prepare_freethreaded_python();
        let result = make_controller(Some("bad"));
        Python::with_gil(|py| match result {
            Ok(_) => panic!("expected a PyValueError for invalid detail value"),
            Err(e) => assert!(e.is_instance_of::<PyValueError>(py)),
        });
    }

    #[test]
    fn test_detail_all_valid_values_accepted() {
        for detail in [None, Some("normal"), Some("full"), Some("minimal")] {
            let result = make_controller(detail);
            assert!(result.is_ok(), "detail={detail:?} should be accepted");
        }
    }

    #[test]
    fn test_render_frame_returns_valid_json() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let (_, mut ctrl) = make_controller(None).unwrap();
            let json_str = ctrl
                .render_frame(80, 24)
                .expect("render_frame should succeed");
            let v: Value = serde_json::from_str(&json_str).expect("output must be valid JSON");
            assert_eq!(v["cols"], 80);
            assert_eq!(v["rows"], 24);
            let cells = v["cells"].as_array().expect("cells must be an array");
            assert_eq!(cells.len(), 80 * 24);
        });
    }
}

/// Instantiate a `GenGraphWidget` from a controller and optional viewport
/// overrides, display it in Jupyter if available, and return it.
///
/// Shared by `PyBlockGroup::plot` and `PyRepository::plot`.
pub fn build_and_display_widget(
    py: Python<'_>,
    ctrl: Py<PyGraphController>,
    rows: Option<u32>,
    cols: Option<u32>,
) -> PyResult<PyObject> {
    let gen_module = py.import("gen")?;
    let widget_cls = gen_module.getattr("GenGraphWidget")?;
    let kwargs = PyDict::new(py);
    if let Some(r) = rows {
        kwargs.set_item("rows", r)?;
    }
    if let Some(c) = cols {
        kwargs.set_item("cols", c)?;
    }
    let widget = widget_cls.call((ctrl,), Some(&kwargs))?;

    if let Ok(ipy) = py.import("IPython.display") {
        let _ = ipy.getattr("display")?.call1((&widget,));
    }

    Ok(widget.into())
}
