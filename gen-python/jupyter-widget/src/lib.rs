use std::{cell::RefCell, collections::HashMap, fmt, rc::Rc};

use gen_tui::{
    geometry::WorldRect,
    graph_controller::GraphController,
    graph_controller::WorldBuffer,
    graph_widget::GraphWidget,
    layout::VisualDetail,
    plotter::{NodeRenderer, NodeSizer},
    theme::Theme,
};
use js_sys::Function;
use petgraph::graphmap::DiGraphMap;
use ratatui::Terminal;
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Style, Stylize},
    widgets::Paragraph,
};
use ratzilla::{
    WebGl2Backend, WebRenderer,
    backend::webgl2::{FontAtlasData, WebGl2BackendOptions},
    event::{KeyCode, KeyEvent, MouseButton, MouseEvent, MouseEventKind},
};
use serde::{
    Deserialize, Serialize, Serializer,
    de::{self, Visitor},
};
use wasm_bindgen::prelude::*;
use web_time::Instant;

// ---------------------------------------------------------------------------
// PATH_START / PATH_END sentinel bytes (mirrors gen-core)
// ---------------------------------------------------------------------------

const PATH_START_BYTES: [u8; 32] = [
    0x84, 0xd6, 0xad, 0xbd, 0x53, 0x95, 0x28, 0x19, 0x33, 0xfe, 0x41, 0xe8, 0x77, 0xd3, 0xa7, 0xf0,
    0x2a, 0x3b, 0x19, 0x90, 0xa6, 0x5b, 0xe1, 0x90, 0x1b, 0x2c, 0x91, 0xfc, 0x68, 0x5e, 0x08, 0x3b,
];

const PATH_END_BYTES: [u8; 32] = [
    0x1c, 0x7d, 0xfc, 0x64, 0x97, 0x7b, 0x08, 0x38, 0xaf, 0x07, 0x62, 0xd7, 0x33, 0x3d, 0xcb, 0x64,
    0xc1, 0x75, 0xb1, 0x5e, 0x65, 0xa7, 0x00, 0x99, 0xec, 0x38, 0xf4, 0x6b, 0xf1, 0xa1, 0x5e, 0xa3,
];

// ---------------------------------------------------------------------------
// NodeId — 32-byte hash, serializes as 64-char hex string
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct NodeId(pub [u8; 32]);

impl NodeId {
    pub fn is_start(&self) -> bool {
        self.0 == PATH_START_BYTES
    }

    pub fn is_end(&self) -> bool {
        self.0 == PATH_END_BYTES
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", hex::encode(&self.0[..4]))
    }
}

impl Serialize for NodeId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(self.0))
    }
}

impl<'de> Deserialize<'de> for NodeId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct HexVisitor;
        impl<'de> Visitor<'de> for HexVisitor {
            type Value = NodeId;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a 64-character hex string")
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<NodeId, E> {
                let bytes = hex::decode(v).map_err(de::Error::custom)?;
                let arr: [u8; 32] = bytes
                    .try_into()
                    .map_err(|_| de::Error::custom("expected 32 bytes (64 hex chars)"))?;
                Ok(NodeId(arr))
            }
        }
        d.deserialize_str(HexVisitor)
    }
}

// ---------------------------------------------------------------------------
// GraphNode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct GraphNode {
    pub block_id: i64,
    pub node_id: NodeId,
    pub sequence_start: i64,
    pub sequence_end: i64,
}

impl GraphNode {
    pub fn sequence_len(&self) -> i64 {
        (self.sequence_end - self.sequence_start).max(0)
    }

    /// Returns the canonical spec string used as the sequence request key.
    /// Format: "node_id_hex:sequence_start-sequence_end"
    pub fn spec(&self) -> String {
        format!(
            "{}:{}-{}",
            self.node_id, self.sequence_start, self.sequence_end
        )
    }
}

// ---------------------------------------------------------------------------
// Topology — the JSON payload sent from Python
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct TopologyResponse {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<(GraphNode, GraphNode)>,
}

pub type WidgetGraph = DiGraphMap<GraphNode, ()>;

// ---------------------------------------------------------------------------
// NodeSizer
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct WidgetNodeSizer;

impl NodeSizer<&WidgetGraph> for WidgetNodeSizer {
    fn get_node_size(&self, node: &GraphNode, detail_level: VisualDetail) -> (u64, u64) {
        if node.node_id.is_start() {
            return (8, 1);
        }
        if node.node_id.is_end() {
            return (7, 1);
        }
        let len = node.sequence_len() as u64;
        match detail_level {
            VisualDetail::Minimal => (1, 1),
            VisualDetail::Truncated => (len.min(12).max(1), 1),
            VisualDetail::Full => (len.max(1), 1),
        }
    }
}

// ---------------------------------------------------------------------------
// WidgetNodeRenderer — fetches sequences via a JS callback rather than HTTP
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct WidgetNodeRenderer {
    /// Truncated-mode sequence cache. Key = spec string ("node_id_hex:start-end").
    cache: Rc<RefCell<HashMap<String, String>>>,
    /// Specs currently awaiting a response from Python.
    pending: Rc<RefCell<Vec<String>>>,
    /// JS callback set by the widget host.
    /// Called with a single spec string each time a sequence is needed.
    sequence_cb: Rc<RefCell<Option<Function>>>,
}

impl WidgetNodeRenderer {
    fn new(
        cache: Rc<RefCell<HashMap<String, String>>>,
        pending: Rc<RefCell<Vec<String>>>,
        sequence_cb: Rc<RefCell<Option<Function>>>,
    ) -> Self {
        Self {
            cache,
            pending,
            sequence_cb,
        }
    }

    fn request_sequence(&self, node: &GraphNode) {
        let spec = node.spec();
        if self.pending.borrow().contains(&spec) {
            return;
        }
        self.pending.borrow_mut().push(spec.clone());

        if let Some(cb) = self.sequence_cb.borrow().as_ref() {
            let arr = js_sys::Array::new();
            arr.push(&JsValue::from_str(&spec));
            if let Err(e) = cb.call1(&JsValue::NULL, &arr) {
                web_sys::console::warn_1(&format!("sequence_cb error: {:?}", e).into());
            }
        }
    }
}

impl NodeRenderer<&WidgetGraph> for WidgetNodeRenderer {
    fn render_node(
        &mut self,
        buffer: &mut WorldBuffer,
        area: WorldRect,
        node: &GraphNode,
        detail_level: VisualDetail,
    ) {
        let node_style = Style::default().fg(Color::White).bg(Color::DarkGray);
        let canvas_style = Style::default().fg(Color::DarkGray).bg(Color::Black);
        buffer.fill_rect(area, ' ');

        if node.node_id.is_start() {
            buffer.set_string_styled(area.left_center(), " Start >", canvas_style);
            return;
        }
        if node.node_id.is_end() {
            buffer.set_string_styled(area.left_center(), "> End ", canvas_style);
            return;
        }

        match detail_level {
            VisualDetail::Minimal => {
                buffer.set_string_styled(area.left_center(), "●", canvas_style);
            }
            VisualDetail::Truncated | VisualDetail::Full => {
                let max_width = (area.max.x - area.min.x + 1) as usize;
                let spec = node.spec();
                let cached = self.cache.borrow().get(&spec).cloned();
                if let Some(seq) = cached {
                    let truncated: String = seq.chars().take(max_width).collect();
                    buffer.set_string_styled(area.left_center(), &truncated, node_style);
                } else {
                    buffer.set_string_styled(
                        area.left_center(),
                        &"~".repeat(max_width),
                        node_style,
                    );
                    self.request_sequence(node);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AppHandle — the public wasm_bindgen handle returned to JS
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct AppHandle {
    cache: Rc<RefCell<HashMap<String, String>>>,
    pending: Rc<RefCell<Vec<String>>>,
    sequence_cb: Rc<RefCell<Option<Function>>>,
}

#[wasm_bindgen]
impl AppHandle {
    /// Register the JS function to call when sequences are needed.
    ///
    /// The function is called with a single-element JS Array containing the
    /// spec string ("node_id_hex:start-end").  The widget host should respond
    /// by eventually calling `deliver_sequences`.
    pub fn set_sequence_callback(&self, cb: Function) {
        *self.sequence_cb.borrow_mut() = Some(cb);
    }

    /// Deliver fetched sequences back to the renderer cache.
    ///
    /// `json` must be a JSON object mapping spec strings to sequence strings,
    /// e.g. `{"abc...:0-100": "ACGT..."}`.
    pub fn deliver_sequences(&self, json: &str) {
        match serde_json::from_str::<HashMap<String, String>>(json) {
            Ok(map) => {
                let mut cache = self.cache.borrow_mut();
                let mut pending = self.pending.borrow_mut();
                for (spec, seq) in map {
                    pending.retain(|s| s != &spec);
                    cache.insert(spec, seq);
                }
            }
            Err(e) => {
                web_sys::console::warn_1(
                    &format!("deliver_sequences: failed to parse JSON: {e}").into(),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// App — internal render-loop state (not exported to JS)
// ---------------------------------------------------------------------------

type WidgetController = GraphController<&'static WidgetGraph, WidgetNodeSizer>;

struct App {
    controller: WidgetController,
    renderer: WidgetNodeRenderer,
    last_frame: Instant,
    cell_size: (u32, u32),
    mouse_down_pos: Option<(u32, u32)>,
    mouse_is_dragging: bool,
    pan_acc: (f64, f64),
}

impl App {
    fn handle_key(&mut self, key: KeyEvent) {
        let controller = &mut self.controller;
        match key.code {
            KeyCode::Enter => {
                controller.cursor.set_coarse_mode(false);
            }
            KeyCode::Esc => {
                if controller.is_panning_mode() {
                    controller.exit_panning_mode();
                } else {
                    controller.cursor.set_coarse_mode(true);
                }
            }
            KeyCode::Left | KeyCode::Char('h') => {
                if controller.is_panning_mode() {
                    controller.exit_panning_mode();
                }
                let vp_w = controller.viewport_state.viewport_bounds.width as i64;
                let delta = if controller.cursor.is_coarse_mode() {
                    -vp_w
                } else {
                    -1
                };
                let _ = controller.navigate_horizontal(delta);
            }
            KeyCode::Right | KeyCode::Char('l') => {
                if controller.is_panning_mode() {
                    controller.exit_panning_mode();
                }
                let vp_w = controller.viewport_state.viewport_bounds.width as i64;
                let delta = if controller.cursor.is_coarse_mode() {
                    vp_w
                } else {
                    1
                };
                let _ = controller.navigate_horizontal(delta);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if controller.is_panning_mode() {
                    controller.exit_panning_mode();
                }
                let vp_h = controller.viewport_state.viewport_bounds.height as i64;
                let delta = if controller.cursor.is_coarse_mode() {
                    vp_h
                } else {
                    1
                };
                let _ = controller.navigate_vertical(delta);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if controller.is_panning_mode() {
                    controller.exit_panning_mode();
                }
                let vp_h = controller.viewport_state.viewport_bounds.height as i64;
                let delta = if controller.cursor.is_coarse_mode() {
                    -vp_h
                } else {
                    -1
                };
                let _ = controller.navigate_vertical(delta);
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                controller.zoom_in();
            }
            KeyCode::Char('-') => {
                controller.zoom_out();
            }
            KeyCode::Char('r') => {
                controller.trigger_rebuild();
            }
            _ => {}
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        let controller = &mut self.controller;
        match mouse.event {
            MouseEventKind::Pressed => {
                if mouse.button == MouseButton::Left {
                    self.mouse_down_pos = Some((mouse.x, mouse.y));
                    self.mouse_is_dragging = false;
                }
            }
            MouseEventKind::Moved => {
                if let Some((lx, ly)) = self.mouse_down_pos {
                    let (cw, ch) = self.cell_size;
                    let dx_px = mouse.x as f64 - lx as f64;
                    let dy_px = mouse.y as f64 - ly as f64;
                    self.pan_acc.0 += dx_px / cw as f64;
                    self.pan_acc.1 += dy_px / ch as f64;
                    let cell_dx = self.pan_acc.0.trunc() as i16;
                    let cell_dy = self.pan_acc.1.trunc() as i16;
                    self.pan_acc.0 = self.pan_acc.0.fract();
                    self.pan_acc.1 = self.pan_acc.1.fract();
                    if cell_dx != 0 || cell_dy != 0 {
                        controller.handle_pan_terminal(cell_dx, cell_dy);
                        controller.sync_cursor_to_closest_node();
                    }
                    self.mouse_down_pos = Some((mouse.x, mouse.y));
                    self.mouse_is_dragging = true;
                }
            }
            MouseEventKind::Released => {
                if mouse.button == MouseButton::Left {
                    if !self.mouse_is_dragging {
                        let (cw, ch) = self.cell_size;
                        let cell_x = (mouse.x / cw) as u16;
                        let cell_y = (mouse.y / ch) as u16;
                        controller.handle_click(cell_x, cell_y);
                    }
                    self.mouse_down_pos = None;
                    self.mouse_is_dragging = false;
                }
            }
            _ => {}
        }
    }

    fn render(&mut self, frame: &mut Frame) {
        let delta = self.last_frame.elapsed();
        self.last_frame = Instant::now();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(1)])
            .split(frame.area());

        let graph_area = chunks[0];
        let is_coarse = self.controller.cursor.is_coarse_mode();
        self.controller.viewport_state.viewport_bounds = graph_area;
        self.controller.update_animations(delta);
        frame.render_stateful_widget(
            GraphWidget::with_renderer(self.renderer.clone()).cursor(),
            graph_area,
            &mut self.controller,
        );

        let footer_text = if self.controller.is_panning_mode() {
            " drag: pan  |  click: select  |  +/-: zoom  |  arrows: nav "
        } else if is_coarse {
            " ← → ↑ ↓: navigate  |  enter: fine nav  |  +/-: zoom "
        } else {
            " ← → ↑ ↓: navigate  |  esc: coarse nav  |  +/-: zoom "
        };
        let footer_style = if is_coarse {
            Color::DarkGray
        } else {
            Color::Yellow
        };
        let footer = Paragraph::new(footer_text)
            .alignment(Alignment::Center)
            .fg(footer_style)
            .bg(Color::Black);
        frame.render_widget(footer, chunks[1]);
    }
}

// ---------------------------------------------------------------------------
// mount_app — the main entry point exported to JS
// ---------------------------------------------------------------------------

/// Mount the gen-tui graph widget into `container_id` using the provided topology JSON.
///
/// Returns an `AppHandle` that the widget host uses to deliver sequences on demand.
/// The topology JSON must match the `TopologyResponse` schema:
/// `{"nodes": [...], "edges": [[src, dst], ...]}`.
#[wasm_bindgen]
pub fn mount_app(container_id: &str, topology_json: &str) -> Result<AppHandle, JsValue> {
    console_error_panic_hook::set_once();

    // Parse topology
    let topology: TopologyResponse = serde_json::from_str(topology_json)
        .map_err(|e| JsValue::from_str(&format!("topology parse error: {e}")))?;

    // Build graph
    let mut graph = WidgetGraph::new();
    for node in topology.nodes {
        graph.add_node(node);
    }
    for (from, to) in topology.edges {
        graph.add_edge(from, to, ());
    }
    let graph: &'static WidgetGraph = Box::leak(Box::new(graph));

    // Build terminal mounted to the container element
    let (terminal, cell_size) = build_terminal(container_id)?;

    // Shared sequence state
    let cache: Rc<RefCell<HashMap<String, String>>> = Rc::new(RefCell::new(HashMap::new()));
    let pending: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let sequence_cb: Rc<RefCell<Option<Function>>> = Rc::new(RefCell::new(None));

    // Build controller
    let node_sizer = WidgetNodeSizer;
    let mut controller = GraphController::new(graph, node_sizer).with_theme(Theme::default());
    controller.set_detail_level(VisualDetail::Truncated);
    controller.enter_panning_mode();

    // Build renderer backed by the shared state
    let renderer = WidgetNodeRenderer::new(
        Rc::clone(&cache),
        Rc::clone(&pending),
        Rc::clone(&sequence_cb),
    );

    let app = Rc::new(RefCell::new(App {
        controller,
        renderer,
        last_frame: Instant::now(),
        cell_size,
        mouse_down_pos: None,
        mouse_is_dragging: false,
        pan_acc: (0.0, 0.0),
    }));

    let event_app = Rc::clone(&app);
    let _ = terminal.on_key_event(move |key| {
        event_app.borrow_mut().handle_key(key);
    });

    let mouse_app = Rc::clone(&app);
    let _ = terminal.on_mouse_event(move |mouse| {
        mouse_app.borrow_mut().handle_mouse(mouse);
    });

    let render_app = Rc::clone(&app);
    terminal.draw_web(move |frame| {
        render_app.borrow_mut().render(frame);
    });

    Ok(AppHandle {
        cache,
        pending,
        sequence_cb,
    })
}

// ---------------------------------------------------------------------------
// Terminal construction
// ---------------------------------------------------------------------------

fn build_terminal(container_id: &str) -> Result<(Terminal<WebGl2Backend>, (u32, u32)), JsValue> {
    // Atlas constructed by
    // beamterm-atlas "Jetbrains Mono" --emoji-font "Jetbrains Mono" --range 0x2500..0x257f  --range 0x2580..0x259f --range 0x25a0..0x25ff --range 0x2190..0x21FF --output JetbrainsMono.atlas
    let atlas = FontAtlasData::from_binary(include_bytes!("../JetbrainsMono.atlas"))
        .map_err(|e| JsValue::from_str(&format!("font atlas error: {e:?}")))?;
    let cell_size = {
        let (w, h) = atlas.cell_size;
        (w as u32, h as u32)
    };
    let options = WebGl2BackendOptions::new()
        .grid_id(container_id)
        .font_atlas(atlas);
    let backend = WebGl2Backend::new_with_options(options).map_err(|e| {
        JsValue::from_str(&format!(
            "WebGL2 is required but could not be initialised: {e}"
        ))
    })?;
    let terminal = Terminal::new(backend)
        .map_err(|e| JsValue::from_str(&format!("terminal init failed: {e}")))?;
    Ok((terminal, cell_size))
}
