mod gfa;

use std::{cell::RefCell, io, rc::Rc};

use gen_tui::{
    graph_controller::{GraphConfig, GraphController},
    graph_widget::GraphWidget,
    layout::VisualDetail,
    theme::Theme,
};
use gfa::{GfaGraph, GfaNodeRenderer, GfaNodeSizer, parse_gfa};
use ratatui::{
    Frame, Terminal,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Stylize},
    widgets::{Block, Paragraph},
};
use ratzilla::{
    WebGl2Backend, WebRenderer,
    backend::webgl2::{FontAtlasData, WebGl2BackendOptions},
    event::{KeyCode, KeyEvent},
};
use web_time::Instant;

type WasmController = GraphController<&'static GfaGraph, GfaNodeSizer>;

fn make_controller(graph: GfaGraph) -> WasmController {
    let graph: &'static GfaGraph = Box::leak(Box::new(graph));
    let node_sizer = GfaNodeSizer;
    let mut config = GraphConfig::default();
    config.partition.layer_count = usize::MAX;
    config.partition.node_count = usize::MAX;
    let mut controller = GraphController::new_with_config(graph, node_sizer, config);
    controller.set_detail_level(VisualDetail::Truncated);
    controller.show_cursor();
    controller.with_theme(Theme::default())
}

struct App {
    controller: WasmController,
    renderer: GfaNodeRenderer,
    last_frame: Instant,
}

impl App {
    fn new() -> Self {
        let gfa_src = include_str!("sample.gfa");
        let (graph, sequences) = parse_gfa(gfa_src);
        let controller = make_controller(graph);
        let renderer = GfaNodeRenderer::new(sequences);
        App {
            controller,
            renderer,
            last_frame: Instant::now(),
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        web_sys::console::log_1(&format!("key: {:?}", key.code).into());

        match key.code {
            KeyCode::Enter => {
                self.controller.cursor.set_coarse_mode(false);
            }
            KeyCode::Esc => {
                self.controller.cursor.set_coarse_mode(true);
            }
            KeyCode::Left | KeyCode::Char('h') => {
                let vp_w = self.controller.viewport_state.viewport_bounds.width as i64;
                let delta = if self.controller.cursor.is_coarse_mode() {
                    -vp_w
                } else {
                    -1
                };
                let _ = self.controller.navigate_horizontal(delta);
            }
            KeyCode::Right | KeyCode::Char('l') => {
                let vp_w = self.controller.viewport_state.viewport_bounds.width as i64;
                let delta = if self.controller.cursor.is_coarse_mode() {
                    vp_w
                } else {
                    1
                };
                let _ = self.controller.navigate_horizontal(delta);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                let vp_h = self.controller.viewport_state.viewport_bounds.height as i64;
                let delta = if self.controller.cursor.is_coarse_mode() {
                    vp_h
                } else {
                    1
                };
                let _ = self.controller.navigate_vertical(delta);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let vp_h = self.controller.viewport_state.viewport_bounds.height as i64;
                let delta = if self.controller.cursor.is_coarse_mode() {
                    -vp_h
                } else {
                    -1
                };
                let _ = self.controller.navigate_vertical(delta);
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.controller.zoom_in();
            }
            KeyCode::Char('-') => {
                self.controller.zoom_out();
            }
            KeyCode::Char('r') => {
                self.controller.trigger_rebuild();
            }
            _ => {}
        }
    }

    fn render(&mut self, frame: &mut Frame) {
        let delta = self.last_frame.elapsed();
        self.last_frame = Instant::now();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(frame.area());

        let header = Paragraph::new(" gen-tui-wasm  |  GFA viewer ")
            .block(Block::bordered())
            .alignment(Alignment::Center)
            .fg(Color::White)
            .bg(Color::DarkGray);
        frame.render_widget(header, chunks[0]);

        let graph_area = chunks[1];
        let is_coarse = self.controller.cursor.is_coarse_mode();
        self.controller.viewport_state.viewport_bounds = graph_area;
        self.controller.update_animations(delta);
        frame.render_stateful_widget(
            GraphWidget::with_renderer(self.renderer.clone()).cursor(),
            graph_area,
            &mut self.controller,
        );

        let footer_text = if is_coarse {
            " h j k l: navigate  |  enter: fine mode  |  +/-: zoom "
        } else {
            " h j k l: navigate  |  esc: coarse mode  |  +/-: zoom "
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
        frame.render_widget(footer, chunks[2]);
    }
}

fn main() -> io::Result<()> {
    console_error_panic_hook::set_once();
    // Font Atlas Data generated using:
    // beamterm-atlas "DejaVu Sans Mono"  --emoji-font "DejaVu Sans Mono" --output src/bitmap_font.atlas --line-height 1.5
    let font_atlas_data = FontAtlasData::from_binary(include_bytes!("bitmap_font.atlas"));
    let backend_options = WebGl2BackendOptions::new().font_atlas(font_atlas_data.unwrap());
    let backend = WebGl2Backend::new_with_options(backend_options)?;
    let terminal = Terminal::new(backend)?;

    let app = Rc::new(RefCell::new(App::new()));

    let event_app = Rc::clone(&app);
    terminal.on_key_event(move |key_event| {
        event_app.borrow_mut().handle_key(key_event);
    });

    let render_app = Rc::clone(&app);
    terminal.draw_web(move |frame| {
        render_app.borrow_mut().render(frame);
    });

    Ok(())
}
