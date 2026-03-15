use std::{cell::RefCell, io, rc::Rc};

use gen_tui::{
    graph_controller::{GraphConfig, GraphController, WorldBuffer},
    layout::VisualDetail,
    plotter::plot_viewport_graph,
    testing::mocks::{DebugNodeRenderer, FixedNodeSizer, MockDomainGraph, TestGraphs},
    theme::Theme,
};
use ratatui::{
    Frame, Terminal,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Stylize},
    widgets::{Block, Paragraph},
};
use ratzilla::{
    DomBackend, WebRenderer,
    event::{KeyCode, KeyEvent},
};

// Use 'static references so the controller can live in the WASM app state.
// Graphs are intentionally leaked (small test data, WASM demo context).
type WasmController = GraphController<&'static MockDomainGraph, FixedNodeSizer>;

fn make_controller(graph: MockDomainGraph) -> WasmController {
    let graph: &'static MockDomainGraph = Box::leak(Box::new(graph));
    let node_sizer = FixedNodeSizer {
        width: 5,
        height: 3,
    };
    let mut config = GraphConfig::default();
    config.partition.layer_count = usize::MAX;
    config.partition.node_count = usize::MAX;
    let mut controller = GraphController::new_with_config(graph, node_sizer, config);
    controller.set_detail_level(VisualDetail::Full);
    controller.show_cursor();
    controller.initialize_cursor();
    controller.with_theme(Theme::default())
}

fn build_named_graphs() -> Vec<(String, MockDomainGraph)> {
    let mut graphs: Vec<(String, MockDomainGraph)> = Vec::new();

    // graphs.push(("Simple Chain".into(), TestGraphs::domain_simple_chain()));
    // graphs.push(("Diamond".into(), TestGraphs::domain_diamond()));
    // graphs.push(("Extended Diamond".into(), TestGraphs::domain_extended_diamond()));
    graphs.push((
        "Complex DAG (9 nodes)".into(),
        TestGraphs::domain_complex_dag(),
    ));
    // graphs.push(("Skip Layer".into(), TestGraphs::domain_skip_layer()));
    // graphs.push(("Single Node".into(), TestGraphs::domain_single_node()));
    // graphs.push(("Star Graph".into(), TestGraphs::domain_star_graph()));
    // graphs.push(("Bridge Graph".into(), TestGraphs::domain_bridge_graph()));
    // graphs.push(("Articulation Graph".into(), TestGraphs::domain_articulation_graph()));

    // Double chain from layout tests
    {
        let mut g = MockDomainGraph::new();
        let nodes: Vec<_> = (0..18).map(|_| g.add_node(())).collect();
        for i in 0..9 {
            g.add_edge(nodes[i], nodes[i + 1], ());
        }
        g.add_edge(nodes[0], nodes[10], ());
        for i in 10..17 {
            g.add_edge(nodes[i], nodes[i + 1], ());
        }
        g.add_edge(nodes[17], nodes[9], ());
        graphs.push(("Double Chain (18 nodes)".into(), g));
    }

    // Asymmetric diamond (3-2 legs)
    {
        let mut g = MockDomainGraph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        let d = g.add_node(());
        let e = g.add_node(());
        let f1 = g.add_node(());
        let f2 = g.add_node(());
        g.add_edge(a, b, ());
        g.add_edge(b, c, ());
        g.add_edge(c, d, ());
        g.add_edge(d, e, ());
        g.add_edge(a, f1, ());
        g.add_edge(f1, f2, ());
        g.add_edge(f2, e, ());
        graphs.push(("Asymmetric Diamond (3-2)".into(), g));
    }

    graphs
}

struct App {
    // Names paired with controllers; graphs are owned by the controllers (leaked for 'static).
    named_controllers: Vec<(String, WasmController)>,
    current_idx: usize,
    renderer: DebugNodeRenderer,
    needs_rebuild: bool,
}

impl App {
    fn new() -> Self {
        let named_graphs = build_named_graphs();
        let named_controllers = named_graphs
            .into_iter()
            .map(|(name, graph)| (name, make_controller(graph)))
            .collect();
        App {
            named_controllers,
            current_idx: 0,
            renderer: DebugNodeRenderer::new(),
            needs_rebuild: true,
        }
    }

    fn current_name(&self) -> &str {
        &self.named_controllers[self.current_idx].0
    }

    fn switch_to(&mut self, idx: usize) {
        self.current_idx = idx;
        self.needs_rebuild = true;
    }

    fn next_graph(&mut self) {
        let next = (self.current_idx + 1) % self.named_controllers.len();
        self.switch_to(next);
    }

    fn prev_graph(&mut self) {
        let prev = if self.current_idx == 0 {
            self.named_controllers.len() - 1
        } else {
            self.current_idx - 1
        };
        self.switch_to(prev);
    }

    fn handle_key(&mut self, key: KeyEvent) {
        match key.code {
            // Graph switching: [ and ] or n and p
            KeyCode::Char(']') | KeyCode::Char('n') => self.next_graph(),
            KeyCode::Char('[') | KeyCode::Char('p') => self.prev_graph(),

            // Navigation within graph
            KeyCode::Left | KeyCode::Char('h') => {
                let ctrl = &mut self.named_controllers[self.current_idx].1;
                let vp_w = ctrl.viewport_state.viewport_bounds.width as i64;
                let delta = if ctrl.cursor.is_coarse_mode() {
                    -vp_w
                } else {
                    -1
                };
                let _ = ctrl.navigate_horizontal(delta);
                self.needs_rebuild = true;
            }
            KeyCode::Right | KeyCode::Char('l') => {
                let ctrl = &mut self.named_controllers[self.current_idx].1;
                let vp_w = ctrl.viewport_state.viewport_bounds.width as i64;
                let delta = if ctrl.cursor.is_coarse_mode() {
                    vp_w
                } else {
                    1
                };
                let _ = ctrl.navigate_horizontal(delta);
                self.needs_rebuild = true;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                let ctrl = &mut self.named_controllers[self.current_idx].1;
                let vp_h = ctrl.viewport_state.viewport_bounds.height as i64;
                let delta = if ctrl.cursor.is_coarse_mode() {
                    vp_h
                } else {
                    1
                };
                let _ = ctrl.navigate_vertical(delta);
                self.needs_rebuild = true;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let ctrl = &mut self.named_controllers[self.current_idx].1;
                let vp_h = ctrl.viewport_state.viewport_bounds.height as i64;
                let delta = if ctrl.cursor.is_coarse_mode() {
                    -vp_h
                } else {
                    -1
                };
                let _ = ctrl.navigate_vertical(delta);
                self.needs_rebuild = true;
            }

            // Zoom
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.named_controllers[self.current_idx].1.zoom_in();
                self.needs_rebuild = true;
            }
            KeyCode::Char('-') => {
                self.named_controllers[self.current_idx].1.zoom_out();
                self.needs_rebuild = true;
            }
            KeyCode::Char('r') => {
                self.named_controllers[self.current_idx].1.trigger_rebuild();
                self.needs_rebuild = true;
            }
            _ => {}
        }
    }

    fn render(&mut self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(frame.area());

        // Header
        let title = format!(
            " gen-tui  [{}/{}]  {} ",
            self.current_idx + 1,
            self.named_controllers.len(),
            self.current_name()
        );
        let header = Paragraph::new(title)
            .block(Block::bordered())
            .alignment(Alignment::Center)
            .fg(Color::White)
            .bg(Color::DarkGray);
        frame.render_widget(header, chunks[0]);

        // Graph area: set viewport bounds, then layout + render
        let graph_area = chunks[1];
        let ctrl = &mut self.named_controllers[self.current_idx].1;
        ctrl.viewport_state.viewport_bounds = graph_area;

        if self.needs_rebuild {
            let _ = ctrl.ensure_camera_coverage();
            let _ = ctrl.rebuild_viewport_graph();
            self.needs_rebuild = false;
        }

        // Collect everything we need from the controller before splitting borrows
        let ctrl = &mut self.named_controllers[self.current_idx].1;
        let viewport_graph = ctrl.get_viewport_graph();
        let detail_level = ctrl.get_detail_level();
        let graph_ref = ctrl.graph;
        let theme = ctrl.theme.clone();
        let viewport_state_snapshot = ctrl.viewport_state.clone();

        let buf = frame.buffer_mut();
        let mut world_buffer = WorldBuffer::new(buf, &viewport_state_snapshot);
        plot_viewport_graph(
            viewport_graph,
            &mut world_buffer,
            &mut self.renderer,
            graph_ref,
            detail_level,
            &theme,
        );

        // Footer
        let footer =
            Paragraph::new(" [/]: prev/next graph  |  h j k l / arrows: navigate  |  +/-: zoom ")
                .alignment(Alignment::Center)
                .fg(Color::DarkGray)
                .bg(Color::Black);
        frame.render_widget(footer, chunks[2]);
    }
}

fn main() -> io::Result<()> {
    console_error_panic_hook::set_once();
    let backend = DomBackend::new()?;
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
