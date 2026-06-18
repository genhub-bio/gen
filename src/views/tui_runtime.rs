#[cfg(all(feature = "browser-tui", not(feature = "native-tui")))]
use std::{io::Read as _, time::Instant};
use std::{
    io::{self, Write},
    time::Duration,
};

#[cfg(feature = "native-tui")]
use crossterm::{
    cursor::Show,
    event::{self, DisableMouseCapture, EnableMouseCapture, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
#[cfg(feature = "native-tui")]
use ratatui::backend::CrosstermBackend;
use ratatui::{
    Terminal,
    backend::{Backend, ClearType, WindowSize},
    buffer::Cell,
    layout::{Position, Size},
    style::{Color, Modifier},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenTuiEvent {
    Key(GenKeyEvent),
    Mouse(GenMouseEvent),
    Resize { cols: u16, rows: u16 },
    Tick,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GenKeyCode {
    Char(char),
    Enter,
    #[default]
    Esc,
    Backspace,
    Delete,
    Tab,
    BackTab,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    F(u8),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenKeyEvent {
    pub code: GenKeyCode,
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
}

impl From<GenKeyCode> for GenKeyEvent {
    fn from(code: GenKeyCode) -> Self {
        Self {
            code,
            ctrl: false,
            alt: false,
            shift: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenMouseEventKind {
    Down,
    Drag,
    Up,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenMouseButton {
    Left,
    Middle,
    Right,
    WheelUp,
    WheelDown,
    Other(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenMouseEvent {
    pub kind: GenMouseEventKind,
    pub button: GenMouseButton,
    pub column: u16,
    pub row: u16,
}

pub trait GenTuiRuntime {
    type Backend: Backend<Error = io::Error>;

    fn enter(&mut self) -> anyhow::Result<()>;
    fn leave(&mut self) -> anyhow::Result<()>;
    fn terminal(&mut self) -> &mut Terminal<Self::Backend>;
    fn poll_event(&mut self, timeout: Duration) -> anyhow::Result<Option<GenTuiEvent>>;
}

pub fn graph_controller_handle_key<G, S>(
    controller: &mut gen_tui::graph_controller::GraphController<G, S>,
    key: GenKeyEvent,
) -> Result<(), String>
where
    G: petgraph::visit::GraphBase
        + petgraph::visit::EdgeIndexable
        + petgraph::visit::NodeIndexable
        + petgraph::visit::NodeCount
        + petgraph::visit::Visitable,
    G::NodeId: Copy + Eq + std::hash::Hash + Ord,
    G::EdgeId: Clone,
    for<'b> &'b G: petgraph::visit::GraphBase<NodeId = G::NodeId, EdgeId = G::EdgeId>
        + petgraph::visit::IntoNodeIdentifiers<NodeId = G::NodeId>
        + petgraph::visit::IntoEdgeReferences<NodeId = G::NodeId, EdgeId = G::EdgeId>
        + petgraph::visit::IntoNeighborsDirected<NodeId = G::NodeId>,
    for<'b> &'b G::NodeId: std::hash::Hash + Ord,
    for<'b> &'b G::EdgeId: Clone,
    S: gen_tui::plotter::NodeSizer<G>,
{
    match key.code {
        GenKeyCode::Char('r') => {
            controller.trigger_rebuild();
        }
        GenKeyCode::Left | GenKeyCode::Char('h') => {
            let vp_w = controller.viewport_state.viewport_bounds.width as i64;
            let delta = if controller.cursor.is_coarse_mode() {
                -vp_w
            } else {
                -1
            };
            controller
                .cursor
                .move_horizontal(delta, &controller.viewport_graph)?;
            controller.trigger_rebuild();
        }
        GenKeyCode::Right | GenKeyCode::Char('l') => {
            let vp_w = controller.viewport_state.viewport_bounds.width as i64;
            let delta = if controller.cursor.is_coarse_mode() {
                vp_w
            } else {
                1
            };
            controller
                .cursor
                .move_horizontal(delta, &controller.viewport_graph)?;
            controller.trigger_rebuild();
        }
        GenKeyCode::Up | GenKeyCode::Char('k') => {
            let vp_h = controller.viewport_state.viewport_bounds.height as i64;
            let delta = if controller.cursor.is_coarse_mode() {
                vp_h
            } else {
                1
            };
            controller
                .cursor
                .move_vertical(delta, &controller.viewport_graph)?;
            controller.trigger_rebuild();
        }
        GenKeyCode::Down | GenKeyCode::Char('j') => {
            let vp_h = controller.viewport_state.viewport_bounds.height as i64;
            let delta = if controller.cursor.is_coarse_mode() {
                -vp_h
            } else {
                -1
            };
            controller
                .cursor
                .move_vertical(delta, &controller.viewport_graph)?;
            controller.trigger_rebuild();
        }
        GenKeyCode::Char('+') | GenKeyCode::Char('=') => {
            if key.shift {
                controller.disperse();
            } else {
                controller.zoom_in();
            }
        }
        GenKeyCode::Char('-') => {
            if key.shift {
                controller.contract();
            } else {
                controller.zoom_out();
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(feature = "native-tui")]
pub type GenTerminal = Terminal<CrosstermBackend<std::io::Stdout>>;

#[cfg(all(feature = "browser-tui", not(feature = "native-tui")))]
pub type GenTerminal = Terminal<BrowserAnsiBackend<std::io::Stdout>>;

#[cfg(feature = "native-tui")]
pub fn restore_terminal() -> io::Result<()> {
    disable_raw_mode()?;
    execute!(
        std::io::stdout(),
        DisableMouseCapture,
        LeaveAlternateScreen,
        Show
    )?;
    Ok(())
}

#[cfg(all(feature = "browser-tui", not(feature = "native-tui")))]
pub fn restore_terminal() -> io::Result<()> {
    let mut stdout = std::io::stdout();
    write!(
        stdout,
        "\x1b[?1000l\x1b[?1002l\x1b[?1006l\x1b[?2004l\x1b[?25h\x1b[?1049l"
    )?;
    stdout.flush()
}

pub fn install_global_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let _ = restore_terminal();
        eprintln!("The application has encountered an unexpected error and must exit.");
        eprintln!("Message: {}", info);
        eprintln!();
        eprintln!("Please file an issue at: https://github.com/genhub-bio/gen/issues");
        eprintln!("Include the full output above, what you were doing, and system info.");
    }));
}

pub struct TuiSession {
    terminal: GenTerminal,
    restored: bool,
    #[cfg(all(feature = "browser-tui", not(feature = "native-tui")))]
    parser: BrowserInputParser,
    #[cfg(all(feature = "browser-tui", not(feature = "native-tui")))]
    last_tick: Instant,
}

impl TuiSession {
    #[cfg(feature = "native-tui")]
    pub fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        if let Err(err) = execute!(stdout, EnterAlternateScreen, EnableMouseCapture) {
            let _ = disable_raw_mode();
            return Err(err);
        }

        match Terminal::new(CrosstermBackend::new(stdout)) {
            Ok(terminal) => Ok(Self {
                terminal,
                restored: false,
            }),
            Err(err) => {
                let _ = restore_terminal();
                Err(err)
            }
        }
    }

    #[cfg(all(feature = "browser-tui", not(feature = "native-tui")))]
    pub fn enter() -> io::Result<Self> {
        let size = browser_size_from_env();
        let mut stdout = std::io::stdout();
        write!(
            stdout,
            "\x1b[?1049h\x1b[?25l\x1b[?2004h\x1b[?1000h\x1b[?1002h\x1b[?1006h"
        )?;
        stdout.flush()?;

        match Terminal::new(BrowserAnsiBackend::new(stdout, size)) {
            Ok(terminal) => Ok(Self {
                terminal,
                restored: false,
                parser: BrowserInputParser::new(),
                last_tick: Instant::now(),
            }),
            Err(err) => {
                let _ = restore_terminal();
                Err(err)
            }
        }
    }

    pub fn terminal_mut(&mut self) -> &mut GenTerminal {
        &mut self.terminal
    }

    #[cfg(feature = "native-tui")]
    pub fn poll_event(&mut self, timeout: Duration) -> io::Result<Option<GenTuiEvent>> {
        if !event::poll(timeout)? {
            return Ok(None);
        }
        loop {
            match event::read()? {
                event::Event::Key(key) if key.kind == KeyEventKind::Press => {
                    return Ok(Some(GenTuiEvent::Key(GenKeyEvent::from(key))));
                }
                event::Event::Mouse(mouse) => {
                    if let Some(mouse) = GenMouseEvent::from_crossterm(mouse) {
                        return Ok(Some(GenTuiEvent::Mouse(mouse)));
                    }
                }
                event::Event::Resize(cols, rows) => {
                    return Ok(Some(GenTuiEvent::Resize { cols, rows }));
                }
                _ => {}
            }
        }
    }

    #[cfg(all(feature = "browser-tui", not(feature = "native-tui")))]
    pub fn poll_event(&mut self, timeout: Duration) -> io::Result<Option<GenTuiEvent>> {
        let start = Instant::now();
        let mut stdin = std::io::stdin();
        let mut buf = [0_u8; 64];

        if let Some(event) = self.parser.next_event() {
            if let GenTuiEvent::Resize { cols, rows } = event {
                self.terminal.backend_mut().set_size(Size::new(cols, rows));
            }
            return Ok(Some(event));
        }

        if timeout.is_zero() {
            return Ok(None);
        }

        while start.elapsed() < timeout {
            match stdin.read(&mut buf) {
                Ok(0) => break,
                Ok(count) => {
                    if let Some(event) = self.parser.push_bytes(&buf[..count]) {
                        if let GenTuiEvent::Resize { cols, rows } = event {
                            self.terminal.backend_mut().set_size(Size::new(cols, rows));
                        }
                        return Ok(Some(event));
                    }
                }
                Err(err) if err.kind() == io::ErrorKind::WouldBlock => break,
                Err(err) => return Err(err),
            }
        }

        if !timeout.is_zero() && self.last_tick.elapsed() >= timeout {
            self.last_tick = Instant::now();
            return Ok(Some(GenTuiEvent::Tick));
        }

        Ok(None)
    }

    pub fn restore(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }

        self.restored = true;
        self.terminal.show_cursor().ok();

        #[cfg(feature = "native-tui")]
        {
            disable_raw_mode()?;
            execute!(
                self.terminal.backend_mut(),
                DisableMouseCapture,
                LeaveAlternateScreen,
                Show
            )?;
        }

        #[cfg(all(feature = "browser-tui", not(feature = "native-tui")))]
        {
            write!(
                self.terminal.backend_mut().writer_mut(),
                "\x1b[?1000l\x1b[?1002l\x1b[?1006l\x1b[?2004l\x1b[?25h\x1b[?1049l"
            )?;
            self.terminal.backend_mut().flush()?;
        }

        Ok(())
    }
}

impl GenTuiRuntime for TuiSession {
    type Backend = <GenTerminal as TerminalBackend>::Backend;

    fn enter(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    fn leave(&mut self) -> anyhow::Result<()> {
        self.restore()?;
        Ok(())
    }

    fn terminal(&mut self) -> &mut Terminal<Self::Backend> {
        self.terminal_mut()
    }

    fn poll_event(&mut self, timeout: Duration) -> anyhow::Result<Option<GenTuiEvent>> {
        Ok(TuiSession::poll_event(self, timeout)?)
    }
}

pub trait TerminalBackend {
    type Backend: Backend<Error = io::Error>;
}

impl<B> TerminalBackend for Terminal<B>
where
    B: Backend<Error = io::Error>,
{
    type Backend = B;
}

impl Drop for TuiSession {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

#[cfg(feature = "native-tui")]
impl From<crossterm::event::KeyEvent> for GenKeyEvent {
    fn from(key: crossterm::event::KeyEvent) -> Self {
        let modifiers = key.modifiers;
        Self {
            code: GenKeyCode::from(key.code),
            ctrl: modifiers.contains(crossterm::event::KeyModifiers::CONTROL),
            alt: modifiers.contains(crossterm::event::KeyModifiers::ALT),
            shift: modifiers.contains(crossterm::event::KeyModifiers::SHIFT),
        }
    }
}

#[cfg(feature = "native-tui")]
impl From<crossterm::event::KeyCode> for GenKeyCode {
    fn from(code: crossterm::event::KeyCode) -> Self {
        match code {
            crossterm::event::KeyCode::Backspace => Self::Backspace,
            crossterm::event::KeyCode::Enter => Self::Enter,
            crossterm::event::KeyCode::Left => Self::Left,
            crossterm::event::KeyCode::Right => Self::Right,
            crossterm::event::KeyCode::Up => Self::Up,
            crossterm::event::KeyCode::Down => Self::Down,
            crossterm::event::KeyCode::Home => Self::Home,
            crossterm::event::KeyCode::End => Self::End,
            crossterm::event::KeyCode::PageUp => Self::PageUp,
            crossterm::event::KeyCode::PageDown => Self::PageDown,
            crossterm::event::KeyCode::Tab => Self::Tab,
            crossterm::event::KeyCode::BackTab => Self::BackTab,
            crossterm::event::KeyCode::Delete => Self::Delete,
            crossterm::event::KeyCode::Insert => Self::Char(' '),
            crossterm::event::KeyCode::F(n) => Self::F(n),
            crossterm::event::KeyCode::Char(c) => Self::Char(c),
            crossterm::event::KeyCode::Esc => Self::Esc,
            _ => Self::Esc,
        }
    }
}

#[cfg(feature = "native-tui")]
impl GenMouseEvent {
    fn from_crossterm(mouse: crossterm::event::MouseEvent) -> Option<Self> {
        use crossterm::event::{MouseButton, MouseEventKind};

        let (kind, button) = match mouse.kind {
            MouseEventKind::Down(button) => (GenMouseEventKind::Down, button),
            MouseEventKind::Drag(button) => (GenMouseEventKind::Drag, button),
            MouseEventKind::Up(button) => (GenMouseEventKind::Up, button),
            MouseEventKind::ScrollUp => (GenMouseEventKind::Down, MouseButton::Left),
            MouseEventKind::ScrollDown => (GenMouseEventKind::Down, MouseButton::Right),
            _ => return None,
        };

        Some(Self {
            kind,
            button: match button {
                MouseButton::Left => GenMouseButton::Left,
                MouseButton::Middle => GenMouseButton::Middle,
                MouseButton::Right => GenMouseButton::Right,
            },
            column: mouse.column,
            row: mouse.row,
        })
    }
}

#[derive(Debug, Default)]
pub struct BrowserInputParser {
    pending: Vec<u8>,
}

impl BrowserInputParser {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> Option<GenTuiEvent> {
        self.pending.extend_from_slice(bytes);
        self.next_event()
    }

    fn next_event(&mut self) -> Option<GenTuiEvent> {
        if self.pending.is_empty() {
            return None;
        }

        if self.pending[0] != 0x1b {
            let byte = self.pending.remove(0);
            return key_from_byte(byte).map(GenTuiEvent::Key);
        }

        if self.pending.len() == 1 {
            return None;
        }

        if self.pending.starts_with(b"\x1b]") {
            return self.parse_osc();
        }

        if self.pending.starts_with(b"\x1b[<") {
            return self.parse_sgr_mouse();
        }

        if let Some(event) = self.parse_escape_key() {
            return Some(event);
        }

        Some(GenTuiEvent::Key(GenKeyCode::Esc.into()))
    }

    fn parse_osc(&mut self) -> Option<GenTuiEvent> {
        let terminator = self.pending.iter().position(|byte| *byte == b'\x07')?;
        let payload = String::from_utf8_lossy(&self.pending[2..terminator]);
        let mut parts = payload.split(';');
        let event = match (
            parts.next(),
            parts.next(),
            parts.next().and_then(|part| part.parse::<u16>().ok()),
            parts.next().and_then(|part| part.parse::<u16>().ok()),
        ) {
            (Some("777"), Some("resize"), Some(cols), Some(rows)) => {
                Some(GenTuiEvent::Resize { cols, rows })
            }
            _ => None,
        };
        self.pending.drain(..=terminator);
        event
    }

    fn parse_sgr_mouse(&mut self) -> Option<GenTuiEvent> {
        let terminator = self
            .pending
            .iter()
            .position(|byte| *byte == b'M' || *byte == b'm')?;
        let release = self.pending[terminator] == b'm';
        let payload = String::from_utf8_lossy(&self.pending[3..terminator]);
        let mut parts = payload.split(';');
        let button_code = parts.next()?.parse::<u16>().ok()?;
        let column = parts.next()?.parse::<u16>().ok()?.saturating_sub(1);
        let row = parts.next()?.parse::<u16>().ok()?.saturating_sub(1);
        self.pending.drain(..=terminator);

        let kind = if release {
            GenMouseEventKind::Up
        } else if button_code & 32 != 0 {
            GenMouseEventKind::Drag
        } else {
            GenMouseEventKind::Down
        };
        let button = match button_code & 0b11 {
            0 => GenMouseButton::Left,
            1 => GenMouseButton::Middle,
            2 => GenMouseButton::Right,
            other => GenMouseButton::Other(other),
        };
        Some(GenTuiEvent::Mouse(GenMouseEvent {
            kind,
            button,
            column,
            row,
        }))
    }

    fn parse_escape_key(&mut self) -> Option<GenTuiEvent> {
        let key = match self.pending.as_slice() {
            bytes if bytes.starts_with(b"\x1b[A") => Some(GenKeyCode::Up.into()),
            bytes if bytes.starts_with(b"\x1b[B") => Some(GenKeyCode::Down.into()),
            bytes if bytes.starts_with(b"\x1b[C") => Some(GenKeyCode::Right.into()),
            bytes if bytes.starts_with(b"\x1b[D") => Some(GenKeyCode::Left.into()),
            bytes if bytes.starts_with(b"\x1b[H") => Some(GenKeyCode::Home.into()),
            bytes if bytes.starts_with(b"\x1b[F") => Some(GenKeyCode::End.into()),
            bytes if bytes.starts_with(b"\x1b[Z") => Some(GenKeyEvent {
                code: GenKeyCode::BackTab,
                shift: true,
                ..GenKeyEvent::default()
            }),
            bytes if bytes.starts_with(b"\x1b[3~") => Some(GenKeyCode::Delete.into()),
            bytes if bytes.starts_with(b"\x1b[5~") => Some(GenKeyCode::PageUp.into()),
            bytes if bytes.starts_with(b"\x1b[6~") => Some(GenKeyCode::PageDown.into()),
            bytes if bytes.starts_with(b"\x1bOH") => Some(GenKeyCode::Home.into()),
            bytes if bytes.starts_with(b"\x1bOF") => Some(GenKeyCode::End.into()),
            bytes if bytes.len() >= 2 && bytes[0] == 0x1b => {
                if let Some(mut key) = key_from_byte(bytes[1]) {
                    key.alt = true;
                    Some(key)
                } else {
                    Some(GenKeyCode::Esc.into())
                }
            }
            _ => None,
        }?;

        let drain = match key.code {
            GenKeyCode::Delete | GenKeyCode::PageUp | GenKeyCode::PageDown => 4,
            GenKeyCode::BackTab => 3,
            _ if self.pending.starts_with(b"\x1bO") => 3,
            _ if self.pending.len() >= 3 && self.pending[1] == b'[' => 3,
            _ => 2,
        };
        self.pending.drain(..drain.min(self.pending.len()));
        Some(GenTuiEvent::Key(key))
    }
}

fn key_from_byte(byte: u8) -> Option<GenKeyEvent> {
    let code = match byte {
        b'\r' | b'\n' => GenKeyCode::Enter,
        b'\t' => GenKeyCode::Tab,
        0x7f | 0x08 => GenKeyCode::Backspace,
        0x03 => {
            return Some(GenKeyEvent {
                code: GenKeyCode::Char('c'),
                ctrl: true,
                ..GenKeyEvent::default()
            });
        }
        0x04 => {
            return Some(GenKeyEvent {
                code: GenKeyCode::Char('d'),
                ctrl: true,
                ..GenKeyEvent::default()
            });
        }
        byte if byte.is_ascii_graphic() || byte == b' ' => GenKeyCode::Char(byte as char),
        _ => return None,
    };
    Some(code.into())
}

#[cfg(all(feature = "browser-tui", not(feature = "native-tui")))]
fn browser_size_from_env() -> Size {
    let cols = std::env::var("GEN_TUI_COLS")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(120);
    let rows = std::env::var("GEN_TUI_ROWS")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(40);
    Size::new(cols, rows)
}

pub struct BrowserAnsiBackend<W: Write> {
    writer: W,
    size: Size,
    cursor: Position,
}

impl<W: Write> BrowserAnsiBackend<W> {
    pub fn new(writer: W, size: Size) -> Self {
        Self {
            writer,
            size,
            cursor: Position::ORIGIN,
        }
    }

    pub fn set_size(&mut self, size: Size) {
        self.size = size;
    }

    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.writer
    }
}

impl<W: Write> Backend for BrowserAnsiBackend<W> {
    type Error = io::Error;

    fn draw<'a, I>(&mut self, content: I) -> io::Result<()>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        let mut fg = Color::Reset;
        let mut bg = Color::Reset;
        let mut modifier = Modifier::empty();
        let mut last_pos: Option<Position> = None;

        for (x, y, cell) in content {
            if !matches!(last_pos, Some(pos) if x == pos.x + 1 && y == pos.y) {
                write!(self.writer, "\x1b[{};{}H", y + 1, x + 1)?;
            }
            last_pos = Some(Position { x, y });

            if cell.modifier != modifier {
                write!(self.writer, "\x1b[0m")?;
                write_style(&mut self.writer, cell.fg, cell.bg, cell.modifier)?;
                modifier = cell.modifier;
                fg = cell.fg;
                bg = cell.bg;
            } else if cell.fg != fg || cell.bg != bg {
                write_colors(&mut self.writer, cell.fg, cell.bg)?;
                fg = cell.fg;
                bg = cell.bg;
            }

            write!(self.writer, "{}", cell.symbol())?;
        }

        write!(self.writer, "\x1b[0m")?;
        Ok(())
    }

    fn hide_cursor(&mut self) -> io::Result<()> {
        write!(self.writer, "\x1b[?25l")
    }

    fn show_cursor(&mut self) -> io::Result<()> {
        write!(self.writer, "\x1b[?25h")
    }

    fn get_cursor_position(&mut self) -> io::Result<Position> {
        Ok(self.cursor)
    }

    fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> io::Result<()> {
        let position = position.into();
        self.cursor = position;
        write!(self.writer, "\x1b[{};{}H", position.y + 1, position.x + 1)
    }

    fn clear(&mut self) -> io::Result<()> {
        write!(self.writer, "\x1b[2J")
    }

    fn clear_region(&mut self, clear_type: ClearType) -> io::Result<()> {
        match clear_type {
            ClearType::All => write!(self.writer, "\x1b[2J"),
            ClearType::AfterCursor => write!(self.writer, "\x1b[J"),
            ClearType::BeforeCursor => write!(self.writer, "\x1b[1J"),
            ClearType::CurrentLine => write!(self.writer, "\x1b[2K"),
            ClearType::UntilNewLine => write!(self.writer, "\x1b[K"),
        }
    }

    fn size(&self) -> io::Result<Size> {
        Ok(self.size)
    }

    fn window_size(&mut self) -> io::Result<WindowSize> {
        Ok(WindowSize {
            columns_rows: self.size,
            pixels: Size::new(0, 0),
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

fn write_style<W: Write>(
    writer: &mut W,
    fg: Color,
    bg: Color,
    modifier: Modifier,
) -> io::Result<()> {
    if modifier.contains(Modifier::BOLD) {
        write!(writer, "\x1b[1m")?;
    }
    if modifier.contains(Modifier::ITALIC) {
        write!(writer, "\x1b[3m")?;
    }
    if modifier.contains(Modifier::UNDERLINED) {
        write!(writer, "\x1b[4m")?;
    }
    if modifier.contains(Modifier::REVERSED) {
        write!(writer, "\x1b[7m")?;
    }
    write_colors(writer, fg, bg)
}

fn write_colors<W: Write>(writer: &mut W, fg: Color, bg: Color) -> io::Result<()> {
    write_color(writer, fg, false)?;
    write_color(writer, bg, true)
}

fn write_color<W: Write>(writer: &mut W, color: Color, background: bool) -> io::Result<()> {
    let base = if background { 48 } else { 38 };
    match color {
        Color::Reset => write!(writer, "\x1b[{}m", if background { 49 } else { 39 }),
        Color::Black => write!(writer, "\x1b[{};5;0m", base),
        Color::Red => write!(writer, "\x1b[{};5;1m", base),
        Color::Green => write!(writer, "\x1b[{};5;2m", base),
        Color::Yellow => write!(writer, "\x1b[{};5;3m", base),
        Color::Blue => write!(writer, "\x1b[{};5;4m", base),
        Color::Magenta => write!(writer, "\x1b[{};5;5m", base),
        Color::Cyan => write!(writer, "\x1b[{};5;6m", base),
        Color::Gray => write!(writer, "\x1b[{};5;7m", base),
        Color::DarkGray => write!(writer, "\x1b[{};5;8m", base),
        Color::LightRed => write!(writer, "\x1b[{};5;9m", base),
        Color::LightGreen => write!(writer, "\x1b[{};5;10m", base),
        Color::LightYellow => write!(writer, "\x1b[{};5;11m", base),
        Color::LightBlue => write!(writer, "\x1b[{};5;12m", base),
        Color::LightMagenta => write!(writer, "\x1b[{};5;13m", base),
        Color::LightCyan => write!(writer, "\x1b[{};5;14m", base),
        Color::White => write!(writer, "\x1b[{};5;15m", base),
        Color::Indexed(index) => write!(writer, "\x1b[{};5;{}m", base, index),
        Color::Rgb(r, g, b) => write!(writer, "\x1b[{};2;{};{};{}m", base, r, g, b),
    }
}

#[cfg(test)]
mod tests {
    use super::{BrowserInputParser, GenKeyCode, GenTuiEvent};

    #[test]
    fn browser_parser_reads_resize_osc() {
        let mut parser = BrowserInputParser::new();

        let event = parser.push_bytes(b"\x1b]777;resize;120;40\x07");

        assert_eq!(
            event,
            Some(GenTuiEvent::Resize {
                cols: 120,
                rows: 40
            })
        );
    }

    #[test]
    fn browser_parser_reads_arrow_key() {
        let mut parser = BrowserInputParser::new();

        let event = parser.push_bytes(b"\x1b[A");

        assert_eq!(event, Some(GenTuiEvent::Key(GenKeyCode::Up.into())));
    }
}
