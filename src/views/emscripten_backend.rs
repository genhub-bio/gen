//! Minimal `ratatui::backend::Backend` implementation for `target_os = "emscripten"`.
//!
//! Ratatui's built-in `CrosstermBackend` lives in the separate `ratatui-crossterm` crate, which
//! depends on `crossterm` with its **default** features enabled -- including `events`, which
//! requires `mio` (no emscripten backend; see `emscripten_input.rs`). Cargo unifies features per
//! package version across the whole dependency graph, so merely linking `ratatui-crossterm`
//! anywhere would re-enable `events` globally and break this target, regardless of our own
//! `crossterm = { default-features = false }` override in `Cargo.toml`.
//!
//! This backend only calls `crossterm::{cursor, style, terminal}`, none of which are gated behind
//! the `events` feature (confirmed: only `event` and `osc52` are `#[cfg(feature = ...)]` in
//! crossterm's `lib.rs`), so it renders correctly with our reduced-feature dependency.

#[cfg(target_os = "emscripten")]
use std::io::{self, Write};

#[cfg(target_os = "emscripten")]
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    execute, queue,
    style::{
        Attribute as CrosstermAttribute, Color as CrosstermColor, Colors as CrosstermColors, Print,
        SetAttribute, SetBackgroundColor, SetColors, SetForegroundColor, SetUnderlineColor,
    },
    terminal::{self, Clear},
};
#[cfg(target_os = "emscripten")]
use ratatui::{
    backend::{Backend, ClearType, WindowSize},
    buffer::Cell,
    layout::{Position, Size},
    style::{Color, Modifier},
};

#[cfg(target_os = "emscripten")]
fn color_to_crossterm(color: Color) -> CrosstermColor {
    match color {
        Color::Reset => CrosstermColor::Reset,
        Color::Black => CrosstermColor::Black,
        Color::Red => CrosstermColor::DarkRed,
        Color::Green => CrosstermColor::DarkGreen,
        Color::Yellow => CrosstermColor::DarkYellow,
        Color::Blue => CrosstermColor::DarkBlue,
        Color::Magenta => CrosstermColor::DarkMagenta,
        Color::Cyan => CrosstermColor::DarkCyan,
        Color::Gray => CrosstermColor::Grey,
        Color::DarkGray => CrosstermColor::DarkGrey,
        Color::LightRed => CrosstermColor::Red,
        Color::LightGreen => CrosstermColor::Green,
        Color::LightBlue => CrosstermColor::Blue,
        Color::LightYellow => CrosstermColor::Yellow,
        Color::LightMagenta => CrosstermColor::Magenta,
        Color::LightCyan => CrosstermColor::Cyan,
        Color::White => CrosstermColor::White,
        Color::Indexed(index) => CrosstermColor::AnsiValue(index),
        Color::Rgb(red, green, blue) => CrosstermColor::Rgb {
            r: red,
            g: green,
            b: blue,
        },
    }
}

/// Queues the SGR attribute commands needed to move from `from` to `to`.
#[cfg(target_os = "emscripten")]
fn queue_modifier_diff<W: Write>(writer: &mut W, from: Modifier, to: Modifier) -> io::Result<()> {
    let removed = from - to;
    if removed.contains(Modifier::REVERSED) {
        queue!(writer, SetAttribute(CrosstermAttribute::NoReverse))?;
    }
    let reset_intensity = removed.contains(Modifier::BOLD) || removed.contains(Modifier::DIM);
    if reset_intensity {
        queue!(writer, SetAttribute(CrosstermAttribute::NormalIntensity))?;
        if to.contains(Modifier::DIM) {
            queue!(writer, SetAttribute(CrosstermAttribute::Dim))?;
        }
        if to.contains(Modifier::BOLD) {
            queue!(writer, SetAttribute(CrosstermAttribute::Bold))?;
        }
    }
    if removed.contains(Modifier::ITALIC) {
        queue!(writer, SetAttribute(CrosstermAttribute::NoItalic))?;
    }
    if removed.contains(Modifier::UNDERLINED) {
        queue!(writer, SetAttribute(CrosstermAttribute::NoUnderline))?;
    }
    if removed.contains(Modifier::CROSSED_OUT) {
        queue!(writer, SetAttribute(CrosstermAttribute::NotCrossedOut))?;
    }
    if removed.contains(Modifier::HIDDEN) {
        queue!(writer, SetAttribute(CrosstermAttribute::NoHidden))?;
    }
    if removed.contains(Modifier::SLOW_BLINK) || removed.contains(Modifier::RAPID_BLINK) {
        queue!(writer, SetAttribute(CrosstermAttribute::NoBlink))?;
    }

    let added = to - from;
    if added.contains(Modifier::REVERSED) {
        queue!(writer, SetAttribute(CrosstermAttribute::Reverse))?;
    }
    if added.contains(Modifier::BOLD) && !reset_intensity {
        queue!(writer, SetAttribute(CrosstermAttribute::Bold))?;
    }
    if added.contains(Modifier::ITALIC) {
        queue!(writer, SetAttribute(CrosstermAttribute::Italic))?;
    }
    if added.contains(Modifier::UNDERLINED) {
        queue!(writer, SetAttribute(CrosstermAttribute::Underlined))?;
    }
    if added.contains(Modifier::DIM) && !reset_intensity {
        queue!(writer, SetAttribute(CrosstermAttribute::Dim))?;
    }
    if added.contains(Modifier::CROSSED_OUT) {
        queue!(writer, SetAttribute(CrosstermAttribute::CrossedOut))?;
    }
    if added.contains(Modifier::HIDDEN) {
        queue!(writer, SetAttribute(CrosstermAttribute::Hidden))?;
    }
    if added.contains(Modifier::SLOW_BLINK) {
        queue!(writer, SetAttribute(CrosstermAttribute::SlowBlink))?;
    }
    if added.contains(Modifier::RAPID_BLINK) {
        queue!(writer, SetAttribute(CrosstermAttribute::RapidBlink))?;
    }
    Ok(())
}

/// A [`Backend`] implementation that writes directly through `crossterm::{cursor, style,
/// terminal}`, skipping `ratatui-crossterm`/`crossterm::event` (and thus `mio`) entirely.
#[cfg(target_os = "emscripten")]
pub struct EmscriptenBackend<W: Write> {
    writer: W,
}

#[cfg(target_os = "emscripten")]
impl<W: Write> EmscriptenBackend<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

#[cfg(target_os = "emscripten")]
impl<W: Write> Backend for EmscriptenBackend<W> {
    type Error = io::Error;

    fn draw<'a, I>(&mut self, content: I) -> io::Result<()>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        let mut fg = Color::Reset;
        let mut bg = Color::Reset;
        let mut underline_color = Color::Reset;
        let mut modifier = Modifier::empty();
        let mut last_pos: Option<Position> = None;
        for (x, y, cell) in content {
            if !matches!(last_pos, Some(p) if x == p.x + 1 && y == p.y) {
                queue!(self.writer, MoveTo(x, y))?;
            }
            last_pos = Some(Position { x, y });
            if cell.modifier != modifier {
                queue_modifier_diff(&mut self.writer, modifier, cell.modifier)?;
                modifier = cell.modifier;
            }
            if cell.fg != fg || cell.bg != bg {
                queue!(
                    self.writer,
                    SetColors(CrosstermColors::new(
                        color_to_crossterm(cell.fg),
                        color_to_crossterm(cell.bg),
                    ))
                )?;
                fg = cell.fg;
                bg = cell.bg;
            }
            if cell.underline_color != underline_color {
                queue!(
                    self.writer,
                    SetUnderlineColor(color_to_crossterm(cell.underline_color))
                )?;
                underline_color = cell.underline_color;
            }
            queue!(self.writer, Print(cell.symbol()))?;
        }

        queue!(
            self.writer,
            SetForegroundColor(CrosstermColor::Reset),
            SetBackgroundColor(CrosstermColor::Reset),
            SetUnderlineColor(CrosstermColor::Reset),
            SetAttribute(CrosstermAttribute::Reset),
        )
    }

    fn hide_cursor(&mut self) -> io::Result<()> {
        execute!(self.writer, Hide)
    }

    fn show_cursor(&mut self) -> io::Result<()> {
        execute!(self.writer, Show)
    }

    fn get_cursor_position(&mut self) -> io::Result<Position> {
        crate::views::emscripten_input::read_cursor_position().map(|(x, y)| Position { x, y })
    }

    fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> io::Result<()> {
        let Position { x, y } = position.into();
        execute!(self.writer, MoveTo(x, y))
    }

    fn clear(&mut self) -> io::Result<()> {
        self.clear_region(ClearType::All)
    }

    fn clear_region(&mut self, clear_type: ClearType) -> io::Result<()> {
        execute!(
            self.writer,
            Clear(match clear_type {
                ClearType::All => crossterm::terminal::ClearType::All,
                ClearType::AfterCursor => crossterm::terminal::ClearType::FromCursorDown,
                ClearType::BeforeCursor => crossterm::terminal::ClearType::FromCursorUp,
                ClearType::CurrentLine => crossterm::terminal::ClearType::CurrentLine,
                ClearType::UntilNewLine => crossterm::terminal::ClearType::UntilNewLine,
            })
        )
    }

    fn append_lines(&mut self, n: u16) -> io::Result<()> {
        for _ in 0..n {
            queue!(self.writer, Print("\n"))?;
        }
        self.writer.flush()
    }

    fn size(&self) -> io::Result<Size> {
        let (width, height) = terminal::size()?;
        Ok(Size { width, height })
    }

    fn window_size(&mut self) -> io::Result<WindowSize> {
        let crossterm::terminal::WindowSize {
            columns,
            rows,
            width,
            height,
        } = terminal::window_size()?;
        Ok(WindowSize {
            columns_rows: Size {
                width: columns,
                height: rows,
            },
            pixels: Size { width, height },
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}
