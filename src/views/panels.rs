use gen_tui::theme::current_theme;
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Clear, Paragraph, Widget},
};

use crate::views::helpers::style_text;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusMode {
    Active,
    Navigation,
}

/// Tracks panel order, selected panel, and whether we are in active or
/// navigation mode.
pub struct PanelFocus<T>
where
    T: Copy + Eq,
{
    order: Vec<T>,
    index: usize,
    mode: FocusMode,
}

impl<T> PanelFocus<T>
where
    T: Copy + Eq,
{
    pub fn new(initial: T) -> Self {
        Self {
            order: vec![initial],
            index: 0,
            mode: FocusMode::Active,
        }
    }

    pub fn current(&self) -> T {
        self.order[self.index]
    }

    pub fn is_active(&self) -> bool {
        self.mode == FocusMode::Active
    }

    pub fn is_navigation(&self) -> bool {
        self.mode == FocusMode::Navigation
    }

    pub fn activate(&mut self) {
        self.mode = FocusMode::Active;
    }

    pub fn deactivate(&mut self) {
        self.mode = FocusMode::Navigation;
    }

    pub fn is_selected(&self, panel: T) -> bool {
        self.current() == panel
    }

    pub fn is_active_panel(&self, panel: T) -> bool {
        self.is_active() && self.is_selected(panel)
    }

    pub fn is_navigation_selected(&self, panel: T) -> bool {
        self.is_navigation() && self.is_selected(panel)
    }

    pub fn focus(&mut self, panel: T) -> bool {
        if let Some(index) = self.order.iter().position(|candidate| *candidate == panel) {
            self.index = index;
            return true;
        }
        false
    }

    pub fn include_panel(&mut self, panel: T) -> usize {
        if let Some(index) = self.order.iter().position(|candidate| *candidate == panel) {
            return index;
        }
        self.order.push(panel);
        self.order.len() - 1
    }

    pub fn remove_panel(&mut self, panel: T) -> bool {
        if self.order.len() <= 1 {
            return false;
        }
        let Some(removed_index) = self.order.iter().position(|candidate| *candidate == panel)
        else {
            return false;
        };

        self.order.remove(removed_index);

        if self.index > removed_index {
            self.index -= 1;
        } else if self.index >= self.order.len() {
            self.index = 0;
        }

        true
    }

    pub fn cycle_next(&mut self) -> T {
        self.index += 1;
        if self.index >= self.order.len() {
            self.index = 0;
        }
        self.current()
    }

    pub fn cycle_prev(&mut self) -> T {
        if self.index == 0 {
            self.index = self.order.len() - 1;
        } else {
            self.index -= 1;
        }
        self.current()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PanelStyles {
    pub focused: Style,
    pub selected: Style,
    pub unfocused: Style,
}

impl Default for PanelStyles {
    fn default() -> Self {
        Self {
            focused: Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD),
            selected: Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
            unfocused: Style::default().fg(Color::Gray),
        }
    }
}

pub fn panel_block<T>(
    title: impl Into<String>,
    focus: &PanelFocus<T>,
    panel: T,
    styles: PanelStyles,
) -> Block<'static>
where
    T: Copy + Eq,
{
    let base_title = title.into();
    let display_title = if focus.is_navigation_selected(panel) {
        format!("[{base_title}]")
    } else {
        base_title
    };

    let border_style = if focus.is_active_panel(panel) {
        styles.focused
    } else if focus.is_navigation_selected(panel) {
        styles.selected
    } else {
        styles.unfocused
    };

    Block::default()
        .title(display_title)
        .borders(Borders::ALL)
        .border_style(border_style)
}

pub fn render_status_bar(frame: &mut Frame, area: Rect, message: &str) {
    let theme = current_theme();
    let status_bar_contents = format!("{message:^width$}", width = area.width as usize);
    let status_line = style_text(
        &status_bar_contents,
        Style::default().fg(theme[0x04]),
        Style::default().fg(theme[0x07]),
    );
    let status_bar = Paragraph::new(status_line).style(Style::default().bg(theme[0x00]));
    frame.render_widget(status_bar, area);
}

pub fn render_with_optional_clear<W: Widget>(
    frame: &mut Frame,
    clear_area: Rect,
    render_area: Rect,
    clear_first: bool,
    widget: W,
) {
    if clear_first {
        frame.render_widget(Clear, clear_area);
    }
    frame.render_widget(widget, render_area);
}
