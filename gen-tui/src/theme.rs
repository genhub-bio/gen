use std::str::FromStr;

use ratatui::style::Color;

/// Theme configuration for graph rendering
#[derive(Debug, Clone, PartialEq)]
pub struct Theme {
    pub canvas: Color,
    pub node_fg: Color,
    pub node_bg: Color,
    pub edge_fg: Color,
    pub edge_bg: Color,
    pub cursor_fg: Color,
    pub cursor_bg: Color,
    pub highlight: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            canvas: Color::Reset,
            node_fg: Color::from_str("#cdd6f4").unwrap_or(Color::White), // base04
            node_bg: Color::from_str("#45475A").unwrap_or(Color::Blue),  // base03
            edge_fg: Color::from_str("#45475A").unwrap_or(Color::Gray),  // node_bg
            edge_bg: Color::Reset,                                       // canvas
            cursor_fg: Color::from_str("#45475A").unwrap_or(Color::Blue), // node_bg
            cursor_bg: Color::from_str("#cdd6f4").unwrap_or(Color::White), // node_fg
            highlight: Color::Cyan,
        }
    }
}
