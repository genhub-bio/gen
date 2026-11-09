use ratatui::style::Color;

use crate::theme::current_theme;

/// Pixel cell dimensions, scale and colors used by the pixel overlay, both when it is shown
/// through terminal graphics and when it is written into a PNG. Sizes are in pixels of the
/// rendered image. [`Self::with_scale`] derives a sharper config from a logical one: the same
/// grid of cells with every length multiplied, so a browser can show the image at its logical
/// size and use the extra pixels on high-density screens.
#[derive(Clone, Copy, Debug)]
pub struct RasterConfig {
    pub cell_width: usize,
    pub cell_height: usize,
    /// How many image pixels draw one logical pixel. Strokes, paddings and offsets that are a
    /// fixed number of logical pixels are multiplied by it.
    pub scale: usize,
    /// Used for cells whose foreground is `Color::Reset`.
    pub foreground: [u8; 3],
    /// Used for cells whose background is `Color::Reset`, and to fill the canvas.
    pub background: [u8; 3],
}

impl Default for RasterConfig {
    /// An 11x21 pixel cell with the current theme's text and canvas colors.
    fn default() -> Self {
        let theme = current_theme();
        Self {
            cell_width: 11,
            cell_height: 21,
            scale: 1,
            foreground: color_to_rgb(theme[0x05], [255; 3]),
            background: color_to_rgb(theme[0x00], [0; 3]),
        }
    }
}

impl RasterConfig {
    /// This config drawn at `scale` image pixels per logical pixel, so a scale of 2 writes an
    /// image twice as wide and tall with the same cell grid. A scale of 0 is treated as 1.
    /// The PNG declares 96 dpi times the scale, so 2 is 192 dpi and 3 is 288 dpi, close to a
    /// 300 dpi print.
    pub fn with_scale(self, scale: usize) -> Self {
        let scale = scale.max(1);
        let (cell_width, cell_height) = self.logical_cell_size();
        Self {
            cell_width: cell_width * scale,
            cell_height: cell_height * scale,
            scale,
            ..self
        }
    }

    /// The cell size in logical pixels, which does not change with the scale. A viewer maps
    /// pointer positions to cells with it.
    pub fn logical_cell_size(&self) -> (usize, usize) {
        let scale = self.scale.max(1);
        (self.cell_width / scale, self.cell_height / scale)
    }
}

/// The RGB value a terminal color is drawn with, or `default` for `Color::Reset` and colors
/// the exporter has no value for.
pub fn color_to_rgb(color: Color, default: [u8; 3]) -> [u8; 3] {
    match color {
        Color::Reset => default,
        Color::Black => [0, 0, 0],
        Color::Red => [205, 49, 49],
        Color::Green => [13, 188, 121],
        Color::Yellow => [229, 229, 16],
        Color::Blue => [36, 114, 200],
        Color::Magenta => [188, 63, 188],
        Color::Cyan => [17, 168, 205],
        Color::Gray => [229, 229, 229],
        Color::DarkGray => [102, 102, 102],
        Color::LightRed => [241, 76, 76],
        Color::LightGreen => [35, 209, 139],
        Color::LightYellow => [245, 245, 67],
        Color::LightBlue => [59, 142, 234],
        Color::LightMagenta => [214, 112, 214],
        Color::LightCyan => [41, 184, 219],
        Color::White => [255, 255, 255],
        Color::Rgb(red, green, blue) => [red, green, blue],
        Color::Indexed(_) => default,
    }
}
