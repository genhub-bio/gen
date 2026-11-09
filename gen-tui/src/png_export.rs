//! Rasterizes a Ratatui `Buffer` to a PNG so graph plots can be produced without a terminal or
//! notebook frontend (the CLI `--png` flag and the Python `save_png` methods).
//!
//! Pixels are produced one terminal row at a time and streamed into a single PNG, so pixel-buffer memory is
//! independent of image height (the input cell buffer still covers the whole view). The font, glyph cache and row-band allocation persist across
//! exports. Glyphs are single-width and clipped to their cell: no shaping, fallback fonts or
//! bold/italic faces.

use core::mem::swap;
use std::{
    collections::HashMap,
    io::{Error as IoError, Write},
};

use fontdue::{Font, FontSettings, Metrics};
use png::{BitDepth, ColorType, Compression, Encoder, EncodingError, PixelDimensions, Unit};
use ratatui::{
    buffer::{Buffer, Cell},
    layout::{Margin, Rect},
    style::{Color, Modifier},
};
use thiserror::Error;

use crate::theme::current_theme;

/// Blank cells kept around the content by [`PngRenderer::crop_to_content`].
pub const CROP_MARGIN: u16 = 1;

/// Largest pixel scale callers should accept: an image grows with the square of its scale.
pub const MAX_SCALE: usize = 8;

/// Scale callers draw at unless asked otherwise: twice the logical pixels stay sharp on
/// high-density screens, and the logical size is the same.
pub const DEFAULT_SCALE: usize = 2;

/// Pixels per meter at the logical 96 dpi, which a scale of one declares in the PNG header.
const LOGICAL_PIXELS_PER_METER: u32 = 3780;

const FONT_BYTES: &[u8] = include_bytes!("../fonts/JetBrainsMono-Regular.ttf");

#[derive(Debug, Error)]
pub enum PngExportError {
    #[error("{0}")]
    InvalidInput(&'static str),
    #[error("could not load the bundled font: {0}")]
    Font(&'static str),
    #[error("PNG encoding failed: {0}")]
    Png(#[from] EncodingError),
    #[error("output failed: {0}")]
    Io(#[from] IoError),
}

/// Pixel cell dimensions, scale and colors shared by PNG export and terminal graphics. Sizes are
/// in pixels of the rendered image. [`Self::with_scale`] derives a sharper config from a
/// logical one: the same grid of cells with every length multiplied, so a browser can show the
/// image at its logical size and use the extra pixels on high-density screens.
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

struct CachedGlyph {
    metrics: Metrics,
    mask: Vec<u8>,
}

/// Font and configuration are immutable, so a character identifies a cache entry. Construct
/// another `PngRenderer` to change the cell size or colors.
///
/// Text is sized to fit the cell and centered. Glyphs are clipped to their cell.
pub struct PngRenderer {
    font: Font,
    config: RasterConfig,
    text_size: f32,
    text_baseline: i64,
    glyphs: HashMap<char, CachedGlyph>,
    band: Vec<u8>,
}

impl PngRenderer {
    pub fn new(config: RasterConfig) -> Result<Self, PngExportError> {
        let font =
            Font::from_bytes(FONT_BYTES, FontSettings::default()).map_err(PngExportError::Font)?;
        const PROBE_SIZE: f32 = 100.0;
        let probe_line =
            font.horizontal_line_metrics(PROBE_SIZE)
                .ok_or(PngExportError::InvalidInput(
                    "font has no horizontal metrics",
                ))?;
        let text_padding = 3.0 * config.scale.max(1) as f32;
        let text_size = ((config.cell_width as f32 - text_padding).max(1.0) * PROBE_SIZE
            / font.metrics('M', PROBE_SIZE).advance_width)
            .min(config.cell_height as f32 * PROBE_SIZE / (probe_line.ascent - probe_line.descent));
        let line = font
            .horizontal_line_metrics(text_size)
            .ok_or(PngExportError::InvalidInput(
                "font has no horizontal metrics",
            ))?;
        let text_baseline = ((config.cell_height as f32 - (line.ascent - line.descent)) / 2.0
            + line.ascent)
            .round() as i64;
        Ok(Self {
            font,
            config,
            text_size,
            text_baseline,
            glyphs: HashMap::new(),
            band: Vec::new(),
        })
    }

    /// Stream a complete PNG of `buffer` into `output`. Use a file for large plots; a `Vec`
    /// retains the whole encoded image.
    pub fn write_png<W: Write>(
        &mut self,
        buffer: &Buffer,
        output: &mut W,
    ) -> Result<(), PngExportError> {
        self.write_png_with_overlay(buffer, output, |_, _, _| {})
    }

    /// Like [`Self::write_png`], but hands each canvas-colored band to `overlay` before cells
    /// are drawn, so a caller can draw behind node backgrounds and text (the edges of a curved
    /// graph) without this module
    /// knowing what is drawn. `overlay` receives the band's RGB pixels, the image width in
    /// pixels, and the image row of the band's first pixel; the band's height is
    /// `pixels.len() / (3 * width)`, and bands arrive top to bottom.
    pub fn write_png_with_overlay<W: Write>(
        &mut self,
        buffer: &Buffer,
        output: &mut W,
        mut overlay: impl FnMut(&mut [u8], usize, usize),
    ) -> Result<(), PngExportError> {
        let columns = usize::from(buffer.area.width);
        let rows = usize::from(buffer.area.height);
        if columns == 0 || rows == 0 {
            return Err(PngExportError::InvalidInput(
                "image dimensions must be nonzero",
            ));
        }
        let width = columns * self.config.cell_width;
        self.band.resize(width * self.config.cell_height * 3, 0);

        let mut encoder = Encoder::new(
            &mut *output,
            width as u32,
            (rows * self.config.cell_height) as u32,
        );
        encoder.set_color(ColorType::Rgb);
        encoder.set_depth(BitDepth::Eight);
        encoder.set_compression(Compression::Fast);
        // Declare the density so viewers that honor it show the image at its logical size.
        let pixels_per_meter = LOGICAL_PIXELS_PER_METER * self.config.scale as u32;
        encoder.set_pixel_dims(Some(PixelDimensions {
            xppu: pixels_per_meter,
            yppu: pixels_per_meter,
            unit: Unit::Meter,
        }));
        let mut writer = encoder.write_header()?;
        {
            // One stream spans all bands so compression is not restarted per row.
            let mut stream = writer.stream_writer_with_size(64 * 1024)?;
            for (row_index, row) in buffer.content.chunks_exact(columns).enumerate() {
                self.render_band(
                    row,
                    width,
                    row_index * self.config.cell_height,
                    &mut overlay,
                );
                stream.write_all(&self.band)?;
            }
            stream.finish()?;
        }
        writer.finish()?;
        output.flush()?;
        Ok(())
    }

    /// Trim blank rows and columns around the content, keeping `CROP_MARGIN` blank cells on each
    /// side where there was room. Plots are drawn onto a fixed-size canvas, so without this a small
    /// graph is lost in empty space. Returns the smallest sub-buffer containing every visible cell
    /// plus the margin. A cell is visible when
    /// it has a glyph, an underline, or a background that differs from the canvas. A buffer with
    /// no visible cells is returned unchanged.
    pub fn crop_to_content(&self, buffer: &Buffer) -> Buffer {
        Self::crop(buffer, self.content_crop(buffer, None))
    }

    /// The rectangle [`Self::crop_to_content`] keeps, in cells of `buffer`. `extra` names cells
    /// that hold content the buffer itself does not show, such as curved edge shapes, so
    /// they stay inside the crop.
    pub fn content_crop(&self, buffer: &Buffer, extra: Option<Rect>) -> Rect {
        let columns = usize::from(buffer.area.width);
        let content = buffer
            .content
            .iter()
            .enumerate()
            .filter(|(_, cell)| self.is_visible(cell))
            .map(|(index, _)| Rect::new((index % columns) as u16, (index / columns) as u16, 1, 1))
            .chain(extra)
            .reduce(|bounds, cell| bounds.union(cell));
        let whole = Rect::new(0, 0, buffer.area.width, buffer.area.height);
        let Some(content) = content else {
            return whole;
        };
        content
            .outer(Margin::new(CROP_MARGIN, CROP_MARGIN))
            .intersection(whole)
    }

    /// The part of `buffer` inside `crop`, a rectangle in its cells.
    pub fn crop(buffer: &Buffer, crop: Rect) -> Buffer {
        let columns = usize::from(buffer.area.width);
        let mut cropped = Buffer::empty(Rect::new(0, 0, crop.width, crop.height));
        for row in 0..usize::from(crop.height) {
            let source = (usize::from(crop.y) + row) * columns + usize::from(crop.x);
            let target = row * usize::from(crop.width);
            cropped.content[target..target + usize::from(crop.width)]
                .clone_from_slice(&buffer.content[source..source + usize::from(crop.width)]);
        }
        cropped
    }

    fn is_visible(&self, cell: &Cell) -> bool {
        let has_glyph = cell
            .symbol()
            .chars()
            .any(|character| !character.is_whitespace());
        let (_, background) = cell_colors(cell, self.config);
        has_glyph
            || cell.modifier.contains(Modifier::UNDERLINED)
            || background != self.config.background
    }

    fn render_band(
        &mut self,
        row: &[Cell],
        width: usize,
        top: usize,
        overlay: &mut impl FnMut(&mut [u8], usize, usize),
    ) {
        let config = self.config;
        for pixel in self.band.as_chunks_mut::<3>().0 {
            *pixel = config.background;
        }
        overlay(&mut self.band, width, top);
        // Node backgrounds and glyphs cover edge crossings, as they do in the terminal.
        for (column, cell) in row.iter().enumerate() {
            let (_, background) = cell_colors(cell, config);
            if background != config.background {
                fill_rect(
                    &mut self.band,
                    width,
                    (column * config.cell_width, 0),
                    (config.cell_width, config.cell_height),
                    background,
                );
            }
        }

        for (column, cell) in row.iter().enumerate() {
            let cell_x = column * config.cell_width;
            let (foreground, _) = cell_colors(cell, config);
            let glyph_char = cell
                .symbol()
                .chars()
                .next()
                .filter(|character| !character.is_whitespace());
            if let Some(character) = glyph_char
                && draw_sentinel_glyph(
                    &mut self.band,
                    width,
                    (cell_x, character),
                    config,
                    foreground,
                )
            {
                continue;
            }
            if let Some(character) = glyph_char {
                let font = &self.font;
                let glyph = self.glyphs.entry(character).or_insert_with(|| {
                    let (metrics, mask) = font.rasterize(character, self.text_size);
                    CachedGlyph { metrics, mask }
                });
                let metrics = &glyph.metrics;
                let origin_x = cell_x as i64
                    + ((config.cell_width as f32 - metrics.advance_width) / 2.0).round() as i64;
                let glyph_x = origin_x + i64::from(metrics.xmin);
                let glyph_y = self.text_baseline - i64::from(metrics.ymin) - metrics.height as i64;
                for mask_y in 0..metrics.height {
                    for mask_x in 0..metrics.width {
                        let pixel_x = glyph_x + mask_x as i64;
                        let pixel_y = glyph_y + mask_y as i64;
                        if pixel_x < cell_x as i64
                            || pixel_x >= (cell_x + config.cell_width) as i64
                            || pixel_y < 0
                            || pixel_y >= config.cell_height as i64
                        {
                            continue;
                        }
                        let offset = (pixel_y as usize * width + pixel_x as usize) * 3;
                        let alpha = u32::from(glyph.mask[mask_y * metrics.width + mask_x]);
                        let pixel = &mut self.band[offset..offset + 3];
                        for (channel, value) in pixel.iter_mut().enumerate() {
                            *value = ((u32::from(foreground[channel]) * alpha
                                + u32::from(*value) * (255 - alpha)
                                + 127)
                                / 255) as u8;
                        }
                    }
                }
            }
            if cell.modifier.contains(Modifier::UNDERLINED) {
                fill_rect(
                    &mut self.band,
                    width,
                    (
                        cell_x,
                        config.cell_height.saturating_sub(2 * config.scale.max(1)),
                    ),
                    (config.cell_width, config.scale.max(1)),
                    foreground,
                );
            }
        }
    }
}

/// Draw the start and end sentinel glyphs `╟` and `╢` with one-pixel lines instead of the font's,
/// whose box-drawing strokes sit half a pixel off the middle row of the cell. Two vertical lines
/// span the cell; a horizontal line on the middle pixel row runs from the line nearer the
/// edges to the cell's open side. Returns whether `character` was one of them.
fn draw_sentinel_glyph(
    pixels: &mut [u8],
    row_width: usize,
    (cell_x, character): (usize, char),
    config: RasterConfig,
    color: [u8; 3],
) -> bool {
    let (width, height) = (config.cell_width, config.cell_height);
    let thickness = config.scale.max(1);
    let middle_column = width / 2;
    let (near, far) = (middle_column - thickness, middle_column + thickness);
    // Keep the horizontal line on a whole logical pixel row so it is not split across two.
    let middle_row = height / 2 / thickness * thickness;
    let stub = match character {
        '╟' => (far + thickness, width - far - thickness),
        '╢' => (0, near),
        _ => return false,
    };
    for line in [near, far] {
        fill_rect(
            pixels,
            row_width,
            (cell_x + line, 0),
            (thickness, height),
            color,
        );
    }
    fill_rect(
        pixels,
        row_width,
        (cell_x + stub.0, middle_row),
        (stub.1, thickness),
        color,
    );
    true
}

/// Fill the rectangle at `origin` of `size` (both `(x, y)`/`(width, height)` in pixels).
fn fill_rect(
    pixels: &mut [u8],
    row_width: usize,
    origin: (usize, usize),
    size: (usize, usize),
    color: [u8; 3],
) {
    for row in origin.1..origin.1 + size.1 {
        let start = (row * row_width + origin.0) * 3;
        let end = start + size.0 * 3;
        for pixel in pixels[start..end].as_chunks_mut::<3>().0 {
            *pixel = color;
        }
    }
}

fn cell_colors(cell: &Cell, config: RasterConfig) -> ([u8; 3], [u8; 3]) {
    let mut colors = (
        color_to_rgb(cell.fg, config.foreground),
        color_to_rgb(cell.bg, config.background),
    );
    if cell.modifier.contains(Modifier::REVERSED) {
        swap(&mut colors.0, &mut colors.1);
    }
    colors
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use png::Decoder;
    use ratatui::{layout::Rect, style::Style};

    use super::{Buffer, Color, PngRenderer, RasterConfig};

    fn small_config() -> RasterConfig {
        RasterConfig {
            cell_width: 8,
            cell_height: 12,
            scale: 1,
            foreground: [255; 3],
            background: [0; 3],
        }
    }

    fn decode(bytes: &[u8]) -> (u32, u32, Vec<u8>) {
        let mut reader = Decoder::new(Cursor::new(bytes))
            .read_info()
            .expect("should read PNG header");
        let mut pixels = vec![0; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut pixels)
            .expect("should decode PNG frame");
        pixels.truncate(info.buffer_size());
        (info.width, info.height, pixels)
    }

    #[test]
    fn test_png_dimensions_and_background() {
        let mut renderer = PngRenderer::new(small_config()).unwrap();
        let mut buffer = Buffer::empty(Rect::new(0, 0, 3, 2));
        buffer.set_style(
            Rect::new(1, 0, 1, 1),
            Style::default().bg(Color::Rgb(10, 20, 30)),
        );
        let mut bytes = Vec::new();
        renderer.write_png(&buffer, &mut bytes).unwrap();
        let (width, height, pixels) = decode(&bytes);
        assert_eq!((width, height), (24, 24));
        assert_eq!(&pixels[0..3], &[0, 0, 0]);
        assert_eq!(&pixels[8 * 3..8 * 3 + 3], &[10, 20, 30]);
    }

    #[test]
    fn test_crop_keeps_one_cell_margin() {
        let renderer = PngRenderer::new(small_config()).unwrap();
        let mut buffer = Buffer::empty(Rect::new(0, 0, 20, 10));
        buffer.set_string(5, 4, "AB", Style::default().fg(Color::White));
        let cropped = renderer.crop_to_content(&buffer);
        assert_eq!((cropped.area.width, cropped.area.height), (4, 3));
    }

    #[test]
    fn test_crop_margin_stops_at_buffer_edge() {
        let renderer = PngRenderer::new(small_config()).unwrap();
        let mut buffer = Buffer::empty(Rect::new(0, 0, 6, 4));
        buffer.set_string(0, 0, "A", Style::default().fg(Color::White));
        let cropped = renderer.crop_to_content(&buffer);
        assert_eq!((cropped.area.width, cropped.area.height), (2, 2));
    }

    #[test]
    fn test_crop_of_blank_buffer_is_unchanged() {
        let renderer = PngRenderer::new(small_config()).unwrap();
        let buffer = Buffer::empty(Rect::new(0, 0, 5, 3));
        let cropped = renderer.crop_to_content(&buffer);
        assert_eq!((cropped.area.width, cropped.area.height), (5, 3));
    }

    fn cell_pixels(character: &str) -> Vec<[u8; 3]> {
        let config = RasterConfig {
            cell_width: 11,
            cell_height: 21,
            scale: 1,
            foreground: [255; 3],
            background: [0; 3],
        };
        let mut renderer = PngRenderer::new(config).unwrap();
        let mut buffer = Buffer::empty(Rect::new(0, 0, 1, 1));
        buffer.set_string(0, 0, character, Style::default().fg(Color::White));
        let mut bytes = Vec::new();
        renderer.write_png(&buffer, &mut bytes).unwrap();
        decode(&bytes)
            .2
            .chunks_exact(3)
            .map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect()
    }

    #[test]
    fn test_start_sentinel_is_drawn_with_one_pixel_lines_on_the_middle_row() {
        let pixels = cell_pixels("╟");
        let lit = |column: usize, row: usize| pixels[row * 11 + column][0] > 200;
        // Two full-height lines around the middle column, and a stub on the middle row to the
        // right edge only.
        assert!((0..21).all(|row| lit(4, row) && lit(6, row)));
        assert!(!lit(5, 3));
        assert!((7..11).all(|column| lit(column, 10)));
        assert!(!lit(7, 9) && !lit(7, 11));
        assert!(!lit(0, 10));
    }

    #[test]
    fn test_end_sentinel_mirrors_the_start_sentinel() {
        let pixels = cell_pixels("╢");
        let lit = |column: usize, row: usize| pixels[row * 11 + column][0] > 200;
        assert!((0..4).all(|column| lit(column, 10)));
        assert!(!lit(10, 10));
    }

    #[test]
    fn test_box_glyphs_use_the_normal_font_layout() {
        let pixels = cell_pixels("█");
        assert!(pixels.iter().any(|pixel| pixel[0] > 128));
        assert!(pixels.iter().any(|pixel| pixel[0] < 128));
    }

    #[test]
    fn test_letters_are_inset_from_cell_edges() {
        let pixels = cell_pixels("M");
        let lit_in_column = |column: usize| (0..21).any(|row| pixels[row * 11 + column][0] > 64);
        let lit_in_row = |row: usize| (0..11).any(|column| pixels[row * 11 + column][0] > 64);
        assert!(!lit_in_column(0) && !lit_in_column(10));
        assert!(!lit_in_row(0) && !lit_in_row(20));
        assert!(pixels.iter().any(|pixel| pixel[0] > 128));
    }

    #[test]
    fn test_descenders_share_the_text_baseline_without_clipping() {
        let config = RasterConfig::default();
        let renderer = PngRenderer::new(config).expect("should load font");
        for character in "MgypqÅ".chars() {
            let metrics = renderer.font.metrics(character, renderer.text_size);
            let top = renderer.text_baseline - i64::from(metrics.ymin) - metrics.height as i64;
            let bottom = renderer.text_baseline - i64::from(metrics.ymin);
            assert!(
                top >= 0 && bottom <= config.cell_height as i64,
                "{character} should fit the cell at the shared baseline"
            );
        }
        let descender = renderer.font.metrics('g', renderer.text_size);
        assert!(descender.ymin < 0, "g should extend below the baseline");
    }

    #[test]
    fn test_overlay_is_behind_node_backgrounds_and_glyphs() {
        let mut renderer = PngRenderer::new(small_config()).expect("should load font");
        let mut buffer = Buffer::empty(Rect::new(0, 0, 2, 1));
        buffer.set_string(0, 0, "M", Style::default().fg(Color::White));
        buffer[(1, 0)].set_bg(Color::Rgb(17, 23, 31));
        let mut bytes = Vec::new();
        renderer
            .write_png_with_overlay(&buffer, &mut bytes, |pixels, _, _| {
                pixels.fill(0);
            })
            .expect("should render text above overlay");
        let pixels = decode(&bytes).2;
        assert!(pixels.iter().any(|&value| value > 31));
        assert_eq!(&pixels[8 * 3..9 * 3], &[17, 23, 31]);
    }

    #[test]
    fn test_empty_buffer_is_rejected() {
        let mut renderer = PngRenderer::new(small_config()).unwrap();
        let buffer = Buffer::empty(Rect::new(0, 0, 0, 0));
        assert!(renderer.write_png(&buffer, &mut Vec::new()).is_err());
    }

    #[test]
    fn test_scale_two_doubles_the_image_and_keeps_the_cell_grid() {
        let config = small_config().with_scale(2);
        assert_eq!((config.cell_width, config.cell_height), (16, 24));
        assert_eq!(config.logical_cell_size(), (8, 12));
        let mut renderer = PngRenderer::new(config).unwrap();
        let buffer = Buffer::empty(Rect::new(0, 0, 3, 2));
        let mut bytes = Vec::new();
        renderer.write_png(&buffer, &mut bytes).unwrap();
        let (width, height, _) = decode(&bytes);
        assert_eq!((width, height), (48, 48));
    }

    #[test]
    fn test_png_declares_96_dpi_times_the_scale() {
        for scale in [1, 2, 3] {
            let mut renderer = PngRenderer::new(small_config().with_scale(scale)).unwrap();
            let buffer = Buffer::empty(Rect::new(0, 0, 2, 2));
            let mut bytes = Vec::new();
            renderer.write_png(&buffer, &mut bytes).unwrap();
            let reader = Decoder::new(bytes.as_slice()).read_info().unwrap();
            let dimensions = reader
                .info()
                .pixel_dims
                .expect("should declare the pixel density");
            let pixels_per_meter = 3780 * scale as u32;
            assert_eq!(
                (dimensions.xppu, dimensions.yppu),
                (pixels_per_meter, pixels_per_meter),
                "a scale of {scale} should declare {} dpi",
                96 * scale
            );
            assert_eq!(dimensions.unit, png::Unit::Meter);
        }
    }

    #[test]
    fn test_with_scale_rescales_from_the_logical_cell_size() {
        let config = small_config().with_scale(3).with_scale(2);
        assert_eq!((config.cell_width, config.cell_height), (16, 24));
        let reset = config.with_scale(0);
        assert_eq!(
            (reset.cell_width, reset.cell_height, reset.scale),
            (8, 12, 1)
        );
    }

    fn scaled_cell_pixels(character: &str, scale: usize) -> (usize, usize, Vec<[u8; 3]>) {
        let config = RasterConfig {
            foreground: [255; 3],
            background: [0; 3],
            ..RasterConfig::default()
        }
        .with_scale(scale);
        let mut renderer = PngRenderer::new(config).unwrap();
        let mut buffer = Buffer::empty(Rect::new(0, 0, 1, 1));
        buffer.set_string(0, 0, character, Style::default().fg(Color::White));
        let mut bytes = Vec::new();
        renderer.write_png(&buffer, &mut bytes).unwrap();
        let pixels = decode(&bytes)
            .2
            .chunks_exact(3)
            .map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect();
        (config.cell_width, config.cell_height, pixels)
    }

    #[test]
    fn test_glyph_cell_is_not_blank_at_scale_two() {
        let (width, height, pixels) = scaled_cell_pixels("M", 2);
        assert_eq!((width, height), (22, 42));
        assert!(pixels.iter().any(|pixel| pixel[0] > 128));
        let lit_in_column =
            |column: usize| (0..height).any(|row| pixels[row * width + column][0] > 64);
        assert!(!lit_in_column(0) && !lit_in_column(width - 1));
    }

    #[test]
    fn test_sentinel_lines_are_two_pixels_thick_at_scale_two() {
        let (width, height, pixels) = scaled_cell_pixels("╟", 2);
        let lit = |column: usize, row: usize| pixels[row * width + column][0] > 200;
        assert!((0..height).all(|row| (9..11).all(|column| lit(column, row))));
        assert!((0..height).all(|row| (13..15).all(|column| lit(column, row))));
        assert!(!lit(11, 5) && !lit(12, 5));
        assert!((15..width).all(|column| lit(column, 20) && lit(column, 21)));
        assert!(!lit(15, 19) && !lit(15, 22));
    }
}
