//! Rasterizes a Ratatui `Buffer` to a PNG so graph plots can be produced without a terminal or
//! notebook frontend (the CLI `--png` flag and the Python `save_png` methods).
//!
//! Pixels are produced one terminal row at a time and streamed into a single PNG, so memory use is
//! independent of image height. The font, glyph cache and row-band allocation persist across
//! exports. Glyphs are single-width and clipped to their cell: no shaping, fallback fonts or
//! bold/italic faces.

use core::mem::swap;
use std::{
    collections::HashMap,
    io::{Error as IoError, Write},
};

use fontdue::{Font, FontSettings, Metrics};
use png::{BitDepth, ColorType, Compression, Encoder, EncodingError};
use ratatui::{
    buffer::{Buffer, Cell},
    layout::{Margin, Rect},
    style::{Color, Modifier},
};
use thiserror::Error;

use crate::theme::current_theme;

/// Text is drawn at this fraction of the size at which a full block fills the cell, so letters
/// sit inside the cell with room around them. Matches the Jupyter widget (`TEXT_SIZE / CELL_SIZE`).
const TEXT_TO_BOX_RATIO: f32 = 0.875;

/// Box-drawing glyphs are drawn this much larger than the cell and clipped to it, so adjacent
/// cells overlap slightly and lines join without gaps. Matches the Jupyter widget's `BOX_SCALE`.
const BOX_SCALE: f32 = 1.15;

/// Upper bound on a cell's pixel size. With `Buffer` dimensions capped at `u16`, it keeps every
/// image and band size well inside `u32`/`usize`, so no overflow checks are needed.
const MAX_CELL_SIZE: usize = 256;

/// Blank cells kept around the content by [`PngRenderer::crop_to_content`].
pub const CROP_MARGIN: u16 = 1;

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

#[derive(Clone, Copy, Debug)]
pub struct PngConfig {
    pub cell_width: usize,
    pub cell_height: usize,
    /// Used for cells whose foreground is `Color::Reset`.
    pub foreground: [u8; 3],
    /// Used for cells whose background is `Color::Reset`, and to fill the canvas.
    pub background: [u8; 3],
}

impl Default for PngConfig {
    /// A 13x28 pixel cell with the current theme's text and canvas colors.
    fn default() -> Self {
        let theme = current_theme();
        Self {
            cell_width: 13,
            cell_height: 28,
            foreground: color_to_rgb(theme[0x05], [255; 3]),
            background: color_to_rgb(theme[0x00], [0; 3]),
        }
    }
}

struct CachedGlyph {
    metrics: Metrics,
    mask: Vec<u8>,
}

/// Font and configuration are immutable, so a character identifies a cache entry. Construct
/// another `PngRenderer` to change the cell size or colors.
///
/// Sizes derive from the cell: box-drawing glyphs are scaled so a full block fills the cell
/// height (then enlarged by `BOX_SCALE` about the cell center and clipped to the cell), and other
/// glyphs are smaller and centered.
pub struct PngRenderer {
    font: Font,
    config: PngConfig,
    text_size: f32,
    box_size: f32,
    text_baseline: i64,
    /// Offset from the cell center to the origin of a box glyph, in pixels.
    box_origin: (f32, f32),
    glyphs: HashMap<char, CachedGlyph>,
    band: Vec<u8>,
}

impl PngRenderer {
    pub fn new(config: PngConfig) -> Result<Self, PngExportError> {
        if !(1..=MAX_CELL_SIZE).contains(&config.cell_width)
            || !(1..=MAX_CELL_SIZE).contains(&config.cell_height)
        {
            return Err(PngExportError::InvalidInput(
                "cell dimensions must be between 1 and 256 pixels",
            ));
        }
        let font =
            Font::from_bytes(FONT_BYTES, FontSettings::default()).map_err(PngExportError::Font)?;
        const PROBE_SIZE: f32 = 100.0;
        let probe = font.metrics('\u{2588}', PROBE_SIZE);
        if probe.height == 0 {
            return Err(PngExportError::InvalidInput("font has no full block glyph"));
        }
        let fit_size = config.cell_height as f32 * PROBE_SIZE / probe.height as f32;
        let text_size = fit_size * TEXT_TO_BOX_RATIO;
        let box_size = fit_size * BOX_SCALE;
        let line = font
            .horizontal_line_metrics(text_size)
            .ok_or(PngExportError::InvalidInput(
                "font has no horizontal metrics",
            ))?;
        let text_baseline = ((config.cell_height as f32 - (line.ascent - line.descent)) / 2.0
            + line.ascent)
            .round() as i64;
        // Center the full block, whose box is the font's cell, on the pixel cell.
        let block = font.metrics('\u{2588}', box_size);
        let box_origin = (
            -(block.xmin as f32 + block.width as f32 / 2.0),
            block.ymin as f32 + block.height as f32 / 2.0,
        );
        Ok(Self {
            font,
            config,
            text_size,
            box_size,
            text_baseline,
            box_origin,
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
        let mut writer = encoder.write_header()?;
        {
            // One stream spans all bands so compression is not restarted per row.
            let mut stream = writer.stream_writer_with_size(64 * 1024)?;
            for row in buffer.content.chunks_exact(columns) {
                self.render_band(row, width);
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
        let columns = usize::from(buffer.area.width);
        let content = buffer
            .content
            .iter()
            .enumerate()
            .filter(|(_, cell)| self.is_visible(cell))
            .map(|(index, _)| Rect::new((index % columns) as u16, (index / columns) as u16, 1, 1))
            .reduce(|bounds, cell| bounds.union(cell));
        let Some(content) = content else {
            return buffer.clone();
        };
        let crop = content
            .outer(Margin::new(CROP_MARGIN, CROP_MARGIN))
            .intersection(Rect::new(0, 0, buffer.area.width, buffer.area.height));
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

    fn render_band(&mut self, row: &[Cell], width: usize) {
        let config = self.config;
        for pixel in self.band.as_chunks_mut::<3>().0 {
            *pixel = config.background;
        }
        // Complete the row's backgrounds before drawing any glyphs.
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
            if let Some(character) = glyph_char {
                let is_box = is_box_like(character);
                let size = if is_box {
                    self.box_size
                } else {
                    self.text_size
                };
                let font = &self.font;
                let glyph = self.glyphs.entry(character).or_insert_with(|| {
                    let (metrics, mask) = font.rasterize(character, size);
                    CachedGlyph { metrics, mask }
                });
                let metrics = &glyph.metrics;
                let (glyph_x, glyph_y) = if is_box {
                    let center_x = cell_x as f32 + config.cell_width as f32 / 2.0;
                    let center_y = config.cell_height as f32 / 2.0;
                    let origin_x = center_x + self.box_origin.0;
                    let baseline = center_y + self.box_origin.1;
                    (
                        origin_x.round() as i64 + i64::from(metrics.xmin),
                        baseline.round() as i64 - i64::from(metrics.ymin) - metrics.height as i64,
                    )
                } else {
                    let origin_x = cell_x as i64
                        + ((config.cell_width as f32 - metrics.advance_width) / 2.0).round() as i64;
                    (
                        origin_x + i64::from(metrics.xmin),
                        self.text_baseline - i64::from(metrics.ymin) - metrics.height as i64,
                    )
                };
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
                    (cell_x, config.cell_height.saturating_sub(2)),
                    (config.cell_width, 1),
                    foreground,
                );
            }
        }
    }
}

/// Box drawing, block elements and braille: glyphs meant to tile edge to edge.
fn is_box_like(character: char) -> bool {
    matches!(character, '\u{2500}'..='\u{259F}' | '\u{2800}'..='\u{28FF}')
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

fn cell_colors(cell: &Cell, config: PngConfig) -> ([u8; 3], [u8; 3]) {
    let mut colors = (
        color_to_rgb(cell.fg, config.foreground),
        color_to_rgb(cell.bg, config.background),
    );
    if cell.modifier.contains(Modifier::REVERSED) {
        swap(&mut colors.0, &mut colors.1);
    }
    colors
}

fn color_to_rgb(color: Color, default: [u8; 3]) -> [u8; 3] {
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
        Color::Indexed(index) => indexed_to_rgb(index),
    }
}

fn indexed_to_rgb(index: u8) -> [u8; 3] {
    const ANSI: [[u8; 3]; 16] = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [192, 192, 192],
        [128, 128, 128],
        [255, 0, 0],
        [0, 255, 0],
        [255, 255, 0],
        [0, 0, 255],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255],
    ];
    match index {
        0..=15 => ANSI[usize::from(index)],
        16..=231 => {
            let offset = index - 16;
            let scale = |level: u8| if level == 0 { 0 } else { 55 + level * 40 };
            [
                scale(offset / 36),
                scale(offset % 36 / 6),
                scale(offset % 6),
            ]
        }
        232..=255 => [8 + (index - 232) * 10; 3],
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use png::Decoder;
    use ratatui::{layout::Rect, style::Style};

    use super::{Buffer, Color, PngConfig, PngRenderer};

    fn small_config() -> PngConfig {
        PngConfig {
            cell_width: 8,
            cell_height: 12,
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
        let config = PngConfig {
            cell_width: 13,
            cell_height: 28,
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
    fn test_box_glyphs_reach_cell_edges_so_lines_join() {
        let pixels = cell_pixels("─");
        let lit = |column: usize| (0..28).any(|row| pixels[row * 13 + column][0] > 128);
        assert!(lit(0) && lit(12));
        let block = cell_pixels("█");
        assert!(block.iter().all(|pixel| pixel[0] > 128));
    }

    #[test]
    fn test_letters_are_inset_from_cell_edges() {
        let pixels = cell_pixels("M");
        let lit_in_column = |column: usize| (0..28).any(|row| pixels[row * 13 + column][0] > 64);
        let lit_in_row = |row: usize| (0..13).any(|column| pixels[row * 13 + column][0] > 64);
        assert!(!lit_in_column(0) && !lit_in_column(12));
        assert!(!lit_in_row(0) && !lit_in_row(27));
        assert!(pixels.iter().any(|pixel| pixel[0] > 128));
    }

    #[test]
    fn test_empty_buffer_is_rejected() {
        let mut renderer = PngRenderer::new(small_config()).unwrap();
        let buffer = Buffer::empty(Rect::new(0, 0, 0, 0));
        assert!(renderer.write_png(&buffer, &mut Vec::new()).is_err());
    }
}
