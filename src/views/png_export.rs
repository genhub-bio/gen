//! Rasterizes a Ratatui `Buffer` to a PNG so graph plots can be produced without a terminal or
//! notebook frontend (the CLI `--png` flag and the Python `save_png` methods).
//!
//! The viewer places every node exactly as it does in a terminal but leaves curved edges out of
//! the buffer and reports them as overlay primitives (see `LayoutMode::Curved`). The text is
//! rasterized here, and the primitives are rasterized by the same `PixelOverlay` a terminal shows
//! through the Kitty graphics protocol, composited band by band as the image streams out.
//!
//! Pixels are produced one terminal row at a time and streamed into a single PNG, so pixel-buffer
//! memory is independent of image height (the input cell buffer still covers the whole view). The
//! font, glyph cache and row-band allocation persist across exports. Glyphs are single-width and
//! clipped to their cell: no shaping, fallback fonts or bold/italic faces.

use core::mem::swap;
use std::{
    collections::HashMap,
    io::{self, Error as IoError, Write},
};

use fontdue::{Font, FontSettings, Metrics};
use gen_tui::{
    OverlayPrimitive, RasterConfig, color_to_rgb,
    pixel_overlay::{PixelOverlay, overlay_primitives_bounds, prepare_node_highlight_cells},
};
use png::{BitDepth, ColorType, Compression, Encoder, EncodingError, PixelDimensions, Unit};
use ratatui::{
    buffer::{Buffer, Cell},
    layout::{Margin, Rect},
    style::Modifier,
};
use thiserror::Error;
use tiny_skia::Pixmap;

/// Blank cells kept around the content by [`PngRenderer::crop_to_content`].
pub const CROP_MARGIN: u16 = 1;

/// Largest pixel scale callers should accept: an image grows with the square of its scale.
pub const MAX_SCALE: usize = 8;

/// Scale callers draw at unless asked otherwise: twice the logical pixels stay sharp on
/// high-density screens, and the logical size is the same.
pub const DEFAULT_SCALE: usize = 2;

/// Pixels per meter at the logical 96 dpi, which a scale of one declares in the PNG header.
const LOGICAL_PIXELS_PER_METER: u32 = 3780;

const FONT_BYTES: &[u8] = include_bytes!("fonts/JetBrainsMono-Regular.ttf");

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

    /// Stream a complete PNG of `buffer` into `output`, with `overlay` (curved edges and other
    /// pixel primitives of the buffer's cells) drawn behind node backgrounds and text, as in the
    /// terminal. Use a file for large plots; a `Vec` retains the whole encoded image.
    pub fn write_png<W: Write>(
        &mut self,
        buffer: &Buffer,
        overlay: Option<&PixelOverlay>,
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
        let mut overlay_band = match overlay.filter(|overlay| !overlay.is_empty()) {
            Some(overlay) => {
                let scratch = overlay.band_pixmap(width, self.config.cell_height).ok_or(
                    PngExportError::InvalidInput("could not allocate PNG edge band"),
                )?;
                Some((overlay, scratch))
            }
            None => None,
        };

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
                    overlay_band.as_mut(),
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
        overlay_band: Option<&mut (&PixelOverlay, Pixmap)>,
    ) {
        let config = self.config;
        for pixel in self.band.as_chunks_mut::<3>().0 {
            *pixel = config.background;
        }
        if let Some((overlay, scratch)) = overlay_band {
            overlay.composite_band(&mut self.band, top, scratch);
        }
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
///
/// These glyphs are produced by gen-tui's plotter to mark where a path starts and ends, so the
/// characters matched here must stay in sync with it.
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

/// Write `buffer` as a PNG cropped to its content, with `overlay_primitives` (in cells of `buffer`) drawn
/// over its blank cells. Every length is multiplied by `scale` image pixels per logical pixel, so
/// a scale of 2 gives the same picture on twice as many pixels.
pub fn write_cropped_png(
    buffer: &Buffer,
    overlay_primitives: &[OverlayPrimitive],
    scale: usize,
    output: &mut impl Write,
) -> io::Result<()> {
    write_png_inner(buffer, overlay_primitives, scale, output, true)
}

/// Write a full viewport PNG without trimming its terminal-cell geometry.
pub fn write_uncropped_png(
    buffer: &Buffer,
    overlay_primitives: &[OverlayPrimitive],
    scale: usize,
    output: &mut impl Write,
) -> io::Result<()> {
    write_png_inner(buffer, overlay_primitives, scale, output, false)
}

fn write_png_inner(
    buffer: &Buffer,
    overlay_primitives: &[OverlayPrimitive],
    scale: usize,
    output: &mut impl Write,
    crop_to_content: bool,
) -> io::Result<()> {
    let config = RasterConfig::default().with_scale(scale);
    let mut renderer = PngRenderer::new(config).map_err(io::Error::other)?;
    let crop = if crop_to_content {
        renderer.content_crop(
            buffer,
            overlay_primitives_bounds(overlay_primitives, buffer.area),
        )
    } else {
        buffer.area
    };
    let mut cropped = PngRenderer::crop(buffer, crop);
    let origin = (crop.x, crop.y);
    prepare_node_highlight_cells(&mut cropped, overlay_primitives, origin, config);
    let overlay = PixelOverlay::new(overlay_primitives, &cropped, origin, config);
    renderer
        .write_png(&cropped, Some(&overlay), output)
        .map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use gen_tui::{BezierCurve, LineStyle};
    use png::Decoder;
    use ratatui::{
        layout::Rect,
        style::{Color, Style},
    };

    use super::{
        Buffer, OverlayPrimitive, PixelOverlay, PngRenderer, RasterConfig, write_cropped_png,
        write_uncropped_png,
    };

    fn small_config() -> RasterConfig {
        RasterConfig {
            cell_width: 8,
            cell_height: 12,
            scale: 1,
            foreground: [255; 3],
            background: [0; 3],
        }
    }

    fn curve(
        from: (f64, f64),
        control1: (f64, f64),
        control2: (f64, f64),
        to: (f64, f64),
        line_style: LineStyle,
    ) -> BezierCurve {
        BezierCurve {
            from,
            control1,
            control2,
            to,
            color: Color::Red,
            line_style,
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
        renderer.write_png(&buffer, None, &mut bytes).unwrap();
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
        renderer.write_png(&buffer, None, &mut bytes).unwrap();
        decode(&bytes).2.as_chunks::<3>().0.to_vec()
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
        let config = small_config();
        let mut renderer = PngRenderer::new(config).expect("should load font");
        let mut buffer = Buffer::empty(Rect::new(0, 0, 3, 1));
        buffer.set_string(0, 0, "M", Style::default().fg(Color::White));
        buffer[(1, 0)].set_bg(Color::Rgb(17, 23, 31));
        let line = [OverlayPrimitive::Bezier(curve(
            (0.0, 0.5),
            (1.0, 0.5),
            (2.0, 0.5),
            (3.0, 0.5),
            LineStyle::Bold,
        ))];
        let overlay = PixelOverlay::new(&line, &buffer, (0, 0), config);
        let mut bytes = Vec::new();
        renderer
            .write_png(&buffer, Some(&overlay), &mut bytes)
            .expect("should render text above overlay");
        let pixels = decode(&bytes).2;
        let cell_pixels = |column: usize| {
            (0..12)
                .flat_map(|row| {
                    let start = (row * 24 + column * 8) * 3;
                    pixels[start..start + 8 * 3].as_chunks::<3>().0.to_vec()
                })
                .collect::<Vec<_>>()
        };
        let is_edge_red = |pixel: &[u8; 3]| pixel[0] > pixel[1] + 50;
        assert!(cell_pixels(2).iter().any(is_edge_red));
        assert!(cell_pixels(1).iter().all(|pixel| *pixel == [17, 23, 31]));
        assert!(
            cell_pixels(0)
                .iter()
                .any(|pixel| pixel[0] > 200 && pixel[1] > 200)
        );
    }

    #[test]
    fn test_empty_buffer_is_rejected() {
        let mut renderer = PngRenderer::new(small_config()).unwrap();
        let buffer = Buffer::empty(Rect::new(0, 0, 0, 0));
        assert!(renderer.write_png(&buffer, None, &mut Vec::new()).is_err());
    }

    #[test]
    fn test_scale_two_doubles_the_image_and_keeps_the_cell_grid() {
        let config = small_config().with_scale(2);
        assert_eq!((config.cell_width, config.cell_height), (16, 24));
        assert_eq!(config.logical_cell_size(), (8, 12));
        let mut renderer = PngRenderer::new(config).unwrap();
        let buffer = Buffer::empty(Rect::new(0, 0, 3, 2));
        let mut bytes = Vec::new();
        renderer.write_png(&buffer, None, &mut bytes).unwrap();
        let (width, height, _) = decode(&bytes);
        assert_eq!((width, height), (48, 48));
    }

    #[test]
    fn test_png_declares_96_dpi_times_the_scale() {
        for scale in [1, 2, 3] {
            let mut renderer = PngRenderer::new(small_config().with_scale(scale)).unwrap();
            let buffer = Buffer::empty(Rect::new(0, 0, 2, 2));
            let mut bytes = Vec::new();
            renderer.write_png(&buffer, None, &mut bytes).unwrap();
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
        renderer.write_png(&buffer, None, &mut bytes).unwrap();
        let pixels = decode(&bytes).2.as_chunks::<3>().0.to_vec();
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

    #[test]
    fn test_png_with_bezier_is_cropped_around_the_curve_not_its_control_points() {
        let area = Rect::new(0, 0, 40, 20);
        let buffer = Buffer::empty(area);
        let curves = [OverlayPrimitive::Bezier(curve(
            (5.0, 10.0),
            (20.0, 1.0),
            (30.0, 1.0),
            (35.0, 10.0),
            LineStyle::Normal,
        ))];
        let mut bytes = Vec::new();
        write_cropped_png(&buffer, &curves, 1, &mut bytes).expect("should write the png");
        let (width, height) = png_dimensions(&bytes);
        let config = RasterConfig::default();
        assert_eq!(width as usize, (35 - 5 + 1 + 2) * config.cell_width);
        // The curve peaks at y = 3.25, three quarters of the way to its handles at y = 1.
        assert_eq!(height as usize, (10 - 3 + 1 + 2) * config.cell_height);
    }

    #[test]
    fn test_full_viewport_png_preserves_cell_dimensions() {
        let area = Rect::new(0, 0, 40, 20);
        let buffer = Buffer::empty(area);
        let mut bytes = Vec::new();
        write_uncropped_png(&buffer, &[], 1, &mut bytes)
            .expect("should write the full viewport png");
        let (width, height) = png_dimensions(&bytes);
        let config = RasterConfig::default();
        assert_eq!(width as usize, 40 * config.cell_width);
        assert_eq!(height as usize, 20 * config.cell_height);
    }

    fn png_dimensions(bytes: &[u8]) -> (u32, u32) {
        let reader = Decoder::new(bytes).read_info().expect("should read header");
        (reader.info().width, reader.info().height)
    }

    #[test]
    fn test_png_with_lines_at_scale_two_doubles_the_pixels() {
        let area = Rect::new(0, 0, 40, 20);
        let buffer = Buffer::empty(area);
        let curves = [OverlayPrimitive::Bezier(curve(
            (5.0, 10.0),
            (20.0, 1.0),
            (30.0, 1.0),
            (35.0, 10.0),
            LineStyle::Bold,
        ))];
        let mut normal = Vec::new();
        write_cropped_png(&buffer, &curves, 1, &mut normal).expect("should write the png");
        let mut doubled = Vec::new();
        write_cropped_png(&buffer, &curves, 2, &mut doubled).expect("should write the scaled png");
        let (width, height) = png_dimensions(&normal);
        assert_eq!(png_dimensions(&doubled), (width * 2, height * 2));
    }
}
