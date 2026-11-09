//! PNG export of graphs drawn with curved edges.
//!
//! The viewer places every node exactly as it does in a terminal but leaves the edges out of the
//! buffer and reports them as shapes (see `EdgeStyle::Curved`). The text is rasterized by
//! `PngRenderer`, and the shapes are rasterized by the same `EdgeLayer` a terminal shows through
//! the Kitty graphics protocol, composited band by band as the image streams out.

use std::io::{self, Write};

use gen_tui::{
    EdgeShape,
    edge_layer::{EdgeLayer, apply_highlight_cells, shapes_bounds},
    png_export::{PngConfig, PngRenderer},
};
use ratatui::buffer::Buffer;

/// Write `buffer` as a PNG cropped to its content, with `shapes` (in cells of `buffer`) drawn
/// over its blank cells. Every length is multiplied by `scale` image pixels per logical pixel, so
/// a scale of 2 gives the same picture on twice as many pixels.
pub fn write_png_with_edges(
    buffer: &Buffer,
    shapes: &[EdgeShape],
    scale: usize,
    output: &mut impl Write,
) -> io::Result<()> {
    write_png_with_edges_inner(buffer, shapes, scale, output, true)
}

/// Write a full viewport PNG without trimming its terminal-cell geometry.
pub fn write_png_with_edges_full_viewport(
    buffer: &Buffer,
    shapes: &[EdgeShape],
    scale: usize,
    output: &mut impl Write,
) -> io::Result<()> {
    write_png_with_edges_inner(buffer, shapes, scale, output, false)
}

fn write_png_with_edges_inner(
    buffer: &Buffer,
    shapes: &[EdgeShape],
    scale: usize,
    output: &mut impl Write,
    crop_to_content: bool,
) -> io::Result<()> {
    let config = PngConfig::default().with_scale(scale);
    let mut renderer = PngRenderer::new(config).map_err(io::Error::other)?;
    let crop = if crop_to_content {
        renderer.content_crop(buffer, shapes_bounds(shapes, buffer.area))
    } else {
        buffer.area
    };
    let mut cropped = PngRenderer::crop(buffer, crop);
    let origin = (crop.x, crop.y);
    apply_highlight_cells(&mut cropped, shapes, origin, config);
    let layer = EdgeLayer::new(shapes, &cropped, origin, config);
    if layer.is_empty() {
        return renderer
            .write_png(&cropped, output)
            .map_err(io::Error::other);
    }
    let width = usize::from(cropped.area.width) * config.cell_width;
    let mut scratch = layer
        .band_pixmap(width, config.cell_height)
        .ok_or_else(|| io::Error::other("could not allocate PNG edge band"))?;
    renderer
        .write_png_with_overlay(&cropped, output, |pixels, _, top| {
            layer.composite_band(pixels, top, &mut scratch);
        })
        .map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use gen_tui::{BezierCurve, EdgeShape, LineStyle, png_export::PngConfig};
    use ratatui::{buffer::Buffer, layout::Rect, style::Color};

    use super::{write_png_with_edges, write_png_with_edges_full_viewport};

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

    #[test]
    fn test_png_with_bezier_is_cropped_around_the_curve_not_its_control_points() {
        let area = Rect::new(0, 0, 40, 20);
        let buffer = Buffer::empty(area);
        let curves = [EdgeShape::Bezier(curve(
            (5.0, 10.0),
            (20.0, 1.0),
            (30.0, 1.0),
            (35.0, 10.0),
            LineStyle::Normal,
        ))];
        let mut bytes = Vec::new();
        write_png_with_edges(&buffer, &curves, 1, &mut bytes).expect("should write the png");
        let dimension = |start: usize| {
            u32::from_be_bytes(
                bytes[start..start + 4]
                    .try_into()
                    .expect("should have a header"),
            ) as usize
        };
        let config = PngConfig::default();
        assert_eq!(dimension(16), (35 - 5 + 1 + 2) * config.cell_width);
        // The curve peaks at y = 3.25, three quarters of the way to its handles at y = 1.
        assert_eq!(dimension(20), (10 - 3 + 1 + 2) * config.cell_height);
    }

    #[test]
    fn test_full_viewport_png_preserves_cell_dimensions() {
        let area = Rect::new(0, 0, 40, 20);
        let buffer = Buffer::empty(area);
        let mut bytes = Vec::new();
        write_png_with_edges_full_viewport(&buffer, &[], 1, &mut bytes)
            .expect("should write the full viewport png");
        let dimension = |start: usize| {
            u32::from_be_bytes(
                bytes[start..start + 4]
                    .try_into()
                    .expect("should have a header"),
            ) as usize
        };
        let config = PngConfig::default();
        assert_eq!(dimension(16), 40 * config.cell_width);
        assert_eq!(dimension(20), 20 * config.cell_height);
    }

    fn png_dimensions(bytes: &[u8]) -> (u32, u32) {
        let read = |start: usize| {
            u32::from_be_bytes(
                bytes[start..start + 4]
                    .try_into()
                    .expect("should have a header"),
            )
        };
        (read(16), read(20))
    }

    #[test]
    fn test_png_with_lines_at_scale_two_doubles_the_pixels() {
        let area = Rect::new(0, 0, 40, 20);
        let buffer = Buffer::empty(area);
        let curves = [EdgeShape::Bezier(curve(
            (5.0, 10.0),
            (20.0, 1.0),
            (30.0, 1.0),
            (35.0, 10.0),
            LineStyle::Bold,
        ))];
        let mut normal = Vec::new();
        write_png_with_edges(&buffer, &curves, 1, &mut normal).expect("should write the png");
        let mut doubled = Vec::new();
        write_png_with_edges(&buffer, &curves, 2, &mut doubled)
            .expect("should write the scaled png");
        let (width, height) = png_dimensions(&normal);
        assert_eq!(png_dimensions(&doubled), (width * 2, height * 2));
    }
}
