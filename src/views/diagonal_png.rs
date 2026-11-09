//! PNG export of graphs drawn with deferred Bézier edges.
//!
//! The viewer places every node exactly as it does in a terminal but leaves the edges out of the
//! buffer (see `DiagonalDrawing::Deferred`) and reports them as paths. The text is rasterized by
//! `PngRenderer`; the curves are drawn behind the text with antialiased tiny-skia strokes.

use std::io::{self, Write};

use gen_tui::{
    ArrowMark, BezierCurve, DeferredEdge, LineStyle, NodeGlow, SymbolGlow,
    graph_widget::NODE_GLYPH,
    png_export::{PngConfig, PngRenderer, color_to_rgb},
};
use ratatui::{buffer::Buffer, layout::Rect, style::Color};
use tiny_skia::{
    FillRule, LineCap, Paint, Path, PathBuilder, PathSegment, Pixmap, Stroke, Transform,
};

// Rasterize beyond each band before cropping, so antialiasing at the clip edge cannot
// change coverage on the rows we actually write.
const BAND_PADDING: u32 = 2;

/// Glow rings are one pixel wide and fade like a Gaussian of this sigma, in pixels.
const GLOW_SIGMA: f32 = 1.5;
const GLOW_RINGS: usize = 5;
const GLOW_PEAK: f32 = 0.25;
/// How far a glow color is pushed toward contrast, leaving the core of a highlighted edge
/// brighter than its halo.
const GLOW_BRIGHTNESS: f32 = 0.30;
/// The core is only partly lightened and then re-saturated, so different highlight colors stay
/// distinguishable instead of all washing out toward white.
const EDGE_CORE_BRIGHTNESS: f32 = 0.55;
const EDGE_CORE_SATURATION: f32 = 1.8;
/// Width of the bright line at the center of a highlighted edge.
const EDGE_CORE_WIDTH: f32 = 1.2;
/// Farthest a highlight's halo reaches from the center of its stroke, for band culling.
const EDGE_GLOW_REACH: f32 = EDGE_CORE_WIDTH / 2.0 + GLOW_RINGS as f32;

/// Radius of the `●` node glyph as a fraction of the cell width.
const SYMBOL_RADIUS: f32 = 0.4;
/// How much larger than the glyph the bright ring of a highlighted symbol is, in pixels, enough
/// to show around the glyph without reaching the symbols next to it.
const SYMBOL_GLOW_MARGIN: f32 = 1.0;

/// Opacity of the halo at `ring` pixels beyond the lit shape's edge (the ring's center is half
/// a pixel further): `peak * exp(-d^2 / (2 * sigma^2))`, evaluated once per ring rather than per
/// pixel.
fn glow_opacity(peak: f32, ring: usize) -> f32 {
    let distance = ring as f32 + 0.5;
    peak * (-distance * distance / (2.0 * GLOW_SIGMA * GLOW_SIGMA)).exp()
}

/// Dash and gap lengths as multiples of a cell's width.
const DASH_CELLS: f32 = 0.6;
const GAP_CELLS: f32 = 0.4;

/// Write `buffer` as a PNG cropped to its content, with `lines` (in cells of `buffer`) drawn
/// behind its glyphs.
pub fn write_png_with_lines(
    buffer: &Buffer,
    lines: &[DeferredEdge],
    output: &mut impl Write,
) -> io::Result<()> {
    write_png_with_lines_inner(buffer, lines, output, true)
}

/// Write a full viewport PNG without trimming its terminal-cell geometry.
pub fn write_png_with_lines_full_viewport(
    buffer: &Buffer,
    lines: &[DeferredEdge],
    output: &mut impl Write,
) -> io::Result<()> {
    write_png_with_lines_inner(buffer, lines, output, false)
}

fn write_png_with_lines_inner(
    buffer: &Buffer,
    lines: &[DeferredEdge],
    output: &mut impl Write,
    crop_to_content: bool,
) -> io::Result<()> {
    let config = PngConfig::default();
    let mut renderer = PngRenderer::new(config).map_err(io::Error::other)?;
    let crop = if crop_to_content {
        renderer.content_crop(buffer, lines_bounds(lines, buffer.area))
    } else {
        buffer.area
    };
    let mut cropped = PngRenderer::crop(buffer, crop);
    let light_theme = luminance(config.background) > luminance(config.foreground);
    for glow in lines.iter().filter_map(|line| match line {
        DeferredEdge::NodeGlow(glow) => Some(glow),
        _ => None,
    }) {
        let original = color_to_rgb(glow.color, config.foreground);
        let [red, green, blue] = highlight_color(original, EDGE_CORE_BRIGHTNESS, light_theme);
        let left = (glow.top_left.0.floor() as i32 - i32::from(crop.x)).max(0);
        let top = (glow.top_left.1.floor() as i32 - i32::from(crop.y)).max(0);
        let right =
            (glow.bottom_right.0.ceil() as i32 - i32::from(crop.x)).min(i32::from(crop.width));
        let bottom =
            (glow.bottom_right.1.ceil() as i32 - i32::from(crop.y)).min(i32::from(crop.height));
        for row in top..bottom {
            for column in left..right {
                let cell =
                    &mut cropped.content[row as usize * usize::from(crop.width) + column as usize];
                // Preserve the winning highlight where multiple loci overlap.
                if color_to_rgb(cell.bg, config.background) == original {
                    cell.set_bg(Color::Rgb(red, green, blue));
                }
            }
        }
    }
    // A minimal highlighted node is drawn as a disc by SymbolGlow. Keep its text glyph
    // out of the PNG so the disc and halo are the whole visible node.
    for glow in lines.iter().filter_map(|line| match line {
        DeferredEdge::SymbolGlow(glow) => Some(glow),
        _ => None,
    }) {
        let column = glow.center.0.floor() as i32 - i32::from(crop.x);
        let row = glow.center.1.floor() as i32 - i32::from(crop.y);
        if column >= 0 && row >= 0 && column < i32::from(crop.width) && row < i32::from(crop.height)
        {
            let index = row as usize * usize::from(crop.width) + column as usize;
            if cropped.content[index].symbol() == NODE_GLYPH.to_string() {
                cropped.content[index].set_symbol(" ");
            }
        }
    }
    let paths = prepare_paths(lines, (crop.x, crop.y), config);
    if paths.is_empty() {
        return renderer
            .write_png(&cropped, output)
            .map_err(io::Error::other);
    }
    let width = usize::from(cropped.area.width) * config.cell_width;
    let mut pixmap = Pixmap::new(width as u32, config.cell_height as u32 + 2 * BAND_PADDING)
        .ok_or_else(|| io::Error::other("could not allocate PNG edge band"))?;
    renderer
        .write_png_with_overlay(&cropped, output, |pixels, _, top| {
            draw_paths(pixels, top, &paths, &mut pixmap);
        })
        .map_err(io::Error::other)
}

struct EdgePath {
    path: Path,
    paint: Paint<'static>,
}

// Build geometry once per export. Each band clips the original path, rather than splitting
// curves, so tangents and dash phase remain continuous across PNG rows.
fn prepare_paths(lines: &[DeferredEdge], origin: (u16, u16), config: PngConfig) -> Vec<EdgePath> {
    // A cell's center falls on a pixel center only when its size is odd; an even size needs half
    // a pixel to put one-pixel strokes on the middle pixel row or column, like the text glyphs.
    let center_offset = |size: usize| if size.is_multiple_of(2) { 0.5 } else { 0.0 };
    let point = |(x, y): (f64, f64)| {
        (
            (x - f64::from(origin.0)) as f32 * config.cell_width as f32
                + center_offset(config.cell_width),
            (y - f64::from(origin.1)) as f32 * config.cell_height as f32
                + center_offset(config.cell_height),
        )
    };
    let mut paths = Vec::new();
    for edge in lines {
        if let DeferredEdge::Arrow(mark) = edge {
            push_arrow(mark, point(mark.center), config, &mut paths);
            continue;
        }
        if let DeferredEdge::NodeGlow(glow) = edge {
            push_node_glow(glow, point, config, &mut paths);
            continue;
        }
        if let DeferredEdge::SymbolGlow(glow) = edge {
            push_symbol_glow(glow, point(glow.center), config, &mut paths);
            continue;
        }
        let mut builder = PathBuilder::new();
        let (color, line_style) = match edge {
            DeferredEdge::Line(line) => {
                let from = point(line.from);
                let to = point(line.to);
                builder.move_to(from.0, from.1);
                builder.line_to(to.0, to.1);
                (line.color, line.line_style)
            }
            DeferredEdge::Bezier(curve) => {
                let from = point(curve.from);
                let control1 = point(curve.control1);
                let control2 = point(curve.control2);
                let to = point(curve.to);
                builder.move_to(from.0, from.1);
                builder.cubic_to(control1.0, control1.1, control2.0, control2.1, to.0, to.1);
                (curve.color, curve.line_style)
            }
            DeferredEdge::Arrow(_) | DeferredEdge::NodeGlow(_) | DeferredEdge::SymbolGlow(_) => {
                continue;
            }
        };
        let Some(path) = builder.finish() else {
            continue;
        };
        prepare_strokes(
            path,
            color_to_rgb(color, config.foreground),
            line_style,
            config,
            &mut paths,
        );
    }
    paths
}

fn flatten_path(path: &Path) -> Option<Path> {
    let mut builder = PathBuilder::new();
    let mut current = (0.0, 0.0);
    let mut start = current;
    for segment in path.segments() {
        match segment {
            PathSegment::MoveTo(point) => {
                current = (point.x, point.y);
                start = current;
                builder.move_to(point.x, point.y);
            }
            PathSegment::LineTo(point) => {
                builder.line_to(point.x, point.y);
                current = (point.x, point.y);
            }
            PathSegment::QuadTo(control, point) => {
                let control1 = (
                    current.0 + (control.x - current.0) * 2.0 / 3.0,
                    current.1 + (control.y - current.1) * 2.0 / 3.0,
                );
                let control2 = (
                    point.x + (control.x - point.x) * 2.0 / 3.0,
                    point.y + (control.y - point.y) * 2.0 / 3.0,
                );
                flatten_curve(
                    &mut builder,
                    [current, control1, control2, (point.x, point.y)],
                    0,
                );
                current = (point.x, point.y);
            }
            PathSegment::CubicTo(control1, control2, point) => {
                flatten_curve(
                    &mut builder,
                    [
                        current,
                        (control1.x, control1.y),
                        (control2.x, control2.y),
                        (point.x, point.y),
                    ],
                    0,
                );
                current = (point.x, point.y);
            }
            PathSegment::Close => {
                builder.close();
                current = start;
            }
        }
    }
    builder.finish()
}

// Flatten in image coordinates once, with subpixel tolerance. Letting each band flatten
// a clipped cubic independently changes its approximation and creates visible seams.
fn flatten_curve(builder: &mut PathBuilder, points: [(f32, f32); 4], depth: u8) {
    let [from, control1, control2, to] = points;
    let delta = (to.0 - from.0, to.1 - from.1);
    let distance =
        |point: (f32, f32)| ((point.0 - from.0) * delta.1 - (point.1 - from.1) * delta.0).abs();
    let deviation = distance(control1) + distance(control2);
    let chord_squared = delta.0 * delta.0 + delta.1 * delta.1;
    let flat = if chord_squared == 0.0 {
        [control1, control2]
            .iter()
            .all(|point| (point.0 - from.0).hypot(point.1 - from.1) <= 0.05)
    } else {
        deviation * deviation <= 0.05 * 0.05 * chord_squared
    };
    // Collinear controls can still double back; retain that part of the path too.
    let polygon_length: f32 = points
        .windows(2)
        .map(|pair| (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1))
        .sum();
    if (flat && polygon_length - chord_squared.sqrt() <= 0.05) || depth == 16 {
        builder.line_to(to.0, to.1);
        return;
    }
    let midpoint =
        |left: (f32, f32), right: (f32, f32)| ((left.0 + right.0) / 2.0, (left.1 + right.1) / 2.0);
    let first = midpoint(from, control1);
    let middle = midpoint(control1, control2);
    let last = midpoint(control2, to);
    let left = midpoint(first, middle);
    let right = midpoint(middle, last);
    let split = midpoint(left, right);
    flatten_curve(builder, [from, first, left, split], depth + 1);
    flatten_curve(builder, [split, right, last, to], depth + 1);
}

/// An arrowhead is a chevron: two strokes that meet at the tip. Both dimensions are fractions
/// of the cell height, so they scale with the plot. The width is how far apart the stroke ends
/// are across the edge, and the length is how far the tip lies from them along it.
const ARROWHEAD_WIDTH_CELLS: f32 = 0.40;
const ARROWHEAD_LENGTH_CELLS: f32 = 0.30;
/// Wider than an edge's one-pixel stroke, so the chevron stays visible in a shrunk plot.
const ARROW_STROKE_WIDTH: f32 = 2.0;
const ARROW_MITER_LIMIT: f32 = 10.0;

/// The horizontal pixel span of an arrow centered on `center_x`, snapped to whole pixels. The
/// span depends only on the center, never on the arrow's direction: a rightward and a leftward
/// arrow at the same point cover the same columns and differ only in which end is the tip.
/// Snapping the tip and base independently (tip first for rightward, base first for leftward)
/// would round them opposite ways and leave the pair a pixel apart.
fn arrow_horizontal_span(center_x: f32, half_length: f32) -> (f32, f32) {
    let length = (2.0 * half_length).round().max(1.0);
    let left = (center_x - length / 2.0 + 0.5).floor();
    (left, left + length)
}

/// A chevron centered on `center`, in pixels, pointing the way `mark` says.
fn push_arrow(mark: &ArrowMark, center: (f32, f32), config: PngConfig, paths: &mut Vec<EdgePath>) {
    let half_width = config.cell_height as f32 * ARROWHEAD_WIDTH_CELLS / 2.0;
    let half_length = config.cell_height as f32 * ARROWHEAD_LENGTH_CELLS / 2.0;
    let pixel_direction = (
        mark.direction.0 as f32 * config.cell_width as f32,
        mark.direction.1 as f32 * config.cell_height as f32,
    );
    let magnitude = pixel_direction.0.hypot(pixel_direction.1);
    let forward = (pixel_direction.0 / magnitude, pixel_direction.1 / magnitude);
    let perpendicular = (-forward.1, forward.0);
    let (tip, base) = if mark.direction.1 == 0.0 {
        let (left, right) = arrow_horizontal_span(center.0, half_length);
        let (tip, base) = if mark.direction.0 > 0.0 {
            (right, left)
        } else {
            (left, right)
        };
        ((tip, center.1), (base, center.1))
    } else {
        (
            (
                center.0 + forward.0 * half_length,
                center.1 + forward.1 * half_length,
            ),
            (
                center.0 - forward.0 * half_length,
                center.1 - forward.1 * half_length,
            ),
        )
    };
    let mut builder = PathBuilder::new();
    builder.move_to(
        base.0 - perpendicular.0 * half_width,
        base.1 - perpendicular.1 * half_width,
    );
    builder.line_to(tip.0, tip.1);
    builder.line_to(
        base.0 + perpendicular.0 * half_width,
        base.1 + perpendicular.1 * half_width,
    );
    let stroke = Stroke {
        width: ARROW_STROKE_WIDTH,
        line_cap: LineCap::Round,
        miter_limit: ARROW_MITER_LIMIT,
        ..Stroke::default()
    };
    let Some(outline) = builder
        .finish()
        .and_then(|path| path.stroke(&stroke, 1.0))
        .and_then(|outline| flatten_path(&outline))
    else {
        return;
    };
    let color = color_to_rgb(mark.color, config.foreground);
    let mut paint = Paint::default();
    paint.set_color_rgba8(color[0], color[1], color[2], 255);
    paint.anti_alias = true;
    paths.push(EdgePath {
        path: outline,
        paint,
    });
}

/// Surround highlighted cells with a soft halo of concentric rounded rings that fade outward
/// like a Gaussian. The overlay is drawn behind node backgrounds, so only the part outside the
/// nodes shows. A side that continues into the rest of its node is left open at the cell
/// boundary, so the glow does not bulge past it above and below the node.
fn push_node_glow(
    glow: &NodeGlow,
    point: impl Fn((f64, f64)) -> (f32, f32),
    config: PngConfig,
    paths: &mut Vec<EdgePath>,
) {
    let light_theme = luminance(config.background) > luminance(config.foreground);
    let color = shift_toward_contrast(
        color_to_rgb(glow.color, config.foreground),
        GLOW_BRIGHTNESS,
        light_theme,
    );
    let (left, top) = point(glow.top_left);
    let (right, bottom) = point(glow.bottom_right);
    for ring in 0..GLOW_RINGS {
        let opacity = (glow_opacity(GLOW_PEAK, ring) * 255.0).round() as u8;
        let stroke = Stroke {
            width: 1.0,
            line_cap: LineCap::Butt,
            ..Stroke::default()
        };
        let distance = ring as f32 + 0.5;
        let mut builder = PathBuilder::new();
        let (x0, x1) = (
            left - if glow.clip_left { 0.0 } else { distance },
            right + if glow.clip_right { 0.0 } else { distance },
        );
        let (y0, y1) = (top - distance, bottom + distance);
        // Corners follow the true distance field, so they are round rather than square.
        let radius = distance;
        // One continuous stroke per ring keeps the corners joined. Starting the path on a
        // clipped side leaves it open there; a second `move_to` would make `close` return to
        // the wrong point and drop the unclipped side.
        match (glow.clip_left, glow.clip_right) {
            (true, true) => {
                builder.move_to(x0, y0);
                builder.line_to(x1, y0);
                builder.move_to(x0, y1);
                builder.line_to(x1, y1);
            }
            (true, false) => {
                builder.move_to(x0, y0);
                builder.line_to(x1 - radius, y0);
                builder.quad_to(x1, y0, x1, y0 + radius);
                builder.line_to(x1, y1 - radius);
                builder.quad_to(x1, y1, x1 - radius, y1);
                builder.line_to(x0, y1);
            }
            (false, true) => {
                builder.move_to(x1, y0);
                builder.line_to(x0 + radius, y0);
                builder.quad_to(x0, y0, x0, y0 + radius);
                builder.line_to(x0, y1 - radius);
                builder.quad_to(x0, y1, x0 + radius, y1);
                builder.line_to(x1, y1);
            }
            (false, false) => {
                builder.move_to(x0, y0 + radius);
                builder.quad_to(x0, y0, x0 + radius, y0);
                builder.line_to(x1 - radius, y0);
                builder.quad_to(x1, y0, x1, y0 + radius);
                builder.line_to(x1, y1 - radius);
                builder.quad_to(x1, y1, x1 - radius, y1);
                builder.line_to(x0 + radius, y1);
                builder.quad_to(x0, y1, x0, y1 - radius);
                builder.close();
            }
        }
        let Some(outline) = builder
            .finish()
            .and_then(|path| path.stroke(&stroke, 1.0))
            .and_then(|outline| flatten_path(&outline))
        else {
            continue;
        };
        let mut paint = Paint::default();
        paint.set_color_rgba8(color[0], color[1], color[2], opacity);
        paint.anti_alias = true;
        paths.push(EdgePath {
            path: outline,
            paint,
        });
    }
}

/// Draw a highlighted minimal node as a disc slightly larger than its text glyph, with the
/// layers of a highlighted edge: a Gaussian halo of nested discs around a bright core. Unlike
/// [`push_node_glow`] this follows the round symbol, whose cell is mostly empty.
fn push_symbol_glow(
    glow: &SymbolGlow,
    center: (f32, f32),
    config: PngConfig,
    paths: &mut Vec<EdgePath>,
) {
    let light_theme = luminance(config.background) > luminance(config.foreground);
    let color = color_to_rgb(glow.color, config.foreground);
    let core_radius = SYMBOL_RADIUS * config.cell_width as f32 + SYMBOL_GLOW_MARGIN;
    for (width, opacity, brightness) in edge_highlight_layers() {
        let radius = core_radius + (width - EDGE_CORE_WIDTH) / 2.0;
        let Some(disc) = PathBuilder::from_circle(center.0, center.1, radius)
            .and_then(|circle| flatten_path(&circle))
        else {
            continue;
        };
        let color = highlight_color(color, brightness, light_theme);
        let mut paint = Paint::default();
        paint.set_color_rgba8(color[0], color[1], color[2], opacity);
        paint.anti_alias = true;
        paths.push(EdgePath { path: disc, paint });
    }
}

/// Strokes of a highlighted edge as (width, opacity, brightness), widest first. The halo rings
/// are nested strokes that overlap, so each stroke's opacity is chosen such that the composite
/// at ring k equals the Gaussian target: (1 - target_k) = (1 - target_{k+1}) * (1 - opacity_k).
fn edge_highlight_layers() -> Vec<(f32, u8, f32)> {
    let mut layers = Vec::with_capacity(GLOW_RINGS + 1);
    for ring in (0..GLOW_RINGS).rev() {
        let target = glow_opacity(GLOW_PEAK, ring);
        let outer = if ring + 1 < GLOW_RINGS {
            glow_opacity(GLOW_PEAK, ring + 1)
        } else {
            0.0
        };
        let opacity = 1.0 - (1.0 - target) / (1.0 - outer);
        let width = EDGE_CORE_WIDTH + 2.0 * (ring + 1) as f32;
        layers.push((width, (opacity * 255.0).round() as u8, GLOW_BRIGHTNESS));
    }
    layers.push((EDGE_CORE_WIDTH, 255, EDGE_CORE_BRIGHTNESS));
    layers
}

fn prepare_strokes(
    mut path: Path,
    color: [u8; 3],
    line_style: LineStyle,
    config: PngConfig,
    paths: &mut Vec<EdgePath>,
) {
    let light_theme = luminance(config.background) > luminance(config.foreground);
    if line_style == LineStyle::Dashed {
        let dash = tiny_skia::StrokeDash::new(
            vec![
                config.cell_width as f32 * DASH_CELLS,
                config.cell_width as f32 * GAP_CELLS,
            ],
            0.0,
        )
        .expect("should have positive dash lengths");
        let Some(dashed) = path.dash(&dash, 1.0) else {
            return;
        };
        path = dashed;
    }
    let layers = if line_style == LineStyle::Bold {
        edge_highlight_layers()
    } else {
        vec![(1.0, 255, 0.0)]
    };
    for (width, opacity, brightness) in layers {
        let stroke = Stroke {
            width,
            // Stop halo layers at the endpoint's tangent plane so adjoining segments
            // do not composite translucent caps over one another.
            line_cap: if line_style == LineStyle::Bold && width > EDGE_CORE_WIDTH {
                LineCap::Butt
            } else {
                LineCap::Round
            },
            ..Stroke::default()
        };
        let Some(outline) = path
            .stroke(&stroke, 1.0)
            .and_then(|outline| flatten_path(&outline))
        else {
            continue;
        };
        let color = highlight_color(color, brightness, light_theme);
        let mut paint = Paint::default();
        paint.set_color_rgba8(color[0], color[1], color[2], opacity);
        paint.anti_alias = true;
        paths.push(EdgePath {
            path: outline,
            paint,
        });
    }
}

/// Points sampled along a curve to find the cells it crosses.
const CURVE_BOUNDS_STEPS: usize = 32;

/// Points along `curve`, from one end to the other.
fn curve_points(curve: &BezierCurve) -> Vec<(f64, f64)> {
    (0..=CURVE_BOUNDS_STEPS)
        .map(|step| {
            let t = step as f64 / CURVE_BOUNDS_STEPS as f64;
            let s = 1.0 - t;
            let weight = |first: f64, second: f64, third: f64, fourth: f64| {
                s * s * s * first
                    + 3.0 * s * s * t * second
                    + 3.0 * s * t * t * third
                    + t * t * t * fourth
            };
            (
                weight(curve.from.0, curve.control1.0, curve.control2.0, curve.to.0),
                weight(curve.from.1, curve.control1.1, curve.control2.1, curve.to.1),
            )
        })
        .collect()
}

/// The cells that `lines` cross, clipped to `area`, or `None` without lines.
fn lines_bounds(lines: &[DeferredEdge], area: Rect) -> Option<Rect> {
    let cell = |value: f64, limit: u16| value.floor().clamp(0.0, f64::from(limit)) as u16;
    lines
        .iter()
        .flat_map(|edge| match edge {
            DeferredEdge::Line(line) => vec![line.from, line.to],
            // The control points of a loop that bulges far can lie well outside the curve, so
            // the curve's own points bound it, not its control polygon.
            DeferredEdge::Bezier(curve) => curve_points(curve),
            DeferredEdge::Arrow(mark) => vec![mark.center],
            DeferredEdge::SymbolGlow(glow) => vec![glow.center],
            DeferredEdge::NodeGlow(glow) => vec![glow.top_left, glow.bottom_right],
        })
        .map(|(x, y)| {
            Rect::new(
                cell(x, area.width.saturating_sub(1)),
                cell(y, area.height.saturating_sub(1)),
                1,
                1,
            )
        })
        .reduce(|bounds, point| bounds.union(point))
}

fn draw_paths(pixels: &mut [u8], top: usize, paths: &[EdgePath], pixmap: &mut Pixmap) {
    pixmap.data_mut().fill(0);
    let transform = Transform::from_translate(0.0, BAND_PADDING as f32 - top as f32);
    for edge in paths {
        let bounds = edge.path.bounds();
        // Include the widest highlight stroke and its antialias fringe in the band test.
        let padding = EDGE_GLOW_REACH + 1.0;
        if bounds.bottom() + padding < top as f32
            || bounds.top() - padding > (top + pixmap.height() as usize) as f32
        {
            continue;
        }
        // Fill a precomputed stroke outline instead of using the hairline fast path, whose
        // clipping can change cubic coverage between bands. Dashing and stroking happen once.
        pixmap.fill_path(&edge.path, &edge.paint, FillRule::Winding, transform, None);
    }

    for (pixel, rgba) in pixels.as_chunks_mut::<3>().0.iter_mut().zip(
        pixmap
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .skip(pixmap.width() as usize * BAND_PADDING as usize),
    ) {
        let alpha = u32::from(rgba[3]);
        if alpha == 0 {
            continue;
        }
        for channel in 0..3 {
            pixel[channel] = (u32::from(rgba[channel])
                + (u32::from(pixel[channel]) * (255 - alpha) + 127) / 255)
                .min(255) as u8;
        }
    }
}

fn luminance(color: [u8; 3]) -> u32 {
    u32::from(color[0]) * 2126 + u32::from(color[1]) * 7152 + u32::from(color[2]) * 722
}

/// Shift toward contrast, then restore saturation for the core layer only so the halo keeps
/// its soft look.
fn highlight_color(color: [u8; 3], brightness: f32, light_theme: bool) -> [u8; 3] {
    let shifted = shift_toward_contrast(color, brightness, light_theme);
    if brightness != EDGE_CORE_BRIGHTNESS {
        return shifted;
    }
    let gray = shifted.map(f32::from).iter().sum::<f32>() / 3.0;
    shifted.map(|channel| {
        (gray + (f32::from(channel) - gray) * EDGE_CORE_SATURATION)
            .clamp(0.0, 255.0)
            .round() as u8
    })
}

fn shift_toward_contrast(color: [u8; 3], amount: f32, light_theme: bool) -> [u8; 3] {
    let target = if light_theme { 0.0 } else { 255.0 };
    color.map(|channel| (f32::from(channel) + (target - f32::from(channel)) * amount).round() as u8)
}

#[cfg(test)]
mod tests {
    use gen_tui::{
        ArrowMark, BezierCurve, DeferredEdge, LineStyle, SymbolGlow, png_export::PngConfig,
    };
    use ratatui::{buffer::Buffer, layout::Rect, style::Color};
    use tiny_skia::Pixmap;

    use super::{
        BAND_PADDING, SYMBOL_RADIUS, arrow_horizontal_span, draw_paths, prepare_paths,
        write_png_with_lines, write_png_with_lines_full_viewport,
    };

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

    fn canvas(width: usize, height: usize) -> Vec<u8> {
        vec![0; width * height * 3]
    }

    fn draw(width: usize, height: usize, top: usize, curves: &[BezierCurve]) -> Vec<u8> {
        draw_with_config(
            width,
            height,
            top,
            curves,
            PngConfig {
                background: [0; 3],
                ..PngConfig::default()
            },
        )
    }

    fn draw_with_config(
        width: usize,
        height: usize,
        top: usize,
        curves: &[BezierCurve],
        config: PngConfig,
    ) -> Vec<u8> {
        let mut pixels = config.background.repeat(width * height);
        let edges: Vec<_> = curves.iter().copied().map(DeferredEdge::Bezier).collect();
        let paths = prepare_paths(&edges, (0, 0), config);
        let mut pixmap = Pixmap::new(width as u32, height as u32 + 2 * BAND_PADDING)
            .expect("should allocate test band");
        draw_paths(&mut pixels, top, &paths, &mut pixmap);
        pixels
    }

    #[test]
    fn test_streamed_bands_match_full_raster_for_curves_and_dashes() {
        for line_style in [LineStyle::Normal, LineStyle::Bold, LineStyle::Dashed] {
            let curve = curve(
                (1.25, 0.5),
                (8.25, 0.5),
                (2.25, 6.5),
                (9.25, 6.5),
                line_style,
            );
            let full = draw(120, 147, 0, &[curve]);
            let mut bands = Vec::new();
            for top in (0..147).step_by(21) {
                bands.extend(draw(120, 21, top, &[curve]));
            }
            let maximum_difference = full
                .iter()
                .zip(&bands)
                .map(|(left, right)| left.abs_diff(*right))
                .max()
                .unwrap_or(0);
            assert!(
                maximum_difference <= 16,
                "band seams changed coverage by {maximum_difference} for {line_style:?}"
            );
        }
    }

    #[test]
    fn test_node_glow_fades_outward_from_node_edge() {
        let config = PngConfig {
            background: [0; 3],
            ..PngConfig::default()
        };
        let edge = DeferredEdge::NodeGlow(gen_tui::NodeGlow {
            top_left: (5.0, 2.0),
            bottom_right: (10.0, 3.0),
            clip_left: false,
            clip_right: false,
            color: Color::Red,
        });
        let paths = prepare_paths(&[edge], (0, 0), config);
        let mut pixmap = Pixmap::new(200, 84 + 2 * BAND_PADDING).expect("should allocate band");
        let mut pixels = canvas(200, 84);
        draw_paths(&mut pixels, 0, &paths, &mut pixmap);
        let pixel = |x: usize, y: usize| pixels[(y * 200 + x) * 3];
        let node_left = 5 * config.cell_width;
        let row = 2 * config.cell_height + config.cell_height / 2;
        assert!(pixel(node_left - 1, row) > pixel(node_left - 4, row));
        assert!(pixel(node_left - 4, row) > 0);
        assert_eq!(pixel(node_left - 7, row), 0);
    }

    #[test]
    fn test_symbol_glow_rings_the_glyph_and_fades_outward() {
        let config = PngConfig {
            background: [0; 3],
            ..PngConfig::default()
        };
        let edge = DeferredEdge::SymbolGlow(SymbolGlow {
            center: (5.5, 2.5),
            color: Color::Red,
        });
        let paths = prepare_paths(&[edge], (0, 0), config);
        let mut pixmap = Pixmap::new(200, 84 + 2 * BAND_PADDING).expect("should allocate band");
        let mut pixels = canvas(200, 84);
        draw_paths(&mut pixels, 0, &paths, &mut pixmap);
        let pixel = |x: usize, y: usize| pixels[(y * 200 + x) * 3];
        let center_x = 5 * config.cell_width + config.cell_width / 2;
        let row = 2 * config.cell_height + config.cell_height / 2;
        let glyph_radius = (SYMBOL_RADIUS * config.cell_width as f32) as usize;
        assert!(pixel(center_x + glyph_radius, row) > pixel(center_x + glyph_radius + 3, row));
        assert!(pixel(center_x + glyph_radius + 3, row) > 0);
        assert_eq!(pixel(center_x + glyph_radius + 8, row), 0);
    }

    #[test]
    fn test_node_glow_stops_at_clipped_side() {
        let config = PngConfig {
            background: [0; 3],
            ..PngConfig::default()
        };
        let edge = DeferredEdge::NodeGlow(gen_tui::NodeGlow {
            top_left: (5.0, 2.0),
            bottom_right: (10.0, 3.0),
            clip_left: true,
            clip_right: false,
            color: Color::Red,
        });
        let paths = prepare_paths(&[edge], (0, 0), config);
        let mut pixmap = Pixmap::new(200, 84 + 2 * BAND_PADDING).expect("should allocate band");
        let mut pixels = canvas(200, 84);
        draw_paths(&mut pixels, 0, &paths, &mut pixmap);
        let pixel = |x: usize, y: usize| pixels[(y * 200 + x) * 3];
        let node_left = 5 * config.cell_width;
        let node_right = 10 * config.cell_width;
        let above = 2 * config.cell_height - 2;
        assert!(pixel(node_left + 2, above) > 0);
        assert_eq!(pixel(node_left - 1, above), 0);
        assert!(pixel(node_right + 1, above) > 0);
    }

    #[test]
    fn test_deferred_straight_edges_are_not_dropped() {
        let edge = DeferredEdge::Line(gen_tui::DiagonalLine {
            from: (1.0, 1.0),
            to: (8.0, 1.0),
            color: Color::Red,
            line_style: LineStyle::Normal,
        });
        let paths = prepare_paths(&[edge], (0, 0), PngConfig::default());
        let mut pixmap =
            Pixmap::new(120, 44 + 2 * BAND_PADDING).expect("should allocate test band");
        let mut pixels = canvas(120, 44);
        draw_paths(&mut pixels, 0, &paths, &mut pixmap);
        assert!(pixels.iter().any(|&value| value > 0));
    }

    #[test]
    fn test_bezier_shape_and_coverage_are_antialiased() {
        let curve = curve(
            (10.0, 20.0),
            (25.0, 2.0),
            (40.0, 2.0),
            (55.0, 20.0),
            LineStyle::Normal,
        );
        let pixels = draw(800, 600, 0, &[curve]);
        let red_at = |x: usize, y: usize| pixels[(y * 800 + x) * 3];
        assert!(
            red_at(357, 136) > 100,
            "curve should rise between its endpoints"
        );
        assert_eq!(
            red_at(357, 275),
            0,
            "curve should not become a straight chord"
        );
        assert!(
            pixels
                .as_chunks::<3>()
                .0
                .iter()
                .any(|p| p[0] > 0 && p[0] < 255),
            "stroke should be antialiased"
        );
    }

    #[test]
    fn test_bezier_styling_preserves_color_width_and_dashes() {
        let normal = curve(
            (10.0, 10.0),
            (20.0, 10.0),
            (30.0, 10.0),
            (40.0, 10.0),
            LineStyle::Normal,
        );
        let bold = BezierCurve {
            line_style: LineStyle::Bold,
            ..normal
        };
        let dashed = BezierCurve {
            line_style: LineStyle::Dashed,
            ..normal
        };
        let normal_pixels = draw(800, 600, 0, &[normal]);
        let bold_pixels = draw(800, 600, 0, &[bold]);
        let dashed_pixels = draw(800, 600, 0, &[dashed]);
        let colored = |pixels: &[u8]| {
            pixels
                .as_chunks::<3>()
                .0
                .iter()
                .filter(|pixel| pixel[0] > 0)
                .count()
        };
        assert!(colored(&bold_pixels) > colored(&normal_pixels));
        assert!(colored(&dashed_pixels) < colored(&normal_pixels));
        assert!(normal_pixels.as_chunks::<3>().0.iter().any(|p| p[0] > p[1]));
    }

    #[test]
    fn test_bold_edge_has_bright_core_and_soft_halo() {
        let bold = curve(
            (10.0, 10.5),
            (20.0, 10.5),
            (30.0, 10.5),
            (40.0, 10.5),
            LineStyle::Bold,
        );
        let pixels = draw(800, 600, 0, &[bold]);
        let pixel = |x: usize, y: usize| &pixels[(y * 800 + x) * 3..][..3];
        let core = pixel(300, 220);
        let halo = pixel(300, 224);
        assert!(core.iter().max().unwrap() > &200);
        assert!(core.iter().max().unwrap() - core.iter().min().unwrap() > 100);
        assert!(halo.iter().any(|channel| *channel > 0));
        assert!(halo.iter().zip(core).all(|(outer, inner)| outer < inner));
    }

    #[test]
    fn test_bold_edge_glow_is_even_across_a_horizontal_join() {
        let left = curve(
            (10.0, 10.5),
            (15.0, 10.5),
            (20.0, 10.5),
            (25.0, 10.5),
            LineStyle::Bold,
        );
        let right = curve(
            (25.0, 10.5),
            (30.0, 10.5),
            (35.0, 10.5),
            (40.0, 10.5),
            LineStyle::Bold,
        );
        let whole = curve(
            (10.0, 10.5),
            (20.0, 10.5),
            (30.0, 10.5),
            (40.0, 10.5),
            LineStyle::Bold,
        );
        let joined = draw(800, 600, 0, &[left, right]);
        let continuous = draw(800, 600, 0, &[whole]);
        for y in 223..230 {
            for x in 270..280 {
                let index = (y * 800 + x) * 3;
                assert_eq!(&joined[index..index + 3], &continuous[index..index + 3]);
            }
        }
    }

    #[test]
    fn test_bold_edge_has_dark_core_on_light_theme() {
        let bold = curve(
            (10.0, 10.5),
            (20.0, 10.5),
            (30.0, 10.5),
            (40.0, 10.5),
            LineStyle::Bold,
        );
        let config = PngConfig {
            background: [250; 3],
            foreground: [20; 3],
            ..PngConfig::default()
        };
        let pixels = draw_with_config(800, 600, 0, &[bold], config);
        let pixel = |x: usize, y: usize| &pixels[(y * 800 + x) * 3..][..3];
        let core = pixel(300, 220);
        let halo = pixel(300, 224);
        assert!(core.iter().max().unwrap() < &140);
        assert!(halo.iter().any(|channel| *channel < 250));
        assert!(halo.iter().zip(core).all(|(outer, inner)| outer > inner));
    }

    #[test]
    fn test_png_with_bezier_is_cropped_around_the_curve_not_its_control_points() {
        let area = Rect::new(0, 0, 40, 20);
        let buffer = Buffer::empty(area);
        let curves = [DeferredEdge::Bezier(curve(
            (5.0, 10.0),
            (20.0, 1.0),
            (30.0, 1.0),
            (35.0, 10.0),
            LineStyle::Normal,
        ))];
        let mut bytes = Vec::new();
        write_png_with_lines(&buffer, &curves, &mut bytes).expect("should write the png");
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
        write_png_with_lines_full_viewport(&buffer, &[], &mut bytes)
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

    #[test]
    fn test_bezier_is_clipped_to_each_band() {
        let curve = curve(
            (4.0, 0.0),
            (4.0, 0.5),
            (4.0, 1.5),
            (4.0, 2.0),
            LineStyle::Normal,
        );
        let middle = draw(100, 100, 20, &[curve]);
        let elsewhere = draw(100, 100, 60, &[curve]);
        assert!(middle.as_chunks::<3>().0.iter().any(|pixel| pixel[0] > 100));
        assert!(elsewhere.iter().all(|&value| value == 0));
    }

    #[test]
    fn test_arrow_span_ignores_direction_and_covers_whole_pixels() {
        for center in [10.0_f32, 10.25, 10.5, 10.75, 11.4, 99.5] {
            let (left, right) = arrow_horizontal_span(center, 2.7);
            assert_eq!(left.fract(), 0.0, "left edge at center {center}");
            assert_eq!(right - left, 5.0, "length at center {center}");
            assert!((center - (left + right) / 2.0).abs() <= 0.5);
        }
    }

    #[test]
    fn test_forward_and_reverse_arrows_mirror_each_other_about_the_same_center() {
        let config = PngConfig {
            background: [0; 3],
            ..PngConfig::default()
        };
        let render = |points_right: bool| {
            let mark = DeferredEdge::Arrow(ArrowMark {
                center: (3.37, 2.0),
                direction: (if points_right { 1.0 } else { -1.0 }, 0.0),
                color: Color::Red,
            });
            let paths = prepare_paths(&[mark], (0, 0), config);
            let bounds = paths[0].path.bounds();
            (bounds.left(), bounds.right())
        };
        // The two arrowheads are mirror images about one vertical line, so the sum of a left
        // edge and the opposite right edge is the same for both pairs.
        let (forward_left, forward_right) = render(true);
        let (reverse_left, reverse_right) = render(false);
        assert!((forward_left + reverse_right - (forward_right + reverse_left)).abs() < 1e-3);
    }
}
