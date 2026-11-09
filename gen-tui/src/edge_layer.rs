//! The curved edges of a graph rasterized at pixel resolution.
//!
//! A curved edge style leaves the edges out of the cell buffer and reports them as
//! [`EdgeShape`]s. An [`EdgeLayer`] draws those shapes with antialiased strokes onto a
//! transparent image laid over the buffer's cell grid, which a terminal can display through the
//! Kitty graphics protocol (see `kitty_graphics`) and the PNG export composites onto its text.
//!
//! Both routes show the same picture because the layer is cut out of every cell that holds text
//! or a node background. Edges attach at node borders, so the cut leaves them visible, and it
//! means the layer can sit on top of the cells without anyone deciding which layer the nodes
//! are on.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::OnceLock,
    thread::{available_parallelism, scope},
};

use ratatui::{buffer::Buffer, layout::Rect, style::Color};
use tiny_skia::{
    FillRule, IntSize, LineCap, Paint, Path, PathBuilder, PathSegment, Pixmap, PixmapMut, Stroke,
    Transform,
};

use crate::{
    curved::{ArrowMark, BezierCurve, EdgeShape, NodeGlow, SymbolGlow},
    graph_widget::NODE_GLYPH,
    plotter::LineStyle,
    png_export::{PngConfig, color_to_rgb},
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

/// How far past a path's bounds a stroke or its antialias fringe can still touch a band, in
/// pixels at `scale`.
fn band_reach(scale: f32) -> f32 {
    (EDGE_GLOW_REACH + 1.0) * scale
}

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

/// The cells of a buffer whose pixels the layer must leave clear: those with text, or with a
/// background other than the canvas.
struct CellMask {
    columns: usize,
    cell_width: usize,
    cell_height: usize,
    occupied: Vec<bool>,
}

impl CellMask {
    #[cfg(test)]
    fn empty() -> Self {
        Self {
            columns: 0,
            cell_width: 0,
            cell_height: 0,
            occupied: Vec::new(),
        }
    }

    fn of(buffer: &Buffer, config: PngConfig) -> Self {
        let occupied = buffer
            .content
            .iter()
            .map(|cell| {
                !cell.symbol().trim().is_empty()
                    || color_to_rgb(cell.bg, config.background) != config.background
            })
            .collect();
        Self {
            columns: usize::from(buffer.area.width),
            cell_width: config.cell_width,
            cell_height: config.cell_height,
            occupied,
        }
    }

    /// Clear the occupied cells' pixels in `row`, one RGBA row of the image at `image_row`.
    fn clear_row(&self, row: &mut [u8], image_row: usize) {
        if self.columns == 0 {
            return;
        }
        let cells = image_row / self.cell_height * self.columns;
        let Some(cells_in_row) = self.occupied.get(cells..cells + self.columns) else {
            return;
        };
        for (column, _) in cells_in_row.iter().enumerate().filter(|(_, taken)| **taken) {
            let start = column * self.cell_width * 4;
            row[start..start + self.cell_width * 4].fill(0);
        }
    }
}

/// The most horizontal strips a full raster is split into, one thread each.
const MAX_STRIPS: usize = 8;

/// Fewer shapes than this are prepared on the calling thread, where spawning would cost more
/// than it saves.
const PARALLEL_PREPARE_SHAPES: usize = 256;

/// Edge shapes ready to rasterize over a buffer. The strokes are built on first use, so a caller
/// can compare [`Self::digest`] with the previous frame's and skip the work entirely.
pub struct EdgeLayer {
    visible: Vec<EdgeShape>,
    origin: (u16, u16),
    paths: OnceLock<Vec<EdgePath>>,
    mask: CellMask,
    config: PngConfig,
    digest: u64,
}

impl EdgeLayer {
    /// Prepare `shapes`, in cells with `origin` as the top-left corner of the image, over
    /// `buffer`. One cell of `config` is one cell of the buffer, so the image is
    /// `buffer.area.width * config.cell_width` pixels wide. Shapes that cannot reach the
    /// buffer are dropped before any stroke is built: a window holds every placed edge, usually
    /// far more than the screen shows.
    pub fn new(
        shapes: &[EdgeShape],
        buffer: &Buffer,
        origin: (u16, u16),
        config: PngConfig,
    ) -> Self {
        let visible: Vec<EdgeShape> = shapes
            .iter()
            .filter(|shape| reaches_view(shape, origin, buffer.area))
            .copied()
            .collect();
        let mask = CellMask::of(buffer, config);
        let mut hasher = DefaultHasher::new();
        (origin, config.cell_width, config.cell_height, config.scale).hash(&mut hasher);
        config.foreground.hash(&mut hasher);
        config.background.hash(&mut hasher);
        mask.columns.hash(&mut hasher);
        mask.occupied.hash(&mut hasher);
        for shape in &visible {
            hash_shape(shape, &mut hasher);
        }
        Self {
            visible,
            origin,
            paths: OnceLock::new(),
            mask,
            config,
            digest: hasher.finish(),
        }
    }

    /// Identifies what the layer draws: equal digests mean equal pictures, so a caller that
    /// already showed one need not draw it again.
    pub const fn digest(&self) -> u64 {
        self.digest
    }

    /// Whether the layer draws nothing.
    pub fn is_empty(&self) -> bool {
        self.visible.is_empty()
    }

    /// The strokes of the visible shapes, in paint order. A large frame is prepared on several
    /// threads, each taking a run of shapes, and the runs are joined back in order.
    fn paths(&self) -> &[EdgePath] {
        self.paths.get_or_init(|| {
            let threads = available_parallelism()
                .map_or(1, usize::from)
                .min(MAX_STRIPS);
            if self.visible.len() < PARALLEL_PREPARE_SHAPES || threads == 1 {
                return prepare_paths(&self.visible, self.origin, self.config);
            }
            let run = self.visible.len().div_ceil(threads);
            scope(|workers| {
                let handles: Vec<_> = self
                    .visible
                    .chunks(run)
                    .map(|shapes| workers.spawn(|| prepare_paths(shapes, self.origin, self.config)))
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|handle| handle.join().expect("should prepare strokes"))
                    .collect()
            })
        })
    }

    /// A scratch pixmap for [`Self::composite_band`] covering `width` pixels and `rows` image
    /// rows, with the margin that keeps antialiasing at a band's edge from changing coverage.
    pub fn band_pixmap(&self, width: usize, rows: usize) -> Option<Pixmap> {
        let padding = BAND_PADDING * self.config.scale as u32;
        Pixmap::new(width as u32, rows as u32 + 2 * padding)
    }

    /// Composite the layer onto `pixels`, the RGB rows of the image starting at image row `top`,
    /// using `scratch` from [`Self::band_pixmap`]. Bands may arrive in any order.
    pub fn composite_band(&self, pixels: &mut [u8], top: usize, scratch: &mut Pixmap) {
        draw_paths(pixels, top, (self.paths(), &self.mask), scratch);
    }

    /// The whole layer as a transparent pixmap, or `None` when nothing is drawn. Pixels are
    /// premultiplied RGBA. Horizontal strips are drawn on separate threads.
    pub fn rasterize(&self, columns: u16, rows: u16) -> Option<Pixmap> {
        let strips = available_parallelism()
            .map_or(1, usize::from)
            .min(MAX_STRIPS);
        self.rasterize_in_strips(columns, rows, strips)
    }

    fn rasterize_in_strips(&self, columns: u16, rows: u16, strips: usize) -> Option<Pixmap> {
        let paths = self.paths();
        if paths.is_empty() {
            return None;
        }
        let width = usize::from(columns) * self.config.cell_width;
        let height = usize::from(rows) * self.config.cell_height;
        let size = IntSize::from_wh(width as u32, height as u32)?;
        let row_bytes = width * 4;
        let mut data = vec![0_u8; row_bytes * height];
        // Strips are whole cell rows, so every thread owns a disjoint block of image rows.
        let strip_rows = usize::from(rows).div_ceil(strips.max(1)) * self.config.cell_height;
        scope(|threads| {
            for (index, block) in data.chunks_mut(strip_rows * row_bytes).enumerate() {
                let top = index * strip_rows;
                threads.spawn(move || self.draw_strip(block, top, (width, row_bytes), paths));
            }
        });
        Pixmap::from_vec(data, size)
    }

    /// Draw the paths that reach the image rows from `top` into `block`, those rows' RGBA, then
    /// clear the occupied cells.
    fn draw_strip(
        &self,
        block: &mut [u8],
        top: usize,
        (width, row_bytes): (usize, usize),
        paths: &[EdgePath],
    ) {
        let rows = block.len() / row_bytes;
        let Some(mut strip) = PixmapMut::from_bytes(block, width as u32, rows as u32) else {
            return;
        };
        let transform = Transform::from_translate(0.0, -(top as f32));
        for edge in paths {
            let bounds = edge.path.bounds();
            if bounds.bottom() + edge.reach < top as f32
                || bounds.top() - edge.reach > (top + rows) as f32
            {
                continue;
            }
            strip.fill_path(&edge.path, &edge.paint, FillRule::Winding, transform, None);
        }
        for (row, bytes) in block.chunks_exact_mut(row_bytes).enumerate() {
            self.mask.clear_row(bytes, top + row);
        }
    }
}

/// Feed everything that decides how `shape` is drawn into `hasher`.
fn hash_shape(shape: &EdgeShape, hasher: &mut impl Hasher) {
    let mut point = |(x, y): (f64, f64)| {
        x.to_bits().hash(hasher);
        y.to_bits().hash(hasher);
    };
    match shape {
        EdgeShape::Line(line) => {
            point(line.from);
            point(line.to);
            (0_u8, line.color, line.line_style).hash(hasher);
        }
        EdgeShape::Bezier(curve) => {
            for control in [curve.from, curve.control1, curve.control2, curve.to] {
                point(control);
            }
            (1_u8, curve.color, curve.line_style).hash(hasher);
        }
        EdgeShape::Arrow(mark) => {
            point(mark.center);
            point(mark.direction);
            (2_u8, mark.color).hash(hasher);
        }
        EdgeShape::NodeGlow(glow) => {
            point(glow.top_left);
            point(glow.bottom_right);
            (3_u8, glow.clip_left, glow.clip_right, glow.color).hash(hasher);
        }
        EdgeShape::SymbolGlow(glow) => {
            point(glow.center);
            (4_u8, glow.color).hash(hasher);
        }
    }
}

/// How far, in cells, a stroke's halo and antialiasing reach beyond its path.
const REACH_MARGIN_CELLS: f64 = 2.0;

/// Whether `shape` can touch the view: `area`'s cells, whose top-left cell sits at `origin`. A
/// curve stays inside the box of its control points, so that box decides without sampling it.
fn reaches_view(shape: &EdgeShape, origin: (u16, u16), area: Rect) -> bool {
    let points: &[(f64, f64)] = match shape {
        EdgeShape::Line(line) => &[line.from, line.to],
        EdgeShape::Bezier(curve) => &[curve.from, curve.control1, curve.control2, curve.to],
        EdgeShape::Arrow(mark) => &[mark.center],
        EdgeShape::NodeGlow(glow) => &[glow.top_left, glow.bottom_right],
        EdgeShape::SymbolGlow(glow) => &[glow.center],
    };
    let (mut left, mut top) = (f64::INFINITY, f64::INFINITY);
    let (mut right, mut bottom) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(x, y) in points {
        left = left.min(x);
        right = right.max(x);
        top = top.min(y);
        bottom = bottom.max(y);
    }
    let view_left = f64::from(origin.0);
    let view_top = f64::from(origin.1);
    let view_right = view_left + f64::from(area.width);
    let view_bottom = view_top + f64::from(area.height);
    right + REACH_MARGIN_CELLS >= view_left
        && left - REACH_MARGIN_CELLS <= view_right
        && bottom + REACH_MARGIN_CELLS >= view_top
        && top - REACH_MARGIN_CELLS <= view_bottom
}

/// Recolor the cells of highlighted nodes in `buffer` (whose top-left cell is at `origin`) the
/// way the picture shows them: a node's background is lightened toward contrast where an
/// [`EdgeShape::NodeGlow`] covers it, and a minimal highlighted node loses its glyph so the
/// glow's disc is the whole node. Call this before building the [`EdgeLayer`] so the recolored
/// cells are cut out of it.
pub fn apply_highlight_cells(
    buffer: &mut Buffer,
    shapes: &[EdgeShape],
    origin: (u16, u16),
    config: PngConfig,
) {
    let width = buffer.area.width;
    let height = buffer.area.height;
    let light_theme = luminance(config.background) > luminance(config.foreground);
    for glow in shapes.iter().filter_map(|shape| match shape {
        EdgeShape::NodeGlow(glow) => Some(glow),
        _ => None,
    }) {
        let original = color_to_rgb(glow.color, config.foreground);
        let [red, green, blue] = highlight_color(original, EDGE_CORE_BRIGHTNESS, light_theme);
        let left = (glow.top_left.0.floor() as i32 - i32::from(origin.0)).max(0);
        let top = (glow.top_left.1.floor() as i32 - i32::from(origin.1)).max(0);
        let right = (glow.bottom_right.0.ceil() as i32 - i32::from(origin.0)).min(i32::from(width));
        let bottom =
            (glow.bottom_right.1.ceil() as i32 - i32::from(origin.1)).min(i32::from(height));
        for row in top..bottom {
            for column in left..right {
                let cell = &mut buffer.content[row as usize * usize::from(width) + column as usize];
                // Preserve the winning highlight where multiple loci overlap.
                if color_to_rgb(cell.bg, config.background) == original {
                    cell.set_bg(Color::Rgb(red, green, blue));
                }
            }
        }
    }
    // A minimal highlighted node is drawn as a disc by SymbolGlow. Keep its text glyph
    // out of the picture so the disc and halo are the whole visible node.
    for glow in shapes.iter().filter_map(|shape| match shape {
        EdgeShape::SymbolGlow(glow) => Some(glow),
        _ => None,
    }) {
        let column = glow.center.0.floor() as i32 - i32::from(origin.0);
        let row = glow.center.1.floor() as i32 - i32::from(origin.1);
        if column >= 0 && row >= 0 && column < i32::from(width) && row < i32::from(height) {
            let index = row as usize * usize::from(width) + column as usize;
            if buffer.content[index].symbol() == NODE_GLYPH.to_string() {
                buffer.content[index].set_symbol(" ");
            }
        }
    }
}

struct EdgePath {
    path: Path,
    paint: Paint<'static>,
    /// Farthest the stroke reaches from the path's bounds, in pixels, for band culling.
    reach: f32,
}

// Build geometry once per export. Each band clips the original path, rather than splitting
// curves, so tangents and dash phase remain continuous across PNG rows.
fn prepare_paths(lines: &[EdgeShape], origin: (u16, u16), config: PngConfig) -> Vec<EdgePath> {
    // A cell's center falls on a pixel center only when its size is odd; an even size needs half
    // a pixel to put one-pixel strokes on the middle pixel row or column, like the text glyphs.
    // A stroke that is an even number of pixels wide is centered on a pixel boundary already.
    let center_offset = |size: usize| {
        if size.is_multiple_of(2) && config.scale % 2 == 1 {
            0.5
        } else {
            0.0
        }
    };
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
        if let EdgeShape::Arrow(mark) = edge {
            push_arrow(mark, point(mark.center), config, &mut paths);
            continue;
        }
        if let EdgeShape::NodeGlow(glow) = edge {
            push_node_glow(glow, point, config, &mut paths);
            continue;
        }
        if let EdgeShape::SymbolGlow(glow) = edge {
            push_symbol_glow(glow, point(glow.center), config, &mut paths);
            continue;
        }
        let mut builder = PathBuilder::new();
        let (color, line_style) = match edge {
            EdgeShape::Line(line) => {
                let from = point(line.from);
                let to = point(line.to);
                builder.move_to(from.0, from.1);
                builder.line_to(to.0, to.1);
                (line.color, line.line_style)
            }
            EdgeShape::Bezier(curve) => {
                let from = point(curve.from);
                let control1 = point(curve.control1);
                let control2 = point(curve.control2);
                let to = point(curve.to);
                builder.move_to(from.0, from.1);
                builder.cubic_to(control1.0, control1.1, control2.0, control2.1, to.0, to.1);
                (curve.color, curve.line_style)
            }
            EdgeShape::Arrow(_) | EdgeShape::NodeGlow(_) | EdgeShape::SymbolGlow(_) => {
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
    let scale = config.scale as f32;
    let stroke = Stroke {
        width: ARROW_STROKE_WIDTH * scale,
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
        reach: band_reach(scale),
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
    let scale = config.scale as f32;
    for ring in 0..GLOW_RINGS {
        let opacity = (glow_opacity(GLOW_PEAK, ring) * 255.0).round() as u8;
        let stroke = Stroke {
            width: scale,
            line_cap: LineCap::Butt,
            ..Stroke::default()
        };
        let distance = (ring as f32 + 0.5) * scale;
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
            reach: band_reach(scale),
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
    let scale = config.scale as f32;
    let core_radius = SYMBOL_RADIUS * config.cell_width as f32 + SYMBOL_GLOW_MARGIN * scale;
    for (width, opacity, brightness) in edge_highlight_layers(scale) {
        let radius = core_radius + (width - EDGE_CORE_WIDTH * scale) / 2.0;
        let Some(disc) = PathBuilder::from_circle(center.0, center.1, radius)
            .and_then(|circle| flatten_path(&circle))
        else {
            continue;
        };
        let color = highlight_color(color, brightness, light_theme);
        let mut paint = Paint::default();
        paint.set_color_rgba8(color[0], color[1], color[2], opacity);
        paint.anti_alias = true;
        paths.push(EdgePath {
            path: disc,
            paint,
            reach: band_reach(scale),
        });
    }
}

/// Strokes of a highlighted edge as (width, opacity, brightness), widest first. The halo rings
/// are nested strokes that overlap, so each stroke's opacity is chosen such that the composite
/// at ring k equals the Gaussian target: (1 - target_k) = (1 - target_{k+1}) * (1 - opacity_k).
fn edge_highlight_layers(scale: f32) -> Vec<(f32, u8, f32)> {
    let mut layers = Vec::with_capacity(GLOW_RINGS + 1);
    for ring in (0..GLOW_RINGS).rev() {
        let target = glow_opacity(GLOW_PEAK, ring);
        let outer = if ring + 1 < GLOW_RINGS {
            glow_opacity(GLOW_PEAK, ring + 1)
        } else {
            0.0
        };
        let opacity = 1.0 - (1.0 - target) / (1.0 - outer);
        let width = (EDGE_CORE_WIDTH + 2.0 * (ring + 1) as f32) * scale;
        layers.push((width, (opacity * 255.0).round() as u8, GLOW_BRIGHTNESS));
    }
    layers.push((EDGE_CORE_WIDTH * scale, 255, EDGE_CORE_BRIGHTNESS));
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
    let scale = config.scale as f32;
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
        edge_highlight_layers(scale)
    } else {
        vec![(scale, 255, 0.0)]
    };
    for (width, opacity, brightness) in layers {
        let stroke = Stroke {
            width,
            // Stop halo layers at the endpoint's tangent plane so adjoining segments
            // do not composite translucent caps over one another.
            line_cap: if line_style == LineStyle::Bold && width > EDGE_CORE_WIDTH * scale {
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
            reach: band_reach(scale),
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

/// The cells that `shapes` cross, clipped to `area`, or `None` without shapes.
pub fn shapes_bounds(shapes: &[EdgeShape], area: Rect) -> Option<Rect> {
    let cell = |value: f64, limit: u16| value.floor().clamp(0.0, f64::from(limit)) as u16;
    shapes
        .iter()
        .flat_map(|edge| match edge {
            EdgeShape::Line(line) => vec![line.from, line.to],
            // The control points of a loop that bulges far can lie well outside the curve, so
            // the curve's own points bound it, not its control polygon.
            EdgeShape::Bezier(curve) => curve_points(curve),
            EdgeShape::Arrow(mark) => vec![mark.center],
            EdgeShape::SymbolGlow(glow) => vec![glow.center],
            EdgeShape::NodeGlow(glow) => vec![glow.top_left, glow.bottom_right],
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

/// Draw `paths` for the image rows from `top` into `pixmap`, which has `BAND_PADDING` scaled rows
/// of margin above and below the band, cut out of the cells `mask` holds, and composite the
/// result onto `pixels`, the band's RGB rows.
fn draw_paths(
    pixels: &mut [u8],
    top: usize,
    (paths, mask): (&[EdgePath], &CellMask),
    pixmap: &mut Pixmap,
) {
    pixmap.data_mut().fill(0);
    // The pixmap has the same padding above and below the band's rows.
    let band_rows = pixels.len() / (3 * pixmap.width() as usize);
    let band_padding = (pixmap.height() as usize - band_rows) / 2;
    let transform = Transform::from_translate(0.0, band_padding as f32 - top as f32);
    for edge in paths {
        let bounds = edge.path.bounds();
        // Include the widest highlight stroke and its antialias fringe in the band test.
        let padding = edge.reach;
        if bounds.bottom() + padding < top as f32
            || bounds.top() - padding > (top + pixmap.height() as usize) as f32
        {
            continue;
        }
        // Fill a precomputed stroke outline instead of using the hairline fast path, whose
        // clipping can change cubic coverage between bands. Dashing and stroking happen once.
        pixmap.fill_path(&edge.path, &edge.paint, FillRule::Winding, transform, None);
    }
    let row_bytes = pixmap.width() as usize * 4;
    for row in 0..band_rows {
        let start = (band_padding + row) * row_bytes;
        mask.clear_row(&mut pixmap.data_mut()[start..start + row_bytes], top + row);
    }

    for (pixel, rgba) in pixels.as_chunks_mut::<3>().0.iter_mut().zip(
        pixmap
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .skip(pixmap.width() as usize * band_padding),
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
    use ratatui::{buffer::Buffer, layout::Rect, style::Color};
    use tiny_skia::Pixmap;

    use super::{
        ArrowMark, BAND_PADDING, BezierCurve, CellMask, EdgeLayer, EdgeShape, SYMBOL_RADIUS,
        SymbolGlow, arrow_horizontal_span, draw_paths, prepare_paths,
    };
    use crate::{plotter::LineStyle, png_export::PngConfig};

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
        let edges: Vec<_> = curves.iter().copied().map(EdgeShape::Bezier).collect();
        let paths = prepare_paths(&edges, (0, 0), config);
        let mut pixmap = Pixmap::new(width as u32, height as u32 + 2 * BAND_PADDING)
            .expect("should allocate test band");
        draw_paths(&mut pixels, top, (&paths, &CellMask::empty()), &mut pixmap);
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
        let edge = EdgeShape::NodeGlow(crate::curved::NodeGlow {
            top_left: (5.0, 2.0),
            bottom_right: (10.0, 3.0),
            clip_left: false,
            clip_right: false,
            color: Color::Red,
        });
        let paths = prepare_paths(&[edge], (0, 0), config);
        let mut pixmap = Pixmap::new(200, 84 + 2 * BAND_PADDING).expect("should allocate band");
        let mut pixels = canvas(200, 84);
        draw_paths(&mut pixels, 0, (&paths, &CellMask::empty()), &mut pixmap);
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
        let edge = EdgeShape::SymbolGlow(SymbolGlow {
            center: (5.5, 2.5),
            color: Color::Red,
        });
        let paths = prepare_paths(&[edge], (0, 0), config);
        let mut pixmap = Pixmap::new(200, 84 + 2 * BAND_PADDING).expect("should allocate band");
        let mut pixels = canvas(200, 84);
        draw_paths(&mut pixels, 0, (&paths, &CellMask::empty()), &mut pixmap);
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
        let edge = EdgeShape::NodeGlow(crate::curved::NodeGlow {
            top_left: (5.0, 2.0),
            bottom_right: (10.0, 3.0),
            clip_left: true,
            clip_right: false,
            color: Color::Red,
        });
        let paths = prepare_paths(&[edge], (0, 0), config);
        let mut pixmap = Pixmap::new(200, 84 + 2 * BAND_PADDING).expect("should allocate band");
        let mut pixels = canvas(200, 84);
        draw_paths(&mut pixels, 0, (&paths, &CellMask::empty()), &mut pixmap);
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
        let edge = EdgeShape::Line(crate::curved::StraightSegment {
            from: (1.0, 1.0),
            to: (8.0, 1.0),
            color: Color::Red,
            line_style: LineStyle::Normal,
        });
        let paths = prepare_paths(&[edge], (0, 0), PngConfig::default());
        let mut pixmap =
            Pixmap::new(120, 44 + 2 * BAND_PADDING).expect("should allocate test band");
        let mut pixels = canvas(120, 44);
        draw_paths(&mut pixels, 0, (&paths, &CellMask::empty()), &mut pixmap);
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
            let mark = EdgeShape::Arrow(ArrowMark {
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

    #[test]
    fn test_edge_strokes_are_as_thick_as_the_scale() {
        let horizontal = curve(
            (1.0, 2.5),
            (2.0, 2.5),
            (5.0, 2.5),
            (6.0, 2.5),
            LineStyle::Normal,
        );
        for scale in [1_usize, 2, 3] {
            let config = PngConfig {
                background: [0; 3],
                ..PngConfig::default()
            }
            .with_scale(scale);
            let pixels = draw_with_config(300, 300, 0, &[horizontal], config);
            let x = 3 * config.cell_width;
            let lit_rows = (0..300)
                .filter(|&y| pixels[(y * 300 + x) * 3] > 200)
                .count();
            assert_eq!(
                lit_rows, scale,
                "fully lit rows of a stroke at scale {scale}"
            );
        }
    }

    #[test]
    fn test_cells_with_text_or_a_node_background_are_cut_out_of_the_layer() {
        let config = PngConfig {
            cell_width: 4,
            cell_height: 8,
            scale: 1,
            ..PngConfig::default()
        };
        let mut buffer = Buffer::empty(Rect::new(0, 0, 3, 1));
        buffer.set_string(1, 0, "A", ratatui::style::Style::default());
        buffer[(2, 0)].set_bg(Color::Rgb(200, 10, 10));
        let across = EdgeShape::Bezier(curve(
            (0.0, 0.5),
            (1.0, 0.5),
            (2.0, 0.5),
            (3.0, 0.5),
            LineStyle::Bold,
        ));
        let layer = EdgeLayer::new(&[across], &buffer, (0, 0), config);

        let pixmap = layer.rasterize(3, 1).expect("should draw the edge");

        let alpha_at = |x: u32, y: u32| pixmap.pixel(x, y).expect("should be in bounds").alpha();
        assert!(alpha_at(1, 4) > 0, "the blank first cell keeps the edge");
        assert_eq!(alpha_at(5, 4), 0, "the cell with text is cut out");
        assert_eq!(
            alpha_at(9, 4),
            0,
            "the cell with a node background is cut out"
        );
    }

    #[test]
    fn test_full_raster_matches_composited_bands_outside_cut_out_cells() {
        let config = PngConfig {
            background: [0; 3],
            ..PngConfig::default()
        };
        let mut buffer = Buffer::empty(Rect::new(0, 0, 6, 4));
        buffer.set_string(2, 1, "N", ratatui::style::Style::default());
        let edges = [EdgeShape::Bezier(curve(
            (0.5, 0.5),
            (4.0, 0.5),
            (2.0, 3.5),
            (5.5, 3.5),
            LineStyle::Normal,
        ))];
        let layer = EdgeLayer::new(&edges, &buffer, (0, 0), config);
        let width = 6 * config.cell_width;
        let rows = 4 * config.cell_height;
        let mut scratch = layer
            .band_pixmap(width, config.cell_height)
            .expect("should allocate a band");
        let mut bands = Vec::new();
        for top in (0..rows).step_by(config.cell_height) {
            let mut band = vec![0; width * config.cell_height * 3];
            layer.composite_band(&mut band, top, &mut scratch);
            bands.extend(band);
        }
        let full = layer.rasterize(6, 4).expect("should draw the edge");
        // Where the full raster is opaque red the bands must be bright too, and inside the
        // cut-out cell neither draws anything.
        let cut_pixel = (2 * config.cell_width + 1, config.cell_height + 3);
        let band_at = |x: usize, y: usize| bands[(y * width + x) * 3];
        assert_eq!(
            full.pixel(cut_pixel.0 as u32, cut_pixel.1 as u32)
                .unwrap()
                .alpha(),
            0
        );
        assert_eq!(band_at(cut_pixel.0, cut_pixel.1), 0);
        assert!(bands.iter().any(|&value| value > 100));
    }

    #[test]
    fn test_shapes_outside_the_view_are_not_prepared() {
        let config = PngConfig::default();
        let buffer = Buffer::empty(Rect::new(0, 0, 20, 5));
        let inside = EdgeShape::Bezier(curve(
            (1.0, 2.5),
            (8.0, 2.5),
            (12.0, 2.5),
            (19.0, 2.5),
            LineStyle::Normal,
        ));
        let far_right = EdgeShape::Bezier(curve(
            (500.0, 2.5),
            (510.0, 2.5),
            (520.0, 2.5),
            (530.0, 2.5),
            LineStyle::Normal,
        ));
        let far_below = EdgeShape::Bezier(curve(
            (2.0, 90.0),
            (4.0, 95.0),
            (6.0, 95.0),
            (8.0, 90.0),
            LineStyle::Normal,
        ));
        let straddling = EdgeShape::Bezier(curve(
            (-30.0, 2.5),
            (-10.0, 2.5),
            (10.0, 2.5),
            (30.0, 2.5),
            LineStyle::Normal,
        ));

        let only_outside = EdgeLayer::new(&[far_right, far_below], &buffer, (0, 0), config);
        let with_inside = EdgeLayer::new(&[far_right, inside, far_below], &buffer, (0, 0), config);
        let crossing = EdgeLayer::new(&[straddling], &buffer, (0, 0), config);

        assert!(only_outside.is_empty());
        assert_eq!(
            with_inside.paths().len(),
            EdgeLayer::new(&[inside], &buffer, (0, 0), config)
                .paths()
                .len()
        );
        assert!(
            !crossing.is_empty(),
            "an edge that crosses the view is kept"
        );
    }

    #[test]
    fn test_view_follows_the_origin_of_a_cropped_buffer() {
        let config = PngConfig::default();
        let buffer = Buffer::empty(Rect::new(0, 0, 10, 4));
        let at_origin = EdgeShape::Bezier(curve(
            (100.0, 52.0),
            (102.0, 52.0),
            (104.0, 52.0),
            (106.0, 52.0),
            LineStyle::Normal,
        ));
        assert!(EdgeLayer::new(&[at_origin], &buffer, (0, 0), config).is_empty());
        assert!(!EdgeLayer::new(&[at_origin], &buffer, (100, 50), config).is_empty());
    }

    #[test]
    fn test_digest_changes_with_the_shapes_and_the_cut_out_but_not_with_hidden_shapes() {
        let config = PngConfig::default();
        let buffer = Buffer::empty(Rect::new(0, 0, 20, 5));
        let shape = EdgeShape::Bezier(curve(
            (1.0, 2.5),
            (8.0, 2.5),
            (12.0, 1.5),
            (19.0, 1.5),
            LineStyle::Normal,
        ));
        let moved = EdgeShape::Bezier(curve(
            (1.0, 2.5),
            (8.0, 2.5),
            (12.0, 1.5),
            (19.0, 2.0),
            LineStyle::Normal,
        ));
        let hidden = EdgeShape::Bezier(curve(
            (900.0, 2.5),
            (908.0, 2.5),
            (912.0, 1.5),
            (919.0, 1.5),
            LineStyle::Normal,
        ));
        let mut with_text = buffer.clone();
        with_text.set_string(5, 2, "N", ratatui::style::Style::default());
        let digest = |shapes: &[EdgeShape], buffer: &Buffer| {
            EdgeLayer::new(shapes, buffer, (0, 0), config).digest()
        };

        assert_eq!(digest(&[shape], &buffer), digest(&[shape], &buffer));
        assert_eq!(digest(&[shape], &buffer), digest(&[hidden, shape], &buffer));
        assert_ne!(digest(&[shape], &buffer), digest(&[moved], &buffer));
        assert_ne!(digest(&[shape], &buffer), digest(&[shape], &with_text));
    }

    #[test]
    fn test_strips_drawn_on_separate_threads_match_one_strip() {
        let config = PngConfig::default();
        let mut buffer = Buffer::empty(Rect::new(0, 0, 30, 12));
        buffer.set_string(10, 5, "NNN", ratatui::style::Style::default());
        let shapes: Vec<_> = (0..40_u32)
            .map(|index| {
                let y = f64::from(index % 11) + 0.5;
                EdgeShape::Bezier(curve(
                    (1.0, y),
                    (12.0, y),
                    (18.0, 11.5 - y),
                    (29.0, 11.5 - y),
                    [LineStyle::Normal, LineStyle::Bold, LineStyle::Dashed][index as usize % 3],
                ))
            })
            .collect();
        let layer = EdgeLayer::new(&shapes, &buffer, (0, 0), config);

        let single = layer.rasterize_in_strips(30, 12, 1).expect("should draw");
        let split = layer.rasterize_in_strips(30, 12, 5).expect("should draw");

        let worst = single
            .data()
            .iter()
            .zip(split.data())
            .map(|(left, right)| left.abs_diff(*right))
            .max()
            .unwrap_or(0);
        // The same bound as streamed bands: coverage at a clip edge can shift by a few levels.
        assert!(worst <= 16, "strips changed a pixel by {worst}");
    }
}
