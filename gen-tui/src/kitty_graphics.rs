//! Curved edges shown through the Kitty graphics protocol.
//!
//! The curved layout mode leaves the edges out of the cell buffer. A [`PixelOverlay`] rasterizes
//! them with the same antialiased strokes as the PNG export onto a transparent image the size of
//! the graph's region of the screen, and the terminal composites that image over the cells. Cells that hold text or a
//! background are cut out of the image, so an edge never needs to know which layer the terminal
//! draws a node on.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io::{self, Write},
    thread::{available_parallelism, scope},
    time::{Duration, Instant},
};

use base64::{Engine, engine::general_purpose::STANDARD};
use crossterm::{
    cursor::{RestorePosition, SavePosition},
    queue,
    terminal::window_size,
};
use flate2::{Compression, write::ZlibEncoder};
use ratatui::{buffer::Buffer, layout::Rect};
use tiny_skia::Pixmap;

use crate::{
    overlay_primitive::OverlayPrimitive, pixel_overlay::PixelOverlay, raster_config::RasterConfig,
};

const IMAGE_ID: u32 = 0x6765_6e31;
const PLACEMENT_ID: u32 = 1;
/// The protocol limits each escape sequence's payload to this many base64 bytes.
const CHUNK_BYTES: usize = 4096;
/// Cell size assumed when the terminal does not report its size in pixels. The image is
/// stretched over the cell grid anyway, so only the stroke weight depends on it.
const FALLBACK_CELL_PIXELS: (usize, usize) = (10, 20);
/// The height in pixels of the logical cell that a one-pixel stroke is drawn for.
const LOGICAL_CELL_HEIGHT: f32 = 21.0;

/// The id of the one-pixel image the support query asks the terminal to accept.
#[cfg(unix)]
const QUERY_ID: u32 = 31;
/// How long to wait for the terminal to answer, long enough for a round trip over ssh.
#[cfg(unix)]
const QUERY_TIMEOUT: Duration = Duration::from_millis(1500);

/// Ask the terminal whether it implements the Kitty graphics protocol. The graphics query is
/// followed by a primary device attributes request, which every terminal answers, so a terminal
/// that ignores the graphics query is told apart from a slow one by the second reply arriving
/// alone. Needs raw mode, and a multiplexer that does not pass the query through counts as
/// unsupported.
#[cfg(unix)]
pub fn supports_kitty_graphics() -> bool {
    let mut stdout = io::stdout();
    let sent = write!(
        stdout,
        "\x1b_Gi={QUERY_ID},s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\\x1b[c"
    )
    .and_then(|()| stdout.flush());
    if sent.is_err() {
        return false;
    }
    let reply = read_until_device_attributes(QUERY_TIMEOUT);
    graphics_acknowledged(&reply)
}

#[cfg(unix)]
fn graphics_acknowledged(reply: &[u8]) -> bool {
    let acknowledgement = format!("\x1b_Gi={QUERY_ID};OK");
    reply
        .windows(acknowledgement.len())
        .any(|window| window == acknowledgement.as_bytes())
}

/// Read raw bytes from standard input until the device attributes reply (`ESC [ ? ... c`) is
/// complete or `timeout` passes.
#[cfg(unix)]
fn read_until_device_attributes(timeout: Duration) -> Vec<u8> {
    let deadline = Instant::now() + timeout;
    let mut reply = Vec::new();
    let mut buffer = [0_u8; 256];
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return reply;
        }
        let mut descriptor = libc::pollfd {
            fd: libc::STDIN_FILENO,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `descriptor` is a valid pollfd and the count of one matches it.
        let ready = unsafe { libc::poll(&mut descriptor, 1, remaining.as_millis() as i32) };
        if ready <= 0 {
            return reply;
        }
        // SAFETY: `buffer` is writable for its whole length.
        let count =
            unsafe { libc::read(libc::STDIN_FILENO, buffer.as_mut_ptr().cast(), buffer.len()) };
        if count <= 0 {
            return reply;
        }
        reply.extend_from_slice(&buffer[..count as usize]);
        if let Some(start) = reply.windows(3).position(|window| window == b"\x1b[?")
            && reply[start..].contains(&b'c')
        {
            return reply;
        }
    }
}

/// Terminals that cannot be queried (no unix terminal to read the reply from) are treated as
/// not supporting the protocol.
#[cfg(not(unix))]
pub fn supports_kitty_graphics() -> bool {
    false
}

/// The most strips the screen is sent as, so they can be compressed on separate threads.
const MAX_RASTER_THREADS: usize = 8;

/// What the terminal currently shows for one strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StripState {
    Empty,
    Shown(u64),
}

/// What encoding one strip decided.
enum StripOutput {
    Unchanged,
    Clear,
    Upload { digest: u64, escapes: Vec<u8> },
}

/// Remembers what the terminal shows so unchanged work is not redone or sent again.
#[derive(Default)]
pub struct KittyOverlayPresenter {
    /// Digest of the overlay drawn last, so an unchanged picture skips the raster altogether.
    last_digest: Option<u64>,
    /// The cells the images cover. Strips are placed relative to it, so when it moves or
    /// resizes the old images are deleted instead of being compared with the new ones.
    region: Option<Rect>,
    /// What the terminal holds for each strip, so a strip whose pixels did not change is not
    /// sent again even when the frame around it did.
    strips: Vec<StripState>,
}

impl KittyOverlayPresenter {
    /// Show `overlay_primitives` over `region`, the part of `buffer` (the frame just drawn) that
    /// the graph widget was rendered into, or remove the images when there is nothing to show.
    /// The primitives are in cells of `buffer`, so they are measured from the region's top-left
    /// cell. Only the region's cells are rasterized and the images are placed at its top-left
    /// cell, so panels around the graph are never covered. The picture is sent as horizontal
    /// strips of whole cell rows, each its own image, encoded in parallel; strips that look as
    /// they did are skipped.
    pub fn update(
        &mut self,
        output: &mut impl Write,
        buffer: &Buffer,
        region: Rect,
        overlay_primitives: &[OverlayPrimitive],
    ) -> io::Result<()> {
        let region = region.intersection(buffer.area);
        if region.is_empty() {
            return self.clear(output);
        }
        let config = raster_config_for_cell_pixels(terminal_cell_pixels());
        let region_buffer = crop_buffer(buffer, region);
        let pixel_overlay = PixelOverlay::new(
            overlay_primitives,
            &region_buffer,
            (region.x, region.y),
            config,
        );
        if pixel_overlay.is_empty() {
            return self.clear(output);
        }
        if self.region != Some(region) {
            self.forget_images(output)?;
        }
        if self.last_digest == Some(pixel_overlay.digest()) {
            return Ok(());
        }
        self.last_digest = Some(pixel_overlay.digest());
        let Some(pixmap) = pixel_overlay.rasterize(region.width, region.height) else {
            return self.clear(output);
        };
        self.region = Some(region);
        let outputs = encode_strips(&pixmap, region, config.cell_height, &self.strips);
        queue!(output, SavePosition)?;
        let mut strips = Vec::with_capacity(outputs.len());
        for (index, result) in outputs.into_iter().enumerate() {
            let previous = self.strips.get(index).copied().unwrap_or(StripState::Empty);
            strips.push(match result {
                StripOutput::Unchanged => previous,
                StripOutput::Clear => {
                    delete_strip(output, index)?;
                    StripState::Empty
                }
                StripOutput::Upload { digest, escapes } => {
                    output.write_all(&escapes)?;
                    StripState::Shown(digest)
                }
            });
        }
        for index in strips.len()..self.strips.len() {
            delete_strip(output, index)?;
        }
        self.strips = strips;
        queue!(output, RestorePosition)?;
        output.flush()
    }

    pub fn clear(&mut self, output: &mut impl Write) -> io::Result<()> {
        self.last_digest = None;
        self.region = None;
        self.forget_images(output)
    }

    /// Delete every image the terminal holds and start from an empty picture.
    fn forget_images(&mut self, output: &mut impl Write) -> io::Result<()> {
        self.last_digest = None;
        let shown = !self.strips.is_empty();
        for (index, state) in self.strips.drain(..).enumerate() {
            if state != StripState::Empty {
                delete_strip(output, index)?;
            }
        }
        if shown {
            output.flush()?;
        }
        Ok(())
    }
}

/// The cells of `buffer` inside `region`, which lies within the buffer's area, keeping their
/// buffer coordinates.
fn crop_buffer(buffer: &Buffer, region: Rect) -> Buffer {
    let mut cropped = Buffer::empty(region);
    for row in region.top()..region.bottom() {
        for column in region.left()..region.right() {
            cropped[(column, row)] = buffer[(column, row)].clone();
        }
    }
    cropped
}

fn delete_strip(output: &mut impl Write, index: usize) -> io::Result<()> {
    write!(
        output,
        "\x1b_Ga=d,d=I,i={},q=2\x1b\\",
        IMAGE_ID + index as u32
    )
}

/// Cells smaller than this in either direction cannot hold a stroke, so the terminal's report
/// is taken to be wrong (some report cells instead of pixels over ssh).
const MIN_CELL_PIXELS: usize = 4;

fn terminal_cell_pixels() -> (usize, usize) {
    let Ok(size) = window_size() else {
        return FALLBACK_CELL_PIXELS;
    };
    if size.columns == 0 || size.rows == 0 {
        return FALLBACK_CELL_PIXELS;
    }
    let width = usize::from(size.width / size.columns);
    let height = usize::from(size.height / size.rows);
    if width < MIN_CELL_PIXELS || height < MIN_CELL_PIXELS {
        return FALLBACK_CELL_PIXELS;
    }
    (width, height)
}

fn raster_config_for_cell_pixels((cell_width, cell_height): (usize, usize)) -> RasterConfig {
    let scale = (cell_height as f32 / LOGICAL_CELL_HEIGHT).round().max(1.0) as usize;
    RasterConfig {
        cell_width,
        cell_height,
        scale,
        ..RasterConfig::default()
    }
}

/// Cut `pixmap` into strips of whole cell rows and encode each on its own thread, comparing it
/// with what the terminal shows for that strip in `shown`.
fn encode_strips(
    pixmap: &Pixmap,
    region: Rect,
    cell_height: usize,
    shown: &[StripState],
) -> Vec<StripOutput> {
    let strips = available_parallelism()
        .map_or(1, usize::from)
        .min(MAX_RASTER_THREADS)
        .min(usize::from(region.height).max(1));
    let strip_rows = usize::from(region.height).div_ceil(strips);
    let row_bytes = pixmap.width() as usize * 4;
    scope(|threads| {
        let handles: Vec<_> = pixmap
            .data()
            .chunks(strip_rows * cell_height * row_bytes)
            .enumerate()
            .map(|(index, block)| {
                let previous = shown.get(index).copied().unwrap_or(StripState::Empty);
                let placement = StripPlacement {
                    index,
                    first_column: region.x,
                    first_row: usize::from(region.y) + index * strip_rows,
                    columns: region.width,
                    rows: block.len() / row_bytes / cell_height,
                    width: pixmap.width(),
                };
                threads.spawn(move || encode_strip(block, &placement, previous))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("should encode a strip"))
            .collect()
    })
}

/// Where a strip goes on screen, in terminal cells: the image id follows its position from the
/// top of the region.
struct StripPlacement {
    index: usize,
    first_column: u16,
    first_row: usize,
    columns: u16,
    rows: usize,
    width: u32,
}

/// Premultiplied RGBA from the rasterizer to the straight alpha the protocol expects.
fn demultiply(premultiplied: &[u8]) -> Vec<u8> {
    let mut straight = vec![0_u8; premultiplied.len()];
    let (source_pixels, _) = premultiplied.as_chunks::<4>();
    let (target_pixels, _) = straight.as_chunks_mut::<4>();
    for (source, target) in source_pixels.iter().zip(target_pixels) {
        let alpha = u32::from(source[3]);
        if alpha == 0 {
            continue;
        }
        if alpha == 255 {
            target.copy_from_slice(source);
            continue;
        }
        for channel in 0..3 {
            target[channel] =
                ((u32::from(source[channel]) * 255 + alpha / 2) / alpha).min(255) as u8;
        }
        target[3] = source[3];
    }
    straight
}

/// Decide what the terminal needs for one strip, and build the escape sequences when it needs
/// an upload. The image sits below the text (`z=-1`), the cursor does not move, and sending the
/// same image and placement ids again replaces the previous strip without a visible gap.
fn encode_strip(block: &[u8], placement: &StripPlacement, previous: StripState) -> StripOutput {
    if block.as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 0) {
        return if previous == StripState::Empty {
            StripOutput::Unchanged
        } else {
            StripOutput::Clear
        };
    }
    let mut hasher = DefaultHasher::new();
    block.hash(&mut hasher);
    let digest = hasher.finish();
    if previous == StripState::Shown(digest) {
        return StripOutput::Unchanged;
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    let compressed = encoder
        .write_all(&demultiply(block))
        .and_then(|()| encoder.finish());
    let Ok(compressed) = compressed else {
        return StripOutput::Unchanged;
    };
    let payload = STANDARD.encode(compressed);
    let image_id = IMAGE_ID + placement.index as u32;
    let mut escapes = Vec::with_capacity(payload.len() + payload.len() / CHUNK_BYTES * 8 + 128);
    // Move to the strip's top-left cell (the sequence counts from 1); the caller saved the
    // cursor and restores it.
    let _ = write!(
        escapes,
        "\x1b[{};{}H",
        placement.first_row + 1,
        usize::from(placement.first_column) + 1
    );
    let mut chunks = payload.as_bytes().chunks(CHUNK_BYTES).peekable();
    let mut first = true;
    while let Some(chunk) = chunks.next() {
        let more = u8::from(chunks.peek().is_some());
        if first {
            let _ = write!(
                escapes,
                "\x1b_Ga=T,f=32,o=z,s={},v={},i={image_id},p={PLACEMENT_ID},c={},r={},C=1,z=-1,q=2,m={more};",
                placement.width,
                block.len() / 4 / placement.width as usize,
                placement.columns,
                placement.rows,
            );
            first = false;
        } else {
            let _ = write!(escapes, "\x1b_Gm={more};");
        }
        escapes.extend_from_slice(chunk);
        escapes.extend_from_slice(b"\x1b\\");
    }
    StripOutput::Upload { digest, escapes }
}

#[cfg(test)]
mod tests {
    use ratatui::{
        buffer::Buffer,
        layout::Rect,
        style::{Color, Style},
    };

    use super::*;
    use crate::{
        highlight_style::LineStyle,
        overlay_primitive::{BezierCurve, OverlayPrimitive},
    };

    fn opaque_strip(columns: u32, rows: u32) -> Vec<u8> {
        vec![255; (columns * 10 * rows * 20 * 4) as usize]
    }

    fn placement(columns: u16, rows: usize) -> StripPlacement {
        StripPlacement {
            index: 2,
            first_column: 0,
            first_row: 7,
            columns,
            rows,
            width: u32::from(columns) * 10,
        }
    }

    fn curve_across(row: f64) -> OverlayPrimitive {
        OverlayPrimitive::Bezier(BezierCurve {
            from: (1.0, row),
            control1: (10.0, row),
            control2: (30.0, row + 2.0),
            to: (39.0, row + 2.0),
            color: Color::Red,
            line_style: LineStyle::Normal,
        })
    }

    fn curve_between(left: f64, right: f64, row: f64) -> OverlayPrimitive {
        OverlayPrimitive::Bezier(BezierCurve {
            from: (left, row),
            control1: (left + 9.0, row),
            control2: (right - 9.0, row + 2.0),
            to: (right, row + 2.0),
            color: Color::Red,
            line_style: LineStyle::Normal,
        })
    }

    fn present(
        presenter: &mut KittyOverlayPresenter,
        buffer: &Buffer,
        region: Rect,
        overlay_primitives: &[OverlayPrimitive],
    ) -> String {
        let mut output = Vec::new();
        presenter
            .update(&mut output, buffer, region, overlay_primitives)
            .expect("should write to memory");
        String::from_utf8(output).expect("should be ASCII")
    }

    /// The (row, column) of every cursor positioning sequence, as the terminal counts them.
    fn cursor_positions(text: &str) -> Vec<(usize, usize)> {
        text.split("\x1b[")
            .skip(1)
            .filter_map(|rest| {
                let (position, _) = rest.split_once('H')?;
                let (row, column) = position.split_once(';')?;
                Some((row.parse().ok()?, column.parse().ok()?))
            })
            .collect()
    }

    /// The value of `key` in the header of every uploaded strip.
    fn header_values(text: &str, key: &str) -> Vec<usize> {
        text.split("\x1b_Ga=T")
            .skip(1)
            .map(|upload| {
                let header = upload.split(';').next().expect("should have a header");
                header
                    .split(',')
                    .find_map(|field| field.strip_prefix(key))
                    .expect("should name the key")
                    .parse()
                    .expect("should be a number")
            })
            .collect()
    }

    #[test]
    fn test_graphics_reply_is_recognized_among_other_bytes() {
        assert!(graphics_acknowledged(b"\x1b_Gi=31;OK\x1b\\\x1b[?62;c"));
        assert!(!graphics_acknowledged(b"\x1b[?62;c"));
        assert!(!graphics_acknowledged(b"\x1b_Gi=31;ENOTSUPPORTED:no\x1b\\"));
    }

    #[test]
    fn test_strip_upload_places_its_own_image_at_its_first_row() {
        let block = opaque_strip(40, 5);

        let StripOutput::Upload { escapes, .. } =
            encode_strip(&block, &placement(40, 5), StripState::Empty)
        else {
            panic!("an opaque strip should be uploaded");
        };

        let text = String::from_utf8(escapes).expect("should be ASCII");
        assert!(text.starts_with("\x1b[8;1H\x1b_Ga=T,f=32,o=z,s=400,v=100,i="));
        assert!(text.contains(&format!("i={},p=", IMAGE_ID + 2)));
        assert!(text.contains(",c=40,r=5,"));
        assert_eq!(
            text.matches("\x1b\\").count(),
            text.matches("\x1b_G").count()
        );
    }

    #[test]
    fn test_strip_that_looks_as_it_did_is_not_sent_again() {
        let block = opaque_strip(40, 5);
        let StripOutput::Upload { digest, .. } =
            encode_strip(&block, &placement(40, 5), StripState::Empty)
        else {
            panic!("should upload the first time");
        };

        let again = encode_strip(&block, &placement(40, 5), StripState::Shown(digest));

        assert!(matches!(again, StripOutput::Unchanged));
    }

    #[test]
    fn test_transparent_strip_is_cleared_only_if_something_was_shown() {
        let block = vec![0; 10 * 20 * 4 * 40 * 5];

        let after_picture = encode_strip(&block, &placement(40, 5), StripState::Shown(9));
        let after_nothing = encode_strip(&block, &placement(40, 5), StripState::Empty);

        assert!(matches!(after_picture, StripOutput::Clear));
        assert!(matches!(after_nothing, StripOutput::Unchanged));
    }

    #[test]
    fn test_demultiply_restores_straight_alpha() {
        let straight = demultiply(&[100, 50, 0, 200, 0, 0, 0, 0, 7, 8, 9, 255]);

        assert_eq!(straight[3], 200);
        assert_eq!(&straight[4..8], &[0, 0, 0, 0]);
        assert_eq!(&straight[8..12], &[7, 8, 9, 255]);
        assert!((i32::from(straight[0]) - 128).abs() <= 1);
    }

    #[test]
    fn test_overlay_sends_once_then_nothing_until_the_picture_changes() {
        let buffer = Buffer::empty(Rect::new(0, 0, 40, 10));
        let mut overlay = KittyOverlayPresenter::default();
        let mut first = Vec::new();
        let mut second = Vec::new();
        let mut moved = Vec::new();
        let mut cleared = Vec::new();

        overlay
            .update(&mut first, &buffer, buffer.area, &[curve_across(3.5)])
            .expect("should write to memory");
        overlay
            .update(&mut second, &buffer, buffer.area, &[curve_across(3.5)])
            .expect("should write to memory");
        overlay
            .update(&mut moved, &buffer, buffer.area, &[curve_across(5.5)])
            .expect("should write to memory");
        overlay
            .update(&mut cleared, &buffer, buffer.area, &[])
            .expect("should write to memory");

        assert!(first.starts_with(b"\x1b7") && first.ends_with(b"\x1b8"));
        assert!(second.is_empty(), "an identical frame sends nothing");
        assert!(!moved.is_empty());
        let removal = String::from_utf8(cleared).expect("should be ASCII");
        assert!(removal.contains("a=d,d=I"));
    }

    #[test]
    fn test_region_away_from_the_origin_places_images_at_its_top_left_cell() {
        let buffer = Buffer::empty(Rect::new(0, 0, 60, 20));
        let region = Rect::new(5, 3, 40, 10);
        let mut presenter = KittyOverlayPresenter::default();

        let text = present(
            &mut presenter,
            &buffer,
            region,
            &[curve_between(6.0, 44.0, 5.5)],
        );

        let positions = cursor_positions(&text);
        assert!(!positions.is_empty());
        for (row, column) in &positions {
            assert_eq!(*column, 6, "images start at the region's first column");
            assert!((4..=13).contains(row), "row {row} is outside the region");
        }
        assert!(
            header_values(&text, "c=")
                .iter()
                .all(|columns| *columns == 40)
        );
        assert!(header_values(&text, "s=").iter().all(|width| *width == 400));
        assert!(header_values(&text, "r=").iter().sum::<usize>() <= 10);
        for (index, (row, _)) in positions.iter().enumerate() {
            let rows = header_values(&text, "r=")[index];
            assert!(
                row - 1 + rows <= 13,
                "a strip runs past the region's bottom row"
            );
        }
    }

    #[test]
    fn test_primitives_outside_the_region_draw_nothing() {
        let buffer = Buffer::empty(Rect::new(0, 0, 60, 20));
        let region = Rect::new(5, 3, 30, 10);
        let mut presenter = KittyOverlayPresenter::default();

        let beside = present(
            &mut presenter,
            &buffer,
            region,
            &[curve_between(45.0, 58.0, 5.5)],
        );
        let below = present(
            &mut presenter,
            &buffer,
            region,
            &[curve_between(6.0, 30.0, 17.5)],
        );

        assert!(beside.is_empty());
        assert!(below.is_empty());
    }

    #[test]
    fn test_cells_outside_the_region_do_not_cut_into_the_image() {
        let mut buffer = Buffer::empty(Rect::new(0, 0, 60, 20));
        buffer.set_string(50, 5, "text", Style::default());
        let region = Rect::new(5, 3, 40, 10);
        let mut presenter = KittyOverlayPresenter::default();

        let text = present(
            &mut presenter,
            &buffer,
            region,
            &[curve_between(6.0, 44.0, 5.5)],
        );

        assert_eq!(header_values(&text, "s=").first(), Some(&400));
    }

    #[test]
    fn test_changing_the_region_deletes_old_images_before_placing_new_ones() {
        let buffer = Buffer::empty(Rect::new(0, 0, 60, 20));
        let curve = [curve_between(10.0, 44.0, 6.5)];
        let mut presenter = KittyOverlayPresenter::default();
        let first = present(&mut presenter, &buffer, Rect::new(5, 3, 40, 10), &curve);
        let shown = header_values(&first, "i=");

        let same = present(&mut presenter, &buffer, Rect::new(5, 3, 40, 10), &curve);
        let moved = present(&mut presenter, &buffer, Rect::new(8, 4, 40, 10), &curve);

        assert!(same.is_empty());
        assert!(!shown.is_empty());
        let first_upload = moved.find("a=T").expect("should place the new images");
        for id in shown {
            let deletion = format!("a=d,d=I,i={id},");
            let position = moved.find(&deletion).expect("should delete a shown image");
            assert!(position < first_upload);
        }
        for (_, column) in cursor_positions(&moved) {
            assert_eq!(column, 9);
        }
    }

    #[test]
    fn test_clearing_after_a_region_deletes_its_images() {
        let buffer = Buffer::empty(Rect::new(0, 0, 60, 20));
        let mut presenter = KittyOverlayPresenter::default();
        present(
            &mut presenter,
            &buffer,
            Rect::new(5, 3, 40, 10),
            &[curve_between(6.0, 44.0, 5.5)],
        );
        let mut cleared = Vec::new();

        presenter
            .clear(&mut cleared)
            .expect("should write to memory");
        let mut again = Vec::new();
        presenter.clear(&mut again).expect("should write to memory");

        assert!(
            String::from_utf8(cleared)
                .expect("should be ASCII")
                .contains("a=d")
        );
        assert!(again.is_empty());
    }
}
