use gen_tui::{ViewportState, WorldPos, WorldRect};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
};

fn truncations(label: &str) -> impl Iterator<Item = String> {
    let chars: Vec<char> = label.chars().collect();
    let full_len = chars.len();

    // For labels shorter than 4 chars, just return the full label.
    if full_len < 4 {
        vec![label.to_string()].into_iter()
    } else {
        let mut candidates = Vec::with_capacity(full_len - 1);

        candidates.push(label.to_string());

        // Keep at least 2 body characters before the ellipsis.
        // TODO: this is char-based, not display-width-based.
        for body_len in (2..full_len).rev() {
            let body: String = chars[..body_len].iter().collect();
            candidates.push(format!("{body}…"));
        }

        candidates.into_iter()
    }
}

/// All top-left positions where a label of `label_width` fits inside `search`,
/// ordered by Manhattan distance from `preferred`.
///
/// Ties prefer positions below `preferred` before positions above it, then smaller
/// horizontal offsets.
fn nearby_candidates(preferred: (u16, u16), search: Rect, label_width: u16) -> Vec<(u16, u16)> {
    let (px, py) = (preferred.0 as i32, preferred.1 as i32);

    let x_lo = search.x as i32;
    let x_hi = search.right() as i32 - label_width as i32;
    let y_lo = search.y as i32;
    let y_hi = search.bottom() as i32;

    if x_hi < x_lo {
        return Vec::new();
    }

    let mut out = Vec::new();

    for y in y_lo..y_hi {
        for x in x_lo..=x_hi {
            let dx = x - px;
            let dy = y - py;
            let dist = dx.abs() + dy.abs();

            // Sort lower/equal rows before upper rows when distance ties.
            let above = (dy < 0) as u8;

            out.push(((x as u16, y as u16), (dist, above, dx.abs())));
        }
    }

    out.sort_by_key(|&(_, key)| key);
    out.into_iter().map(|(pos, _)| pos).collect()
}

/// Try to place `label` near `annotation`.
///
/// For each text form, this tries:
///
/// 1. centered below the annotation rectangle
/// 2. centered above the annotation rectangle
/// 3. nearby positions inside the bounded search rectangle
///
/// The full label is tried first. Shorter ellipsis forms are only tried after
/// all positions have failed for the current text.
fn try_draw_label_at(
    buf: &mut Buffer,
    area: Rect,
    x: u16,
    y: u16,
    text: &str,
    color: Color,
) -> bool {
    if y < area.y || y >= area.bottom() {
        return false;
    }

    let width = text.chars().count() as u16;
    let x_start = x.max(area.x);
    let visible_width = (area.right() - x_start).min(width) as usize;
    let skip_chars = (x_start - x) as usize;

    if visible_width == 0 {
        return false;
    }

    let visible_text: String = text.chars().skip(skip_chars).take(visible_width).collect();

    if !(x_start..x_start + visible_width as u16)
        .all(|col| buf.cell((col, y)).is_some_and(|cell| cell.symbol() == " "))
    {
        return false;
    }

    buf.set_string(x_start, y, visible_text, Style::default().fg(color));
    true
}

/// Draw a label near a world-space position on the canvas.
///
/// `area_of_interest` is a tuple of (left_pos, right_pos) world-space corners of the annotation span.
/// - left_pos.x and right_pos.x define the x range (first to last block)
/// - left_pos.y and right_pos.y define the y range (min to max y across all blocks)
///   `max_distance` is the maximum Manhattan distance from the annotation to search for label placement.
///
/// For each text form, this tries:
/// 1. centered below the annotation rectangle
/// 2. centered above the annotation rectangle
/// 3. nearby positions inside the bounded search rectangle
///
/// The full label is tried first. Shorter ellipsis forms are only tried after
/// all positions have failed for the current text.
pub fn draw_label_near_pos(
    buf: &mut Buffer,
    area: Rect,
    area_of_interest: (WorldPos, WorldPos),
    label: &str,
    color: Color,
    viewport_state: &ViewportState,
    max_distance: u16,
) -> Option<(u16, u16)> {
    if area.width == 0 || area.height == 0 {
        return None;
    }

    let (left_pos, right_pos) = area_of_interest;
    let (term_x_a, term_y_a) = viewport_state.world_to_terminal(left_pos)?;
    let (term_x_b, term_y_b) = viewport_state.world_to_terminal(right_pos)?;

    let x_min = term_x_a.min(term_x_b);
    let x_max = term_x_a.max(term_x_b);
    let y_min = term_y_a.min(term_y_b);
    let y_max = term_y_a.max(term_y_b);

    if y_max < area.y || y_min >= area.bottom() || x_max < area.x || x_min >= area.right() {
        return None;
    }

    let annotation_width = x_max - x_min + 1;
    let label_width = label.chars().count() as u16;
    let x_margin = (annotation_width / 2).saturating_add(label_width / 2);

    let annotation = Rect {
        x: x_min,
        y: y_min,
        width: x_max - x_min + 1,
        height: y_max - y_min + 1,
    };

    let center_x = annotation.x + annotation.width / 2;
    let annotation_top = annotation.y;
    let annotation_bottom = annotation.bottom();

    let search_left = annotation.x.saturating_sub(x_margin).max(area.x);
    let search_right = annotation
        .right()
        .saturating_add(x_margin)
        .min(area.right());

    let search_top = annotation.y.saturating_sub(max_distance).max(area.y);
    let search_bottom = annotation
        .bottom()
        .saturating_add(max_distance)
        .min(area.bottom());

    if search_right <= search_left || search_bottom <= search_top {
        return None;
    }

    let search = Rect {
        x: search_left,
        y: search_top,
        width: search_right - search_left,
        height: search_bottom - search_top,
    };

    for text in truncations(label) {
        let width = text.chars().count() as u16;

        let below = (center_x.saturating_sub(width / 2), annotation_bottom);

        if try_draw_label_at(buf, area, below.0, below.1, &text, color) {
            return Some(below);
        }

        let above = (
            center_x.saturating_sub(width / 2),
            annotation_top.saturating_sub(1),
        );

        if try_draw_label_at(buf, area, above.0, above.1, &text, color) {
            return Some(above);
        }

        for (x, y) in nearby_candidates(below, search, width) {
            if (x, y) == below || (x, y) == above {
                continue;
            }

            if try_draw_label_at(buf, area, x, y, &text, color) {
                return Some((x, y));
            }
        }
    }

    None
}

/// Render a footer note summarizing annotations that didn't get an inline label
/// in the current frame.
pub fn draw_annotation_overflow_note(
    buf: &mut Buffer,
    area: Rect,
    hidden_count: usize,
    style: Style,
    truncated: bool,
) {
    if hidden_count == 0 {
        return;
    }

    let header = if truncated {
        format!(" {hidden_count} annotations hidden in truncated view ")
    } else {
        format!(" {hidden_count} annotations hidden due to space constraints ")
    };
    let footer_y = area.bottom().saturating_sub(1);
    buf.set_string(area.x, footer_y, &header, style);
}

/// Mark a node that has annotations hidden from the current frame with a small
/// asterisk placed just above its top-right corner. That cell sits outside the
/// node's own rect and off any routed edge, so overwriting it doesn't clobber
/// other graph content.
pub fn draw_hidden_annotation_marker(
    buf: &mut Buffer,
    area: Rect,
    node_center: WorldPos,
    node_size: (u64, u64),
    style: Style,
    viewport_state: &ViewportState,
) {
    let rect = WorldRect::from_center_and_size(node_center, node_size);
    let marker_pos = WorldPos::new(rect.right() + 1, rect.top() + 1);
    let Some((x, y)) = viewport_state.world_to_terminal(marker_pos) else {
        return;
    };
    if x >= area.x && x < area.right() && y >= area.y && y < area.bottom() {
        buf.set_string(x, y, "*", style);
    }
}

/// Render a footer legend explaining the hidden-annotation marker, shown only
/// when at least one marker was drawn this frame.
pub fn draw_hidden_annotation_legend(
    buf: &mut Buffer,
    area: Rect,
    any_hidden: bool,
    style: Style,
    truncated: bool,
) {
    if !any_hidden {
        return;
    }

    let text = if truncated {
        " * zoom in for more features "
    } else {
        " * some annotations hidden due to space constraints "
    };
    let footer_y = area.bottom().saturating_sub(1);
    buf.set_string(area.x, footer_y, text, style);
}
