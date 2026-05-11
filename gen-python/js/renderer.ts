import { CELL_SIZE, TEXT_SIZE, BOX_SCALE, cellFont, isBoxLike, Cell, GridMetrics } from "./grid";
import { GLYPH_SPLIT, NODE_GLYPH_CP } from "./glyphs";

export interface Frame {
  cols: number;
  rows: number;
  cells: Cell[];
  neutral_fg?: string;
  neutral_bg?: string;
}

// ─── Three-layer split-glyph rendering ───────────────────────────────────────
//
// Mixed-weight box-drawing chars (e.g. heavy-horizontal + light-vertical) cannot
// be represented by a single font glyph with per-cell colouring.  Instead they
// are decomposed into two sub-glyphs drawn in two passes inside a clipped cell:
//
//   1. light layer (lightCp)  — Plain arms, drawn in neutral_fg (may be absent)
//   2. heavy layer (heavyCp)  — Thick arms, drawn in cell fg (highlight colour)
//
// Combined with the canvas background this produces true three-colour rendering:
//   canvas bg  /  neutral_fg  /  cell fg
//
// The GLYPH_SPLIT map in glyphs.ts encodes the decomposition for every
// mixed- or pure-heavy codepoint.

function drawSplitGlyph(
  ctx: CanvasRenderingContext2D,
  heavyCp: number,
  lightCp: number | undefined,
  px: number,
  py: number,
  cellW: number,
  cellH: number,
  boxDrawX: number,
  boxDrawY: number,
  fg: string,
  neutral_fg: string,
  boxFont: string,
): void {
  ctx.save();
  ctx.beginPath();
  ctx.rect(px, py, cellW, cellH);
  ctx.clip();

  if (lightCp !== undefined) {
    ctx.fillStyle = neutral_fg;
    ctx.font = boxFont;
    ctx.save();
    ctx.translate(px + cellW / 2, py + cellH / 2);
    ctx.scale(BOX_SCALE, BOX_SCALE);
    ctx.fillText(String.fromCodePoint(lightCp), boxDrawX - cellW / 2, boxDrawY - cellH / 2);
    ctx.restore();
  }

  ctx.fillStyle = fg;
  ctx.font = boxFont;
  ctx.save();
  ctx.translate(px + cellW / 2, py + cellH / 2);
  ctx.scale(BOX_SCALE, BOX_SCALE);
  ctx.fillText(String.fromCodePoint(heavyCp), boxDrawX - cellW / 2, boxDrawY - cellH / 2);
  ctx.restore();

  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────

export function paintFrame(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  grid: GridMetrics,
  frame: Frame | null | undefined,
): Set<string> {
  const nodeCells = new Set<string>();
  if (!frame?.cells) return nodeCells;

  const { cellW, cellH, boxDrawX, boxDrawY, textX, textBaseline } = grid;
  const { cols, rows, cells, neutral_fg = "#cdd6f4", neutral_bg = "#1e1e2e" } = frame;

  canvas.width = cols * cellW;
  canvas.height = rows * cellH;
  ctx.fillStyle = neutral_bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";

  for (const cell of cells) {
    const { x: c, y: r, text } = cell;
    const fg = cell.fg ?? neutral_fg;
    const bg = cell.bg ?? neutral_bg;
    const px = c * cellW;
    const py = r * cellH;

    if (!text || text === " ") continue;

    const cp = text.codePointAt(0)!;
    if (!isBoxLike(text)) nodeCells.add(`${c},${r}`);

    if (cp === NODE_GLYPH_CP) {
      if (bg !== neutral_bg) {
        ctx.fillStyle = bg;
        ctx.fillRect(px, py, cellW, cellH);
      }
      ctx.fillStyle = fg;
      ctx.beginPath();
      ctx.arc(px + cellW / 2, py + cellH / 2, cellW / 2, 0, Math.PI * 2);
      ctx.fill();
      continue;
    }

    if (bg !== neutral_bg) {
      ctx.fillStyle = bg;
      ctx.fillRect(px, py, cellW, cellH);
    }

    const split = GLYPH_SPLIT.get(cp);
    if (split !== undefined) {
      // Mixed/pure-heavy box char: decompose into heavy + light sub-glyphs.
      const [heavyCp, lightCp] = split;
      const boxFont = cellFont(CELL_SIZE, { bold: false, italic: !!cell.italic });
      drawSplitGlyph(ctx, heavyCp, lightCp, px, py, cellW, cellH, boxDrawX, boxDrawY, fg, neutral_fg, boxFont);
    } else if (isBoxLike(text)) {
      // Plain box-drawing / block element: render at BOX_SCALE in cell fg.
      ctx.fillStyle = fg;
      ctx.font = cellFont(CELL_SIZE, cell);
      ctx.save();
      ctx.beginPath();
      ctx.rect(px, py, cellW, cellH);
      ctx.clip();
      ctx.translate(px + cellW / 2, py + cellH / 2);
      ctx.scale(BOX_SCALE, BOX_SCALE);
      ctx.fillText(text, boxDrawX - cellW / 2, boxDrawY - cellH / 2);
      ctx.restore();
    } else {
      // Regular text cell.
      ctx.fillStyle = fg;
      ctx.font = cellFont(TEXT_SIZE, cell);
      ctx.fillText(text, px + textX, py + textBaseline);
    }

    if (cell.underline) {
      ctx.fillStyle = fg;
      const underlineY = Math.min(py + cellH - 1, py + textBaseline + 1);
      ctx.fillRect(px, underlineY, cellW, 1);
    }
  }

  return nodeCells;
}
