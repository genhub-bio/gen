import { CELL_SIZE, TEXT_SIZE, BOX_SCALE, cellFont, isBoxLike, Cell, GridMetrics } from "./grid";

export interface Frame {
  cols: number;
  rows: number;
  cells: Cell[];
  neutral_fg?: string;
  neutral_bg?: string;
}

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

    if (!isBoxLike(text)) nodeCells.add(`${c},${r}`);

    if (bg !== neutral_bg) {
      ctx.fillStyle = bg;
      ctx.fillRect(px, py, cellW, cellH);
    }

    ctx.fillStyle = fg;

    if (isBoxLike(text)) {
      ctx.font = cellFont(CELL_SIZE, { bold: false, italic: !!cell.italic });
      ctx.save();
      ctx.beginPath();
      ctx.rect(px, py, cellW, cellH);
      ctx.clip();
      ctx.translate(px + cellW / 2, py + cellH / 2);
      ctx.scale(BOX_SCALE, BOX_SCALE);
      ctx.fillText(text, boxDrawX - cellW / 2, boxDrawY - cellH / 2);
      ctx.restore();
    } else {
      ctx.font = cellFont(TEXT_SIZE, cell);
      ctx.fillText(text, px + textX, py + textBaseline);
    }

    if (cell.underline) {
      const underlineY = Math.min(py + cellH - 1, py + textBaseline + 1);
      ctx.fillRect(px, underlineY, cellW, 1);
    }
  }

  return nodeCells;
}
