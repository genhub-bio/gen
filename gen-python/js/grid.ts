// Data shapes (from Rust via Python):
//   RenderedFrame  { cols, rows, neutral_fg, neutral_bg, cells: RenderedCell[] }
//   RenderedCell   { x, y, text, fg?, bg?, bold?, italic?, underline? }
//
// Sparse protocol: only non-blank or non-neutral cells are sent.
// fg/bg are omitted when equal to neutral_fg/neutral_bg; bool flags omitted when false.

export const TEXT_SIZE = 14;
export const CELL_SIZE = Math.round(TEXT_SIZE / 0.875);
export const BOX_SCALE = 1.15;
export const FONT_FAMILY = "monospace";
export const BOX_FONT = `${CELL_SIZE}px ${FONT_FAMILY}`;
export const TEXT_FONT = `${TEXT_SIZE}px ${FONT_FAMILY}`;

export interface Cell {
  x: number;
  y: number;
  text: string;
  fg?: string;
  bg?: string;
  bold?: boolean;
  italic?: boolean;
  underline?: boolean;
}

export interface GridMetrics {
  cellW: number;
  cellH: number;
  boxDrawX: number;
  boxDrawY: number;
  textX: number;
  textBaseline: number;
}

export function cellFont(size: number, cell: Pick<Cell, "bold" | "italic">): string {
  const style = cell.bold && cell.italic ? "bold italic" : cell.bold ? "bold" : cell.italic ? "italic" : "";
  return style ? `${style} ${size}px ${FONT_FAMILY}` : `${size}px ${FONT_FAMILY}`;
}

function measure(ctx: CanvasRenderingContext2D, font: string, text: string): TextMetrics {
  ctx.font = font;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  return ctx.measureText(text);
}

export function isBoxLike(ch: string): boolean {
  if (!ch || ch === " ") return false;
  const cp = ch.codePointAt(0)!;
  return (
    (cp >= 0x2500 && cp <= 0x257F) || // box drawing
    (cp >= 0x2580 && cp <= 0x259F) || // block elements
    (cp >= 0x2800 && cp <= 0x28FF)    // braille
  );
}

export function makeGridMetrics(ctx: CanvasRenderingContext2D): GridMetrics {
  const block = measure(ctx, BOX_FONT, "\u2588");
  const cellW = Math.ceil(block.actualBoundingBoxLeft + block.actualBoundingBoxRight);
  const cellH = Math.ceil(block.actualBoundingBoxAscent + block.actualBoundingBoxDescent);
  const boxDrawX = block.actualBoundingBoxLeft;
  const boxDrawY = block.actualBoundingBoxAscent;
  const textProbe = measure(ctx, TEXT_FONT, "Mg");
  const textMono = measure(ctx, TEXT_FONT, "M");
  const textAscent =
    textProbe.emHeightAscent ?? textProbe.fontBoundingBoxAscent ?? textProbe.actualBoundingBoxAscent;
  const textDescent =
    textProbe.emHeightDescent ?? textProbe.fontBoundingBoxDescent ?? textProbe.actualBoundingBoxDescent;
  const textX = Math.round((cellW - textMono.width) / 2);
  const textBaseline = Math.round((cellH + textAscent - textDescent) / 2) + 1;
  return { cellW, cellH, boxDrawX, boxDrawY, textX, textBaseline };
}
