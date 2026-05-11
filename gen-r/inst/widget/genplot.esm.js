// Generated from js/index.ts by `npm run build-r` / `make r-widget`. Edit TypeScript sources, not this file.

// js/grid.ts
var CELL_SIZE = Math.round(16), BOX_SCALE = 1.15, FONT_FAMILY = "monospace", BOX_FONT = `${CELL_SIZE}px ${FONT_FAMILY}`, TEXT_FONT = `14px ${FONT_FAMILY}`;
function cellFont(size, cell) {
  let style = cell.bold && cell.italic ? "bold italic" : cell.bold ? "bold" : cell.italic ? "italic" : "";
  return style ? `${style} ${size}px ${FONT_FAMILY}` : `${size}px ${FONT_FAMILY}`;
}
function measure(ctx, font, text) {
  return ctx.font = font, ctx.textAlign = "left", ctx.textBaseline = "alphabetic", ctx.measureText(text);
}
function isBoxLike(ch) {
  if (!ch || ch === " ") return !1;
  let cp = ch.codePointAt(0);
  return cp >= 9472 && cp <= 9599 || // box drawing
  cp >= 9600 && cp <= 9631 || // block elements
  cp >= 10240 && cp <= 10495;
}
function makeGridMetrics(ctx) {
  let block = measure(ctx, BOX_FONT, "\u2588"), cellW = Math.ceil(block.actualBoundingBoxLeft + block.actualBoundingBoxRight), cellH = Math.ceil(block.actualBoundingBoxAscent + block.actualBoundingBoxDescent), boxDrawX = block.actualBoundingBoxLeft, boxDrawY = block.actualBoundingBoxAscent, textProbe = measure(ctx, TEXT_FONT, "Mg"), textMono = measure(ctx, TEXT_FONT, "M"), textAscent = textProbe.emHeightAscent ?? textProbe.fontBoundingBoxAscent ?? textProbe.actualBoundingBoxAscent, textDescent = textProbe.emHeightDescent ?? textProbe.fontBoundingBoxDescent ?? textProbe.actualBoundingBoxDescent, textX = Math.round((cellW - textMono.width) / 2), textBaseline = Math.round((cellH + textAscent - textDescent) / 2) + 1;
  return { cellW, cellH, boxDrawX, boxDrawY, textX, textBaseline };
}

// js/glyphs.ts
var GLYPH_SPLIT = /* @__PURE__ */ new Map([
  // Pure-heavy lines and half-lines: map to their light/rounded equivalents
  [9473, [9472, void 0]],
  [9475, [9474, void 0]],
  [9477, [9476, void 0]],
  [9479, [9478, void 0]],
  [9481, [9480, void 0]],
  [9483, [9482, void 0]],
  [9592, [9588, void 0]],
  [9593, [9589, void 0]],
  [9594, [9590, void 0]],
  [9595, [9591, void 0]],
  // Pure-heavy corners
  [9487, [9581, void 0]],
  [9491, [9582, void 0]],
  [9495, [9584, void 0]],
  [9499, [9583, void 0]],
  // Pure-heavy tees and cross
  [9507, [9500, void 0]],
  [9515, [9508, void 0]],
  [9523, [9516, void 0]],
  [9531, [9524, void 0]],
  [9547, [9532, void 0]],
  // Mixed-weight corners (one heavy arm, one light arm)
  [9485, [9590, 9591]],
  [9486, [9591, 9590]],
  [9489, [9588, 9591]],
  [9490, [9591, 9588]],
  [9493, [9590, 9589]],
  [9494, [9589, 9590]],
  [9497, [9588, 9589]],
  [9498, [9589, 9588]],
  // Mixed-weight tees (heavy stem, mixed crossbar)
  [9501, [9590, 9474]],
  [9502, [9589, 9484]],
  [9503, [9591, 9492]],
  [9504, [9474, 9590]],
  [9505, [9492, 9591]],
  [9506, [9484, 9589]],
  [9509, [9588, 9474]],
  [9510, [9589, 9488]],
  [9511, [9591, 9496]],
  [9512, [9474, 9588]],
  [9513, [9496, 9591]],
  [9514, [9488, 9589]],
  [9517, [9588, 9484]],
  [9518, [9590, 9488]],
  [9519, [9472, 9591]],
  [9520, [9591, 9472]],
  [9521, [9488, 9590]],
  [9522, [9484, 9588]],
  [9525, [9588, 9492]],
  [9526, [9590, 9496]],
  [9527, [9472, 9589]],
  [9528, [9589, 9472]],
  [9529, [9496, 9590]],
  [9530, [9492, 9588]],
  // Mixed-weight crosses
  [9533, [9588, 9500]],
  [9534, [9590, 9508]],
  [9535, [9472, 9474]],
  [9536, [9589, 9516]],
  [9537, [9591, 9524]],
  [9538, [9474, 9472]],
  [9539, [9496, 9484]],
  [9540, [9492, 9488]],
  [9541, [9488, 9492]],
  [9542, [9484, 9496]],
  [9543, [9524, 9591]],
  [9544, [9516, 9589]],
  [9545, [9508, 9590]],
  [9546, [9500, 9588]]
]), NODE_GLYPH_CP = 9679;

// js/renderer.ts
function drawSplitGlyph(ctx, heavyCp, lightCp, px, py, cellW, cellH, boxDrawX, boxDrawY, fg, neutral_fg, boxFont) {
  ctx.save(), ctx.beginPath(), ctx.rect(px, py, cellW, cellH), ctx.clip(), lightCp !== void 0 && (ctx.fillStyle = neutral_fg, ctx.font = boxFont, ctx.save(), ctx.translate(px + cellW / 2, py + cellH / 2), ctx.scale(BOX_SCALE, BOX_SCALE), ctx.fillText(String.fromCodePoint(lightCp), boxDrawX - cellW / 2, boxDrawY - cellH / 2), ctx.restore()), ctx.fillStyle = fg, ctx.font = boxFont, ctx.save(), ctx.translate(px + cellW / 2, py + cellH / 2), ctx.scale(BOX_SCALE, BOX_SCALE), ctx.fillText(String.fromCodePoint(heavyCp), boxDrawX - cellW / 2, boxDrawY - cellH / 2), ctx.restore(), ctx.restore();
}
function paintFrame(ctx, canvas, grid, frame) {
  let nodeCells = /* @__PURE__ */ new Set();
  if (!frame?.cells) return nodeCells;
  let { cellW, cellH, boxDrawX, boxDrawY, textX, textBaseline } = grid, { cols, rows, cells, neutral_fg = "#cdd6f4", neutral_bg = "#1e1e2e" } = frame;
  canvas.width = cols * cellW, canvas.height = rows * cellH, ctx.fillStyle = neutral_bg, ctx.fillRect(0, 0, canvas.width, canvas.height), ctx.textAlign = "left", ctx.textBaseline = "alphabetic";
  for (let cell of cells) {
    let { x: c, y: r, text } = cell, fg = cell.fg ?? neutral_fg, bg = cell.bg ?? neutral_bg, px = c * cellW, py = r * cellH;
    if (!text || text === " ") continue;
    let cp = text.codePointAt(0);
    if (isBoxLike(text) || nodeCells.add(`${c},${r}`), cp === NODE_GLYPH_CP) {
      bg !== neutral_bg && (ctx.fillStyle = bg, ctx.fillRect(px, py, cellW, cellH)), ctx.fillStyle = fg, ctx.beginPath(), ctx.arc(px + cellW / 2, py + cellH / 2, cellW / 2, 0, Math.PI * 2), ctx.fill();
      continue;
    }
    bg !== neutral_bg && (ctx.fillStyle = bg, ctx.fillRect(px, py, cellW, cellH));
    let split = GLYPH_SPLIT.get(cp);
    if (split !== void 0) {
      let [heavyCp, lightCp] = split, boxFont = cellFont(CELL_SIZE, { bold: !1, italic: !!cell.italic });
      drawSplitGlyph(ctx, heavyCp, lightCp, px, py, cellW, cellH, boxDrawX, boxDrawY, fg, neutral_fg, boxFont);
    } else isBoxLike(text) ? (ctx.fillStyle = fg, ctx.font = cellFont(CELL_SIZE, cell), ctx.save(), ctx.beginPath(), ctx.rect(px, py, cellW, cellH), ctx.clip(), ctx.translate(px + cellW / 2, py + cellH / 2), ctx.scale(BOX_SCALE, BOX_SCALE), ctx.fillText(text, boxDrawX - cellW / 2, boxDrawY - cellH / 2), ctx.restore()) : (ctx.fillStyle = fg, ctx.font = cellFont(14, cell), ctx.fillText(text, px + textX, py + textBaseline));
    if (cell.underline) {
      ctx.fillStyle = fg;
      let underlineY = Math.min(py + cellH - 1, py + textBaseline + 1);
      ctx.fillRect(px, underlineY, cellW, 1);
    }
  }
  return nodeCells;
}

// js/index.ts
function render({ model, el }) {
  let scratchCtx = document.createElement("canvas").getContext("2d"), grid = makeGridMetrics(scratchCtx), wrapper = document.createElement("div");
  wrapper.style.cssText = "position: relative; display: inline-block; line-height: 0;";
  let canvas = document.createElement("canvas");
  canvas.style.cssText = "display: block; cursor: default; border: 2px solid #45475a;", wrapper.appendChild(canvas), el.appendChild(wrapper);
  let ctx = canvas.getContext("2d"), nodeCells = /* @__PURE__ */ new Set(), frozen = !1;
  function repaint(frame) {
    nodeCells = paintFrame(ctx, canvas, grid, frame);
  }
  repaint(model.get("frame")), model.on("change:frame", () => repaint(model.get("frame")));
}
var index_default = { render };
export {
  index_default as default
};
