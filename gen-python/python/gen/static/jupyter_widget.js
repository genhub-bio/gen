// Generated from js/index.ts by `npm run build` / `make jupyter`. Edit TypeScript sources, not this file.

// js/grid.ts
var TEXT_SIZE = 14;
var CELL_SIZE = Math.round(TEXT_SIZE / 0.875);
var BOX_SCALE = 1.15;
var FONT_FAMILY = "monospace";
var BOX_FONT = `${CELL_SIZE}px ${FONT_FAMILY}`;
var TEXT_FONT = `${TEXT_SIZE}px ${FONT_FAMILY}`;
function cellFont(size, cell) {
  const style = cell.bold && cell.italic ? "bold italic" : cell.bold ? "bold" : cell.italic ? "italic" : "";
  return style ? `${style} ${size}px ${FONT_FAMILY}` : `${size}px ${FONT_FAMILY}`;
}
function measure(ctx, font, text) {
  ctx.font = font;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  return ctx.measureText(text);
}
function isBoxLike(ch) {
  if (!ch || ch === " ") return false;
  const cp = ch.codePointAt(0);
  return cp >= 9472 && cp <= 9599 || // box drawing
  cp >= 9600 && cp <= 9631 || // block elements
  cp >= 10240 && cp <= 10495;
}
function makeGridMetrics(ctx) {
  const block = measure(ctx, BOX_FONT, "\u2588");
  const cellW = Math.ceil(block.actualBoundingBoxLeft + block.actualBoundingBoxRight);
  const cellH = Math.ceil(block.actualBoundingBoxAscent + block.actualBoundingBoxDescent);
  const boxDrawX = block.actualBoundingBoxLeft;
  const boxDrawY = block.actualBoundingBoxAscent;
  const textProbe = measure(ctx, TEXT_FONT, "Mg");
  const textMono = measure(ctx, TEXT_FONT, "M");
  const textAscent = textProbe.emHeightAscent ?? textProbe.fontBoundingBoxAscent ?? textProbe.actualBoundingBoxAscent;
  const textDescent = textProbe.emHeightDescent ?? textProbe.fontBoundingBoxDescent ?? textProbe.actualBoundingBoxDescent;
  const textX = Math.round((cellW - textMono.width) / 2);
  const textBaseline = Math.round((cellH + textAscent - textDescent) / 2) + 1;
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
]);
var NODE_GLYPH_CP = 9679;

// js/renderer.ts
function drawSplitGlyph(ctx, heavyCp, lightCp, px, py, cellW, cellH, boxDrawX, boxDrawY, fg, neutral_fg, boxFont) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(px, py, cellW, cellH);
  ctx.clip();
  if (lightCp !== void 0) {
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
function paintFrame(ctx, canvas, grid, frame) {
  const nodeCells = /* @__PURE__ */ new Set();
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
    const cp = text.codePointAt(0);
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
    if (split !== void 0) {
      const [heavyCp, lightCp] = split;
      const boxFont = cellFont(CELL_SIZE, { bold: false, italic: !!cell.italic });
      drawSplitGlyph(ctx, heavyCp, lightCp, px, py, cellW, cellH, boxDrawX, boxDrawY, fg, neutral_fg, boxFont);
    } else if (isBoxLike(text)) {
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

// js/interaction.ts
var DRAG_THRESHOLD = 4;
function attachInteraction(canvas, model, grid, getNodeCells, isFrozen) {
  const { cellW, cellH } = grid;
  canvas.addEventListener("mousemove", (e) => {
    if (isFrozen()) return;
    if (pointerDown) return;
    const rect = canvas.getBoundingClientRect();
    const col = Math.floor((e.clientX - rect.left) / cellW);
    const row = Math.floor((e.clientY - rect.top) / cellH);
    canvas.style.cursor = getNodeCells().has(`${col},${row}`) ? "pointer" : "grab";
  });
  let pointerDown = false;
  let isDragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let lastDragX = 0;
  let lastDragY = 0;
  let dragRemX = 0;
  let dragRemY = 0;
  canvas.addEventListener("mousedown", (e) => {
    if (isFrozen() || e.button !== 0) return;
    pointerDown = true;
    isDragging = false;
    dragStartX = lastDragX = e.clientX;
    dragStartY = lastDragY = e.clientY;
    dragRemX = 0;
    dragRemY = 0;
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!pointerDown) return;
    const totalDx = e.clientX - dragStartX;
    const totalDy = e.clientY - dragStartY;
    if (!isDragging && (Math.abs(totalDx) > DRAG_THRESHOLD || Math.abs(totalDy) > DRAG_THRESHOLD)) {
      isDragging = true;
      canvas.style.cursor = "grabbing";
    }
    if (isDragging) {
      dragRemX += e.clientX - lastDragX;
      dragRemY += e.clientY - lastDragY;
      const termDx = Math.trunc(dragRemX / cellW);
      const termDy = Math.trunc(dragRemY / cellH);
      if (termDx !== 0 || termDy !== 0) {
        model.send({ type: "pan", dx: termDx, dy: termDy });
        dragRemX -= termDx * cellW;
        dragRemY -= termDy * cellH;
      }
    }
    lastDragX = e.clientX;
    lastDragY = e.clientY;
  });
  window.addEventListener("mouseup", (e) => {
    if (!pointerDown || e.button !== 0) return;
    if (!isDragging) {
      const rect = canvas.getBoundingClientRect();
      const col = Math.floor((e.clientX - rect.left) / cellW);
      const row = Math.floor((e.clientY - rect.top) / cellH);
      model.send({ type: "mouse_click", col, row, button: e.button });
    }
    pointerDown = false;
    isDragging = false;
    canvas.style.cursor = "grab";
  });
}

// js/index.ts
function render({ model, el }) {
  const scratch = document.createElement("canvas");
  const scratchCtx = scratch.getContext("2d");
  const grid = makeGridMetrics(scratchCtx);
  const wrapper = document.createElement("div");
  wrapper.style.cssText = "position: relative; display: inline-block; line-height: 0;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display: block; cursor: grab; border: 2px solid #45475a;";
  const sharedBtnStyle = [
    "width: 24px",
    "height: 24px",
    "font-size: 16px",
    "line-height: 1",
    "cursor: pointer",
    "background: rgba(30,30,46,0.85)",
    "color: #cdd6f4",
    "border: 1px solid #45475a",
    "border-radius: 4px",
    "display: flex",
    "align-items: center",
    "justify-content: center",
    "user-select: none",
    "padding: 0"
  ].join("; ");
  const btnContainer = document.createElement("div");
  btnContainer.style.cssText = "position: absolute; bottom: 8px; right: 8px; display: flex; flex-direction: column; gap: 4px; z-index: 1;";
  const zoomInBtn = document.createElement("button");
  zoomInBtn.textContent = "+";
  zoomInBtn.setAttribute("style", sharedBtnStyle);
  zoomInBtn.title = "Zoom in (+)";
  const zoomOutBtn = document.createElement("button");
  zoomOutBtn.textContent = "\u2212";
  zoomOutBtn.setAttribute("style", sharedBtnStyle);
  zoomOutBtn.title = "Zoom out (-)";
  btnContainer.appendChild(zoomInBtn);
  btnContainer.appendChild(zoomOutBtn);
  wrapper.appendChild(canvas);
  wrapper.appendChild(btnContainer);
  el.appendChild(wrapper);
  const ctx = canvas.getContext("2d");
  let nodeCells = /* @__PURE__ */ new Set();
  let frozen = false;
  function repaint(frame) {
    nodeCells = paintFrame(ctx, canvas, grid, frame);
  }
  repaint(model.get("frame"));
  model.on("change:frame", () => repaint(model.get("frame")));
  model.on("msg:custom", (msg) => {
    if (msg.type !== "freeze") return;
    frozen = true;
    btnContainer.style.display = "none";
    canvas.style.cursor = "default";
    wrapper.style.border = "none";
    const dataUrl = canvas.toDataURL("image/png");
    const img = document.createElement("img");
    img.src = dataUrl;
    img.width = canvas.width;
    img.height = canvas.height;
    img.style.cssText = "display:block;cursor:default;font-family:monospace";
    el.replaceChild(img, wrapper);
    model.send({ type: "snapshot", data: dataUrl });
  });
  zoomInBtn.addEventListener("mousedown", (e) => e.preventDefault());
  zoomOutBtn.addEventListener("mousedown", (e) => e.preventDefault());
  zoomInBtn.addEventListener("click", () => model.send({ type: "zoom", direction: "in" }));
  zoomOutBtn.addEventListener("click", () => model.send({ type: "zoom", direction: "out" }));
  attachInteraction(canvas, model, grid, () => nodeCells, () => frozen);
}
var index_default = { render };
export {
  index_default as default
};
//# sourceMappingURL=jupyter_widget.js.map
