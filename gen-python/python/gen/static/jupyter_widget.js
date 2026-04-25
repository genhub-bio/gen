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

// js/renderer.ts
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

// js/interaction.ts
var DRAG_THRESHOLD = 4;
function attachInteraction(canvas, model, grid, getNodeCells, isFrozen) {
  const { cellW, cellH } = grid;
  canvas.addEventListener("mousemove", (e) => {
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
    const dataUrl = canvas.toDataURL("image/png");
    const img = document.createElement("img");
    img.src = dataUrl;
    img.style.cssText = "display:block;font-family:monospace";
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
