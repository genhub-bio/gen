/**
 * Interaction model (Jupyter-safe):
 *   - Drag to pan the graph.
 *   - +/- overlay buttons in the bottom-right corner for zoom.
 *   - Click on a node to select it.
 *   - Wheel events are NOT captured — they scroll the notebook normally.
 *
 * Data shapes (from Rust via Python):
 *   RenderedFrame  { cols: number, rows: number, cells: RenderedCell[] }
 *   RenderedCell   { text: string, fg: string, bg: string, bold: boolean, italic: boolean, underline: boolean }
 */

// Monospace font used for cell measurement and drawing.
const FONT_SIZE = 14;
const FONT_FAMILY = "monospace";
const FONT = `${FONT_SIZE}px ${FONT_FAMILY}`;

function fontForCell(cell) {
  if (cell.bold && cell.italic) return `bold italic ${FONT_SIZE}px ${FONT_FAMILY}`;
  if (cell.bold) return `bold ${FONT_SIZE}px ${FONT_FAMILY}`;
  if (cell.italic) return `italic ${FONT_SIZE}px ${FONT_FAMILY}`;
  return FONT;
}

function render({ model, el }) {
  // Measure a tight cell directly from a light box-drawing cross.
  const scratch = document.createElement("canvas");
  const scratchCtx = scratch.getContext("2d");
  scratchCtx.font = FONT;
  scratchCtx.textBaseline = "alphabetic";

  const metrics = scratchCtx.measureText("┼");
  const cellW = Math.ceil(metrics.width);
  const ascent = Math.ceil(metrics.actualBoundingBoxAscent || FONT_SIZE);
  const descent = Math.ceil(metrics.actualBoundingBoxDescent || 0);
  const cellH = ascent + descent;
  const baseline = ascent;

  // ── DOM setup ──────────────────────────────────────────────────────────────
  const wrapper = document.createElement("div");
  wrapper.style.cssText = "position: relative; display: inline-block; line-height: 0; border: 2px solid #45475a;";

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display: block; cursor: grab;";

  // ── Zoom button overlay (bottom-right corner) ──────────────────────────────
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
    "padding: 0",
  ].join("; ");

  const btnContainer = document.createElement("div");
  btnContainer.style.cssText =
    "position: absolute; bottom: 8px; right: 8px; display: flex; flex-direction: column; gap: 4px; z-index: 1;";

  const zoomInBtn = document.createElement("button");
  zoomInBtn.textContent = "+";
  zoomInBtn.setAttribute("style", sharedBtnStyle);
  zoomInBtn.title = "Zoom in (+)";

  const zoomOutBtn = document.createElement("button");
  zoomOutBtn.textContent = "−";
  zoomOutBtn.setAttribute("style", sharedBtnStyle);
  zoomOutBtn.title = "Zoom out (-)";

  btnContainer.appendChild(zoomInBtn);
  btnContainer.appendChild(zoomOutBtn);

  wrapper.appendChild(canvas);
  wrapper.appendChild(btnContainer);
  el.appendChild(wrapper);

  const ctx = canvas.getContext("2d");
  ctx.textBaseline = "alphabetic";

  function paintFrame(frame) {
    if (!frame || !frame.cells) return;

    const { cols, rows, cells } = frame;
    canvas.width = cols * cellW;
    canvas.height = rows * cellH;

    // Canvas resize resets drawing state.
    ctx.textBaseline = "alphabetic";

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cell = cells[r * cols + c];
        const x = c * cellW;
        const y = r * cellH;

        ctx.fillStyle = cell.bg || "#000000";
        ctx.fillRect(x, y, cellW, cellH);

        const text = cell.text;
        if (text && text !== " " && text !== "") {
          ctx.font = fontForCell(cell);
          ctx.fillStyle = cell.fg || "#ffffff";
          ctx.fillText(text, x, y + baseline);
        }

        if (cell.underline) {
          ctx.fillStyle = cell.fg || "#ffffff";
          const underlineY = Math.min(y + cellH - 1, y + baseline + 1);
          ctx.fillRect(x, underlineY, cellW, 1);
        }
      }
    }
  }

  paintFrame(model.get("frame"));
  model.on("change:frame", () => paintFrame(model.get("frame")));

  // ── Freeze ─────────────────────────────────────────────────────────────────
  let frozen = false;

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

  // ── Zoom buttons ───────────────────────────────────────────────────────────
  zoomInBtn.addEventListener("mousedown", (e) => e.preventDefault());
  zoomOutBtn.addEventListener("mousedown", (e) => e.preventDefault());

  zoomInBtn.addEventListener("click", () => {
    model.send({ type: "zoom", direction: "in" });
  });

  zoomOutBtn.addEventListener("click", () => {
    model.send({ type: "zoom", direction: "out" });
  });

  // ── Drag-to-pan ────────────────────────────────────────────────────────────
  const DRAG_THRESHOLD = 4;
  let pointerDown = false;
  let isDragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let lastDragX = 0;
  let lastDragY = 0;
  let dragRemX = 0;
  let dragRemY = 0;

  canvas.addEventListener("mousedown", (e) => {
    if (frozen || e.button !== 0) return;
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

export default { render };
