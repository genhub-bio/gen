import { GridMetrics } from "./grid";

interface AnywidgetModel {
  send(msg: Record<string, unknown>): void;
}

const DRAG_THRESHOLD = 4;

export function attachInteraction(
  canvas: HTMLCanvasElement,
  model: AnywidgetModel,
  grid: GridMetrics,
  getNodeCells: () => Set<string>,
  isFrozen: () => boolean,
): () => void {
  const { cellW, cellH } = grid;

  canvas.addEventListener("mousemove", (e) => {
    if (isFrozen()) { canvas.style.cursor = "default"; return; }
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

  const onWindowMouseMove = (e: MouseEvent) => {
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
  };

  const onWindowMouseUp = (e: MouseEvent) => {
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
  };

  window.addEventListener("mousemove", onWindowMouseMove);
  window.addEventListener("mouseup", onWindowMouseUp);

  return () => {
    window.removeEventListener("mousemove", onWindowMouseMove);
    window.removeEventListener("mouseup", onWindowMouseUp);
  };
}
