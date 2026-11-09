interface AnywidgetModel {
  send(msg: Record<string, unknown>): void;
}

const DRAG_THRESHOLD = 4;

export interface HitRegion {
  x: number;
  y: number;
  kind: string;
}

export function attachInteraction(
  image: HTMLImageElement,
  model: AnywidgetModel,
  getGeometry: () => { cellWidth: number; cellHeight: number; hitRegions: HitRegion[] },
  isFrozen: () => boolean,
): { detach: () => void; resetPreview: () => void; finishPan: () => void; panBy: (dx: number, dy: number) => boolean } {
  let pointerDown = false;
  // True from the pan request until Python acknowledges it. Large graphs can take
  // many seconds to re-render, so the preview stays put (no timeout) and further
  // drags are ignored rather than queueing more requests behind the slow one.
  let panInFlight = false;
  let isDragging = false;
  let dragStartX = 0;
  let dragStartY = 0;

  const resetPreview = () => {
    image.style.transform = "";
  };

  function cellAt(clientX: number, clientY: number): { col: number; row: number } {
    const rect = image.getBoundingClientRect();
    const { cellWidth, cellHeight } = getGeometry();
    const scaleX = image.naturalWidth / rect.width;
    const scaleY = image.naturalHeight / rect.height;
    return {
      col: Math.floor(((clientX - rect.left) * scaleX) / cellWidth),
      row: Math.floor(((clientY - rect.top) * scaleY) / cellHeight),
    };
  }

  const onMouseMove = (e: MouseEvent) => {
    if (isFrozen()) { image.style.cursor = "default"; return; }
    if (pointerDown || panInFlight) return;
    const { col, row } = cellAt(e.clientX, e.clientY);
    image.style.cursor = getGeometry().hitRegions.some((region) => region.x === col && region.y === row)
      ? "pointer"
      : "grab";
  };

  image.addEventListener("mousemove", onMouseMove);

  const onMouseDown = (e: MouseEvent) => {
    if (isFrozen() || e.button !== 0 || panInFlight) return;
    pointerDown = true;
    isDragging = false;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    resetPreview();
    e.preventDefault();
  };
  image.addEventListener("mousedown", onMouseDown);

  const onWindowMouseMove = (e: MouseEvent) => {
    if (!pointerDown) return;
    if (isFrozen()) { pointerDown = false; resetPreview(); return; }
    const totalDx = e.clientX - dragStartX;
    const totalDy = e.clientY - dragStartY;
    if (!isDragging && (Math.abs(totalDx) > DRAG_THRESHOLD || Math.abs(totalDy) > DRAG_THRESHOLD)) {
      isDragging = true;
      image.style.cursor = "grabbing";
    }
    if (isDragging) {
      image.style.transform = `translate(${totalDx}px, ${totalDy}px)`;
    }
  };

  // Single entry point for drags and the arrow buttons so both share the
  // one-request-in-flight lock. Returns false when nothing was sent.
  const panBy = (dx: number, dy: number): boolean => {
    if (isFrozen() || panInFlight || (dx === 0 && dy === 0)) return false;
    const rect = image.getBoundingClientRect();
    const { cellWidth, cellHeight } = getGeometry();
    const cssCellWidth = cellWidth * rect.width / image.naturalWidth;
    const cssCellHeight = cellHeight * rect.height / image.naturalHeight;
    image.style.transform = `translate(${dx * cssCellWidth}px, ${dy * cssCellHeight}px)`;
    panInFlight = true;
    image.style.cursor = "progress";
    model.send({ type: "pan", dx, dy });
    return true;
  };

  const onWindowMouseUp = (e: MouseEvent) => {
    if (!pointerDown || e.button !== 0) return;
    if (isFrozen()) { pointerDown = false; resetPreview(); return; }
    if (!isDragging) {
      const { col, row } = cellAt(e.clientX, e.clientY);
      model.send({ type: "mouse_click", col, row, button: e.button });
    } else {
      const rect = image.getBoundingClientRect();
      const { cellWidth, cellHeight } = getGeometry();
      const cssCellWidth = cellWidth * rect.width / image.naturalWidth;
      const cssCellHeight = cellHeight * rect.height / image.naturalHeight;
      const termDx = Math.trunc((e.clientX - dragStartX) / cssCellWidth);
      const termDy = Math.trunc((e.clientY - dragStartY) / cssCellHeight);
      if (!panBy(termDx, termDy)) resetPreview();
    }
    pointerDown = false;
    isDragging = false;
    if (!panInFlight) image.style.cursor = "grab";
  };

  const finishPan = () => {
    panInFlight = false;
    image.style.cursor = "grab";
  };

  window.addEventListener("mousemove", onWindowMouseMove);
  window.addEventListener("mouseup", onWindowMouseUp);

  const detach = () => {
    resetPreview();
    image.removeEventListener("mousemove", onMouseMove);
    image.removeEventListener("mousedown", onMouseDown);
    window.removeEventListener("mousemove", onWindowMouseMove);
    window.removeEventListener("mouseup", onWindowMouseUp);
  };
  return { detach, resetPreview, finishPan, panBy };
}
