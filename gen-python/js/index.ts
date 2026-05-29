import { makeGridMetrics } from "./grid";
import { paintFrame, Frame } from "./renderer";
import { attachInteraction } from "./interaction";

declare const INTERACTIVE: boolean;

interface AnywidgetModel {
  get(key: "frame"): Frame;
  on(event: "change:frame", callback: () => void): void;
  on(event: "msg:custom", callback: (msg: { type: string; data?: string }) => void): void;
  send(msg: Record<string, unknown>): void;
}

interface RenderContext {
  model: AnywidgetModel;
  el: HTMLElement;
}

function render({ model, el }: RenderContext): { destroy(): void } | void {
  const scratch = document.createElement("canvas");
  const scratchCtx = scratch.getContext("2d")!;
  const grid = makeGridMetrics(scratchCtx);

  el.style.overflowX = "auto";

  const wrapper = document.createElement("div");
  wrapper.style.cssText = "position: relative; display: inline-block; line-height: 0;";

  const canvas = document.createElement("canvas");
  canvas.style.cssText = `display: block; cursor: ${INTERACTIVE ? "grab" : "default"}; box-shadow: inset 0 0 0 2px #45475a;`;

  wrapper.appendChild(canvas);
  el.appendChild(wrapper);

  const ctx = canvas.getContext("2d")!;

  paintFrame(ctx, canvas, grid, model.get("frame"));

  if (INTERACTIVE) {
    let nodeCells = new Set<string>();
    let frozen = false;
    let snapshotTimer: ReturnType<typeof setTimeout> | null = null;

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
      "position: absolute; top: 8px; right: 8px; display: flex; flex-direction: row; gap: 4px; z-index: 1;";

    const zoomInBtn = document.createElement("button");
    zoomInBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
  <circle cx="10" cy="10" r="7" stroke-width="1.2" />
  <line x1="16" y1="16" x2="21" y2="21" stroke-width="2.5" />
  <line x1="10" y1="6" x2="10" y2="14" />
  <line x1="6" y1="10" x2="14" y2="10" />
</svg>`;
    zoomInBtn.setAttribute("style", sharedBtnStyle);
    zoomInBtn.title = "Zoom in (+)";

    const zoomOutBtn = document.createElement("button");
    zoomOutBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
  <circle cx="10" cy="10" r="7" stroke-width="1.2" />
  <line x1="16" y1="16" x2="21" y2="21" stroke-width="2.5" />
  <line x1="6" y1="10" x2="14" y2="10" />
</svg>`;
    zoomOutBtn.setAttribute("style", sharedBtnStyle);
    zoomOutBtn.title = "Zoom out (-)";

    const freezeBtn = document.createElement("button");
    freezeBtn.innerHTML = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
  <rect x="5" y="10" width="14" height="11" rx="2" />
  <path d="M8 10V7a4 4 0 0 1 8 0v3" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" />
</svg>
`;
    freezeBtn.setAttribute("style", sharedBtnStyle);
    freezeBtn.title = "Freeze as static image";

    btnContainer.appendChild(zoomOutBtn);
    btnContainer.appendChild(zoomInBtn);
    btnContainer.appendChild(freezeBtn);
    wrapper.appendChild(btnContainer);

    function captureHiRes(): { dataUrl: string; width: number; height: number } {
      const dpr = window.devicePixelRatio || 1;
      const hiResGrid = makeGridMetrics(scratchCtx, dpr);
      const offscreen = document.createElement("canvas");
      const offCtx = offscreen.getContext("2d")!;
      paintFrame(offCtx, offscreen, hiResGrid, model.get("frame"));
      return { dataUrl: offscreen.toDataURL("image/png"), width: canvas.width, height: canvas.height };
    }

    function sendSnapshot(): void {
      const { dataUrl, width, height } = captureHiRes();
      model.send({ type: "snapshot", data: dataUrl, width, height });
    }

    function scheduleSnapshot(): void {
      if (snapshotTimer) clearTimeout(snapshotTimer);
      snapshotTimer = setTimeout(sendSnapshot, 1000);
    }

    function repaint(frame: Frame): void {
      nodeCells = paintFrame(ctx, canvas, grid, frame);
    }

    repaint(model.get("frame"));
    model.on("change:frame", () => { repaint(model.get("frame")); scheduleSnapshot(); });

    function doFreeze(): void {
      if (snapshotTimer) { clearTimeout(snapshotTimer); snapshotTimer = null; }
      frozen = true;
      btnContainer.style.display = "none";
      canvas.style.cursor = "default";
      canvas.style.boxShadow = "none";
      const { dataUrl, width, height } = captureHiRes();
      const img = document.createElement("img");
      img.src = dataUrl;
      img.style.cssText = `display:block;cursor:default;width:${width}px;height:${height}px`;
      wrapper.replaceChild(img, canvas);
      model.send({ type: "freeze", data: dataUrl, width, height });
    }

    model.on("msg:custom", (msg) => {
      if (msg.type === "freeze") doFreeze();
    });

    zoomInBtn.addEventListener("mousedown", (e) => e.preventDefault());
    zoomOutBtn.addEventListener("mousedown", (e) => e.preventDefault());
    freezeBtn.addEventListener("mousedown", (e) => e.preventDefault());
    zoomInBtn.addEventListener("click", () => model.send({ type: "zoom", direction: "in" }));
    zoomOutBtn.addEventListener("click", () => model.send({ type: "zoom", direction: "out" }));
    freezeBtn.addEventListener("click", () => doFreeze());

    const detachInteraction = attachInteraction(canvas, model, grid, () => nodeCells, () => frozen);

    return {
      destroy() {
        if (snapshotTimer) clearTimeout(snapshotTimer);
        detachInteraction();
      },
    };
  }
}

export default { render };
