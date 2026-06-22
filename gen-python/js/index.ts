import { makeGridMetrics } from "./grid";
import { paintFrame, Frame } from "./renderer";
import { attachInteraction } from "./interaction";

declare const INTERACTIVE: boolean;

interface AnywidgetModel {
  get(key: "frame"): Frame;
  get(key: "page_count"): number;
  get(key: "page_index"): number;
  on(event: "change:frame", callback: () => void): void;
  on(event: "change:page_index", callback: () => void): void;
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
      "box-sizing: border-box",
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

    const pageArrowStyle = [
      "box-sizing: border-box",
      "width: 20px",
      "height: 24px",
      "font-size: 16px",
      "line-height: 1",
      "display: flex",
      "align-items: center",
      "justify-content: center",
      "background: transparent",
      "color: inherit",
      "border: none",
      "cursor: pointer",
      "padding: 0",
    ].join("; ");

    function makeBtn(svg: string, title: string, onClick: () => void): HTMLButtonElement {
      const btn = document.createElement("button");
      btn.innerHTML = svg;
      btn.setAttribute("style", sharedBtnStyle);
      btn.title = title;
      btn.addEventListener("mousedown", (e) => e.preventDefault());
      btn.addEventListener("click", onClick);
      return btn;
    }

    const pageable = (model.get("page_count") ?? 1) > 1;

    // Floating "<index/count>" pager indicator: same height/style as the
    // other floating buttons, with a clickable arrow on each side.
    const pageIndicator = document.createElement("div");
    pageIndicator.style.cssText = [
      "box-sizing: border-box",
      "height: 24px",
      "display: flex",
      "align-items: center",
      "background: rgba(30,30,46,0.85)",
      "color: #cdd6f4",
      "border: 1px solid #45475a",
      "border-radius: 4px",
      "overflow: hidden",
      "user-select: none",
    ].join("; ");
    pageIndicator.style.cssText += "; position: absolute; top: 8px; left: 8px; z-index: 1;";
    pageIndicator.style.display = pageable ? "flex" : "none";

    function makePageArrow(svg: string, title: string, direction: "prev" | "next"): HTMLButtonElement {
      const arrow = document.createElement("button");
      arrow.innerHTML = svg;
      arrow.title = title;
      arrow.setAttribute("style", pageArrowStyle);
      arrow.addEventListener("mousedown", (e) => e.preventDefault());
      arrow.addEventListener("click", () => model.send({ type: "page", direction }));
      return arrow;
    }

    const pageLeftBtn = makePageArrow(
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <polyline points="15 18 9 12 15 6" />
</svg>`,
      "Previous sequence graph",
      "prev",
    );

    const pageLabel = document.createElement("span");
    pageLabel.style.cssText = "font-size: 12px; line-height: 24px; padding: 0 2px; white-space: nowrap;";

    const pageRightBtn = makePageArrow(
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <polyline points="9 18 15 12 9 6" />
</svg>`,
      "Next sequence graph",
      "next",
    );

    function updatePageLabel(): void {
      const count = model.get("page_count") ?? 1;
      const index = model.get("page_index") ?? 0;
      pageLabel.textContent = `${index + 1}/${count}`;
    }
    updatePageLabel();
    model.on("change:page_index", updatePageLabel);

    pageIndicator.appendChild(pageLeftBtn);
    pageIndicator.appendChild(pageLabel);
    pageIndicator.appendChild(pageRightBtn);

    const zoomInBtn = makeBtn(
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
  <circle cx="10" cy="10" r="7" stroke-width="1.2" />
  <line x1="16" y1="16" x2="21" y2="21" stroke-width="2.5" />
  <line x1="10" y1="6" x2="10" y2="14" />
  <line x1="6" y1="10" x2="14" y2="10" />
</svg>`,
      "Zoom in (+)",
      () => model.send({ type: "zoom", direction: "in" }),
    );

    const zoomOutBtn = makeBtn(
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
  <circle cx="10" cy="10" r="7" stroke-width="1.2" />
  <line x1="16" y1="16" x2="21" y2="21" stroke-width="2.5" />
  <line x1="6" y1="10" x2="14" y2="10" />
</svg>`,
      "Zoom out (-)",
      () => model.send({ type: "zoom", direction: "out" }),
    );

    const freezeBtn = makeBtn(
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
  <rect x="5" y="10" width="14" height="11" rx="2" />
  <path d="M8 10V7a4 4 0 0 1 8 0v3" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" />
</svg>`,
      "Freeze as static image",
      () => doFreeze(),
    );

    wrapper.appendChild(pageIndicator);

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
      pageIndicator.style.display = "none";
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
