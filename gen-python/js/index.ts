import { makeGridMetrics } from "./grid";
import { paintFrame, Frame } from "./renderer";
import { attachInteraction } from "./interaction";

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

function render({ model, el }: RenderContext): void {
  const scratch = document.createElement("canvas");
  const scratchCtx = scratch.getContext("2d")!;
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
  zoomOutBtn.textContent = "\u2212";
  zoomOutBtn.setAttribute("style", sharedBtnStyle);
  zoomOutBtn.title = "Zoom out (-)";

  btnContainer.appendChild(zoomInBtn);
  btnContainer.appendChild(zoomOutBtn);
  wrapper.appendChild(canvas);
  wrapper.appendChild(btnContainer);
  el.appendChild(wrapper);

  const ctx = canvas.getContext("2d")!;
  let nodeCells = new Set<string>();
  let frozen = false;

  function repaint(frame: Frame): void {
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

export default { render };
