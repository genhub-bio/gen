import { attachInteraction, HitRegion } from "./interaction";

// Jupyter sends the PNG as bytes. A host that cannot carry binary values, such as R Shiny, sends
// it as a base64 string, with or without the `data:image/png;base64,` prefix.
export type PngSource = ArrayBuffer | Uint8Array | DataView | number[] | string;

export interface AnywidgetModel {
  get(key: "viewport_png"): PngSource;
  get(key: "cell_width"): number;
  get(key: "cell_height"): number;
  get(key: "scale"): number;
  get(key: "background_color"): string;
  get(key: "hit_regions"): HitRegion[];
  get(key: "page_count"): number;
  get(key: "page_index"): number;
  get(key: "cols"): number;
  on(event: "change:viewport_png", callback: () => void): void;
  on(event: "change:page_index", callback: () => void): void;
  on(event: "change:hit_regions", callback: () => void): void;
  on(event: "change:background_color", callback: () => void): void;
  on(event: "msg:custom", callback: (msg: { type: string }) => void): void;
  off(event: string, callback: (...args: never[]) => void): void;
  send(msg: Record<string, unknown>): void;
}

export interface RenderContext {
  model: AnywidgetModel;
  el: HTMLElement;
}

function render({ model, el }: RenderContext): { destroy(): void } {
  el.style.overflowX = "auto";

  const wrapper = document.createElement("div");
  wrapper.style.cssText = "position: relative; display: inline-block; line-height: 0;";

  wrapper.style.overflow = "hidden";
  const updateBackground = () => { wrapper.style.backgroundColor = model.get("background_color"); };
  updateBackground();
  model.on("change:background_color", updateBackground);

  const image = document.createElement("img");
  image.style.cssText = "display: block; cursor: grab;";
  image.draggable = false;
  wrapper.appendChild(image);
  el.appendChild(wrapper);

  let objectUrl: string | null = null;
  let pendingObjectUrl: string | null = null;
  let imageRevision = 0;
  let resetPanPreview: (() => void) | null = null;
  function pngBytes(source: PngSource): Uint8Array {
    if (typeof source === "string") {
      const binary = atob(source.slice(source.indexOf(",") + 1));
      return Uint8Array.from(binary, (character) => character.charCodeAt(0));
    }
    return source instanceof ArrayBuffer
      ? new Uint8Array(source)
      : source instanceof DataView
        ? new Uint8Array(source.buffer, source.byteOffset, source.byteLength)
        : source instanceof Uint8Array
          ? source
          : new Uint8Array(source);
  }

  function updateImage(): void {
    const data = pngBytes(model.get("viewport_png"));
    const png = new Uint8Array(data.byteLength);
    png.set(data);
    if (png.byteLength === 0) return;
    const revision = ++imageRevision;
    if (pendingObjectUrl) URL.revokeObjectURL(pendingObjectUrl);
    pendingObjectUrl = URL.createObjectURL(new Blob([png.buffer], { type: "image/png" }));
    const nextUrl = pendingObjectUrl;
    const decodedImage = new Image();
    decodedImage.src = nextUrl;
    void decodedImage.decode().then(() => {
      if (revision !== imageRevision) return;
      const previousUrl = objectUrl;
      image.src = nextUrl;
      objectUrl = nextUrl;
      pendingObjectUrl = null;
      // A scaled PNG has `scale` image pixels per CSS pixel; show it at its logical size.
      const scale = Math.max(1, model.get("scale") || 1);
      image.style.width = `${decodedImage.naturalWidth / scale}px`;
      image.style.height = `${decodedImage.naturalHeight / scale}px`;
      resetPanPreview?.();
      if (previousUrl) URL.revokeObjectURL(previousUrl);
    }).catch(() => {
      if (revision === imageRevision) pendingObjectUrl = null;
      URL.revokeObjectURL(nextUrl);
    });
  }
  updateImage();

  let frozen = false;

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

  // Step by 80% of the viewport so a strip of the old view stays visible and
  // the user keeps their bearings. Negative dx moves the camera downstream
  // (drag semantics, see Python _move_by).
  const PAN_STEP_FRACTION = 0.8;
  function makeScrollBtn(side: "left" | "right"): HTMLButtonElement {
    const points = side === "left" ? "15 18 9 12 15 6" : "9 18 15 12 9 6";
    const btn = makeBtn(
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="${points}" /></svg>`,
      side === "left" ? "Scroll left (80% of the view)" : "Scroll right (80% of the view)",
      () => {
        const step = Math.max(1, Math.round((model.get("cols") ?? 0) * PAN_STEP_FRACTION));
        interaction.panBy(side === "left" ? step : -step, 0);
      },
    );
    btn.style.cssText += `; position: absolute; top: 50%; ${side}: 8px; transform: translateY(-50%); z-index: 1;`;
    return btn;
  }
  const scrollLeftBtn = makeScrollBtn("left");
  const scrollRightBtn = makeScrollBtn("right");
  wrapper.appendChild(scrollLeftBtn);
  wrapper.appendChild(scrollRightBtn);

  btnContainer.appendChild(zoomOutBtn);
  btnContainer.appendChild(zoomInBtn);
  btnContainer.appendChild(freezeBtn);
  wrapper.appendChild(btnContainer);

  const onPngChange = () => updateImage();
  model.on("change:viewport_png", onPngChange);

  function doFreeze(): void {
    frozen = true;
    btnContainer.style.display = "none";
    scrollLeftBtn.style.display = "none";
    scrollRightBtn.style.display = "none";
    pageIndicator.style.display = "none";
    image.style.cursor = "default";
    model.send({ type: "freeze" });
  }

  const onCustomMessage = (msg: { type: string }) => {
    if (msg.type === "freeze") doFreeze();
    if (msg.type === "pan_done") {
      interaction.finishPan();
      // If a new image is still decoding it will clear the preview itself;
      // otherwise (e.g. the pan was clamped and nothing changed) clear it now.
      if (!pendingObjectUrl) interaction.resetPreview();
    }
  };
  model.on("msg:custom", onCustomMessage);

  const interaction = attachInteraction(
    image,
    model,
    () => ({ cellWidth: model.get("cell_width"), cellHeight: model.get("cell_height"), hitRegions: model.get("hit_regions") }),
    () => frozen,
  );
  resetPanPreview = interaction.resetPreview;

  return {
    destroy() {
      interaction.detach();
      imageRevision += 1;
      model.off("change:viewport_png", onPngChange);
      model.off("change:background_color", updateBackground);
      model.off("msg:custom", onCustomMessage);
      model.off("change:page_index", updatePageLabel);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      if (pendingObjectUrl) URL.revokeObjectURL(pendingObjectUrl);
    },
  };
}

export default { render };
