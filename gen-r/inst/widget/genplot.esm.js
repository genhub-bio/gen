// Generated from js/shiny.ts by `npm run build-r` / `make r`. Edit TypeScript sources, not this file.

// js/interaction.ts
function attachInteraction(image, model, getGeometry, isFrozen) {
  let pointerDown = !1, panInFlight = !1, isDragging = !1, dragStartX = 0, dragStartY = 0, resetPreview = () => {
    image.style.transform = "";
  };
  function cellAt(clientX, clientY) {
    let rect = image.getBoundingClientRect(), { cellWidth, cellHeight } = getGeometry(), scaleX = image.naturalWidth / rect.width, scaleY = image.naturalHeight / rect.height;
    return {
      col: Math.floor((clientX - rect.left) * scaleX / cellWidth),
      row: Math.floor((clientY - rect.top) * scaleY / cellHeight)
    };
  }
  let onMouseMove = (e) => {
    if (isFrozen()) {
      image.style.cursor = "default";
      return;
    }
    if (pointerDown || panInFlight) return;
    let { col, row } = cellAt(e.clientX, e.clientY);
    image.style.cursor = getGeometry().hitRegions.some((region) => region.x === col && region.y === row) ? "pointer" : "grab";
  };
  image.addEventListener("mousemove", onMouseMove);
  let onMouseDown = (e) => {
    isFrozen() || e.button !== 0 || panInFlight || (pointerDown = !0, isDragging = !1, dragStartX = e.clientX, dragStartY = e.clientY, resetPreview(), e.preventDefault());
  };
  image.addEventListener("mousedown", onMouseDown);
  let onWindowMouseMove = (e) => {
    if (!pointerDown) return;
    if (isFrozen()) {
      pointerDown = !1, resetPreview();
      return;
    }
    let totalDx = e.clientX - dragStartX, totalDy = e.clientY - dragStartY;
    !isDragging && (Math.abs(totalDx) > 4 || Math.abs(totalDy) > 4) && (isDragging = !0, image.style.cursor = "grabbing"), isDragging && (image.style.transform = `translate(${totalDx}px, ${totalDy}px)`);
  }, panBy = (dx, dy) => {
    if (isFrozen() || panInFlight || dx === 0 && dy === 0) return !1;
    let rect = image.getBoundingClientRect(), { cellWidth, cellHeight } = getGeometry(), cssCellWidth = cellWidth * rect.width / image.naturalWidth, cssCellHeight = cellHeight * rect.height / image.naturalHeight;
    return image.style.transform = `translate(${dx * cssCellWidth}px, ${dy * cssCellHeight}px)`, panInFlight = !0, image.style.cursor = "progress", model.send({ type: "pan", dx, dy }), !0;
  }, onWindowMouseUp = (e) => {
    if (!(!pointerDown || e.button !== 0)) {
      if (isFrozen()) {
        pointerDown = !1, resetPreview();
        return;
      }
      if (isDragging) {
        let rect = image.getBoundingClientRect(), { cellWidth, cellHeight } = getGeometry(), cssCellWidth = cellWidth * rect.width / image.naturalWidth, cssCellHeight = cellHeight * rect.height / image.naturalHeight, termDx = Math.trunc((e.clientX - dragStartX) / cssCellWidth), termDy = Math.trunc((e.clientY - dragStartY) / cssCellHeight);
        panBy(termDx, termDy) || resetPreview();
      } else {
        let { col, row } = cellAt(e.clientX, e.clientY);
        model.send({ type: "mouse_click", col, row, button: e.button });
      }
      pointerDown = !1, isDragging = !1, panInFlight || (image.style.cursor = "grab");
    }
  }, finishPan = () => {
    panInFlight = !1, image.style.cursor = "grab";
  };
  return window.addEventListener("mousemove", onWindowMouseMove), window.addEventListener("mouseup", onWindowMouseUp), { detach: () => {
    resetPreview(), image.removeEventListener("mousemove", onMouseMove), image.removeEventListener("mousedown", onMouseDown), window.removeEventListener("mousemove", onWindowMouseMove), window.removeEventListener("mouseup", onWindowMouseUp);
  }, resetPreview, finishPan, panBy };
}

// js/index.ts
function render({ model, el }) {
  el.style.overflowX = "auto";
  let wrapper = document.createElement("div");
  wrapper.style.cssText = "position: relative; display: inline-block; line-height: 0;", wrapper.style.overflow = "hidden";
  let updateBackground = () => {
    wrapper.style.backgroundColor = model.get("background_color");
  };
  updateBackground(), model.on("change:background_color", updateBackground);
  let image = document.createElement("img");
  image.style.cssText = "display: block; cursor: grab;", image.draggable = !1, wrapper.appendChild(image), el.appendChild(wrapper);
  let objectUrl = null, pendingObjectUrl = null, imageRevision = 0, resetPanPreview = null;
  function pngBytes(source) {
    if (typeof source == "string") {
      let binary = atob(source.slice(source.indexOf(",") + 1));
      return Uint8Array.from(binary, (character) => character.charCodeAt(0));
    }
    return source instanceof ArrayBuffer ? new Uint8Array(source) : source instanceof DataView ? new Uint8Array(source.buffer, source.byteOffset, source.byteLength) : source instanceof Uint8Array ? source : new Uint8Array(source);
  }
  function updateImage() {
    let data = pngBytes(model.get("viewport_png")), png = new Uint8Array(data.byteLength);
    if (png.set(data), png.byteLength === 0) return;
    let revision = ++imageRevision;
    pendingObjectUrl && URL.revokeObjectURL(pendingObjectUrl), pendingObjectUrl = URL.createObjectURL(new Blob([png.buffer], { type: "image/png" }));
    let nextUrl = pendingObjectUrl, decodedImage = new Image();
    decodedImage.src = nextUrl, decodedImage.decode().then(() => {
      if (revision !== imageRevision) return;
      let previousUrl = objectUrl;
      image.src = nextUrl, objectUrl = nextUrl, pendingObjectUrl = null;
      let scale = Math.max(1, model.get("scale") || 1);
      image.style.width = `${decodedImage.naturalWidth / scale}px`, image.style.height = `${decodedImage.naturalHeight / scale}px`, resetPanPreview?.(), previousUrl && URL.revokeObjectURL(previousUrl);
    }).catch(() => {
      revision === imageRevision && (pendingObjectUrl = null), URL.revokeObjectURL(nextUrl);
    });
  }
  updateImage();
  let frozen = !1, sharedBtnStyle = [
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
    "padding: 0"
  ].join("; "), btnContainer = document.createElement("div");
  btnContainer.style.cssText = "position: absolute; top: 8px; right: 8px; display: flex; flex-direction: row; gap: 4px; z-index: 1;";
  let pageArrowStyle = [
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
    "padding: 0"
  ].join("; ");
  function makeBtn(svg, title, onClick) {
    let btn = document.createElement("button");
    return btn.innerHTML = svg, btn.setAttribute("style", sharedBtnStyle), btn.title = title, btn.addEventListener("mousedown", (e) => e.preventDefault()), btn.addEventListener("click", onClick), btn;
  }
  let pageable = (model.get("page_count") ?? 1) > 1, pageIndicator = document.createElement("div");
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
    "user-select: none"
  ].join("; "), pageIndicator.style.cssText += "; position: absolute; top: 8px; left: 8px; z-index: 1;", pageIndicator.style.display = pageable ? "flex" : "none";
  function makePageArrow(svg, title, direction) {
    let arrow = document.createElement("button");
    return arrow.innerHTML = svg, arrow.title = title, arrow.setAttribute("style", pageArrowStyle), arrow.addEventListener("mousedown", (e) => e.preventDefault()), arrow.addEventListener("click", () => model.send({ type: "page", direction })), arrow;
  }
  let pageLeftBtn = makePageArrow(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<polyline points="15 18 9 12 15 6" />
</svg>`,
    "Previous sequence graph",
    "prev"
  ), pageLabel = document.createElement("span");
  pageLabel.style.cssText = "font-size: 12px; line-height: 24px; padding: 0 2px; white-space: nowrap;";
  let pageRightBtn = makePageArrow(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<polyline points="9 18 15 12 9 6" />
</svg>`,
    "Next sequence graph",
    "next"
  );
  function updatePageLabel() {
    let count = model.get("page_count") ?? 1, index = model.get("page_index") ?? 0;
    pageLabel.textContent = `${index + 1}/${count}`;
  }
  updatePageLabel(), model.on("change:page_index", updatePageLabel), pageIndicator.appendChild(pageLeftBtn), pageIndicator.appendChild(pageLabel), pageIndicator.appendChild(pageRightBtn);
  let zoomInBtn = makeBtn(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
<circle cx="10" cy="10" r="7" stroke-width="1.2" />
<line x1="16" y1="16" x2="21" y2="21" stroke-width="2.5" />
<line x1="10" y1="6" x2="10" y2="14" />
<line x1="6" y1="10" x2="14" y2="10" />
</svg>`,
    "Zoom in (+)",
    () => model.send({ type: "zoom", direction: "in" })
  ), zoomOutBtn = makeBtn(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
<circle cx="10" cy="10" r="7" stroke-width="1.2" />
<line x1="16" y1="16" x2="21" y2="21" stroke-width="2.5" />
<line x1="6" y1="10" x2="14" y2="10" />
</svg>`,
    "Zoom out (-)",
    () => model.send({ type: "zoom", direction: "out" })
  ), freezeBtn = makeBtn(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
<rect x="5" y="10" width="14" height="11" rx="2" />
<path d="M8 10V7a4 4 0 0 1 8 0v3" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" />
</svg>`,
    "Freeze as static image",
    () => doFreeze()
  );
  wrapper.appendChild(pageIndicator);
  let PAN_STEP_FRACTION = 0.8;
  function makeScrollBtn(side) {
    let btn = makeBtn(
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="${side === "left" ? "15 18 9 12 15 6" : "9 18 15 12 9 6"}" /></svg>`,
      side === "left" ? "Scroll left (80% of the view)" : "Scroll right (80% of the view)",
      () => {
        let step = Math.max(1, Math.round((model.get("cols") ?? 0) * PAN_STEP_FRACTION));
        interaction.panBy(side === "left" ? step : -step, 0);
      }
    );
    return btn.style.cssText += `; position: absolute; top: 50%; ${side}: 8px; transform: translateY(-50%); z-index: 1;`, btn;
  }
  let scrollLeftBtn = makeScrollBtn("left"), scrollRightBtn = makeScrollBtn("right");
  wrapper.appendChild(scrollLeftBtn), wrapper.appendChild(scrollRightBtn), btnContainer.appendChild(zoomOutBtn), btnContainer.appendChild(zoomInBtn), btnContainer.appendChild(freezeBtn), wrapper.appendChild(btnContainer);
  let onPngChange = () => updateImage();
  model.on("change:viewport_png", onPngChange);
  function doFreeze() {
    frozen = !0, btnContainer.style.display = "none", scrollLeftBtn.style.display = "none", scrollRightBtn.style.display = "none", pageIndicator.style.display = "none", image.style.cursor = "default", model.send({ type: "freeze" });
  }
  let onCustomMessage = (msg) => {
    msg.type === "freeze" && doFreeze(), msg.type === "pan_done" && (interaction.finishPan(), pendingObjectUrl || interaction.resetPreview());
  };
  model.on("msg:custom", onCustomMessage);
  let interaction = attachInteraction(
    image,
    model,
    () => ({ cellWidth: model.get("cell_width"), cellHeight: model.get("cell_height"), hitRegions: model.get("hit_regions") }),
    () => frozen
  );
  return resetPanPreview = interaction.resetPreview, {
    destroy() {
      interaction.detach(), imageRevision += 1, model.off("change:viewport_png", onPngChange), model.off("change:background_color", updateBackground), model.off("msg:custom", onCustomMessage), model.off("change:page_index", updatePageLabel), objectUrl && URL.revokeObjectURL(objectUrl), pendingObjectUrl && URL.revokeObjectURL(pendingObjectUrl);
    }
  };
}
var index_default = { render };

// js/shiny.ts
var FRAME_KEYS = ["hit_regions", "cell_width", "cell_height", "background_color", "viewport_png", "ack"], ShinyModelAdapter = class {
  constructor(base) {
    this.base = base;
    this.frame = /* @__PURE__ */ new Map();
    this.listeners = /* @__PURE__ */ new Map();
  }
  get(key) {
    return this.frame.has(key) ? this.frame.get(key) : this.base.get(key);
  }
  on(event, callback) {
    let callbacks = this.listeners.get(event) ?? /* @__PURE__ */ new Set();
    callbacks.add(callback), this.listeners.set(event, callbacks);
  }
  off(event, callback) {
    this.listeners.get(event)?.delete(callback);
  }
  send(message) {
    message.type !== "freeze" && (this.base.set("op", message), this.base.set("seq", (this.base.get("seq") ?? 0) + 1), this.base.save_changes());
  }
  applyFrame(next) {
    for (let key of FRAME_KEYS)
      key in next && (this.frame.set(key, next[key]), key !== "ack" && this.emit(`change:${key}`, void 0));
    "ack" in next && this.emit("msg:custom", { type: "pan_done" });
  }
  emit(event, payload) {
    for (let callback of [...this.listeners.get(event) ?? []])
      callback(payload);
  }
};
function render2({ model, el }) {
  let adapter = new ShinyModelAdapter(model), shiny = window.Shiny, destroyed = !1, messageName = adapter.get("view_message");
  shiny && typeof messageName == "string" && shiny.addCustomMessageHandler(messageName, (message) => {
    destroyed || adapter.applyFrame(message);
  });
  let rendered = index_default.render({ model: adapter, el });
  return {
    destroy() {
      destroyed = !0, rendered.destroy();
    }
  };
}
var shiny_default = { render: render2 };
export {
  shiny_default as default
};
