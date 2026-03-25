// graph_widget_src.js — pre-bundle source for the gen Jupyter widget.
//
// This file is NOT loaded directly by the browser.  build_bundle.py reads it,
// prepends the wasm-pack-generated glue + the WASM binary (base64-encoded),
// and writes the combined self-contained bundle to static/graph_widget.js.
//
// The bundle is then inlined by anywidget via _esm = Path("static/widget.js"),
// which creates a blob URL from the file contents and imports it as an ES
// module — no Jupyter server extension or separate asset serving required.

// These names are injected by build_bundle.py:
//   __GEN_WASM_INIT__  — the default export (init function) from the
//                        wasm-pack-generated glue (gen_wasm_widget.js)
//   mount_app          — the named export from the same glue
//   __GEN_WASM_BYTES__ — Uint8Array containing the raw .wasm bytes

async function render({ model, el }) {
  // Initialise the WASM module from the inlined bytes.
  // Passing a BufferSource to init() bypasses any fetch / import.meta.url
  // resolution, so the bundle is fully self-contained.
  await __GEN_WASM_INIT__(__GEN_WASM_BYTES__.buffer);

  // Create a sized container for the ratzilla terminal.
  const uid = "gen-" + crypto.randomUUID();
  const container = document.createElement("div");
  container.id = uid;
  container.style.cssText = "width:100%;height:500px;overflow:hidden;";
  el.appendChild(container);

  // Mount the gen-tui widget into the container.
  const topology = model.get("topology");
  const palette = model.get("palette");
  const pathNodes = model.get("path_nodes");
  let app;
  try {
    app = mount_app(uid, topology, palette, pathNodes);
  } catch (err) {
    el.textContent = "gen widget error: " + err;
    console.error("gen widget mount_app failed", err);
    return;
  }

  // Wire up sequence fetching: WASM → Python kernel → WASM.
  //
  // When the renderer needs a sequence it calls our callback with a
  // single-element JS Array containing the spec string.
  // We forward it to Python via model.send(); Python looks up the sequence
  // from the gen database and replies with a sequences_response message.
  app.set_sequence_callback((specsArray) => {
    const nodes = Array.from(specsArray);
    model.send({ type: "get_sequences", nodes });
  });

  model.on("msg:custom", (msg) => {
    if (msg.type === "sequences_response") {
      app.deliver_sequences(msg.data_json);
    }
  });

  // Prevent widget navigation keys from leaking to Jupyter / the browser.
  // Arrow keys would scroll the page; hjkl/Enter/Esc trigger notebook shortcuts.
  const WIDGET_KEYS = new Set([
    "h", "j", "k", "l",
    "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
    "Enter", "Escape", "+", "=", "-", "r", "p",
  ]);
  container.addEventListener("keydown", (e) => {
    if (WIDGET_KEYS.has(e.key)) {
      e.preventDefault();
      e.stopPropagation();
    }
  });

  // anywidget calls the returned function on widget teardown.
  return () => {
    try { app.destroy?.(); } catch (_) {}
    container.remove();
  };
}

export default { render };
