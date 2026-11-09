import widget, { AnywidgetModel, RenderContext } from "./index";

// The model anyhtmlwidget gives a widget inside a Shiny session. Values set here are uploaded
// to R by `save_changes`, and `model.send` is not implemented, so operations travel as the
// widget values `op` and `seq`.
interface ShinyBaseModel {
  get(key: string): unknown;
  set(key: string, value: unknown): void;
  save_changes(): void;
}

interface ShinyGlobal {
  addCustomMessageHandler(name: string, handler: (message: Record<string, unknown>) => void): void;
}

type Listener = (...args: never[]) => void;

// Entries of a frame in the order the widget script should see them: geometry before the image
// that is positioned with it, and the acknowledgement last so the pan lock is released after
// the new image has started to decode.
const FRAME_KEYS = ["hit_regions", "cell_width", "cell_height", "background_color", "viewport_png", "ack"];

/**
 * Presents a Shiny widget model as the anywidget model the shared script expects.
 *
 * R answers each operation with one custom message holding a complete frame. Frames are kept
 * here instead of being written into the base model, because the base model would upload every
 * value it was given back to R on the next `save_changes`, and a frame carries a whole PNG.
 */
class ShinyModelAdapter {
  private readonly frame = new Map<string, unknown>();
  private readonly listeners = new Map<string, Set<Listener>>();

  constructor(private readonly base: ShinyBaseModel) {}

  get(key: string): unknown {
    return this.frame.has(key) ? this.frame.get(key) : this.base.get(key);
  }

  on(event: string, callback: Listener): void {
    const callbacks = this.listeners.get(event) ?? new Set<Listener>();
    callbacks.add(callback);
    this.listeners.set(event, callbacks);
  }

  off(event: string, callback: Listener): void {
    this.listeners.get(event)?.delete(callback);
  }

  send(message: Record<string, unknown>): void {
    // Freezing only locks the controls in the page; there is no Python-side image to write.
    if (message.type === "freeze") return;
    this.base.set("op", message);
    this.base.set("seq", ((this.base.get("seq") as number | undefined) ?? 0) + 1);
    this.base.save_changes();
  }

  applyFrame(next: Record<string, unknown>): void {
    for (const key of FRAME_KEYS) {
      if (!(key in next)) continue;
      this.frame.set(key, next[key]);
      if (key === "ack") continue;
      this.emit(`change:${key}`, undefined);
    }
    if ("ack" in next) this.emit("msg:custom", { type: "pan_done" });
  }

  private emit(event: string, payload: unknown): void {
    for (const callback of [...(this.listeners.get(event) ?? [])]) {
      (callback as (payload: unknown) => void)(payload);
    }
  }
}

function render({ model, el }: RenderContext): { destroy(): void } {
  const adapter = new ShinyModelAdapter(model as unknown as ShinyBaseModel);
  const shiny = (window as unknown as { Shiny?: ShinyGlobal }).Shiny;
  let destroyed = false;
  // Shiny cannot remove a message handler, so a destroyed widget's handler goes quiet instead.
  const messageName = adapter.get("view_message");
  if (shiny && typeof messageName === "string") {
    shiny.addCustomMessageHandler(messageName, (message) => {
      if (!destroyed) adapter.applyFrame(message);
    });
  }
  const rendered = widget.render({ model: adapter as unknown as AnywidgetModel, el });
  return {
    destroy() {
      destroyed = true;
      rendered.destroy();
    },
  };
}

export default { render };
