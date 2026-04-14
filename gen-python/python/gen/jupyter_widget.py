"""Anywidget-based Jupyter frontend for the Gen graph viewer.

Architecture
------------
- Native Rust (GraphController) owns all state, performs layout, renders into
  a ratatui Buffer, and serialises the result to a RenderedFrame JSON string.
- Python (GenGraphWidget) is a thin bridge: it holds the Rust controller,
  requests frames, and syncs them to the frontend via the `frame` traitlet.
- The frontend (static/jupyter_widget.js) is a dumb canvas painter that also sends
  mouse events back as custom messages.
"""

from __future__ import annotations

import json
import pathlib

import anywidget
import traitlets

# Default viewport dimensions (terminal columns × rows).
DEFAULT_COLS = 80
DEFAULT_ROWS = 24

_ESM = pathlib.Path(__file__).parent / "static" / "jupyter_widget.js"


class GenGraphWidget(anywidget.AnyWidget):
    """Jupyter widget that displays a Gen graph using the native Rust renderer.

    Usage
    -----
    ::

        repo   = gen.Repository()
        bg     = repo.get_block_groups()[0]
        widget = repo.plot(bg)   # or bg.plot()
        widget  # display in Jupyter cell

        # Optionally send commands from Python afterwards:
        widget.move_by(-5, 0)
        widget.zoom_in()
    """

    _esm = _ESM

    # ── Traitlets synced with the frontend ────────────────────────────────────

    # The rendered frame.  Updated on every render call.
    frame: dict = traitlets.Dict({}).tag(sync=True)

    # Viewport size in terminal cells.  The frontend can resize these.
    cols: int = traitlets.Int(DEFAULT_COLS).tag(sync=True)
    rows: int = traitlets.Int(DEFAULT_ROWS).tag(sync=True)

    def __init__(self, controller, **kwargs):
        """
        Parameters
        ----------
        controller:
            A ``gen.PyGraphController`` instance.  Normally obtained via
            ``repo.plot(bg)`` or ``bg.plot()``.
        """
        super().__init__(**kwargs)
        self._controller = controller
        self._frozen = False
        self._static_png: str = ""

        # Re-render when the viewport size changes.
        self.observe(self._on_resize, names=["cols", "rows"])

        # Handle custom messages from the frontend (keyboard / mouse).
        self.on_msg(self._on_frontend_msg)

        # Initial render.
        self._render()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _render(self) -> None:
        """Ask Rust to render a frame and push it to the frontend."""
        frame_json = self._controller.render_frame(self.cols, self.rows)
        self.frame = json.loads(frame_json)

    def _on_resize(self, change) -> None:
        # change is required by the traitlets observe protocol but we only need
        # to re-render; the new cols/rows values are read directly from self.
        self._render()

    def _on_frontend_msg(self, widget, msg: dict, buffers) -> None:
        """Dispatch a message from the frontend to the Rust controller."""
        # widget and buffers are required by the anywidget on_msg protocol;
        # all the information we need is in msg.
        msg_type = msg.get("type")

        if msg_type == "snapshot":
            self._static_png = msg.get("data", "")
            return

        if msg_type == "mouse_click":
            self.handle_click(int(msg.get("col", 0)), int(msg.get("row", 0)))

        elif msg_type == "zoom":
            if msg.get("direction") == "in":
                self.zoom_in()
            else:
                self.zoom_out()

        elif msg_type == "pan":
            self.move_by(int(msg.get("dx", 0)), int(msg.get("dy", 0)))

    # ── Public command API ────────────────────────────────────────────────────

    def handle_click(self, col: int, row: int) -> bool:
        """Send a mouse click to the controller and re-render. Returns True if a node was hit."""
        if self._frozen:
            return False
        hit = self._controller.handle_click(col, row)
        self._render()
        return hit

    def zoom_in(self) -> None:
        """Step one zoom level in."""
        if self._frozen:
            return
        self._controller.zoom_in()
        self._render()

    def zoom_out(self) -> None:
        """Step one zoom level out."""
        if self._frozen:
            return
        self._controller.zoom_out()
        self._render()

    def move_by(self, dx: int, dy: int) -> None:
        """Move the viewport by dx, dy cells."""
        if self._frozen:
            return
        self._controller.move_by(dx, dy)
        self._render()

    def freeze(self) -> None:
        """Capture current canvas as static PNG, disable interactivity.

        After calling this the widget becomes a static snapshot.  The canvas
        border indicator is hidden and all interaction methods become no-ops.
        The captured PNG is embedded in ``_repr_html_`` so the notebook can be
        distributed to environments without the Python module installed.
        """
        self._frozen = True
        self.send({"type": "freeze"})

    def _repr_html_(self) -> str:
        if self._static_png:
            return f'<img src="{self._static_png}" style="display:block;font-family:monospace" />'
        return "<pre>Call widget.freeze() to generate a static snapshot.</pre>"

    def go_to(self, pos) -> None:
        """Instantly move the camera to a graph position.

        Parameters
        ----------
        pos:
            A ``GraphPos`` obtained from ``locus.start()`` or ``locus.end()``.

        Example
        -------
        ::

            matches = repo.search(bg, "ACGT...")
            widget.go_to(matches[0].start())
        """
        if self._frozen:
            return
        self._controller.go_to_pos(pos)
        self._render()

    def show(self, locus, color: str | None = None) -> None:
        """Navigate to and highlight a graph locus in one call.

        Parameters
        ----------
        locus:
            A ``GraphLocus`` returned by ``repo.search()``.
        color:
            Optional highlight colour.  Accepts named colours
            (``"yellow"``, ``"cyan"``, ``"red"``, …) or a CSS hex string
            (``"#ff8800"``).  When omitted the next unused theme accent
            colour is chosen automatically.

        Example
        -------
        ::

            matches = repo.search(bg, "ACGT...")
            widget.show(matches[0])
        """
        if self._frozen:
            return
        self._controller.go_to_pos(locus.start())
        self._controller.highlight_match(locus, color)
        self._render()

    def highlight_match(self, locus, color: str | None = None) -> None:
        """Highlight the nodes covered by a graph locus.

        Parameters
        ----------
        locus:
            A ``GraphLocus`` returned by ``repo.search()``.
        color:
            Optional colour for the highlight.  Accepts named colours
            (``"yellow"``, ``"cyan"``, ``"red"``, …) or a CSS hex string
            (``"#ff8800"``).  When omitted the next unused theme accent
            colour is chosen automatically, so multiple ``highlight_match``
            calls without an explicit colour each get a distinct colour.

        Example
        -------
        ::

            matches = repo.search(bg, "ACGT...")
            widget.go_to(matches[0].start())
            widget.highlight_match(matches[0])
        """
        if self._frozen:
            return
        self._controller.highlight_match(locus, color)
        self._render()

    def clear_highlights(self) -> None:
        """Remove all highlights from the graph."""
        if self._frozen:
            return
        self._controller.clear_highlights()
        self._render()

    def show_path(self, color: str | None = None) -> None:
        """Highlight the most recent path for this block group.

        Parameters
        ----------
        color:
            Optional colour for the highlight.  Accepts named colours
            (``"yellow"``, ``"cyan"``, ``"red"``, …) or a CSS hex string
            (``"#ff4444"``).  When omitted the next unused theme accent
            colour is chosen automatically.
        """
        self._controller.show_path(color)
        self._render()

    def clear_path(self) -> None:
        """Remove path highlighting applied by :meth:`show_path`."""
        self._controller.clear_path()
        self._render()

    def refresh(self) -> None:
        """Force a re-render from the current controller state."""
        if self._frozen:
            return
        self._render()
