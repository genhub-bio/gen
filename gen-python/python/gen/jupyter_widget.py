"""Anywidget-based Jupyter frontend for the Gen graph viewer.

Architecture
------------
- Native Rust (GraphController) owns all state, performs layout, renders into
  a ratatui Buffer, and serialises the result to a RenderedFrame JSON string.
- Python (GraphWidget) is a thin bridge: it holds the Rust controller,
  requests frames, and syncs them to the frontend via the `frame` traitlet.
- The frontend (static/jupyter_widget.js) is a dumb canvas painter that also sends
  mouse events back as custom messages.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import anywidget
import traitlets

# Default viewport dimensions (terminal columns × rows).
DEFAULT_COLS = 60
DEFAULT_ROWS = 12


def _highlighted(cell: dict, frame: dict) -> bool:
    """True when the cell carries a highlight colour.

    The graph renderer paints text cells with inverted neutral colours
    (fg=neutral_bg, bg=neutral_fg) by default.  A true highlight changes
    the cell's *bg* to an accent colour while keeping the inverted fg.
    Edge cells never set *bg* (highlighted edges are already drawn with
    heavy/dashed box-drawing glyphs by the Rust renderer), so this only
    ever fires for text cells.
    """
    bg = cell.get("bg")
    if bg is None:
        return False
    return bg != frame.get("neutral_fg")


def _transform(text: str, highlighted: bool) -> str:
    """Adjust casing based on highlight state."""
    return text.upper() if highlighted else text.lower()


_ESM = pathlib.Path(__file__).parent / "static" / "jupyter_widget.js"

# All live widget instances, so that module-level helpers can operate on them.


class GraphWidget(anywidget.AnyWidget):
    """Jupyter widget that displays a Gen graph using the native Rust renderer.

    A widget obtained from a single ``SequenceGraph`` (via ``repo.plot(sg)`` or
    ``sg.plot()``) shows just that graph. A widget obtained from a ``Sample``
    (via ``sample.plot()``) pages through every sequence graph it contains,
    showing a header row with the sequence graph name plus a floating
    ``<index/count>`` pager indicator next to the zoom buttons.

    Usage
    -----
    ::

        repo   = gen.Repository()
        sg     = repo.get_sequence_graphs()[0]
        widget = repo.plot(sg)   # or sg.plot()
        widget  # display in Jupyter cell

        # Optionally send commands from Python afterwards:
        widget.scroll_left()
        widget.zoom_in()

        sample = repo.import_fasta(...)
        sample_widget = sample.plot()
        sample_widget.next_page()
        sample_widget.prev_page()
    """

    _esm = _ESM

    # ── Traitlets synced with the frontend ────────────────────────────────────

    # The rendered frame.  Updated on every render call.
    frame: dict = traitlets.Dict({}).tag(sync=True)

    # Viewport size in terminal cells.  The frontend can resize these.
    cols: int = traitlets.Int(DEFAULT_COLS).tag(sync=True)
    rows: int = traitlets.Int(DEFAULT_ROWS).tag(sync=True)

    # Number of pages available. The frontend only shows pager arrows when
    # this is greater than 1 (plain GraphWidgets have exactly one page).
    page_count: int = traitlets.Int(1).tag(sync=True)

    # Index of the currently active page, for the frontend's <index/count>
    # pager indicator.
    page_index: int = traitlets.Int(0).tag(sync=True)

    def __init__(self, controller, *, colors=None, **kwargs):
        """
        Parameters
        ----------
        controller:
            A ``gen.PyGraphController`` instance.  Normally obtained via
            ``repo.plot(sg)``, ``sg.plot()``, or ``sample.plot()``.
        colors : callable | dict | list, optional
            Controls annotation colours loaded from the repository.

            - **callable** ``(ann: Annotation) -> str | None`` — called once per
              annotation; return a CSS hex colour to paint it, or ``None`` to hide it.
            - **dict** ``{name: color}`` — maps ``ann.name`` to a colour; annotations
              absent from the dict are hidden.
            - **list** ``[color, ...]`` — assigns colours from the list cyclically.

            When omitted the theme accent palette is used automatically.
        """
        kwargs.setdefault("page_count", controller.page_count)
        super().__init__(**kwargs)
        self._controller = controller
        self._frozen = False
        self._static_png: str = ""
        self._display_handle = None

        # Re-render when the viewport size changes.
        self.observe(self._on_resize, names=["cols", "rows"])

        # Handle custom messages from the frontend (keyboard / mouse).
        self.on_msg(self._on_frontend_msg)

        # Load annotation groups (skipped if controller is a clone with groups already loaded).
        self._load_initial_annotations(colors)

        # Initial render.
        self._render()

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Render the current frame as plain ASCII/Unicode text.

        This is what shows up as the `text/plain` fallback wherever the
        widget JS can't run (nbconvert text/markdown export, `print()`, a
        plain terminal). The frame is already a full character grid computed
        server-side by the Rust renderer, so no separate rendering path is
        needed.
        """
        cols, rows = (
            self.frame.get("cols", self.cols),
            self.frame.get("rows", self.rows),
        )
        grid = [[" "] * cols for _ in range(rows)]
        for cell in self.frame.get("cells", []):
            x, y, text = cell["x"], cell["y"], cell["text"]
            if text and 0 <= y < rows and 0 <= x < cols:
                grid[y][x] = _transform(text, _highlighted(cell, self.frame))
        lines = ["".join(row).rstrip() for row in grid]
        if self.page_count > 1 and lines:
            prefix = f"[{self.page_index + 1}/{self.page_count}] "
            lines[0] = (prefix + lines[0].lstrip()).rstrip()
        return "\n".join(lines)

    def _ipython_display_(self, **kwargs):
        """Clone the controller and display an independent widget in this cell.

        Each cell gets its own copy of the graph and computed layouts so that
        programmatic changes to the original widget (in another cell) do not
        affect previously displayed outputs.  Mouse interaction and canvas
        buttons work on the per-cell clone.
        """
        from IPython.display import display

        cloned_ctrl = self._controller.clone_controller()
        snapshot = type(self)(cloned_ctrl, cols=self.cols, rows=self.rows)
        data = {
            "text/plain": repr(snapshot),
            "application/vnd.jupyter.widget-view+json": {
                "version_major": 2,
                "version_minor": 0,
                "model_id": snapshot._model_id,
            },
        }
        snapshot._display_handle = display(data, raw=True, display_id=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_initial_annotations(self, colors) -> None:
        """Load annotation groups from the repository, applying the colors mapping.

        Skips if the controller already has groups loaded (e.g. it is a clone).
        """
        if self._controller.annotations_loaded:
            return

        if colors is None:
            self._controller.trigger_auto_load()
            return

        color_fn = self._build_color_fn(colors)
        annotations = self._controller.list_annotations()
        color_map = {ann.id: color_fn(ann) for ann in annotations}
        self._controller.load_annotation_groups_with_colors(color_map)

    @staticmethod
    def _build_color_fn(colors):
        """Return a callable ``(Annotation) -> str | None`` from any supported colors value."""
        if callable(colors):
            return colors
        if isinstance(colors, dict):
            return lambda ann: colors.get(ann.name)
        if isinstance(colors, list):
            palette = list(colors)
            n = len(palette)
            if n == 0:
                raise ValueError("colors list must not be empty")
            assigned: dict = {}
            counter = [0]

            def _cyclic(ann):
                if ann.id not in assigned:
                    assigned[ann.id] = palette[counter[0] % n]
                    counter[0] += 1
                return assigned[ann.id]

            return _cyclic
        raise TypeError(
            f"colors must be a callable, dict, or list; got {type(colors).__name__}"
        )

    def _render(self) -> None:
        """Ask Rust to render a frame and push it to the frontend."""
        frame_json = self._controller.render_frame(self.cols, self.rows)
        self.frame = json.loads(frame_json)
        self.page_index = self._controller.page_index

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
            data_url = msg.get("data", "")
            self._static_png = data_url
            if self._display_handle is not None and data_url.startswith(
                "data:image/png;base64,"
            ):
                b64 = data_url.split(",", 1)[1]
                self._display_handle.update(
                    {
                        "application/vnd.jupyter.widget-view+json": {
                            "version_major": 2,
                            "version_minor": 0,
                            "model_id": self._model_id,
                        },
                        "image/png": b64,
                        "text/plain": repr(self),
                    },
                    raw=True,
                )
            return

        if msg_type == "freeze":
            data_url = msg.get("data", "")
            self._static_png = data_url
            self._frozen = True
            if self._display_handle is not None:
                from IPython.display import HTML

                w, h = msg.get("width"), msg.get("height")
                size = f";width:{w}px;height:{h}px" if w and h else ""
                self._display_handle.update(
                    HTML(
                        f'<img src="{data_url}" style="display:block;font-family:monospace{size}" />'
                    )
                )
            return

        if msg_type == "mouse_click":
            self.handle_click(int(msg.get("col", 0)), int(msg.get("row", 0)))
            return

        if msg_type == "zoom":
            if msg.get("direction") == "in":
                self.zoom_in()
            else:
                self.zoom_out()
            return

        if msg_type == "pan":
            self._move_by(int(msg.get("dx", 0)), int(msg.get("dy", 0)))
            return

        if msg_type == "page":
            if msg.get("direction") == "next":
                self.next_page()
            else:
                self.prev_page()
            return

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

    def _move_by(self, dx: int, dy: int) -> None:
        """Move the viewport like a mouse drag of (dx, dy) terminal cells.

        ``dx``/``dy`` follow drag semantics, not camera-direction semantics:
        the Rust controller negates dx internally (dragging right pulls
        upstream/earlier content into view, like dragging a map), so a
        *negative* dx here is what moves the camera rightward/downstream.
        """
        if self._frozen:
            return
        self._controller.move_by(dx, dy)
        self._render()

    def scroll_right(self) -> None:
        """Scroll the view right by one page, to show further-downstream sequence."""
        self._move_by(
            -self.cols, 0
        )  # negative dx -> camera moves downstream (see _move_by)

    def scroll_left(self) -> None:
        """Scroll the view left by one page, back toward earlier/upstream sequence."""
        self._move_by(
            self.cols, 0
        )  # positive dx -> camera moves upstream (see _move_by)

    def scroll_down(self) -> None:
        """Scroll the view down by one page, to show content below the current view."""
        self._move_by(0, -self.rows)

    def scroll_up(self) -> None:
        """Scroll the view up by one page, to show content above the current view."""
        self._move_by(0, self.rows)

    def next_page(self) -> None:
        """Advance to the next sequence graph (only meaningful for a ``Sample``-backed widget)."""
        if self._frozen:
            return
        self._controller.next_page()
        self._render()

    def prev_page(self) -> None:
        """Go back to the previous sequence graph (only meaningful for a ``Sample``-backed widget)."""
        if self._frozen:
            return
        self._controller.prev_page()
        self._render()

    def go_to(self, target, *, center: bool = False) -> None:
        """Instantly move the camera to a graph position, locus, or annotation.

        Parameters
        ----------
        target:
            A ``Position`` (from ``locus.start()`` / ``locus.end()``),
            a ``Locus`` (from ``repo.search()``), or
            an ``Annotation`` object (e.g. from ``widget.list_annotations()``).
        center:
            When ``True``, center the target in the viewport instead of the
            default snap-left placement.

        Example
        -------
        ::

            matches = repo.search(bg, "ACGT...")
            widget.go_to(matches[0].start())
            widget.go_to(matches[0])
            widget.go_to(matches[0], center=True)

            records = widget.list_annotations()
            widget.go_to(records[0])
        """
        if self._frozen:
            return
        from gen import Annotation, Locus  # noqa: PLC0415

        if isinstance(target, Annotation):
            self._controller.go_to_annotation_obj(target, center)
        elif isinstance(target, Locus):
            self._controller.go_to_locus(target, center)
        else:
            self._controller.go_to_pos(target, center)
        self._render()

    def show(self, target, color: str | None = None, *, center: bool = False) -> None:
        """Navigate to and highlight a graph locus or annotation in one call.

        Parameters
        ----------
        target:
            A ``Locus`` returned by ``repo.search()``, or an ``Annotation``
            object (e.g. from ``widget.list_annotations()``).
        color:
            Optional highlight colour.  Accepts named colours
            (``"yellow"``, ``"cyan"``, ``"red"``, …) or a CSS hex string
            (``"#ff8800"``).  When omitted the next unused theme accent
            colour is chosen automatically.
        center:
            When ``True``, center the target in the viewport instead of the
            default snap-left placement.

        Example
        -------
        ::

            matches = repo.search(bg, "ACGT...")
            widget.show(matches[0])

            records = widget.list_annotations()
            widget.show(records[0])
        """
        if self._frozen:
            return
        from gen import Annotation  # noqa: PLC0415

        if isinstance(target, Annotation):
            self._controller.go_to_annotation_obj(target, center)
            self._controller.highlight_annotation_obj(target, color)
        else:
            self._controller.go_to_pos(target.start(), center)
            self._controller.highlight_match(target, color)
        self._render()

    def highlight_match(self, locus, color: str | None = None) -> None:
        """Highlight the nodes covered by a graph locus.

        Parameters
        ----------
        locus:
            A ``Locus`` returned by ``repo.search()``.
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

    # ── Annotation API ────────────────────────────────────────────────────

    def add_annotation_track(
        self,
        annotations=None,
        *,
        file: str | None = None,
        group: str | None = None,
        name: str | None = None,
        from_sample: str | None = None,
        filter=None,
    ) -> None:
        """Add annotations as inline graph highlights with floating labels.

        Exactly one of *annotations*, *file*, or *group* must be supplied.

        Parameters
        ----------
        annotations : list[Annotation], optional
            Annotations built with ``Annotation(locus, name)``.  *name* is
            required when using this form.
        file : str, optional
            Path to a GFF3 or BED annotation file.  Both standard files
            (chromosome/contig names as reference) and pre-translated files
            (node hash-IDs as reference) are accepted; standard files are
            translated automatically.  *name* defaults to the file path.
        group : str, optional
            Annotation group name stored in the repository.
        name : str, optional
            Display label for this annotation track.  Required when *annotations* is supplied.
        from_sample : str, optional
            Sample whose coordinate space the file uses (file tracks only).
            Defaults to ``"reference"``.
        filter : callable, optional
            ``(row: str) -> bool`` predicate applied to each non-header line
            (file tracks only).
        """
        if self._frozen:
            return
        given = sum(x is not None for x in (annotations, file, group))
        if given != 1:
            raise ValueError(
                "exactly one of annotations, file, or group must be supplied"
            )
        if annotations is not None:
            if name is None:
                raise ValueError("name is required when annotations is supplied")
            self._controller.add_track_annotations(annotations, name)
        elif file is not None:
            if filter is not None:
                file = self._apply_row_filter(file, filter)
            self._controller.add_track_file(file, name, from_sample)
        else:
            self._controller.add_track_group(group)
        self._render()

    @staticmethod
    def _apply_row_filter(file: str, filter) -> str:
        # TODO: temporary bandaid — move row filtering to Rust once annotation
        # metadata infrastructure is built out further.
        """Write header lines + approved data rows to a temp file; return its path."""
        suffix = pathlib.Path(file).suffix
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        )
        with open(file, encoding="utf-8") as fh:
            for line in fh:
                if (
                    line.startswith("#")
                    or line.startswith("track")
                    or line.startswith("browser")
                ):
                    tmp.write(line)
                elif filter(line):
                    tmp.write(line)
        tmp.close()
        return tmp.name

    def annotation_tracks(self) -> list:
        """Return list of annotation track names currently loaded."""
        return json.loads(self._controller.get_track_names())

    def remove_annotation_track(self, name: str) -> None:
        """Remove an annotation track by name."""
        if self._frozen:
            return
        self._controller.remove_track(name)
        self._render()

    def clear_all_annotations(self) -> None:
        """Clear all annotations from the graph."""
        if self._frozen:
            return
        self._controller.clear_all_annotations()
        self._render()

    def add_annotation(self, annotation) -> None:
        """Render an annotation inline on the graph canvas.

        The annotation is tinted with an accent colour and its name is placed
        below its bounding box.  Labels avoid each other but give up rather
        than overwrite existing graph content.

        Parameters
        ----------
        annotation : Annotation
            A named annotation built with ``Annotation(locus, name)``.
        """
        if self._frozen:
            return
        self._controller.add_annotation([annotation], annotation.name)
        self._render()

    def annotations(self) -> list:
        """Return list of annotation names currently displayed."""
        return json.loads(self._controller.get_annotation_names())

    def list_annotations(self) -> list:
        """Return all annotations loaded into the widget as ``Annotation`` objects.

        Example
        -------
        ::

            records = widget.list_annotations()
            mcs = next(r for r in records if r.name == "MCS")
            widget.go_to(mcs)
        """
        return self._controller.list_annotations()

    def remove_annotation(self, name: str) -> None:
        """Remove all annotations with the given name.

        If ``add_annotation`` was called more than once with annotations
        sharing the same name, every copy is removed.
        """
        if self._frozen:
            return
        self._controller.remove_annotation(name)
        self._render()
