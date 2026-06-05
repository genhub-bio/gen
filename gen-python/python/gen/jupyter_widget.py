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

_ESM = pathlib.Path(__file__).parent / "static" / "jupyter_widget.js"

# All live widget instances, so that module-level helpers can operate on them.


class GraphWidget(anywidget.AnyWidget):
    """Jupyter widget that displays a Gen graph using the native Rust renderer.

    Usage
    -----
    ::

        repo   = gen.Repository()
        sg     = repo.get_sequence_graphs()[0]
        widget = repo.plot(sg)   # or sg.plot()
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
            ``repo.plot(sg)`` or ``sg.plot()``.
        """
        super().__init__(**kwargs)
        self._controller = controller
        self._frozen = False
        self._static_png: str = ""
        self._display_handle = None

        # Re-render when the viewport size changes.
        self.observe(self._on_resize, names=["cols", "rows"])

        # Handle custom messages from the frontend (keyboard / mouse).
        self.on_msg(self._on_frontend_msg)

        # Initial render.
        self._render()

    # ── Display ───────────────────────────────────────────────────────────────

    def _ipython_display_(self, **kwargs):
        """Clone the controller and display an independent widget in this cell.

        Each cell gets its own copy of the graph and computed layouts so that
        programmatic changes to the original widget (in another cell) do not
        affect previously displayed outputs.  Mouse interaction and canvas
        buttons work on the per-cell clone.
        """
        from IPython.display import display

        cloned_ctrl = self._controller.clone_controller()
        snapshot = GraphWidget(cloned_ctrl, cols=self.cols, rows=self.rows)
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

    def go_to(self, target) -> None:
        """Instantly move the camera to a graph position, locus, or annotation.

        Parameters
        ----------
        target:
            A ``Position`` (from ``locus.start()`` / ``locus.end()``),
            a ``Locus`` (from ``repo.search()``), or
            an ``Annotation`` object (e.g. from ``widget.list_annotations()``).

        Example
        -------
        ::

            matches = repo.search(bg, "ACGT...")
            widget.go_to(matches[0].start())
            widget.go_to(matches[0])

            records = widget.list_annotations()
            widget.go_to(records[0])
        """
        if self._frozen:
            return
        from gen import Annotation, Locus  # noqa: PLC0415

        if isinstance(target, Annotation):
            self._controller.go_to_annotation_obj(target)
        elif isinstance(target, Locus):
            self._controller.go_to_locus(target)
        else:
            self._controller.go_to_pos(target)
        self._render()

    def show(self, target, color: str | None = None) -> None:
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
            self._controller.go_to_annotation_obj(target)
            self._controller.highlight_annotation_obj(target, color)
        else:
            self._controller.go_to_pos(target.start())
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
        """Add an annotation track panel below the graph.

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
            Track panel label.  Required when *annotations* is supplied.
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
        """Return list of loaded track-panel annotation names."""
        return json.loads(self._controller.get_track_names())

    def remove_annotation_track(self, name: str) -> None:
        """Remove an annotation track panel by name."""
        if self._frozen:
            return
        self._controller.remove_track(name)
        self._render()

    def clear_all_annotations(self) -> None:
        """Clear all annotation track panels and inline annotations."""
        if self._frozen:
            return
        self._controller.clear_all_annotations()
        self._controller.clear_all_inline_annotations()
        self._render()

    # ── Inline annotation API ─────────────────────────────────────────────────
    #
    # Inline annotations are rendered directly on the graph — each annotation
    # is tinted on the nodes it covers and labelled below its bounding box.
    # Use add_annotation_track() for grouped annotations in a separate aligned
    # panel below the graph.

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
        self._controller.add_inline_annotation([annotation], annotation.name)
        self._render()

    def inline_annotations(self) -> list:
        """Return list of inline annotation names currently displayed."""
        return json.loads(self._controller.get_inline_annotation_names())

    def list_annotations(self) -> list:
        """Return all loaded annotations (track and inline) as ``Annotation`` objects.

        Example
        -------
        ::

            records = widget.list_annotations()
            mcs = next(r for r in records if r.name == "MCS")
            widget.go_to(mcs)
        """
        return self._controller.list_annotations()

    def remove_annotation(self, name: str) -> None:
        """Remove all inline annotations with the given name.

        If ``add_annotation`` was called more than once with annotations
        sharing the same name, every copy is removed.
        """
        if self._frozen:
            return
        self._controller.remove_inline_annotation(name)
        self._render()

    def clear_all_inline_annotations(self) -> None:
        """Clear all inline annotations from the graph."""
        if self._frozen:
            return
        self._controller.clear_all_inline_annotations()
        self._render()
