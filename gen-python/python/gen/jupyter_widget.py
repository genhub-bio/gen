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

import asyncio
import json
import pathlib
import tempfile
import warnings

import anywidget
import ipywidgets
import traitlets

from .ascii_render import frame_text

# Default viewport dimensions (terminal columns × rows).
DEFAULT_COLS = 60
DEFAULT_ROWS = 12


_ESM = pathlib.Path(__file__).parent / "static" / "jupyter_widget.js"

# Prefixed onto the text/plain fallback of a frozen widget only (not the live
# widget's __repr__): a frozen output has no kernel or JS behind it, so unlike
# an interactive display, there is nothing to inspect except this text. It
# orients a reader (human or LLM) skimming the raw notebook JSON who has never
# seen a Gen graph widget before.
_FREEZE_TEXT_HINT = (
    "# Gen graph widget, frozen to a static image for GitHub viewing.\n"
    "# The text below is an ASCII rendering of the same frame: each character is one\n"
    "# terminal cell from the Rust layout engine; UPPERCASE marks a highlighted\n"
    "# annotation region, lowercase is unhighlighted sequence/graph structure.\n"
    "# To interact with this graph instead of reading the ASCII, rerun the notebook's\n"
    "# own cell (e.g. `repo.plot(sg)` or `sample.plot()`) in a live Jupyter kernel; the\n"
    "# returned GraphWidget supports .zoom_in()/.zoom_out(), .scroll_left()/.scroll_right()/\n"
    "# .scroll_up()/.scroll_down(), and .next_page()/.prev_page() for multi-page samples.\n"
)


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

        # Configure before displaying: each display below clones the
        # controller's *current* state, so commands compose into whatever
        # is shown next.
        widget.scroll_left()
        widget.zoom_in()
        widget  # display in Jupyter cell

        # A cell's displayed output is an independent snapshot from this
        # point on. Calling commands on `widget` again will not change what
        # that cell already shows -- it only affects a later display of
        # `widget`, e.g. if it is shown again in a subsequent cell.

        sample = repo.import_fasta(...)
        sample_widget = sample.plot()
        sample_widget.next_page()
        sample_widget  # display in Jupyter cell
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
        return frame_text(self.frame, self.page_count, self.page_index)

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
        annotations = self._controller.annotations
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

    def _freeze(self) -> None:
        """Ask the frontend to replace the live canvas with a static PNG.

        Only meaningful on the actual displayed clone (``_display_handle is
        not None``): unlike the zoom/pan/page commands, which mutate the
        controller state a future display will clone, freezing has nothing
        to compose into for an object that hasn't been displayed yet — the
        object a user's own variable holds is never that clone (see
        ``_ipython_display_``), so this is kept internal and reached only
        through ``freeze_all_widgets()``, which already holds the actual
        displayed instances. Once the frontend confirms, the canvas is
        swapped for a plain ``<img>`` and the widget is closed, since a
        frozen widget can no longer be interacted with anyway.
        """
        if self._frozen:
            return
        self.send({"type": "freeze"})

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
            ascii_repr = repr(self)
            self._frozen = True
            if self._display_handle is not None:
                w, h = msg.get("width"), msg.get("height")
                size = f";width:{w}px;height:{h}px" if w and h else ""
                self._display_handle.update(
                    {
                        "text/html": (
                            f'<img src="{data_url}" '
                            f'style="display:block;font-family:monospace{size}" />'
                        ),
                        "text/plain": _FREEZE_TEXT_HINT + ascii_repr,
                    },
                    raw=True,
                )
            # A frozen widget is inert (every mutating method below early-returns
            # on self._frozen), so nothing is lost by releasing the controller
            # and comm now instead of waiting for the kernel to shut down.
            self.close()
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
            A ``Position`` (from ``locus.start()`` / ``locus.end()``), a
            ``SuperPosition`` (centers on its first position), a ``Locus``
            (from ``repo.search()``), or an ``Annotation`` object (e.g. from
            ``sequence_graph.annotations``).
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

            records = sequence_graph.annotations
            widget.go_to(records[0])
        """
        if self._frozen:
            return
        from gen import Annotation, Locus, SuperPosition  # noqa: PLC0415

        if isinstance(target, Annotation):
            self._controller.go_to_annotation_obj(target, center)
        elif isinstance(target, Locus):
            self._controller.go_to_locus(target, center)
        elif isinstance(target, SuperPosition):
            self._controller.go_to_pos(target.positions[0], center)
        else:
            self._controller.go_to_pos(target, center)
        self._render()

    def show(self, target, color: str | None = None, *, center: bool = False) -> None:
        """Navigate to and highlight a graph locus or annotation in one call.

        Parameters
        ----------
        target:
            A ``Locus`` returned by ``repo.search()``, an ``Annotation``
            object (e.g. from ``sequence_graph.annotations``), or a
            ``Position``/``SuperPosition`` (centers on the position, or the
            first position of a superposition; a position is a single point,
            so nothing is highlighted).
        color:
            Optional highlight colour.  Accepts named colours
            (``"yellow"``, ``"cyan"``, ``"red"``, …) or a CSS hex string
            (``"#ff8800"``).  When omitted the next unused theme accent
            colour is chosen automatically.  Ignored for a
            ``Position``/``SuperPosition`` target, which has nothing to
            highlight.
        center:
            When ``True``, center the target in the viewport instead of the
            default snap-left placement.

        Example
        -------
        ::

            matches = repo.search(bg, "ACGT...")
            widget.show(matches[0])

            records = sequence_graph.annotations
            widget.show(records[0])

            widget.show(matches[0].start())
        """
        if self._frozen:
            return
        from gen import Annotation, Position, SuperPosition  # noqa: PLC0415

        if isinstance(target, Annotation):
            self._controller.go_to_annotation_obj(target, center)
            self._controller.highlight_annotation_obj(target, color)
        elif isinstance(target, SuperPosition):
            self._controller.go_to_pos(target.positions[0], center)
        elif isinstance(target, Position):
            self._controller.go_to_pos(target, center)
        else:
            self._controller.go_to_locus(target, center)
            self._controller.highlight_match(target, color)
        self._render()

    def clear_highlights(self) -> None:
        """Remove the ephemeral highlights added via :meth:`show`.

        Persistent tracks (from :meth:`load_track`/:meth:`show_track`) and
        the path highlight from :meth:`show_path` are left untouched.
        """
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

        Raises
        ------
        RuntimeError
            If the current path runs through nodes this widget has pruned
            from display, which happens by default. Replot with
            ``SequenceGraph.plot(show_history=True)`` to include them.
        """
        self._controller.show_path(color)
        self._render()

    def hide_path(self) -> None:
        """Remove path highlighting applied by :meth:`show_path`."""
        self._controller.hide_path()
        self._render()

    def refresh(self) -> None:
        """Force a re-render from the current controller state."""
        if self._frozen:
            return
        self._render()

    # ── Track API ─────────────────────────────────────────────────────────

    def load_track(
        self,
        file: str,
        *,
        name: str | None = None,
        from_sample: str | None = None,
        filter=None,
    ) -> None:
        """Load a GFF3 or BED file as a persistent annotation track.

        Parameters
        ----------
        file : str
            Path to a GFF3 or BED annotation file.  Both standard files
            (chromosome/contig names as reference) and pre-translated files
            (node hash-IDs as reference) are accepted; standard files are
            translated automatically.  *name* defaults to the file path.
        name : str, optional
            Display label for this annotation track.
        from_sample : str, optional
            Sample whose coordinate space the file uses.  Defaults to
            ``"reference"``.
        filter : callable, optional
            ``(row: str) -> bool`` predicate applied to each non-header line.
        """
        if self._frozen:
            return
        if filter is not None:
            file = self._apply_row_filter(file, filter)
        self._controller.add_track_file(file, name, from_sample)
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

    def show_track(self, name: str) -> None:
        """Load and display a DB-stored annotation group by name.

        See :attr:`tracks` for the full list of group names available to
        this widget's sequence graph.
        """
        if self._frozen:
            return
        self._controller.add_track_group(name)
        self._render()

    def hide_track(self, name: str) -> None:
        """Remove a displayed track, however it was shown (:meth:`load_track` or :meth:`show_track`)."""
        if self._frozen:
            return
        self._controller.remove_track(name)
        self._render()

    @property
    def tracks(self) -> list:
        """Every annotation-group name visible to this widget's sequence graph.

        This is the full menu of names that can be passed to
        :meth:`show_track` — its own group plus any inherited from ancestor
        samples — independent of which tracks are currently displayed.
        """
        return self._controller.track_names

    def hide_all_tracks(self) -> None:
        """Suppress every displayed track.

        This also suppresses annotations that auto-load from the database
        on first plot, giving a blank canvas to build tracks up from.
        """
        if self._frozen:
            return
        self._controller.clear_all_annotations()
        self._render()


async def freeze_all_widgets(timeout: float = 10.0, quiet: float = 1.0) -> None:
    """Freeze every constructed, not-yet-frozen ``GraphWidget`` that has a live view.

    ``ipywidgets`` already keeps a strong reference to every widget it has
    ever constructed (``Widget.widgets``), including any per-cell clones
    ``_ipython_display_`` creates for classic Jupyter — so no separate
    bookkeeping is needed to find them. Not every host renders through that
    clone-and-display path, though: marimo, for example, dispatches on
    ``isinstance(obj, anywidget.AnyWidget)`` directly rather than calling
    ``_ipython_display_``, so under marimo the object a user's own variable
    holds *is* the one with a live view, and no clone ever exists. This
    freezes every widget with an open comm rather than trying to detect
    "was actually displayed" per host.

    That means some widgets in the set (a Jupyter clone's own now-orphaned
    original, something constructed but never shown at all) may never
    confirm ``_frozen``, since they have no browser view to respond from.
    Rather than wait the full ``timeout`` for those every time, this stops
    ``quiet`` seconds after the last widget confirms — so a mixed batch
    still returns quickly once the real work is done, and only pays the
    full ``timeout`` when nothing responds at all (e.g. a headless kernel
    with no real browser view).

    Safe to call more than once, or repeatedly as new widgets get displayed
    across a session — each call only acts on the widgets not already frozen.

    Must be awaited from a cell (``await gen.freeze_all_widgets()``); a
    synchronous wrapper cannot work here because it would block the same
    event loop the frontend's response needs to be delivered on.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        widgets = [
            widget
            for widget in ipywidgets.Widget.widgets.values()
            if isinstance(widget, GraphWidget)
            and widget.comm is not None
            and not widget._frozen
        ]
    for widget in widgets:
        widget._freeze()

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    last_progress = loop.time()
    remaining = [widget for widget in widgets if not widget._frozen]
    while remaining:
        now = loop.time()
        if now > deadline or now - last_progress > quiet:
            break
        await asyncio.sleep(0.05)
        still_remaining = [widget for widget in remaining if not widget._frozen]
        if len(still_remaining) < len(remaining):
            last_progress = loop.time()
        remaining = still_remaining
