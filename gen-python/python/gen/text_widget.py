"""Dependency-free textual fallback for the Gen graph viewer."""

from __future__ import annotations

import json
import os
from typing import Self

from .ascii_render import frame_text
from .png_image import PngImage


def _in_jupyter_kernel() -> bool:
    """Return whether this process is running inside a live Jupyter kernel."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ipython = get_ipython()
    return ipython is not None and ipython.__class__.__name__ == "ZMQInteractiveShell"


# Shown to a notebook user who has a live kernel but installed gen without the
# jupyter extra: the fix is one command, so the hint stays short.
_INSTALL_HINT = (
    "# Gen graph textual output (the optional Jupyter widget is not installed).\n"
    "# To include the notebook widget, please reinstall as gen[jupyter].\n"
)

# Shown everywhere outside a live Jupyter kernel (plain scripts, terminals, AI
# agent tool calls). The Jupyter extra may be installed, but no browser canvas
# can be displayed in this session, so orient a reader to the ASCII grid.
_TEXT_FALLBACK_HINT = (
    "# AI agents: Gen SequenceGraph textual output below. It represents a population or\n"
    "# library of sequences: read it left to right, where nodes are sequence fragments\n"
    "# and solid edges are the ways fragments can be joined into full paths. A node\n"
    "# shows its complete sequence, a truncated one with an ellipsis where sequence is\n"
    "# omitted, or (most minimal view) just a filled circle. For an image file instead\n"
    '# of text, call widget.save_png("plot.png"). Omit these # lines when showing the\n'
    "# graph to a user; paste the frame in a code fence.\n"
)

# Text fallback output can occur many times in one notebook or agent session.
# Show the orientation once, when the first frame is actually displayed, then
# keep later frames compact. A module global naturally lasts for the current
# Python process (and therefore the current Jupyter kernel).
_text_fallback_hint_shown = False


def _text_fallback_hint() -> str:
    """Return the one-time explanation for the first displayed text frame."""
    global _text_fallback_hint_shown

    # A notebook frozen with text plots chose text on purpose, so a hint to install the
    # widget would only mislead its readers.
    if os.environ.get("GEN_NOTEBOOK_TEXT_OUTPUT") == "1" or _text_fallback_hint_shown:
        return ""
    _text_fallback_hint_shown = True
    return _INSTALL_HINT if _in_jupyter_kernel() else _TEXT_FALLBACK_HINT


class TextGraphWidget:
    """Render a Gen graph as plain text when no live Jupyter canvas is available.

    The native controller still owns layout and graph state, so this fallback
    remains useful in notebooks, terminals, and environments used by AI agents.
    """

    def __init__(self, controller, *, colors=None, sequence_graphs=(), **kwargs):
        self._controller = controller
        self._sequence_graphs = sequence_graphs
        self._colors = colors
        self.cols = kwargs.get("cols", 60)
        self.rows = kwargs.get("rows", 12)
        # Image pixels per logical pixel in the PNGs this widget writes.
        self.scale = kwargs.get("scale", 2)
        self.page_count = controller.page_count
        self.page_index = 0
        self.frame = {}
        self._load_initial_annotations()
        self._render()

    def __repr__(self) -> str:
        """Return the current native render frame as readable plain text."""
        return _text_fallback_hint() + frame_text(
            self.frame, self.page_count, self.page_index
        )

    def _load_initial_annotations(self) -> None:
        """Apply the same annotation loading policy as the interactive widget."""
        if self._controller.annotations_loaded:
            return
        if self._colors is None:
            self._controller.trigger_auto_load()
            return

        if callable(self._colors):
            color_fn = self._colors
        elif isinstance(self._colors, dict):

            def color_fn(annotation):
                return self._colors.get(annotation.name)

        elif isinstance(self._colors, list):
            if not self._colors:
                raise ValueError("colors list must not be empty")
            assigned = {}

            def color_fn(annotation):
                key = str(annotation.id)
                if key not in assigned:
                    assigned[key] = self._colors[len(assigned) % len(self._colors)]
                return assigned[key]
        else:
            raise TypeError(
                f"colors must be a callable, dict, or list; got {type(self._colors).__name__}"
            )

        color_map = {
            str(annotation.id): color_fn(annotation)
            for annotation in self._sequence_graphs[
                self._controller.page_index
            ].annotations
        }
        self._controller.load_annotation_groups_with_colors(color_map)

    def _render(self) -> None:
        """Refresh the text frame from the native controller."""
        self.frame = json.loads(self._controller.render_frame(self.cols, self.rows))
        self.page_index = self._controller.page_index

    def zoom_in(self) -> Self:
        """Step one zoom level in and refresh the text output."""
        self._controller.zoom_in()
        self._render()
        return self

    def zoom_out(self) -> Self:
        """Step one zoom level out and refresh the text output."""
        self._controller.zoom_out()
        self._render()
        return self

    def show_labels(self) -> Self:
        """Show annotation names in the normal-zoom view.

        Plots open with names hidden, so annotations draw as bars and long names don't
        stretch the nodes. Zooming in once also shows them, and the view with the full
        sequence always names every annotation."""
        self._controller.set_labels_shown(True)
        self._render()
        return self

    def hide_labels(self) -> Self:
        """Hide annotation names in the normal-zoom view, leaving only the bars."""
        self._controller.set_labels_shown(False)
        self._render()
        return self

    def scroll_right(self) -> Self:
        """Scroll the view right by one screenful."""
        self._controller.move_by(-self.cols, 0)
        self._render()
        return self

    def scroll_left(self) -> Self:
        """Scroll the view left by one screenful."""
        self._controller.move_by(self.cols, 0)
        self._render()
        return self

    def scroll_down(self) -> Self:
        """Scroll the view down by one screenful."""
        self._controller.move_by(0, -self.rows)
        self._render()
        return self

    def scroll_up(self) -> Self:
        """Scroll the view up by one screenful."""
        self._controller.move_by(0, self.rows)
        self._render()
        return self

    def go_to(self, target, *, center: bool = False) -> Self:
        """Instantly move the camera to a graph position, locus, or annotation.

        The cursor is drawn on the target; call :meth:`hide_cursor` to leave it out of
        saved images.

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
        return self

    def show(self, target, color: str | None = None, *, center: bool = False) -> Self:
        """Navigate to and highlight a graph locus or annotation in one call.

        Parameters
        ----------
        target:
            A ``Locus`` returned by ``repo.search()``, an ``Annotation``
            object (e.g. from ``sequence_graph.annotations``), or a
            ``Position``/``SuperPosition`` (highlights each named base and
            navigates to the leftmost position).
        color:
            Optional highlight colour.  Accepts named colours
            (``"yellow"``, ``"cyan"``, ``"red"``, …) or a CSS hex string
            (``"#ff8800"``).  When omitted the theme's highlight accent
            is used, the same colour on every call.
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
        from gen import Annotation, Position, SuperPosition  # noqa: PLC0415

        if isinstance(target, Annotation):
            self._controller.go_to_annotation_obj(target, center)
            self._controller.highlight_annotation_obj(target, color)
        elif isinstance(target, SuperPosition):
            self._controller.show_positions(target.positions, color, center)
        elif isinstance(target, Position):
            self._controller.show_positions([target], color, center)
        else:
            self._controller.go_to_locus(target, center)
            self._controller.highlight_match(target, color)
        self._controller.set_cursor_hidden(True)
        self._render()
        return self

    def next_page(self) -> Self:
        """Advance to the next sequence graph and refresh the text output."""
        self._controller.next_page()
        self._load_initial_annotations()
        self._render()
        return self

    def prev_page(self) -> Self:
        """Return to the previous sequence graph and refresh the text output."""
        self._controller.prev_page()
        self._load_initial_annotations()
        self._render()
        return self

    def _render_png(
        self,
        *,
        cols: int | None = None,
        rows: int | None = None,
        scale: int | None = None,
    ) -> PngImage:
        scale = self.scale if scale is None else scale
        data = self._controller.render_png(
            self.cols if cols is None else cols,
            self.rows if rows is None else rows,
            scale,
        )
        return PngImage(data, scale)

    def save_png(
        self,
        path: str | os.PathLike,
        *,
        cols: int | None = None,
        rows: int | None = None,
        scale: int | None = None,
    ) -> None:
        """Write the current view of the graph to a PNG file.

        Needs no Jupyter dependencies. Each terminal cell becomes 11x21 pixels times
        ``scale``, and blank rows and columns around the graph are trimmed, leaving a
        one-cell margin.

        Parameters
        ----------
        path:
            Destination file.
        cols, rows:
            Size of the render canvas in terminal cells before trimming. Each omitted
            dimension uses the widget's current viewport size. Specify larger values
            to capture more of the graph; blank space is trimmed afterward.
        scale:
            Image pixels per logical pixel, from 1 to 8. A scale of 2 writes the same
            picture on twice as many pixels. Defaults to the widget's scale.
        """
        image = self._render_png(cols=cols, rows=rows, scale=scale)
        with open(path, "wb") as handle:
            handle.write(image.data)

    def refresh(self) -> Self:
        """Refresh the text output from the current controller state."""
        self._render()
        return self

    def handle_click(self, col: int, row: int) -> bool:
        """Click the text cell at ``(col, row)`` and re-render. Returns True if a node was hit."""
        hit = self._controller.handle_click(col, row)
        self._render()
        return hit

    def clear_highlights(self) -> Self:
        """Remove the ephemeral highlights added via :meth:`show`.

        Persistent tracks (from :meth:`show_track`) and the path highlight from
        :meth:`show_path` are left untouched.
        """
        self._controller.clear_highlights()
        self._render()
        return self

    def hide_cursor(self) -> Self:
        """Keep the cursor out of the view, for a clean image, until :meth:`show_cursor`."""
        self._controller.set_cursor_hidden(True)
        self._render()
        return self

    def show_cursor(self) -> Self:
        """Draw the cursor again after :meth:`hide_cursor`."""
        self._controller.set_cursor_hidden(False)
        self._render()
        return self

    def show_path(self, color: str | None = None) -> Self:
        """Highlight the most recent path for this sequence graph.

        Raises ``RuntimeError`` if the path cannot be traced through the plotted
        graph. A path copied before later edits runs through nodes that the
        default view prunes; plot with ``show_history=True`` to keep them.
        """
        self._controller.show_path(color)
        self._render()
        return self

    def hide_path(self) -> Self:
        """Remove path highlighting applied by :meth:`show_path`."""
        self._controller.hide_path()
        self._render()
        return self

    def show_track(self, name: str) -> Self:
        """Load and display a DB-stored annotation group by name (see :attr:`tracks`)."""
        self._controller.add_track_group(name)
        self._render()
        return self

    def hide_track(self, name: str) -> Self:
        """Remove a displayed annotation track."""
        self._controller.remove_track(name)
        self._render()
        return self

    @property
    def tracks(self) -> list:
        """Every annotation-group name that can be passed to :meth:`show_track`."""
        return self._controller.track_names

    def hide_all_tracks(self) -> Self:
        """Suppress every displayed track, including annotations auto-loaded on first plot."""
        self._controller.clear_all_annotations()
        self._render()
        return self
