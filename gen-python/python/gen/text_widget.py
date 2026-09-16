"""Dependency-free textual fallback for the Gen graph viewer."""

from __future__ import annotations

import json

from .ascii_render import frame_text


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

# Shown everywhere else (plain scripts, terminals, AI agent tool calls): there
# is no interactive canvas to fall back to, so this orients a reader who has
# never seen a Gen graph widget before at reading the ASCII grid itself.
_TEXT_FALLBACK_HINT = (
    "# Gen graph textual output (the optional Jupyter widget is not installed).\n"
    "# This is an ASCII rendering of the native layout: each character is one\n"
    "# terminal cell from the Rust layout engine; UPPERCASE marks a highlighted\n"
    "# annotation region, lowercase is unhighlighted sequence/graph structure.\n"
    "# This object supports .zoom_in()/.zoom_out(), .scroll_left()/.scroll_right()/\n"
    "# .scroll_up()/.scroll_down(), .next_page()/.prev_page() for multi-page samples,\n"
    "# and .go_to(target)/.show(target) to jump to a Position/Locus/Annotation; call\n"
    "# refresh() or repr() again after any of these to see the update.\n"
    "# To install the interactive notebook widget instead, reinstall as gen[jupyter].\n"
)


# Public GraphWidget methods with no TextGraphWidget equivalent. Kept as a
# literal list rather than introspected from jupyter_widget.GraphWidget:
# importing that module here would itself raise ImportError (it requires
# anywidget/ipywidgets/traitlets), which is exactly the case this file exists
# to handle.
_GRAPHWIDGET_ONLY_METHODS = frozenset(
    {
        "handle_click",
        "highlight_match",
        "clear_highlights",
        "show_path",
        "clear_path",
        "add_annotation_track",
        "annotation_tracks",
        "remove_annotation_track",
        "clear_all_annotations",
        "add_annotation",
        "annotations",
        "list_annotations",
        "remove_annotation",
    }
)


class TextGraphWidget:
    """Render a Gen graph as plain text when Jupyter extras are unavailable.

    The native controller still owns layout and graph state, so this fallback
    remains useful in notebooks, terminals, and environments used by AI agents.
    """

    def __init__(self, controller, *, colors=None, **kwargs):
        self._controller = controller
        self.cols = kwargs.get("cols", 60)
        self.rows = kwargs.get("rows", 12)
        self.page_count = controller.page_count
        self.page_index = 0
        self.frame = {}
        self._load_initial_annotations(colors)
        self._render()

    def __repr__(self) -> str:
        """Return the current native render frame as readable plain text."""
        hint = _INSTALL_HINT if _in_jupyter_kernel() else _TEXT_FALLBACK_HINT
        return hint + frame_text(self.frame, self.page_count, self.page_index)

    def _load_initial_annotations(self, colors) -> None:
        """Apply the same annotation loading policy as the interactive widget."""
        if self._controller.annotations_loaded:
            return
        if colors is None:
            self._controller.trigger_auto_load()
            return

        if callable(colors):
            color_fn = colors
        elif isinstance(colors, dict):

            def color_fn(annotation):
                return colors.get(annotation.name)

        elif isinstance(colors, list):
            if not colors:
                raise ValueError("colors list must not be empty")
            assigned = {}

            def color_fn(annotation):
                if annotation.id not in assigned:
                    assigned[annotation.id] = colors[len(assigned) % len(colors)]
                return assigned[annotation.id]
        else:
            raise TypeError(
                f"colors must be a callable, dict, or list; got {type(colors).__name__}"
            )

        color_map = {
            annotation.id: color_fn(annotation)
            for annotation in self._controller.list_annotations()
        }
        self._controller.load_annotation_groups_with_colors(color_map)

    def _render(self) -> None:
        """Refresh the text frame from the native controller."""
        self.frame = json.loads(self._controller.render_frame(self.cols, self.rows))
        self.page_index = self._controller.page_index

    def zoom_in(self) -> None:
        """Step one zoom level in and refresh the text output."""
        self._controller.zoom_in()
        self._render()

    def zoom_out(self) -> None:
        """Step one zoom level out and refresh the text output."""
        self._controller.zoom_out()
        self._render()

    def scroll_right(self) -> None:
        """Scroll the view right by one screenful."""
        self._controller.move_by(-self.cols, 0)
        self._render()

    def scroll_left(self) -> None:
        """Scroll the view left by one screenful."""
        self._controller.move_by(self.cols, 0)
        self._render()

    def scroll_down(self) -> None:
        """Scroll the view down by one screenful."""
        self._controller.move_by(0, -self.rows)
        self._render()

    def scroll_up(self) -> None:
        """Scroll the view up by one screenful."""
        self._controller.move_by(0, self.rows)
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
        """
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
        """
        from gen import Annotation  # noqa: PLC0415

        if isinstance(target, Annotation):
            self._controller.go_to_annotation_obj(target, center)
            self._controller.highlight_annotation_obj(target, color)
        else:
            self._controller.go_to_pos(target.start(), center)
            self._controller.highlight_match(target, color)
        self._render()

    def next_page(self) -> None:
        """Advance to the next sequence graph and refresh the text output."""
        self._controller.next_page()
        self._render()

    def prev_page(self) -> None:
        """Return to the previous sequence graph and refresh the text output."""
        self._controller.prev_page()
        self._render()

    def refresh(self) -> None:
        """Refresh the text output from the current controller state."""
        self._render()

    def __getattr__(self, name: str):
        """Point a call to a GraphWidget-only method at the install fix.

        Only names that actually exist on GraphWidget (handle_click,
        highlight_match, clear_highlights, show_path, clear_path, and the
        annotation-track methods) get the install hint; anything else falls
        through to Python's normal AttributeError so a typo still reads as a
        typo.
        """
        if name in _GRAPHWIDGET_ONLY_METHODS:
            raise AttributeError("install gen[jupyter] to get the full suite of functions")
        raise AttributeError(f"'TextGraphWidget' object has no attribute {name!r}")
