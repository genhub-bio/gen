"""Dependency-free rendering of a native RenderedFrame as ASCII/Unicode text.

Shared by ``jupyter_widget.GraphWidget.__repr__`` (the ``text/plain``
fallback used whenever the widget JS can't run) and
``text_widget.TextGraphWidget`` (used when the ``jupyter`` extra isn't
installed at all). Neither caller's import requirements apply here: this
module must stay free of anywidget/ipywidgets/traitlets imports.
"""

from __future__ import annotations


def highlighted(cell: dict, frame: dict) -> bool:
    """True when the cell carries a highlight colour.

    The graph renderer paints text cells with inverted neutral colours
    (fg=neutral_bg, bg=neutral_fg) by default. A true highlight changes
    the cell's *bg* to an accent colour while keeping the inverted fg.
    Edge cells never set *bg* (highlighted edges are already drawn with
    heavy/dashed box-drawing glyphs by the Rust renderer), so this only
    ever fires for text cells.
    """
    bg = cell.get("bg")
    if bg is None:
        return False
    return bg != frame.get("neutral_fg")


def transform(text: str, is_highlighted: bool) -> str:
    """Adjust casing based on highlight state."""
    return text.upper() if is_highlighted else text.lower()


def frame_text(frame: dict, page_count: int, page_index: int) -> str:
    """Convert a sparse native render frame into its terminal-cell text."""
    cols = frame.get("cols", 0)
    rows = frame.get("rows", 0)
    grid = [[" "] * cols for _ in range(rows)]
    for cell in frame.get("cells", []):
        x, y, text = cell["x"], cell["y"], cell["text"]
        if text and 0 <= y < rows and 0 <= x < cols:
            grid[y][x] = transform(text, highlighted(cell, frame))
    lines = ["".join(row).rstrip() for row in grid]
    if page_count > 1 and lines:
        prefix = f"[{page_index + 1}/{page_count}] "
        lines[0] = (prefix + lines[0].lstrip()).rstrip()
    return "\n".join(lines)
