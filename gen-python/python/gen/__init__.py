"""Python bindings to the Gen version control system."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gen")
except PackageNotFoundError:
    __version__ = "0.0.0"


# Bindings can come through a Python intermediate layer (helpers.py) or the compiled Rust library itself

# Directly from Rust
from .gen import (
    Annotation,
    Asset,
    Branch,
    HashId,
    Locus,
    Node,
    NodeSlice,
    Operation,
    Position,
    Remote,
    Repository,
    Sample,
    SequenceGraph,
    SequencePart,
    clone,
)

# Jupyter widget — use a dependency-free textual fallback when the extra is absent.
try:
    from .jupyter_widget import GraphWidget, freeze_all_widgets
except ImportError:
    import warnings

    from .text_widget import TextGraphWidget as GraphWidget
    from .text_widget import _in_jupyter_kernel

    freeze_all_widgets = None

    if _in_jupyter_kernel():
        warnings.warn(
            "gen[jupyter] is not installed, so GraphWidget falls back to a plain-text "
            "renderer: no interactive canvas, highlighting, or annotation tracks. Run "
            "`pip install gen[jupyter]` and restart the kernel for the full widget.",
            stacklevel=2,
        )

__all__ = [
    "Annotation",
    "Asset",
    "Branch",
    "GraphWidget",
    "freeze_all_widgets",
    "HashId",
    "Locus",
    "Node",
    "NodeSlice",
    "Operation",
    "Position",
    "Remote",
    "Repository",
    "Sample",
    "SequenceGraph",
    "SequencePart",
    "clone",
]
