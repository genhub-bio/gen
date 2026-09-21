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
    SuperPosition,
    clone,
)

# Only a live Jupyter kernel can display the browser canvas. Route terminals,
# scripts, and AI REPLs to the text widget even when the optional dependencies
# happen to be installed.
from .text_widget import TextGraphWidget, _in_jupyter_kernel

if _in_jupyter_kernel():
    try:
        from .jupyter_widget import GraphWidget, freeze_all_widgets
    except ImportError:
        GraphWidget = TextGraphWidget
        freeze_all_widgets = None
else:
    GraphWidget = TextGraphWidget
    freeze_all_widgets = None

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
    "SuperPosition",
    "clone",
]
