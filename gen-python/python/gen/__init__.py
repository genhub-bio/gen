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
    Branch,
    HashId,
    Locus,
    Node,
    NodeSlice,
    Operation,
    Position,
    Repository,
    Sample,
    SequenceGraph,
    SequencePart,
    clone,
)

# Jupyter widget — only available with `pip install gen[jupyter]`
try:
    from .jupyter_widget import GraphWidget, freeze_all_widgets
except ImportError:
    GraphWidget = None
    freeze_all_widgets = None

__all__ = [
    "Annotation",
    "Branch",
    "GraphWidget",
    "freeze_all_widgets",
    "HashId",
    "Locus",
    "Node",
    "NodeSlice",
    "Operation",
    "Position",
    "Repository",
    "Sample",
    "SequenceGraph",
    "SequencePart",
    "clone",
]
