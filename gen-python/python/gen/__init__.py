"""Python bindings to the Gen version control system."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("gen")
except _PackageNotFoundError:
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
    Operation,
    Position,
    Remote,
    Repository,
    Sample,
    Sequence,
    SequenceGraph,
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

# Widgets are only ever returned by plot(), so they are not part of the public namespace.
__all__ = [
    "Annotation",
    "Asset",
    "Branch",
    "HashId",
    "Locus",
    "Node",
    "Operation",
    "Position",
    "Remote",
    "Repository",
    "Sample",
    "Sequence",
    "SequenceGraph",
    "SuperPosition",
    "clone",
    "freeze_all_widgets",
]


def __dir__() -> list[str]:
    # Keep submodules and helper imports out of dir(gen) and tab completion.
    return sorted(__all__)
