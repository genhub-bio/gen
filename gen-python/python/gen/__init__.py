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
    AnnotationOffset,
    BlockGroup,
    HashId,
    Node,
    NodeSlice,
    GraphPos,
    GraphLocus,
    SequencePart,
    Repository,
)

# Jupyter widget — only available with `pip install gen[jupyter]`
try:
    from .jupyter_widget import GenGraphWidget
except ImportError:
    GenGraphWidget = None

__all__ = [
    "Annotation",
    "AnnotationOffset",
    "BlockGroup",
    "GenGraphWidget",
    "GraphLocus",
    "GraphPos",
    "HashId",
    "Node",
    "NodeSlice",
    "Repository",
    "SequencePart",
]
