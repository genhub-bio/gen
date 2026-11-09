"""PNG image result that notebooks can display without an IPython dependency."""

from __future__ import annotations

import struct

_PNG_HEADER_SIZE = 24


class PngImage:
    """Hold rendered PNG bytes for notebook display.

    ``scale`` is how many image pixels draw one logical pixel. A notebook shows a scaled image
    at its logical size, so it stays sharp on high-density screens instead of growing.
    """

    def __init__(self, data: bytes, scale: int = 1):
        self.data = data
        self.scale = scale

    def _repr_png_(self) -> bytes | tuple[bytes, dict[str, int]]:
        """Provide the image to IPython's PNG display formatter."""
        if self.scale <= 1 or len(self.data) < _PNG_HEADER_SIZE:
            return self.data
        width, height = struct.unpack(">II", self.data[16:_PNG_HEADER_SIZE])
        return self.data, {"width": width // self.scale, "height": height // self.scale}

    def __repr__(self) -> str:
        return f"PngImage({len(self.data)} bytes)"
