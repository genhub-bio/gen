"""Render Cre-loxP style rearrangements to PNG to check how edge sides are routed.

Each scenario cuts one 9 bp molecule into left | segment | right and joins the pieces with
`SequenceGraph.connect`. Joins that read a segment on the reverse strand attach to the opposite
side of a node, so they exercise the pins: east to east (right pin), west to west (left pin),
west to east (both pins), a reversed plain edge, and an east-to-west edge against the order.
"""

from __future__ import annotations

import argparse
import pathlib
import tempfile
from collections.abc import Callable

import gen

SEQUENCE = "AAACCGTTT"
DETAILS = ("minimal", "normal", "full")


class Sites:
    """The bases next to the two cut points (3 and 6) and the molecule's two ends."""

    def __init__(self, graph: gen.SequenceGraph) -> None:
        self.first = graph.locus(0, 1).start()
        self.left_last = graph.locus(2, 3).start()
        self.segment_first = graph.locus(3, 4).start()
        self.segment_last = graph.locus(5, 6).start()
        self.right_first = graph.locus(6, 7).start()
        self.last = graph.locus(8, 9).start()


def linear(graph: gen.SequenceGraph, sites: Sites) -> None:
    """No edits: every edge runs east to west."""


def inversion(graph: gen.SequenceGraph, sites: Sites) -> None:
    """Invert the segment: left east to segment east, then segment west to right west."""
    graph.connect(after=sites.left_last, before=sites.segment_last.reverse_complement())
    graph.connect(after=sites.segment_first.reverse_complement(), before=sites.right_first)


def stacked_inversion(graph: gen.SequenceGraph, sites: Sites) -> None:
    """The inversion next to the original molecule."""
    graph.connect(
        after=sites.left_last,
        before=sites.segment_last.reverse_complement(),
        stack=True,
    )
    graph.connect(
        after=sites.segment_first.reverse_complement(),
        before=sites.right_first,
        stack=True,
    )


def excision(graph: gen.SequenceGraph, sites: Sites) -> None:
    """Excise the segment into a circle: a skip edge and a segment self-loop."""
    graph.connect(after=sites.left_last, before=sites.right_first)
    graph.connect(after=sites.segment_last, before=sites.segment_first)


def circle(graph: gen.SequenceGraph, sites: Sites) -> None:
    """Close the molecule: the last base joins the first, an east-to-west edge against the order."""
    graph.connect(after=sites.last, before=sites.first, stack=True)


def head_to_head(graph: gen.SequenceGraph, sites: Sites) -> None:
    """Left east to segment east only: both ends on the east side, one right pin."""
    graph.connect(
        after=sites.left_last,
        before=sites.segment_last.reverse_complement(),
        stack=True,
    )


def tail_to_tail(graph: gen.SequenceGraph, sites: Sites) -> None:
    """Segment west to right west only: both ends on the west side, one left pin."""
    graph.connect(
        after=sites.segment_first.reverse_complement(),
        before=sites.right_first,
        stack=True,
    )


def west_to_east_forward(graph: gen.SequenceGraph, sites: Sites) -> None:
    """Segment west to right east: the target follows the source, so both pins are needed."""
    graph.connect(
        after=sites.segment_first.reverse_complement(),
        before=sites.last.reverse_complement(),
        stack=True,
    )


def west_to_east_reversed(graph: gen.SequenceGraph, sites: Sites) -> None:
    """Right west to segment east: the target comes first, a plain edge read backwards."""
    graph.connect(
        after=sites.right_first.reverse_complement(),
        before=sites.segment_last.reverse_complement(),
        stack=True,
    )


SCENARIOS: dict[str, Callable[[gen.SequenceGraph, Sites], None]] = {
    "linear": linear,
    "inversion": inversion,
    "stacked_inversion": stacked_inversion,
    "excision": excision,
    "circle": circle,
    "head_to_head": head_to_head,
    "tail_to_tail": tail_to_tail,
    "west_to_east_forward": west_to_east_forward,
    "west_to_east_reversed": west_to_east_reversed,
}


# Scenarios whose interesting part is a separate component, so the view opens on it instead.
FOCUS: dict[str, Callable[[Sites], gen.Position]] = {
    "excision": lambda sites: sites.segment_first,
}


def render(name: str, output_dir: pathlib.Path, details: tuple[str, ...]) -> list[pathlib.Path]:
    """Build one scenario in a fresh repository and save one PNG per detail level."""
    written = []
    with tempfile.TemporaryDirectory(prefix=f"gen-pins-{name}-") as work_dir:
        repository = gen.Repository(work_dir)
        graph = repository.import_sequence(SEQUENCE, name="molecule", sample="cre")
        sites = Sites(graph)
        # Stacking the original joins cuts the molecule into its three blocks, so every
        # scenario draws the same nodes whichever points its own joins name.
        graph.connect(after=sites.left_last, before=sites.segment_first, stack=True)
        graph.connect(after=sites.segment_last, before=sites.right_first, stack=True)
        SCENARIOS[name](graph, sites)
        for detail in details:
            path = output_dir / f"{name}_{detail}.png"
            focus = FOCUS.get(name)
            widget = graph.plot(detail=detail, go_to=focus(sites) if focus else None)
            widget.save_png(str(path))
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*", choices=[*SCENARIOS, []], default=[])
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("pin_routing"))
    parser.add_argument("--detail", action="append", choices=DETAILS)
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    details = tuple(arguments.detail or DETAILS)
    for name in arguments.scenarios or SCENARIOS:
        for path in render(name, arguments.output_dir, details):
            print(path)


if __name__ == "__main__":
    main()
