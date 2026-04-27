import networkx as nx

from .gen import Block, EndBlock, StartBlock

VALID_STRANDS = {"+", "-", "forward", "reverse", "Forward", "Reverse"}


class GenGraph(nx.DiGraph):
    """A NetworkX DiGraph that enforces Block-typed nodes."""

    def add_node(self, block, **attrs):
        if not isinstance(block, Block):
            raise TypeError(f"Nodes must be Block instances, got {type(block)!r}")
        node_attrs = {
            "node_id": str(block.node_id),
            "sequence": block._node_sequence,
            "sequence_start": block.sequence_start,
            "sequence_end": block.sequence_end,
        }
        node_attrs.update(attrs)
        super().add_node(block, **node_attrs)
        return block

    def add_edge(self, src, dst, source_strand="+", target_strand="+", **attrs):
        if not isinstance(src, Block):
            raise TypeError(f"Edge source must be a Block, got {type(src)!r}")
        if not isinstance(dst, Block):
            raise TypeError(f"Edge target must be a Block, got {type(dst)!r}")
        if source_strand not in VALID_STRANDS:
            raise ValueError(f"Invalid source_strand {source_strand!r}")
        if target_strand not in VALID_STRANDS:
            raise ValueError(f"Invalid target_strand {target_strand!r}")
        if src not in self:
            self.add_node(src)
        if dst not in self:
            self.add_node(dst)
        super().add_edge(src, dst, source_strand=source_strand,
                         target_strand=target_strand, **attrs)

    def validate(self):
        """Raise TypeError/ValueError for any invariant violation."""
        for node in self.nodes():
            if not isinstance(node, Block):
                raise TypeError(f"Non-Block node found: {node!r}")
            if not isinstance(node, (StartBlock, EndBlock)) and not node._node_sequence:
                raise ValueError(f"Block has empty sequence: {node!r}")
        for src, dst, edge_attrs in self.edges(data=True):
            ss = edge_attrs.get("source_strand", "+")
            ts = edge_attrs.get("target_strand", "+")
            if ss not in VALID_STRANDS:
                raise ValueError(
                    f"Invalid source_strand {ss!r} on edge {src!r} -> {dst!r}"
                )
            if ts not in VALID_STRANDS:
                raise ValueError(
                    f"Invalid target_strand {ts!r} on edge {src!r} -> {dst!r}"
                )
