import pytest

from gen import Block, EndBlock, GenGraph, StartBlock


def test_add_node_returns_block():
    g = GenGraph()
    b = Block("ACGT")
    result = g.add_node(b)
    assert result is b
    assert b in g.nodes()


def test_add_node_rejects_non_block():
    g = GenGraph()
    with pytest.raises(TypeError):
        g.add_node("not-a-block")


def test_add_edge_auto_adds_nodes():
    g = GenGraph()
    a = Block("A")
    c = Block("C")
    g.add_edge(a, c)
    assert a in g.nodes()
    assert c in g.nodes()


def test_add_edge_with_start_block():
    g = GenGraph()
    b = Block("ACGT")
    g.add_edge(StartBlock(), b)
    assert StartBlock() in g.nodes()
    assert b in g.nodes()


def test_add_edge_with_end_block():
    g = GenGraph()
    b = Block("ACGT")
    g.add_edge(b, EndBlock())
    assert b in g.nodes()
    assert EndBlock() in g.nodes()


def test_validate_passes_well_formed():
    g = GenGraph()
    b = Block("ACGT")
    g.add_edge(StartBlock(), b)
    g.add_edge(b, EndBlock())
    g.validate()


def test_validate_raises_empty_sequence():
    g = GenGraph()
    b1 = Block("ACGT")
    b2 = Block("TTTT")
    g.add_edge(b1, b2)
    # Manually inject a bad node
    g._node[b1]["sequence"] = ""
    b1_bad = Block("")
    super(GenGraph, g).add_node(b1_bad)
    with pytest.raises((ValueError, TypeError)):
        g.validate()


def test_start_block_equality():
    assert StartBlock() == StartBlock()


def test_end_block_equality():
    assert EndBlock() == EndBlock()


def test_two_blocks_same_seq_are_distinct():
    g = GenGraph()
    b1 = Block("ACGT")
    b2 = Block("ACGT")
    g.add_node(b1)
    g.add_node(b2)
    assert b1 != b2
    assert len(g.nodes()) == 2


def test_node_id_sequence_collision_raises():
    import tempfile, gen
    tmp = tempfile.TemporaryDirectory()
    repo = gen.Repository(tmp.name + '/.gen')
    bg = repo.create_block_group_from_sequence(name='orig', sequence='ACGT')
    original_node = next(
        n for n in bg.to_networkx(include_sentinels=False).nodes()
    )
    g = GenGraph()
    g.add_node(Block('TTTT', node_id=original_node.node_id))
    with pytest.raises(RuntimeError, match='already exists.*different sequence'):
        repo.create_block_group_from_graph(g, name='conflict')


def test_add_edge_rejects_invalid_strand():
    g = GenGraph()
    a = Block("ACGT")
    b = Block("TTTT")
    with pytest.raises(ValueError):
        g.add_edge(a, b, source_strand="bad")
