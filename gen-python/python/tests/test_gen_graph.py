import pytest

from gen import Block, EndBlock, StartBlock


def test_start_block_equality():
    assert StartBlock() == StartBlock()


def test_end_block_equality():
    assert EndBlock() == EndBlock()


def test_two_blocks_same_seq_are_distinct():
    import networkx as nx
    b1 = Block("ACGT")
    b2 = Block("ACGT")
    g = nx.DiGraph()
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
        n for n in bg.to_networkx().nodes()
        if not isinstance(n, (StartBlock, EndBlock))
    )
    import networkx as nx
    g = nx.DiGraph()
    g.add_node(Block('TTTT', node_id=original_node.node_id))
    with pytest.raises(RuntimeError, match='already exists.*different sequence'):
        repo.create_block_group_from_graph(g, name='conflict')
