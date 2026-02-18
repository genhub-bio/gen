use petgraph::stable_graph::{NodeIndex, StableDiGraph};

use super::{Edge, Vertex, reset_alignment};
use crate::p3_calculate_coordinates::{create_vertical_alignments, mark_type_1_conflicts};

fn create_test_layout() -> (StableDiGraph<Vertex, Edge>, Vec<Vec<NodeIndex>>) {
    let edges: [(u32, u32); 30] = [
        (0, 2),
        (0, 6),
        (0, 18),
        (1, 16),
        (1, 17),
        (3, 8),
        (16, 8),
        (4, 8),
        (17, 19),
        (18, 20),
        (5, 8),
        (5, 9),
        (6, 8),
        (6, 21),
        (7, 10),
        (7, 11),
        (7, 12),
        (19, 23),
        (20, 24),
        (21, 12),
        (9, 22),
        (9, 25),
        (10, 13),
        (10, 14),
        (11, 14),
        (22, 13),
        (23, 15),
        (24, 15),
        (12, 15),
        (25, 15),
    ];

    let mut graph = StableDiGraph::<Vertex, Edge>::from_edges(edges);
    let layers: Vec<Vec<NodeIndex>> = [
        vec![0, 1],
        vec![2, 3, 16, 4, 17, 18, 5, 6],
        vec![7, 8, 19, 20, 21, 9],
        vec![10, 11, 22, 23, 24, 12, 25],
        vec![13, 14, 15],
    ]
    .into_iter()
    .map(|row| row.into_iter().map(|id| id.into()).collect())
    .collect();

    for (rank, row) in layers.iter().enumerate() {
        for (pos, v) in row.iter().enumerate() {
            let weight = &mut graph[*v];
            *weight = Vertex {
                rank: rank as _,
                pos,
                root: *v,
                align: *v,
                sink: *v,
                is_dummy: v.index() >= 16,
                size: if v.index() >= 16 {
                    (1.0, 1.0)
                } else {
                    (10.0, 10.0)
                },
                ..Default::default()
            };
        }
    }

    (graph, layers)
}

#[test]
fn type_1() {
    let (mut graph, l) = create_test_layout();
    mark_type_1_conflicts(&mut graph, &l);
    assert!(graph[graph.find_edge(6.into(), 8.into()).unwrap()].has_type_1_conflict);
    assert!(graph[graph.find_edge(7.into(), 12.into()).unwrap()].has_type_1_conflict);
    assert!(graph[graph.find_edge(5.into(), 8.into()).unwrap()].has_type_1_conflict);
    assert!(graph[graph.find_edge(9.into(), 22.into()).unwrap()].has_type_1_conflict);
}

#[test]
fn alignment_down_right() {
    let (mut g, mut l) = create_test_layout();
    mark_type_1_conflicts(&mut g, &l);

    // down right means no rotation
    reset_alignment(&mut g, &l);
    create_vertical_alignments(&mut g, &mut l);
    // verify roots
    assert_eq!(g[NodeIndex::from(0)].root, 0.into());
    assert_eq!(g[NodeIndex::from(1)].root, 1.into());
    assert_eq!(g[NodeIndex::from(2)].root, 0.into());
    assert_eq!(g[NodeIndex::from(3)].root, 3.into());
    assert_eq!(g[NodeIndex::from(4)].root, 4.into());
    assert_eq!(g[NodeIndex::from(5)].root, 5.into());
    assert_eq!(g[NodeIndex::from(6)].root, 6.into());
    assert_eq!(g[NodeIndex::from(7)].root, 7.into());
    assert_eq!(g[NodeIndex::from(8)].root, 4.into());
    assert_eq!(g[NodeIndex::from(9)].root, 9.into());
    assert_eq!(g[NodeIndex::from(10)].root, 7.into());
    assert_eq!(g[NodeIndex::from(11)].root, 11.into());
    assert_eq!(g[NodeIndex::from(12)].root, 6.into());
    assert_eq!(g[NodeIndex::from(13)].root, 7.into());
    assert_eq!(g[NodeIndex::from(14)].root, 11.into());
    assert_eq!(g[NodeIndex::from(15)].root, 18.into());
    assert_eq!(g[NodeIndex::from(16)].root, 1.into());
    assert_eq!(g[NodeIndex::from(17)].root, 17.into());
    assert_eq!(g[NodeIndex::from(18)].root, 18.into());
    assert_eq!(g[NodeIndex::from(19)].root, 17.into());
    assert_eq!(g[NodeIndex::from(20)].root, 18.into());
    assert_eq!(g[NodeIndex::from(21)].root, 6.into());
    assert_eq!(g[NodeIndex::from(22)].root, 22.into());
    assert_eq!(g[NodeIndex::from(23)].root, 17.into());
    assert_eq!(g[NodeIndex::from(24)].root, 18.into());
    assert_eq!(g[NodeIndex::from(25)].root, 9.into());

    // verify alignments
    assert_eq!(g[NodeIndex::from(0)].align, 2.into());
    assert_eq!(g[NodeIndex::from(1)].align, 16.into());
    assert_eq!(g[NodeIndex::from(2)].align, 0.into());
    assert_eq!(g[NodeIndex::from(3)].align, 3.into());
    assert_eq!(g[NodeIndex::from(4)].align, 8.into());
    assert_eq!(g[NodeIndex::from(5)].align, 5.into());
    assert_eq!(g[NodeIndex::from(6)].align, 21.into());
    assert_eq!(g[NodeIndex::from(7)].align, 10.into());
    assert_eq!(g[NodeIndex::from(8)].align, 4.into());
    assert_eq!(g[NodeIndex::from(9)].align, 25.into());
    assert_eq!(g[NodeIndex::from(10)].align, 13.into());
    assert_eq!(g[NodeIndex::from(11)].align, 14.into());
    assert_eq!(g[NodeIndex::from(12)].align, 6.into());
    assert_eq!(g[NodeIndex::from(13)].align, 7.into());
    assert_eq!(g[NodeIndex::from(14)].align, 11.into());
    assert_eq!(g[NodeIndex::from(15)].align, 18.into());
    assert_eq!(g[NodeIndex::from(16)].align, 1.into());
    assert_eq!(g[NodeIndex::from(17)].align, 19.into());
    assert_eq!(g[NodeIndex::from(18)].align, 20.into());
    assert_eq!(g[NodeIndex::from(19)].align, 23.into());
    assert_eq!(g[NodeIndex::from(20)].align, 24.into());
    assert_eq!(g[NodeIndex::from(21)].align, 12.into());
    assert_eq!(g[NodeIndex::from(22)].align, 22.into());
    assert_eq!(g[NodeIndex::from(23)].align, 17.into());
    assert_eq!(g[NodeIndex::from(24)].align, 15.into());
    assert_eq!(g[NodeIndex::from(25)].align, 9.into());
}

#[test]
fn alignment_down_left() {
    let (mut g, mut l) = create_test_layout();
    mark_type_1_conflicts(&mut g, &l);
    // down left means reverse each layer
    l.iter_mut().for_each(|l| l.reverse());
    reset_alignment(&mut g, &l);
    create_vertical_alignments(&mut g, &mut l);

    // block root 0
    for n in [0, 6] {
        assert_eq!(g[NodeIndex::from(n)].root, 0.into());
    }
    // block root 1
    {
        let n = 1;
        assert_eq!(g[NodeIndex::from(n)].root, 1.into());
    }
    // block root 2
    {
        let n = 2;
        assert_eq!(g[NodeIndex::from(n)].root, 2.into());
    }
    // block root 3
    {
        let n = 3;
        assert_eq!(g[NodeIndex::from(n)].root, 3.into());
    }
    // block root 16
    {
        let n = 16;
        assert_eq!(g[NodeIndex::from(n)].root, 16.into());
    }
    // block root 4
    for n in [4, 8] {
        assert_eq!(g[NodeIndex::from(n)].root, 4.into());
    }
    // block root 17
    for n in [17, 19, 23] {
        assert_eq!(g[NodeIndex::from(n)].root, 17.into());
    }
    // block root 18
    for n in [18, 20, 24] {
        assert_eq!(g[NodeIndex::from(n)].root, 18.into());
    }
    // block root 5
    for n in [5, 9, 25] {
        assert_eq!(g[NodeIndex::from(n)].root, 5.into());
    }
    // block root 7
    for n in [7, 11, 14] {
        assert_eq!(g[NodeIndex::from(n)].root, 7.into());
    }
    // block root 21
    for n in [21, 12, 15] {
        assert_eq!(g[NodeIndex::from(n)].root, 21.into());
    }
    // block root 10
    for n in [10, 13] {
        assert_eq!(g[NodeIndex::from(n)].root, 10.into());
    }
    // block root 22
    {
        let n = 22;
        assert_eq!(g[NodeIndex::from(n)].root, 22.into());
    }
}

#[test]
fn alignment_up_right() {
    let (mut g, mut l) = create_test_layout();
    mark_type_1_conflicts(&mut g, &l);

    // up right means reverse the edges of the graph and the ranks in the layer
    g.reverse();
    l.reverse();
    reset_alignment(&mut g, &l);
    create_vertical_alignments(&mut g, &mut l);

    for n in [13, 10] {
        assert_eq!(g[NodeIndex::from(n)].root, 13.into())
    }
    for n in [14, 11, 7] {
        assert_eq!(g[NodeIndex::from(n)].root, 14.into())
    }
    for n in [15, 23, 19, 17] {
        assert_eq!(g[NodeIndex::from(n)].root, 15.into())
    }
    for n in [22] {
        assert_eq!(g[NodeIndex::from(n)].root, 22.into())
    }
    for n in [24, 20, 18, 0] {
        assert_eq!(g[NodeIndex::from(n)].root, 24.into())
    }
    for n in [12, 21] {
        assert_eq!(g[NodeIndex::from(n)].root, 12.into())
    }
    for n in [25, 9, 5] {
        assert_eq!(g[NodeIndex::from(n)].root, 25.into())
    }
    for n in [8, 3] {
        assert_eq!(g[NodeIndex::from(n)].root, 8.into())
    }
    for n in [2] {
        assert_eq!(g[NodeIndex::from(n)].root, 2.into())
    }
    for n in [16] {
        assert_eq!(g[NodeIndex::from(n)].root, 16.into())
    }
    for n in [4] {
        assert_eq!(g[NodeIndex::from(n)].root, 4.into())
    }
    for n in [6] {
        assert_eq!(g[NodeIndex::from(n)].root, 6.into())
    }
    for n in [1] {
        assert_eq!(g[NodeIndex::from(n)].root, 1.into())
    }
}

#[test]
fn alignment_up_left() {
    let (mut g, mut l) = create_test_layout();
    mark_type_1_conflicts(&mut g, &l);
    // up left means reverse edges, ranks and order in layers
    g.reverse();
    l.reverse();
    l.iter_mut().for_each(|l| l.reverse());
    reset_alignment(&mut g, &l);
    create_vertical_alignments(&mut g, &mut l);

    for n in [15, 25, 9] {
        assert_eq!(g[NodeIndex::from(n)].root, 15.into())
    }
    for n in [14] {
        assert_eq!(g[NodeIndex::from(n)].root, 14.into())
    }
    for n in [13, 22] {
        assert_eq!(g[NodeIndex::from(n)].root, 13.into())
    }
    for n in [12, 21, 6] {
        assert_eq!(g[NodeIndex::from(n)].root, 12.into())
    }
    for n in [24, 20, 18] {
        assert_eq!(g[NodeIndex::from(n)].root, 24.into())
    }
    for n in [23, 19, 17, 1] {
        assert_eq!(g[NodeIndex::from(n)].root, 23.into())
    }
    for n in [11, 7] {
        assert_eq!(g[NodeIndex::from(n)].root, 11.into())
    }
    for n in [10] {
        assert_eq!(g[NodeIndex::from(n)].root, 10.into())
    }
    for n in [4, 8] {
        assert_eq!(g[NodeIndex::from(n)].root, 8.into())
    }
    for n in [0] {
        assert_eq!(g[NodeIndex::from(n)].root, 0.into())
    }
    for n in [2] {
        assert_eq!(g[NodeIndex::from(n)].root, 2.into())
    }
    for n in [3] {
        assert_eq!(g[NodeIndex::from(n)].root, 3.into())
    }
    for n in [16] {
        assert_eq!(g[NodeIndex::from(n)].root, 16.into())
    }
}

#[test]
fn place_blocks() {
    let (mut g, mut l) = create_test_layout();
    mark_type_1_conflicts(&mut g, &l);
    create_vertical_alignments(&mut g, &mut l);

    let block_1: Vec<NodeIndex> = [
        0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25,
    ]
    .into_iter()
    .map(|v| v.into())
    .collect();
    let block_2: Vec<NodeIndex> = [7, 10, 11, 22, 13, 14]
        .into_iter()
        .map(|v| v.into())
        .collect();

    let x_coordinates = super::place_blocks(&mut g, &l);

    assert_eq!(x_coordinates.len(), 26);
    for v in block_1 {
        assert_eq!(g[v].sink, 0.into());
    }

    for v in block_2 {
        assert_eq!(g[v].sink, 7.into());
    }
}
