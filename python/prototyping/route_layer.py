#! /usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from typing import List, Set, Dict, Any, Tuple
import route_channel
import unittest

class Terminal:
    def __init__(self, node: Any, position: int, net_index: int):
        self.node = node
        self.position = position
        self.net_index = net_index

    def __repr__(self):
        return f"Terminal(node={self.node}, position={self.position}, net_index={self.net_index})"

class Pin:
    def __init__(self, obj: Any, pos: int):
        self.obj = obj
        self.pos = pos

    def __repr__(self):
        return f"Pin(obj={self.obj}, pos={self.pos})"

def enumerate_bicliques(edges: List[Tuple[Any, Any]]):
    """
    Enumerate all maximal bicliques in the graph using the MBEA algorithm from:
    Zhang et al. 2014 "On finding bicliques in bipartite graphs: a novel algorithm and its application to the
    integration of diverse biological data types"
    """
    # Setup the bipartite graph
    U, V = map(set, zip(*edges))

    # The algorithm assumes that the left side is smaller than the right side,
    # so we need to check for that and swap if necessary.
    if len(U) > len(V):
        U, V = V, U
        swap = True
    else:
        swap = False   

    G = nx.Graph()
    G.add_edges_from(edges)

    # Initial algorithm state
    L0 = U                      # L  ← U
    R0: Set[Any] = set()        # R  ← ∅
    P0 = sorted(V, key=G.degree)  # P  ← V  (sorted by |N(u)|)
    Q0: List[Any] = []          # Q  ← ∅
    bicliques: List[tuple[Set[Any], Set[Any]]] = []

    # Execute algorithm, collecting bicliques directly in the bicliques list
    find_biclique(G, L0, R0, P0, Q0, bicliques)

    if swap:
        bicliques = [(R, L) for L, R in bicliques]

    # Sanity check: all edges from the bicliques should be in the original graph,
    # and all edges from the original graph should be in the bicliques
    for biclique in bicliques:
        assert all((u, v) in edges for u in biclique[0] for v in biclique[1])
    for (u,v) in edges:
        assert any((u in L) and (v in R) for L, R in bicliques)

    return bicliques

    
def find_biclique(
    G: nx.Graph,
    L: Set[Any],            # L  – common neighbours of vertices in R
    R: Set[Any],            # R  – right vertices already in the biclique
    P: List[Any],           # P  – candidate right vertices (ordered list!)
    Q: List[Any],           # Q  – tested candidates that are *not* in R
    bicliques: List[tuple[Set[Any], Set[Any]]]
) -> None:
    """
    Recursive function at the core of the MBEA algorithm.

    Parameters
    ----------
    G
        A *bipartite* NetworkX graph. 
    L, R, P, Q
        The four state variables of Algorithm 1 (see paper):
            • L : set of vertices ∈ U that are common neighbors of vertices in R, initially L = U;
            • R : set of vertices ∈ V belonging to the current biclique, initially R = ∅;
            • P : still-untested vertices ∈ V, initially P = V;
            • Q : set of vertices used to determine maximality, initially Q = ∅.
    bicliques
        Output list that collects pairs ``(L', R')`` for every biclique found.
        Modified in-place.
    """

    while P:
        # (*1) Select next candidate 'x' from P and attempt to extend the biclique
        x = P.pop(0)              
        R_prime = R | {x}  # R' = R ∪ {x}
        L_prime = L & set(G.neighbors(x))   # L' = L ∩ N(x)

        # New containers for the *next* level's candidates / non-candidates
        P_prime: List[Any] = []  
        Q_prime: List[Any] = []

        # (*2)  Check if the biclique is maximal by looking at L'
        is_maximal = True
        for v in Q:
            Nv = set(G.neighbors(v))
            if L_prime == Nv:
                is_maximal = False
                break
            elif L_prime & Nv:
                Q_prime.append(v)

        Q.append(x)

        # Stop exploring this branch if the biclique is not maximal
        if not is_maximal:
            continue

        # (*3)  Expand R' to maximal size
        for v in P:                                  
            Nv = set(G.neighbors(v))
            if L_prime <= Nv:                         # fully connected
                R_prime.add(v)                        # → extend biclique
            elif L_prime & Nv:                        # partially connected
                P_prime.append(v)                     # → stay as candidate

        # Store the biclique
        bicliques.append((L_prime, R_prime))

        # (*4)  Recurse if there are still candidates left
        if P_prime:   
            find_biclique(
                G,
                L_prime,
                R_prime,
                P_prime,
                Q_prime,
                bicliques
            )

def make_nets(bicliques: List[Tuple[Set[Any], Set[Any]]]):
    """
    Partition the edges of each biclique into nets, which are non-overlapping
    bicliques chosen such that each edge is included in exactly one net,
    preferentially the largest biclique.

    Returns a list of sets, each containing the edges of a net.
    """

    # Sort the bicliques by number of edges (descending)
    bicliques.sort(key=lambda x: len(x[0]) * len(x[1]), reverse=True)

    # Each biclique results in a net, with the restriction that:
    # - nodes may be shared between nets
    # - edges are not duplicated

    # We will build the nets one by one, starting with the largest biclique
    # and removing edges that have already been used.
    nets = []
    for L, R in bicliques:
        biclique_edges = set([(u, v) for u in L for v in R])
        unique_edges = biclique_edges - set(chain(*nets))
        if not unique_edges:
            continue
        nets.append(unique_edges)
    return nets

def make_terminals(nets: List[Tuple[Any, Any]], 
                   node_pos: Dict[Any, Tuple[int, int]]) -> List[Terminal]:
    """
    Nodes connect to nets via terminal nodes that become the pins in the routing algorithm.
    We place the terminal nodes at the same within-rank position as the original nodes, except
    if there is already a terminal node there, in which case we move everything up.

    Returns a list of Terminal objects that each refer to a node, 
    net index (starting from 1) and within-rank position.
    """
    # Assign nodes and an average position to each net
    net_nodes = []
    for net_idx, net in enumerate(nets):
        L_nodes, R_nodes = map(set, zip(*net))
        net_pos = sum(node_pos[node][1] for node in L_nodes.union(R_nodes)) / len(L_nodes.union(R_nodes))
        net_nodes.append((net_idx+1,
                          net_pos, 
                          L_nodes,
                          R_nodes))
        
    # Assign nets to each node, in the order of the position of the net
    L_node_nets = {}
    R_node_nets = {}
    for (net_idx, net_pos, L_nodes, R_nodes) in sorted(net_nodes, key=lambda x: x[1]):
        for node in L_nodes:
            L_node_nets[node] = L_node_nets.get(node, []) + [net_idx]
        for node in R_nodes:
            R_node_nets[node] = R_node_nets.get(node, []) + [net_idx]

    # Loop over the nodes in the order of their position within each rank
    # to place the terminals.
    L_terminals: List[Terminal] = [] 
    R_terminals: List[Terminal] = [] 
    for node in sorted(L_node_nets.keys(), key=lambda u: node_pos[u][1]):
        for net_idx in L_node_nets[node]:
            # Try to place it in the node position, but if that's already taken,
            # place it one unit away from the last terminal.
            attempt_pos = round(node_pos[node][1])
            if not L_terminals or L_terminals[-1].position < attempt_pos:
                L_terminals.append(Terminal(node, attempt_pos, net_idx))
            else:
                last_pos = L_terminals[-1].position
                L_terminals.append(Terminal(node, last_pos+1, net_idx))
    
    # Same for the right nodes
    for node in sorted(R_node_nets.keys(), key=lambda v: node_pos[v][1]):
        for net_idx in R_node_nets[node]:
            attempt_pos = round(node_pos[node][1])
            if not R_terminals or R_terminals[-1].position < attempt_pos:
                R_terminals.append(Terminal(node, attempt_pos, net_idx))
            else:
                last_pos = R_terminals[-1].position
                R_terminals.append(Terminal(node, last_pos+1, net_idx))

    return L_terminals + R_terminals

# def make_pin_list(terminals: list[tuple[any, int, int]]):
#     """
#     Convert the list of terminals into a list of pins for use in the routing algorithm.
#     This list has 0 for empty positions, and the net index + 1 for occupied positions.
#     """
#     pin_list = [0] * (max(pos for (node, pos, net_idx) in terminals) + 1)
#     for (node, pos, net_idx) in terminals:
#         pin_list[pos] = net_idx
#     return pin_list

def make_pin_lists(L: List[Pin],
                  R: List[Pin]):
    """
    Converts lists consisting of Pin objects (with 'obj' and 'pos' attributes)
    into longer lists where each element is either an object or 0. 
    E.g. [Pin("foo", 2), Pin("bar", 4)] becomes [0, 0, "foo", 0, "bar"]
    When applied to multiple lists at the same time, trailing 0s are
    added to normalize their lengths.
    """
    assert len(L) + len(R) > 0, "Both lists are empty"
    
    length = max(pin.pos for pin in L + R) + 1

    L_out = [0] * length
    for pin in L:
        L_out[pin.pos] = pin.obj
    
    R_out = [0] * length
    for pin in R:
        R_out[pin.pos] = pin.obj

    return (L_out, R_out)

def transform_graph(G: nx.Graph, 
                    scale: Tuple[int, int]=(1,2 ), 
                    shift: Tuple[int, int]=(0,0)):
    """
    Transform a graph from the channel router's coordinate system (top-to-bottom)
    to the left-to-right coordinates we use to display the graph.
    This encompasses:
    - transposing (left-to-right instead of top-to-bottom)
    - scaling (hscale, vscale)
    - shifting (x,y) to (x+shift,y+shift)
    """
    hscale, vscale = scale
    xshift, yshift = shift
    return nx.relabel_nodes(G, lambda xy: (xy[1] * hscale + xshift, xy[0] * vscale + yshift), copy=True)

def plot_graph(G: nx.Graph):
    plt.figure(figsize=(5, 5))
    node_positions = {node: node for node in G.nodes()}
    nx.draw(G, pos=node_positions, with_labels=True)
    plt.show()


def assign_ports(G: nx.Graph):
    for node in G.nodes():
        x, y = node
        neighbors = G.neighbors(node)
        north, south, east, west = False, False, False, False
        for neighbor in neighbors:
            nx, ny = neighbor
            # Use Router's coordinate system (y increases upwards)
            if ny > y: north = True
            if ny < y: south = True
            # Use Router's coordinate system (x increases rightwards)
            if nx > x: east = True
            if nx < x: west = True
        # Store the orientation in the node attribute
        G.nodes[node]['ports'] = (north, south, east, west)

def colinear(p1, p2, p3):
    """
    Tests if the points, given as 3 tuples (x, y), are collinear.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (y2 - y1) * (x3 - x2) == (y3 - y2) * (x2 - x1)

def simplify(G: nx.Graph):
    """
    Build a new graph by removing all degree-2 nodes that are colinear with their neighbors.
    Assumes that each node has an (x,y) tuple in the 'pos' attribute.
    """
    if not G or G.number_of_edges() == 0:
        return

    simplified_G = nx.Graph()
    processed_segments = set()

    # 1. Identify and add all critical nodes (degree != 2 or not colinear with neighbors)
    critical_nodes = set()
    original_nodes = list(G.nodes())
    for node in original_nodes:
        neighbors = list(G.neighbors(node))
        if len(neighbors) != 2:
            critical_nodes.add(node)
            simplified_G.add_node(node, **G.nodes[node])
        else:
            # Check if node is colinear with its two neighbors
            p1 = G.nodes[neighbors[0]].get('pos')
            p2 = G.nodes[node].get('pos')
            p3 = G.nodes[neighbors[1]].get('pos')
            if not (p1 and p2 and p3 and colinear(p1, p2, p3)):
                critical_nodes.add(node)
                simplified_G.add_node(node, **G.nodes[node])

    # Handle case where graph might be a single loop of colinear segments
    if not critical_nodes and G.number_of_nodes() > 0:
        start_node = original_nodes[0]
        critical_nodes.add(start_node)
        simplified_G.add_node(start_node, **G.nodes[start_node])

    # 2. Iterate through critical nodes and trace segments using true colinearity
    for start_node in critical_nodes:
        for neighbor in list(G.neighbors(start_node)):
            if neighbor in critical_nodes:
                segment_endpoints = tuple(sorted((start_node, neighbor)))
                if segment_endpoints not in processed_segments:
                    net_attr = G.edges[start_node, neighbor].get('net')
                    if not simplified_G.has_edge(start_node, neighbor):
                        simplified_G.add_edge(start_node, neighbor, net=net_attr)
                    processed_segments.add(segment_endpoints)
            else:
                # Start of a colinear segment
                prev = start_node
                curr = neighbor
                segment_net = G.edges[prev, curr].get('net')
                path = [prev, curr]
                while True:
                    neighbors = list(G.neighbors(curr))
                    # Remove the previous node from neighbors
                    next_candidates = [n for n in neighbors if n != prev]
                    if len(next_candidates) != 1:
                        # Dead-end or branch, stop here
                        break
                    next_node = next_candidates[0]
                    # Check net consistency
                    next_net = G.edges[curr, next_node].get('net')
                    if next_net != segment_net:
                        break
                    # Check colinearity
                    p1 = G.nodes[prev].get('pos')
                    p2 = G.nodes[curr].get('pos')
                    p3 = G.nodes[next_node].get('pos')
                    if not (p1 and p2 and p3 and colinear(p1, p2, p3)):
                        break
                    prev, curr = curr, next_node
                    path.append(curr)
                    if curr in critical_nodes:
                        break

                end_node = curr
                segment_endpoints = tuple(sorted((start_node, end_node)))
                if not simplified_G.has_node(end_node):
                    simplified_G.add_node(end_node, **G.nodes[end_node])
                if segment_endpoints not in processed_segments and start_node != end_node:
                    if not simplified_G.has_edge(start_node, end_node):
                        simplified_G.add_edge(start_node, end_node, net=segment_net)
                    processed_segments.add(segment_endpoints)

    return simplified_G

def route_layer(U_pos: Dict[Any, Tuple[int, int]], 
                V_pos: Dict[Any, Tuple[int, int]], 
                E: Set[Tuple[Any, Any]]) -> nx.Graph:
    """
    Build a rectilinear routing between two layers of the graph.
    A layer is defined as the bipartite subgraph between two sets of nodes that have
    been assigned consecutive ranks in the Sugiyama algorithm.
    Where appropriate, edges are routed over a common bus to save visual clutter.
    This does introduce a problem of ambiguity when routing edges over a common bus:
    "do all nodes have the same neighbors, or just a subset?"
    We solve this problem by connecting nodes through distinct terminals in cases
    where it would otherwise be ambiguous. We refer to the area between the nodes
    and the terminals, and between the two sets of terminals as "channels"
    of terminals as the "channel".
    """
    U = set(U_pos.keys())
    V = set(V_pos.keys())

    assert all((source in U and target in V) for source, target in E)
    assert not any((target in U and source in V) for source, target in E)

    # Nets are non-overlapping subgraphs where every node on one side,
    # has an edge to every node on the other side.
    # This is similar to the definition of bicliques, except that those may overlap.
    bicliques = enumerate_bicliques(E)
    nets = make_nets(bicliques) 

    # For each node, we create one terminal per net that it is part of.
    # Each terminal is defined by a position within its rank, and its net index.
    terminals = make_terminals(nets, U_pos | V_pos) 

    layer_graph = nx.Graph()
    # Add U nodes to the layer graph and set up a pin list
    for u in U:
        x, y = U_pos[u]
        layer_graph.add_node(u, pos=(x,y))

    # First channel: U nodes to U terminals (as nodes)
    L_pins = [Pin(node, y) for node, (_x,y) in U_pos.items()]
    R_pins = [Pin(terminal.node, terminal.position) 
                for terminal in terminals 
                if terminal.node in U]

    L_pin_list, R_pin_list = make_pin_lists(L_pins, R_pins)
    channel_router = route_channel.Router(R_pin_list, L_pin_list)
    channel_graph = channel_router.route()
    channel_graph = transform_graph(channel_graph)
    max_x = max(xy[0] for xy in channel_graph.nodes())

    # Second channel: U terminals (as nets) to V terminals (as nets)
    L_pins = [Pin(terminal.net_index, terminal.position) 
                for terminal in terminals 
                if terminal.node in U]
    R_pins = [Pin(terminal.net_index, terminal.position) 
                for terminal in terminals 
                if terminal.node in V]
    
    L_pin_list, R_pin_list = make_pin_lists(L_pins, R_pins)
    channel_router = route_channel.Router(R_pin_list, L_pin_list)
    channel_graph2 = channel_router.route()
    channel_graph2 = transform_graph(channel_graph2, shift=(max_x, 0))

    # Combine the two channel graphs
    combined_graph = nx.compose(channel_graph, channel_graph2)
    max_x = max(xy[0] for xy in combined_graph.nodes())

    # Third channel: V terminals (as nodes) to V nodes
    L_pins = [Pin(terminal.node, terminal.position) 
                for terminal in terminals 
                if terminal.node in V]
    R_pins = [Pin(node, y) for node, (_x,y) in V_pos.items()]

    L_pin_list, R_pin_list = make_pin_lists(L_pins, R_pins)
    channel_router = route_channel.Router(R_pin_list, L_pin_list)
    channel_graph3 = channel_router.route()
    channel_graph3 = transform_graph(channel_graph3, shift=(max_x, 0))

    # Combine and clean up the three channel graphs
    combined_graph = nx.compose(combined_graph, channel_graph3)
    plot_graph(combined_graph)
    #assign_ports(combined_graph)
    simplified_graph = simplify(combined_graph)
    plot_graph(simplified_graph)

    return simplified_graph




if __name__ == "__main__":
    # Create an example graph with integer nodes and 'pos' attributes
    # (x,y) coordinates. x=0 for left, x=1 for right.
   
    edges_int = [(1,3), (1,4), (2,3), (2,4), (2,5)]
    node_pos = [(1, {'pos': (0, 0)}), (2, {'pos': (0, 1)}), 
                (3, {'pos': (1, 0)}), (4, {'pos': (1, 1)}), (5, {'pos': (1, 2)})]
    
    # Determine U and V sets from integer edges to assign y-coordinates
    U, V = map(set, zip(*edges_int))
    G_example = nx.Graph()
    G_example.add_nodes_from(node_pos)
    G_example.add_edges_from(edges_int)
    
    # Define U and V for the example graph
    U_example, V_example = map(set, zip(*edges_int))

    # Call the new function
    layer_subgraph = route_layer(dict((n, attr['pos']) for n, attr in node_pos if n in U),
                                 dict((n, attr['pos']) for n, attr in node_pos if n in V),
                                 edges_int)
    
    print(layer_subgraph)
    


    
# Tests:
class TestMakePinList(unittest.TestCase):
    def test_basic_case(self):
        # Basic test case - verifies the core functionality works
        L = [Pin("foo", 2), Pin("bar", 4)]
        R = [Pin("baz", 1), Pin("qux", 3)]
        result_L, result_R = make_pin_lists(L, R)
        self.assertEqual(result_L, [0, 0, "foo", 0, "bar"])
        self.assertEqual(result_R, [0, "baz", 0, "qux", 0])

    def test_different_lengths(self):
        # Tests the normalization feature - both lists get same length
        L = [Pin("a", 1), Pin("b", 3)]
        R = [Pin("x", 0), Pin("y", 2), Pin("z", 5)]
        result_L, result_R = make_pin_lists(L, R)
        self.assertEqual(result_L, [0, "a", 0, "b", 0, 0])
        self.assertEqual(result_R, ["x", 0, "y", 0, 0, "z"])

    def test_one_empty_list(self):
        L = [Pin("a", 2)]
        R = []
        result_L, result_R = make_pin_lists(L, R)
        self.assertEqual(result_L, [0, 0, "a"])
        self.assertEqual(result_R, [0, 0, 0])

    def test_both_lists_empty(self):
        L = []
        R = []
        with self.assertRaises(AssertionError) as context:
            make_pin_lists(L, R)
        self.assertEqual(str(context.exception), "Both lists are empty")



    

