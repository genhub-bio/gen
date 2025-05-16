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
    to the left-to-right coordinates we use to display the graph. Optionally applies
    a scaling and/or translation as well. Modifies the graph in-place.
    This encompasses:
    - transposing (left-to-right instead of top-to-bottom)
    - scaling (hscale, vscale)
    - shifting (x,y) to (x+shift,y+shift)
    """
    hscale, vscale = scale
    xshift, yshift = shift

    for node_id, data in G.nodes(data=True):
        if 'pos' not in data:
            print(f"Warning: Node {node_id} in transform_graph is missing 'pos' attribute and was not transformed.")
            continue

        x, y = data['pos'] # These are from channel_router's perspective

        new_x = x * hscale + xshift
        new_y = y * vscale + yshift
        
        G.nodes[node_id]['pos'] = (new_x, new_y)
        
    return G

def plot_graph(G: nx.Graph):
    # Refuse to plot graphs with nodes missing the 'pos' attribute
    if not all('pos' in data for _node, data in G.nodes(data=True)):
        raise ValueError("Warning: One or more nodes are missing the 'pos' attribute and were not plotted.")
    
    node_positions = {node: data['pos'] for node, data in G.nodes(data=True)}
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos=node_positions, with_labels=True)
    plt.show()


def assign_ports(G: nx.Graph):
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        neighbors = G.neighbors(node)
        north, south, east, west = False, False, False, False
        for neighbor in neighbors:
            nx, ny = G.nodes[neighbor]['pos']
            # Use Router's coordinate system (y increases upwards)
            if ny > y: north = True
            if ny < y: south = True
            # Use Router's coordinate system (x increases rightwards)
            if nx > x: east = True
            if nx < x: west = True
        # Store the orientation in the node attribute
        G.nodes[node]['ports'] = (north, south, east, west)



def route_layer(U_pos: Dict[Any, Tuple[int, int]], 
                V_pos: Dict[Any, Tuple[int, int]], 
                E: List[Tuple[Any, Any]]) -> nx.Graph:
    """
    Build a rectilinear routing between two layers of the graph.
    A layer is defined as the bipartite subgraph between two sets of nodes that have
    been assigned consecutive ranks in the Sugiyama algorithm.
    Where appropriate, edges are routed over a common bus to save visual clutter.
    This does introduce a problem of ambiguity when routing edges over a common bus:
    "do *all* nodes on this bus have the same neighbors, or do they share a subset?"
    We solve this problem by connecting nodes through distinct terminals in cases
    where it would otherwise be ambiguous. The area between the nodes and the terminals,
     and the area between sets of terminals are called "channels" (so one layer
    can have up to 3 channels".
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
    # Each terminal is defined by a linear position within its rank, and its net index.
    terminals = make_terminals(nets, U_pos | V_pos) 

    layer_graph = nx.Graph()
    # Add U and V nodes to the layer graph with their positions
    for node, (x,y) in list(U_pos.items()) + list(V_pos.items()):
        layer_graph.add_node(node, pos=(x,y))
 
    # First channel: U nodes to U terminals (as nodes)
    L_pins = [Pin(node, y) for node, (_x,y) in U_pos.items()]
    R_pins = [Pin(terminal.node, terminal.position) 
                for terminal in terminals 
                if terminal.node in U]

    L_pin_list, R_pin_list = make_pin_lists(L_pins, R_pins)
    G1 = route_channel.Router(R_pin_list, L_pin_list).route()

    # Track vertical extent of the first channel so we can offset the next
    # (Note that we are operating in a top-to-bottom coordinate system here (TODO: fix this))
    y_offset = 0
    if G1.number_of_nodes() > 0:
        y_offset = max(y for _n, (_x, y) in nx.get_node_attributes(G1, 'pos').items())
    
    # Second channel: U terminals (as nets) to V terminals (as nets)
    L_pins = [Pin(terminal.net_index, terminal.position) 
                for terminal in terminals 
                if terminal.node in U]
    R_pins = [Pin(terminal.net_index, terminal.position) 
                for terminal in terminals 
                if terminal.node in V]
    
    L_pin_list, R_pin_list = make_pin_lists(L_pins, R_pins)
    G2 = route_channel.Router(R_pin_list, L_pin_list).route()

    # Shift G2 vertically so it sits below the first channel
    for node, data in G2.nodes(data=True):
        if 'pos' in data:
            x, y = data['pos']
            G2.nodes[node]['pos'] = (x, y + y_offset)

    # Update y_offset for the following stage
    if G2.number_of_nodes() > 0:
        y_offset = max(y for _n, (_x, y) in nx.get_node_attributes(G2, 'pos').items())

    # Third channel: V terminals (as nodes) to V nodes
    L_pins = [Pin(terminal.node, terminal.position)
                for terminal in terminals
                if terminal.node in V]
    R_pins = [Pin(node, y) for node, (_x,y) in V_pos.items()]

    L_pin_list, R_pin_list = make_pin_lists(L_pins, R_pins)
    G3 = route_channel.Router(R_pin_list, L_pin_list).route()

    # Shift G3 below the previous channels using current y_offset
    for node, data in G3.nodes(data=True):
        if 'pos' in data:
            x, y = data['pos']
            G3.nodes[node]['pos'] = (x, y + y_offset)




    pos_to_id: Dict[Tuple[int, int], int] = {layer_graph.nodes[n]['pos']: n for n in layer_graph.nodes() if 'pos' in layer_graph.nodes[n] and isinstance(layer_graph.nodes[n]['pos'], tuple)}

    next_free_id = max(layer_graph.nodes) + 1

    # Merge all channel graphs into layer_graph
    for src_G in (G1, G2, G3):
        # 1) Merge / reuse nodes based on their coordinates
        for n, attrs in src_G.nodes(data=True):
            pos = attrs.get('pos')
            if pos is None:
                continue  # skip nodes without positional information

            if pos in pos_to_id:
                node_id = pos_to_id[pos]
            else:
                # Assign next available integer identifier (auto_id_counter is always unique
                # because we initialized it to max(existing_ids)+1 and only increment it here).
                node_id = next_free_id
                next_free_id += 1

                layer_graph.add_node(node_id, **attrs)
                pos_to_id[pos] = node_id

        # 2) Add edges using the mapped node IDs
        for u, v, e_attrs in src_G.edges(data=True):
            pos_u = src_G.nodes[u].get('pos')
            pos_v = src_G.nodes[v].get('pos')
            if pos_u is None or pos_v is None:
                continue
            u_id = pos_to_id[pos_u]
            v_id = pos_to_id[pos_v]
            if layer_graph.has_edge(u_id, v_id):
                layer_graph.edges[u_id, v_id].update(e_attrs)
            else:
                layer_graph.add_edge(u_id, v_id, **e_attrs)

    # Recompute port orientation for the merged graph
    assign_ports(layer_graph)

    return layer_graph




if __name__ == "__main__":
    edges = [(1,3), (1,4), (2,3), (2,4), (2,5)]
    node_pos = {1: (0, 0), 2: (0, 1), 
                3: (1, 0), 4: (1, 1), 5: (1, 2)}
    
    U, V = map(set, zip(*edges))

    G = route_layer(dict((k,v) for k,v in node_pos.items() if k in U),
                    dict((k,v) for k,v in node_pos.items() if k in V),
                    edges)
    
    plot_graph(G)    

    

