#! /usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from typing import List, Set, Dict, Any
import python.prototyping.route_channel as route_channel

def enumerate_bicliques(edges: list[tuple[any, any]]):
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

def make_nets(bicliques: list[tuple[Set[Any], Set[Any]]]):
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

def make_terminals(nets: list[set[tuple[any, any]]], node_pos: dict[any, (int, int)]):
    """
    Nodes connect to nets via terminal nodes that become the pins in the routing algorithm.
    We place the terminal nodes at the same within-rank position as the original nodes, except
    if there is already a terminal node there, in which case we move everything up.

    Returns a dictionary mapping each node to a list of net indices (1-indexed).
    """
    # Assign nodes and an average position to each net
    net_nodes = []
    for net_idx, net in enumerate(nets):
        L_nodes, R_nodes = map(set, zip(*net))
        net_pos = sum(node_pos[node] for node in L_nodes.union(R_nodes)) / len(L_nodes.union(R_nodes))
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

    # If each node has only one net, we don't have to create terminals
    if all(len(nets) == 1 for nets in L_node_nets.values()):
        return None




    # Loop over the nodes in the order of their position within each rank
    # to place the terminals.
    L_terminals = [] # (node, position, net_idx)
    R_terminals = [] 
    for node in sorted(L_node_nets.keys(), key=lambda p: node_pos[p][1]):
        for net_idx in L_node_nets[node]:
            # Try to place it in the node position, but if that's already taken,
            # place it one unit away from the last terminal.
            attempt_pos = round(node_pos[node])
            if not L_terminals or L_terminals[-1][1] < attempt_pos:
                L_terminals.append((node, attempt_pos, net_idx))
            else:
                last_pos = L_terminals[-1][1]
                L_terminals.append((node, last_pos+1, net_idx))
    
    # Same for the right nodes
    for node in sorted(R_node_nets.keys(), key=lambda x: node_pos[x]):
        for net_idx in R_node_nets[node]:
            attempt_pos = round(node_pos[node])
            if not R_terminals or R_terminals[-1][1] < attempt_pos:
                R_terminals.append((node, attempt_pos, net_idx))
            else:
                last_pos = R_terminals[-1][1]
                R_terminals.append((node, last_pos+1, net_idx))

    return L_terminals, R_terminals

def make_pin_list(terminals: list[tuple[any, int, int]]):
    """
    Convert the list of terminals into a list of pins for use in the routing algorithm.
    This list has 0 for empty positions, and the net index + 1for occupied positions.
    """
    pin_list = [0] * (max(pos for (node, pos, net_idx) in terminals) + 1)
    for (node, pos, net_idx) in terminals:
        pin_list[pos] = net_idx
    return pin_list

def route_layer(graph: nx.Graph, V: Set[Any], E: Set[tuple[Any, Any]]) -> Any:
    """
    Adds a layer to a growing graph by building a rectilinear routing from nodes
    in set V (not part of the graph) to nodes in set U (already part of the graph).
    The layer is defined by a set of nodes that hold a y-coordinate as attribute,
    and a set of edges E that connect nodes in V to nodes in U.
    Where appropriate, edges are routed over a common bus to save visual clutter.
    This does introduce a problem of ambiguity when routing edges over a common bus:
    "do all nodes have the same neighbors, or just a subset?"
    We solve this problem by connecting nodes through distinct terminals in cases
    where it would otherwise be ambiguous.
    """

    # Assert that the source nodes are part of the graph, and the target nodes are not
    assert all((source in graph and target in V) for source, target in E)
    assert not any((target in graph and source in V) for source, target in E)

    node_pos = {node: data['pos'] for node, data in graph.nodes(data=True) if node in U or node in V}

    # Nets are non-overlapping subgraphs where every node on one side,
    # has an edge to every node on the other side.
    # This is similar to the definition of bicliques, except that those may overlap.
    bicliques = enumerate_bicliques(E)
    nets = make_nets(bicliques) 



    # AREA BELOW STILL NEEDS TO BE REFACTORED: make_terminals should add to the graph
    # and so should the actual channel routing. Otherwise it'll be annoying to have ids
    # still line up if merges happen much later.

    # For each node, we create one terminal per net that it is part of.
    L_terminals, R_terminals = make_terminals(nets, node_pos)

    L_pin_list = make_pin_list(L_terminals)
    R_pin_list = make_pin_list(R_terminals)

    # Add trailing zeros to the pin lists to make them the same length
    if len(L_pin_list) > len(R_pin_list):
        R_pin_list += [0] * (len(L_pin_list) - len(R_pin_list))
    elif len(R_pin_list) > len(L_pin_list):
        L_pin_list += [0] * (len(R_pin_list) - len(L_pin_list))

    # Print net information (part of original selection)
    for i, net_edge_set in enumerate(nets):
        if net_edge_set:
            # map(set, zip(*net_edge_set)) creates [L_nodes_in_net, R_nodes_in_net]
            net_node_parts = list(map(set, zip(*net_edge_set)))
            print(f"Net {i+1}: edges {net_edge_set}, nodes {net_node_parts}")  
        else:
            print(f"Net {i+1}: edges {net_edge_set}, nodes []")

    # Print pin lists for routing (part of original selection)
    print("Pin lists for routing:")
    
    # Create dictionaries to map terminal positions back to node IDs for printing
    L_nodes_at_pin_pos = {pos: str(node) for node, pos, _ in L_terminals}
    R_nodes_at_pin_pos = {pos: str(node) for node, pos, _ in R_terminals}
    max_len_pins = len(L_pin_list) # L_pin_list and R_pin_list are same length

    for i in reversed(range(max_len_pins)):
        ln_str = L_nodes_at_pin_pos.get(i, '  ')
        lp_val = L_pin_list[i]
        rp_val = R_pin_list[i]
        rn_str = R_nodes_at_pin_pos.get(i, '  ')
        # Adjust spacing if node IDs can be long; assuming short integers for now
        print(f"{ln_str:<3s} ... {lp_val:<2} ... {rp_val:<2} ... {rn_str:<3s}")

    # Setup the router and plotter (part of original selection)
    # The original selection used R_pin_list, L_pin_list in this order for Router
    router = route_channel.Router(R_pin_list, L_pin_list)
    g_routed = router.route()
    plotter = route_channel.Plotter(g_routed)
    
    return plotter

if __name__ == "__main__":
    # Create an example graph with integer nodes and 'pos' attributes
    # (x,y) coordinates. x=0 for left, x=1 for right.
   
    edges_int = [(0, 10), (0, 11), (0, 12), (1, 11), (1, 12), (2, 12), (3, 11), (4, 13)]

    node_pos = [(4, {'pos': (0, 0)}), (3, {'pos': (0, 2)}), (2, {'pos': (0, 4)}), (1, {'pos': (0, 6)}), (0, {'pos': (0, 8)}), (13, {'pos': (1, 0)}), (12, {'pos': (1, 2)}), (11, {'pos': (1, 4)}), (10, {'pos': (1, 6)})]
    # Determine U and V sets from integer edges to assign y-coordinates
    U, V = map(set, zip(*edges_int))
    G_example = nx.Graph()
    G_example.add_nodes_from(node_pos)
    G_example.add_edges_from(edges_int)
    
    # Define U and V for the example graph
    U_example, V_example = map(set, zip(*edges_int))

    # Call the new function
    plotter_obj = route_layer(G_example, U_example, V_example)
    
    # Print the rendered graph from the plotter object
    if plotter_obj:
        print("\nRendered graph from route_valley:")
        print(plotter_obj.render_text_graph())
    else:
        print("route_valley did not return a plotter object.")


    

  



   

