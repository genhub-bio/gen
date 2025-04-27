#! /usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from typing import List, Set, Dict, Any

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

def make_terminals(nets: list[set[tuple[any, any]]], node_pos: dict[any, float]):
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

    # Loop over the nodes in the order of their position within each rank
    # to place the terminals.
    L_terminals = [] # (node, position, net_idx)
    R_terminals = [] 
    for node in sorted(L_node_nets.keys(), key=lambda x: node_pos[x]):
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

    




if __name__ == "__main__":
    # Create example graph
    edges = [("u1", "v1"),
             ("u1", "v2"),
             ("u1", "v3"),
             ("u2", "v2"),
             ("u2", "v3"),
             ("u3", "v3"),
             ("u4", "v2"),
             ("u5", "v4")]
    
    U, V = map(set, zip(*edges))
    node_pos = {}
    node_pos.update((node, 2*i) for i, node in enumerate(sorted(U, reverse=True)))
    node_pos.update((node, 2*i) for i, node in enumerate(sorted(V, reverse=True)))
    
    # Make terminals
    bicliques = enumerate_bicliques(edges)
    nets = make_nets(bicliques) 
    L_terminals, R_terminals = make_terminals(nets, node_pos)

    # Setup the input to the routing algorithm
    L_pin_list = make_pin_list(L_terminals)
    R_pin_list = make_pin_list(R_terminals)

    # Add trailing zeros to the pin lists to make them the same length
    L_pin_list += [0] * (len(R_pin_list) - len(L_pin_list))
    R_pin_list += [0] * (len(L_pin_list) - len(R_pin_list))

    # TODO: verify that we won't have negative positions in the real application

    for i, net in enumerate(nets):
        print(f"Net {i+1}: edges {net}, nodes {list(map(set, zip(*net)))}")  

    print("Pin lists for routing:")
    # Make a similar list for the node positions
    L_node_pos = ['  '] * len(L_pin_list)
    R_node_pos = ['  '] * len(R_pin_list)
    for node in U:
        L_node_pos[node_pos[node]] = node
    for node in V:
        R_node_pos[node_pos[node]] = node

    for (Ln, Lp, Rp, Rn) in reversed(list(zip(L_node_pos, L_pin_list, R_pin_list, R_node_pos))):
        print(f"{Ln} ... {Lp} ... {Rp} ... {Rn}")



        
    # Draw the nets (just for testing)
    G = nx.Graph()

    U, V = map(set, zip(*edges))
    G.add_nodes_from(U, bipartite=0)
    G.add_nodes_from(V, bipartite=1)
    G.add_edges_from(edges)

    coord = {}
    coord.update((node, (0, i)) for i, node in enumerate(sorted(U, reverse=True)))
    coord.update((node, (1, i)) for i, node in enumerate(sorted(V, reverse=True)))


    plt.figure(figsize=(6,4))
    nx.draw(G, coord, with_labels=True, node_size=500, node_color='lightblue', edge_color='grey')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(nets)))
    for i, net in enumerate(nets):
        nx.draw_networkx_edges(G, coord, edgelist=net, edge_color=[colors[i],]*len(net), width=2)
    plt.show()


   

