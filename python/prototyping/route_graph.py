import networkx as nx

def route_graph(graph: nx.Graph):
    print("hello from python")
    print(graph.nodes())
    print(graph.edges())


    # 10 times the coordinates of the nodes
    for node in graph.nodes():
        graph.nodes[node]['pos'] = (100, 
                                    100)

    return graph






