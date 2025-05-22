import networkx as nx
def build_graph(nodes, edges):
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node["id"], **node)
    for edge in edges:
        G.add_edge(edge["from"], edge["to"], label=edge["label"])
    return G