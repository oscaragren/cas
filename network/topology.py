from __future__ import annotations

import networkx as nx

def to_adjacency(graph: nx.Graph) -> dict[str, set[str]]:
    """
    Convert a networkx Graph into the plain dict[id, set[id]] form
    the rest of the codebase consumes, so nothing outside this file
    needs to import networkx directly.
    """
    return {str(node): {str(neighbor) for neighbor in graph.neighbors(node)} for node in graph.nodes}

def ring_lattice(n: int, k: int, seed: int | None = None) -> dict[str, set[str]]:
    """
    Each agent connected to its k nearest neighbors on a ring.
    k must be even (k/2 neighbors on each side).
    """
    graph = nx.watts_strogatz_graph(n, k, p=0.0, seed=seed)
    return _relabel(graph, n)

def watts_strogatz(n: int, k: int, p: float, seed: int | None = None) -> dict[str, set[str]]:
    """
    Start from a ring lattice, rewire each edge with probability p.
    p=0 is the ring, p approaching 1 is close to a random graph.
    """
    graph = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return _relabel(graph, n)

def erdos_renyi(n: int, p: float, seed: int | None = None) -> dict[str, set[str]]:
    """
    Each agent connected to its k nearest neighbors on a ring.
    k must be even (k/2 neighbors on each side).
    """
    graph = nx.erdos_renyi_graph(n, p, seed=seed)
    return _relabel(graph, n)

def _relabel(graph: nx.Graph, n: int) -> dict[str, set[str]]:
    mapping = {i: f"agent_{i}" for i in range(n)}
    graph = nx.relabel_nodes(graph, mapping)
    return to_adjacency(graph)

def clustering_coefficient(adjacency: dict[str, set[str]]) -> float:
    graph = _from_adjacency(adjacency)
    return nx.average_clustering(graph)

def average_path_length(adjacency: dict[str, set[str]]) -> float:
    graph = _from_adjacency(adjacency)
    if not nx.is_connected(graph):
        # disconnected graphs don't have a well-defined average path length
        # across the whole graph; report it per connected component instead
        # of silently producing a misleading number
        largest_cc = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_cc)
    return nx.average_shortest_path_length(graph)

def _from_adjacency(adjacency: dict[str, set[str]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(adjacency.keys())
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)
    return graph