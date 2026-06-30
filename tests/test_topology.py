from network.topology import (
    ring_lattice,
    watts_strogatz,
    erdos_renyi,
    clustering_coefficient,
    average_path_length,
)

def test_ring_lattice_every_node_has_degree_k():
    adjacency = ring_lattice(n=20, k=4, seed=0)
    for node, neighbors in adjacency.items():
        assert len(neighbors) == 4

def test_ring_lattice_has_high_clustering():
    adjacency = ring_lattice(n=20, k=4, seed=0)
    coeff = clustering_coefficient(adjacency)
    # a ring lattice is maximally clustered relative to a random graph
    # of the same size/degree. Exact value depends on k, but it should
    # be well above what a random graph of the same density would show
    assert coeff > 0.3

def test_watts_strogatz_at_p_zero_matches_ring_lattice():
    ring = ring_lattice(n=20, k=4, seed=0)
    ws_zero = watts_strogatz(n=20, k=4, p=0.0, seed=0)
    assert ring == ws_zero

def test_watts_strogatz_preserves_total_edge_count_across_rewiring():
    # per-node degree is NOT strictly conserved by rewiring, only the
    # total edge count is.
    def edge_count(adjacency):
        return sum(len(neighbors) for neighbors in adjacency.values()) // 2

    base_edges = edge_count(watts_strogatz(n=20, k=4, p=0.0, seed=1))
    for p in (0.0, 0.1, 0.5, 1.0):
        adjacency = watts_strogatz(n=20, k=4, p=p, seed=1)
        assert edge_count(adjacency) == base_edges, f"edge count changed at p={p}"


def test_watts_strogatz_average_degree_close_to_k_across_rewiring():
    # individual nodes can drift from k, but the population average should
    # stay close to k throughout the sweep range
    for p in (0.0, 0.1, 0.5, 1.0):
        adjacency = watts_strogatz(n=20, k=4, p=p, seed=1)
        avg_degree = sum(len(n) for n in adjacency.values()) / len(adjacency)
        assert abs(avg_degree - 4) < 0.5

def test_watts_strogatz_clustering_decreases_as_p_increases():
    low_p = clustering_coefficient(watts_strogatz(n=30, k=6, p=0.0, seed=2))
    high_p = clustering_coefficient(watts_strogatz(n=30, k=6, p=1.0, seed=2))
    assert high_p < low_p

def test_erdos_renyi_produces_requested_node_count():
    adjacency = erdos_renyi(n=15, p=0.3, seed=3)
    assert len(adjacency) == 15

def test_topology_generation_is_reproducible_with_same_seed():
    graph_a = watts_strogatz(n=20, k=4, p=0.3, seed=42)
    graph_b = watts_strogatz(n=20, k=4, p=0.3, seed=42)
    assert graph_a == graph_b

def test_topology_generation_differs_with_different_seed():
    graph_a = watts_strogatz(n=20, k=4, p=0.3, seed=1)
    graph_b = watts_strogatz(n=20, k=4, p=0.3, seed=2)
    # not a strict guarantee for all n/k/p combos, but true often enough
    # at this size/p that a persistent failure here is worth investigating
    assert graph_a != graph_b

def test_average_path_length_handles_disconnected_graph_without_raising():
    adjacency = erdos_renyi(n=10, p=0.05, seed=4)
    # should not raise regardless of how fragmented the graph is --
    # but with very sparse graphs the "largest connected component"
    # can legitimately be a single isolated node, giving a path length of 0
    length = average_path_length(adjacency)
    assert length >= 0

def test_average_path_length_handles_disconnected_graph_without_raising():
    # a moderately sparse but not pathologically sparse graph -- likely
    # disconnected into a few components, but the largest one should
    # have real internal structure to measure
    adjacency = erdos_renyi(n=30, p=0.08, seed=4)
    length = average_path_length(adjacency)
    assert length > 0

def test_average_path_length_handles_fully_isolated_graph():
    # the genuinely degenerate case: confirm it doesn't raise, but don't
    # assert a specific meaningful value since none exists
    adjacency = {"agent_0": set(), "agent_1": set(), "agent_2": set()}
    length = average_path_length(adjacency)
    assert length == 0  # single-node "component" has no paths, by definition    

def test_adjacency_is_symmetric():
    # if agent_x is a neighbor of agent_y, agent_y must be a neighbor of agent_x
    adjacency = watts_strogatz(n=20, k=4, p=0.4, seed=5)
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            assert node in adjacency[neighbor], f"{node}->{neighbor} not symmetric"