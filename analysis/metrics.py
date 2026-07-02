from __future__ import annotations

import statistics
from agents.schemas import ActionType, RoundOutcome

_COOPERATIVE_ACTIONS = {ActionType.COOPERATE, ActionType.OFFER_RESOURCE}

def population_cooperation_rate(
        history: list[RoundOutcome],
        burn_in: int = 0,
) -> float:
    """
    Fraction of all actions in the history (after burn_in rounds) that
    were cooperative. This is your primary order parameter.
    """
    if not history:
        return 0.0
    relevant = [o for o in history if o.round_num > burn_in]
    if not relevant:
        return 0.0
    total = 0
    coop = 0
    for outcome in relevant:
        for action in (outcome.action_a, outcome.action_b):
            total += 1
            if action in _COOPERATIVE_ACTIONS:
                coop += 1
    return coop / total if total > 0 else 0.0

def behavioral_clustering(
        agents: dict[str, object], 
        adjacency: dict[str, set[str]],
) -> float:
    """
    Fraction of neighbor pairs that share the same behavioral phase.
    High values mean the population has self-organized into spatially
    coherent clusters -- a hallmark of emergence. Low values mean
    behavioral types are randomly intermixed.
    """
    signatures = {aid: agent.behavioral_signature() for aid, agent in agents.items()}

    same = 0
    total = 0
    seen_edges: set[frozenset] = set()

    for agent_id, neighbors in adjacency.items():
        for neighbor_id in neighbors:
            edge = frozenset({agent_id, neighbor_id})
            if edge in seen_edges:
                continue
            seen_edges.add(edge)

            sig_a = signatures.get(agent_id, "no_history")
            sig_b = signatures.get(neighbor_id, "no_history")

            if sig_a == "no_history" or sig_b == "no_history":
                continue
            
            total += 1
            if sig_a == sig_b:
                same += 1
    return same / total if total > 0 else 0.0

def cascade_size(
        history: list[RoundOutcome],
        agents: dict[str, object],
        adjacency: dict[str, set[str]],
        shock_round: int,
        window: int = 5,
        signature_window: int = 10,
) -> dict:
    """
    Measure the behavioral impact of a shock or seeded defection.

    Returns a dict with:
      - 'flipped': number of agents that changed from cooperative to
                   defecting phase in the window after the shock round
      - 'max_hop_distance': furthest network distance from any initially-
                            flipped agent that another flip was observed
    """
    # classify agent phases just BEfORe the shock
    pre_shock = [o for o in history if o.round_num < shock_round]
    post_shock = [o for o in history if shock_round <= o.round_num < shock_round + window]
    
    if not pre_shock or not post_shock:
        return {"flipped": 0, "max_hop_distance": 0}
    
    def phase_from_outcomes(agent_id, outcomes, w=signature_window):
        """
        Classify an agent's phase from the last w outcomes in a list.
        """
        agent_outcomes = [
            o for o in outcomes
            if o.agent_a == agent_id or o.agent_b == agent_id
        ][-w:]
        if not agent_outcomes:
            return "no_history"
        coop = sum(
            1 for o in agent_outcomes
            for action in (
                (o.action_a if o.agent_a == agent_id else o.action_b),
            )
            if action in _COOPERATIVE_ACTIONS
        )
        rate = coop / len(agent_outcomes)
        if rate >= 0.7:
            return "mostly_cooperative"
        if rate <= 0.3:
            return "mostly_defecting"
        return "mixed"
    
    pre_phases = {aid: phase_from_outcomes(aid, pre_shock) for aid in agents}
    post_phases = {aid: phase_from_outcomes(aid, post_shock) for aid in agents}

    # an agent "flipped" if it was cooperative before and defecting after
    flipped = {
        aid for aid in agents
        if pre_phases[aid] == "mostly_cooperative"
        and post_phases[aid] == "mostly_defecting"
    }

    if not flipped:
        return {"flipped": 0, "max_hop_distance": 0}
    
    # measure how far the flip propagated in network hops from the
    # initially-flipped agents. Requires a BFS over the adjacency dict
    max_distance = _max_bfs_distance(flipped, adjacency)

    return {
        "flipped": len(flipped),
        "max_hop_distance": max_distance,
    }

def _max_bfs_distance(sources: set[str], adjacency: dict[str, set[str]]) -> int:
    """
    BFS from a set of source nodes; return the max distance reached.
    """
    visited = {s: 0 for s in sources}
    queue = list(sources)
    max_dist = 0
    while queue:
        current = queue.pop(0)
        for neighbor in adjacency.get(current, set()):
            if neighbor not in visited:
                visited[neighbor] = visited[current] + 1
                max_dist = max(max_dist, visited[neighbor])
                queue.append(neighbor)
    return max_dist

def order_parameter_variance_near_transition(
        sweep_results: list[dict],
) -> list[dict]:
    """
    For each sweep point, compute the variance in cooperation rate
    across replicates. A spike here near the transition is evidence of
    critical fluctuations -- a signature of a genuine phase transition
    rather than a smooth crossover.

    Expects sweep_results as a list of dicts, each with:
      'control_param': float
      'cooperation_rates': list[float]  (one per replicate)
    Returns the same list augmented with 'variance' and 'ci_95' keys
    """
    augmented = []
    for point in sweep_results:
        rates = point["cooperation_rates"]
        n = len(rates)
        mean = statistics.mean(rates)
        variance = statistics.variance(rates) if n > 1 else 0.0
        stdev = statistics.stdev(rates) if n > 1 else 0.0
        margin = 1.96 * (stdev / (n ** 0.5)) if n > 1 else 0.0
        augmented.append({
            **point,
            "mean_cooperation_rate": mean,
            "variance": variance,
            "ci_95": (mean - margin, mean + margin),
        })
    return augmented
