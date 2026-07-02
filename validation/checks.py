from __future__ import annotations

import statistics
from dataclasses import dataclass

from agents.agent import Agent, AgentConfig
from agents.llm_client import get_client
from agents.schemas import ActionType, RoundOutcome
from analysis.metrics import population_cooperation_rate
from network.topology import watts_strogatz, ring_lattice, erdos_renyi
from sim.orchestrator import Simulation, SimulationConfig

class _FrozenAffectAgent(Agent):
    """
    An Agent whose affect state never changes.
    """

    def observe_outcome(self, outcome: RoundOutcome) -> None:
        self.memory.record_episode(outcome)

    def end_round_decay(self) -> None:
        self.memory.clear_working()

@dataclass
class AblationResult:
    cooperation_rate_with_affect: float
    cooperation_rate_without_affect: float
    difference: float
    interpretation: str
    is_meaningful: bool

def run_affect_ablation(
    agent_ids: list[str],
    adjacency: dict[str, set[str]],
    n_rounds: int = 30,
    burn_in: int = 0,
    scarcity_shock_rounds: tuple[int, ...] = (),
    seed: int = 0,
    llm_kind: str = "mock",
    noise_threshold: float = 0.05,
) -> AblationResult:
    
    def _run(agent_cls) -> float:
        client = get_client(llm_kind, seed=seed) if llm_kind == "mock" else get_client(llm_kind)
        agents = [agent_cls(AgentConfig(agent_id=aid), llm_client=client) for aid in agent_ids]
        sim = Simulation(
            agents=agents,
            config=SimulationConfig(
                n_rounds=n_rounds,
                scarcity_shock_rounds=scarcity_shock_rounds,
                seed=seed,
            ),
            adjacency=adjacency,
        )
        return population_cooperation_rate(sim.run(), burn_in=burn_in)
    
    with_affect = _run(Agent)
    without_affect = _run(_FrozenAffectAgent)
    diff = with_affect - without_affect
    meaningful = abs(diff) >= noise_threshold

    if not meaningful:
        interpretation = (
            f"Difference of {diff:+.3f} is below the noise threshold ({noise_threshold}). "
            "Affect state may not be meaningfully influencing behavior at this "
            "parameter setting. Check prompt salience of affect cues and whether "
            "resentment/fear thresholds are being reached in practice before "
            "making affect-driven claims."
        )
    else:
        direction = "lower" if diff < 0 else "higher"
        interpretation = (
            f"Affect dynamics produce a {direction} cooperation rate "
            f"({with_affect:.3f} vs {without_affect:.3f} without affect, "
            f"difference {diff:+.3f}), consistent with affect having a real "
            "behavioral effect at this parameter setting."
        )
    return AblationResult(
        cooperation_rate_with_affect=with_affect,
        cooperation_rate_without_affect=without_affect,
        difference=diff,
        interpretation=interpretation,
        is_meaningful=meaningful,
    )

@dataclass
class TopologyAblationResult:
    cooperation_rate_structured: float
    cooperation_rate_random: float
    difference: float
    structured_topology: str
    n_edges_structured: int
    n_edges_random: int
    edge_counts_match: bool
    interpretation: str

def run_topology_ablation(
    agent_ids: list[str],
    structured_adjacency: dict[str, set[str]],
    n_rounds: int = 30,
    burn_in: int = 0,
    scarcity_shock_rounds: tuple[int, ...] = (),
    seed: int = 0,
    llm_kind: str = "mock",
    structured_label: str = "ring_lattice",    
) -> TopologyAblationResult:
    """
    Compare behavior on a structured topology vs a degree-matched
    random graph with the same total edge count.
    """

    def _count_edges(adjacency: dict[str, set[str]]) -> int:
        return sum(len(neighbors) for neighbors in adjacency.values())
    
    def _make_random_matched(adjacency, seed) -> dict[str, set[str]]:
        """
        Build an Erdos-Renyi graph matched to the structured graph's
        edge density (p = observed_edges / possible_edges).
        """
        n = len(adjacency)
        n_edges = _count_edges(adjacency)
        possible = n * (n - 1) / 2
        p_matched = n_edges / possible if possible > 0 else 0.0
        return erdos_renyi(n=n, p=p_matched, seed=seed)

    def _run(adjacency) -> float:
        client = get_client(llm_kind, seed=seed) \
            if llm_kind == "mock" else get_client(llm_kind)
        agents = [
            Agent(AgentConfig(agent_id=aid), llm_client=client)
            for aid in agent_ids
        ]
        sim = Simulation(
            agents=agents,
            config=SimulationConfig(
                n_rounds=n_rounds,
                scarcity_shock_rounds=scarcity_shock_rounds,
                seed=seed,
            ),
            adjacency=adjacency,
        )
        return population_cooperation_rate(sim.run(), burn_in=burn_in)

    random_adjacency = _make_random_matched(structured_adjacency, seed=seed + 1)

    n_edges_structured = _count_edges(structured_adjacency)
    n_edges_random = _count_edges(random_adjacency)
    edge_counts_match = abs(n_edges_structured - n_edges_random) <= 1

    rate_structured = _run(structured_adjacency)
    rate_random = _run(random_adjacency)
    diff = rate_structured - rate_random

    if not edge_counts_match:
        interpretation = (
            f"WARNING: edge counts differ ({n_edges_structured} structured vs "
            f"{n_edges_random} random). The comparison may be confounded by "
            "connectivity differences, not just structural ones. Recheck the "
            "degree-matching logic before interpreting this result."
        )
    elif abs(diff) < 0.05:
        interpretation = (
            f"Minimal difference ({diff:+.3f}) between structured ({structured_label}) "
            "and degree-matched random graph. Network structure (clustering, path length) "
            "may not be the primary driver of behavior at this parameter setting -- "
            "the effect could be explained by connectivity alone."
        )
    else:
        direction = "higher" if diff > 0 else "lower"
        interpretation = (
            f"Cooperation rate is {direction} on the {structured_label} topology "
            f"({rate_structured:.3f}) than on a degree-matched random graph "
            f"({rate_random:.3f}, difference {diff:+.3f}), consistent with network "
            "structure (not just connectivity) affecting collective behavior."
        )
    
    return TopologyAblationResult(
        cooperation_rate_structured=rate_structured,
        cooperation_rate_random=rate_random,
        difference=diff,
        structured_topology=structured_label,
        n_edges_structured=n_edges_structured,
        n_edges_random=n_edges_random,
        edge_counts_match=edge_counts_match,
        interpretation=interpretation,
    )

@dataclass
class FiniteSizeResult:
    population_sizes: list[int]
    mean_cooperation_rates: list[float]
    transition_sharpness: list[float]  # stdev across sweep at each size
    interpretation: str


def run_finite_size_check(
    population_sizes: list[int],
    control_param_values: list[float],
    n_replicates: int = 5,
    n_rounds: int = 20,
    burn_in: int = 0,
    k: int = 4,
    base_seed: int = 0,
    llm_kind: str = "mock",
) -> FiniteSizeResult:
    """
    Run a coarse sweep at multiple population sizes.

    For a real phase transition, two things should happen as N grows:
      1. The mean curve stays roughly the same shape (transition at
         the same critical parameter value)
      2. The *variance* across sweep points spikes more sharply at
         the critical point (the transition sharpens)
    """
    from sim.sweep import run_sweep

    mean_rates_by_size = []
    sharpness_by_size = []

    for n in population_sizes:
        agent_ids = [f"agent_{i}" for i in range(n)]
        results = run_sweep(
            control_param_values=control_param_values,
            n_replicates=n_replicates,
            agent_ids=agent_ids,
            n_rounds=n_rounds,
            burn_in=burn_in,
            k=k,
            base_seed=base_seed,
            llm_kind=llm_kind,
        )
        means = [r.mean_cooperation_rate for r in results]
        mean_rates_by_size.append(statistics.mean(means))
        # sharpness: stdev of the cooperation rate *across sweep points*
        # (not across replicates) -- a sharper transition = higher variance
        # across the p-axis at the same N
        sharpness_by_size.append(statistics.stdev(means) if len(means) > 1 else 0.0)

    sharpness_increases = all(
        sharpness_by_size[i] <= sharpness_by_size[i + 1]
        for i in range(len(sharpness_by_size) - 1)
    )

    if sharpness_increases:
        interpretation = (
            "Transition sharpness increases with population size "
            f"({', '.join(f'{s:.3f}' for s in sharpness_by_size)} for N="
            f"{', '.join(str(n) for n in population_sizes)}), "
            "consistent with a genuine finite-size phase transition."
        )
    else:
        interpretation = (
            "Transition sharpness does not consistently increase with population "
            f"size ({', '.join(f'{s:.3f}' for s in sharpness_by_size)} for N="
            f"{', '.join(str(n) for n in population_sizes)}). "
            "The apparent transition may reflect sampling noise rather than "
            "a genuine phase transition. Consider more replicates or a wider "
            "parameter range before drawing strong conclusions."
        )

    return FiniteSizeResult(
        population_sizes=population_sizes,
        mean_cooperation_rates=mean_rates_by_size,
        transition_sharpness=sharpness_by_size,
        interpretation=interpretation,
    )

@dataclass
class HysteresisResult:
    max_gap: float
    mean_gap: float
    forward_rates: list[float]
    backward_rates: list[float]
    control_params: list[float]
    shows_hysteresis: bool
    interpretation: str


def summarize_hysteresis(
    hysteresis_sweep: dict,
    gap_threshold: float = 0.05,
) -> HysteresisResult:
    """
    Interpret a hysteresis sweep result from run_hysteresis_sweep().
    """
    forward = hysteresis_sweep["forward"]
    backward = hysteresis_sweep["backward"]

    forward_by_param = {r.control_param: r.mean_cooperation_rate for r in forward}
    backward_by_param = {r.control_param: r.mean_cooperation_rate for r in backward}
    common_params = sorted(set(forward_by_param) & set(backward_by_param))

    if not common_params:
        return HysteresisResult(
            max_gap=0.0, mean_gap=0.0,
            forward_rates=[], backward_rates=[],
            control_params=[],
            shows_hysteresis=False,
            interpretation="No common parameter values between forward and backward sweeps.",
        )

    gaps = [abs(forward_by_param[p] - backward_by_param[p]) for p in common_params]
    max_gap = max(gaps)
    mean_gap = statistics.mean(gaps)
    shows_hysteresis = max_gap >= gap_threshold

    if shows_hysteresis:
        interpretation = (
            f"Forward and backward sweeps differ by up to {max_gap:.3f} "
            f"(mean gap {mean_gap:.3f}), exceeding the threshold of {gap_threshold}. "
            "This is consistent with path dependence: the system's state depends "
            "not just on the current parameter value but on its history of "
            "parameter values. A hallmark of complex adaptive systems."
        )
    else:
        interpretation = (
            f"Forward and backward sweeps agree closely (max gap {max_gap:.3f}, "
            f"mean gap {mean_gap:.3f}, threshold {gap_threshold}). "
            "No strong evidence of path dependence at this population size and "
            "parameter range. This is a valid finding -- not every CAS shows "
            "hysteresis, and reporting its absence is as useful as reporting "
            "its presence."
        )

    return HysteresisResult(
        max_gap=max_gap,
        mean_gap=mean_gap,
        forward_rates=[forward_by_param[p] for p in common_params],
        backward_rates=[backward_by_param[p] for p in common_params],
        control_params=common_params,
        shows_hysteresis=shows_hysteresis,
        interpretation=interpretation,
    )

_COOPERATIVE_WORDS = {
    "trust", "build", "mutual", "benefit", "honest",
    "repair", "cooperate", "reward", "together", "share",
}
_DEFENSIVE_WORDS = {
    "defect", "risk", "protect", "punish", "retaliat",
    "exploit", "betray", "guard", "withhold", "threat",
    "distrust", "suspicious",
}


@dataclass
class CoherenceResult:
    agent_id: str
    n_checked: int
    n_consistent: int
    n_inconsistent: int
    n_skipped: int
    consistency_rate: float
    inconsistent_examples: list[dict]
    interpretation: str


def check_rationale_coherence(
    agent,
    max_examples: int = 5,
) -> CoherenceResult:
    """
    Heuristic check: does an agent's stated rationale directionally
    match its chosen action?
    """
    cooperative_actions = {"cooperate", "offer_resource", "signal_trust"}
    n_checked = n_consistent = n_skipped = 0
    examples = []

    for entry in agent.decision_log:
        rationale = entry.get("rationale", "")
        action = entry.get("action", "")

        if not rationale or rationale == "[unparsed response]":
            n_skipped += 1
            continue

        n_checked += 1
        rationale_lower = rationale.lower()

        coop_hits = sum(1 for w in _COOPERATIVE_WORDS if w in rationale_lower)
        def_hits = sum(1 for w in _DEFENSIVE_WORDS if w in rationale_lower)
        action_is_coop = action in cooperative_actions

        # consistent cases:
        #   - action cooperative, rationale not defensively leaning
        #   - action defensive, rationale not cooperatively leaning
        #   - no signal either way (can't call it inconsistent)
        no_signal = coop_hits == 0 and def_hits == 0
        consistent = (
            no_signal
            or (action_is_coop and coop_hits >= def_hits)
            or (not action_is_coop and def_hits >= coop_hits)
        )

        if consistent:
            n_consistent += 1
        elif len(examples) < max_examples:
            examples.append({
                "round": entry.get("round"),
                "partner": entry.get("partner"),
                "action": action,
                "rationale": rationale,
                "coop_hits": coop_hits,
                "def_hits": def_hits,
            })

    n_inconsistent = n_checked - n_consistent
    rate = n_consistent / n_checked if n_checked > 0 else 1.0

    interpretation = (
        f"Consistency rate: {rate:.0%} ({n_consistent}/{n_checked} checked, "
        f"{n_skipped} skipped due to parse failures). "
        "NOTE: This is a keyword-direction heuristic, not a causal proof. "
        "Flagged inconsistencies are candidates for manual review, not "
        "confirmed reasoning failures."
    )

    return CoherenceResult(
        agent_id=agent.id,
        n_checked=n_checked,
        n_consistent=n_consistent,
        n_inconsistent=n_inconsistent,
        n_skipped=n_skipped,
        consistency_rate=rate,
        inconsistent_examples=examples,
        interpretation=interpretation,
    )


def check_population_coherence(agents: dict) -> list[CoherenceResult]:
    """Run coherence check across all agents and return a list of results."""
    return [check_rationale_coherence(agent) for agent in agents.values()]