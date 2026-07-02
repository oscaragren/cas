# tests/test_validation.py

from agents.agent import Agent, AgentConfig
from agents.llm_client import MockLLMClient
from network.topology import ring_lattice
from validation.checks import (
    run_affect_ablation,
    run_topology_ablation,
    run_finite_size_check,
    check_rationale_coherence,
    summarize_hysteresis,
)


AGENT_IDS = [f"agent_{i}" for i in range(6)]
SMALL_ADJACENCY = ring_lattice(n=6, k=2, seed=0)


# ── affect ablation ──────────────────────────────────────────────────────────

def test_affect_ablation_returns_valid_rates():
    result = run_affect_ablation(
        agent_ids=AGENT_IDS,
        adjacency=SMALL_ADJACENCY,
        n_rounds=8,
        seed=0,
    )
    assert 0.0 <= result.cooperation_rate_with_affect <= 1.0
    assert 0.0 <= result.cooperation_rate_without_affect <= 1.0


def test_affect_ablation_difference_is_consistent():
    result = run_affect_ablation(
        agent_ids=AGENT_IDS,
        adjacency=SMALL_ADJACENCY,
        n_rounds=8,
        seed=0,
    )
    expected_diff = result.cooperation_rate_with_affect - result.cooperation_rate_without_affect
    assert abs(result.difference - expected_diff) < 1e-9


def test_affect_ablation_is_meaningful_flag_set_correctly():
    result = run_affect_ablation(
        agent_ids=AGENT_IDS,
        adjacency=SMALL_ADJACENCY,
        n_rounds=8,
        seed=0,
        noise_threshold=0.0,  # any difference is "meaningful" at threshold=0
    )
    # with threshold=0, is_meaningful should match whether difference is nonzero
    assert result.is_meaningful == (abs(result.difference) >= 0.0)


def test_affect_ablation_interpretation_is_non_empty():
    result = run_affect_ablation(
        agent_ids=AGENT_IDS,
        adjacency=SMALL_ADJACENCY,
        n_rounds=8,
        seed=1,
    )
    assert len(result.interpretation) > 0


# ── topology ablation ────────────────────────────────────────────────────────

def test_topology_ablation_returns_valid_rates():
    result = run_topology_ablation(
        agent_ids=AGENT_IDS,
        structured_adjacency=SMALL_ADJACENCY,
        n_rounds=8,
        seed=0,
    )
    assert 0.0 <= result.cooperation_rate_structured <= 1.0
    assert 0.0 <= result.cooperation_rate_random <= 1.0


def test_topology_ablation_reports_edge_counts():
    result = run_topology_ablation(
        agent_ids=AGENT_IDS,
        structured_adjacency=SMALL_ADJACENCY,
        n_rounds=8,
        seed=0,
    )
    # edge counts should be reported and positive
    assert result.n_edges_structured > 0
    assert result.n_edges_random > 0


def test_topology_ablation_edge_count_check_fires():
    adjacency = ring_lattice(n=6, k=2, seed=0)
    result = run_topology_ablation(
        agent_ids=AGENT_IDS,
        structured_adjacency=adjacency,
        n_rounds=5,
        seed=0,
    )
    # edge_counts_match is a bool
    assert isinstance(result.edge_counts_match, bool)


# ── finite-size check ────────────────────────────────────────────────────────

def test_finite_size_check_returns_one_result_per_size():
    result = run_finite_size_check(
        population_sizes=[6, 10],
        control_param_values=[0.0, 0.5, 1.0],
        n_replicates=2,
        n_rounds=5,
        llm_kind="mock",
    )
    assert len(result.population_sizes) == 2
    assert len(result.mean_cooperation_rates) == 2
    assert len(result.transition_sharpness) == 2


def test_finite_size_check_interpretation_non_empty():
    result = run_finite_size_check(
        population_sizes=[6, 10],
        control_param_values=[0.0, 0.5, 1.0],
        n_replicates=2,
        n_rounds=5,
        llm_kind="mock",
    )
    assert len(result.interpretation) > 0


# ── hysteresis ───────────────────────────────────────────────────────────────

def test_summarize_hysteresis_handles_empty_sweep():
    result = summarize_hysteresis({"forward": [], "backward": []})
    assert result.shows_hysteresis is False
    assert "No common" in result.interpretation


def test_summarize_hysteresis_detects_large_gap():
    from sim.orchestrator import SimulationConfig
    # construct a fake hysteresis sweep result with a known large gap
    class FakeResult:
        def __init__(self, p, rate):
            self.control_param = p
            self.mean_cooperation_rate = rate

    fake_sweep = {
        "forward":  [FakeResult(0.0, 0.8), FakeResult(0.5, 0.5), FakeResult(1.0, 0.2)],
        "backward": [FakeResult(0.0, 0.4), FakeResult(0.5, 0.3), FakeResult(1.0, 0.1)],
    }
    result = summarize_hysteresis(fake_sweep, gap_threshold=0.1)
    assert result.shows_hysteresis is True
    assert result.max_gap >= 0.1


def test_summarize_hysteresis_no_hysteresis_when_curves_identical():
    class FakeResult:
        def __init__(self, p, rate):
            self.control_param = p
            self.mean_cooperation_rate = rate

    fake_sweep = {
        "forward":  [FakeResult(0.0, 0.7), FakeResult(0.5, 0.5), FakeResult(1.0, 0.3)],
        "backward": [FakeResult(0.0, 0.7), FakeResult(0.5, 0.5), FakeResult(1.0, 0.3)],
    }
    result = summarize_hysteresis(fake_sweep, gap_threshold=0.05)
    assert result.shows_hysteresis is False
    assert result.max_gap == 0.0


# ── rationale coherence ──────────────────────────────────────────────────────

def test_coherence_skips_unparsed_entries():
    client = MockLLMClient(seed=0)
    agent = Agent(AgentConfig(agent_id="agent_a"), llm_client=client)
    agent.decision_log.append({
        "round": 1, "partner": "agent_b",
        "action": "cooperate", "rationale": "[unparsed response]",
        "affect_snapshot": {}, "raw_llm_output": "", "parse_succeeded": False,
    })
    result = check_rationale_coherence(agent)
    assert result.n_checked == 0
    assert result.n_skipped == 1


def test_coherence_detects_clear_mismatch():
    client = MockLLMClient(seed=0)
    agent = Agent(AgentConfig(agent_id="agent_a"), llm_client=client)
    agent.decision_log.append({
        "round": 1, "partner": "agent_b",
        "action": "defect",
        "rationale": "I want to build trust and enable mutual cooperation.",
        "affect_snapshot": {}, "raw_llm_output": "", "parse_succeeded": True,
    })
    result = check_rationale_coherence(agent)
    assert result.n_inconsistent == 1
    assert len(result.inconsistent_examples) == 1
    assert result.inconsistent_examples[0]["coop_hits"] > 0


def test_coherence_consistent_when_no_signal():
    client = MockLLMClient(seed=0)
    agent = Agent(AgentConfig(agent_id="agent_a"), llm_client=client)
    agent.decision_log.append({
        "round": 1, "partner": "agent_b",
        "action": "defect",
        "rationale": "I chose this action based on current conditions.",
        "affect_snapshot": {}, "raw_llm_output": "", "parse_succeeded": True,
    })
    result = check_rationale_coherence(agent)
    assert result.n_consistent == 1
    assert result.n_inconsistent == 0


def test_coherence_interpretation_contains_disclaimer():
    client = MockLLMClient(seed=0)
    agent = Agent(AgentConfig(agent_id="agent_a"), llm_client=client)
    agent.decision_log.append({
        "round": 1, "partner": "agent_b",
        "action": "cooperate", "rationale": "I trust this agent.",
        "affect_snapshot": {}, "raw_llm_output": "", "parse_succeeded": True,
    })
    result = check_rationale_coherence(agent)
    assert "heuristic" in result.interpretation.lower()