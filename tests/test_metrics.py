from agents.schemas import ActionType, RoundOutcome
from analysis.metrics import (
    population_cooperation_rate,
    behavioral_clustering,
    cascade_size,
    order_parameter_variance_near_transition,
)

def make_outcome(round_num, a="agent_a", b="agent_b",
                 action_a=ActionType.COOPERATE, action_b=ActionType.COOPERATE):
    return RoundOutcome(
        round_num=round_num, agent_a=a, agent_b=b,
        action_a=action_a, action_b=action_b,
        payoff_a=3.0, payoff_b=3.0,
    )


class FakeAgent:
    """Minimal stand-in for Agent -- only needs behavioral_signature()."""
    def __init__(self, signature: str):
        self._sig = signature

    def behavioral_signature(self) -> str:
        return self._sig
    
# ── population_cooperation_rate ──────────

def test_all_cooperate_gives_rate_one():
    history = [make_outcome(r) for r in range(1, 6)]
    assert population_cooperation_rate(history) == 1.0


def test_all_defect_gives_rate_zero():
    history = [
        make_outcome(r, action_a=ActionType.DEFECT, action_b=ActionType.DEFECT)
        for r in range(1, 6)
    ]
    assert population_cooperation_rate(history) == 0.0


def test_half_cooperate_gives_rate_half():
    history = [
        make_outcome(1, action_a=ActionType.COOPERATE, action_b=ActionType.DEFECT),
        make_outcome(2, action_a=ActionType.DEFECT, action_b=ActionType.COOPERATE),
    ]
    assert population_cooperation_rate(history) == 0.5


def test_burn_in_excludes_early_rounds():
    early = [
        make_outcome(r, action_a=ActionType.COOPERATE, action_b=ActionType.COOPERATE)
        for r in range(1, 6)
    ]
    late = [
        make_outcome(r, action_a=ActionType.DEFECT, action_b=ActionType.DEFECT)
        for r in range(6, 11)
    ]
    history = early + late
    # with burn_in=5, only rounds 6-10 count: all defect -> rate = 0.0
    assert population_cooperation_rate(history, burn_in=5) == 0.0
    # without burn_in, all rounds count: 50/50 -> rate = 0.5
    assert population_cooperation_rate(history, burn_in=0) == 0.5


def test_empty_history_returns_zero():
    assert population_cooperation_rate([]) == 0.0


def test_offer_resource_counts_as_cooperative():
    history = [make_outcome(1, action_a=ActionType.OFFER_RESOURCE,
                            action_b=ActionType.OFFER_RESOURCE)]
    assert population_cooperation_rate(history) == 1.0


# ── behavioral_clustering ──────────

def test_all_same_phase_gives_clustering_one():
    agents = {
        "agent_0": FakeAgent("mostly_cooperative"),
        "agent_1": FakeAgent("mostly_cooperative"),
        "agent_2": FakeAgent("mostly_cooperative"),
    }
    adjacency = {
        "agent_0": {"agent_1"},
        "agent_1": {"agent_0", "agent_2"},
        "agent_2": {"agent_1"},
    }
    assert behavioral_clustering(agents, adjacency) == 1.0


def test_alternating_phases_gives_clustering_zero():
    # a chain where cooperative and defecting agents alternate:
    # coop -- defect -- coop -- defect
    agents = {
        "agent_0": FakeAgent("mostly_cooperative"),
        "agent_1": FakeAgent("mostly_defecting"),
        "agent_2": FakeAgent("mostly_cooperative"),
        "agent_3": FakeAgent("mostly_defecting"),
    }
    adjacency = {
        "agent_0": {"agent_1"},
        "agent_1": {"agent_0", "agent_2"},
        "agent_2": {"agent_1", "agent_3"},
        "agent_3": {"agent_2"},
    }
    assert behavioral_clustering(agents, adjacency) == 0.0


def test_no_history_agents_excluded_from_clustering():
    agents = {
        "agent_0": FakeAgent("mostly_cooperative"),
        "agent_1": FakeAgent("no_history"),
    }
    adjacency = {
        "agent_0": {"agent_1"},
        "agent_1": {"agent_0"},
    }
    # the only edge involves a no_history agent -- should be excluded,
    # giving 0 total edges counted, returning 0.0
    assert behavioral_clustering(agents, adjacency) == 0.0


# ── order_parameter_variance_near_transition ──────────

def test_variance_computed_correctly():
    sweep = [
        {"control_param": 0.0, "cooperation_rates": [0.8, 0.8, 0.8]},
        {"control_param": 0.5, "cooperation_rates": [0.3, 0.7, 0.5]},
        {"control_param": 1.0, "cooperation_rates": [0.2, 0.2, 0.2]},
    ]
    results = order_parameter_variance_near_transition(sweep)
    # variance at p=0.0 (all 0.8) should be 0 or very close
    assert results[0]["variance"] < 0.001
    # variance at p=0.5 (mixed) should be higher
    assert results[1]["variance"] > results[0]["variance"]


def test_ci_bounds_straddle_mean():
    sweep = [{"control_param": 0.5, "cooperation_rates": [0.4, 0.5, 0.6, 0.45, 0.55]}]
    results = order_parameter_variance_near_transition(sweep)
    point = results[0]
    assert point["ci_95"][0] <= point["mean_cooperation_rate"] <= point["ci_95"][1]


def test_augmented_results_preserve_original_fields():
    sweep = [{"control_param": 0.3, "cooperation_rates": [0.5, 0.6], "my_extra": "field"}]
    results = order_parameter_variance_near_transition(sweep)
    assert results[0]["my_extra"] == "field"
    assert results[0]["control_param"] == 0.3