from agents.agent import Agent, AgentConfig
from agents.llm_client import MockLLMClient
from network.topology import ring_lattice
from sim.orchestrator import Simulation, SimulationConfig

def make_population(agent_ids: list[str], seed: int = 0) -> list[Agent]:
    client = MockLLMClient(seed=seed)
    return [Agent(AgentConfig(agent_id=aid), llm_client=client) for aid in agent_ids]

def make_ring_sim(n: int = 6, k: int = 2, n_rounds: int = 5, seed: int = 0):
    agent_ids = [f"agent_{i}" for i in range(n)]
    agents = make_population(agent_ids, seed=seed)
    adjacency = ring_lattice(n=n, k=k, seed=seed)
    config = SimulationConfig(n_rounds=n_rounds, seed=seed)
    return Simulation(agents, config, adjacency)

def test_simulation_requires_at_least_two_agents():
    client = MockLLMClient(seed=0)
    solo = [Agent(AgentConfig(agent_id="agent_0"), llm_client=client)]
    adjacency = {"agent_0": set()}
    try:
        Simulation(solo, SimulationConfig(), adjacency)
        assert False, "expected ValueError"
    except ValueError:
        pass

def test_simulation_produces_outcomes():
    sim = make_ring_sim(n=6, k=2, n_rounds=5)
    history = sim.run()
    assert len(history) > 0

def test_agents_only_interact_with_neighbors():
    sim = make_ring_sim(n=6, k=2, n_rounds=10, seed=1)
    history = sim.run()
    adjacency = ring_lattice(n=6, k=2, seed=1)
    for outcome in history:
        a, b = outcome.agent_a, outcome.agent_b
        assert b in adjacency[a], f"{a} interacted with non-neighbor {b}"
        assert a in adjacency[b], f"{b} interacted with non-neighbor {a}"

def test_scarcity_shock_rounds_flagged_correctly():
    agent_ids = [f"agent_{i}" for i in range(4)]
    agents = make_population(agent_ids)
    adjacency = ring_lattice(n=4, k=2, seed=0)
    config = SimulationConfig(n_rounds=6, scarcity_shock_rounds=(3, 4), seed=0)
    sim = Simulation(agents, config, adjacency)
    history = sim.run()
    shocked = {o.round_num for o in history if o.scarcity_shock}
    assert shocked == {3, 4}

def test_all_outcomes_have_valid_action_types():
    from agents.schemas import ActionType
    sim = make_ring_sim(n=4, k=2, n_rounds=5)
    history = sim.run()
    for outcome in history:
        assert isinstance(outcome.action_a, ActionType)
        assert isinstance(outcome.action_b, ActionType)

def test_every_agent_has_decision_log_entries_after_run():
    sim = make_ring_sim(n=4, k=2, n_rounds=5)
    sim.run()
    for agent in sim.agents.values():
        assert len(agent.decision_log) > 0

def test_reproducible_with_same_seed():
    def run_and_extract(seed):
        sim = make_ring_sim(n=6, k=2, n_rounds=8, seed=seed)
        history = sim.run()
        return [(o.action_a.value, o.action_b.value) for o in history]

    assert run_and_extract(42) == run_and_extract(42)

def test_different_seeds_produce_different_trajectories():
    def run_and_extract(seed):
        sim = make_ring_sim(n=6, k=2, n_rounds=8, seed=seed)
        history = sim.run()
        return [(o.action_a.value, o.action_b.value) for o in history]

    # not guaranteed for all possible seeds but reliable enough at this size
    assert run_and_extract(1) != run_and_extract(2)

def test_export_log_produces_valid_json(tmp_path):
    import json
    sim = make_ring_sim(n=4, k=2, n_rounds=3)
    sim.run()
    out = tmp_path / "log.json"
    sim.export_log(str(out))
    with open(out) as f:
        data = json.load(f)
    assert "outcomes" in data
    assert "agent_decision_logs" in data
    assert "config" in data
    assert len(data["outcomes"]) > 0

def test_affect_state_changes_after_run():
    # after running, agents should have different affect states than
    # their initial defaults -- confirms observe_outcome is actually
    # being called and updating state
    sim = make_ring_sim(n=4, k=2, n_rounds=10, seed=7)
    initial_trusts = {aid: 0.5 for aid in sim.agents}
    sim.run()
    final_trusts = {aid: agent.affect.trust for aid, agent in sim.agents.items()}
    # at least one agent should have a different trust level after 10 rounds
    assert any(
        abs(final_trusts[aid] - initial_trusts[aid]) > 0.01
        for aid in sim.agents
    )