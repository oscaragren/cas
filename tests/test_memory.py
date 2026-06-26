from __future__ import annotations
from agents.memory import AgentMemory
from agents.schemas import ActionType, RoundOutcome

def make_outcome(round_num, a="agent_a", b="agent_b", action_a=ActionType.COOPERATE, action_b=ActionType.DEFECT):
    return RoundOutcome(
        round_num=round_num, agent_a=a, agent_b=b,
        action_a=action_a, action_b=action_b,
        payoff_a=0.0, payoff_b=5.0,
    )

def test_working_memory_resets():
    mem = AgentMemory(agent_id="agent_a")
    mem.set_working("foo", "bar")
    assert mem.working["foo"] == "bar"
    mem.clear_working()
    assert mem.working == {}

def test_episodic_memory_bounded_window():
    mem = AgentMemory(agent_id="agent_a", episodic_window=3)
    for r in range(1, 6):
        mem.record_episode(make_outcome(r))
    entries = mem.episodic_for("agent_b")
    assert len(entries) == 3
    assert entries[-1].round_num == 5 # oldest two were dropped

def test_episodic_memory_irrelevant_outcome_ignored():
    mem = AgentMemory(agent_id="agent_c")  # not a participant in the outcome
    mem.record_episode(make_outcome(1))
    assert mem.episodic_for("agent_b") == []

def test_record_episode_correct_from_agent_a_prespective():
    mem = AgentMemory(agent_id="agent_a")
    mem.record_episode(make_outcome(1, action_a=ActionType.COOPERATE, action_b=ActionType.DEFECT))
    entry = mem.episodic_for("agent_b")[0]
    assert entry.my_action == ActionType.COOPERATE
    assert entry.their_action == ActionType.DEFECT
    assert entry.payoff == 0.0

def test_record_episode_correct_from_agent_b_prespective():
    mem = AgentMemory(agent_id="agent_b")
    mem.record_episode(make_outcome(1, action_a=ActionType.COOPERATE, action_b=ActionType.DEFECT))
    entry = mem.episodic_for("agent_a")[0]
    assert entry.my_action == ActionType.DEFECT
    assert entry.their_action == ActionType.COOPERATE
    assert entry.payoff == 5.0

def test_semantic_memory_reputation_updates_from_episodic():
    mem = AgentMemory(agent_id="agent_a")
    mem.record_episode(make_outcome(1, action_a=ActionType.COOPERATE, action_b=ActionType.DEFECT))
    rep = mem.reputation_of("agent_b")
    assert rep.defections_observed == 1
    assert rep.cooperations_observed == 0
    assert rep.last_action == ActionType.DEFECT

def test_mental_model_predicts_repeat_of_last_action():
    mem = AgentMemory(agent_id="agent_a")
    mem.record_episode(make_outcome(1, action_a=ActionType.COOPERATE, action_b=ActionType.DEFECT))
    rep = mem.reputation_of("agent_b")
    assert rep.predict_next_action() == ActionType.DEFECT

def test_reputation_of_unknown_partner_returns_uninformative_default():
    mem = AgentMemory(agent_id="agent_a")
    rep = mem.reputation_of("agent_never_seen")
    assert rep.cooperation_rate == 0.5
    assert rep.total_observed == 0   
