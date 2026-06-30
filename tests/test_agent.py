# tests/test_agent.py

from agents.agent import Agent, AgentConfig
from agents.llm_client import MockLLMClient
from agents.schemas import ActionType, RoundOutcome


def make_agent(agent_id="agent_a", seed=0, base_cooperativeness=0.5):
    client = MockLLMClient(seed=seed)
    return Agent(AgentConfig(agent_id=agent_id, base_cooperativeness=base_cooperativeness), llm_client=client)


def test_neutral_affect_allows_all_six_actions():
    agent = make_agent()
    assert len(agent.available_actions()) == 6


def test_high_resentment_removes_cooperate():
    agent = make_agent()
    agent.affect.resentment = 0.9
    actions = agent.available_actions()
    assert ActionType.COOPERATE not in actions
    assert ActionType.DEFECT in actions  # unrelated action unaffected


def test_high_fear_removes_offer_resource_and_signal_trust():
    agent = make_agent()
    agent.affect.fear = 0.9
    actions = agent.available_actions()
    assert ActionType.OFFER_RESOURCE not in actions
    assert ActionType.SIGNAL_TRUST not in actions
    assert ActionType.COOPERATE in actions  # unrelated action unaffected


def test_high_resentment_and_high_fear_combined_dont_crash():
    agent = make_agent()
    agent.affect.resentment = 0.9
    agent.affect.fear = 0.9
    actions = agent.available_actions()
    assert ActionType.COOPERATE not in actions
    assert ActionType.OFFER_RESOURCE not in actions
    assert ActionType.SIGNAL_TRUST not in actions


def test_resolve_action_falls_back_to_withhold_on_unparsed_response():
    agent = make_agent()
    assert agent._resolve_action(None) == ActionType.WITHHOLD


def test_resolve_action_falls_back_to_withhold_on_hallucinated_action():
    agent = make_agent()
    assert agent._resolve_action({"action": "betray_everyone"}) == ActionType.WITHHOLD


def test_resolve_action_falls_back_to_withhold_on_gated_out_action():
    agent = make_agent()
    agent.affect.resentment = 0.9  # COOPERATE is now gated out
    assert agent._resolve_action({"action": "cooperate"}) == ActionType.WITHHOLD


def test_resolve_action_accepts_valid_ungated_action():
    agent = make_agent()
    assert agent._resolve_action({"action": "defect"}) == ActionType.DEFECT


def test_decide_returns_properly_typed_message():
    agent = make_agent()
    msg = agent.decide(partner_id="agent_b", round_num=1)
    assert msg.sender_id == "agent_a"
    assert msg.receiver_id == "agent_b"
    assert msg.round_num == 1
    assert isinstance(msg.action, ActionType)


def test_decide_appends_one_entry_to_decision_log():
    agent = make_agent()
    agent.decide(partner_id="agent_b", round_num=1)
    assert len(agent.decision_log) == 1
    entry = agent.decision_log[0]
    assert entry["round"] == 1
    assert entry["partner"] == "agent_b"
    assert "affect_snapshot" in entry
    assert entry["affect_snapshot"]["trust"] == agent.affect.trust


def test_observe_outcome_betrayal_only_fires_when_i_cooperated_and_they_defected():
    agent = make_agent("agent_a")
    trust_before = agent.affect.trust
    outcome = RoundOutcome(
        round_num=1, agent_a="agent_a", agent_b="agent_b",
        action_a=ActionType.COOPERATE, action_b=ActionType.DEFECT,
        payoff_a=0.0, payoff_b=5.0,
    )
    agent.observe_outcome(outcome)
    assert agent.affect.trust < trust_before  # betrayal lowered trust


def test_observe_outcome_mutual_defection_is_not_betrayal():
    agent = make_agent("agent_a")
    trust_before = agent.affect.trust
    fear_before = agent.affect.fear
    outcome = RoundOutcome(
        round_num=1, agent_a="agent_a", agent_b="agent_b",
        action_a=ActionType.DEFECT, action_b=ActionType.DEFECT,
        payoff_a=1.0, payoff_b=1.0,
    )
    agent.observe_outcome(outcome)
    # neither cooperated_with nor betrayed should fire here, so fear/trust shouldn't move
    assert agent.affect.fear == fear_before
    assert agent.affect.trust == trust_before


def test_end_round_decay_clears_working_memory():
    agent = make_agent()
    agent.memory.set_working("scratch", 123)
    agent.end_round_decay()
    assert agent.memory.working == {}


def test_behavioral_signature_no_history():
    agent = make_agent()
    assert agent.behavioral_signature() == "no_history"


def test_behavioral_signature_mostly_cooperative():
    agent = make_agent()
    for _ in range(9):
        agent.decision_log.append({"action": "cooperate"})
    agent.decision_log.append({"action": "defect"})
    assert agent.behavioral_signature() == "mostly_cooperative"


def test_behavioral_signature_mostly_defecting():
    agent = make_agent()
    for _ in range(9):
        agent.decision_log.append({"action": "defect"})
    agent.decision_log.append({"action": "cooperate"})
    assert agent.behavioral_signature() == "mostly_defecting"


def test_behavioral_signature_mixed():
    agent = make_agent()
    for i in range(10):
        agent.decision_log.append({"action": "cooperate" if i % 2 == 0 else "defect"})
    assert agent.behavioral_signature() == "mixed"


def test_behavioral_signature_only_considers_recent_window():
    agent = make_agent()
    for _ in range(20):
        agent.decision_log.append({"action": "defect"})
    for _ in range(10):
        agent.decision_log.append({"action": "cooperate"})
    # default window=10 should only see the most recent 10 (all cooperate)
    assert agent.behavioral_signature() == "mostly_cooperative"