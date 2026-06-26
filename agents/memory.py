from __future__ import annotations
from collections import deque
from dataclasses import dataclass

from agents.schemas import ActionType, ReputationRecord, RoundOutcome

@dataclass
class EpisodicEntry:
    round_num: int
    partner_id: str
    my_action: ActionType
    their_action: ActionType
    payoff: float
    summary: str


class AgentMemory:

    def __init__(self, agent_id: str, episodic_window: int=8) -> None:
        self.agent_id = agent_id
        self.episodic_window = episodic_window

        self.working: dict = {}
        self._episodic: dict[str, deque[EpisodicEntry]] = {}
        self.reputations: dict[str, ReputationRecord] = {}

    def clear_working(self) -> None:
        self.working = {}

    def set_working(self, key: str, value) -> None:
        self.working[key] = value


    def record_episode(self, outcome: RoundOutcome) -> None:
        if outcome.agent_a == self.agent_id:
            partner, my_action, their_action, my_payoff = (
                outcome.agent_b, outcome.action_a, outcome.action_b, outcome.payoff_a
            )
        elif outcome.agent_b == self.agent_id:
            partner, my_action, their_action, my_payoff = (
                outcome.agent_a, outcome.action_b, outcome.action_a, outcome.payoff_b
            )
        else:
            return # This outcome doesn't involve this agent at all
        
        summary = (
            f"R{outcome.round_num}: I {my_action.value}, {partner} {their_action.value}, "
            f"my payoff {my_payoff:+.1f}"
        )

        entry = EpisodicEntry(
            round_num=outcome.round_num,
            partner_id=partner,
            my_action=my_action,
            their_action=their_action,
            payoff=my_payoff,
            summary=summary,
        )

        buf = self._episodic.setdefault(partner, deque(maxlen=self.episodic_window))
        buf.append(entry)

        self._update_reputation(partner, their_action)

    def episodic_for(self, partner_id: str) -> list[EpisodicEntry]:
        return list(self._episodic.get(partner_id, []))
    
    def episodic_summary_text(self, partner_id: str) -> str:
        entries = self.episodic_for(partner_id)
        if not entries:
            return f"No prior history with {partner_id}."
        return "\n".join(e.summary for e in entries)

    def _update_reputation(self, partner_id: str, their_action: ActionType) -> None:
        rec = self.reputations.setdefault(partner_id, ReputationRecord(target_id=partner_id))
        if their_action in (ActionType.COOPERATE, ActionType.OFFER_RESOURCE):
            rec.cooperations_observed += 1
        elif their_action == ActionType.DEFECT:
            rec.defections_observed += 1
            rec.resentment = min(1.0, rec.resentment + 0.3)
        rec.last_action = their_action
        rec.resentment = max(0.0, rec.resentment - 0.02)

    def reputation_of(self, partner_id: str) -> ReputationRecord:
        return self.reputations.setdefault(partner_id, ReputationRecord(target_id=partner_id))
    
    def mental_model_text(self, partner_id: str) -> str:
        rec = self.reputation_of(partner_id)
        predicted = rec.predict_next_action()
        return (
            f"Mental model of {partner_id}: observed cooperation rate "
            f"{rec.cooperation_rate:.0%} over {rec.total_observed} rounds, "
            f"your grudge level against them is {rec.resentment:.2f}, "
            f"predicted next action: {predicted.value}."
        )