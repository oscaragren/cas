from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
import time, uuid


class ActionType(str, Enum):
    COOPORATE = "cooporate"
    DEFECT = "defect"
    SIGNAL_TRUST = "signal_trust"
    SIGNAL_THREAT = "signal_threat"
    OFFER_RESOURCE = "offer_resource"
    WITHHOLD = "withhold"

@dataclass
class Message:
    sender_id: str
    receiver_id: str | None # None = broadcast
    round_num: int
    action: ActionType
    payload: dict = field(default_factory=dict)
    rationale: str = ""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[8:])
    timestamp: float = field(default_factory=time.time)

@dataclass
class RoundOutcome:
    round_num: int
    agent_a: str
    agent_b: str
    action_a: ActionType
    action_b: ActionType
    payoff_a: float
    payoff_b: float
    scarcity_shock: bool = False

@dataclass
class AffectState:

    trust: float = 0.5
    fear: float = 0.0
    resentment: float = 0.0
    decay_rate: float = 0.05

    def decay(self) -> None:
        baseline = {"trust": 0.5, "fear": 0.0, "resentment": 0.0}
        for dim, base in baseline.items():
            current = getattr(self, dim)
            setattr(self, dim, current + (base - current) * self.decay_rate)

    def apply_event(self, betrayed: bool, cooperated_with: bool, scarcity: bool) -> None:
        if betrayed:
            self.trust = max(0.0, self.trust - 0.25)
            self.fear = min(1.0, self.fear + 0.2)
            self.resentment = min(1.0, self.resentment + 0.3)

        if cooperated_with:
            self.trust = min(1.0, self.trust + 0.15)
            self.resentment = max(0.0, self.resentment - 0.1)

        if scarcity:
            self.fear = min(1.0, self.fear + 0.15)

    def to_prompt_fragment(self) -> str:
        """LLM understand trust=high better than trust=0.73"""
        pass


@dataclass
class ReputationRecord:
    target_id: str
    cooperations_observed: int = 0
    defections_observed: int = 0
    last_action: ActionType | None = None
    resentment: float = 0.0

    @property
    def total_observed(self) -> int:
        return self.cooperations_observed + self.defections_observed
    
    @property
    def cooperate_rate(self) -> float:
        if self.total_observed == 0:
            return 0.5 # uninformative prior
        return self.cooperations_observed/self.total_observed
    
    def predict_next_action(self) -> ActionType:
        if self.last_action is not None:
            return self.last_action
        return ActionType.COOPERATE if self.cooperate_rate >= 0.5 else ActionType.DEFECT
