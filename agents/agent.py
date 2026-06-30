from __future__ import annotations

from dataclasses import dataclass
from collections import deque

from agents.llm_client import LLMClient
from agents.memory import AgentMemory
from agents.schemas import ActionType, AffectState, Message, RoundOutcome

@dataclass
class AgentConfig:
    agent_id: str
    persona: str = "a rational self-interested actor in a repeated trust game"
    base_cooperativeness: float = 0.5

class Agent:
    def __init__(self, config: AgentConfig, llm_client: LLMClient) -> None:
        self.id = config.agent_id
        self.persona = config.persona
        self.llm = llm_client
        self.memory = AgentMemory(agent_id=self.id)
        self.affect = AffectState(trust=config.base_cooperativeness)
        self.decision_log: list[dict] = []


    def available_actions(self) -> list[ActionType]:
        actions = [
            ActionType.COOPERATE,
            ActionType.DEFECT,
            ActionType.SIGNAL_THREAT,
            ActionType.SIGNAL_TRUST,
            ActionType.OFFER_RESOURCE,
            ActionType.WITHHOLD,
        ]
        if self.affect.resentment > 0.6:
            actions.remove(ActionType.COOPERATE)
        if self.affect.fear > 0.6:
            if ActionType.OFFER_RESOURCE in actions:
                actions.remove(ActionType.OFFER_RESOURCE)
            if ActionType.SIGNAL_TRUST in actions:
                actions.remove(ActionType.SIGNAL_TRUST)
        return actions

    def build_prompt(self, partner_id: str, round_num: int, scarcity_shock: bool) -> tuple[str, str]:
        system = (
            f"You are agent {self.id}, {self.persona}. "
            "You are participating in a repeated interaction with other agents. "
            "Each round you choose exactly one action toward your current partner. "
            "Respond with a single JSON object: "
            '{"action": "<one of the allowed actions>", "rationale": "<one sentence>"}. '
            "No text outside the JSON object."
        )

        allowed = [a.value for a in self.available_actions()]
        history = self.memory.episodic_summary_text(partner_id)
        mental_model = self.memory.mental_model_text(partner_id)
        affect_text = self.affect.to_prompt_fragment()

        scarcity_note = (
            "\nNOTE: A resource scarcity shock has hit the population this round. "
            "Resources are scarcer than usual for everyone."
            if scarcity_shock else ""
        )

        user = (
            f"Round {round_num}. Your partner this round: {partner_id}.\n\n"
            f"Interaction history with {partner_id}:\n{history}\n\n"
            f"{mental_model}\n\n"
            f"{affect_text}\n"
            f"{scarcity_note}\n\n"
            f"Allowed actions this round: {allowed}\n"
            "Choose one action and give a one-sentence rationale."
        )
        return system, user
    
    def decide(self, partner_id: str, round_num: int, scarcity_shock: bool = False) -> Message:
        system, user = self.build_prompt(partner_id, round_num, scarcity_shock)
        response = self.llm.complete(system, user)

        action = self._resolve_action(response.parsed)
        rationale = (
            (response.parsed or {}).get("rationale", "")
            if response.parsed else "[unparsed response]"
        )

        self.decision_log.append({
            "round": round_num,
            "partner": partner_id,
            "action": action.value,
            "rationale": rationale,
            "affect_snapshot": {
                "trust": self.affect.trust,
                "fear": self.affect.fear,
                "resentment": self.affect.resentment,
            },
            "raw_llm_output": response.raw_text,
            "parse_succeeded": response.parsed is not None,
        })

        return Message(
            sender_id=self.id,
            receiver_id=partner_id,
            round_num=round_num,
            action=action,
            rationale=rationale
        )

    def _resolve_action(self, parsed: dict | None) -> ActionType:
        allowed = set(self.available_actions())
        if not parsed or "action" not in parsed:
            return ActionType.WITHHOLD
        try:
            candidate = ActionType(parsed["action"])
        except ValueError:
            return ActionType.WITHHOLD
        return candidate if candidate in allowed else ActionType.WITHHOLD
    
    def observe_outcome(self, outcome: RoundOutcome) -> None:
        self.memory.record_episode(outcome)
        
        if outcome.agent_a == self.id:
            my_action, their_action = outcome.action_a, outcome.action_b
        else:
            my_action, their_action = outcome.action_b, outcome.action_a

        betrayed = (
            my_action in (ActionType.COOPERATE, ActionType.OFFER_RESOURCE)
            and their_action == ActionType.DEFECT
        )
        cooperated_with = their_action in (ActionType.COOPERATE, ActionType.OFFER_RESOURCE)

        self.affect.apply_event(
            betrayed=betrayed,
            cooperated_with=cooperated_with,
            scarcity=outcome.scarcity_shock,
        )

    def end_round_decay(self) -> None:
        self.affect.decay()
        self.memory.clear_working()

    def behavioral_signature(self, window: int = 10) -> str:
        recent = self.decision_log[window]
        if not recent:
            return "no_history"
        coop_actions = {"cooperate", "offer_resource"}
        coop_count = sum(1 for entry in recent if entry["action"] in coop_actions)
        rate = coop_count / len(recent)
        if rate >= 0.7:
            return "mostly_cooperative"
        if rate <= 0.3:
            return "mostly_defection"
        return "mixed"
