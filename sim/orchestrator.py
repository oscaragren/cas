from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field, asdict

from agents.agent import Agent
from agents.schemas import ActionType, RoundOutcome
from sim.payoffs import resolve_payoffs

@dataclass
class SimulationConfig:
    n_rounds: int = 30
    scarcity_shock_rounds: tuple[int, ...] = field(default_factory=tuple)
    seed: int | None = None
    interaction_rate: float = 1.0 # fraction of edges activated each round

class Simulation:
    def __init__(
        self,
        agents: list[Agent],
        config: SimulationConfig,
        adjacency: dict[str, set[str]],
    ) -> None:
        if len(agents) < 2:
            raise ValueError("Need at least 2 agents.")
        self.agents = {a.id: a for a in agents}
        self.config = config
        self.adjacency = adjacency
        self._rng = random.Random(config.seed)

        self.history: list[RoundOutcome] = []
        self._round_pairings: list[list[tuple[str, str]]] = []

    def _pair_agents(self) -> list[tuple[str, str]]:
        # collect all edges in the graph, as ordered (a < b) pairs to avoid
        # double-counting. Since adjacency is symmetric, every edge appears
        # as both (a, b) and (b, a), so we only keep the lexicographically first
        all_edges = [
            (a, b)
            for a, neighbors in self.adjacency.items()
            for b in neighbors
            if a < b
        ]
        
        if self.config.interaction_rate < 1.0:
            n_active = max(1, int(len(all_edges) * self.config.interaction_rate))
            all_edges = self._rng.sample(all_edges, n_active)

        return all_edges
    
    async def _run_round(self, round_num: int) -> list[RoundOutcome]:
        scarcity = round_num in self.config.scarcity_shock_rounds
        pairs = self._pair_agents()
        self._round_pairings.append(pairs)

        async def decide_pair(a_id: str, b_id: str) -> tuple:
            agent_a = self.agents[a_id]
            agent_b = self.agents[b_id]
            msg_a, msg_b = await asyncio.gather(
                asyncio.to_thread(agent_a.decide, b_id, round_num, scarcity),
                asyncio.to_thread(agent_b.decide, a_id, round_num, scarcity),
            )
            return msg_a, msg_b
    
        results = await asyncio.gather(
            *(decide_pair(a, b) for a, b in pairs)
        )

        outcomes: list[RoundOutcome] = []
        for (a_id, b_id), (msg_a, msg_b) in zip(pairs, results):
            payoff_a, payoff_b = resolve_payoffs(msg_a.action, msg_b.action, scarcity)
            outcome = RoundOutcome(
                round_num=round_num,
                agent_a=a_id,
                agent_b=b_id,
                action_a=msg_a.action,
                action_b=msg_b.action,
                payoff_a=payoff_a,
                payoff_b=payoff_b,
                scarcity_shock=scarcity,
            )
            outcomes.append(outcome)
            self.agents[a_id].observe_outcome(outcome)
            self.agents[b_id].observe_outcome(outcome)
        
        for agent in self.agents.values():
            agent.end_round_decay()

        return outcomes
    
    def run(self) -> list[RoundOutcome]:
        return asyncio.run(self._run_all())
    
    async def _run_all(self) -> list[RoundOutcome]:
        for round_num in range(1, self.config.n_rounds + 1):
            outcomes = await self._run_round(round_num)
            self.history.extend(outcomes)
        return self.history
    
    def export_log(self, path: str) -> None:
        data = {
            "config": {
                "n_rounds": self.config.n_rounds,
                "scarcity_shock_rounds": list(self.config.scarcity_shock_rounds),
                "seed": self.config.seed,
                "interaction_rate": self.config.interaction_rate,
            },
            "outcomes": [
                {
                    "round_num": o.round_num,
                    "agent_a": o.agent_a,
                    "agent_b": o.agent_b,
                    "action_a": o.action_a.value,
                    "action_b": o.action_b.value,
                    "payoff_a": o.payoff_a,
                    "payoff_b": o.payoff_b,
                    "scarcity_shock": o.scarcity_shock,
                }
                for o in self.history
            ],
            "agent_decision_logs": {
                agent_id: agent.decision_log
                for agent_id, agent in self.agents.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)