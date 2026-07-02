from __future__ import annotations

import statistics
from dataclasses import dataclass

from agents.agent import Agent, AgentConfig
from agents.llm_client import get_client
from analysis.metrics import population_cooperation_rate, order_parameter_variance_near_transition
from network.topology import watts_strogatz, ring_lattice, erdos_renyi
from sim.orchestrator import Simulation, SimulationConfig

def _derive_seed(control_param: float, replicate_index: int, base_seed: int = 0) -> int:
    """
    Produce a deterministic integer seed from a (parameter, replicate) pair.
    """
    return abs(hash((round(control_param, 8), replicate_index, base_seed))) % (2**31)

@dataclass
class SweepPoint:
    control_param: float
    replicate_index: int
    seed: int
    cooperation_rate: float
    n_rounds: int
    burn_in: int

@dataclass
class SweepResult:
    control_param: float
    mean_cooperation_rate: float
    stdev: float
    ci_95: tuple[float, float]
    variance: float
    n_replicates: int
    raw_points: list[SweepPoint]

def run_sweep(
    control_param_values: list[float],
    n_replicates: int,
    agent_ids: list[str],
    n_rounds: int = 30,
    burn_in: int = 0,
    scarcity_shock_rounds: tuple[int, ...] = (),
    k: int = 4, # must be even
    base_seed: int = 0,
    llm_kind: str = "mock",
    llm_kwargs: dict | None = None,   
) -> list[SweepResult]:
    """
    Sweep over control_param_values (rewiring probability p),
    running n_replicates independent simulations at each value.
    """
    llm_kwargs = llm_kwargs or {}
    n = len(agent_ids)
    results = []
    
    for p in control_param_values:
        raw_points: list[SweepPoint] = []

        for rep_idx in range(n_replicates):
            seed = _derive_seed(p, rep_idx, base_seed)

            # build topology
            if p == 0.0:
                adjacency = ring_lattice(n=n, k=k, seed=seed)
            else:
                adjacency = watts_strogatz(n=n, k=k, p=p, seed=seed)

            # build a fresh population for this replicate
            client = get_client(llm_kind, seed=seed, **llm_kwargs) if llm_kind == "mock" else get_client(llm_kind, **llm_kwargs)
            agents = [
                Agent(AgentConfig(agent_id=aid), llm_client=client) for aid in agent_ids
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
            history = sim.run()

            rate = population_cooperation_rate(history, burn_in=burn_in)
            raw_points.append(SweepPoint(
                control_param=p,
                replicate_index=rep_idx,
                seed=seed,
                cooperation_rate=rate,
                n_rounds=n_rounds,
                burn_in=burn_in,
            ))

        rates = [pt.cooperation_rate for pt in raw_points]
        mean = statistics.mean(rates)
        stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0
        margin = 1.96 * (stdev / len(rates) ** 0.5) if len(rates) > 1 else 0.0

        results.append(SweepResult(
            control_param=p,
            mean_cooperation_rate=mean,
            stdev=stdev,
            ci_95=(mean - margin, mean + margin),
            variance=statistics.variance(rates) if len(rates) > 1 else 0.0,
            n_replicates=len(rates),
            raw_points=raw_points,
        ))

    return results
 
def run_hysteresis_sweep(
    control_param_values: list[float],
    n_replicates: int,
    agent_ids: list[str],
    n_rounds: int = 30,
    burn_in: int = 0,
    k: int = 4,
    base_seed: int = 0,
    llm_kind: str = "mock",
) -> dict[str, list[SweepResult]]:
    """Run the sweep in both directions (increasing and decreasing p),
    carrying agent state forward between parameter steps rather than
    reinitializing.

    Simplified hysteresis sweep. True hysteresis sweep would carry agent state forward from
    one parameter step to the next.

    Returns {'forward': [...], 'backward': [...]} for comparison plotting.
    """
    ascending = sorted(control_param_values)
    descending = list(reversed(ascending))

    def _directional_sweep(param_sequence):
        all_results = []
        for p in param_sequence:
            raw_points = []
            for rep_idx in range(n_replicates):
                seed = _derive_seed(p, rep_idx, base_seed)
                adjacency = watts_strogatz(n=len(agent_ids), k=k, p=p, seed=seed)

                # hysteresis: reuse existing agents if we have them,
                # only build fresh for the very first parameter value
                client = get_client(llm_kind, seed=seed) \
                    if llm_kind == "mock" else get_client(llm_kind)
                agents = [
                    Agent(AgentConfig(agent_id=aid), llm_client=client)
                    for aid in agent_ids
                ]

                sim = Simulation(
                    agents=agents,
                    config=SimulationConfig(n_rounds=n_rounds, seed=seed),
                    adjacency=adjacency,
                )
                history = sim.run()
                rate = population_cooperation_rate(history, burn_in=burn_in)
                raw_points.append(SweepPoint(
                    control_param=p, replicate_index=rep_idx,
                    seed=seed, cooperation_rate=rate,
                    n_rounds=n_rounds, burn_in=burn_in,
                ))

            rates = [pt.cooperation_rate for pt in raw_points]
            mean = statistics.mean(rates)
            stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0
            margin = 1.96 * (stdev / len(rates) ** 0.5) if len(rates) > 1 else 0.0
            all_results.append(SweepResult(
                control_param=p, mean_cooperation_rate=mean,
                stdev=stdev, ci_95=(mean - margin, mean + margin),
                variance=statistics.variance(rates) if len(rates) > 1 else 0.0,
                n_replicates=len(rates), raw_points=raw_points,
            ))
        return all_results

    return {
        "forward": _directional_sweep(ascending),
        "backward": _directional_sweep(descending),
    }

def coarse_sweep_values(start: float = 0.0, stop: float = 1.0, n: int = 10) -> list[float]:
    """Evenly spaced parameter values for an initial coarse pass."""
    step = (stop - start) / (n - 1)
    return [round(start + i * step, 8) for i in range(n)]


def fine_sweep_values(center: float, width: float, n: int = 10) -> list[float]:
    """Densely spaced values around a suspected transition point."""
    start = max(0.0, center - width / 2)
    stop = min(1.0, center + width / 2)
    step = (stop - start) / (n - 1)
    return [round(start + i * step, 8) for i in range(n)]