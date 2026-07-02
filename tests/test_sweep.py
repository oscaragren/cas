# tests/test_sweep.py

from sim.sweep import (
    run_sweep,
    run_hysteresis_sweep,
    coarse_sweep_values,
    fine_sweep_values,
    _derive_seed,
)

AGENT_IDS = [f"agent_{i}" for i in range(6)]


# ── seed derivation ──────────────────────────────────────────────────────────

def test_seed_deterministic_same_inputs():
    assert _derive_seed(0.3, 5) == _derive_seed(0.3, 5)


def test_seed_differs_for_different_params():
    assert _derive_seed(0.1, 0) != _derive_seed(0.2, 0)


def test_seed_differs_for_different_replicates():
    assert _derive_seed(0.3, 0) != _derive_seed(0.3, 1)


def test_seed_no_collision_between_param_and_replicate():
    # the naive int(param*1000) + replicate approach collides here;
    # confirm the hash-based approach doesn't
    assert _derive_seed(0.1, 10) != _derive_seed(0.0, 110)


def test_seed_is_non_negative():
    for p in (0.0, 0.1, 0.5, 1.0):
        for rep in range(5):
            assert _derive_seed(p, rep) >= 0


# ── coarse and fine sweep values ─────────────────────────────────────────────

def test_coarse_sweep_produces_correct_count():
    values = coarse_sweep_values(0.0, 1.0, n=5)
    assert len(values) == 5


def test_coarse_sweep_endpoints():
    values = coarse_sweep_values(0.0, 1.0, n=5)
    assert values[0] == 0.0
    assert values[-1] == 1.0


def test_fine_sweep_centered_on_target():
    values = fine_sweep_values(center=0.3, width=0.2, n=5)
    mid = values[len(values) // 2]
    assert abs(mid - 0.3) < 0.05


def test_fine_sweep_stays_in_zero_one():
    values = fine_sweep_values(center=0.05, width=0.2, n=5)
    assert all(0.0 <= v <= 1.0 for v in values)


# ── run_sweep ────────────────────────────────────────────────────────────────

def test_sweep_returns_one_result_per_param_value():
    params = coarse_sweep_values(0.0, 1.0, n=3)
    results = run_sweep(
        control_param_values=params,
        n_replicates=3,
        agent_ids=AGENT_IDS,
        n_rounds=5,
        llm_kind="mock",
    )
    assert len(results) == 3


def test_sweep_result_has_correct_n_replicates():
    results = run_sweep(
        control_param_values=[0.3],
        n_replicates=4,
        agent_ids=AGENT_IDS,
        n_rounds=5,
        llm_kind="mock",
    )
    assert results[0].n_replicates == 4
    assert len(results[0].raw_points) == 4


def test_sweep_cooperation_rates_in_valid_range():
    results = run_sweep(
        control_param_values=[0.0, 0.5, 1.0],
        n_replicates=3,
        agent_ids=AGENT_IDS,
        n_rounds=5,
        llm_kind="mock",
    )
    for result in results:
        assert 0.0 <= result.mean_cooperation_rate <= 1.0
        for pt in result.raw_points:
            assert 0.0 <= pt.cooperation_rate <= 1.0


def test_sweep_ci_straddles_mean():
    results = run_sweep(
        control_param_values=[0.3],
        n_replicates=5,
        agent_ids=AGENT_IDS,
        n_rounds=5,
        llm_kind="mock",
    )
    r = results[0]
    assert r.ci_95[0] <= r.mean_cooperation_rate <= r.ci_95[1]


def test_sweep_is_reproducible_with_same_base_seed():
    kwargs = dict(
        control_param_values=[0.0, 0.5],
        n_replicates=3,
        agent_ids=AGENT_IDS,
        n_rounds=5,
        llm_kind="mock",
        base_seed=99,
    )
    first = [(r.control_param, r.mean_cooperation_rate) for r in run_sweep(**kwargs)]
    second = [(r.control_param, r.mean_cooperation_rate) for r in run_sweep(**kwargs)]
    assert first == second


def test_sweep_raw_points_carry_correct_param_value():
    params = [0.1, 0.5, 0.9]
    results = run_sweep(
        control_param_values=params,
        n_replicates=2,
        agent_ids=AGENT_IDS,
        n_rounds=5,
        llm_kind="mock",
    )
    for result in results:
        for pt in result.raw_points:
            assert pt.control_param == result.control_param


# ── hysteresis sweep ─────────────────────────────────────────────────────────

def test_hysteresis_sweep_returns_forward_and_backward():
    result = run_hysteresis_sweep(
        control_param_values=[0.0, 0.5, 1.0],
        n_replicates=2,
        agent_ids=AGENT_IDS,
        n_rounds=5,
        llm_kind="mock",
    )
    assert "forward" in result
    assert "backward" in result


def test_hysteresis_forward_and_backward_cover_same_params():
    result = run_hysteresis_sweep(
        control_param_values=[0.0, 0.5, 1.0],
        n_replicates=2,
        agent_ids=AGENT_IDS,
        n_rounds=5,
        llm_kind="mock",
    )
    forward_params = {r.control_param for r in result["forward"]}
    backward_params = {r.control_param for r in result["backward"]}
    assert forward_params == backward_params