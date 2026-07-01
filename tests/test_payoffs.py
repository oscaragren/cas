from sim.payoffs import resolve_payoffs
from agents.schemas import ActionType

def test_mutual_cooperation_beats_mutual_defection_total():
    coop_a, coop_b = resolve_payoffs(ActionType.COOPERATE, ActionType.COOPERATE)
    defe_a, defe_b = resolve_payoffs(ActionType.DEFECT, ActionType.DEFECT)
    assert coop_a + coop_b > defe_a + defe_b

def test_defector_profits_over_cooperator():
    defector, sucker = resolve_payoffs(ActionType.DEFECT, ActionType.COOPERATE)
    assert defector > sucker

def test_cooperating_against_defector_is_worse_than_mutual_defection():
    # this is what makes cooperation risky: you're better off defecting
    # against a defector than cooperating against one
    _, sucker_payoff = resolve_payoffs(ActionType.DEFECT, ActionType.COOPERATE)
    _, mutual_defect_payoff = resolve_payoffs(ActionType.DEFECT, ActionType.DEFECT)
    assert sucker_payoff < mutual_defect_payoff

def test_nash_equilibrium_unilateral_switch_from_defect_to_cooperate_hurts():
    # if your partner is defecting, you should defect too
    # confirms this is a genuine prisoner's dilemma, not some other game
    against_defector_if_i_cooperate, _ = resolve_payoffs(ActionType.COOPERATE, ActionType.DEFECT)
    against_defector_if_i_defect, _ = resolve_payoffs(ActionType.DEFECT, ActionType.DEFECT)
    assert against_defector_if_i_cooperate < against_defector_if_i_defect

def test_offer_resource_costs_sender_and_benefits_receiver():
    plain_a, plain_b = resolve_payoffs(ActionType.COOPERATE, ActionType.COOPERATE)
    offer_a, offer_b = resolve_payoffs(ActionType.OFFER_RESOURCE, ActionType.COOPERATE)
    assert offer_a < plain_a   # sender pays a cost
    assert offer_b > plain_b   # receiver gets a bonus

def test_mutual_offer_resource_both_benefit_relative_to_plain_cooperate():
    plain, _ = resolve_payoffs(ActionType.COOPERATE, ActionType.COOPERATE)
    mutual_offer_a, mutual_offer_b = resolve_payoffs(
        ActionType.OFFER_RESOURCE, ActionType.OFFER_RESOURCE
    )
    assert mutual_offer_a > plain
    assert mutual_offer_b > plain

def test_scarcity_shock_reduces_both_payoffs():
    normal_a, normal_b = resolve_payoffs(ActionType.COOPERATE, ActionType.COOPERATE)
    shock_a, shock_b = resolve_payoffs(
        ActionType.COOPERATE, ActionType.COOPERATE, scarcity_shock=True
    )
    assert shock_a < normal_a
    assert shock_b < normal_b

def test_scarcity_shock_preserves_incentive_ratios():
    # multiplying both payoffs by the same constant should keep the
    # ordering intact -- defecting should still be individually rational
    # under scarcity, just with lower absolute values
    normal_defector, _ = resolve_payoffs(ActionType.DEFECT, ActionType.COOPERATE)
    normal_sucker, _ = resolve_payoffs(ActionType.COOPERATE, ActionType.DEFECT)
    shock_defector, _ = resolve_payoffs(
        ActionType.DEFECT, ActionType.COOPERATE, scarcity_shock=True
    )
    shock_sucker, _ = resolve_payoffs(
        ActionType.COOPERATE, ActionType.DEFECT, scarcity_shock=True
    )
    assert shock_defector > shock_sucker  # ordering preserved

def test_cheap_talk_actions_treated_as_defect_for_payoff():
    for cheap_talk in (ActionType.SIGNAL_TRUST, ActionType.SIGNAL_THREAT, ActionType.WITHHOLD):
        cheap_a, cheap_b = resolve_payoffs(cheap_talk, ActionType.COOPERATE)
        defect_a, defect_b = resolve_payoffs(ActionType.DEFECT, ActionType.COOPERATE)
        assert cheap_a == defect_a, f"{cheap_talk} should yield same payoff as DEFECT"
        assert cheap_b == defect_b, f"{cheap_talk} partner should get same payoff as vs DEFECT"


def test_payoffs_are_symmetric_for_symmetric_actions():
    # if both agents do the same thing, both should get the same payoff
    for action in (ActionType.COOPERATE, ActionType.DEFECT):
        a, b = resolve_payoffs(action, action)
        assert a == b