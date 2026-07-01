from __future__ import annotations

from agents.schemas import ActionType

# Prisoner's-dilemma-shaped base matrix.
# (action_a, action_b) -> (payoff_a, payoff_b)

_BASE_MATRIX: dict[tuple[ActionType, ActionType], tuple[float, float]] = {
    (ActionType.COOPERATE, ActionType.COOPERATE): (3.0, 3.0),
    (ActionType.COOPERATE, ActionType.DEFECT):    (0.0, 5.0),
    (ActionType.DEFECT,    ActionType.COOPERATE): (5.0, 0.0),
    (ActionType.DEFECT,    ActionType.DEFECT):    (1.0, 1.0),
}

_OFFER_COST    = 0.5   # sender pays this on top of the base cooperate payoff
_OFFER_BONUS   = 1.0   # receiver gets this on top of the base cooperate payoff

def _normalize(action: ActionType) -> ActionType:
    """
    Map the full six-action vocabulary down to the two-action
    payoff matrix. OFFER_RESOURCE is cooperative in outcome;
    cheap-talk and withhold have no resource consequence.
    """
    if action == ActionType.OFFER_RESOURCE:
        return ActionType.COOPERATE
    if action in (ActionType.SIGNAL_TRUST, ActionType.SIGNAL_THREAT, ActionType.WITHHOLD):
        return ActionType.DEFECT
    return action # COOPERATE and DEFECT pass through unchanged

def resolve_payoffs(
        action_a: ActionType,
        action_b: ActionType,
        scarcity_shock: bool = False,
        scarcity_multiplier: float = 0.5,    
) -> tuple[float, float]:
    """
    Return (payoff_a, payoff_b) for one round between two agents.
    """
    norm_a = _normalize(action_a)
    norm_b = _normalize(action_b)

    payoff_a, payoff_b = _BASE_MATRIX[(norm_a, norm_b)]

    # OFFER_RESOURCE is a costly signal on top of the base cooperate payoff:
    # the sender pays extra, the receiver gets extra.
    if action_a == ActionType.OFFER_RESOURCE:
        payoff_a -= _OFFER_COST
        payoff_b += _OFFER_BONUS
    if action_b == ActionType.OFFER_RESOURCE:
        payoff_b -= _OFFER_COST
        payoff_a += _OFFER_BONUS

    if scarcity_shock:
        payoff_a *= scarcity_multiplier
        payoff_b *= scarcity_multiplier

    return payoff_a, payoff_b
