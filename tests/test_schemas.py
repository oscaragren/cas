from agents.schemas import AffectState, ReputationRecord

def test_affect_decay_pulls_toward_baseline():
    affect = AffectState(trust=0.9, fear=0.8, resentment=0.7)
    affect.decay()
    assert affect.trust < 0.9
    assert affect.fear < 0.8
    assert affect.resentment < 0.7


def test_affect_betrayal_increases_fear_and_resentment():
    affect = AffectState()
    affect.apply_event(betrayed=True, cooperated_with=False, scarcity=False)
    assert affect.trust < 0.5
    assert affect.fear > 0.0
    assert affect.resentment > 0.0

def test_reputation_uninformative_prior():
    rec = ReputationRecord(target_id="agent_b")
    assert rec.cooperate_rate == 0.5

def test_reputation_rate_updates_correctly():
    rec = ReputationRecord(target_id="agent_b", cooperations_observed=3, defections_observed=1)
    assert rec.cooperate_rate == 0.75

def test_cooperation_with_partner_increases_trust_and_lowers_resentment():
    affect = AffectState(resentment=0.5)
    affect.apply_event(betrayed=False, cooperated_with=True, scarcity=False)
    assert affect.trust > 0.5
    assert affect.resentment < 0.5



    