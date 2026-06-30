from agents.llm_client import MockLLMClient, get_client, _safe_json_extract

def test_mock_client_deterministic_with_same_seed():
    client_a = MockLLMClient(seed=42)
    client_b = MockLLMClient(seed=42)
    response_a = client_a.complete("system prompt", "user prompt")
    response_b = client_b.complete("system prompt", "user prompt")
    assert response_a.parsed == response_b.parsed

def test_mock_client_repeated_calls_advance_independently_of_other_instances():
    # two independently-seeded clients shouldn't affect each other's sequence
    client_a = MockLLMClient(seed=1)
    client_b = MockLLMClient(seed=2)
    response_a1 = client_a.complete("", "neutral prompt")
    response_b1 = client_b.complete("", "neutral prompt")
    # re-seed a fresh client_a the same way and confirm its first call matches
    client_a_again = MockLLMClient(seed=1)
    response_a1_again = client_a_again.complete("", "neutral prompt")
    assert response_a1.parsed == response_a1_again.parsed

def test_mock_client_high_trust_cue_increases_cooperate_rate():
    high_trust_client = MockLLMClient(seed=7)
    neutral_client = MockLLMClient(seed=7)

    high_trust_coops = sum(
        1 for _ in range(300)
        if high_trust_client.complete("", "Your current internal state: trust=high (0.80)").parsed["action"] == "cooperate"
    )
    neutral_coops = sum(
        1 for _ in range(300)
        if neutral_client.complete("", "Your current internal state: trust=moderate (0.50)").parsed["action"] == "cooperate"
    )

    assert high_trust_coops > neutral_coops

def test_mock_client_resentment_cue_decreases_cooperate_rate():
    resentful_client = MockLLMClient(seed=3)
    neutral_client = MockLLMClient(seed=3)

    resentful_coops = sum(
        1 for _ in range(300)
        if resentful_client.complete("", "your grudge level against them is resentment=high (0.80)").parsed["action"] == "cooperate"
    )
    neutral_coops = sum(
        1 for _ in range(300)
        if neutral_client.complete("", "your grudge level against them is moderate").parsed["action"] == "cooperate"
    )
    assert resentful_coops < neutral_coops

def test_mock_client_always_returns_parseable_response():
    client = MockLLMClient(seed=0)
    response = client.complete("any system prompt", "any user prompt")
    assert response.parsed is not None
    assert response.parsed["action"] in ("cooperate", "defect")

def test_get_client_factory_returns_mock_by_default():
    client = get_client()
    assert isinstance(client, MockLLMClient)

def test_get_client_factory_rejects_unknown_kind():
    try:
        get_client(kind="not_a_real_backend")
        assert False, "expected ValueError"
    except ValueError:
        pass