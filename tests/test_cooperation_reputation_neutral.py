from __future__ import annotations

from game_theory_agent.experiments.cooperation_reputation_neutral_real import (
    CONDITIONS,
    _credibility_record,
    build_reputation_observation,
    cooperation_capability_overlay,
)
from game_theory_agent.experiments.persona_pilot import PROJECT_ROOT, _observation
from game_theory_agent.market import MarketEnv, load_market_config


def test_reputation_treatments_only_change_observation_not_true_market_state() -> None:
    base_config = load_market_config(PROJECT_ROOT / "configs" / "market_v4.yaml")
    config = cooperation_capability_overlay(base_config)
    assert base_config.mapping("persona_utilities", "capabilities")["cooperation"] is False
    assert config.mapping("persona_utilities", "capabilities")["cooperation"] is True
    assert config.config_sha256 != base_config.config_sha256
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B", "company_C", "company_D"),
        episode_id="reputation-test",
        episode_seed=920_201,
        max_rounds=10,
        cooperation_mode="shared_resilience_v1",
    )
    base = _observation(config, state, "company_B")
    observations = {condition: build_reputation_observation(base, condition) for condition in CONDITIONS}
    assert {item["state_hash"] for item in observations.values()} == {state.state_hash}
    assert len({item["observation_hash"] for item in observations.values()}) == 3
    messages = [item["communication_view"]["visible_messages"] for item in observations.values()]
    assert messages[0] == messages[1] == messages[2]
    assert [_credibility_record(condition)["credibility_ppm"] for condition in CONDITIONS] == [500_000, 100_000, 700_000]
