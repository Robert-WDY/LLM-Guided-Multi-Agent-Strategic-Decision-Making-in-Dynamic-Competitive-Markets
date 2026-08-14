from game_theory_agent.gameplay import build_company_analysis, build_rule_action
from game_theory_agent.market import MarketEnv


def test_company_analysis_is_decision_focused(initial_state):
    analysis = build_company_analysis(initial_state, "company_A")
    assert 0 <= analysis["health_score"] <= 100
    assert {factor["key"] for factor in analysis["factors"]} == {
        "liquidity",
        "market_position",
        "capacity",
        "brand",
        "risk_readiness",
    }
    assert analysis["decision_context"]["rounds_remaining"] == 10


def test_rule_opponent_action_is_valid(config, env, initial_state):
    action = build_rule_action(config, initial_state, "company_B")
    result = env.validator.validate(action, state=initial_state, company_id="company_B")
    assert result.valid, result.errors
    assert action.strategy_summary.startswith("rule-opponent")


def test_seeded_rule_opponents_are_repeatable_and_varied(config):
    profiles = set()
    first_round_prices = set()
    for seed in range(20):
        env = MarketEnv(config)
        state = env.reset(
            ("company_A", "company_B", "company_C", "company_D"),
            episode_id=f"rule-style-{seed}",
            episode_seed=seed,
        )
        for company_id in state.company_ids[1:]:
            first = build_rule_action(config, state, company_id)
            second = build_rule_action(config, state, company_id)
            assert first == second
            profiles.add(first.strategy_summary.split(":")[1])
            first_round_prices.add(first.price_cents)

    assert profiles == {"value", "premium", "growth", "cautious"}
    assert len(first_round_prices) >= 8


def test_rule_prices_stay_anchored_instead_of_ratchet_up(config):
    for model in ("balanced", "value_oriented", "quality_oriented", "service_oriented"):
        env = MarketEnv(config)
        state = env.reset(
            ("company_A", "company_B", "company_C", "company_D"),
            episode_id=f"anchor-{model}",
            episode_seed=73,
            market_model=model,
        )
        prices = []
        while not state.terminal:
            actions = {
                company_id: build_rule_action(config, state, company_id)
                for company_id in state.company_ids
            }
            prices.extend(action.price_cents for action in actions.values())
            state = env.step(
                f"{state.episode_id}:{state.round}:{state.state_version}", actions
            ).state_after
        assert max(prices) <= state.market.price_anchor_cents + state.market.price_band_cents + 700
        assert min(prices) >= state.market.price_anchor_cents - state.market.price_band_cents
