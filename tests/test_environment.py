from dataclasses import FrozenInstanceError, replace

import pytest

from game_theory_agent.market import MarketEnv
from game_theory_agent.market.exceptions import (
    ActionValidationError,
    EpisodeCompleteError,
    IdempotencyConflictError,
)
from game_theory_agent.market.protocols import state_hash


def _rehash(state):
    state = replace(state, state_hash="")
    return replace(state, state_hash=state_hash(state.to_dict()))


def _paired_step(config, state, joint_action):
    env = MarketEnv(config)
    env.load_state(state)
    return env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", joint_action
    ).state_after


def test_reset_uses_v4_round_semantics_and_fixed_point(initial_state):
    assert initial_state.round == 1
    assert initial_state.state_version == 0
    assert initial_state.rounds_remaining == 10
    assert (
        sum(company.commercial.market_share_ppm for company in initial_state.companies)
        == 1_000_000
    )
    assert initial_state.state_hash.startswith("sha256:")
    with pytest.raises(FrozenInstanceError):
        initial_state.round = 2


def test_market_models_change_public_preferences_and_are_seeded(config):
    value_env = MarketEnv(config)
    value = value_env.reset(
        ("company_A", "company_B"),
        episode_id="value-market",
        episode_seed=9,
        market_model="value_oriented",
    )
    quality_env = MarketEnv(config)
    quality = quality_env.reset(
        ("company_A", "company_B"),
        episode_id="quality-market",
        episode_seed=9,
        market_model="quality_oriented",
    )
    assert value.market.market_model_id == "value_oriented"
    assert quality.market.market_model_id == "quality_oriented"
    assert dict(value.consumer_segments)["price_sensitive"] > dict(quality.consumer_segments)["price_sensitive"]
    assert value.market.price_anchor_cents < quality.market.price_anchor_cents

    random_one = MarketEnv(config).reset(
        ("company_A", "company_B"), episode_id="random-one", episode_seed=91
    )
    random_two = MarketEnv(config).reset(
        ("company_A", "company_B"), episode_id="random-two", episode_seed=91
    )
    assert random_one.market.market_model_id == random_two.market.market_model_id
    assert random_one.market.demand_bias_ppm == random_two.market.demand_bias_ppm
    assert random_one.market.price_anchor_cents == random_two.market.price_anchor_cents
    assert 900_000 <= random_one.market.demand_bias_ppm <= 1_100_000


def test_quality_rewards_brand_while_service_rewards_current_service(
    config, make_actions
):
    """Paired actions must produce opposite preferences in the two models."""

    company_ids = ("company_A", "company_B", "company_C", "company_D")
    results = {}
    for model in ("quality_oriented", "service_oriented"):
        model_results = {}
        for focus, override in {
            "brand": {
                "advertising_budget_cents": 5_000_000,
                "service_budget_cents": 0,
            },
            "service": {
                "advertising_budget_cents": 0,
                "service_budget_cents": 5_000_000,
            },
        }.items():
            env = MarketEnv(config)
            state = env.reset(
                company_ids,
                episode_id=f"paired-{model}-{focus}",
                episode_seed=424_242,
                market_model=model,
            )
            actions = make_actions(
                state,
                nonce=focus,
                overrides={"company_A": override},
            )
            settled = env.step(
                f"{state.episode_id}:1:0", actions
            ).state_after
            model_results[focus] = settled.company(
                "company_A"
            ).commercial.potential_demand_orders
        results[model] = model_results

    assert (
        results["quality_oriented"]["brand"]
        > results["quality_oriented"]["service"]
    )
    assert (
        results["service_oriented"]["service"]
        > results["service_oriented"]["brand"]
    )


def test_configurable_horizon_terminates_at_selected_round(config, make_actions):
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B"),
        episode_id="five-round-episode",
        episode_seed=7,
        market_model="balanced",
        max_rounds=5,
    )
    assert state.max_rounds == 5
    assert state.rounds_remaining == 5

    while not state.terminal:
        state = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}",
            make_actions(state, nonce=str(state.round)),
        ).state_after

    assert state.state_version == 5
    assert state.round == 6
    assert state.rounds_remaining == 0


def test_same_actions_across_rounds_produce_dynamic_results(
    env, initial_state, make_actions
):
    state = initial_state
    demands = []
    awareness = []
    for index in range(4):
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}",
            make_actions(state, nonce=str(index)),
        )
        state = result.state_after
        demands.append(state.market.realized_demand_orders)
        awareness.append(state.company("company_A").brand.brand_awareness_ppm)

    assert len(set(demands)) > 1
    assert len(set(awareness)) > 1
    assert state.company("company_A").history.recent_profit_cents


def test_lower_relative_price_wins_more_orders_with_paired_seed(
    config, initial_state, make_actions
):
    baseline = make_actions(initial_state, nonce="paired")
    low_price = dict(baseline)
    low_price["company_A"] = replace(low_price["company_A"], price_cents=8000)
    high_price = dict(baseline)
    high_price["company_A"] = replace(high_price["company_A"], price_cents=12500)

    low_state = _paired_step(config, initial_state, low_price)
    high_state = _paired_step(config, initial_state, high_price)
    assert (
        low_state.company("company_A").commercial.potential_demand_orders
        > high_state.company("company_A").commercial.potential_demand_orders
    )


def test_advertising_has_diminishing_input_and_builds_awareness(
    config, initial_state, make_actions
):
    zero = make_actions(
        initial_state,
        nonce="paired",
        overrides={"company_A": {"advertising_budget_cents": 0}},
    )
    high = make_actions(
        initial_state,
        nonce="paired",
        overrides={"company_A": {"advertising_budget_cents": 5_000_000}},
    )
    zero_state = _paired_step(config, initial_state, zero)
    high_state = _paired_step(config, initial_state, high)
    assert (
        high_state.company("company_A").brand.brand_awareness_ppm
        > zero_state.company("company_A").brand.brand_awareness_ppm
    )
    assert (
        high_state.company("company_A").financial.cash_balance_cents
        < zero_state.company("company_A").financial.cash_balance_cents
    )


def test_capacity_investment_only_changes_next_round_base_capacity(
    config, initial_state, make_actions
):
    no_invest = make_actions(initial_state, nonce="paired")
    invest = make_actions(
        initial_state,
        nonce="paired",
        overrides={"company_A": {"capacity_investment_cents": 3_000_000}},
    )
    no_state = _paired_step(config, initial_state, no_invest)
    invest_state = _paired_step(config, initial_state, invest)
    assert (
        no_state.company("company_A").operations.effective_capacity_orders
        == invest_state.company("company_A").operations.effective_capacity_orders
    )
    assert invest_state.company("company_A").operations.base_capacity_orders == (
        no_state.company("company_A").operations.base_capacity_orders + 300
    )


def test_outside_option_and_demand_closure(env, initial_state, make_actions):
    actions = make_actions(
        initial_state,
        overrides={
            company_id: {
                "price_cents": 13000,
                "advertising_budget_cents": 0,
                "service_budget_cents": 0,
            }
            for company_id in initial_state.company_ids
        },
    )
    state = env.step("episode-0001:1:0", actions).state_after
    assert state.market.no_purchase_orders > 0
    assert (
        state.market.no_purchase_orders
        + state.market.lost_after_stockout_orders
        + sum(company.commercial.sales_orders for company in state.companies)
        == state.market.realized_demand_orders
    )


def test_stockout_triggers_one_pass_redistribution(config, initial_state, make_actions):
    company_a = initial_state.company("company_A")
    constrained_a = replace(
        company_a,
        operations=replace(company_a.operations, base_capacity_orders=100),
    )
    forced = _rehash(
        replace(
            initial_state,
            companies=(constrained_a,) + initial_state.companies[1:],
        )
    )
    actions = make_actions(
        forced,
        overrides={"company_A": {"price_cents": 7500}},
    )
    state = _paired_step(config, forced, actions)
    assert state.company("company_A").commercial.attempted_unfulfilled_orders > 0
    assert (
        sum(
            company.commercial.orders_received_from_redistribution
            for company in state.companies[1:]
        )
        > 0
    )


def test_step_is_idempotent_and_conflicts_on_changed_payload(
    env, initial_state, make_actions
):
    actions = make_actions(initial_state)
    first = env.step("episode-0001:1:0", actions)
    second = env.step("episode-0001:1:0", actions)
    assert second is first

    changed = dict(actions)
    changed["company_A"] = replace(changed["company_A"], price_cents=9000)
    with pytest.raises(IdempotencyConflictError):
        env.step("episode-0001:1:0", changed)


def test_horizon_and_last_round_investment_rule(env, initial_state, make_actions):
    state = initial_state
    for index in range(9):
        state = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}",
            make_actions(state, nonce=str(index)),
        ).state_after
    invalid = make_actions(
        state,
        nonce="last",
        overrides={"company_A": {"capacity_investment_cents": 10000}},
    )
    with pytest.raises(ActionValidationError, match="last round"):
        env.step(f"{state.episode_id}:{state.round}:{state.state_version}", invalid)

    valid = make_actions(state, nonce="terminal")
    terminal = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", valid
    ).state_after
    assert terminal.terminal
    assert terminal.round == 11
    assert len(terminal.terminal_enterprise_values_cents) == 4
    with pytest.raises(EpisodeCompleteError):
        env.step("episode-0001:11:10", valid)
