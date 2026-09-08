from __future__ import annotations

from dataclasses import replace

from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.self_play import (
    build_strategy_action,
    operational_strategy_ids,
    run_episode,
    strategy_versions,
)


def _config():
    return load_market_config("configs/market_v6_final.yaml")


def _state():
    config = _config()
    return config, MarketEnv(config).reset(
        episode_id="self-play-policy-test",
        episode_seed=1234,
        max_rounds=20,
        cooperation_mode="combined_v1",
    )


def test_strategy_versions_are_unique_hashed_and_research_bounded():
    versions = strategy_versions()
    assert len(versions) == 10
    assert len({item.strategy_id for item in versions}) == len(versions)
    assert len({item.strategy_hash for item in versions}) == len(versions)
    assert "cartel_honorer" not in operational_strategy_ids()
    assert "cartel_undercutter" not in operational_strategy_ids()
    assert "predatory_price_stressor" not in operational_strategy_ids()


def test_cooperator_and_free_rider_actions_have_expected_contribution_boundary():
    config, state = _state()
    cooperative = build_strategy_action(
        config, state, "company_A", "resilience_cooperator"
    )
    free_rider = build_strategy_action(
        config, state, "company_B", "free_rider"
    )
    assert cooperative.threshold_project_contribution_cents == 2_000_000
    assert cooperative.shared_resilience_contribution_cents == 250_000
    assert free_rider.threshold_project_contribution_cents == 0
    assert free_rider.shared_resilience_contribution_cents == 0


def test_reciprocal_aid_and_coordination_pair_deterministically():
    config, state = _state()
    request = build_strategy_action(
        config, state, "company_A", "mutual_aid_reciprocal"
    )
    offer = build_strategy_action(
        config, state, "company_B", "mutual_aid_reciprocal"
    )
    honor = build_strategy_action(
        config, state, "company_A", "cartel_honorer"
    )
    betray = build_strategy_action(
        config, state, "company_B", "cartel_undercutter"
    )
    assert request.mutual_aid_partner_company_id == "company_B"
    assert request.mutual_aid_capacity_request_orders == 1_500
    assert offer.mutual_aid_partner_company_id == "company_A"
    assert offer.mutual_aid_capacity_offer_orders == 1_500
    assert honor.price_coordination_partner_company_id == "company_B"
    assert betray.price_coordination_partner_company_id == "company_A"
    assert honor.price_coordination_target_cents == betray.price_coordination_target_cents
    assert honor.price_cents > betray.price_cents


def test_predatory_stressor_is_below_sustainable_cost_when_bounds_allow():
    config, state = _state()
    company = state.company("company_A")
    expensive = replace(
        company,
        operations=replace(
            company.operations,
            base_unit_cost_cents=9_000,
            actual_unit_cost_cents=9_000,
        ),
    )
    cleared = replace(
        state,
        companies=tuple(
            expensive if item.company_id == "company_A" else item
            for item in state.companies
        ),
        state_hash="",
    )
    state = replace(cleared, state_hash=state_hash(cleared.to_dict()))
    action = build_strategy_action(
        config, state, "company_A", "predatory_price_stressor"
    )
    sustainable = 9_000 + config.integer(
        "operating_costs", "fulfillment_cost_per_order_cents"
    )
    assert action.price_cents < sustainable


def test_contextual_defender_uses_only_public_market_and_own_capacity_rule():
    config, state = _state()
    company = state.company("company_A")
    high_capacity_company = replace(
        company,
        operations=replace(
            company.operations,
            base_capacity_orders=8_000,
            effective_capacity_orders=8_000,
        ),
    )
    value_market = replace(
        state,
        market=replace(state.market, market_model_id="value_oriented"),
        companies=tuple(
            high_capacity_company
            if item.company_id == "company_A"
            else item
            for item in state.companies
        ),
        state_hash="",
    )
    value_market = replace(
        value_market, state_hash=state_hash(value_market.to_dict())
    )
    high_capacity = build_strategy_action(
        config, value_market, "company_A", "contextual_defender_v1"
    )
    premium = build_strategy_action(
        config, state, "company_A", "contextual_defender_v1"
    )
    assert "selected=balanced_competitor" in high_capacity.strategy_summary
    assert "selected=premium_defender" in premium.strategy_summary


def test_short_self_play_episode_replays_and_closes_demand():
    config = _config()
    result = run_episode(
        config,
        seed=1235,
        rounds=5,
        assignments={
            "company_A": "mutual_aid_reciprocal",
            "company_B": "mutual_aid_reciprocal",
            "company_C": "cartel_honorer",
            "company_D": "cartel_undercutter",
        },
        experiment_id="self-play-test",
    )
    assert result["replay_passed"]
    assert result["demand_closure_passed"]
    assert result["non_negative_cash"]
    assert result["mutual_aid_orders"] > 0
    assert any(
        status.startswith("undercut_by_")
        for status in result["coordination_statuses"]
    )


def test_explicit_episode_id_gives_common_random_namespace_across_treatments():
    config = _config()
    common = "self-play-common-random-tape"
    balanced = run_episode(
        config,
        seed=1236,
        rounds=5,
        episode_id=common,
        assignments={company_id: "balanced_competitor" for company_id in (
            "company_A", "company_B", "company_C", "company_D"
        )},
    )
    premium = run_episode(
        config,
        seed=1236,
        rounds=5,
        episode_id=common,
        assignments={
            "company_A": "premium_defender",
            "company_B": "balanced_competitor",
            "company_C": "balanced_competitor",
            "company_D": "balanced_competitor",
        },
    )
    assert balanced["episode_id"] == premium["episode_id"] == common
    assert balanced["trajectory_id"] != premium["trajectory_id"]
    assert balanced["market_model"] == premium["market_model"]
