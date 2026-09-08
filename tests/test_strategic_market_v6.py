from dataclasses import replace
from pathlib import Path

from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.market import (
    CompanyOperatingStatus,
    MarketEnv,
    MarketState,
    PriceCoordinationStatus,
    ThresholdProjectStatus,
    load_market_config,
)
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition, verify_replay


ROOT = Path(__file__).resolve().parents[1]
V5_HASH = "sha256:bc5ba80831795e29c483bb7a4315da74759e322ebade4a4768c9b82edcbbba61"


def _rehash(state: MarketState) -> MarketState:
    cleared = replace(state, state_hash="")
    return replace(cleared, state_hash=state_hash(cleared.to_dict()))


def _rule_actions(config, state):
    return {
        company_id: build_rule_action(config, state, company_id)
        for company_id in state.company_ids
    }


def test_v6_config_inherits_without_mutating_v5():
    v5 = load_market_config(ROOT / "configs" / "market_v5_cooperation.yaml")
    v6 = load_market_config(ROOT / "configs" / "market_v6_strategic.yaml")

    assert v5.config_sha256 == V5_HASH
    assert v6.config_id == "market-v6-strategic"
    assert v6.environment_version == "market-env-v6.0.0"
    assert v6.mapping("strategic_market")["enabled"] is True
    assert "extends" not in v6.to_dict()


def test_final_combined_market_manifest_is_versioned_and_replayable():
    config = load_market_config(ROOT / "configs" / "market_v6_final.yaml")
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B", "company_C", "company_D"),
        episode_id="v6-combined-manifest",
        episode_seed=7000,
        max_rounds=5,
        cooperation_mode="combined_v1",
    )

    manifest = EpisodeManifest.create(
        env,
        state,
        experiment_id="v6-combined-manifest",
        information_mode="public",
        communication_mode="public_private",
        cooperation_mode="combined_v1",
        belief_mode="public_action_v1",
        opponent_model_mode="public_strategy_v1",
        utility_inference_mode="strategy_utility_v1",
        advisor_mode="strategic_market_v9",
    )

    assert manifest.config_id == "market-v6-final"
    assert manifest.cooperation_mode == "combined_v1"
    assert (
        manifest.cooperation_protocol_version
        == "final-strategic-cooperation-v1.0.0"
    )
    assert manifest.advisor_mode == "strategic_market_v9"
    assert state.strategic_market is not None
    assert state.strategic_market.threshold_project is not None
    assert state.strategic_market.mutual_aid_enabled
    assert state.strategic_market.price_coordination_enabled


def test_financial_distress_exit_monopoly_and_absorbing_shutdown():
    config = load_market_config(ROOT / "configs" / "market_v6_strategic.yaml")
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B"),
        episode_id="v6-bankruptcy",
        episode_seed=7001,
        max_rounds=5,
    )
    weak = state.company("company_A")
    weak = replace(
        weak,
        financial=replace(
            weak.financial,
            cash_balance_cents=200_000,
            capacity_book_value_cents=1_000_000,
        ),
        operations=replace(
            weak.operations,
            base_capacity_orders=0,
            effective_capacity_orders=0,
            financial_capacity_orders=0,
        ),
    )
    state = _rehash(
        replace(
            state,
            companies=(weak, state.company("company_B")),
        )
    )
    env.load_state(state)
    manifest = EpisodeManifest.create(env, state, experiment_id="v6-lifecycle")

    actions_1 = _rule_actions(config, state)
    result_1 = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions_1
    )
    after_exit = result_1.state_after
    lifecycle = after_exit.strategic_market.lifecycle("company_A")
    assert lifecycle.status is CompanyOperatingStatus.EXITED
    assert lifecycle.exit_reason == "cash_exhausted"
    assert after_exit.strategic_market.active_company_count == 1
    assert after_exit.strategic_market.concentration_regime.value == "monopoly"
    assert after_exit.company("company_A").financial.capacity_book_value_cents == 0

    actions_2 = _rule_actions(config, after_exit)
    assert actions_2["company_A"].fixed_spend_cents == 0
    result_2 = env.step(
        (
            f"{after_exit.episode_id}:{after_exit.round}:"
            f"{after_exit.state_version}"
        ),
        actions_2,
    )
    final = result_2.state_after
    assert final.company("company_A").commercial.sales_orders == 0
    assert final.company("company_A").commercial.market_share_ppm == 0
    assert final.strategic_market.lifecycle("company_A") == lifecycle

    transitions = (
        MarketTransition.create(state, actions_1, result_1),
        MarketTransition.create(after_exit, actions_2, result_2),
    )
    assert verify_replay(MarketEnv(config), manifest, transitions)[-1] == final


def test_below_sustainable_cost_is_audited_as_predatory_pressure():
    config = load_market_config(ROOT / "configs" / "market_v6_strategic.yaml")
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B"),
        episode_id="v6-price-war",
        episode_seed=7002,
        max_rounds=5,
    )
    costly = state.company("company_A")
    costly = replace(
        costly,
        operations=replace(costly.operations, base_unit_cost_cents=8_000),
    )
    state = _rehash(
        replace(state, companies=(costly, state.company("company_B")))
    )
    env.load_state(state)
    actions = _rule_actions(config, state)
    actions["company_A"] = replace(
        actions["company_A"],
        price_cents=7_500,
        advertising_budget_cents=0,
        service_budget_cents=0,
        strategy_summary="deliberate below-cost pressure test",
    )

    after = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    ).state_after

    assert after.strategic_market.predatory_pricing_company_ids == ("company_A",)
    assert after.strategic_market.price_war_intensity_ppm > 0
    assert after.strategic_market.regulatory_pressure_ppm > 0
    assert after.strategic_market.consumer_surplus_proxy_cents > 0


def test_v6_state_round_trip_preserves_strategic_contract():
    config = load_market_config(ROOT / "configs" / "market_v6_strategic.yaml")
    state = MarketEnv(config).reset(episode_id="v6-round-trip", episode_seed=7003)

    assert MarketState.from_dict(state.to_dict()) == state


def test_threshold_project_succeeds_only_at_provision_point():
    config = load_market_config(ROOT / "configs" / "market_v6_strategic.yaml")
    env = MarketEnv(config)
    state = env.reset(
        episode_id="v6-threshold-success",
        episode_seed=7201,
        max_rounds=5,
        cooperation_modes=("threshold_project_v1",),
    )
    actions = _rule_actions(config, state)
    actions = {
        company_id: replace(
            action,
            threshold_project_contribution_cents=2_000_000,
        )
        for company_id, action in actions.items()
    }

    after = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    ).state_after
    project = after.strategic_market.threshold_project

    assert project.status is ThresholdProjectStatus.SUCCEEDED
    assert project.accumulated_total_contribution_cents == 8_000_000
    assert project.public_protection_bonus_ppm == 180_000
    assert project.supply_cost_reduction_ppm == 60_000
    assert after.market.actual_supply_cost_index_ppm < (
        after.market.base_supply_cost_index_ppm
    )

    invalid = replace(
        _rule_actions(config, after)["company_A"],
        threshold_project_contribution_cents=1,
    )
    validation = env.validate_action(invalid, "company_A")
    assert validation.valid is False
    assert "threshold project no longer accepts contributions" in validation.errors


def test_threshold_project_failure_refunds_only_configured_fraction():
    config = load_market_config(ROOT / "configs" / "market_v6_strategic.yaml")
    env = MarketEnv(config)
    state = env.reset(
        episode_id="v6-threshold-failure",
        episode_seed=7202,
        max_rounds=5,
        cooperation_modes=("threshold_project_v1",),
    )
    for index in range(4):
        actions = _rule_actions(config, state)
        if index == 0:
            actions["company_A"] = replace(
                actions["company_A"],
                threshold_project_contribution_cents=1_000_000,
            )
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", actions
        )
        state = result.state_after

    project = state.strategic_market.threshold_project
    assert project.status is ThresholdProjectStatus.FAILED
    assert dict(project.last_refund_by_company_cents)["company_A"] == 200_000
    assert project.public_protection_bonus_ppm == 0
    assert project.supply_cost_reduction_ppm == 0


def test_resolver_and_market_keep_identical_post_project_action_schema():
    config = load_market_config(ROOT / "configs" / "market_v6_final.yaml")
    env = MarketEnv(config)
    state = env.reset(
        episode_id="v6-post-project-action-schema",
        episode_seed=7203,
        max_rounds=10,
        cooperation_mode="combined_v1",
    )
    for _ in range(4):
        actions = _rule_actions(config, state)
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", actions
        )
        state = result.state_after
    assert state.strategic_market.threshold_project.status is ThresholdProjectStatus.FAILED

    resolved = resolve_action_request(
        config,
        state,
        "company_A",
        {
            "price_cents": state.company("company_A").commercial.price_cents,
            "threshold_project_contribution_cents": 0,
        },
        source="post-project-schema-test",
    )
    validated = env.validate_action(resolved.action, "company_A").require_valid()

    assert resolved.action.threshold_project_contribution_cents == 0
    assert validated.threshold_project_contribution_cents == 0
    assert resolved.action.to_dict() == validated.to_dict()


def _mutual_aid_shortage_state(config, *, episode_id: str, seed: int):
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B"),
        episode_id=episode_id,
        episode_seed=seed,
        max_rounds=5,
        cooperation_modes=("mutual_aid_v1",),
    )
    receiver = state.company("company_A")
    donor = state.company("company_B")
    receiver = replace(
        receiver,
        operations=replace(
            receiver.operations,
            base_capacity_orders=300,
            effective_capacity_orders=300,
        ),
    )
    donor = replace(
        donor,
        operations=replace(
            donor.operations,
            base_capacity_orders=10_000,
            effective_capacity_orders=10_000,
        ),
    )
    state = _rehash(replace(state, companies=(receiver, donor)))
    env.load_state(state)
    return env, state


def test_bilateral_mutual_aid_requires_a_reciprocal_match_and_replays():
    config = load_market_config(ROOT / "configs" / "market_v6_final.yaml")
    env, state = _mutual_aid_shortage_state(
        config, episode_id="v6-mutual-aid", seed=7301
    )
    manifest = EpisodeManifest.create(env, state, experiment_id="v6-mutual-aid")
    actions = _rule_actions(config, state)
    actions["company_A"] = replace(
        actions["company_A"],
        mutual_aid_partner_company_id="company_B",
        mutual_aid_capacity_request_orders=2_000,
    )
    actions["company_B"] = replace(
        actions["company_B"],
        mutual_aid_partner_company_id="company_A",
        mutual_aid_capacity_offer_orders=2_000,
    )

    result = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    )
    after = result.state_after
    transfer = after.strategic_market.last_mutual_aid_transfers[0]

    assert transfer.donor_company_id == "company_B"
    assert transfer.recipient_company_id == "company_A"
    assert 0 < transfer.fulfilled_orders <= 2_000
    assert after.company("company_A").commercial.mutual_aid_fulfilled_orders == (
        transfer.fulfilled_orders
    )
    assert after.company("company_B").commercial.mutual_aid_provided_orders == (
        transfer.fulfilled_orders
    )
    assert verify_replay(
        MarketEnv(config),
        manifest,
        (MarketTransition.create(state, actions, result),),
    )[-1] == after


def test_unilateral_mutual_aid_request_does_not_create_a_transfer():
    config = load_market_config(ROOT / "configs" / "market_v6_final.yaml")
    env, state = _mutual_aid_shortage_state(
        config, episode_id="v6-mutual-aid-unilateral", seed=7302
    )
    actions = _rule_actions(config, state)
    actions["company_A"] = replace(
        actions["company_A"],
        mutual_aid_partner_company_id="company_B",
        mutual_aid_capacity_request_orders=2_000,
    )

    after = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    ).state_after

    assert after.strategic_market.last_mutual_aid_transfers == ()
    assert after.company("company_A").commercial.mutual_aid_fulfilled_orders == 0


def test_mutual_aid_is_bounded_by_real_spare_capacity_and_shortfall():
    config = load_market_config(ROOT / "configs" / "market_v6_final.yaml")
    env, state = _mutual_aid_shortage_state(
        config, episode_id="v6-mutual-aid-cap", seed=7303
    )
    donor = state.company("company_B")
    donor = replace(
        donor,
        operations=replace(
            donor.operations,
            base_capacity_orders=3_100,
            effective_capacity_orders=3_100,
        ),
    )
    state = _rehash(replace(state, companies=(state.company("company_A"), donor)))
    env.load_state(state)
    actions = _rule_actions(config, state)
    actions["company_A"] = replace(
        actions["company_A"],
        mutual_aid_partner_company_id="company_B",
        mutual_aid_capacity_request_orders=2_000,
    )
    actions["company_B"] = replace(
        actions["company_B"],
        mutual_aid_partner_company_id="company_A",
        mutual_aid_capacity_offer_orders=2_000,
    )

    after = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    ).state_after
    provided = after.company("company_B").commercial.mutual_aid_provided_orders
    own_sales = (
        after.company("company_B").commercial.sales_orders
        - after.company("company_B").commercial.mutual_aid_fulfilled_orders
    )

    assert provided <= 2_000
    assert own_sales + provided <= 3_100


def test_price_coordination_keeps_actual_price_independent_and_updates_trust():
    config = load_market_config(ROOT / "configs" / "market_v6_final.yaml")
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B"),
        episode_id="v6-price-coordination-undercut",
        episode_seed=7401,
        max_rounds=5,
        cooperation_modes=("price_coordination_v1",),
    )
    actions = _rule_actions(config, state)
    actions["company_A"] = replace(
        actions["company_A"],
        price_cents=10_000,
        price_coordination_partner_company_id="company_B",
        price_coordination_target_cents=12_000,
    )
    actions["company_B"] = replace(
        actions["company_B"],
        price_cents=12_000,
        price_coordination_partner_company_id="company_A",
        price_coordination_target_cents=12_000,
    )

    after = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    ).state_after
    outcome = after.strategic_market.last_price_coordination_outcomes[0]
    credibility = dict(
        after.strategic_market.coordination_credibility_by_company_ppm
    )

    assert outcome.status is PriceCoordinationStatus.UNDERCUT_BY_A
    assert outcome.company_a_actual_price_cents == 10_000
    assert outcome.target_price_cents == 12_000
    assert credibility["company_A"] == 325_000
    assert credibility["company_B"] == 675_000


def test_price_coordination_requires_matching_partner_and_target():
    config = load_market_config(ROOT / "configs" / "market_v6_final.yaml")
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B"),
        episode_id="v6-price-coordination-mismatch",
        episode_seed=7402,
        max_rounds=5,
        cooperation_modes=("price_coordination_v1",),
    )
    actions = _rule_actions(config, state)
    actions["company_A"] = replace(
        actions["company_A"],
        price_coordination_partner_company_id="company_B",
        price_coordination_target_cents=12_000,
    )
    actions["company_B"] = replace(
        actions["company_B"],
        price_coordination_partner_company_id="company_A",
        price_coordination_target_cents=11_500,
    )

    after = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    ).state_after

    assert after.strategic_market.last_price_coordination_outcomes == ()
    assert dict(
        after.strategic_market.coordination_credibility_by_company_ppm
    ) == {"company_A": 500_000, "company_B": 500_000}
