from dataclasses import replace
from pathlib import Path

from game_theory_agent.market import CompanyAction, MarketEnv
from game_theory_agent.market.config import load_market_config
from game_theory_agent.market.protocols import state_hash


ROOT = Path(__file__).resolve().parents[1]


def _env(seed: int = 81) -> tuple[MarketEnv, object]:
    env = MarketEnv(load_market_config(ROOT / "configs/market_v8_supply_chain.yaml"))
    return env, env.reset(
        episode_id=f"supply-test-{seed}",
        episode_seed=seed,
        market_model="balanced",
        max_rounds=20,
    )


def _actions(state, mode: str) -> dict[str, CompanyAction]:
    result = {}
    for company_id in state.company_ids:
        if mode == "cheap":
            primary, backup, share = "economy_supplier", None, 1_000_000
        elif mode == "stable":
            primary, backup, share = "resilient_supplier", None, 1_000_000
        else:
            primary, backup, share = (
                "economy_supplier",
                "resilient_supplier",
                500_000,
            )
        result[company_id] = CompanyAction(
            action_id=f"{state.episode_id}:{state.round}:{company_id}:{mode}",
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=10_000,
            primary_supplier_id=primary,
            backup_supplier_id=backup,
            primary_supplier_share_ppm=share,
        )
    return result


def _force_economy_disruption(env: MarketEnv, state):
    supply = state.supply_chain
    assert supply is not None
    suppliers = tuple(
        replace(
            item,
            disrupted=item.supplier_id == "economy_supplier",
            available_capacity_orders=(
                3_300
                if item.supplier_id == "economy_supplier"
                else item.base_capacity_orders
            ),
        )
        for item in supply.suppliers
    )
    forced = replace(state, supply_chain=replace(supply, suppliers=suppliers), state_hash="")
    forced = replace(forced, state_hash=state_hash(forced.to_dict()))
    env.load_state(forced)
    return forced


def test_supplier_congestion_and_diversification_change_real_capacity() -> None:
    cheap_env, cheap = _env()
    cheap_after = cheap_env.step(
        f"{cheap.episode_id}:{cheap.round}:{cheap.state_version}",
        _actions(cheap, "cheap"),
    ).state_after
    diverse_env, diverse = _env()
    diverse_after = diverse_env.step(
        f"{diverse.episode_id}:{diverse.round}:{diverse.state_version}",
        _actions(diverse, "diverse"),
    ).state_after
    assert sum(
        item.operations.procurement_fulfilled_orders or 0
        for item in cheap_after.companies
    ) == 11_000
    assert sum(
        item.operations.procurement_fulfilled_orders or 0
        for item in diverse_after.companies
    ) == 14_000
    assert cheap_after.company("company_A").operations.actual_unit_cost_cents < (
        diverse_after.company("company_A").operations.actual_unit_cost_cents
    )


def test_diversification_limits_a_forced_supplier_disruption() -> None:
    cheap_env, cheap = _env(82)
    cheap = _force_economy_disruption(cheap_env, cheap)
    cheap_after = cheap_env.step(
        f"{cheap.episode_id}:{cheap.round}:{cheap.state_version}",
        _actions(cheap, "cheap"),
    ).state_after
    diverse_env, diverse = _env(82)
    diverse = _force_economy_disruption(diverse_env, diverse)
    diverse_after = diverse_env.step(
        f"{diverse.episode_id}:{diverse.round}:{diverse.state_version}",
        _actions(diverse, "diverse"),
    ).state_after
    cheap_inputs = sum(
        item.operations.procurement_fulfilled_orders or 0
        for item in cheap_after.companies
    )
    diverse_inputs = sum(
        item.operations.procurement_fulfilled_orders or 0
        for item in diverse_after.companies
    )
    assert cheap_inputs == 3_300
    assert diverse_inputs == 10_300
    assert diverse_inputs > cheap_inputs


def test_supply_chain_roundtrip_and_action_validation() -> None:
    env, state = _env(83)
    action = _actions(state, "diverse")["company_A"]
    assert env.validate_action(action, "company_A").valid
    restored = type(state).from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()

