from dataclasses import replace
from pathlib import Path

import pytest

from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketEnv, MarketState, load_market_config
from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.market.exceptions import StateInvariantError
from game_theory_agent.persistence import CODEC
from game_theory_agent.agents.observation import ObservationBuilder

ROOT = Path(__file__).resolve().parents[1]


def environment(**changes):
    config = load_market_config(ROOT / "configs/market_v12_accounting.yaml")
    data = config.to_dict()
    for section, fields in changes.items():
        data[section].update(fields)
    env = MarketEnv(MarketConfig.from_mapping(data))
    return env, env.reset(episode_id="cash-material-test", episode_seed=16101, max_rounds=20, cooperation_mode="combined_v1")


def step(env, state):
    return env.step(f"{state.episode_id}:{state.round}:{state.state_version}", {
        cid: build_rule_action(env.config, state, cid) for cid in state.company_ids}).state_after


def test_invoices_receipts_costs_and_perishable_materials_close():
    env, state = environment(market={"base_demand_orders": 5000})
    after = step(env, state)
    assert sum(o.material.wasted_orders for o in after.supply_chain.last_procurement_outcomes) > 0
    assert sum(o.material.payment_cents for o in after.supply_chain.last_procurement_outcomes) == sum(s.account.receipts_cents for s in after.supply_chain.suppliers)
    for company in after.companies:
        assert company.financial.cash_balance_cents == state.company(company.company_id).financial.cash_balance_cents + company.financial.round_profit_cents
    env.assert_invariants(after)


@pytest.mark.parametrize("cash", [0, 1000000, 12000000])
def test_low_cash_limits_prepaid_orders_and_never_borrows(cash):
    env, state = environment(company_initial={"cash_balance_cents": cash})
    while not state.terminal:
        state = step(env, state)
        assert all(c.financial.cash_balance_cents >= 0 for c in state.companies)
        assert all(o.material.payment_cents <= o.material.budget_cents for o in state.supply_chain.last_procurement_outcomes)


def test_restore_round_trip_continues_exactly():
    env, state = environment()
    for _ in range(4):
        state = step(env, state)
    assert MarketState.from_dict(state.to_dict()) == state
    other = MarketEnv(env.config)
    other.load_state(CODEC.decode(CODEC.encode(state)))
    while not state.terminal:
        expected = step(other, state)
        state = step(env, state)
        assert state == expected


def test_balanced_tampering_still_rejected_by_cash_and_physical_invariants():
    env, state = environment()
    state = step(env, state)
    original = state.supply_chain.last_procurement_outcomes[0]
    for material in [replace(original.material, payment_cents=original.material.payment_cents+1),
                     replace(original.material, used_orders=original.material.used_orders+1, wasted_orders=original.material.wasted_orders-1)]:
        outcomes = (replace(original, material=material),) + state.supply_chain.last_procurement_outcomes[1:]
        invalid = replace(state, supply_chain=replace(state.supply_chain, last_procurement_outcomes=outcomes))
        invalid = replace(invalid, state_hash=state_hash(invalid.to_dict()))
        with pytest.raises(StateInvariantError):
            env.load_state(invalid)


def test_public_view_excludes_supplier_cash_and_buyer_budget():
    env, state = environment()
    state = step(env, state)
    public = ObservationBuilder().build(state, state.company_ids[0], "public")
    supply = public["market"]["supply_chain"]
    assert all("account" not in s and "unit_cost_cents" not in s for s in supply["suppliers"].values())
    assert all("material" not in o for o in supply["last_procurement_outcomes"].values())
    assert public["own_company"]["financial"]["round_material_payment_cents"] > 0
    perfect = ObservationBuilder().build(state, state.company_ids[0], "perfect")
    assert all("account" in s for s in perfect["market"]["supply_chain"]["suppliers"].values())
