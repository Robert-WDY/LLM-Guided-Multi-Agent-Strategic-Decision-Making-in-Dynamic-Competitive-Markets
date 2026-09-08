from pathlib import Path

from game_theory_agent.market import CompanyAction, MarketEnv, MarketState
from game_theory_agent.market.config import load_market_config


ROOT = Path(__file__).resolve().parents[1]


def _one_round(seed: int, price: int):
    env = MarketEnv(load_market_config(ROOT / "configs/market_v9_welfare.yaml"))
    state = env.reset(
        episode_id=f"welfare-{seed}-{price}",
        episode_seed=seed,
        market_model="balanced",
        max_rounds=20,
    )
    actions = {
        company_id: CompanyAction(
            action_id=f"{state.episode_id}:{company_id}",
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=price,
            primary_supplier_id="economy_supplier",
            backup_supplier_id="resilient_supplier",
            primary_supplier_share_ppm=500_000,
        )
        for company_id in state.company_ids
    }
    return env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    ).state_after


def test_explicit_wtp_changes_consumer_surplus_and_ledger_closes() -> None:
    affordable = _one_round(91, 9_000)
    expensive = _one_round(91, 14_000)
    low = affordable.welfare_accounting
    high = expensive.welfare_accounting
    assert low is not None and high is not None
    assert low.round_consumer_surplus_cents > high.round_consumer_surplus_cents
    assert affordable.market.no_purchase_orders < expensive.market.no_purchase_orders
    assert low.round_total_economic_welfare_cents == (
        low.round_consumer_surplus_cents
        + low.round_downstream_producer_surplus_cents
        + low.round_upstream_producer_surplus_cents
        + low.round_government_net_budget_cents
        - low.round_stockout_externality_cents
        - low.round_business_exit_externality_cents
    )


def test_regulatory_fines_are_attributed_as_government_transfers() -> None:
    found = None
    for seed in range(1, 100):
        env = MarketEnv(load_market_config(ROOT / "configs/market_v9_welfare.yaml"))
        state = env.reset(
            episode_id=f"government-{seed}",
            episode_seed=seed,
            market_model="balanced",
            max_rounds=20,
            cooperation_modes=("price_coordination_v1",),
        )
        pairs = {
            "company_A": "company_B",
            "company_B": "company_A",
            "company_C": "company_D",
            "company_D": "company_C",
        }
        actions = {
            company_id: CompanyAction(
                action_id=f"{state.episode_id}:{company_id}",
                episode_id=state.episode_id,
                agent_id=company_id,
                round=state.round,
                state_version=state.state_version,
                price_cents=14_000,
                price_coordination_partner_company_id=pairs[company_id],
                price_coordination_target_cents=14_000,
                primary_supplier_id="economy_supplier",
                backup_supplier_id="resilient_supplier",
                primary_supplier_share_ppm=500_000,
            )
            for company_id in state.company_ids
        }
        after = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", actions
        ).state_after
        welfare = after.welfare_accounting
        assert welfare is not None
        if welfare.round_government_fine_revenue_cents:
            found = after
            break
    assert found is not None
    welfare = found.welfare_accounting
    assert welfare is not None
    assert welfare.round_government_fine_revenue_cents == sum(
        item.financial.round_regulatory_fine_cents or 0
        for item in found.companies
    )
    assert welfare.round_government_net_budget_cents == (
        welfare.round_government_fine_revenue_cents
        - welfare.round_government_enforcement_cost_cents
    )


def test_welfare_state_roundtrips() -> None:
    state = _one_round(93, 10_000)
    assert MarketState.from_dict(state.to_dict()).to_dict() == state.to_dict()

