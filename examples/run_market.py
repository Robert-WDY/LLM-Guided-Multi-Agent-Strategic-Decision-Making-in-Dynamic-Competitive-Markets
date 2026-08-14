"""Run one reproducible Engineering MVP v4 market episode."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game_theory_agent.market import (  # noqa: E402
    CompanyAction,
    IncidentResponse,
    IncidentResponseMode,
    MarketEnv,
    load_market_config,
)


def rule_actions(env: MarketEnv):
    state = env.get_state()
    actions = {}
    for index, company_id in enumerate(state.company_ids):
        company = state.company(company_id)
        incident = company.risk.active_incident
        repair = IncidentResponse()
        if (
            incident
            and incident.remaining_repair_cents <= company.financial.cash_balance_cents
        ):
            repair = IncidentResponse(
                IncidentResponseMode.FULL_REPAIR,
                incident.remaining_repair_cents,
            )
        risk_warning = bool(state.risk_signals)
        last_round = state.rounds_remaining <= 1
        actions[company_id] = CompanyAction(
            action_id=f"{state.episode_id}:{state.round}:{company_id}:rule-v1",
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=9_200 + index * 350,
            advertising_budget_cents=600_000 + index * 150_000,
            service_budget_cents=900_000,
            capacity_investment_cents=0 if last_round else 800_000,
            resilience_budget_cents=(
                0 if last_round else (1_200_000 if risk_warning else 200_000)
            ),
            incident_response=repair,
            strategy_summary="deterministic rule baseline",
        )
    return actions


def main() -> None:
    config = load_market_config(ROOT / "configs" / "market_v4.yaml")
    env = MarketEnv(config)
    state = env.reset(episode_id="demo-v4", episode_seed=42)

    while not state.terminal:
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}",
            rule_actions(env),
        )
        state = result.state_after
        print(
            f"Round {result.settled_round}: demand={state.market.realized_demand_orders}, "
            f"no_purchase={state.market.no_purchase_orders}, "
            f"events={[event.event_type for event in state.active_market_events]}"
        )
        for company in state.companies:
            print(
                f"  {company.company_id}: price={company.commercial.price_cents / 100:.2f}, "
                f"share={company.commercial.market_share_ppm / 10_000:.2f}%, "
                f"sales={company.commercial.sales_orders}, "
                f"profit={company.financial.round_profit_cents / 100:.2f}, "
                f"cash={company.financial.cash_balance_cents / 100:.2f}"
            )
        print(f"  state_hash={state.state_hash}")


if __name__ == "__main__":
    main()
