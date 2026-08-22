"""Controlled same-seed one-step counterfactual evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.economics import decision_support_metrics
from game_theory_agent.market import CompanyAction, MarketConfig, MarketEnv, MarketState
from game_theory_agent.market.config import load_market_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CounterfactualEvaluator:
    def __init__(self, config: MarketConfig | None = None) -> None:
        self.config = config or load_market_config(
            Path(
                os.environ.get(
                    "MARKET_CONFIG_PATH",
                    PROJECT_ROOT / "configs" / "market_v4.yaml",
                )
            )
        )

    def evaluate(
        self,
        state_before: MarketState,
        state_after: MarketState,
        joint_actions: dict[str, dict[str, Any]],
        company_id: str,
    ) -> dict[str, Any]:
        actual = state_after.company(company_id)
        actual_action = CompanyAction.from_dict(joint_actions[company_id])
        support = decision_support_metrics(self.config, state_before, company_id)
        candidates = {
            "cash_preservation": {
                "price_cents": max(
                    actual_action.price_cents,
                    int(support["minimum_safe_price_cents"]),
                ),
                "advertising_budget_cents": 0,
                "service_budget_cents": 0,
                "capacity_investment_cents": 0,
                "resilience_budget_cents": 0,
                "incident_response": {
                    "mode": "wait",
                    "repair_budget_cents": 0,
                },
                "strategy_summary": "counterfactual cash preservation",
            },
            "margin_recovery": {
                "price_cents": max(
                    actual_action.price_cents,
                    int(support["estimated_break_even_price_cents"]),
                    int(support["minimum_safe_price_cents"]),
                ),
                "advertising_budget_cents": min(
                    actual_action.advertising_budget_cents, 300_000
                ),
                "service_budget_cents": min(
                    actual_action.service_budget_cents, 300_000
                ),
                "capacity_investment_cents": 0,
                "resilience_budget_cents": 0,
                "incident_response": {
                    "mode": "wait",
                    "repair_budget_cents": 0,
                },
                "strategy_summary": "counterfactual margin recovery",
            },
        }
        outcomes: list[dict[str, Any]] = []
        for name, request in candidates.items():
            resolution = resolve_action_request(
                self.config,
                state_before,
                company_id,
                request,
                source=f"counterfactual:{name}",
                action_id=(
                    f"counterfactual:{state_before.episode_id}:"
                    f"{state_before.round}:{company_id}:{name}"
                ),
            )
            actions = {
                item_id: CompanyAction.from_dict(payload)
                for item_id, payload in joint_actions.items()
            }
            actions[company_id] = resolution.action
            env = MarketEnv(self.config)
            env.load_state(state_before)
            result = env.step(
                (
                    f"{state_before.episode_id}:{state_before.round}:"
                    f"{state_before.state_version}"
                ),
                actions,
            )
            company = result.state_after.company(company_id)
            outcomes.append(
                {
                    "alternative": name,
                    "action": resolution.action.to_dict(),
                    "adjustments": [
                        item.to_dict() for item in resolution.adjustments
                    ],
                    "round_profit_cents": company.financial.round_profit_cents,
                    "cash_balance_cents": company.financial.cash_balance_cents,
                    "market_share_ppm": company.commercial.market_share_ppm,
                    "sales_orders": company.commercial.sales_orders,
                    "delta_vs_actual": {
                        "round_profit_cents": (
                            company.financial.round_profit_cents
                            - actual.financial.round_profit_cents
                        ),
                        "cash_balance_cents": (
                            company.financial.cash_balance_cents
                            - actual.financial.cash_balance_cents
                        ),
                        "market_share_ppm": (
                            company.commercial.market_share_ppm
                            - actual.commercial.market_share_ppm
                        ),
                        "sales_orders": (
                            company.commercial.sales_orders
                            - actual.commercial.sales_orders
                        ),
                    },
                    "state_hash": result.state_after.state_hash,
                    "invariant_results": list(result.invariant_results),
                }
            )
        return {
            "counterfactual_schema_version": "counterfactual-v1.0.0",
            "method": "same_state_same_seed_other_company_actions_fixed",
            "state_before_hash": state_before.state_hash,
            "actual_state_after_hash": state_after.state_hash,
            "company_id": company_id,
            "alternatives": outcomes,
        }
