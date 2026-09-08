"""Versioned visibility policies and the single TrueState -> view projection."""

from __future__ import annotations
from copy import deepcopy

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from game_theory_agent.market.models import CompanyState, MarketState


class InformationMode(str, Enum):
    PERFECT = "perfect"
    PUBLIC = "public"


@dataclass(frozen=True, slots=True)
class VisibilityPolicy:
    """Auditable field-level policy for one information treatment."""

    policy_version: str
    information_mode: InformationMode
    public_state_schema_version: str
    private_state_schema_version: str
    public_company_fields: tuple[str, ...]
    public_market_fields: tuple[str, ...]
    public_event_fields: tuple[str, ...]
    opponents_receive_full_state: bool
    own_company_receives_full_state: bool = True
    belief_schema_version: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "information_mode": self.information_mode.value,
            "public_state_schema_version": self.public_state_schema_version,
            "private_state_schema_version": self.private_state_schema_version,
            "public_company_fields": list(self.public_company_fields),
            "public_market_fields": list(self.public_market_fields),
            "public_event_fields": list(self.public_event_fields),
            "opponents_receive_full_state": self.opponents_receive_full_state,
            "own_company_receives_full_state": (
                self.own_company_receives_full_state
            ),
            "belief_schema_version": self.belief_schema_version,
            "controller_only_fields": [
                "episode_seed",
                "future_random_draws",
                "opponent_private_financials",
                "opponent_internal_operations",
                "opponent_persona",
                "opponent_plan",
            ],
        }


PUBLIC_COMPANY_FIELDS = (
    "company_id",
    "price_cents",
    "market_share_ppm",
    "sales_orders",
    "reputation_ppm",
)
PUBLIC_MARKET_FIELDS = (
    "base_demand_orders",
    "realized_demand_orders",
    "no_purchase_orders",
    "lost_after_stockout_orders",
    "market_sentiment_ppm",
    "actual_supply_cost_index_ppm",
    "average_paid_price_cents",
    "market_model_id",
    "market_model_label",
    "market_model_description",
)
PUBLIC_EVENT_FIELDS = (
    "event_id",
    "event_type",
    "severity",
    "started_round",
    "remaining_rounds",
)

PERFECT_VISIBILITY_POLICY = VisibilityPolicy(
    policy_version="visibility-perfect-v1.0.0",
    information_mode=InformationMode.PERFECT,
    public_state_schema_version="public-state-v1.0.0",
    private_state_schema_version="private-state-v1.0.0",
    public_company_fields=PUBLIC_COMPANY_FIELDS,
    public_market_fields=PUBLIC_MARKET_FIELDS,
    public_event_fields=PUBLIC_EVENT_FIELDS,
    opponents_receive_full_state=True,
)
PUBLIC_VISIBILITY_POLICY = VisibilityPolicy(
    policy_version="visibility-public-v2.0.0",
    information_mode=InformationMode.PUBLIC,
    public_state_schema_version="public-state-v1.0.0",
    private_state_schema_version="private-state-v1.0.0",
    public_company_fields=PUBLIC_COMPANY_FIELDS,
    public_market_fields=PUBLIC_MARKET_FIELDS,
    public_event_fields=PUBLIC_EVENT_FIELDS,
    opponents_receive_full_state=False,
)
VISIBILITY_POLICIES = {
    InformationMode.PERFECT: PERFECT_VISIBILITY_POLICY,
    InformationMode.PUBLIC: PUBLIC_VISIBILITY_POLICY,
}


def visibility_policy_for(
    information_mode: str | InformationMode,
) -> VisibilityPolicy:
    try:
        mode = InformationMode(information_mode)
    except ValueError as exc:
        raise ValueError(
            f"unsupported information mode: {information_mode}"
        ) from exc
    return VISIBILITY_POLICIES[mode]


class ObservationBuilder:
    """The only authoritative TrueState -> company view projection."""

    @staticmethod
    def _company_source(company: CompanyState) -> dict[str, Any]:
        return {
            "company_id": company.company_id,
            "price_cents": company.commercial.price_cents,
            "market_share_ppm": company.commercial.market_share_ppm,
            "sales_orders": company.commercial.sales_orders,
            "reputation_ppm": company.brand.reputation_ppm,
        }

    @classmethod
    def public_company(
        cls,
        company: CompanyState,
        policy: VisibilityPolicy = PUBLIC_VISIBILITY_POLICY,
    ) -> dict[str, Any]:
        source = cls._company_source(company)
        return {field: source[field] for field in policy.public_company_fields}

    @staticmethod
    def _project(
        source: dict[str, Any], fields: tuple[str, ...]
    ) -> dict[str, Any]:
        return {field: source[field] for field in fields}

    def build(
        self,
        state: MarketState,
        company_id: str,
        information_mode: str | InformationMode,
        *,
        belief_state: Mapping[str, Any] | None = None,
        belief_hash: str | None = None,
        belief_schema_version: str = "none",
    ) -> dict[str, Any]:
        policy = visibility_policy_for(information_mode)
        if company_id not in state.company_ids:
            raise KeyError(company_id)

        public_companies = [
            self.public_company(item, policy) for item in state.companies
        ]
        public_market = self._project(
            state.market.to_dict(), policy.public_market_fields
        )
        full_market = state.market.to_dict()
        if state.strategic_market is not None:
            strategic_market = state.strategic_market.to_dict()
            public_market["strategic_market"] = strategic_market
            full_market["strategic_market"] = strategic_market
        if state.supply_chain is not None:
            supply_chain = state.supply_chain.to_dict()
            public_market["supply_chain"] = supply_chain
            full_market["supply_chain"] = supply_chain
            if any(s.account is not None for s in state.supply_chain.suppliers):
                public_supply = deepcopy(supply_chain)
                for supplier in public_supply["suppliers"].values():
                    supplier.pop("account", None)
                    supplier.pop("unit_cost_cents", None)
                    supplier.pop("investment_decision", None)
                    supplier.pop("round_requested_orders", None)
                    strategic=supplier.pop("strategic_ledger",None)
                    if strategic is not None:
                        supplier["bankrupt"]=strategic["bankrupt"]
                        # v14 declares settled contracts, stock and creditor books
                        # public. Supplier cash, costs and rejected bids stay private.
                        report=deepcopy(strategic)
                        if report.get("audit"):
                            for key in ("opening_cash_cents","opening_lender_cash_cents","unit_cost_cents","hash","negotiations"):
                                report["audit"].pop(key,None)
                        supplier["strategic_public"]=report
                for outcome in public_supply["last_procurement_outcomes"].values():
                    outcome.pop("material", None)
                public_market["supply_chain"] = public_supply
        if state.government is not None:
            public_market["government"] = state.government.to_dict()
            public_market["consumer_decisions"] = [d.to_dict() for d in state.consumer_decisions]
            full_market.update({k:public_market[k] for k in ("government","consumer_decisions")})
        if state.welfare_accounting is not None:
            welfare = state.welfare_accounting.to_dict()
            public_market["welfare_accounting"] = welfare
            full_market["welfare_accounting"] = welfare
        public_events = [
            self._project(item.to_dict(), policy.public_event_fields)
            for item in state.active_market_events
        ]
        shared_resilience = (
            state.shared_resilience.to_dict()
            if state.shared_resilience is not None
            else None
        )
        risk_signals = [item.to_dict() for item in state.risk_signals]
        own_company = state.company(company_id).to_dict()
        competitors = [
            (
                item.to_dict()
                if policy.opponents_receive_full_state
                else self.public_company(item, policy)
            )
            for item in state.companies
            if item.company_id != company_id
        ]
        public_state = {
            "schema_version": policy.public_state_schema_version,
            "episode_id": state.episode_id,
            "round": state.round,
            "rounds_remaining": state.rounds_remaining,
            "state_version": state.state_version,
            "terminal": state.terminal,
            "market": public_market,
            "shared_resilience": shared_resilience,
            "risk_signals": risk_signals,
            "active_market_events": public_events,
            "companies": public_companies,
        }
        private_state = {
            "schema_version": policy.private_state_schema_version,
            "company_id": company_id,
            "company": own_company,
        }
        return {
            "information_mode": policy.information_mode.value,
            "visibility_policy_version": policy.policy_version,
            "visibility_policy": {
                **policy.to_dict(),
                "belief_schema_version": belief_schema_version,
            },
            "belief_schema_version": belief_schema_version,
            "belief_hash": belief_hash,
            "belief_state": dict(belief_state) if belief_state is not None else None,
            "public_state": public_state,
            "private_state": private_state,
            "own_company": own_company,
            "competitors": competitors,
            "public_companies": public_companies,
            "market": (
                full_market
                if policy.opponents_receive_full_state
                else public_market
            ),
            "shared_resilience": shared_resilience,
            "risk_signals": risk_signals,
            "active_market_events": (
                [item.to_dict() for item in state.active_market_events]
                if policy.opponents_receive_full_state
                else public_events
            ),
        }

    def build_company_views(
        self,
        state: MarketState,
        company_id: str,
        information_mode: str | InformationMode,
    ) -> dict[str, Any]:
        """Compatibility alias; all callers still use the same projection."""

        return self.build(state, company_id, information_mode)
