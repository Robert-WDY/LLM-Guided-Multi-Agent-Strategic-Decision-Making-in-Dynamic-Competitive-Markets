"""Immutable fixed-point domain models for Engineering MVP v4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class Level(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Persona(StrEnum):
    NONE = "none"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    STAKEHOLDER_BALANCED = "stakeholder_balanced"
    PUBLIC_SERVICE = "public_service"
    RESILIENCE_STEWARD = "resilience_steward"


class IncidentResponseMode(StrEnum):
    WAIT = "wait"
    PARTIAL_REPAIR = "partial_repair"
    FULL_REPAIR = "full_repair"


class CompanyOperatingStatus(StrEnum):
    OPERATING = "operating"
    DISTRESSED = "distressed"
    EXITED = "exited"


class ConcentrationRegime(StrEnum):
    COMPETITIVE = "competitive"
    MODERATE = "moderate"
    CONCENTRATED = "concentrated"
    DOMINANT = "dominant"
    MONOPOLY = "monopoly"
    NO_ACTIVE_MARKET = "no_active_market"


class ThresholdProjectStatus(StrEnum):
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PriceCoordinationStatus(StrEnum):
    HONORED = "honored"
    UNDERCUT_BY_A = "undercut_by_a"
    UNDERCUT_BY_B = "undercut_by_b"
    MUTUAL_DEVIATION = "mutual_deviation"


@dataclass(frozen=True, slots=True)
class IncidentResponse:
    mode: IncidentResponseMode = IncidentResponseMode.WAIT
    repair_budget_cents: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "repair_budget_cents": self.repair_budget_cents,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IncidentResponse":
        return cls(
            mode=IncidentResponseMode(data.get("mode", "wait")),
            repair_budget_cents=int(data.get("repair_budget_cents", 0)),
        )


@dataclass(frozen=True, slots=True)
class CompanyAction:
    action_id: str
    episode_id: str
    agent_id: str
    round: int
    state_version: int
    price_cents: int
    advertising_budget_cents: int = 0
    service_budget_cents: int = 0
    capacity_investment_cents: int = 0
    resilience_budget_cents: int = 0
    shared_resilience_contribution_cents: int | None = None
    threshold_project_contribution_cents: int | None = None
    mutual_aid_partner_company_id: str | None = None
    mutual_aid_capacity_offer_orders: int | None = None
    mutual_aid_capacity_request_orders: int | None = None
    price_coordination_partner_company_id: str | None = None
    price_coordination_target_cents: int | None = None
    primary_supplier_id: str | None = None
    backup_supplier_id: str | None = None
    primary_supplier_share_ppm: int | None = None
    contract_quantity_orders: int | None = None
    contract_duration_rounds: int | None = None
    contract_bid_cents: int | None = None
    procurement_quantity_orders: int | None = None
    incident_response: IncidentResponse = IncidentResponse()
    strategy_summary: str = ""

    @property
    def fixed_spend_cents(self) -> int:
        return (
            self.advertising_budget_cents
            + self.service_budget_cents
            + self.capacity_investment_cents
            + self.resilience_budget_cents
            + (self.shared_resilience_contribution_cents or 0)
            + (self.threshold_project_contribution_cents or 0)
            + self.incident_response.repair_budget_cents
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "action_id": self.action_id,
            "episode_id": self.episode_id,
            "agent_id": self.agent_id,
            "round": self.round,
            "state_version": self.state_version,
            "price_cents": self.price_cents,
            "advertising_budget_cents": self.advertising_budget_cents,
            "service_budget_cents": self.service_budget_cents,
            "capacity_investment_cents": self.capacity_investment_cents,
            "resilience_budget_cents": self.resilience_budget_cents,
            "incident_response": self.incident_response.to_dict(),
            "strategy_summary": self.strategy_summary,
        }
        if self.shared_resilience_contribution_cents is not None:
            payload["shared_resilience_contribution_cents"] = (
                self.shared_resilience_contribution_cents
            )
        if self.threshold_project_contribution_cents is not None:
            payload["threshold_project_contribution_cents"] = (
                self.threshold_project_contribution_cents
            )
        if self.mutual_aid_partner_company_id is not None:
            payload["mutual_aid_partner_company_id"] = (
                self.mutual_aid_partner_company_id
            )
        if self.mutual_aid_capacity_offer_orders is not None:
            payload["mutual_aid_capacity_offer_orders"] = (
                self.mutual_aid_capacity_offer_orders
            )
        if self.mutual_aid_capacity_request_orders is not None:
            payload["mutual_aid_capacity_request_orders"] = (
                self.mutual_aid_capacity_request_orders
            )
        if self.price_coordination_partner_company_id is not None:
            payload["price_coordination_partner_company_id"] = (
                self.price_coordination_partner_company_id
            )
        if self.price_coordination_target_cents is not None:
            payload["price_coordination_target_cents"] = (
                self.price_coordination_target_cents
            )
        if self.primary_supplier_id is not None:
            payload["primary_supplier_id"] = self.primary_supplier_id
        if self.backup_supplier_id is not None:
            payload["backup_supplier_id"] = self.backup_supplier_id
        if self.primary_supplier_share_ppm is not None:
            payload["primary_supplier_share_ppm"] = (
                self.primary_supplier_share_ppm
            )
        for key in ("contract_quantity_orders", "contract_duration_rounds", "contract_bid_cents"):
            if getattr(self,key) is not None: payload[key]=getattr(self,key)
        if self.procurement_quantity_orders is not None:
            payload["procurement_quantity_orders"] = self.procurement_quantity_orders
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompanyAction":
        return cls(
            **{key:data.get(key) for key in ("contract_quantity_orders", "contract_duration_rounds", "contract_bid_cents")},
            procurement_quantity_orders=data.get("procurement_quantity_orders"),
            action_id=str(data["action_id"]),
            episode_id=str(data["episode_id"]),
            agent_id=str(data["agent_id"]),
            round=int(data["round"]),
            state_version=int(data["state_version"]),
            price_cents=int(data["price_cents"]),
            advertising_budget_cents=int(data.get("advertising_budget_cents", 0)),
            service_budget_cents=int(data.get("service_budget_cents", 0)),
            capacity_investment_cents=int(data.get("capacity_investment_cents", 0)),
            resilience_budget_cents=int(data.get("resilience_budget_cents", 0)),
            shared_resilience_contribution_cents=(
                int(data["shared_resilience_contribution_cents"])
                if data.get("shared_resilience_contribution_cents") is not None
                else None
            ),
            threshold_project_contribution_cents=(
                int(data["threshold_project_contribution_cents"])
                if data.get("threshold_project_contribution_cents") is not None
                else None
            ),
            mutual_aid_partner_company_id=(
                str(data["mutual_aid_partner_company_id"])
                if data.get("mutual_aid_partner_company_id") is not None
                else None
            ),
            mutual_aid_capacity_offer_orders=(
                int(data["mutual_aid_capacity_offer_orders"])
                if data.get("mutual_aid_capacity_offer_orders") is not None
                else None
            ),
            mutual_aid_capacity_request_orders=(
                int(data["mutual_aid_capacity_request_orders"])
                if data.get("mutual_aid_capacity_request_orders") is not None
                else None
            ),
            price_coordination_partner_company_id=(
                str(data["price_coordination_partner_company_id"])
                if data.get("price_coordination_partner_company_id")
                is not None
                else None
            ),
            price_coordination_target_cents=(
                int(data["price_coordination_target_cents"])
                if data.get("price_coordination_target_cents") is not None
                else None
            ),
            primary_supplier_id=(
                str(data["primary_supplier_id"])
                if data.get("primary_supplier_id") is not None
                else None
            ),
            backup_supplier_id=(
                str(data["backup_supplier_id"])
                if data.get("backup_supplier_id") is not None
                else None
            ),
            primary_supplier_share_ppm=(
                int(data["primary_supplier_share_ppm"])
                if data.get("primary_supplier_share_ppm") is not None
                else None
            ),
            incident_response=IncidentResponse.from_dict(
                data.get("incident_response", {})
            ),
            strategy_summary=str(data.get("strategy_summary", "")),
        )


Action = CompanyAction


@dataclass(frozen=True, slots=True)
class RiskSignal:
    signal_id: str
    event_type: str
    target_round: int
    estimated_probability_ppm: int
    severity: str
    lead_time_rounds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "event_type": self.event_type,
            "target_round": self.target_round,
            "estimated_probability_ppm": self.estimated_probability_ppm,
            "severity": self.severity,
            "lead_time_rounds": self.lead_time_rounds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RiskSignal":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class MarketEvent:
    event_id: str
    event_type: str
    severity: str
    started_round: int
    remaining_rounds: int
    demand_multiplier_ppm: int
    supply_cost_multiplier_ppm: int
    capacity_multiplier_ppm: int
    advertising_multiplier_ppm: int
    service_penalty_ppm: int
    reputation_penalty_ppm: int

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketEvent":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class CompanyIncident:
    incident_id: str
    incident_type: str
    severity: str
    started_round: int
    remaining_rounds: int
    repair_required_cents: int
    accumulated_repair_cents: int
    capacity_multiplier_ppm: int
    advertising_multiplier_ppm: int
    service_penalty_ppm: int
    reputation_penalty_ppm: int
    refund_rate_ppm: int

    @property
    def remaining_repair_cents(self) -> int:
        return max(0, self.repair_required_cents - self.accumulated_repair_cents)

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompanyIncident":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class FinancialState:
    cash_balance_cents: int
    round_revenue_cents: int = 0
    round_variable_cost_cents: int = 0
    round_fixed_spend_cents: int = 0
    round_incident_cost_cents: int = 0
    round_operating_cost_cents: int = 0
    round_profit_cents: int = 0
    cumulative_profit_cents: int = 0
    capacity_book_value_cents: int = 0
    round_regulatory_fine_cents: int | None = None
    round_material_payment_cents: int | None = None
    round_government_support_cents: int | None = None

    def to_dict(self) -> dict[str, int]:
        return {
            field: value
            for field in self.__dataclass_fields__
            if (value := getattr(self, field)) is not None
        }


@dataclass(frozen=True, slots=True)
class CommercialState:
    price_cents: int
    market_share_ppm: int = 0
    potential_demand_orders: int = 0
    sales_orders: int = 0
    attempted_unfulfilled_orders: int = 0
    orders_received_from_redistribution: int = 0
    orders_lost_after_redistribution: int = 0
    mutual_aid_fulfilled_orders: int | None = None
    mutual_aid_provided_orders: int | None = None

    def to_dict(self) -> dict[str, int]:
        return {
            field: value
            for field in self.__dataclass_fields__
            if (value := getattr(self, field)) is not None
        }


@dataclass(frozen=True, slots=True)
class OperationsState:
    base_capacity_orders: int
    effective_capacity_orders: int
    financial_capacity_orders: int
    capacity_utilization_ppm: int
    base_unit_cost_cents: int
    actual_unit_cost_cents: int
    procurement_requested_orders: int | None = None
    procurement_fulfilled_orders: int | None = None
    procurement_unit_input_price_cents: int | None = None

    def to_dict(self) -> dict[str, int]:
        return {
            field: value
            for field in self.__dataclass_fields__
            if (value := getattr(self, field)) is not None
        }


@dataclass(frozen=True, slots=True)
class SupplierQuoteDecision:
    policy_version: str
    observed_round: int
    applies_round: int
    previous_price_cents: int
    quoted_price_cents: int
    observed_sales_orders: int
    observed_capacity_orders: int
    utilization_ppm: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class SupplierAccount:
    opening_cash_cents: int
    cash_cents: int
    receipts_cents: int = 0
    production_cost_cents: int = 0
    settlement_price_cents: int = 0
    investment_cents: int | None = None

    def to_dict(self) -> dict[str, int]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__ if getattr(self, key) is not None}


@dataclass(frozen=True, slots=True)
class SupplierInvestmentDecision:
    observed_round: int
    applies_round: int
    observed_sales_orders: int
    observed_capacity_orders: int
    observed_cash_cents: int
    previous_base_capacity_orders: int
    added_capacity_orders: int
    investment_cents: int
    reason: str
    policy_version: str = "cash-bounded-investment-v1.0.0"
    observed_requested_orders: int | None = None
    observed_unit_margin_cents: int | None = None
    remaining_rounds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__ if getattr(self,key) is not None}


@dataclass(frozen=True, slots=True)
class MaterialSettlement:
    budget_cents: int
    payment_cents: int
    payments_by_supplier: tuple[tuple[str, int], ...]
    used_orders: int = 0
    wasted_orders: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"budget_cents": self.budget_cents, "payment_cents": self.payment_cents,
                "payments_by_supplier": dict(self.payments_by_supplier),
                "used_orders": self.used_orders, "wasted_orders": self.wasted_orders}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaterialSettlement":
        return cls(**{**data, "payments_by_supplier": tuple(sorted(data["payments_by_supplier"].items()))})


@dataclass(frozen=True, slots=True)
class SupplierState:
    supplier_id: str
    label: str
    unit_price_cents: int
    unit_cost_cents: int
    base_capacity_orders: int
    available_capacity_orders: int
    reliability_ppm: int
    disrupted: bool = False
    round_sales_orders: int = 0
    round_profit_cents: int = 0
    cumulative_profit_cents: int = 0
    last_settled_unit_price_cents: int | None = None
    quote_decision: SupplierQuoteDecision | None = None
    account: SupplierAccount | None = None
    investment_decision: SupplierInvestmentDecision | None = None
    round_requested_orders: int | None = None
    strategic_ledger: str | None = None

    def to_dict(self) -> dict[str, Any]:
        # Omit optional v11 fields so legacy state/replay hashes remain stable.
        payload = {field: getattr(self, field) for field in self.__dataclass_fields__
                   if field not in {"last_settled_unit_price_cents", "quote_decision", "account", "investment_decision", "round_requested_orders", "strategic_ledger"}}
        if self.strategic_ledger is not None: payload["strategic_ledger"] = json.loads(self.strategic_ledger)
        if self.round_requested_orders is not None:
            payload["round_requested_orders"] = self.round_requested_orders
        if self.investment_decision is not None:
            payload["investment_decision"] = self.investment_decision.to_dict()
        if self.account is not None:
            payload["account"] = self.account.to_dict()
        if self.last_settled_unit_price_cents is not None:
            payload["last_settled_unit_price_cents"] = self.last_settled_unit_price_cents
        if self.quote_decision is not None:
            payload["quote_decision"] = self.quote_decision.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SupplierState":
        payload = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        if payload.get("strategic_ledger") is not None: payload["strategic_ledger"] = json.dumps(payload["strategic_ledger"],ensure_ascii=False,sort_keys=True,separators=(",",":"))
        if payload.get("quote_decision") is not None:
            payload["quote_decision"] = SupplierQuoteDecision(**payload["quote_decision"])
        if payload.get("account") is not None:
            payload["account"] = SupplierAccount(**payload["account"])
        if payload.get("investment_decision") is not None:
            payload["investment_decision"] = SupplierInvestmentDecision(**payload["investment_decision"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ProcurementOutcome:
    company_id: str
    primary_supplier_id: str
    backup_supplier_id: str | None
    primary_supplier_share_ppm: int
    requested_orders: int
    fulfilled_orders: int
    unfulfilled_orders: int
    weighted_unit_input_price_cents: int
    supplier_allocation_orders: tuple[tuple[str, int], ...]
    material: MaterialSettlement | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_id": self.company_id,
            **({"material": self.material.to_dict()} if self.material is not None else {}),
            "primary_supplier_id": self.primary_supplier_id,
            "backup_supplier_id": self.backup_supplier_id,
            "primary_supplier_share_ppm": self.primary_supplier_share_ppm,
            "requested_orders": self.requested_orders,
            "fulfilled_orders": self.fulfilled_orders,
            "unfulfilled_orders": self.unfulfilled_orders,
            "weighted_unit_input_price_cents": (
                self.weighted_unit_input_price_cents
            ),
            "supplier_allocation_orders": dict(
                self.supplier_allocation_orders
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcurementOutcome":
        return cls(
            material=(MaterialSettlement.from_dict(data["material"]) if data.get("material") is not None else None),
            company_id=str(data["company_id"]),
            primary_supplier_id=str(data["primary_supplier_id"]),
            backup_supplier_id=(
                str(data["backup_supplier_id"])
                if data.get("backup_supplier_id") is not None
                else None
            ),
            primary_supplier_share_ppm=int(
                data["primary_supplier_share_ppm"]
            ),
            requested_orders=int(data["requested_orders"]),
            fulfilled_orders=int(data["fulfilled_orders"]),
            unfulfilled_orders=int(data["unfulfilled_orders"]),
            weighted_unit_input_price_cents=int(
                data["weighted_unit_input_price_cents"]
            ),
            supplier_allocation_orders=tuple(
                sorted(
                    (str(key), int(value))
                    for key, value in data.get(
                        "supplier_allocation_orders", {}
                    ).items()
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class SupplyChainState:
    protocol_version: str
    suppliers: tuple[SupplierState, ...]
    last_procurement_outcomes: tuple[ProcurementOutcome, ...] = ()
    round_upstream_producer_surplus_cents: int = 0
    cumulative_upstream_producer_surplus_cents: int = 0

    @property
    def supplier_ids(self) -> tuple[str, ...]:
        return tuple(item.supplier_id for item in self.suppliers)

    def supplier(self, supplier_id: str) -> SupplierState:
        for item in self.suppliers:
            if item.supplier_id == supplier_id:
                return item
        raise KeyError(supplier_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "supplier_order": list(self.supplier_ids),
            "suppliers": {
                item.supplier_id: item.to_dict() for item in self.suppliers
            },
            "last_procurement_outcomes": {
                item.company_id: item.to_dict()
                for item in self.last_procurement_outcomes
            },
            "round_upstream_producer_surplus_cents": (
                self.round_upstream_producer_surplus_cents
            ),
            "cumulative_upstream_producer_surplus_cents": (
                self.cumulative_upstream_producer_surplus_cents
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SupplyChainState":
        suppliers = data.get("suppliers", {})
        order = tuple(data.get("supplier_order", suppliers.keys()))
        outcomes = data.get("last_procurement_outcomes", {})
        return cls(
            protocol_version=str(data["protocol_version"]),
            suppliers=tuple(
                SupplierState.from_dict(suppliers[key]) for key in order
            ),
            last_procurement_outcomes=tuple(
                ProcurementOutcome.from_dict(outcomes[key])
                for key in sorted(outcomes)
            ),
            round_upstream_producer_surplus_cents=int(
                data.get("round_upstream_producer_surplus_cents", 0)
            ),
            cumulative_upstream_producer_surplus_cents=int(
                data.get("cumulative_upstream_producer_surplus_cents", 0)
            ),
        )


@dataclass(frozen=True, slots=True)
class WelfareAccountingState:
    """Explicit value-chain and government welfare ledger."""

    protocol_version: str
    valuation_source: str
    round_consumer_surplus_cents: int = 0
    round_downstream_producer_surplus_cents: int = 0
    round_upstream_producer_surplus_cents: int = 0
    round_government_fine_revenue_cents: int = 0
    round_government_enforcement_cost_cents: int = 0
    round_government_net_budget_cents: int = 0
    round_stockout_externality_cents: int = 0
    round_business_exit_externality_cents: int = 0
    round_total_economic_welfare_cents: int = 0
    round_service_continuity_orders: int = 0
    round_voluntary_no_purchase_orders: int = 0
    cumulative_consumer_surplus_cents: int = 0
    cumulative_downstream_producer_surplus_cents: int = 0
    cumulative_upstream_producer_surplus_cents: int = 0
    cumulative_government_net_budget_cents: int = 0
    cumulative_externality_cost_cents: int = 0
    cumulative_total_economic_welfare_cents: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WelfareAccountingState":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class BrandState:
    brand_awareness_ppm: int
    service_quality_ppm: int
    reputation_ppm: int
    last_attempted_unfulfilled_rate_ppm: int = 0

    def to_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class RiskState:
    resilience_ppm: int
    active_incident: CompanyIncident | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "resilience_ppm": self.resilience_ppm,
            "active_incident": self.active_incident.to_dict()
            if self.active_incident
            else None,
        }


@dataclass(frozen=True, slots=True)
class CompanyHistory:
    last_action_id: str | None = None
    last_action: CompanyAction | None = None
    recent_profit_cents: tuple[int, ...] = ()
    recent_market_share_ppm: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_action_id": self.last_action_id,
            "last_action": self.last_action.to_dict() if self.last_action else None,
            "recent_profit_cents": list(self.recent_profit_cents),
            "recent_market_share_ppm": list(self.recent_market_share_ppm),
        }


@dataclass(frozen=True, slots=True)
class CompanyState:
    company_id: str
    persona: Persona
    financial: FinancialState
    commercial: CommercialState
    operations: OperationsState
    brand: BrandState
    risk: RiskState
    history: CompanyHistory = CompanyHistory()

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_id": self.company_id,
            "persona": self.persona.value,
            "financial": self.financial.to_dict(),
            "commercial": self.commercial.to_dict(),
            "operations": self.operations.to_dict(),
            "brand": self.brand.to_dict(),
            "risk": self.risk.to_dict(),
            "history": self.history.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompanyState":
        risk = data["risk"]
        history = data.get("history", {})
        return cls(
            company_id=str(data["company_id"]),
            persona=Persona(data.get("persona", "none")),
            financial=FinancialState(**data["financial"]),
            commercial=CommercialState(**data["commercial"]),
            operations=OperationsState(**data["operations"]),
            brand=BrandState(**data["brand"]),
            risk=RiskState(
                resilience_ppm=int(risk["resilience_ppm"]),
                active_incident=(
                    CompanyIncident.from_dict(risk["active_incident"])
                    if risk.get("active_incident")
                    else None
                ),
            ),
            history=CompanyHistory(
                last_action_id=history.get("last_action_id"),
                last_action=(
                    CompanyAction.from_dict(history["last_action"])
                    if history.get("last_action")
                    else None
                ),
                recent_profit_cents=tuple(history.get("recent_profit_cents", ())),
                recent_market_share_ppm=tuple(
                    history.get("recent_market_share_ppm", ())
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    base_demand_orders: int
    realized_demand_orders: int
    no_purchase_orders: int
    lost_after_stockout_orders: int
    market_sentiment_ppm: int
    base_supply_cost_index_ppm: int
    actual_supply_cost_index_ppm: int
    average_paid_price_cents: int
    market_model_id: str = "balanced"
    market_model_label: str = "均衡市场"
    market_model_description: str = "价格、品牌和服务共同影响消费者选择。"
    demand_bias_ppm: int = 1_000_000
    price_anchor_cents: int = 10_400
    price_band_cents: int = 1_200
    utility_price_multiplier_ppm: int = 1_000_000
    utility_awareness_multiplier_ppm: int = 1_000_000
    utility_service_multiplier_ppm: int = 1_000_000
    utility_reputation_multiplier_ppm: int = 1_000_000
    utility_prior_stockout_multiplier_ppm: int = 1_000_000

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class SharedResilienceState:
    """Public-good stock created by settled company contributions."""

    protocol_version: str = "shared-resilience-market-v1.0.0"
    industry_resilience_ppm: int = 0
    last_total_contribution_cents: int = 0
    last_contribution_by_company_cents: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "industry_resilience_ppm": self.industry_resilience_ppm,
            "last_total_contribution_cents": (
                self.last_total_contribution_cents
            ),
            "last_contribution_by_company_cents": dict(
                self.last_contribution_by_company_cents
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SharedResilienceState":
        by_company = data.get("last_contribution_by_company_cents", {})
        return cls(
            protocol_version=str(
                data.get(
                    "protocol_version", "shared-resilience-market-v1.0.0"
                )
            ),
            industry_resilience_ppm=int(
                data.get("industry_resilience_ppm", 0)
            ),
            last_total_contribution_cents=int(
                data.get("last_total_contribution_cents", 0)
            ),
            last_contribution_by_company_cents=tuple(
                sorted((str(key), int(value)) for key, value in by_company.items())
            ),
        )


@dataclass(frozen=True, slots=True)
class CompanyLifecycleState:
    """Public v6 operating status; exit is absorbing within an episode."""

    company_id: str
    status: CompanyOperatingStatus = CompanyOperatingStatus.OPERATING
    distress_streak: int = 0
    first_distress_round: int | None = None
    exit_round: int | None = None
    exit_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_id": self.company_id,
            "status": self.status.value,
            "distress_streak": self.distress_streak,
            "first_distress_round": self.first_distress_round,
            "exit_round": self.exit_round,
            "exit_reason": self.exit_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompanyLifecycleState":
        return cls(
            company_id=str(data["company_id"]),
            status=CompanyOperatingStatus(data.get("status", "operating")),
            distress_streak=int(data.get("distress_streak", 0)),
            first_distress_round=(
                int(data["first_distress_round"])
                if data.get("first_distress_round") is not None
                else None
            ),
            exit_round=(
                int(data["exit_round"])
                if data.get("exit_round") is not None
                else None
            ),
            exit_reason=(
                str(data["exit_reason"])
                if data.get("exit_reason") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ThresholdCooperationProjectState:
    """A provision-point public project with an auditable deadline."""

    project_id: str
    status: ThresholdProjectStatus
    required_total_contribution_cents: int
    accumulated_total_contribution_cents: int
    contribution_by_company_cents: tuple[tuple[str, int], ...]
    deadline_round: int
    success_round: int | None = None
    failure_round: int | None = None
    failure_refund_rate_ppm: int = 0
    public_protection_bonus_ppm: int = 0
    supply_cost_reduction_ppm: int = 0
    last_contribution_by_company_cents: tuple[tuple[str, int], ...] = ()
    last_refund_by_company_cents: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "status": self.status.value,
            "required_total_contribution_cents": (
                self.required_total_contribution_cents
            ),
            "accumulated_total_contribution_cents": (
                self.accumulated_total_contribution_cents
            ),
            "contribution_by_company_cents": dict(
                self.contribution_by_company_cents
            ),
            "deadline_round": self.deadline_round,
            "success_round": self.success_round,
            "failure_round": self.failure_round,
            "failure_refund_rate_ppm": self.failure_refund_rate_ppm,
            "public_protection_bonus_ppm": self.public_protection_bonus_ppm,
            "supply_cost_reduction_ppm": self.supply_cost_reduction_ppm,
            "last_contribution_by_company_cents": dict(
                self.last_contribution_by_company_cents
            ),
            "last_refund_by_company_cents": dict(
                self.last_refund_by_company_cents
            ),
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any]
    ) -> "ThresholdCooperationProjectState":
        return cls(
            project_id=str(data["project_id"]),
            status=ThresholdProjectStatus(data["status"]),
            required_total_contribution_cents=int(
                data["required_total_contribution_cents"]
            ),
            accumulated_total_contribution_cents=int(
                data["accumulated_total_contribution_cents"]
            ),
            contribution_by_company_cents=tuple(
                sorted(
                    (str(key), int(value))
                    for key, value in data.get(
                        "contribution_by_company_cents", {}
                    ).items()
                )
            ),
            deadline_round=int(data["deadline_round"]),
            success_round=(
                int(data["success_round"])
                if data.get("success_round") is not None
                else None
            ),
            failure_round=(
                int(data["failure_round"])
                if data.get("failure_round") is not None
                else None
            ),
            failure_refund_rate_ppm=int(data["failure_refund_rate_ppm"]),
            public_protection_bonus_ppm=int(
                data.get("public_protection_bonus_ppm", 0)
            ),
            supply_cost_reduction_ppm=int(
                data.get("supply_cost_reduction_ppm", 0)
            ),
            last_contribution_by_company_cents=tuple(
                sorted(
                    (str(key), int(value))
                    for key, value in data.get(
                        "last_contribution_by_company_cents", {}
                    ).items()
                )
            ),
            last_refund_by_company_cents=tuple(
                sorted(
                    (str(key), int(value))
                    for key, value in data.get(
                        "last_refund_by_company_cents", {}
                    ).items()
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class MutualAidTransfer:
    donor_company_id: str
    recipient_company_id: str
    fulfilled_orders: int
    fee_per_order_cents: int
    total_transfer_fee_cents: int

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MutualAidTransfer":
        return cls(
            donor_company_id=str(data["donor_company_id"]),
            recipient_company_id=str(data["recipient_company_id"]),
            fulfilled_orders=int(data["fulfilled_orders"]),
            fee_per_order_cents=int(data["fee_per_order_cents"]),
            total_transfer_fee_cents=int(data["total_transfer_fee_cents"]),
        )


@dataclass(frozen=True, slots=True)
class PriceCoordinationOutcome:
    company_a_id: str
    company_b_id: str
    target_price_cents: int
    company_a_actual_price_cents: int
    company_b_actual_price_cents: int
    company_a_adhered: bool
    company_b_adhered: bool
    status: PriceCoordinationStatus
    detection_probability_ppm: int
    detected: bool
    fine_by_company_cents: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_a_id": self.company_a_id,
            "company_b_id": self.company_b_id,
            "target_price_cents": self.target_price_cents,
            "company_a_actual_price_cents": self.company_a_actual_price_cents,
            "company_b_actual_price_cents": self.company_b_actual_price_cents,
            "company_a_adhered": self.company_a_adhered,
            "company_b_adhered": self.company_b_adhered,
            "status": self.status.value,
            "detection_probability_ppm": self.detection_probability_ppm,
            "detected": self.detected,
            "fine_by_company_cents": dict(self.fine_by_company_cents),
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any]
    ) -> "PriceCoordinationOutcome":
        return cls(
            company_a_id=str(data["company_a_id"]),
            company_b_id=str(data["company_b_id"]),
            target_price_cents=int(data["target_price_cents"]),
            company_a_actual_price_cents=int(
                data["company_a_actual_price_cents"]
            ),
            company_b_actual_price_cents=int(
                data["company_b_actual_price_cents"]
            ),
            company_a_adhered=bool(data["company_a_adhered"]),
            company_b_adhered=bool(data["company_b_adhered"]),
            status=PriceCoordinationStatus(data["status"]),
            detection_probability_ppm=int(
                data["detection_probability_ppm"]
            ),
            detected=bool(data["detected"]),
            fine_by_company_cents=tuple(
                sorted(
                    (str(key), int(value))
                    for key, value in data.get(
                        "fine_by_company_cents", {}
                    ).items()
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class StrategicMarketState:
    """Auditable v6 market structure, lifecycle and welfare state."""

    protocol_version: str
    company_lifecycle: tuple[CompanyLifecycleState, ...]
    active_company_count: int
    hhi_ppm: int
    concentration_regime: ConcentrationRegime
    dominant_company_id: str | None
    dominant_share_ppm: int
    predatory_pricing_company_ids: tuple[str, ...] = ()
    price_war_intensity_ppm: int = 0
    market_power_markup_ppm: int = 0
    regulatory_pressure_ppm: int = 0
    consumer_surplus_proxy_cents: int = 0
    producer_welfare_cents: int = 0
    social_welfare_proxy_cents: int = 0
    cumulative_consumer_surplus_proxy_cents: int = 0
    cumulative_producer_welfare_cents: int = 0
    cumulative_social_welfare_proxy_cents: int = 0
    threshold_project: ThresholdCooperationProjectState | None = None
    mutual_aid_enabled: bool = False
    last_mutual_aid_transfers: tuple[MutualAidTransfer, ...] = ()
    price_coordination_enabled: bool = False
    coordination_credibility_by_company_ppm: tuple[tuple[str, int], ...] = ()
    last_price_coordination_outcomes: tuple[
        PriceCoordinationOutcome, ...
    ] = ()

    @property
    def active_company_ids(self) -> tuple[str, ...]:
        return tuple(
            item.company_id
            for item in self.company_lifecycle
            if item.status is not CompanyOperatingStatus.EXITED
        )

    def lifecycle(self, company_id: str) -> CompanyLifecycleState:
        for item in self.company_lifecycle:
            if item.company_id == company_id:
                return item
        raise KeyError(company_id)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "protocol_version": self.protocol_version,
            "company_lifecycle": [
                item.to_dict() for item in self.company_lifecycle
            ],
            "active_company_count": self.active_company_count,
            "hhi_ppm": self.hhi_ppm,
            "concentration_regime": self.concentration_regime.value,
            "dominant_company_id": self.dominant_company_id,
            "dominant_share_ppm": self.dominant_share_ppm,
            "predatory_pricing_company_ids": list(
                self.predatory_pricing_company_ids
            ),
            "price_war_intensity_ppm": self.price_war_intensity_ppm,
            "market_power_markup_ppm": self.market_power_markup_ppm,
            "regulatory_pressure_ppm": self.regulatory_pressure_ppm,
            "consumer_surplus_proxy_cents": self.consumer_surplus_proxy_cents,
            "producer_welfare_cents": self.producer_welfare_cents,
            "social_welfare_proxy_cents": self.social_welfare_proxy_cents,
            "cumulative_consumer_surplus_proxy_cents": (
                self.cumulative_consumer_surplus_proxy_cents
            ),
            "cumulative_producer_welfare_cents": (
                self.cumulative_producer_welfare_cents
            ),
            "cumulative_social_welfare_proxy_cents": (
                self.cumulative_social_welfare_proxy_cents
            ),
        }
        if self.threshold_project is not None:
            payload["threshold_project"] = self.threshold_project.to_dict()
        if self.mutual_aid_enabled:
            payload["mutual_aid"] = {
                "enabled": True,
                "last_transfers": [
                    item.to_dict() for item in self.last_mutual_aid_transfers
                ],
            }
        if self.price_coordination_enabled:
            payload["price_coordination"] = {
                "enabled": True,
                "credibility_by_company_ppm": dict(
                    self.coordination_credibility_by_company_ppm
                ),
                "last_outcomes": [
                    item.to_dict()
                    for item in self.last_price_coordination_outcomes
                ],
            }
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StrategicMarketState":
        lifecycle = data.get("company_lifecycle", ())
        lifecycle_rows = (
            [lifecycle[key] for key in sorted(lifecycle)]
            if isinstance(lifecycle, Mapping)
            else list(lifecycle)
        )
        return cls(
            protocol_version=str(data["protocol_version"]),
            company_lifecycle=tuple(
                CompanyLifecycleState.from_dict(item)
                for item in lifecycle_rows
            ),
            active_company_count=int(data["active_company_count"]),
            hhi_ppm=int(data["hhi_ppm"]),
            concentration_regime=ConcentrationRegime(
                data["concentration_regime"]
            ),
            dominant_company_id=(
                str(data["dominant_company_id"])
                if data.get("dominant_company_id") is not None
                else None
            ),
            dominant_share_ppm=int(data["dominant_share_ppm"]),
            predatory_pricing_company_ids=tuple(
                str(value)
                for value in data.get("predatory_pricing_company_ids", ())
            ),
            price_war_intensity_ppm=int(data.get("price_war_intensity_ppm", 0)),
            market_power_markup_ppm=int(
                data.get("market_power_markup_ppm", 0)
            ),
            regulatory_pressure_ppm=int(
                data.get("regulatory_pressure_ppm", 0)
            ),
            consumer_surplus_proxy_cents=int(
                data.get("consumer_surplus_proxy_cents", 0)
            ),
            producer_welfare_cents=int(data.get("producer_welfare_cents", 0)),
            social_welfare_proxy_cents=int(
                data.get("social_welfare_proxy_cents", 0)
            ),
            cumulative_consumer_surplus_proxy_cents=int(
                data.get("cumulative_consumer_surplus_proxy_cents", 0)
            ),
            cumulative_producer_welfare_cents=int(
                data.get("cumulative_producer_welfare_cents", 0)
            ),
            cumulative_social_welfare_proxy_cents=int(
                data.get("cumulative_social_welfare_proxy_cents", 0)
            ),
            threshold_project=(
                ThresholdCooperationProjectState.from_dict(
                    data["threshold_project"]
                )
                if data.get("threshold_project") is not None
                else None
            ),
            mutual_aid_enabled=bool(
                data.get("mutual_aid", {}).get("enabled", False)
            ),
            last_mutual_aid_transfers=tuple(
                MutualAidTransfer.from_dict(item)
                for item in data.get("mutual_aid", {}).get(
                    "last_transfers", ()
                )
            ),
            price_coordination_enabled=bool(
                data.get("price_coordination", {}).get("enabled", False)
            ),
            coordination_credibility_by_company_ppm=tuple(
                sorted(
                    (str(key), int(value))
                    for key, value in data.get(
                        "price_coordination", {}
                    ).get("credibility_by_company_ppm", {})
                    .items()
                )
            ),
            last_price_coordination_outcomes=tuple(
                PriceCoordinationOutcome.from_dict(item)
                for item in data.get("price_coordination", {}).get(
                    "last_outcomes", ()
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class GovernmentDecision:
    observed_round: int
    applies_round: int
    opening_cash_cents: int
    observed_stockout_orders: int
    observed_demand_orders: int
    observed_hhi_ppm: int
    active_company_ids: tuple[str, ...]
    inspection_cases: int
    inspection_cost_cents: int
    detection_boost_ppm: int
    support_by_company_cents: tuple[tuple[str, int], ...]
    reason: str
    policy_version: str = "budgeted-market-steward-v1.0.0"
    strategic_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**{key: getattr(self,key) for key in self.__dataclass_fields__ if getattr(self,key) is not None},
                "active_company_ids": list(self.active_company_ids), "support_by_company_cents": dict(self.support_by_company_cents)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GovernmentDecision":
        return cls(**{**data,"active_company_ids":tuple(data["active_company_ids"]),"support_by_company_cents":tuple(sorted(data["support_by_company_cents"].items()))})


@dataclass(frozen=True, slots=True)
class GovernmentState:
    cash_cents: int
    cumulative_net_cents: int = 0
    round_fines_cents: int = 0
    last_decision: GovernmentDecision | None = None
    policy_memory: str | None = None
    round_consumer_rebate_cents: int | None = None
    round_matched_support_cents: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"cash_cents":self.cash_cents,"cumulative_net_cents":self.cumulative_net_cents,
                "round_fines_cents":self.round_fines_cents,"last_decision":self.last_decision.to_dict() if self.last_decision else None,
                **{key:getattr(self,key) for key in ("policy_memory","round_consumer_rebate_cents","round_matched_support_cents") if getattr(self,key) is not None}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GovernmentState":
        return cls(**{**data,"last_decision":GovernmentDecision.from_dict(data["last_decision"]) if data.get("last_decision") else None})


@dataclass(frozen=True, slots=True)
class ConsumerDecisionAudit:
    group_id: str
    settled_round: int
    demand_orders: int
    unit_budget_cents: int
    voluntary_no_purchase_orders: int
    stockout_orders: int
    purchases_by_company: tuple[tuple[str,int], ...]
    posted_prices_cents: tuple[tuple[str,int], ...]
    spending_cents: int
    refund_cents: int = 0
    policy_version: str = "budget-constrained-logit-v1.0.0"
    government_rebate_cents: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**{key:getattr(self,key) for key in self.__dataclass_fields__ if getattr(self,key) is not None},
                "purchases_by_company":dict(self.purchases_by_company),"posted_prices_cents":dict(self.posted_prices_cents)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConsumerDecisionAudit":
        return cls(**{**data,"purchases_by_company":tuple(sorted(data["purchases_by_company"].items())),
                      "posted_prices_cents":tuple(sorted(data["posted_prices_cents"].items()))})


@dataclass(frozen=True, slots=True)
class MarketState:
    episode_id: str
    episode_seed: int
    round: int
    rounds_remaining: int
    state_version: int
    terminal: bool
    max_rounds: int
    market: MarketSnapshot
    consumer_segments: tuple[tuple[str, int], ...]
    risk_signals: tuple[RiskSignal, ...]
    active_market_events: tuple[MarketEvent, ...]
    companies: tuple[CompanyState, ...]
    shared_resilience: SharedResilienceState | None = None
    strategic_market: StrategicMarketState | None = None
    supply_chain: SupplyChainState | None = None
    welfare_accounting: WelfareAccountingState | None = None
    government: GovernmentState | None = None
    consumer_decisions: tuple[ConsumerDecisionAudit, ...] = ()
    last_joint_action: tuple[CompanyAction, ...] = ()
    terminal_enterprise_values_cents: tuple[tuple[str, int], ...] = ()
    state_hash: str = ""

    @property
    def company_ids(self) -> tuple[str, ...]:
        return tuple(company.company_id for company in self.companies)

    def company(self, company_id: str) -> CompanyState:
        for company in self.companies:
            if company.company_id == company_id:
                return company
        raise KeyError(company_id)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "round": self.round,
            "rounds_remaining": self.rounds_remaining,
            "state_version": self.state_version,
            "terminal": self.terminal,
            "max_rounds": self.max_rounds,
            "market": self.market.to_dict(),
            "consumer_segments": dict(self.consumer_segments),
            "risk_signals": [item.to_dict() for item in self.risk_signals],
            "active_market_events": [
                item.to_dict() for item in self.active_market_events
            ],
            "company_order": list(self.company_ids),
            "companies": {
                company.company_id: company.to_dict() for company in self.companies
            },
            "last_joint_action": {
                action.agent_id: action.to_dict() for action in self.last_joint_action
            },
            "terminal_enterprise_values_cents": dict(
                self.terminal_enterprise_values_cents
            ),
            "state_hash": self.state_hash,
        }
        if self.shared_resilience is not None:
            payload["shared_resilience"] = self.shared_resilience.to_dict()
        if self.strategic_market is not None:
            payload["strategic_market"] = self.strategic_market.to_dict()
        if self.supply_chain is not None:
            payload["supply_chain"] = self.supply_chain.to_dict()
        if self.welfare_accounting is not None:
            payload["welfare_accounting"] = self.welfare_accounting.to_dict()
        if self.government is not None:
            payload["government"] = self.government.to_dict()
            payload["consumer_decisions"] = [d.to_dict() for d in self.consumer_decisions]
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketState":
        companies_data = data["companies"]
        order = tuple(data.get("company_order", companies_data.keys()))
        actions = data.get("last_joint_action", {})
        segments = data.get("consumer_segments", {})
        terminal_values = data.get("terminal_enterprise_values_cents", {})
        return cls(
            government=GovernmentState.from_dict(data["government"]) if data.get("government") else None,
            consumer_decisions=tuple(ConsumerDecisionAudit.from_dict(d) for d in data.get("consumer_decisions", ())),
            episode_id=str(data["episode_id"]),
            episode_seed=int(data["episode_seed"]),
            round=int(data["round"]),
            rounds_remaining=int(data["rounds_remaining"]),
            state_version=int(data["state_version"]),
            terminal=bool(data["terminal"]),
            max_rounds=int(
                data.get(
                    "max_rounds",
                    int(data["state_version"]) + int(data["rounds_remaining"]),
                )
            ),
            market=MarketSnapshot(**data["market"]),
            consumer_segments=tuple(
                sorted((str(k), int(v)) for k, v in segments.items())
            ),
            risk_signals=tuple(
                RiskSignal.from_dict(item) for item in data.get("risk_signals", ())
            ),
            active_market_events=tuple(
                MarketEvent.from_dict(item)
                for item in data.get("active_market_events", ())
            ),
            companies=tuple(
                CompanyState.from_dict(companies_data[key]) for key in order
            ),
            shared_resilience=(
                SharedResilienceState.from_dict(data["shared_resilience"])
                if data.get("shared_resilience") is not None
                else None
            ),
            strategic_market=(
                StrategicMarketState.from_dict(data["strategic_market"])
                if data.get("strategic_market") is not None
                else None
            ),
            supply_chain=(
                SupplyChainState.from_dict(data["supply_chain"])
                if data.get("supply_chain") is not None
                else None
            ),
            welfare_accounting=(
                WelfareAccountingState.from_dict(data["welfare_accounting"])
                if data.get("welfare_accounting") is not None
                else None
            ),
            last_joint_action=tuple(
                CompanyAction.from_dict(actions[key]) for key in order if key in actions
            ),
            terminal_enterprise_values_cents=tuple(
                sorted((str(k), int(v)) for k, v in terminal_values.items())
            ),
            state_hash=str(data.get("state_hash", "")),
        )


@dataclass(frozen=True, slots=True)
class StepResult:
    step_id: str
    settled_round: int
    state_before_hash: str
    state_after: MarketState
    joint_action_hash: str
    random_draw_summary: tuple[tuple[str, int], ...]
    invariant_results: tuple[str, ...]
    actor_choices: tuple[tuple[str,str], ...] = ()

    @property
    def done(self) -> bool:
        return self.state_after.terminal

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            **({"actor_choices":dict(self.actor_choices)} if self.actor_choices else {}),
            "settled_round": self.settled_round,
            "state_before_hash": self.state_before_hash,
            "state_after_hash": self.state_after.state_hash,
            "state_after": self.state_after.to_dict(),
            "joint_action_hash": self.joint_action_hash,
            "random_draw_summary": dict(self.random_draw_summary),
            "invariant_results": list(self.invariant_results),
            "done": self.done,
        }
