"""Immutable fixed-point domain models for Engineering MVP v4."""

from __future__ import annotations

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


class IncidentResponseMode(StrEnum):
    WAIT = "wait"
    PARTIAL_REPAIR = "partial_repair"
    FULL_REPAIR = "full_repair"


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
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompanyAction":
        return cls(
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

    def to_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CommercialState:
    price_cents: int
    market_share_ppm: int = 0
    potential_demand_orders: int = 0
    sales_orders: int = 0
    attempted_unfulfilled_orders: int = 0
    orders_received_from_redistribution: int = 0
    orders_lost_after_redistribution: int = 0

    def to_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class OperationsState:
    base_capacity_orders: int
    effective_capacity_orders: int
    financial_capacity_orders: int
    capacity_utilization_ppm: int
    base_unit_cost_cents: int
    actual_unit_cost_cents: int

    def to_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


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
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketState":
        companies_data = data["companies"]
        order = tuple(data.get("company_order", companies_data.keys()))
        actions = data.get("last_joint_action", {})
        segments = data.get("consumer_segments", {})
        terminal_values = data.get("terminal_enterprise_values_cents", {})
        return cls(
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

    @property
    def done(self) -> bool:
        return self.state_after.terminal

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "settled_round": self.settled_round,
            "state_before_hash": self.state_before_hash,
            "state_after_hash": self.state_after.state_hash,
            "state_after": self.state_after.to_dict(),
            "joint_action_hash": self.joint_action_hash,
            "random_draw_summary": dict(self.random_draw_summary),
            "invariant_results": list(self.invariant_results),
            "done": self.done,
        }
