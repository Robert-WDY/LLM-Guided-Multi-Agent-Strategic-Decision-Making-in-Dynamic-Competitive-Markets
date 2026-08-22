"""Bounded LangGraph for a single company's market decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .gateway import GatewaySnapshot
from .models import (
    DecisionContext,
    DecisionProposal,
    DecisionStatus,
    DecisionTrace,
    IntentDraft,
    PromptAudit,
    PromptTemplate,
    PersonaTraceManifest,
    StrategyReflection,
)
from .nodes import build_single_agent_nodes
from .provider import ProviderError, ProviderResult


class DecisionProvider(Protocol):
    def generate_decision(self, **kwargs: Any) -> ProviderResult: ...


class Gateway(Protocol):
    def load_snapshot(self, episode_id: str, company_id: str) -> GatewaySnapshot: ...

    def submit_intent(self, **kwargs: Any) -> dict[str, Any]: ...


class TraceStore(Protocol):
    def append(self, trace: DecisionTrace) -> None: ...

    def read_company_before_round(
        self,
        episode_id: str,
        company_id: str,
        round_number: int,
        limit: int = 5,
    ) -> list[DecisionTrace]: ...


class AgentState(TypedDict, total=False):
    episode_id: str
    company_id: str
    model_id: str
    persona_manifest: PersonaTraceManifest
    snapshot: GatewaySnapshot
    decision_context: DecisionContext
    strategy_reflection: StrategyReflection
    provider_result: ProviderResult
    proposal: DecisionProposal
    validation_errors: list[str]
    repair_attempts: int
    prepared_intent: IntentDraft
    intent_receipt: dict[str, Any]
    status: DecisionStatus
    error_code: str
    trace: DecisionTrace
    prompt_audit: PromptAudit
    prompt_template: PromptTemplate
    provider_attempt_usage: dict[str, int]
    provider_attempt_latency_ms: int
    provider_finish_reason: str
    provider_error_category: str
    provider_usage_available: bool


@dataclass(frozen=True, slots=True)
class GraphDependencies:
    provider: DecisionProvider
    gateway: Gateway
    trace_store: TraceStore
    history_limit: int = 2
    progress: Any = None


def normalize_evidence_paths(
    proposal: DecisionProposal, context: dict[str, Any]
) -> DecisionProposal:
    """Keep only evidence references that resolve inside the visible context."""

    normalized_candidates = []
    for candidate in proposal.candidates:
        normalized_paths: list[str] = []
        for raw_path in candidate.evidence_paths:
            if raw_path.startswith("/visible_context/"):
                parts = [
                    part.replace("~1", "/").replace("~0", "~")
                    for part in raw_path.removeprefix("/visible_context/").split("/")
                    if part
                ]
            elif raw_path.startswith("/"):
                continue
            else:
                parts = [part for part in raw_path.split(".") if part]
                if parts and parts[0] not in {"observation", "action_contract"}:
                    parts.insert(0, "observation")
            if not parts or parts[0] not in {"observation", "action_contract"}:
                continue

            current: Any = context
            resolved = True
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                    current = current[int(part)]
                else:
                    resolved = False
                    break
            normalized = ".".join(parts)
            if resolved and normalized not in normalized_paths:
                normalized_paths.append(normalized)
        normalized_candidates.append(
            candidate.model_copy(update={"evidence_paths": normalized_paths})
        )
    return proposal.model_copy(update={"candidates": normalized_candidates})


def validate_selected_action(
    proposal: DecisionProposal, action_contract: dict[str, Any]
) -> list[str]:
    action = proposal.selected_candidate.action
    constraints = action_contract.get("constraints") or {}
    bounds = constraints.get("bounds") or {}
    errors: list[str] = []
    values = action.model_dump(mode="python")
    for field in (
        "price_cents",
        "advertising_budget_cents",
        "service_budget_cents",
        "capacity_investment_cents",
        "resilience_budget_cents",
        "shared_resilience_contribution_cents",
    ):
        bound = bounds.get(field) or {}
        value = int(values[field])
        if "min" in bound and value < int(bound["min"]):
            errors.append(f"{field}_below_min")
        if "max" in bound and value > int(bound["max"]):
            errors.append(f"{field}_above_max")

    repair = action.incident_response.repair_budget_cents
    repair_bound = bounds.get("repair_budget_cents") or {}
    if "max" in repair_bound and repair > int(repair_bound["max"]):
        errors.append("repair_budget_cents_above_max")
    if not constraints.get("active_incident") and (
        action.incident_response.mode != "wait" or repair != 0
    ):
        errors.append("incident_response_without_active_incident")
    if not constraints.get("capacity_investment_enabled", True) and action.capacity_investment_cents:
        errors.append("capacity_investment_disabled")
    if not constraints.get("resilience_investment_enabled", True) and action.resilience_budget_cents:
        errors.append("resilience_investment_disabled")
    if not constraints.get("shared_resilience_contribution_enabled", False) and action.shared_resilience_contribution_cents:
        errors.append("shared_resilience_contribution_disabled")

    fixed_spend = (
        action.advertising_budget_cents
        + action.service_budget_cents
        + action.capacity_investment_cents
        + action.resilience_budget_cents
        + action.shared_resilience_contribution_cents
        + repair
    )
    cash = constraints.get("cash_available_cents")
    if cash is not None and fixed_spend > int(cash):
        errors.append("total_fixed_spend_above_cash")
    return errors


def build_single_agent_graph(deps: GraphDependencies, checkpointer: Any = None):
    graph = StateGraph(AgentState)
    nodes = build_single_agent_nodes(
        deps,
        validate_action=validate_selected_action,
        normalize_evidence_paths=normalize_evidence_paths,
    )

    graph.add_node("load_snapshot", nodes.load_snapshot)
    graph.add_node("build_context", nodes.build_context)
    graph.add_node("reflect_strategy", nodes.reflect_strategy)
    graph.add_node("generate_candidates", nodes.generate_candidates)
    graph.add_node("validate", nodes.validate)
    graph.add_node("repair_decision", nodes.repair_decision)
    graph.add_node("prepare_intent", nodes.prepare_intent)
    graph.add_node("submit_intent", nodes.submit_intent)
    graph.add_node("finalize", nodes.finalize)
    graph.add_edge(START, "load_snapshot")
    graph.add_conditional_edges(
        "load_snapshot",
        lambda state: "build_context" if state.get("status") == "running" else "finalize",
        {"build_context": "build_context", "finalize": "finalize"},
    )
    graph.add_edge("build_context", "reflect_strategy")
    graph.add_edge("reflect_strategy", "generate_candidates")
    graph.add_conditional_edges(
        "generate_candidates",
        lambda state: (
            "finalize"
            if state.get("status") == "no_intent"
            else "validate"
            if state.get("proposal")
            else "repair_decision"
        ),
        {
            "validate": "validate",
            "repair_decision": "repair_decision",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "validate",
        lambda state: (
            "prepare_intent"
            if not state.get("validation_errors")
            else "repair_decision"
            if state.get("repair_attempts", 0) < 1
            else "finalize"
        ),
        {"prepare_intent": "prepare_intent", "repair_decision": "repair_decision", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "repair_decision",
        lambda state: "finalize" if state.get("status") == "no_intent" else "validate",
        {"validate": "validate", "finalize": "finalize"},
    )
    graph.add_edge("prepare_intent", "submit_intent")
    graph.add_edge("submit_intent", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
