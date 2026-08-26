"""Build and reuse AgentRuntime seats for a hosted RoundCoordinator."""

from __future__ import annotations

import os
from typing import Any

from game_theory_agent.agents import AgentRuntime, load_persona_registry
from game_theory_agent.agents.contracts import AgentRequestedAction, IncidentIntent
from game_theory_agent.model_clients import (
    DeepSeekModelClient,
    DoubaoModelClient,
    MockModelClient,
)
from game_theory_agent.model_clients.fixed_action import FixedActionModelClient
from game_theory_agent.orchestration.coordinator import RoundCoordinator, StaleRoundError
from game_theory_agent.orchestration.local import (
    LocalControllerClient,
    LocalGatewayClient,
    run_coroutine_sync,
)

_PERSONA_ALIASES = {
    "cooperator": "balanced",
    "free_rider": "profit_myopic",
    "retaliator": "aggressive",
}


def client_kind(config: dict[str, Any]) -> str:
    agent_type = str(config.get("agent_type") or "").lower()
    provider = str(config.get("provider") or "").lower()
    model = str(config.get("model") or "").lower()
    if agent_type == "human" or provider == "human":
        return "human"
    if provider in {"doubao", "deepseek", "mock"}:
        return provider
    if "doubao" in model:
        return "doubao"
    if "deepseek" in model:
        return "deepseek"
    if "mock" in model:
        return "mock"
    if agent_type == "model":
        return "mock"
    return "rule"


def episode_requires_coordinator(session: Any) -> bool:
    if session.communication_mode != "off" or session.cooperation_mode != "off":
        return True
    if session.belief_mode != "off" or session.opponent_model_mode != "off":
        return True
    if session.utility_inference_mode != "off" or session.advisor_mode != "off":
        return True
    if session.repeated_game_mode != "off":
        return True
    configs = dict(session.manifest.agent_configs)
    return any(
        client_kind(config) in {"doubao", "deepseek", "mock"}
        for config in configs.values()
    )


def requested_action_from_payload(
    payload: dict[str, Any] | None,
) -> AgentRequestedAction | None:
    if not payload:
        return None
    incident = payload.get("incident_response") or {}
    return AgentRequestedAction(
        price_cents=int(payload.get("price_cents", 10_000)),
        advertising_budget_cents=int(payload.get("advertising_budget_cents", 0)),
        service_budget_cents=int(payload.get("service_budget_cents", 0)),
        capacity_investment_cents=int(payload.get("capacity_investment_cents", 0)),
        resilience_budget_cents=int(payload.get("resilience_budget_cents", 0)),
        shared_resilience_contribution_cents=payload.get(
            "shared_resilience_contribution_cents"
        ),
        incident_response=IncidentIntent(
            mode=incident.get("mode", "wait"),
            repair_budget_cents=int(incident.get("repair_budget_cents", 0)),
        ),
        strategy_summary=str(payload.get("strategy_summary") or "human-submitted action"),
    )


def _persona_profile(registry: Any, raw_name: object) -> Any:
    name = str(raw_name or "").strip()
    if not name:
        return registry.get("none")
    try:
        return registry.get(name)
    except ValueError:
        try:
            return registry.get(_PERSONA_ALIASES.get(name, "none"))
        except ValueError:
            return registry.get("none")


def _model_client(kind: str) -> Any:
    if kind == "doubao" and os.getenv("ARK_API_KEY"):
        return DoubaoModelClient()
    if kind == "deepseek" and os.getenv("DEEPSEEK_API_KEY"):
        return DeepSeekModelClient()
    return MockModelClient()


def sync_episode_runtimes(
    session: Any,
    *,
    human_action: dict[str, Any] | None = None,
    include_human: bool = False,
) -> dict[str, AgentRuntime]:
    registry = load_persona_registry()
    configs = dict(session.manifest.agent_configs)
    existing: dict[str, AgentRuntime] = dict(getattr(session, "agent_runtimes", {}) or {})
    runtimes: dict[str, AgentRuntime] = {}
    requested = requested_action_from_payload(human_action)

    for company_id in session.env.get_state().company_ids:
        config = dict(configs.get(company_id) or {})
        kind = client_kind(config)
        if kind == "rule":
            continue
        if kind == "human" and not include_human:
            continue
        target = (human_action or {}).get("agent_id")
        if kind == "human" and target and target != company_id:
            continue
        prior = existing.get(company_id)
        if kind == "human":
            client: Any
            if isinstance(getattr(prior, "model_client", None), FixedActionModelClient):
                client = prior.model_client
            else:
                client = FixedActionModelClient()
            if requested is not None:
                client.set_requested(requested)
            runtime = prior if prior is not None and prior.model_client is client else AgentRuntime(
                str(config.get("agent_id") or f"human-{company_id}"),
                company_id,
                client,
                persona_profile=_persona_profile(
                    registry, config.get("persona_name") or config.get("persona")
                ),
                persona_registry=registry,
            )
            runtimes[company_id] = runtime
            continue
        if (
            prior is not None
            and not isinstance(prior.model_client, FixedActionModelClient)
        ):
            runtimes[company_id] = prior
            continue
        runtimes[company_id] = AgentRuntime(
            str(config.get("agent_id") or f"{kind}-{company_id}"),
            company_id,
            _model_client(kind),
            persona_profile=_persona_profile(
                registry, config.get("persona_name") or config.get("persona")
            ),
            persona_registry=registry,
        )

    session.agent_runtimes = runtimes
    return runtimes


def run_hosted_coordinator(
    session: Any,
    episode_id: str,
    controller_token: str,
    *,
    max_rounds: int | None = None,
    human_action: dict[str, Any] | None = None,
    include_human: bool = False,
) -> tuple[Any, ...]:
    runtimes = sync_episode_runtimes(
        session,
        human_action=human_action,
        include_human=include_human,
    )
    coordinator = RoundCoordinator(
        LocalControllerClient(controller_token),
        LocalGatewayClient(),
        runtimes,
    )
    try:
        return run_coroutine_sync(
            coordinator.run_episode(episode_id, max_rounds=max_rounds)
        )
    except StaleRoundError as exc:
        raise RuntimeError(str(exc)) from exc
