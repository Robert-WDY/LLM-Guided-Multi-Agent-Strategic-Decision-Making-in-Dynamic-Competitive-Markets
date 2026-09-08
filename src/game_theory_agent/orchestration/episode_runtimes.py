"""Build and reuse AgentRuntime seats for a hosted RoundCoordinator."""

from __future__ import annotations
from game_theory_agent.local_budget import protected as local_budget_protected

import os
from typing import Any

from game_theory_agent.agents import AgentRuntime, load_persona_registry
from game_theory_agent.agents.personas import PersonaRegistry, PersonaUtilityTracker
from game_theory_agent.agents.contracts import AgentRequestedAction, IncidentIntent
from game_theory_agent.model_clients import (
    BudgetedModelClient,
    DeepSeekModelClient,
    DoubaoModelClient,
    MockModelClient,
)
from game_theory_agent.strategic_reliability import RealModelCostGuard
from game_theory_agent.model_clients.fixed_action import FixedActionModelClient
from game_theory_agent.orchestration.coordinator import RoundCoordinator, StaleRoundError
from game_theory_agent.agents.counterfactual import CounterfactualEvaluator
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
    values = {name:payload[name] for name in AgentRequestedAction.model_fields if name in payload}
    values.setdefault("price_cents", 10_000)
    values["incident_response"] = IncidentIntent.model_validate(payload.get("incident_response") or {})
    values.setdefault("strategy_summary", "human-submitted action")
    return AgentRequestedAction.model_validate(values)


def _persona_profile(registry: Any, raw_name: object) -> Any:
    if isinstance(raw_name, dict):
        raw_name = raw_name.get("persona_id")
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


def real_model_seats(session: Any) -> dict[str, dict[str, Any]]:
    configs = dict(session.manifest.agent_configs)
    return {
        company_id: dict(configs.get(company_id) or {})
        for company_id in session.env.get_state().company_ids
        if client_kind(dict(configs.get(company_id) or {})) in {"doubao", "deepseek"}
    }


def _model_client(
    kind: str,
    config: dict[str, Any],
    cost_guard: RealModelCostGuard | None,
    *, local_budget: bool = False,
) -> Any:
    if kind == "mock":
        return MockModelClient()
    if kind not in {"doubao", "deepseek"}:
        raise ValueError(f"unsupported model provider: {kind}")
    if cost_guard is None:
        raise ValueError("real-model runtime requires an authorized cost guard")
    model = str(config.get("model") or "").strip()
    if local_budget:
        from game_theory_agent.local_budget import LocalBudgetError, status
        valid_deepseek=kind=="deepseek" and model=="deepseek-v4-flash" and os.getenv("DEEPSEEK_BASE_URL","https://api.deepseek.com").rstrip("/") in {"https://api.deepseek.com","https://api.deepseek.com/v1"}
        valid_doubao=kind=="doubao" and model=="doubao-seed-2-0-lite-260215" and os.getenv("ARK_BASE_URL","https://ark.cn-beijing.volces.com/api/v3").rstrip("/")=="https://ark.cn-beijing.volces.com/api/v3"
        if not (valid_deepseek or valid_doubao):raise LocalBudgetError("本机仅开放已核价的官方 DeepSeek Flash 和豆包 Seed 2.0 Lite。")
        budget_status = status()
        if not budget_status['ready']:
            raise LocalBudgetError(budget_status['message'])
    if not model:
        raise ValueError(f"{kind} model id is required")
    if kind == "doubao":
        if not os.getenv("ARK_API_KEY"):
            raise ValueError("ARK_API_KEY is required; refusing silent Mock fallback")
        paid = DoubaoModelClient(model=model, max_schema_attempts=1,max_transport_retries=0)
    else:
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY is required; refusing silent Mock fallback")
        paid = DeepSeekModelClient(
            model=model, max_schema_attempts=1, max_transport_retries=0
        )
    if local_budget:
        from types import SimpleNamespace
        from game_theory_agent.local_budget import GuardedCompletions,GuardedResponses
        if kind=="doubao":paid._client=SimpleNamespace(responses=GuardedResponses(paid._client.responses))
        else:paid._client = SimpleNamespace(chat=SimpleNamespace(completions=GuardedCompletions(paid._client.chat.completions)))
        return BudgetedModelClient(paid, cost_guard=cost_guard, reserved_prompt_tokens_per_call=128_000, input_price_microunits_per_token=2 if kind=="doubao" else 3, output_price_microunits_per_token=11 if kind=="doubao" else 9)
    return BudgetedModelClient(paid, cost_guard=cost_guard)


def sync_episode_runtimes(
    session: Any,
    *,
    human_action: dict[str, Any] | None = None,
    human_communication: dict[str, Any] | None = None,
    include_human: bool = False,
    real_model_cost_guard: RealModelCostGuard | None = None,
) -> dict[str, AgentRuntime]:
    registry = PersonaRegistry.from_market_config(session.env.config)
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
            client.set_communication(human_communication)
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
        if kind == "mock" and (
            prior is not None
            and not isinstance(prior.model_client, FixedActionModelClient)
        ):
            runtimes[company_id] = prior
            continue
        runtimes[company_id] = AgentRuntime(
            str(config.get("agent_id") or f"{kind}-{company_id}"),
            company_id,
            _model_client(kind, config, real_model_cost_guard, local_budget=local_budget_protected(session.env.config)),
            memory=(prior.memory if prior is not None else None),
            persona_profile=_persona_profile(
                registry, config.get("persona_name") or config.get("persona")
            ),
            persona_registry=registry,
        )
        if prior is not None:
            runtimes[company_id]._utility_episode_id = prior._utility_episode_id
            runtimes[company_id]._utility_tracker = prior._utility_tracker

    saved_states = getattr(session, "restored_runtime_states", {})
    for company_id, runtime in runtimes.items():
        saved = saved_states.pop(company_id, None)
        if saved is not None:
            runtime.memory = saved["memory"]
            runtime._utility_episode_id = saved["utility_episode_id"]
            if saved["utility"] is not None:
                utility = saved["utility"]
                runtime._utility_tracker = PersonaUtilityTracker(registry.evaluator(registry.get(utility["profile_id"])))
                runtime._utility_tracker.discount_multiplier_ppm = utility["discount_multiplier_ppm"]
                runtime._utility_tracker.cumulative_discounted_utility_ppm = utility["cumulative_discounted_utility_ppm"]
    session.agent_runtimes = runtimes
    return runtimes


def run_hosted_coordinator(
    session: Any,
    episode_id: str,
    controller_token: str,
    *,
    max_rounds: int | None = None,
    actor_choices: dict[str,str] | None = None,
    human_action: dict[str, Any] | None = None,
    human_communication: dict[str, Any] | None = None,
    include_human: bool = False,
    real_model_cost_guard: RealModelCostGuard | None = None,
) -> tuple[Any, ...]:
    runtimes = sync_episode_runtimes(
        session,
        human_action=human_action,
        human_communication=human_communication,
        include_human=include_human,
        real_model_cost_guard=real_model_cost_guard,
    )
    coordinator = RoundCoordinator(
        LocalControllerClient(controller_token,actor_choices),
        LocalGatewayClient(),
        runtimes,
        counterfactual_evaluator=CounterfactualEvaluator(session.env.config),
    )
    try:
        return run_coroutine_sync(
            coordinator.run_episode(episode_id, max_rounds=max_rounds)
        )
    except StaleRoundError as exc:
        raise RuntimeError(str(exc)) from exc
