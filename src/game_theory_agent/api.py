"""HTTP adapter for the Engineering MVP v4 market environment."""

from __future__ import annotations

import os
import secrets
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from game_theory_agent.calibration import evaluate_presets
from game_theory_agent.decisioning import (
    POLICY_VERSION,
    ResolvedDecision,
    resolve_action_request,
)

from game_theory_agent.gameplay import (
    build_company_analysis,
    build_retrospective,
    build_rule_action,
    build_terminal_rankings,
    settled_market_snapshot,
)
from game_theory_agent.market import (
    CompanyAction,
    MarketEnv,
    Persona,
    load_market_config,
)
from game_theory_agent.market.exceptions import MarketError
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(
    os.environ.get("MARKET_CONFIG_PATH", PROJECT_ROOT / "configs" / "market_v4.yaml")
)
CONFIG = load_market_config(CONFIG_PATH)


class CreateEpisodeRequest(BaseModel):
    episode_id: str | None = None
    episode_seed: int | None = Field(default=None, ge=0, le=(1 << 64) - 1)
    company_ids: list[str] = Field(
        default_factory=lambda: [
            "company_A",
            "company_B",
            "company_C",
            "company_D",
        ]
    )
    personas: dict[str, Persona] = Field(default_factory=dict)
    game_mode: Literal["market", "single_company"] = "market"
    player_company_id: str | None = None
    market_model: Literal[
        "random",
        "balanced",
        "value_oriented",
        "quality_oriented",
        "service_oriented",
    ] = "random"
    max_rounds: Literal[5, 10, 15, 20] = 10


class StepRequest(BaseModel):
    step_id: str
    joint_action: dict[str, dict[str, Any]]


class PlayerStepRequest(BaseModel):
    step_id: str
    player_action: dict[str, Any]


class IncidentIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["wait", "partial_repair", "full_repair"] = "wait"
    repair_budget_cents: int = Field(default=0, ge=0)


class AgentRequestedAction(BaseModel):
    """Economic intent only; identity and protocol fields are controller-owned."""

    model_config = ConfigDict(extra="forbid")

    price_cents: int
    advertising_budget_cents: int = Field(default=0, ge=0)
    service_budget_cents: int = Field(default=0, ge=0)
    capacity_investment_cents: int = Field(default=0, ge=0)
    resilience_budget_cents: int = Field(default=0, ge=0)
    incident_response: IncidentIntent = Field(default_factory=IncidentIntent)
    strategy_summary: str = Field(default="", max_length=500)


class SubmitAgentIntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1, max_length=120)
    company_id: str = Field(min_length=1, max_length=120)
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    requested_action: AgentRequestedAction
    rationale: str = Field(default="", max_length=2000)
    expected_outcome: str = Field(default="", max_length=1000)


class SettleAgentRoundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    intent_ids: dict[str, str] = Field(default_factory=dict)
    fallback: Literal["rule", "error"] = "rule"


class PresetCalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_start: int = Field(default=0, ge=0, le=(1 << 63) - 1)
    seed_count: int = Field(default=200, ge=1, le=500)


@dataclass(slots=True)
class AgentIntentRecord:
    intent_id: str
    agent_id: str
    company_id: str
    round: int
    state_version: int
    requested_action: dict[str, Any]
    rationale: str
    expected_outcome: str
    submitted_at: str
    status: Literal["accepted", "executed", "rejected"] = "accepted"
    resolution: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "agent_id": self.agent_id,
            "company_id": self.company_id,
            "round": self.round,
            "state_version": self.state_version,
            "requested_action": self.requested_action,
            "rationale": self.rationale,
            "expected_outcome": self.expected_outcome,
            "submitted_at": self.submitted_at,
            "status": self.status,
            "resolution": self.resolution,
        }


@dataclass(slots=True)
class EpisodeSession:
    env: MarketEnv
    manifest: EpisodeManifest
    transitions: list[MarketTransition] = field(default_factory=list)
    game_mode: Literal["market", "single_company"] = "market"
    player_company_id: str | None = None
    agent_intents: dict[str, AgentIntentRecord] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


SESSIONS: dict[str, EpisodeSession] = {}
SESSIONS_LOCK = threading.RLock()


app = FastAPI(
    title="Fresh Market Lab API",
    version=CONFIG.environment_version,
    description="Backend-only Engineering MVP v4 market calculation source.",
)
agent_app = FastAPI(
    title="Fresh Market Agent Gateway",
    version="1.0.0",
    description=(
        "Read market observations and submit non-executable intents. "
        "This gateway never settles a round."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://[::1]:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://[::1]:3001",
        "http://localhost:3210",
        "http://127.0.0.1:3210",
        "http://[::1]:3210",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(MarketError)
async def market_error_handler(_request: Any, exc: MarketError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@agent_app.exception_handler(MarketError)
async def agent_market_error_handler(
    _request: Any, exc: MarketError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def _session(episode_id: str) -> EpisodeSession:
    with SESSIONS_LOCK:
        session = SESSIONS.get(episode_id)
    if session is None:
        raise HTTPException(status_code=404, detail="episode not found")
    return session


def _constraints(env: MarketEnv) -> dict[str, Any]:
    state = env.get_state()
    if state.terminal:
        return {}
    return {
        company_id: env.get_action_constraints(company_id, state.state_version)
        for company_id in state.company_ids
    }


def _episode_payload(session: EpisodeSession) -> dict[str, Any]:
    state = session.env.get_state()
    payload = {
        "manifest": session.manifest.to_dict(),
        "state": state.to_dict(),
        "action_constraints": _constraints(session.env),
        "action_presets": CONFIG.to_dict()["action"]["presets"],
        "game_mode": session.game_mode,
        "player_company_id": session.player_company_id,
        "market_model_options": {
            model_id: {
                "label": profile["label"],
                "description": profile["description"],
            }
            for model_id, profile in CONFIG.mapping(
                "market_models", "profiles"
            ).items()
        },
        "episode_options": _episode_options(),
    }
    if session.player_company_id:
        payload["company_analysis"] = build_company_analysis(
            state, session.player_company_id, CONFIG
        )
    return payload


def _record_transition(
    session: EpisodeSession,
    state_before: Any,
    actions: dict[str, CompanyAction],
    result: Any,
) -> None:
    if (
        not session.transitions
        or session.transitions[-1].step_result.step_id != result.step_id
    ):
        session.transitions.append(
            MarketTransition.create(state_before, actions, result)
        )


def _resolve(
    state: Any,
    company_id: str,
    requested_action: dict[str, Any],
    *,
    source: str,
    action_id: str | None = None,
) -> ResolvedDecision:
    return resolve_action_request(
        CONFIG,
        state,
        company_id,
        requested_action,
        source=source,
        action_id=action_id,
    )


def _agent_observation(session: EpisodeSession, company_id: str) -> dict[str, Any]:
    state = session.env.get_state()
    if company_id not in state.company_ids:
        raise HTTPException(status_code=404, detail="company not found")
    public_companies = [_public_company(item) for item in state.companies]
    public_history = [
        _public_transition(transition, company_id)
        for transition in session.transitions
    ]
    terminal_summary = None
    if state.terminal:
        ranking_payload = build_terminal_rankings(state, CONFIG)
        composite = ranking_payload["composite"]
        assets = ranking_payload["total_assets"]
        own_composite = next(
            item for item in composite if item["company_id"] == company_id
        )
        own_assets = next(
            item for item in assets if item["company_id"] == company_id
        )
        terminal_summary = {
            "status": "complete",
            "settled_rounds": state.max_rounds,
            "ranking": [
                {
                    "rank": item["rank"],
                    "company_id": item["company_id"],
                    "enterprise_value_cents": item["value_cents"],
                }
                for item in composite
            ],
            "rankings": ranking_payload,
            "own_rank": own_composite["rank"],
            "own_asset_rank": own_assets["rank"],
            "own_enterprise_value_cents": own_composite["value_cents"],
            "own_total_assets_cents": own_assets["value_cents"],
        }
    return {
        "observation_schema_version": "agent-observation-v1.0.0",
        "episode_id": state.episode_id,
        "episode_seed": state.episode_seed,
        "round": state.round,
        "decision_round": None if state.terminal else state.round,
        "last_settled_round": min(state.state_version, state.max_rounds),
        "rounds_remaining": state.rounds_remaining,
        "state_version": state.state_version,
        "state_hash": state.state_hash,
        "terminal": state.terminal,
        "episode_config": {
            "max_rounds": state.max_rounds,
            "episode_seed": state.episode_seed,
            "market_model_id": state.market.market_model_id,
        },
        "market": state.market.to_dict(),
        "risk_signals": [item.to_dict() for item in state.risk_signals],
        "active_market_events": [
            item.to_dict() for item in state.active_market_events
        ],
        "public_companies": public_companies,
        "public_history": public_history,
        "own_company": state.company(company_id).to_dict(),
        "company_analysis": build_company_analysis(state, company_id, CONFIG),
        "action_constraints": (
            session.env.get_action_constraints(company_id, state.state_version)
            if not state.terminal
            else {}
        ),
        "terminal_summary": terminal_summary,
    }


def _public_company(company: Any) -> dict[str, Any]:
    return {
        "company_id": company.company_id,
        "price_cents": company.commercial.price_cents,
        "market_share_ppm": company.commercial.market_share_ppm,
        "sales_orders": company.commercial.sales_orders,
        "round_revenue_cents": company.financial.round_revenue_cents,
        "reputation_ppm": company.brand.reputation_ppm,
    }


def _event_impact_summary(state: Any, company_id: str) -> dict[str, Any]:
    """Explain declared event transmission using only public event fields."""

    ppm = 1_000_000
    resilience = state.company(company_id).risk.resilience_ppm
    max_reduction = int(CONFIG.mapping("events")["resilience_max_reduction_ppm"])
    remaining_impact = ppm - round(max_reduction * resilience / ppm)
    demand = supply = capacity = advertising = ppm
    service_penalty = reputation_penalty = 0
    for event in state.active_market_events:
        demand = round(demand * event.demand_multiplier_ppm / ppm)
        protected_supply = ppm + round(
            (event.supply_cost_multiplier_ppm - ppm) * remaining_impact / ppm
        )
        protected_capacity = ppm - round(
            (ppm - event.capacity_multiplier_ppm) * remaining_impact / ppm
        )
        protected_advertising = ppm - round(
            (ppm - event.advertising_multiplier_ppm) * remaining_impact / ppm
        )
        supply = round(supply * protected_supply / ppm)
        capacity = round(capacity * protected_capacity / ppm)
        advertising = round(advertising * protected_advertising / ppm)
        service_penalty += round(event.service_penalty_ppm * remaining_impact / ppm)
        reputation_penalty += round(
            event.reputation_penalty_ppm * remaining_impact / ppm
        )
    return {
        "active_event_count": len(state.active_market_events),
        "resilience_at_round_start_ppm": resilience,
        "combined_effective_multipliers": {
            "market_demand_ppm": demand,
            "own_supply_cost_ppm": supply,
            "own_capacity_ppm": capacity,
            "own_advertising_ppm": advertising,
        },
        "combined_effective_penalties": {
            "service_quality_ppm": service_penalty,
            "reputation_ppm": reputation_penalty,
        },
        "note": "韧性按回合开始值缓冲事件；本轮新增韧性只保护后续回合。",
    }


def _public_transition(transition: MarketTransition, company_id: str) -> dict[str, Any]:
    before = transition.state_before
    after = transition.state_after
    before_company = before.company(company_id)
    after_company = after.company(company_id)
    action = dict(transition.joint_action)[company_id]
    signal_outcomes = []
    active_event_ids = {item.event_id for item in after.active_market_events}
    for signal in before.risk_signals:
        if signal.target_round != after.round:
            continue
        event_id = f"{before.episode_id}:event:{signal.signal_id}"
        signal_outcomes.append(
            {
                **signal.to_dict(),
                "outcome": "realized" if event_id in active_event_ids else "not_realized",
            }
        )
    return {
        "settled_round": transition.step_result.settled_round,
        "market": settled_market_snapshot(transition),
        "active_events_during_round": [
            item.to_dict() for item in before.active_market_events
        ],
        "event_impact_explanation": _event_impact_summary(before, company_id),
        "incident_context": {
            "active_at_round_start": (
                before_company.risk.active_incident.to_dict()
                if before_company.risk.active_incident
                else None
            ),
            "repair_response": action.incident_response.to_dict(),
            "active_after_round": (
                after_company.risk.active_incident.to_dict()
                if after_company.risk.active_incident
                else None
            ),
            "repair_timing_note": (
                "维修降低本轮事故影响；完全维修仍保留本轮残余影响，"
                "并从下一轮清除事故。"
            ),
        },
        "resolved_signal_outcomes": signal_outcomes,
        "public_companies": [_public_company(item) for item in after.companies],
        "own_action": action.to_dict(),
        "own_result": {
            "round_profit_cents": after_company.financial.round_profit_cents,
            "round_revenue_cents": after_company.financial.round_revenue_cents,
            "round_product_cost_cents": after_company.financial.round_variable_cost_cents,
            "round_operating_cost_cents": after_company.financial.round_operating_cost_cents,
            "round_decision_spend_cents": after_company.financial.round_fixed_spend_cents,
            "round_incident_cost_cents": after_company.financial.round_incident_cost_cents,
            "market_share_ppm": after_company.commercial.market_share_ppm,
            "share_delta_ppm": (
                after_company.commercial.market_share_ppm
                - before_company.commercial.market_share_ppm
            ),
            "sales_orders": after_company.commercial.sales_orders,
            "stockout_orders": after_company.commercial.attempted_unfulfilled_orders,
            "capacity_utilization_ppm": after_company.operations.capacity_utilization_ppm,
            "awareness_delta_ppm": (
                after_company.brand.brand_awareness_ppm
                - before_company.brand.brand_awareness_ppm
            ),
            "service_delta_ppm": (
                after_company.brand.service_quality_ppm
                - before_company.brand.service_quality_ppm
            ),
            "reputation_delta_ppm": (
                after_company.brand.reputation_ppm
                - before_company.brand.reputation_ppm
            ),
        },
    }


def _require_controller_token(token: str | None) -> None:
    expected = os.environ.get("MARKET_CONTROLLER_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="controller is disabled until MARKET_CONTROLLER_TOKEN is set",
        )
    if token is None or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="invalid controller token")


@agent_app.get("/health")
def agent_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "agent-gateway",
        "execution_access": False,
        "policy_version": POLICY_VERSION,
    }


@agent_app.get("/v1/capabilities")
def agent_capabilities() -> dict[str, Any]:
    return {
        "protocol_version": "agent-gateway-v1.0.0",
        "can_read_observation": True,
        "can_submit_intent": True,
        "can_execute_action": False,
        "can_settle_round": False,
        "can_discover_episode_options": True,
        "identity_fields_controller_owned": [
            "action_id",
            "episode_id",
            "agent_id",
            "round",
            "state_version",
        ],
        "decision_policy_version": POLICY_VERSION,
    }


def _episode_options() -> dict[str, Any]:
    return {
        "round_options": list(CONFIG.get("episode_options", "round_options")),
        "default_rounds": CONFIG.integer("episode_options", "default_rounds"),
        "seed": {
            "min": CONFIG.integer("episode_options", "seed_min"),
            "max": CONFIG.integer("episode_options", "seed_max"),
            "random_supported": True,
            "fixed_supported": True,
            "request_semantics": "episode_seed omitted/null = random uint64; integer = fixed",
            "note": "随机 Seed 由受信任 Controller 生成；固定 Seed 用于可重放与配对评估。",
        },
        "market_models": {
            "random": {
                "label": "随机市场",
                "description": "按 Seed 选择市场类型并施加有限需求与价格锚点扰动。",
            },
            **{
                model_id: {
                    "label": profile["label"],
                    "description": profile["description"],
                }
                for model_id, profile in CONFIG.mapping(
                    "market_models", "profiles"
                ).items()
            },
        },
        "creation_boundary": {
            "agent_gateway_can_create": False,
            "controller_endpoint": "POST /api/episodes on private port 8010",
        },
    }


@agent_app.get("/v1/episode-options")
def get_agent_episode_options() -> dict[str, Any]:
    """Discover Controller-owned Episode choices without creating state."""

    return _episode_options()


@agent_app.get(
    "/v1/episodes/{episode_id}/companies/{company_id}/observation"
)
def get_agent_observation(episode_id: str, company_id: str) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        return _agent_observation(session, company_id)


@agent_app.get(
    "/v1/episodes/{episode_id}/companies/{company_id}/action-contract"
)
def get_agent_action_contract(episode_id: str, company_id: str) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        state = session.env.get_state()
        if state.terminal:
            raise HTTPException(status_code=409, detail="episode is terminal")
        if company_id not in state.company_ids:
            raise HTTPException(status_code=404, detail="company not found")
        return {
            "protocol_version": "agent-intent-v1.0.0",
            "decision_policy_version": POLICY_VERSION,
            "episode_id": episode_id,
            "company_id": company_id,
            "round": state.round,
            "state_version": state.state_version,
            "allowed_intent_fields": list(AgentRequestedAction.model_fields),
            "controller_owned_fields": [
                "action_id",
                "episode_id",
                "agent_id",
                "round",
                "state_version",
            ],
            "constraints": session.env.get_action_constraints(
                company_id, state.state_version
            ),
            "execution_note": (
                "POST creates an accepted intent only. The protected controller "
                "resolves guardrails and settles the joint action separately."
            ),
        }


@agent_app.post("/v1/episodes/{episode_id}/intents", status_code=202)
def submit_agent_intent(
    episode_id: str, request: SubmitAgentIntentRequest
) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        state = session.env.get_state()
        if state.terminal:
            raise HTTPException(status_code=409, detail="episode is terminal")
        if request.company_id not in state.company_ids:
            raise HTTPException(status_code=404, detail="company not found")
        if request.round != state.round or request.state_version != state.state_version:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "STALE_OBSERVATION",
                    "expected_round": state.round,
                    "expected_state_version": state.state_version,
                },
            )
        requested_action = request.requested_action.model_dump()
        intent_id = f"intent-{uuid.uuid4()}"
        preview = _resolve(
            state,
            request.company_id,
            requested_action,
            source=f"agent-intent:{request.agent_id}",
            action_id=f"preview:{intent_id}",
        )
        record = AgentIntentRecord(
            intent_id=intent_id,
            agent_id=request.agent_id,
            company_id=request.company_id,
            round=request.round,
            state_version=request.state_version,
            requested_action=requested_action,
            rationale=request.rationale,
            expected_outcome=request.expected_outcome,
            submitted_at=datetime.now(UTC).isoformat(),
            resolution=preview.to_dict(),
        )
        session.agent_intents[intent_id] = record
        return {
            **record.to_dict(),
            "executed": False,
            "message": "意图已接收；市场状态未改变。",
        }


@agent_app.get("/v1/episodes/{episode_id}/intents/{intent_id}")
def get_agent_intent(episode_id: str, intent_id: str) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        record = session.agent_intents.get(intent_id)
        if record is None:
            raise HTTPException(status_code=404, detail="intent not found")
        return record.to_dict()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "environment_version": CONFIG.environment_version,
        "config_id": CONFIG.config_id,
        "config_version": CONFIG.config_version,
        "config_sha256": CONFIG.config_sha256,
        "decision_policy_version": POLICY_VERSION,
    }


@app.post("/api/v1/controller/evaluations/presets")
def evaluate_preset_endpoint(
    request: PresetCalibrationRequest,
    controller_token: str | None = Header(
        default=None, alias="X-Controller-Token"
    ),
) -> dict[str, Any]:
    _require_controller_token(controller_token)
    return evaluate_presets(
        CONFIG, seed_start=request.seed_start, seed_count=request.seed_count
    )


@app.post("/api/episodes", status_code=201)
def create_episode(request: CreateEpisodeRequest) -> dict[str, Any]:
    episode_id = request.episode_id or f"episode-{uuid.uuid4()}"
    episode_seed = (
        secrets.randbits(64)
        if request.episode_seed is None
        else request.episode_seed
    )
    player_company_id = request.player_company_id
    if request.game_mode == "single_company":
        player_company_id = player_company_id or request.company_ids[0]
        if player_company_id not in request.company_ids:
            raise HTTPException(
                status_code=422, detail="player_company_id must be in company_ids"
            )
    else:
        player_company_id = None
    env = MarketEnv(CONFIG)
    state = env.reset(
        request.company_ids,
        episode_id=episode_id,
        episode_seed=episode_seed,
        personas=request.personas,
        market_model=request.market_model,
        max_rounds=request.max_rounds,
    )
    manifest = EpisodeManifest.create(env, state, experiment_id="frontend-live")
    session = EpisodeSession(
        env=env,
        manifest=manifest,
        game_mode=request.game_mode,
        player_company_id=player_company_id,
    )
    with SESSIONS_LOCK:
        if episode_id in SESSIONS:
            raise HTTPException(status_code=409, detail="episode_id already exists")
        SESSIONS[episode_id] = session
    return _episode_payload(session)


@app.get("/api/episodes/{episode_id}/state")
def get_state(episode_id: str) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        return _episode_payload(session)


@app.get("/api/episodes/{episode_id}/agents/{agent_id}/action-constraints")
def get_action_constraints(episode_id: str, agent_id: str) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        state = session.env.get_state()
        return session.env.get_action_constraints(agent_id, state.state_version)


@app.post("/api/episodes/{episode_id}/steps")
def step_episode(episode_id: str, request: StepRequest) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        state_before = session.env.get_state()
        if set(request.joint_action) != set(state_before.company_ids):
            raise HTTPException(
                status_code=422, detail="joint_action must contain every company"
            )
        resolutions = {
            company_id: _resolve(
                state_before,
                company_id,
                request.joint_action[company_id],
                source="control-api",
                action_id=str(
                    request.joint_action[company_id].get(
                        "action_id",
                        f"control:{request.step_id}:{company_id}",
                    )
                ),
            )
            for company_id in state_before.company_ids
        }
        actions = {
            company_id: decision.action
            for company_id, decision in resolutions.items()
        }
        result = session.env.step(request.step_id, actions)
        _record_transition(session, state_before, actions, result)
        return {
            "step_result": result.to_dict(),
            "state": result.state_after.to_dict(),
            "settled_market": settled_market_snapshot(session.transitions[-1]),
            "action_constraints": _constraints(session.env),
            "decision_resolutions": {
                company_id: decision.to_dict()
                for company_id, decision in resolutions.items()
            },
        }


@app.post("/api/episodes/{episode_id}/player-steps")
def step_player_episode(episode_id: str, request: PlayerStepRequest) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        if session.game_mode != "single_company" or not session.player_company_id:
            raise HTTPException(
                status_code=409, detail="episode is not in single_company mode"
            )
        state_before = session.env.get_state()
        requested_agent = request.player_action.get("agent_id")
        if requested_agent != session.player_company_id:
            raise HTTPException(
                status_code=422, detail="player action agent_id does not match player"
            )
        resolutions: dict[str, ResolvedDecision] = {}
        for company_id in state_before.company_ids:
            if company_id == session.player_company_id:
                raw = request.player_action
                source = "human-player"
                action_id = str(
                    raw.get("action_id", f"player:{request.step_id}:{company_id}")
                )
            else:
                rule_action = build_rule_action(CONFIG, state_before, company_id)
                raw = rule_action.to_dict()
                source = "rule-opponent"
                action_id = rule_action.action_id
            resolutions[company_id] = _resolve(
                state_before,
                company_id,
                raw,
                source=source,
                action_id=action_id,
            )
        actions = {
            company_id: decision.action
            for company_id, decision in resolutions.items()
        }
        result = session.env.step(request.step_id, actions)
        _record_transition(session, state_before, actions, result)
        state = result.state_after
        payload: dict[str, Any] = {
            "step_result": result.to_dict(),
            "state": state.to_dict(),
            "settled_market": settled_market_snapshot(session.transitions[-1]),
            "action_constraints": _constraints(session.env),
            "company_analysis": build_company_analysis(
                state, session.player_company_id, CONFIG
            ),
            "opponent_actions": {
                company_id: action.to_dict()
                for company_id, action in actions.items()
                if company_id != session.player_company_id
            },
            "decision_resolutions": {
                company_id: decision.to_dict()
                for company_id, decision in resolutions.items()
            },
        }
        if state.terminal:
            payload["retrospective"] = build_retrospective(
                session.manifest,
                session.transitions,
                session.player_company_id,
                CONFIG,
            )
        return payload


@app.post("/api/v1/controller/episodes/{episode_id}/settle-agent-round")
def settle_agent_round(
    episode_id: str,
    request: SettleAgentRoundRequest,
    controller_token: str | None = Header(
        default=None, alias="X-Controller-Token"
    ),
) -> dict[str, Any]:
    """Protected control-plane operation; never mounted on ``agent_app``."""

    _require_controller_token(controller_token)
    session = _session(episode_id)
    with session.lock:
        state_before = session.env.get_state()
        if state_before.terminal:
            raise HTTPException(status_code=409, detail="episode is terminal")
        unknown_companies = set(request.intent_ids) - set(state_before.company_ids)
        if unknown_companies:
            raise HTTPException(
                status_code=422,
                detail=f"unknown companies: {sorted(unknown_companies)}",
            )
        resolutions: dict[str, ResolvedDecision] = {}
        used_records: list[AgentIntentRecord] = []
        for company_id in state_before.company_ids:
            intent_id = request.intent_ids.get(company_id)
            record = session.agent_intents.get(intent_id) if intent_id else None
            if record is not None:
                if record.company_id != company_id:
                    raise HTTPException(
                        status_code=422,
                        detail=f"intent {record.intent_id} belongs to another company",
                    )
                if record.status != "accepted":
                    raise HTTPException(
                        status_code=409,
                        detail=f"intent {record.intent_id} is {record.status}",
                    )
                if (
                    record.round != state_before.round
                    or record.state_version != state_before.state_version
                ):
                    record.status = "rejected"
                    raise HTTPException(
                        status_code=409,
                        detail=f"intent {record.intent_id} is stale",
                    )
                raw = record.requested_action
                source = f"agent-intent:{record.agent_id}"
                used_records.append(record)
            elif intent_id:
                raise HTTPException(status_code=404, detail=f"intent {intent_id} not found")
            elif request.fallback == "rule":
                raw = build_rule_action(CONFIG, state_before, company_id).to_dict()
                source = "controller-rule-fallback"
            else:
                raise HTTPException(
                    status_code=409, detail=f"missing intent for {company_id}"
                )
            resolutions[company_id] = _resolve(
                state_before,
                company_id,
                raw,
                source=source,
                action_id=f"controller:{request.step_id}:{company_id}",
            )

        actions = {
            company_id: decision.action
            for company_id, decision in resolutions.items()
        }
        result = session.env.step(request.step_id, actions)
        _record_transition(session, state_before, actions, result)
        for record in used_records:
            record.status = "executed"
            record.resolution = resolutions[record.company_id].to_dict()
        return {
            "step_result": result.to_dict(),
            "state": result.state_after.to_dict(),
            "action_constraints": _constraints(session.env),
            "decision_resolutions": {
                company_id: decision.to_dict()
                for company_id, decision in resolutions.items()
            },
            "executed_intent_ids": [record.intent_id for record in used_records],
        }


@app.get("/api/episodes/{episode_id}/events")
def get_events(episode_id: str) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        return {
            "episode_id": episode_id,
            "events": [transition.to_dict() for transition in session.transitions],
        }


@app.get("/api/episodes/{episode_id}/retrospective")
def get_retrospective(episode_id: str) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        if not session.player_company_id:
            raise HTTPException(
                status_code=409, detail="retrospective requires single_company mode"
            )
        return build_retrospective(
            session.manifest,
            session.transitions,
            session.player_company_id,
            CONFIG,
        )


def run() -> None:
    """Run the private engine and read/intent Agent Gateway on separate ports."""

    import uvicorn

    if os.environ.get("MARKET_AGENT_GATEWAY_ENABLED", "1") != "0":
        agent_config = uvicorn.Config(
            agent_app,
            host=os.environ.get("MARKET_AGENT_HOST", "127.0.0.1"),
            port=int(os.environ.get("MARKET_AGENT_PORT", "8011")),
            log_level="info",
        )
        agent_server = uvicorn.Server(agent_config)
        threading.Thread(
            target=agent_server.run,
            name="market-agent-gateway",
            daemon=True,
        ).start()
    uvicorn.run(
        app,
        host=os.environ.get("MARKET_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("MARKET_API_PORT", "8010")),
        reload=False,
    )


if __name__ == "__main__":
    run()
