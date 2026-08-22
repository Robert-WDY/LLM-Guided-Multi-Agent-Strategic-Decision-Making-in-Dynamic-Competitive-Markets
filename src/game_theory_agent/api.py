"""HTTP adapter for the Engineering MVP v4 market environment."""

from __future__ import annotations

import hashlib
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

from game_theory_agent.agents.contracts import AgentRequestedAction
from game_theory_agent.agents.market_regime import MarketRegimeEvaluator
from game_theory_agent.agents.observation import (
    InformationMode,
    ObservationBuilder,
    visibility_policy_for,
)
from game_theory_agent.calibration import evaluate_presets
from game_theory_agent.belief import (
    BELIEF_SCHEMA_VERSION,
    SIGNAL_BELIEF_SCHEMA_VERSION,
    BeliefLedger,
    BeliefMode,
)
from game_theory_agent.advisor import (
    AdvisorMode,
    BayesianGameAdvisor,
    BayesianStrategyAdvisor,
)
from game_theory_agent.opponent import (
    OpponentModelLedger,
    OpponentModelMode,
)
from game_theory_agent.utility_inference import (
    OpponentUtilityInferer,
    UtilityInferenceMode,
)
from game_theory_agent.repeated_game import (
    RepeatedGameMode,
    RepeatedGameStrategist,
)
from game_theory_agent.decisioning import (
    POLICY_VERSION,
    ResolvedDecision,
    resolve_action_request,
)
from game_theory_agent.cooperation import (
    CooperativeBenefitAttribution,
    CooperationLedger,
    CooperationMode,
    CooperationProtocolError,
)
from game_theory_agent.economics import decision_support_metrics
from game_theory_agent.interaction import (
    CommunicationConflictError,
    CommunicationMode,
    CommunicationRoundLedger,
    CommunicationStateError,
    CommunicationSubmission,
    CommunicationValidationError,
)
from game_theory_agent.information import seal_observation

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
MARKET_REGIME_EVALUATOR = MarketRegimeEvaluator(CONFIG)


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
    agent_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)
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
    information_mode: Literal["perfect", "public"] = "perfect"
    observer_information_modes: dict[
        str, Literal["perfect", "public"]
    ] = Field(default_factory=dict)
    communication_mode: CommunicationMode = "off"
    cooperation_mode: CooperationMode = "off"
    belief_mode: BeliefMode = "off"
    opponent_model_mode: OpponentModelMode = "off"
    utility_inference_mode: UtilityInferenceMode = "off"
    advisor_mode: AdvisorMode = "off"
    repeated_game_mode: RepeatedGameMode = "off"


class StepRequest(BaseModel):
    step_id: str
    joint_action: dict[str, dict[str, Any]]


class PlayerStepRequest(BaseModel):
    step_id: str
    player_action: dict[str, Any]


class SubmitAgentIntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1, max_length=120)
    company_id: str = Field(min_length=1, max_length=120)
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    observation_hash: str = Field(min_length=1)
    requested_action: AgentRequestedAction
    rationale: str = Field(default="", max_length=2000)
    expected_outcome: str = Field(default="", max_length=1000)
    communication_view_digest: str | None = Field(default=None, min_length=1)


class SubmitCommunicationRequest(BaseModel):
    """Bind a model-authored draft to one frozen market state."""

    model_config = ConfigDict(extra="forbid")

    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    state_hash: str = Field(min_length=1)
    submission: CommunicationSubmission


class CloseCommunicationRequest(BaseModel):
    """Controller-owned close command for a single communication barrier."""

    model_config = ConfigDict(extra="forbid")

    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    state_hash: str = Field(min_length=1)


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
    observation_hash: str
    requested_action: dict[str, Any]
    rationale: str
    expected_outcome: str
    communication_view_digest: str | None
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
            "observation_hash": self.observation_hash,
            "requested_action": self.requested_action,
            "rationale": self.rationale,
            "expected_outcome": self.expected_outcome,
            "communication_view_digest": self.communication_view_digest,
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
    agent_settlements: dict[str, tuple[dict[str, Any], dict[str, Any]]] = field(
        default_factory=dict
    )
    communication_mode: CommunicationMode = "off"
    cooperation_mode: CooperationMode = "off"
    cooperation_ledger: CooperationLedger | None = None
    belief_mode: BeliefMode = "off"
    belief_ledger: BeliefLedger | None = None
    opponent_model_mode: OpponentModelMode = "off"
    opponent_model_ledger: OpponentModelLedger | None = None
    utility_inference_mode: UtilityInferenceMode = "off"
    advisor_mode: AdvisorMode = "off"
    repeated_game_mode: RepeatedGameMode = "off"
    agent_token_hashes: dict[str, str] = field(default_factory=dict)
    communication_ledgers: dict[
        tuple[int, int, str], CommunicationRoundLedger
    ] = field(default_factory=dict)
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


def _hash_agent_token(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def _new_agent_credentials(
    company_ids: list[str] | tuple[str, ...],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return one-time plaintext credentials plus their persistence-safe hashes."""

    raw_tokens = {
        company_id: secrets.token_urlsafe(32) for company_id in company_ids
    }
    token_hashes = {
        company_id: _hash_agent_token(token)
        for company_id, token in raw_tokens.items()
    }
    return raw_tokens, token_hashes


def _require_agent_token(
    session: EpisodeSession,
    company_id: str,
    token: str | None,
    *,
    always: bool = False,
) -> None:
    """Authenticate the company without retaining or logging the raw token."""

    if (
        not always
        and session.communication_mode == "off"
        and session.cooperation_mode == "off"
        and session.belief_mode == "off"
        and session.opponent_model_mode == "off"
        and session.utility_inference_mode == "off"
        and session.advisor_mode == "off"
        and session.repeated_game_mode == "off"
        and not session.manifest.observer_information_modes
    ):
        return
    expected_hash = session.agent_token_hashes.get(company_id)
    if token is None or expected_hash is None:
        raise HTTPException(status_code=401, detail="invalid agent token")
    supplied_hash = _hash_agent_token(token)
    if not secrets.compare_digest(supplied_hash, expected_hash):
        raise HTTPException(status_code=401, detail="invalid agent token")


def _canonical_agent_id(
    session: EpisodeSession,
    company_id: str,
    requested_agent_id: str,
) -> str:
    """Bind attribution to Controller manifest data, never to a client claim."""

    configured = dict(session.manifest.agent_configs).get(company_id, {})
    configured_agent_id = configured.get("agent_id")
    if configured_agent_id:
        canonical = str(configured_agent_id)
        if requested_agent_id != canonical:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "AGENT_IDENTITY_MISMATCH",
                    "expected_agent_id": canonical,
                },
            )
        return canonical
    return company_id


def _communication_key(
    round_number: int,
    state_version: int,
    state_hash: str,
) -> tuple[int, int, str]:
    return round_number, state_version, state_hash


def _require_current_communication_state(
    state: Any,
    *,
    round_number: int,
    state_version: int,
    state_hash: str,
) -> None:
    if state.terminal:
        raise HTTPException(status_code=409, detail="episode is terminal")
    if (
        round_number != state.round
        or state_version != state.state_version
        or state_hash != state.state_hash
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "STALE_COMMUNICATION_STATE",
                "expected_round": state.round,
                "expected_state_version": state.state_version,
                "expected_state_hash": state.state_hash,
            },
        )


def _communication_ledger(
    session: EpisodeSession,
    state: Any,
) -> CommunicationRoundLedger:
    key = _communication_key(state.round, state.state_version, state.state_hash)
    ledger = session.communication_ledgers.get(key)
    if ledger is None:
        ledger = CommunicationRoundLedger(
            episode_id=state.episode_id,
            round_number=state.round,
            state_version=state.state_version,
            state_hash=state.state_hash,
            company_ids=state.company_ids,
            mode=session.communication_mode,
        )
        session.communication_ledgers[key] = ledger
    if session.communication_mode == "off" and ledger.status == "open":
        ledger.close()
    return ledger


def _current_communication_view(
    session: EpisodeSession,
    state: Any,
    company_id: str,
) -> dict[str, Any] | None:
    if state.terminal:
        return None
    ledger = _communication_ledger(session, state)
    if ledger.status != "closed":
        return None
    return ledger.close().views[company_id].model_dump(mode="json")


def _recent_communication_history(
    session: EpisodeSession,
    state: Any,
    company_id: str,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Derive bounded company-scoped history from the authoritative ledgers."""

    if session.communication_mode == "off" or limit < 1:
        return []
    prior: list[tuple[int, int, dict[str, Any]]] = []
    for (round_number, state_version, _state_hash), ledger in (
        session.communication_ledgers.items()
    ):
        if state_version >= state.state_version or ledger.status != "closed":
            continue
        closure = ledger.close()
        view = closure.views.get(company_id)
        if view is None:
            continue
        prior.append(
            (
                state_version,
                round_number,
                view.model_dump(mode="json"),
            )
        )
    prior.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in prior[-limit:]]


def _require_decision_communication_view(
    session: EpisodeSession,
    state: Any,
    company_id: str,
    supplied_digest: str | None,
) -> None:
    if session.communication_mode == "off":
        return
    ledger = _communication_ledger(session, state)
    if ledger.status != "closed":
        raise HTTPException(
            status_code=409,
            detail={"code": "COMMUNICATION_NOT_CLOSED"},
        )
    expected_digest = ledger.close().views[company_id].view_digest
    if supplied_digest is None or not secrets.compare_digest(
        supplied_digest, expected_digest
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "COMMUNICATION_VIEW_MISMATCH",
                "expected_communication_view_digest": expected_digest,
            },
        )


def _reject_direct_step_when_interaction_enabled(
    session: EpisodeSession,
) -> None:
    """Keep legacy direct-step routes from bypassing the interaction barrier."""

    if session.communication_mode != "off" or session.cooperation_mode != "off":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "INTERACTION_REQUIRES_AGENT_BARRIER",
                "message": (
                    "direct step routes are disabled when communication is enabled; "
                    "submit authenticated intents after Communication Close and use "
                    "the protected controller settlement endpoint"
                ),
            },
        )


def _communication_phase_payload(ledger: CommunicationRoundLedger) -> dict[str, Any]:
    closure = ledger.close()
    return {
        "phase_schema_version": "communication-phase-v1.0.0",
        "mode": closure.mode,
        "status": "closed",
        "closed": True,
        "closure": closure.model_dump(mode="json"),
        "company_views": {
            company_id: {
                "company_id": company_id,
                "view_digest": view.view_digest,
                "visible_message_ids": [
                    message.message_id for message in view.visible_messages
                ],
                "own_message_ids": list(view.own_message_ids),
            }
            for company_id, view in closure.views.items()
        },
        "generation_traces": [],
    }


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
    communication_status = (
        "closed"
        if state.terminal
        else _communication_ledger(session, state).status
    )
    payload = {
        "manifest": session.manifest.to_dict(),
        "state": state.to_dict(),
        "action_constraints": _constraints(session.env),
        "action_presets": CONFIG.to_dict()["action"]["presets"],
        "game_mode": session.game_mode,
        "player_company_id": session.player_company_id,
        "communication": {
            "mode": session.communication_mode,
            "status": communication_status,
            "messages_are_non_binding": True,
        },
        "cooperation": {
            "mode": session.cooperation_mode,
            "mechanism": (
                "shared_resilience_contribution"
                if session.cooperation_mode == "shared_resilience_v1"
                else None
            ),
            "commitments_are_non_binding": True,
        },
        "belief": {
            "mode": session.belief_mode,
            "schema_version": session.manifest.belief_schema_version,
            "updater_version": session.manifest.belief_updater_version,
            "uses_hidden_state": False,
        },
        "game_theory_advisor": {
            "mode": session.advisor_mode,
            "schema_version": session.manifest.advisor_schema_version,
            "model_version": session.manifest.advisor_model_version,
            "recommendations_are_non_binding": True,
        },
        "opponent_model": {
            "mode": session.opponent_model_mode,
            "schema_version": session.manifest.opponent_model_schema_version,
            "updater_version": session.manifest.opponent_model_updater_version,
            "uses_hidden_state": False,
        },
        "utility_inference": {
            "mode": session.utility_inference_mode,
            "schema_version": session.manifest.utility_inference_schema_version,
            "model_version": session.manifest.utility_inference_model_version,
            "uses_hidden_persona": False,
        },
        "repeated_game": {
            "mode": session.repeated_game_mode,
            "schema_version": session.manifest.repeated_game_schema_version,
            "changes_market_directly": False,
        },
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
        if session.belief_ledger is not None:
            ledger = _communication_ledger(session, state_before)
            messages = (
                ledger.close().all_messages if ledger.status == "closed" else ()
            )
            session.belief_ledger.update_after_settlement(
                state_before, actions, communication_messages=messages
            )
        if session.opponent_model_ledger is not None:
            session.opponent_model_ledger.update_after_settlement(
                state_before, result.state_after, actions
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
    belief_state = None
    belief_hash = None
    communication_view = _current_communication_view(session, state, company_id)
    if session.belief_ledger is not None and session.belief_mode != "off":
        resolved_belief, belief_hash = session.belief_ledger.company_view(
            observer_company_id=company_id,
            round_number=state.round,
            state_version=state.state_version,
            visible_messages=(
                communication_view.get("visible_messages", [])
                if communication_view is not None
                else ()
            ),
            public_prices={
                item.company_id: item.commercial.price_cents
                for item in state.companies
            },
        )
        belief_state = resolved_belief.model_dump(mode="json")
    opponent_model_state = None
    opponent_model_hash = None
    if session.opponent_model_ledger is not None:
        resolved_opponent_model, opponent_model_hash = (
            session.opponent_model_ledger.company_view(
                observer_company_id=company_id,
                round_number=state.round,
                state_version=state.state_version,
            )
        )
        opponent_model_state = resolved_opponent_model.model_dump(mode="json")
    utility_inference_state = None
    utility_inference_hash = None
    if (
        session.utility_inference_mode == "strategy_utility_v1"
        and opponent_model_state is not None
    ):
        resolved_utility, utility_inference_hash = OpponentUtilityInferer().infer(
            opponent_model_state
        )
        utility_inference_state = resolved_utility.model_dump(mode="json")
    observer_information_mode = session.manifest.information_mode_for(company_id)
    company_views = ObservationBuilder().build(
        state,
        company_id,
        observer_information_mode,
        belief_state=belief_state,
        belief_hash=belief_hash,
        belief_schema_version=(
            (
                SIGNAL_BELIEF_SCHEMA_VERSION
                if session.belief_mode == "public_action_signal_v2"
                else BELIEF_SCHEMA_VERSION
            )
            if belief_state is not None
            else "none"
        ),
    )
    public_history = [
        _public_transition(
            transition, company_id, observer_information_mode
        )
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
    cooperation_view = (
        session.cooperation_ledger.company_view(
            company_id, round_number=state.round
        )
        if session.cooperation_ledger is not None
        and session.cooperation_mode != "off"
        else None
    )
    repeated_game_strategy = None
    repeated_game_strategy_hash = None
    if (
        session.repeated_game_mode == "reciprocity_v1"
        and cooperation_view is not None
    ):
        resolved_strategy, repeated_game_strategy_hash = (
            RepeatedGameStrategist().build(
                episode_id=state.episode_id,
                observer_company_id=company_id,
                round_number=state.round,
                cooperation_view=cooperation_view,
            )
        )
        repeated_game_strategy = resolved_strategy.model_dump(mode="json")
    observation = {
        "observation_schema_version": "agent-observation-v1.8.0",
        "episode_id": state.episode_id,
        "round": state.round,
        "decision_round": None if state.terminal else state.round,
        "last_settled_round": min(state.state_version, state.max_rounds),
        "rounds_remaining": state.rounds_remaining,
        "state_version": state.state_version,
        "state_hash": state.state_hash,
        "terminal": state.terminal,
        "episode_config": {
            "max_rounds": state.max_rounds,
            "market_model_id": state.market.market_model_id,
            "information_mode": observer_information_mode,
            "market_information_mode": session.manifest.information_mode,
            "company_scoped_information_treatment": (
                company_id in dict(
                    session.manifest.observer_information_modes
                )
            ),
            "communication_mode": session.communication_mode,
            "cooperation_mode": session.cooperation_mode,
            "belief_mode": session.belief_mode,
            "advisor_mode": session.advisor_mode,
            "opponent_model_mode": session.opponent_model_mode,
            "utility_inference_mode": session.utility_inference_mode,
            "repeated_game_mode": session.repeated_game_mode,
        },
        "information_mode": company_views["information_mode"],
        "visibility_policy_version": company_views[
            "visibility_policy_version"
        ],
        "visibility_policy": company_views["visibility_policy"],
        "belief_schema_version": company_views["belief_schema_version"],
        "belief_hash": company_views["belief_hash"],
        "belief_state": company_views["belief_state"],
        "opponent_model_hash": opponent_model_hash,
        "opponent_model_state": opponent_model_state,
        "utility_inference_hash": utility_inference_hash,
        "utility_inference_state": utility_inference_state,
        "public_state": company_views["public_state"],
        "private_state": company_views["private_state"],
        "communication_mode": session.communication_mode,
        "cooperation_mode": session.cooperation_mode,
        "market": company_views["market"],
        "shared_resilience": company_views["shared_resilience"],
        "market_regime": MARKET_REGIME_EVALUATOR.evaluate(
            state, information_mode=observer_information_mode
        ),
        "decision_support": decision_support_metrics(CONFIG, state, company_id),
        "risk_signals": company_views["risk_signals"],
        "active_market_events": company_views["active_market_events"],
        "public_companies": company_views["public_companies"],
        "competitors": company_views["competitors"],
        "public_history": public_history,
        "own_company": company_views["own_company"],
        "company_analysis": build_company_analysis(state, company_id, CONFIG),
        "action_constraints": (
            session.env.get_action_constraints(company_id, state.state_version)
            if not state.terminal
            else {}
        ),
        "communication_view": communication_view,
        "communication_history": _recent_communication_history(
            session, state, company_id
        ),
        "cooperation": cooperation_view,
        "repeated_game_strategy_hash": repeated_game_strategy_hash,
        "repeated_game_strategy": repeated_game_strategy,
        "terminal_summary": terminal_summary,
    }
    if (
        session.advisor_mode == "bayesian_price_v1"
        and belief_state is not None
        and not state.terminal
    ):
        observation["game_theory_advice"] = BayesianGameAdvisor().advise(
            belief_state=belief_state,
            own_company=company_views["own_company"],
            action_constraints=observation["action_constraints"],
        ).model_dump(mode="json")
    elif (
        session.advisor_mode == "bayesian_strategy_v2"
        and belief_state is not None
        and opponent_model_state is not None
        and utility_inference_state is not None
        and not state.terminal
    ):
        observation["game_theory_advice"] = BayesianStrategyAdvisor().advise(
            belief_state=belief_state,
            opponent_model=opponent_model_state,
            utility_inference=utility_inference_state,
            own_company=company_views["own_company"],
            action_constraints=observation["action_constraints"],
        ).model_dump(mode="json")
    return seal_observation(observation)


def _public_company(company: Any) -> dict[str, Any]:
    return ObservationBuilder.public_company(company)


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


def _public_transition(
    transition: MarketTransition,
    company_id: str,
    information_mode: str = "perfect",
) -> dict[str, Any]:
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
    policy = visibility_policy_for(information_mode)
    settled_market = settled_market_snapshot(transition)
    if information_mode == InformationMode.PUBLIC.value:
        settled_market = {
            field_name: settled_market[field_name]
            for field_name in policy.public_market_fields
        }
        active_events = [
            {
                field_name: item.to_dict()[field_name]
                for field_name in policy.public_event_fields
            }
            for item in before.active_market_events
        ]
        impact_explanation: dict[str, Any] = {
            "visibility": "coarse_public_outcome_only",
            "note": (
                "精确事件倍率与内部韧性传导不属于公开状态；"
                "Agent 只观察已实现的自身经营结果。"
            ),
        }
    else:
        active_events = [
            item.to_dict() for item in before.active_market_events
        ]
        impact_explanation = _event_impact_summary(before, company_id)
    return {
        "settled_round": transition.step_result.settled_round,
        "market": settled_market,
        "active_events_during_round": active_events,
        "event_impact_explanation": impact_explanation,
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
        "protocol_version": "agent-gateway-v1.2.0",
        "can_read_observation": True,
        "can_submit_intent": True,
        "can_submit_communication": True,
        "can_read_communication_view": True,
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
        "observation_schema_version": "agent-observation-v1.8.0",
        "observation_hash_protocol_version": (
            "observation-view-hash-v1.0.0"
        ),
        "belief_schema_versions": [
            BELIEF_SCHEMA_VERSION,
            SIGNAL_BELIEF_SCHEMA_VERSION,
        ],
        "belief_modes": [
            "off", "public_action_v1", "public_action_signal_v2"
        ],
        "opponent_model_modes": ["off", "public_strategy_v1"],
        "utility_inference_modes": ["off", "strategy_utility_v1"],
        "advisor_modes": [
            "off", "bayesian_price_v1", "bayesian_strategy_v2"
        ],
        "repeated_game_modes": ["off", "reciprocity_v1"],
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
        "information_modes": {
            "default": "perfect",
            "options": {
                "perfect": {
                    "description": "完全信息基线：Agent 可见对手完整状态。",
                    "visibility_policy_version": (
                        "visibility-perfect-v1.0.0"
                    ),
                },
                "public": {
                    "description": (
                        "公开信息处理：Agent 只见共同 PublicState、"
                        "自身完整 PrivateState 与对手公开结果。"
                    ),
                    "visibility_policy_version": "visibility-public-v2.0.0",
                },
            },
            "observation_hash_protocol_version": (
                "observation-view-hash-v1.0.0"
            ),
            "belief": {
                "default": "off",
                "options": {
                    "off": "不提供对手动作概率，保留 Phase A 基线。",
                    "public_action_v1": (
                        "仅从已结算公开价格历史预测对手下一轮降价、持平或涨价。"
                    ),
                    "public_action_signal_v2": (
                        "在公开动作信念上，只融合本公司可见的非绑定结构化价格声明，"
                        "并按历史声明履约率降权。"
                    ),
                },
                "uses_hidden_state": False,
                "probability_unit": "ppm",
            },
            "game_theory_advisor": {
                "default": "off",
                "options": {
                    "off": "不提供策略建议。",
                    "bayesian_price_v1": (
                        "对公开价格方向信念做近似边际化并给出非绑定报价建议。"
                    ),
                    "bayesian_strategy_v2": (
                        "联合对手策略类型、推断效用和预期回应，比较有限动作的"
                        "近似期望效用；不求解 Nash。"
                    ),
                },
                "uses_hidden_opponent_state": False,
                "recommendations_are_non_binding": True,
            },
            "opponent_model": {
                "default": "off",
                "options": {
                    "off": "不推断对手策略类型。",
                    "public_strategy_v1": (
                        "只从公开价格、销量、份额、声誉和公开贡献推断策略分布。"
                    ),
                },
                "uses_hidden_state": False,
            },
            "utility_inference": {
                "default": "off",
                "options": {
                    "off": "不推断对手效用。",
                    "strategy_utility_v1": (
                        "从可回放策略分布推断效用权重分布。"
                    ),
                },
                "uses_hidden_persona": False,
            },
        },
        "communication_modes": {
            "default": "off",
            "options": {
                "off": "无通信，保持阶段 1 决策路径。",
                "public_only": "每轮一次同步公开发言。",
                "public_private": "每轮一次同步公开发言和一对一私信。",
            },
            "messages_are_non_binding": True,
            "maximum_messages_per_agent_per_round": 2,
        },
        "cooperation_modes": {
            "default": "off",
            "options": {
                "off": "无合作动作，保持阶段 2 行为。",
                "shared_resilience_v1": (
                    "仅允许私密韧性提议、非约束承诺和真实公共韧性贡献。"
                ),
            },
            "supported_communication_modes": ["off", "public_private"],
            "repeated_game": {
                "default": "off",
                "options": {
                    "off": "不生成重复博弈策略建议。",
                    "reciprocity_v1": (
                        "从权威履约记忆生成 Tit-for-Tat、Grim Trigger 和"
                        "Generous Tit-for-Tat 非绑定建议。"
                    ),
                },
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
def get_agent_observation(
    episode_id: str,
    company_id: str,
    agent_token: str | None = Header(default=None, alias="X-Agent-Token"),
) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        _require_agent_token(session, company_id, agent_token)
        return _agent_observation(session, company_id)


@agent_app.post(
    "/v1/episodes/{episode_id}/companies/{company_id}/communication/submissions",
    status_code=202,
)
def submit_agent_communication(
    episode_id: str,
    company_id: str,
    request: SubmitCommunicationRequest,
    agent_token: str | None = Header(default=None, alias="X-Agent-Token"),
) -> dict[str, Any]:
    """Accept a non-executable draft without exposing concurrent submissions."""

    session = _session(episode_id)
    with session.lock:
        _require_agent_token(session, company_id, agent_token, always=True)
        state = session.env.get_state()
        _require_current_communication_state(
            state,
            round_number=request.round,
            state_version=request.state_version,
            state_hash=request.state_hash,
        )
        before_hash = state.state_hash
        ledger = _communication_ledger(session, state)
        try:
            if session.cooperation_ledger is not None:
                session.cooperation_ledger.validate_submission(
                    sender_company_id=company_id,
                    round_number=state.round,
                    submission=request.submission,
                )
            delivered = ledger.submit(company_id, request.submission)
        except CommunicationConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CommunicationStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CommunicationValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except CooperationProtocolError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        after_hash = session.env.get_state().state_hash
        if after_hash != before_hash:
            raise RuntimeError("communication unexpectedly changed market state")
        return {
            "communication_submission_schema_version": (
                "communication-submission-result-v1.0.0"
            ),
            "episode_id": episode_id,
            "company_id": company_id,
            "round": request.round,
            "state_version": request.state_version,
            "state_hash": request.state_hash,
            "status": ledger.status,
            "message_ids": [message.message_id for message in delivered],
            "messages": [
                message.model_dump(mode="json") for message in delivered
            ],
            "market_state_unchanged": after_hash == before_hash,
        }


@agent_app.get(
    "/v1/episodes/{episode_id}/companies/{company_id}/communication/view"
)
def get_agent_communication_view(
    episode_id: str,
    company_id: str,
    agent_token: str | None = Header(default=None, alias="X-Agent-Token"),
) -> dict[str, Any]:
    """Return only the closed, company-scoped view authorized by the token."""

    session = _session(episode_id)
    with session.lock:
        _require_agent_token(session, company_id, agent_token, always=True)
        state = session.env.get_state()
        if state.terminal:
            raise HTTPException(status_code=409, detail="episode is terminal")
        ledger = _communication_ledger(session, state)
        if ledger.status != "closed":
            raise HTTPException(
                status_code=409,
                detail={"code": "COMMUNICATION_NOT_CLOSED"},
            )
        return ledger.close().views[company_id].model_dump(mode="json")


@agent_app.get(
    "/v1/episodes/{episode_id}/companies/{company_id}/action-contract"
)
def get_agent_action_contract(
    episode_id: str,
    company_id: str,
    agent_token: str | None = Header(default=None, alias="X-Agent-Token"),
) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        _require_agent_token(session, company_id, agent_token)
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
    episode_id: str,
    request: SubmitAgentIntentRequest,
    agent_token: str | None = Header(default=None, alias="X-Agent-Token"),
) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        _require_agent_token(session, request.company_id, agent_token)
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
        _require_decision_communication_view(
            session,
            state,
            request.company_id,
            request.communication_view_digest,
        )
        expected_observation_hash = _agent_observation(
            session, request.company_id
        )["observation_hash"]
        if not secrets.compare_digest(
            request.observation_hash, expected_observation_hash
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "OBSERVATION_VIEW_MISMATCH",
                    "expected_observation_hash": expected_observation_hash,
                },
            )
        canonical_agent_id = _canonical_agent_id(
            session, request.company_id, request.agent_id
        )
        requested_action = request.requested_action.model_dump()
        intent_id = f"intent-{uuid.uuid4()}"
        preview = _resolve(
            state,
            request.company_id,
            requested_action,
            source=f"agent-intent:{canonical_agent_id}",
            action_id=f"preview:{intent_id}",
        )
        record = AgentIntentRecord(
            intent_id=intent_id,
            agent_id=canonical_agent_id,
            company_id=request.company_id,
            round=request.round,
            state_version=request.state_version,
            observation_hash=request.observation_hash,
            requested_action=requested_action,
            rationale=request.rationale,
            expected_outcome=request.expected_outcome,
            communication_view_digest=request.communication_view_digest,
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
def get_agent_intent(
    episode_id: str,
    intent_id: str,
    agent_token: str | None = Header(default=None, alias="X-Agent-Token"),
) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        record = session.agent_intents.get(intent_id)
        if record is None:
            raise HTTPException(status_code=404, detail="intent not found")
        _require_agent_token(session, record.company_id, agent_token)
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
def create_episode(
    request: CreateEpisodeRequest,
    controller_token: str | None = Header(
        default=None, alias="X-Controller-Token"
    ),
) -> dict[str, Any]:
    episode_id = request.episode_id or f"episode-{uuid.uuid4()}"
    protected_creation = (
        request.communication_mode != "off"
        or request.cooperation_mode != "off"
        or request.belief_mode != "off"
        or request.opponent_model_mode != "off"
        or request.utility_inference_mode != "off"
        or request.advisor_mode != "off"
        or request.repeated_game_mode != "off"
        or bool(request.observer_information_modes)
        or controller_token is not None
    )
    if protected_creation:
        _require_controller_token(controller_token)
    if (
        request.cooperation_mode == "shared_resilience_v1"
        and request.communication_mode == "public_only"
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "shared_resilience_v1 supports communication off or public_private"
            ),
        )
    if request.belief_mode == "public_action_signal_v2" and request.communication_mode == "off":
        raise HTTPException(
            status_code=422,
            detail="public_action_signal_v2 requires communication",
        )
    if request.advisor_mode != "off" and request.belief_mode == "off":
        raise HTTPException(
            status_code=422,
            detail="Bayesian advisor requires an enabled belief mode",
        )
    if (
        request.utility_inference_mode != "off"
        and request.opponent_model_mode == "off"
    ):
        raise HTTPException(
            status_code=422,
            detail="strategy_utility_v1 requires public_strategy_v1",
        )
    if request.advisor_mode == "bayesian_strategy_v2" and (
        request.opponent_model_mode == "off"
        or request.utility_inference_mode == "off"
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "bayesian_strategy_v2 requires opponent model and utility inference"
            ),
        )
    if request.repeated_game_mode != "off" and request.cooperation_mode == "off":
        raise HTTPException(
            status_code=422,
            detail="reciprocity_v1 requires shared_resilience_v1",
        )
    unknown_agent_configs = set(request.agent_configs) - set(request.company_ids)
    if unknown_agent_configs:
        raise HTTPException(
            status_code=422,
            detail=(
                "agent_configs contains unknown companies: "
                f"{sorted(unknown_agent_configs)}"
            ),
        )
    unknown_observer_modes = set(request.observer_information_modes) - set(
        request.company_ids
    )
    if unknown_observer_modes:
        raise HTTPException(
            status_code=422,
            detail=(
                "observer_information_modes contains unknown companies: "
                f"{sorted(unknown_observer_modes)}"
            ),
        )
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
        cooperation_mode=request.cooperation_mode,
    )
    manifest = EpisodeManifest.create(
        env,
        state,
        experiment_id="frontend-live",
        information_mode=request.information_mode,
        communication_mode=request.communication_mode,
        cooperation_mode=request.cooperation_mode,
        belief_mode=request.belief_mode,
        opponent_model_mode=request.opponent_model_mode,
        utility_inference_mode=request.utility_inference_mode,
        advisor_mode=request.advisor_mode,
        repeated_game_mode=request.repeated_game_mode,
        observer_information_modes=request.observer_information_modes,
        agent_configs=request.agent_configs,
    )
    raw_agent_tokens, agent_token_hashes = _new_agent_credentials(
        tuple(state.company_ids)
    )
    session = EpisodeSession(
        env=env,
        manifest=manifest,
        game_mode=request.game_mode,
        player_company_id=player_company_id,
        communication_mode=request.communication_mode,
        cooperation_mode=request.cooperation_mode,
        belief_mode=request.belief_mode,
        opponent_model_mode=request.opponent_model_mode,
        utility_inference_mode=request.utility_inference_mode,
        advisor_mode=request.advisor_mode,
        repeated_game_mode=request.repeated_game_mode,
        cooperation_ledger=CooperationLedger(
            mode=request.cooperation_mode,
            episode_id=state.episode_id,
            company_ids=state.company_ids,
            max_rounds=state.max_rounds,
        ),
        belief_ledger=(
            BeliefLedger(
                episode_id=state.episode_id,
                company_ids=state.company_ids,
                mode=request.belief_mode,
            )
            if request.belief_mode != "off"
            else None
        ),
        opponent_model_ledger=(
            OpponentModelLedger(
                episode_id=state.episode_id,
                company_ids=state.company_ids,
            )
            if request.opponent_model_mode == "public_strategy_v1"
            else None
        ),
        agent_token_hashes=agent_token_hashes,
    )
    with SESSIONS_LOCK:
        if episode_id in SESSIONS:
            raise HTTPException(status_code=409, detail="episode_id already exists")
        SESSIONS[episode_id] = session
    payload = _episode_payload(session)
    if protected_creation:
        payload["agent_tokens"] = raw_agent_tokens
        payload["agent_token_header"] = "X-Agent-Token"
        payload["agent_tokens_returned_once"] = True
    return payload


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
        _reject_direct_step_when_interaction_enabled(session)
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
        payload = {
            "step_result": result.to_dict(),
            "state": result.state_after.to_dict(),
            "settled_market": settled_market_snapshot(session.transitions[-1]),
            "action_constraints": _constraints(session.env),
            "decision_resolutions": {
                company_id: decision.to_dict()
                for company_id, decision in resolutions.items()
            },
        }
        return payload


@app.post("/api/episodes/{episode_id}/player-steps")
def step_player_episode(episode_id: str, request: PlayerStepRequest) -> dict[str, Any]:
    session = _session(episode_id)
    with session.lock:
        _reject_direct_step_when_interaction_enabled(session)
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


@app.post(
    "/api/v1/controller/episodes/{episode_id}/communication/close"
)
def close_agent_communication(
    episode_id: str,
    request: CloseCommunicationRequest,
    controller_token: str | None = Header(
        default=None, alias="X-Controller-Token"
    ),
) -> dict[str, Any]:
    """Atomically close one round; repeated identical close calls are safe."""

    _require_controller_token(controller_token)
    session = _session(episode_id)
    with session.lock:
        key = _communication_key(
            request.round, request.state_version, request.state_hash
        )
        prior = session.communication_ledgers.get(key)
        if prior is not None and prior.status == "closed":
            cooperation_close = (
                session.cooperation_ledger.close_round(prior.close())
                if session.cooperation_ledger is not None
                and session.cooperation_mode != "off"
                else None
            )
            return {
                "communication_phase": _communication_phase_payload(prior),
                "cooperation_close": (
                    cooperation_close.model_dump(mode="json")
                    if cooperation_close is not None
                    else None
                ),
                "market_state_unchanged": True,
            }

        state = session.env.get_state()
        _require_current_communication_state(
            state,
            round_number=request.round,
            state_version=request.state_version,
            state_hash=request.state_hash,
        )
        before_hash = state.state_hash
        ledger = _communication_ledger(session, state)
        communication_phase = _communication_phase_payload(ledger)
        try:
            cooperation_close = (
                session.cooperation_ledger.close_round(ledger.close())
                if session.cooperation_ledger is not None
                and session.cooperation_mode != "off"
                else None
            )
        except CooperationProtocolError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        after_hash = session.env.get_state().state_hash
        if after_hash != before_hash:
            raise RuntimeError("communication close unexpectedly changed market state")
        return {
            "communication_phase": communication_phase,
            "cooperation_close": (
                cooperation_close.model_dump(mode="json")
                if cooperation_close is not None
                else None
            ),
            "market_state_unchanged": after_hash == before_hash,
        }


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
        request_key = {
            "intent_ids": dict(sorted(request.intent_ids.items())),
            "fallback": request.fallback,
        }
        cached = session.agent_settlements.get(request.step_id)
        if cached is not None:
            cached_key, cached_payload = cached
            if cached_key != request_key:
                raise HTTPException(
                    status_code=409,
                    detail="step_id was already used with a different settlement request",
                )
            return cached_payload
        state_before = session.env.get_state()
        if state_before.terminal:
            raise HTTPException(status_code=409, detail="episode is terminal")
        if session.communication_mode != "off":
            ledger = _communication_ledger(session, state_before)
            if ledger.status != "closed":
                raise HTTPException(
                    status_code=409,
                    detail={"code": "COMMUNICATION_NOT_CLOSED"},
                )
        if (
            session.cooperation_mode == "shared_resilience_v1"
            and session.cooperation_ledger is not None
            and not session.cooperation_ledger.has_closed_round(
                state_before.round
            )
        ):
            raise HTTPException(
                status_code=409,
                detail={"code": "COOPERATION_NOT_CLOSED"},
            )
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
        no_public_protection_result = None
        if session.cooperation_mode == "shared_resilience_v1":
            no_public_protection_result = (
                session.env.counterfactual_without_public_resilience(
                    state_before, actions
                )
            )
        result = session.env.step(request.step_id, actions)
        _record_transition(session, state_before, actions, result)
        cooperation_round = None
        if (
            session.cooperation_ledger is not None
            and session.cooperation_mode == "shared_resilience_v1"
        ):
            before_shared = state_before.shared_resilience
            after_shared = result.state_after.shared_resilience
            if before_shared is None or after_shared is None:
                raise RuntimeError("cooperation state is missing during settlement")
            protection_weight = int(
                CONFIG.mapping("shared_resilience")[
                    "public_protection_weight_ppm"
                ]
            )
            public_protection = min(
                1_000_000,
                protection_weight
                * before_shared.industry_resilience_ppm
                // 1_000_000,
            )
            if no_public_protection_result is None:
                raise RuntimeError("public protection counterfactual is missing")
            benefit_attribution = {}
            for company_id in state_before.company_ids:
                actual_company = result.state_after.company(company_id)
                counterfactual_company = (
                    no_public_protection_result.state_after.company(company_id)
                )
                actual_incident = actual_company.risk.active_incident
                counterfactual_incident = (
                    counterfactual_company.risk.active_incident
                )
                benefit_attribution[company_id] = (
                    CooperativeBenefitAttribution.from_counterfactual(
                        company_id=company_id,
                        current_contribution_cost_cents=int(
                            actions[
                                company_id
                            ].shared_resilience_contribution_cents
                            or 0
                        ),
                        latest_source_contribution_cents=int(
                            dict(
                                before_shared.last_contribution_by_company_cents
                            ).get(company_id, 0)
                        ),
                        public_protection_received_ppm=public_protection,
                        actual_round_profit_cents=(
                            actual_company.financial.round_profit_cents
                        ),
                        no_public_protection_round_profit_cents=(
                            counterfactual_company.financial.round_profit_cents
                        ),
                        avoided_next_incident=(
                            actual_incident is None
                            and counterfactual_incident is not None
                        ),
                    )
                )
            cooperation_round = session.cooperation_ledger.settle_round(
                round_number=state_before.round,
                final_actions={
                    company_id: action.to_dict()
                    for company_id, action in actions.items()
                },
                industry_resilience_before_ppm=(
                    before_shared.industry_resilience_ppm
                ),
                public_protection_applied_ppm=public_protection,
                industry_resilience_after_ppm=(
                    after_shared.industry_resilience_ppm
                ),
                benefit_attribution_by_company=benefit_attribution,
            )
        for record in used_records:
            record.status = "executed"
            record.resolution = resolutions[record.company_id].to_dict()
        payload = {
            "step_result": result.to_dict(),
            "state": result.state_after.to_dict(),
            "action_constraints": _constraints(session.env),
            "decision_resolutions": {
                company_id: decision.to_dict()
                for company_id, decision in resolutions.items()
            },
            "executed_intent_ids": [record.intent_id for record in used_records],
            "cooperation_round": (
                cooperation_round.model_dump(mode="json")
                if cooperation_round is not None
                else None
            ),
        }
        session.agent_settlements[request.step_id] = (request_key, payload)
        return payload


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
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)

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
