"""In-process four-Agent, 20-round orchestration and replay acceptance."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from dotenv import dotenv_values, load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Resolve only the config selector before importing api.CONFIG. Loading the
# whole .env here would leak controller credentials as an import side effect.
_PROJECT_ENV = dotenv_values(PROJECT_ROOT / ".env")
if _PROJECT_ENV.get("MARKET_CONFIG_PATH"):
    os.environ.setdefault(
        "MARKET_CONFIG_PATH", str(_PROJECT_ENV["MARKET_CONFIG_PATH"])
    )

from game_theory_agent.agents import (
    AgentCommunicationResult,
    AgentDecisionResult,
    AgentRuntime,
    DecisionContextBuilder,
    load_persona_registry,
)
from game_theory_agent.api import CONFIG, CONFIG_PATH, SESSIONS, agent_app, app
from game_theory_agent.market import MarketEnv
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.market.replay import verify_replay
from game_theory_agent.interaction.replay import (
    InteractionReplayMismatchError,
    verify_interaction_replay,
)
from game_theory_agent.interaction import (
    CommunicationSubmission,
    MessageDraft,
    PartialActionClaim,
)
from game_theory_agent.information import (
    InformationReplayMismatchError,
    verify_information_replay,
)
from game_theory_agent.belief import (
    BeliefReplayMismatchError,
    compute_belief_calibration,
    verify_belief_replay,
)
from game_theory_agent.game_theory import verify_game_theory_replay
from game_theory_agent.model_clients import MockModelClient
from game_theory_agent.orchestration import JsonlRoundEventLogger, RoundCoordinator
from game_theory_agent.experiments.persona_pilot import _model_client
from game_theory_agent.experiments.market_metrics import compute_research_metrics
from game_theory_agent.experiments.interaction_metrics import (
    compute_interaction_metrics,
)


COMPANIES = ("company_A", "company_B", "company_C", "company_D")
ACTIVE_COMMUNICATION_SCENARIOS = frozenset(
    {"public_price", "private_price", "mixed"}
)
EXPERIMENT_CONDITIONS = {
    "A_balanced_legacy": {
        "personas": ("balanced_v1",),
        "decision_support_version": "legacy_v1",
        "persona_semantics_version": "legacy_v1",
        "diagnostic_mode": "off",
    },
    "B_extreme_legacy": {
        "personas": (
            "balanced_v1",
            "aggressive_v1_extreme",
            "conservative_v1_extreme",
            "selfish_long_term_v1",
        ),
        "decision_support_version": "legacy_v1",
        "persona_semantics_version": "legacy_v1",
        "diagnostic_mode": "off",
    },
    "C_extreme_support": {
        "personas": (
            "balanced_v1",
            "aggressive_v1_extreme",
            "conservative_v1_extreme",
            "selfish_long_term_v1",
        ),
        "decision_support_version": "economic_v2",
        "persona_semantics_version": "legacy_v1",
        "diagnostic_mode": "off",
    },
    "D_extreme_semantics": {
        "personas": (
            "balanced_v1",
            "aggressive_v1_extreme",
            "conservative_v1_extreme",
            "selfish_long_term_v1",
        ),
        "decision_support_version": "economic_v2",
        "persona_semantics_version": "economic_v2",
        "diagnostic_mode": "off",
    },
    "E_moderate_semantics": {
        "personas": (
            "balanced_v1",
            "disciplined_growth_v1",
            "risk_guarded_v1",
            "selfish_long_term_v1",
        ),
        "decision_support_version": "economic_v2",
        "persona_semantics_version": "economic_v2",
        "diagnostic_mode": "off",
    },
    "F_moderate_diagnostics": {
        "personas": (
            "balanced_v1",
            "disciplined_growth_v1",
            "risk_guarded_v1",
            "selfish_long_term_v1",
        ),
        "decision_support_version": "economic_v2",
        "persona_semantics_version": "economic_v2",
        "diagnostic_mode": "observe",
    },
}


def _project_version() -> str:
    try:
        return importlib.metadata.version("game-theory-agent")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _validate_acceptance_args(args: argparse.Namespace) -> None:
    """Reject configurations that cannot prove an active interaction effect."""

    scenario = getattr(args, "mock_communication_scenario", "silence")
    if scenario == "silence":
        return
    if scenario not in ACTIVE_COMMUNICATION_SCENARIOS:
        raise ValueError(f"unsupported mock communication scenario: {scenario}")
    if args.provider != "mock":
        raise ValueError(
            "an active mock communication scenario requires --provider mock"
        )
    if args.rounds != 20:
        raise ValueError(
            "an active mock communication scenario requires exactly 20 rounds"
        )
    communication_mode = getattr(args, "communication_mode", "off")
    if communication_mode == "off":
        raise ValueError("an active mock communication scenario requires communication on")
    if scenario in {"public_price", "mixed"} and args.llm_count != 4:
        raise ValueError(
            "the public and mixed scenarios require --llm-count 4"
        )
    if scenario in {"private_price", "mixed"}:
        if args.llm_count < 2:
            raise ValueError(
                "the private and mixed scenarios require --llm-count >= 2"
            )
        if communication_mode != "public_private":
            raise ValueError(
                "the private and mixed scenarios require public_private mode"
            )


def _ensure_clean_round_event_log(path: Path) -> None:
    """Never append an acceptance run to an existing non-empty event log."""

    if path.exists() and path.stat().st_size > 0:
        raise FileExistsError(
            "refusing to append acceptance events to non-empty log: "
            f"{path.resolve()}"
        )


def _active_round_evidence(
    events: list[Any], scenario: str
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    """Evaluate every active-scenario contract separately for every round."""

    if scenario not in ACTIVE_COMMUNICATION_SCENARIOS:
        return {}, []

    expected_channels = {
        "public_price": ["public"],
        "private_price": ["private"],
        "mixed": ["private", "public"],
    }[scenario]
    expected_target_prices = {
        "public_price": {
            "company_B": 11_000,
            "company_C": 11_000,
            "company_D": 11_000,
        },
        "private_price": {"company_B": 12_345},
        "mixed": {
            "company_B": 12_345,
            "company_C": 11_000,
            "company_D": 11_000,
        },
    }[scenario]

    evidence: list[dict[str, Any]] = []
    for event in events:
        phase = event.communication_phase
        messages = list(phase.closure.all_messages) if phase is not None else []
        a_messages = [
            message
            for message in messages
            if message.sender_company_id == "company_A"
        ]
        actual_channels = sorted(message.channel for message in a_messages)
        messages_by_channel = {
            message.channel: message for message in a_messages
        }
        message_contract_met = actual_channels == expected_channels
        if "public" in expected_channels:
            public_message = messages_by_channel.get("public")
            message_contract_met = bool(
                message_contract_met
                and public_message is not None
                and list(public_message.recipients) == []
                and public_message.speech_act == "proposal"
                and public_message.requested_peer_action is not None
                and public_message.requested_peer_action.price_cents == 11_000
            )
        if "private" in expected_channels:
            private_message = messages_by_channel.get("private")
            message_contract_met = bool(
                message_contract_met
                and private_message is not None
                and list(private_message.recipients) == ["company_B"]
                and private_message.speech_act == "proposal"
                and private_message.requested_peer_action is not None
                and private_message.requested_peer_action.price_cents == 12_345
            )

        a_generation_traces = (
            [
                trace
                for trace in phase.generation_traces
                if trace.company_id == "company_A"
            ]
            if phase is not None
            else []
        )
        a_generation = (
            a_generation_traces[0] if len(a_generation_traces) == 1 else None
        )
        accepted_message_ids = {
            message_id
            for trace in a_generation_traces
            for message_id in trace.accepted_message_ids
        }
        actual_message_ids = {message.message_id for message in a_messages}
        generation_submitted = bool(
            a_generation is not None
            and a_generation.generation_status == "submitted"
            and a_generation.submission is not None
            and sorted(
                message.channel for message in a_generation.submission.messages
            )
            == expected_channels
            and accepted_message_ids == actual_message_ids
        )

        actual_target_prices = {
            company_id: event.joint_action.get(company_id, {}).get("price_cents")
            for company_id in expected_target_prices
        }
        target_prices_met = actual_target_prices == expected_target_prices
        traces = {trace.company_id: trace for trace in event.traces}
        visible_ids_by_company = {
            company_id: {
                message.message_id
                for message in (trace.communication_view.visible_messages or [])
            }
            for company_id, trace in traces.items()
            if trace.communication_view is not None
        }
        accepted_response_ids_by_company: dict[str, set[str]] = {}
        for company_id, trace in traces.items():
            accepted_response_ids_by_company[company_id] = {
                (
                    response.get("message_id")
                    if isinstance(response, dict)
                    else response.message_id
                )
                for response in (trace.message_responses or [])
                if (
                    response.get("disposition")
                    if isinstance(response, dict)
                    else response.disposition
                )
                == "accepted"
            }
        visibility_and_response_met = True
        if "public" in expected_channels:
            public_id = messages_by_channel["public"].message_id
            visibility_and_response_met = visibility_and_response_met and all(
                public_id in visible_ids_by_company.get(company_id, set())
                and public_id
                in accepted_response_ids_by_company.get(company_id, set())
                for company_id in ("company_B", "company_C", "company_D")
            )
        if "private" in expected_channels:
            private_id = messages_by_channel["private"].message_id
            visibility_and_response_met = bool(
                visibility_and_response_met
                and private_id in visible_ids_by_company.get("company_A", set())
                and private_id in visible_ids_by_company.get("company_B", set())
                and private_id
                in accepted_response_ids_by_company.get("company_B", set())
                and private_id
                not in visible_ids_by_company.get("company_C", set())
                and private_id
                not in visible_ids_by_company.get("company_D", set())
            )
        actual_target_requests = {
            company_id: (
                traces.get(company_id).requested_action or {}
            ).get("price_cents")
            if traces.get(company_id) is not None
            else None
            for company_id in expected_target_prices
        }
        target_requests_met = actual_target_requests == expected_target_prices
        row = {
            "round": event.settled_round,
            "a_expected_message_count": len(expected_channels),
            "a_actual_message_count": len(a_messages),
            "a_expected_channels": expected_channels,
            "a_actual_channels": actual_channels,
            "a_message_contract_met": message_contract_met,
            "a_generation_status": (
                a_generation.generation_status
                if a_generation is not None
                else None
            ),
            "a_generation_submitted": generation_submitted,
            "message_visibility_and_accepted_response_met": (
                visibility_and_response_met
            ),
            "expected_target_prices": expected_target_prices,
            "actual_target_requested_prices": actual_target_requests,
            "target_requested_prices_met": target_requests_met,
            "actual_target_prices": actual_target_prices,
            "target_prices_met": target_prices_met,
        }
        row["passed"] = all(
            (
                row["a_message_contract_met"],
                row["a_generation_submitted"],
                row["message_visibility_and_accepted_response_met"],
                row["target_requested_prices_met"],
                row["target_prices_met"],
            )
        )
        evidence.append(row)

    checks = {
        "active_interaction_exactly_20_rounds": len(events) == 20,
        "active_a_message_contract_every_round": bool(evidence)
        and all(row["a_message_contract_met"] for row in evidence),
        "active_a_generation_submitted_every_round": bool(evidence)
        and all(row["a_generation_submitted"] for row in evidence),
        "active_visibility_and_accepted_response_every_round": bool(evidence)
        and all(
            row["message_visibility_and_accepted_response_met"]
            for row in evidence
        ),
        "active_target_requested_price_every_round": bool(evidence)
        and all(row["target_requested_prices_met"] for row in evidence),
        "active_target_final_price_every_round": bool(evidence)
        and all(row["target_prices_met"] for row in evidence),
        "active_round_contract_every_round": bool(evidence)
        and all(row["passed"] for row in evidence),
    }
    return checks, evidence


def _seed_split(seed: int) -> str:
    if 101 <= seed <= 110:
        return "development"
    if 201 <= seed <= 220:
        return "validation"
    if 1001 <= seed <= 1030:
        return "final_holdout"
    return "custom"


def _acceptance_model_client(
    args: argparse.Namespace,
    company_id: str,
) -> object:
    scenario = getattr(args, "mock_communication_scenario", "silence")
    communication_mode = getattr(args, "communication_mode", "off")
    honor_advice = bool(
        getattr(args, "honor_game_theory_advice", False)
    )
    if args.provider == "mock" and scenario == "silence" and honor_advice:
        return MockModelClient(honor_game_theory_advice=True)
    if args.provider != "mock" or scenario == "silence":
        return _model_client(
            args.provider,
            args.model,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    if communication_mode == "off":
        raise ValueError("a mock communication scenario requires communication on")
    if scenario == "mixed" and communication_mode != "public_private":
        raise ValueError("the mixed scenario requires public_private mode")

    messages: list[MessageDraft] = []
    if company_id == "company_A" and scenario in {"public_price", "mixed"}:
        messages.append(
            MessageDraft(
                channel="public",
                speech_act="proposal",
                content="Consider a price of 11000 cents this round.",
                requested_peer_action=PartialActionClaim(price_cents=11_000),
            )
        )
    if company_id == "company_A" and scenario in {"private_price", "mixed"}:
        if communication_mode != "public_private":
            raise ValueError("the private scenario requires public_private mode")
        messages.append(
            MessageDraft(
                channel="private",
                recipients=["company_B"],
                speech_act="proposal",
                content="Privately consider a price of 12345 cents this round.",
                requested_peer_action=PartialActionClaim(price_cents=12_345),
            )
        )
    return MockModelClient(
        model_name=f"mock-interaction-{company_id}",
        communication_submission=CommunicationSubmission(messages=messages),
        honor_requested_price=company_id != "company_A",
        honor_game_theory_advice=honor_advice,
    )


class _GatewayAdapter:
    def __init__(self, agent_tokens: dict[str, str] | None = None) -> None:
        self.client = TestClient(agent_app)
        self.agent_tokens = dict(agent_tokens or {})

    def _headers(self, company_id: str) -> dict[str, str]:
        token = self.agent_tokens.get(company_id)
        return {"X-Agent-Token": token} if token else {}

    async def get_observation(
        self, episode_id: str, company_id: str
    ) -> dict[str, Any]:
        response = self.client.get(
            f"/v1/episodes/{episode_id}/companies/{company_id}/observation",
            headers=self._headers(company_id),
        )
        response.raise_for_status()
        return response.json()

    async def submit_communication(
        self, episode_id: str, result: AgentCommunicationResult
    ) -> dict[str, Any]:
        context = result.context
        response = self.client.post(
            (
                f"/v1/episodes/{episode_id}/companies/{result.company_id}/"
                "communication/submissions"
            ),
            headers=self._headers(result.company_id),
            json={
                "round": context.round,
                "state_version": context.state_version,
                "state_hash": context.state_hash,
                "submission": result.submission.model_dump(mode="json"),
            },
        )
        response.raise_for_status()
        return response.json()

    async def submit_intent(
        self, episode_id: str, result: AgentDecisionResult
    ) -> dict[str, Any]:
        if result.decision is None:
            raise ValueError("successful result must contain a decision")
        response = self.client.post(
            f"/v1/episodes/{episode_id}/intents",
            headers=self._headers(result.company_id),
            json={
                "agent_id": result.agent_id,
                "company_id": result.company_id,
                "round": result.context.round,
                "state_version": result.context.state_version,
                "observation_hash": result.context.meta.observation_hash,
                "requested_action": result.decision.requested_action.model_dump(
                    mode="json"
                ),
                "rationale": result.decision.plan.situation_summary,
                "expected_outcome": json.dumps(
                    result.decision.plan.expected_outcome.model_dump()
                ),
                "communication_view_digest": (
                    result.context.communication_view.view_digest
                    if result.context.communication_view is not None
                    else None
                ),
            },
        )
        response.raise_for_status()
        return response.json()


class _ControllerAdapter:
    def __init__(self, token: str) -> None:
        self.client = TestClient(app)
        self.headers = {"X-Controller-Token": token}

    async def get_episode(self, episode_id: str) -> dict[str, Any]:
        response = self.client.get(f"/api/episodes/{episode_id}/state")
        response.raise_for_status()
        return response.json()

    async def settle_agent_round(
        self,
        episode_id: str,
        step_id: str,
        intent_ids: dict[str, str],
    ) -> dict[str, Any]:
        response = self.client.post(
            f"/api/v1/controller/episodes/{episode_id}/settle-agent-round",
            headers=self.headers,
            json={
                "step_id": step_id,
                "intent_ids": intent_ids,
                "fallback": "rule",
            },
        )
        response.raise_for_status()
        return response.json()

    async def close_communication(
        self,
        episode_id: str,
        round_number: int,
        state_version: int,
        state_hash: str,
    ) -> dict[str, Any]:
        response = self.client.post(
            f"/api/v1/controller/episodes/{episode_id}/communication/close",
            headers=self.headers,
            json={
                "round": round_number,
                "state_version": state_version,
                "state_hash": state_hash,
            },
        )
        response.raise_for_status()
        return response.json()


async def run(args: argparse.Namespace) -> int:
    _validate_acceptance_args(args)
    load_dotenv(PROJECT_ROOT / ".env")
    output = Path(args.output).resolve()
    round_event_path = output / "round-events.jsonl"
    _ensure_clean_round_event_log(round_event_path)

    token = f"four-agent-acceptance-{uuid.uuid4().hex}"
    os.environ["MARKET_CONTROLLER_TOKEN"] = token
    SESSIONS.clear()
    episode_id = args.episode_id or f"four-agent-acceptance-{uuid.uuid4().hex[:8]}"
    communication_mode = getattr(args, "communication_mode", "off")
    information_mode = getattr(args, "information_mode", "perfect")
    privileged_observer_company_id = getattr(
        args, "privileged_observer_company_id", None
    )
    observer_information_modes = (
        {privileged_observer_company_id: "perfect"}
        if privileged_observer_company_id is not None
        else {}
    )
    belief_mode = getattr(args, "belief_mode", "off")
    communication_timeout = float(
        getattr(args, "communication_timeout", 30.0)
    )
    mock_communication_scenario = getattr(
        args, "mock_communication_scenario", "silence"
    )
    created = TestClient(app).post(
        "/api/episodes",
        headers={"X-Controller-Token": token},
        json={
            "episode_id": episode_id,
            "episode_seed": args.seed,
            "company_ids": list(COMPANIES),
            "max_rounds": args.rounds,
            "market_model": args.market_model,
            "information_mode": information_mode,
            "observer_information_modes": observer_information_modes,
            "communication_mode": communication_mode,
            "belief_mode": belief_mode,
            "opponent_model_mode": getattr(
                args, "opponent_model_mode", "off"
            ),
            "utility_inference_mode": getattr(
                args, "utility_inference_mode", "off"
            ),
            "advisor_mode": getattr(args, "advisor_mode", "off"),
            "repeated_game_mode": getattr(
                args, "repeated_game_mode", "off"
            ),
            "cooperation_mode": getattr(args, "cooperation_mode", "off"),
        },
    )
    created.raise_for_status()
    logger = JsonlRoundEventLogger(round_event_path)
    llm_companies = COMPANIES[: args.llm_count]
    config_path = Path(
        os.environ.get(
            "MARKET_CONFIG_PATH", PROJECT_ROOT / "configs" / "market_v4.yaml"
        )
    )
    registry = load_persona_registry(config_path)
    condition = EXPERIMENT_CONDITIONS.get(args.condition)
    persona_ids = (
        tuple(condition["personas"])
        if condition is not None
        else (
            tuple(item.strip() for item in args.personas.split(",") if item.strip())
            if args.personas
            else (args.persona,)
        )
    )
    if len(persona_ids) == 1:
        persona_ids = persona_ids * args.llm_count
    if len(persona_ids) != args.llm_count:
        raise ValueError(
            "--personas must contain either one persona or exactly --llm-count "
            f"personas; received {len(persona_ids)} for {args.llm_count} LLMs"
        )
    if len(persona_ids) > 1 and args.rotation_index:
        shift = args.rotation_index % len(persona_ids)
        persona_ids = persona_ids[shift:] + persona_ids[:shift]
    decision_support_version = str(
        condition["decision_support_version"]
        if condition is not None
        else args.decision_support_version
    )
    persona_semantics_version = str(
        condition["persona_semantics_version"]
        if condition is not None
        else args.persona_semantics_version
    )
    diagnostic_mode = str(
        condition["diagnostic_mode"]
        if condition is not None
        else args.diagnostic_mode
    )
    persona_by_company = dict(zip(llm_companies, persona_ids, strict=True))
    runtimes = {
        company_id: AgentRuntime(
            f"{args.provider}-{persona_by_company[company_id]}-{company_id}",
            company_id,
            _acceptance_model_client(args, company_id),
            context_builder=DecisionContextBuilder(
                persona_profile=registry.get(persona_by_company[company_id]),
                persona_registry=registry,
                decision_support_version=decision_support_version,
                persona_semantics_version=persona_semantics_version,
                diagnostic_mode=diagnostic_mode,
            ),
            persona_profile=registry.get(persona_by_company[company_id]),
            persona_registry=registry,
        )
        for company_id in llm_companies
    }
    coordinator = RoundCoordinator(
        _ControllerAdapter(token),
        _GatewayAdapter(created.json().get("agent_tokens", {})),
        runtimes,
        event_logger=logger,
        decision_timeout_seconds=args.timeout,
        communication_timeout_seconds=communication_timeout,
    )
    rounds = await coordinator.run_episode(episode_id)
    runtime_events = [item.event for item in rounds]
    logged_all_events = list(logger.read_all())
    events = [
        event for event in logged_all_events if event.episode_id == episode_id
    ]
    event_log_episode_only = len(events) == len(logged_all_events)
    event_log_count_matches_runtime = len(events) == len(runtime_events)
    event_log_content_matches_runtime = (
        event_log_count_matches_runtime
        and [event.model_dump(mode="json") for event in events]
        == [event.model_dump(mode="json") for event in runtime_events]
    )
    same_version = all(
        trace.decision_context is not None
        and trace.decision_context["meta"]["state_version"]
        == event.state_before["state_version"]
        and trace.decision_context["meta"]["state_hash"]
        == event.state_before_hash
        for event in events
        for trace in event.traces
        if trace.company_id in llm_companies
    )
    complete_events = all(
        len(event.traces) == 4
        and bool(event.state_before)
        and bool(event.state_after)
        and len(event.joint_action) == 4
        and all(
            trace.final_action
            and trace.result_analysis is not None
            and (
                trace.company_id not in llm_companies
                or (
                    trace.observation is not None
                    and trace.decision_context is not None
                    and trace.planner_output is not None
                    and trace.requested_action is not None
                )
            )
            for trace in event.traces
        )
        for event in events
    )
    action_ids = [
        action["action_id"]
        for event in events
        for action in event.joint_action.values()
    ]
    no_partial_updates = all(
        event.state_after["state_version"]
        == event.state_before["state_version"] + 1
        for event in events
    )
    session = SESSIONS[episode_id]
    replay_states = verify_replay(
        MarketEnv(CONFIG), session.manifest, session.transitions
    )
    replay_match = (
        replay_states[-1].state_hash == session.env.get_state().state_hash
    )
    interaction_replay_error: str | None = None
    try:
        interaction_replay = verify_interaction_replay(events)
        interaction_replay_match = len(interaction_replay) == len(events)
    except InteractionReplayMismatchError as exc:
        interaction_replay_match = False
        interaction_replay_error = str(exc)
    information_replay_error: str | None = None
    try:
        information_replay = verify_information_replay(
            events, session.manifest
        )
        information_replay_match = (
            len(information_replay)
            == len(events)
            * args.llm_count
            * (2 if communication_mode != "off" else 1)
        )
    except InformationReplayMismatchError as exc:
        information_replay_match = False
        information_replay_error = str(exc)
    belief_replay_error: str | None = None
    try:
        belief_replay = verify_belief_replay(events, session.manifest)
        expected_belief_snapshots = (
            len(events) * args.llm_count
            if belief_mode != "off"
            else 0
        )
        belief_replay_match = len(belief_replay) == expected_belief_snapshots
    except BeliefReplayMismatchError as exc:
        belief_replay_match = False
        belief_replay_error = str(exc)
    game_theory_replay_error: str | None = None
    game_theory_replay = None
    try:
        game_theory_replay = verify_game_theory_replay(
            events, session.manifest
        )
        game_theory_replay_match = (
            game_theory_replay.hidden_state_leak_count == 0
        )
    except RuntimeError as exc:
        game_theory_replay_match = False
        game_theory_replay_error = str(exc)
    communication_complete = all(
        event.communication_phase is not None
        and event.communication_phase.closed
        and event.communication_phase.closure.state_hash
        == event.state_before_hash
        and all(
            trace.communication_view
            == event.communication_phase.closure.views[trace.company_id]
            for trace in event.traces
        )
        for event in events
    )
    interaction_metrics = compute_interaction_metrics(events)
    protocol_checks = {
        "episode_complete": bool(events)
        and len(events) == args.rounds
        and session.env.get_state().terminal,
        "round_event_log_episode_only": event_log_episode_only,
        "round_event_log_count_matches_runtime": event_log_count_matches_runtime,
        "round_event_log_content_matches_runtime": (
            event_log_content_matches_runtime
        ),
        "same_state_version_for_all_agents": same_version,
        "single_settlement_per_round": len(session.transitions) == len(events),
        "duplicate_action_zero": len(action_ids) == len(set(action_ids)),
        "partial_state_update_zero": no_partial_updates,
        "round_event_complete_100pct": complete_events,
        "llm_fallback_zero": all(
            trace.decision_status == "submitted"
            for event in events
            for trace in event.traces
            if trace.company_id in llm_companies
        ),
        "rule_fallback_count_expected": sum(
            trace.decision_status == "fallback"
            for event in events
            for trace in event.traces
            if trace.company_id not in llm_companies
        )
        == args.rounds * (len(COMPANIES) - args.llm_count),
        "replay_match_100pct": replay_match,
        "communication_phase_complete_100pct": communication_complete,
        "interaction_replay_match_100pct": interaction_replay_match,
        "interaction_replay_from_round_event_log_100pct": (
            interaction_replay_match
        ),
        "information_replay_match_100pct": information_replay_match,
        "belief_replay_match_100pct": belief_replay_match,
        "game_theory_replay_match_100pct": game_theory_replay_match,
    }
    active_interaction_checks, active_round_evidence = _active_round_evidence(
        events, mock_communication_scenario
    )
    protocol_passed = all(protocol_checks.values())
    active_interaction_passed: bool | None = (
        all(active_interaction_checks.values())
        if active_interaction_checks
        else None
    )
    passed = protocol_passed and active_interaction_passed is not False

    run_configuration = {
        "episode_id": episode_id,
        "seed": args.seed,
        "rounds": args.rounds,
        "market_model": args.market_model,
        "information_mode": information_mode,
        "observer_information_modes": observer_information_modes,
        "privileged_observer_company_id": privileged_observer_company_id,
        "communication_mode": communication_mode,
        "belief_mode": belief_mode,
        "opponent_model_mode": getattr(args, "opponent_model_mode", "off"),
        "utility_inference_mode": getattr(
            args, "utility_inference_mode", "off"
        ),
        "advisor_mode": getattr(args, "advisor_mode", "off"),
        "repeated_game_mode": getattr(args, "repeated_game_mode", "off"),
        "cooperation_mode": getattr(args, "cooperation_mode", "off"),
        "honor_game_theory_advice": bool(
            getattr(args, "honor_game_theory_advice", False)
        ),
        "communication_timeout_seconds": communication_timeout,
        "mock_communication_scenario": mock_communication_scenario,
        "provider": args.provider,
        "model": args.model,
        "llm_count": args.llm_count,
        "llm_companies": list(llm_companies),
        "personas": list(persona_ids),
        "persona_by_company": persona_by_company,
        "decision_support_version": decision_support_version,
        "persona_semantics_version": persona_semantics_version,
        "diagnostic_mode": diagnostic_mode,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "decision_timeout_seconds": args.timeout,
    }
    manifest = session.manifest.to_dict()
    event_schema_versions = sorted(
        {event.event_schema_version for event in events}
    )
    summary = {
        "acceptance_schema_version": "mixed-agent-acceptance-v1.3.0",
        "episode_id": episode_id,
        "seed": args.seed,
        "seed_split": _seed_split(args.seed),
        "experiment_condition": args.condition,
        "rotation_index": args.rotation_index,
        "rounds": len(events),
        "agent_count": 4,
        "provider": args.provider,
        "model": args.model,
        "persona": persona_ids[0] if len(set(persona_ids)) == 1 else None,
        "personas": list(persona_ids),
        "persona_by_company": persona_by_company,
        "persona_treatment": (
            "homogeneous" if len(set(persona_ids)) == 1 else "heterogeneous"
        ),
        "decision_support_version": decision_support_version,
        "persona_semantics_version": persona_semantics_version,
        "diagnostic_mode": diagnostic_mode,
        "information_mode": information_mode,
        "observer_information_modes": observer_information_modes,
        "privileged_observer_company_id": privileged_observer_company_id,
        "belief_mode": belief_mode,
        "communication_mode": communication_mode,
        "communication_timeout_seconds": communication_timeout,
        "mock_communication_scenario": mock_communication_scenario,
        "acceptance_scope": (
            "interaction_effect"
            if mock_communication_scenario != "silence"
            else "communication_barrier"
        ),
        "llm_count": args.llm_count,
        "rule_count": len(COMPANIES) - args.llm_count,
        "llm_companies": list(llm_companies),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "decision_count": sum(len(event.traces) for event in events),
        "protocol_checks": protocol_checks,
        "active_interaction_checks": active_interaction_checks,
        "active_round_evidence": active_round_evidence,
        # Backward-compatible aggregate index; pass/fail semantics are explicit
        # in the three fields below.
        "checks": {**protocol_checks, **active_interaction_checks},
        "protocol_passed": protocol_passed,
        "active_interaction_passed": active_interaction_passed,
        "passed": passed,
        "research_metrics": compute_research_metrics(events, CONFIG),
        "interaction_metrics": interaction_metrics,
        "interaction_replay_error": interaction_replay_error,
        "information_replay_error": information_replay_error,
        "belief_replay_error": belief_replay_error,
        "game_theory_replay_error": game_theory_replay_error,
        "game_theory_replay": (
            game_theory_replay.model_dump(mode="json")
            if game_theory_replay is not None
            else None
        ),
        "belief_calibration": compute_belief_calibration(events),
        "reproducibility": {
            "project_name": "game-theory-agent",
            "project_version": _project_version(),
            "acceptance_runner_sha256": _file_sha256(Path(__file__).resolve()),
            "market_config_path": str(CONFIG_PATH.resolve()),
            "market_config_id": CONFIG.config_id,
            "market_config_version": CONFIG.config_version,
            "market_config_sha256": CONFIG.config_sha256,
            "environment_version": CONFIG.environment_version,
            "episode_manifest_version": manifest["manifest_version"],
            "episode_manifest_sha256": sha256_hash(manifest),
            "event_schema_versions": event_schema_versions,
            "interaction_metrics_schema_version": interaction_metrics[
                "metrics_schema_version"
            ],
            "persona_catalog_version": registry.catalog_version,
            "persona_profile_hashes": {
                company_id: registry.get(persona_id).profile_hash
                for company_id, persona_id in persona_by_company.items()
            },
            "run_configuration": run_configuration,
            "run_configuration_sha256": _json_sha256(run_configuration),
            "round_event_log_sha256": _file_sha256(logger.path.resolve()),
        },
        "final_state_hash": session.env.get_state().state_hash,
        "round_event_path": str(logger.path.resolve()),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not getattr(args, "quiet", False):
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--market-model", default="balanced")
    parser.add_argument(
        "--information-mode",
        choices=("perfect", "public"),
        default="perfect",
    )
    parser.add_argument(
        "--privileged-observer-company-id",
        choices=COMPANIES,
        help=(
            "keep the episode in the selected information mode but give one "
            "company a perfect-information observation"
        ),
    )
    parser.add_argument(
        "--provider", choices=("mock", "doubao", "deepseek"), default="mock"
    )
    parser.add_argument(
        "--belief-mode",
        choices=("off", "public_action_v1", "public_action_signal_v2"),
        default="off",
    )
    parser.add_argument(
        "--opponent-model-mode",
        choices=("off", "public_strategy_v1"),
        default="off",
    )
    parser.add_argument(
        "--utility-inference-mode",
        choices=("off", "strategy_utility_v1"),
        default="off",
    )
    parser.add_argument(
        "--advisor-mode",
        choices=("off", "bayesian_price_v1", "bayesian_strategy_v2"),
        default="off",
    )
    parser.add_argument(
        "--repeated-game-mode",
        choices=("off", "reciprocity_v1"),
        default="off",
    )
    parser.add_argument(
        "--cooperation-mode",
        choices=("off", "shared_resilience_v1"),
        default="off",
    )
    parser.add_argument(
        "--honor-game-theory-advice",
        action="store_true",
        help="deterministic Mock policy adopts the advisor price",
    )
    parser.add_argument("--model")
    parser.add_argument("--persona", default="balanced")
    parser.add_argument("--condition", choices=tuple(EXPERIMENT_CONDITIONS))
    parser.add_argument(
        "--personas",
        help=(
            "comma-separated personas mapped to LLM companies in A-D order; "
            "one value is broadcast"
        ),
    )
    parser.add_argument("--llm-count", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--rotation-index", type=int, default=0)
    parser.add_argument(
        "--decision-support-version",
        choices=("legacy_v1", "economic_v2"),
        default="economic_v2",
    )
    parser.add_argument(
        "--persona-semantics-version",
        choices=("legacy_v1", "economic_v2"),
        default="economic_v2",
    )
    parser.add_argument(
        "--diagnostic-mode", choices=("off", "observe"), default="off"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--communication-mode",
        choices=("off", "public_only", "public_private"),
        default="off",
    )
    parser.add_argument("--communication-timeout", type=float, default=30.0)
    parser.add_argument(
        "--mock-communication-scenario",
        choices=("silence", "public_price", "private_price", "mixed"),
        default="silence",
        help="deterministic Mock-only message influence scenario",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    return asyncio.run(run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
