"""End-to-end P0-P2 acceptance using registered immutable Agent versions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from game_theory_agent import api
from game_theory_agent.agent_registry import (
    AgentInstanceBinding,
    AgentRegistry,
    replay_agent_attribution,
)
from game_theory_agent.agent_registry.bootstrap import CurrentAgentVersionBuilder
from game_theory_agent.agent_registry.runtime_factory import (
    RegisteredAgentRuntimeFactory,
    common_episode_settings,
)
from game_theory_agent.agents.contracts import AgentCommunicationResult, AgentDecisionResult
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.game_theory.replay import verify_game_theory_replay
from game_theory_agent.information import verify_information_replay
from game_theory_agent.interaction.replay import verify_interaction_replay
from game_theory_agent.market import MarketEnv
from game_theory_agent.market.protocols import canonical_json, sha256_hash
from game_theory_agent.market.replay import verify_replay
from game_theory_agent.orchestration import JsonlRoundEventLogger, RoundCoordinator


PERSONAS = (
    "balanced_v1",
    "aggressive_v1_extreme",
    "risk_guarded_v1",
    "profit_myopic",
)
COMPANIES = ("company_A", "company_B", "company_C", "company_D")


class InProcessGateway:
    def __init__(self, agent_tokens: dict[str, str]) -> None:
        self.client = TestClient(api.agent_app)
        self.agent_tokens = dict(agent_tokens)

    def _headers(self, company_id: str) -> dict[str, str]:
        return {"X-Agent-Token": self.agent_tokens[company_id]}

    async def get_observation(
        self, episode_id: str, company_id: str
    ) -> dict[str, Any]:
        response = self.client.get(
            f"/v1/episodes/{episode_id}/companies/{company_id}/observation",
            headers=self._headers(company_id),
        )
        if response.status_code != 200:
            raise RuntimeError(response.text)
        return response.json()

    async def submit_intent(
        self, episode_id: str, result: AgentDecisionResult
    ) -> dict[str, Any]:
        if result.decision is None:
            raise RuntimeError("cannot submit a missing decision")
        context = result.context
        response = self.client.post(
            f"/v1/episodes/{episode_id}/intents",
            headers=self._headers(result.company_id),
            json={
                "agent_id": result.agent_id,
                "company_id": result.company_id,
                "round": context.round,
                "state_version": context.state_version,
                "observation_hash": context.meta.observation_hash,
                "requested_action": result.decision.requested_action.model_dump(
                    mode="json"
                ),
                "rationale": result.decision.plan.situation_summary,
                "expected_outcome": result.decision.plan.strategy_summary,
                "communication_view_digest": (
                    context.communication_view.view_digest
                    if context.communication_view is not None
                    else None
                ),
            },
        )
        if response.status_code != 202:
            raise RuntimeError(response.text)
        return response.json()

    async def submit_communication(
        self, episode_id: str, result: AgentCommunicationResult
    ) -> dict[str, Any]:
        context = result.context
        response = self.client.post(
            f"/v1/episodes/{episode_id}/companies/{result.company_id}/communication/submissions",
            headers=self._headers(result.company_id),
            json={
                "round": context.round,
                "state_version": context.state_version,
                "observation_hash": context.meta.observation_hash,
                "submission": result.submission.model_dump(mode="json"),
            },
        )
        if response.status_code != 202:
            raise RuntimeError(response.text)
        return response.json()


class InProcessController:
    def __init__(self, token: str) -> None:
        self.client = TestClient(api.app)
        self.headers = {"X-Controller-Token": token}

    async def get_episode(self, episode_id: str) -> dict[str, Any]:
        response = self.client.get(f"/api/episodes/{episode_id}/state")
        if response.status_code != 200:
            raise RuntimeError(response.text)
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
        if response.status_code != 200:
            raise RuntimeError(response.text)
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
        if response.status_code != 200:
            raise RuntimeError(response.text)
        return response.json()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


async def run_acceptance(
    *,
    registry_path: Path,
    output_dir: Path,
    seed: int,
    rounds: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    token = "immutable-version-acceptance-controller"
    os.environ["MARKET_CONTROLLER_TOKEN"] = token
    api.AGENT_REGISTRY_PATH = registry_path
    api.SESSIONS.clear()
    registry = AgentRegistry(registry_path)
    personas = PersonaRegistry.from_market_config(api.CONFIG)
    builder = CurrentAgentVersionBuilder(registry, api.CONFIG, personas)
    manifests = [
        builder.register(
            provider="mock",
            model="mock-balanced-v1",
            model_revision="mock-policy-v1.1.0",
            persona_id=persona_id,
            human_version=f"validation-{persona_id}-1.0.0",
        )
        for persona_id in PERSONAS
    ]
    settings = common_episode_settings(manifests)
    episode_id = f"immutable-p0-p2-seed-{seed}"
    agent_ids = {
        company_id: f"registered-mock-{persona_id}"
        for company_id, persona_id in zip(COMPANIES, PERSONAS)
    }
    controller_client = TestClient(api.app)
    create_response = controller_client.post(
        "/api/episodes",
        headers={"X-Controller-Token": token},
        json={
            "episode_id": episode_id,
            "episode_seed": seed,
            "company_ids": list(COMPANIES),
            "max_rounds": rounds,
            "market_model": "balanced",
            "agent_versioning_mode": "immutable_v1",
            "agent_version_ids": {
                company_id: manifest.agent_version_id
                for company_id, manifest in zip(COMPANIES, manifests)
            },
            "agent_configs": {
                company_id: {
                    "agent_id": agent_ids[company_id],
                    "provider": "mock",
                    "persona_id": persona_id,
                }
                for company_id, persona_id in zip(COMPANIES, PERSONAS)
            },
            **settings,
        },
    )
    if create_response.status_code != 201:
        raise RuntimeError(create_response.text)
    created = create_response.json()
    session = api.SESSIONS[episode_id]
    runtime_factory = RegisteredAgentRuntimeFactory(
        registry,
        api.CONFIG,
        personas,
    )
    runtimes = {
        company_id: runtime_factory.build(
            agent_version_id=manifest.agent_version_id,
            episode_id=episode_id,
            company_id=company_id,
            agent_id=agent_ids[company_id],
        )
        for company_id, manifest in zip(COMPANIES, manifests)
    }
    event_path = output_dir / "agent-rounds.jsonl"
    if event_path.exists():
        event_path.unlink()
    coordinator = RoundCoordinator(
        InProcessController(token),
        InProcessGateway(created["agent_tokens"]),
        runtimes,
        event_logger=JsonlRoundEventLogger(event_path),
    )
    coordinated = await coordinator.run_episode(episode_id)
    events = JsonlRoundEventLogger(event_path).read_all()
    session.manifest.verify_agent_roster_hash()
    economic_states = verify_replay(
        MarketEnv(api.CONFIG), session.manifest, session.transitions
    )
    information_snapshots = verify_information_replay(events, session.manifest)
    interaction_closures = verify_interaction_replay(events)
    game_theory = verify_game_theory_replay(events, session.manifest)
    roster = {
        company_id: AgentInstanceBinding.model_validate(payload)
        for company_id, payload in dict(session.manifest.agent_roster).items()
    }
    replay_agent_attribution(
        episode_id=episode_id,
        company_ids=COMPANIES,
        roster=roster,
        events=events,
        registry=registry,
    )
    trace_count = sum(len(event.traces) for event in events)
    complete_version_traces = sum(
        bool(
            trace.agent_version_id
            and trace.agent_instance_id
            and trace.behavior_spec_hash
            and trace.prompt_bundle_hash
            and trace.source_bundle_hash
        )
        for event in events
        for trace in event.traces
    )
    input_tokens = sum(
        trace.input_tokens or 0 for event in events for trace in event.traces
    )
    output_tokens = sum(
        trace.output_tokens or 0 for event in events for trace in event.traces
    )
    payload = {
        "report_schema_version": "immutable-agent-acceptance-v1.0.0",
        "episode_id": episode_id,
        "seed": seed,
        "rounds": len(coordinated),
        "companies": list(COMPANIES),
        "agent_version_ids": {
            company_id: manifest.agent_version_id
            for company_id, manifest in zip(COMPANIES, manifests)
        },
        "unique_agent_version_count": len(
            {manifest.agent_version_id for manifest in manifests}
        ),
        "manifest_version": session.manifest.to_dict()["manifest_version"],
        "agent_versioning_mode": session.manifest.agent_versioning_mode,
        "agent_roster_hash": session.manifest.agent_roster_hash,
        "round_event_schema_versions": sorted(
            {event.event_schema_version for event in events}
        ),
        "trace_count": trace_count,
        "complete_version_trace_count": complete_version_traces,
        "version_attribution_rate_ppm": (
            complete_version_traces * 1_000_000 // max(1, trace_count)
        ),
        "replay": {
            "economic_transition_count": len(session.transitions),
            "economic_replayed_state_count": len(economic_states),
            "information_snapshot_count": len(information_snapshots),
            "interaction_closure_count": len(interaction_closures),
            "game_theory_event_count": game_theory.event_count,
            "game_theory_trace_binding_count": game_theory.trace_binding_count,
            "hidden_state_leak_count": game_theory.hidden_state_leak_count,
            "agent_attribution": "passed",
        },
        "terminal": coordinated[-1].settlement["state"]["terminal"],
        "final_state_hash": economic_states[-1].state_hash,
        "model_usage": {
            "real_llm_calls": 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_cny_micros": 0,
        },
        "acceptance_passed": bool(
            coordinated[-1].settlement["state"]["terminal"]
            and trace_count == rounds * len(COMPANIES)
            and complete_version_traces == trace_count
            and len(session.transitions) == rounds
            and len(economic_states) == rounds + 1
            and len(interaction_closures) == rounds
            and game_theory.hidden_state_leak_count == 0
        ),
    }
    payload["report_hash"] = sha256_hash(payload)
    _write_json(output_dir / "summary.json", payload)
    _write_json(output_dir / "episode-manifest.json", session.manifest.to_dict())
    _write_json(
        output_dir / "registered-versions.json",
        [manifest.model_dump(mode="json") for manifest in manifests],
    )
    (output_dir / "summary.canonical.json").write_text(
        canonical_json(payload), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--rounds", type=int, choices=(5, 10, 15, 20), default=5)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("runs/immutable-version-acceptance/registry"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/immutable-version-acceptance"),
    )
    args = parser.parse_args()
    result = asyncio.run(
        run_acceptance(
            registry_path=args.registry,
            output_dir=args.output,
            seed=args.seed,
            rounds=args.rounds,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
