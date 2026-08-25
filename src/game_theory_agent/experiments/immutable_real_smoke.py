"""One-call remote-model smoke for the immutable Agent version path."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

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
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.experiments.immutable_version_acceptance import (
    COMPANIES,
    PERSONAS,
    InProcessController,
    InProcessGateway,
    _write_json,
)
from game_theory_agent.game_theory.replay import verify_game_theory_replay
from game_theory_agent.information import verify_information_replay
from game_theory_agent.interaction.replay import verify_interaction_replay
from game_theory_agent.market import MarketEnv
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.market.replay import verify_replay
from game_theory_agent.orchestration import JsonlRoundEventLogger, RoundCoordinator
from game_theory_agent.strategic_reliability.cost_guard import (
    RealModelBudget,
    RealModelCostGuard,
)


async def run_real_smoke(
    *,
    registry_path: Path,
    output_dir: Path,
    seed: int,
    provider: str,
    model: str,
) -> dict[str, object]:
    if provider not in {"deepseek", "doubao"}:
        raise ValueError("provider must be deepseek or doubao")
    credential_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "ARK_API_KEY"
    if not os.environ.get(credential_name):
        raise RuntimeError(f"{credential_name} is not set")
    output_dir.mkdir(parents=True, exist_ok=True)
    controller_token = "immutable-real-smoke-controller"
    os.environ["MARKET_CONTROLLER_TOKEN"] = controller_token
    api.AGENT_REGISTRY_PATH = registry_path
    api.SESSIONS.clear()
    registry = AgentRegistry(registry_path)
    personas = PersonaRegistry.from_market_config(api.CONFIG)
    builder = CurrentAgentVersionBuilder(registry, api.CONFIG, personas)
    conservative_input_price = 2 if provider == "deepseek" else 1
    conservative_output_price = 4
    remote = builder.register(
        provider=provider,
        model=model,
        model_revision=f"provider-alias:{model}:2026-08-25",
        persona_id=PERSONAS[0],
        family_id="current-remote-pareto-agent",
        human_version=f"real-smoke-{provider}-{PERSONAS[0]}-1.0.0",
        max_schema_attempts=1,
        policy_parameters={
            "paid_rounds_csv": "1",
            "reserved_prompt_tokens_per_call": 32_000,
            "reserved_completion_tokens_per_call": 4_000,
            "input_price_microunits_per_token": conservative_input_price,
            "output_price_microunits_per_token": conservative_output_price,
        },
    )
    mocks = [
        builder.register(
            provider="mock",
            model="mock-balanced-v1",
            model_revision="mock-policy-v1.1.0",
            persona_id=persona_id,
            human_version=f"real-smoke-peer-{persona_id}-1.0.0",
        )
        for persona_id in PERSONAS[1:]
    ]
    manifests = [remote, *mocks]
    settings = common_episode_settings(manifests)
    episode_id = f"immutable-real-{provider}-seed-{seed}"
    agent_ids = {
        company_id: (
            f"registered-{provider}-{company_id}"
            if company_id == "company_A"
            else f"registered-mock-{company_id}"
        )
        for company_id in COMPANIES
    }
    create = TestClient(api.app).post(
        "/api/episodes",
        headers={"X-Controller-Token": controller_token},
        json={
            "episode_id": episode_id,
            "episode_seed": seed,
            "company_ids": list(COMPANIES),
            "max_rounds": 5,
            "market_model": "balanced",
            "agent_versioning_mode": "immutable_v1",
            "agent_version_ids": {
                company_id: manifest.agent_version_id
                for company_id, manifest in zip(COMPANIES, manifests)
            },
            "agent_configs": {
                company_id: {
                    "agent_id": agent_ids[company_id],
                    "provider": provider if company_id == "company_A" else "mock",
                    "persona_id": persona_id,
                }
                for company_id, persona_id in zip(COMPANIES, PERSONAS)
            },
            **settings,
        },
    )
    if create.status_code != 201:
        raise RuntimeError(create.text)
    created = create.json()
    guard = RealModelCostGuard(
        RealModelBudget(
            max_calls=1,
            max_prompt_tokens=32_000,
            max_completion_tokens=4_000,
            max_estimated_cost_microunits=(
                32_000 * conservative_input_price
                + 4_000 * conservative_output_price
            ),
        ),
        explicitly_authorized=True,
    )
    factory = RegisteredAgentRuntimeFactory(
        registry,
        api.CONFIG,
        personas,
        real_model_cost_guard=guard,
    )
    runtimes = {
        company_id: factory.build(
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
        InProcessController(controller_token),
        InProcessGateway(created["agent_tokens"]),
        runtimes,
        event_logger=JsonlRoundEventLogger(event_path),
    )
    coordinated = await coordinator.run_episode(episode_id, max_rounds=1)
    events = JsonlRoundEventLogger(event_path).read_all()
    session = api.SESSIONS[episode_id]
    session.manifest.verify_agent_roster_hash()
    economic_states = verify_replay(
        MarketEnv(api.CONFIG), session.manifest, session.transitions
    )
    information = verify_information_replay(events, session.manifest)
    interaction = verify_interaction_replay(events)
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
    remote_trace = next(
        trace for trace in events[0].traces if trace.company_id == "company_A"
    )
    actual = guard.actual
    reserved = guard.reserved
    payload: dict[str, object] = {
        "report_schema_version": "immutable-agent-real-smoke-v1.0.0",
        "episode_id": episode_id,
        "seed": seed,
        "provider": provider,
        "model": model,
        "agent_version_id": remote.agent_version_id,
        "reproducibility_tier": remote.reproducibility_tier,
        "rounds_executed": len(coordinated),
        "terminal_expected": False,
        "remote_decision_status": remote_trace.decision_status,
        "remote_agent_type": remote_trace.agent_type,
        "remote_version_attributed": bool(
            remote_trace.agent_version_id == remote.agent_version_id
            and remote_trace.registry_manifest_hash == remote.manifest_hash
        ),
        "provider_audit": (
            remote_trace.provider_audit.model_dump(mode="json")
            if remote_trace.provider_audit is not None
            else None
        ),
        "usage": {
            "reserved_calls": reserved.calls,
            "successful_calls": actual.calls,
            "input_tokens": actual.prompt_tokens,
            "output_tokens": actual.completion_tokens,
            "estimated_cost_cny_microunits": actual.estimated_cost_microunits,
            "reserved_cost_cny_microunits": reserved.estimated_cost_microunits,
            "pricing_snapshot": (
                "deepseek-official-2026-08-25-peak-conservative"
                if provider == "deepseek"
                else "project-doubao-conservative"
            ),
        },
        "replay": {
            "economic_transition_count": len(session.transitions),
            "economic_replayed_state_count": len(economic_states),
            "information_snapshot_count": len(information),
            "interaction_closure_count": len(interaction),
            "game_theory_trace_binding_count": game_theory.trace_binding_count,
            "hidden_state_leak_count": game_theory.hidden_state_leak_count,
            "agent_attribution": "passed",
        },
        "smoke_passed": bool(
            remote_trace.decision_status == "submitted"
            and remote_trace.agent_type == "model"
            and actual.calls == 1
            and actual.prompt_tokens > 0
            and actual.completion_tokens > 0
            and remote_trace.provider_audit is not None
            and bool(remote_trace.provider_audit.request_id)
            and bool(remote_trace.provider_audit.response_model)
            and len(session.transitions) == 1
            and len(economic_states) == 2
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
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("deepseek", "doubao"), default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--registry", type=Path, default=Path("runs/immutable-real-smoke/registry")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/immutable-real-smoke")
    )
    args = parser.parse_args()
    result = asyncio.run(
        run_real_smoke(
            registry_path=args.registry,
            output_dir=args.output,
            seed=args.seed,
            provider=args.provider,
            model=args.model,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["smoke_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
