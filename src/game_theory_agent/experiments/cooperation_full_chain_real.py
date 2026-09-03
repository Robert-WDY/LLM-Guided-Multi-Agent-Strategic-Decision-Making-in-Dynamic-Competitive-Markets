"""Run the smallest real-LLM proposal-to-verification cooperation chain.

Round 1 uses a deterministic proposer.  In round 2 the selected provider first
responds to that private proposal and then makes the real contribution decision.
All other agents and phases are deterministic, keeping paid calls to two per
episode while still exercising the production coordinator and four replay layers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from game_theory_agent.agents import (
    AgentRuntime,
    DecisionContextBuilder,
    EpisodeMemory,
    load_persona_registry,
)
from game_theory_agent.agents.contracts import (
    CommunicationContext,
    DecisionContext,
    ModelGeneration,
)
from game_theory_agent.cooperation.replay import verify_cooperation_replay
from game_theory_agent.experiments.canonical_validity_real import PRICE_SNAPSHOT
from game_theory_agent.experiments.cooperation_metrics import (
    compute_cooperation_metrics,
)
from game_theory_agent.experiments.persona_pilot import _model_client
from game_theory_agent.information import verify_information_replay
from game_theory_agent.interaction.replay import verify_interaction_replay
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.replay import verify_replay
from game_theory_agent.model_clients import BudgetedModelClient, MockModelClient
from game_theory_agent.orchestration import JsonlRoundEventLogger, RoundCoordinator
from game_theory_agent.strategic_reliability import RealModelBudget, RealModelCostGuard


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "market_v5_cooperation.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "cooperation-full-chain-real-v1"
COMPANIES = ("company_A", "company_B", "company_C", "company_D")
PROVIDERS = ("doubao", "deepseek")
SEEDS = (920_301, 920_302, 920_303)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


class AuditedRoundSelectiveModelClient:
    """Use the paid client only in one round and persist each successful boundary."""

    def __init__(
        self,
        *,
        paid_client: Any,
        fallback_client: Any,
        real_round: int,
        provider: str,
        seed: int,
        audit_path: Path,
    ) -> None:
        self.paid_client = paid_client
        self.fallback_client = fallback_client
        self.real_round = int(real_round)
        self.provider = provider
        self.seed = int(seed)
        self.audit_path = audit_path

    @property
    def model(self) -> str | None:
        return getattr(self.paid_client, "model", None)

    def _persist(
        self,
        phase: str,
        context: CommunicationContext | DecisionContext,
        result: ModelGeneration,
    ) -> None:
        _append_jsonl(
            self.audit_path,
            {
                "audit_schema_version": "real-model-boundary-v1.0.0",
                "provider": self.provider,
                "seed": self.seed,
                "episode_id": context.episode_id,
                "company_id": context.company_id,
                "round": context.round,
                "phase": phase,
                "model_name": result.model_name,
                "prompt_version": result.prompt_version,
                "raw_response": result.raw_response,
                "parsed_output": result.parsed_output,
                "input_tokens": int(result.input_tokens or 0),
                "output_tokens": int(result.output_tokens or 0),
                "retry_count": result.retry_count,
                "latency_ms": result.latency_ms,
                "provider_audit": (
                    result.provider_audit.model_dump(mode="json")
                    if result.provider_audit is not None
                    else None
                ),
            },
        )

    async def generate_communication(
        self, context: CommunicationContext
    ) -> ModelGeneration:
        if context.round != self.real_round:
            return await self.fallback_client.generate_communication(context)
        result = await self.paid_client.generate_communication(context)
        self._persist("communication", context, result)
        return result

    async def generate_decision(self, context: DecisionContext) -> ModelGeneration:
        if context.round != self.real_round:
            return await self.fallback_client.generate_decision(context)
        result = await self.paid_client.generate_decision(context)
        self._persist("decision", context, result)
        return result


def _make_context_builder(config_path: Path) -> DecisionContextBuilder:
    registry = load_persona_registry(config_path)
    return DecisionContextBuilder(
        persona_profile=registry.get("balanced_v1"),
        persona_registry=registry,
        cooperation_history_mode="full",
        cooperation_prompt_variant="neutral_numeric_v1",
    )


async def run(output: Path, *, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    output = output.resolve()
    config_path = config_path.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("output directory must be new and empty")
    output.mkdir(parents=True, exist_ok=True)

    # api.CONFIG is process-global, so select and validate v5 before importing it.
    os.environ["MARKET_CONFIG_PATH"] = str(config_path)
    from game_theory_agent.api import CONFIG, SESSIONS, app
    from game_theory_agent.experiments.four_agent_acceptance import (
        _ControllerAdapter,
        _GatewayAdapter,
    )

    file_config = load_market_config(config_path)
    if CONFIG.config_sha256 != file_config.config_sha256:
        raise RuntimeError("api.CONFIG was imported before the cooperation v5 config")
    if CONFIG.config_id != "market-v5-cooperation":
        raise RuntimeError("the real cooperation chain requires market-v5-cooperation")
    capability = CONFIG.to_dict()["persona_utilities"]["capabilities"]["cooperation"]
    if capability is not True:
        raise RuntimeError("cooperation persona capability is not enabled")

    manifest = {
        "schema_version": "cooperation-full-chain-real-manifest-v1.0.0",
        "config_path": str(config_path),
        "config_id": CONFIG.config_id,
        "config_version": CONFIG.config_version,
        "config_sha256": CONFIG.config_sha256,
        "environment_version": CONFIG.environment_version,
        "providers": list(PROVIDERS),
        "common_seeds": list(SEEDS),
        "episodes": len(PROVIDERS) * len(SEEDS),
        "planned_real_generation_tasks": len(PROVIDERS) * len(SEEDS) * 2,
        "real_phases": ["round_2_communication", "round_2_decision"],
        "proposal": {
            "sender": "company_A",
            "receiver": "company_B",
            "created_round": 1,
            "target_round": 2,
            "requested_contribution_cents": 1_000_000,
        },
        "persona": "balanced_v1",
        "cooperation_prompt_variant": "neutral_numeric_v1",
        "price_snapshot": PRICE_SNAPSHOT,
        "evidence_level": "native_two_round_real_model_chain",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    guard = RealModelCostGuard(
        RealModelBudget(
            max_calls=12,
            max_prompt_tokens=240_000,
            max_completion_tokens=48_000,
            max_estimated_cost_microunits=1_500_000,
        ),
        explicitly_authorized=True,
    )
    paid_clients: dict[str, Any] = {}
    for provider in PROVIDERS:
        price = PRICE_SNAPSHOT[provider]
        paid_clients[provider] = BudgetedModelClient(
            _model_client(provider, None, temperature=0.0, top_p=1.0),
            cost_guard=guard,
            reserved_prompt_tokens_per_call=20_000,
            reserved_completion_tokens_per_call=4_000,
            input_price_microunits_per_token=int(
                price["input_microunits_per_token"]
            ),
            output_price_microunits_per_token=int(
                price["output_microunits_per_token"]
            ),
        )

    SESSIONS.clear()
    rows: list[dict[str, Any]] = []
    boundary_path = output / "provider-boundaries.jsonl"
    with TestClient(app) as client:
        for provider in PROVIDERS:
            for repetition, seed in enumerate(SEEDS, start=1):
                episode_id = f"cooperation-full-chain-{provider}-{seed}"
                token = f"controller-{uuid.uuid4().hex}"
                os.environ["MARKET_CONTROLLER_TOKEN"] = token
                created_response = client.post(
                    "/api/episodes",
                    headers={"X-Controller-Token": token},
                    json={
                        "episode_id": episode_id,
                        "episode_seed": seed,
                        "company_ids": list(COMPANIES),
                        # The API supports 5/10/15/20 rounds.  We execute only
                        # two, so round 2 remains non-terminal and contributions
                        # can create a future public benefit.
                        "max_rounds": 5,
                        "market_model": "balanced",
                        "information_mode": "perfect",
                        "communication_mode": "public_private",
                        "cooperation_mode": "shared_resilience_v1",
                    },
                )
                created_response.raise_for_status()
                created = created_response.json()
                context_builder = _make_context_builder(config_path)
                b_client = AuditedRoundSelectiveModelClient(
                    paid_client=paid_clients[provider],
                    fallback_client=MockModelClient(
                        model_name=f"mock-{provider}-company_B-round1"
                    ),
                    real_round=2,
                    provider=provider,
                    seed=seed,
                    audit_path=boundary_path,
                )
                runtimes = {
                    "company_A": AgentRuntime(
                        f"mock-company_A-{seed}",
                        "company_A",
                        MockModelClient(
                            model_name="mock-proposer-v1",
                            cooperation_proposal_receiver="company_B",
                            cooperation_proposal_round=1,
                            cooperation_proposal_target_round=2,
                            cooperation_proposal_amount_cents=1_000_000,
                        ),
                        context_builder=_make_context_builder(config_path),
                    ),
                    "company_B": AgentRuntime(
                        f"real-{provider}-company_B-{seed}",
                        "company_B",
                        b_client,
                        memory=EpisodeMemory(),
                        context_builder=context_builder,
                    ),
                    **{
                        company_id: AgentRuntime(
                            f"mock-{company_id}-{seed}",
                            company_id,
                            MockModelClient(model_name=f"mock-{company_id}"),
                            context_builder=_make_context_builder(config_path),
                        )
                        for company_id in ("company_C", "company_D")
                    },
                }
                event_path = output / "episodes" / episode_id / "round-events.jsonl"
                coordinator = RoundCoordinator(
                    _ControllerAdapter(token),
                    _GatewayAdapter(created["agent_tokens"]),
                    runtimes,
                    event_logger=JsonlRoundEventLogger(event_path),
                )
                rounds = await coordinator.run_episode(episode_id, max_rounds=2)
                events = [item.event for item in rounds]
                session = SESSIONS[episode_id]
                interaction = verify_interaction_replay(events)
                information = verify_information_replay(events, session.manifest)
                cooperation = verify_cooperation_replay(events, MarketEnv(CONFIG))
                economic = verify_replay(
                    MarketEnv(CONFIG), session.manifest, session.transitions
                )
                persisted = list(JsonlRoundEventLogger(event_path).read_all())
                if persisted != events:
                    raise RuntimeError("persisted RoundEvents differ from memory")

                metrics = compute_cooperation_metrics(events)
                verifications = [
                    item
                    for record in cooperation
                    for item in record.verifications
                    if item.company_id == "company_B"
                ]
                contribution = int(
                    events[1].joint_action["company_B"].get(
                        "shared_resilience_contribution_cents", 0
                    )
                    or 0
                )
                credibility = session.cooperation_ledger.credibility()[
                    "company_B"
                ].model_dump(mode="json")
                communication_trace = next(
                    trace
                    for trace in events[1].communication_phase.generation_traces
                    if trace.company_id == "company_B"
                )
                decision_trace = next(
                    trace for trace in events[1].traces if trace.company_id == "company_B"
                )
                row = {
                    "provider": provider,
                    "repetition": repetition,
                    "seed": seed,
                    "episode_id": episode_id,
                    "proposal_count": metrics["proposal_count"],
                    "response_count": metrics["response_count"],
                    "acceptance_count": metrics["acceptance_count"],
                    "commitment_count": metrics["commitment_count"],
                    "actual_contribution_cents": contribution,
                    "verification": (
                        verifications[0].model_dump(mode="json")
                        if verifications
                        else None
                    ),
                    "credibility_after": credibility,
                    "round_2_industry_resilience_ppm": events[1].state_after[
                        "shared_resilience"
                    ]["industry_resilience_ppm"],
                    "communication_status": communication_trace.generation_status,
                    "communication_error": communication_trace.error_code,
                    "decision_status": decision_trace.decision_status,
                    "decision_error": decision_trace.error_code,
                    "replay": {
                        "economic_states": len(economic),
                        "interaction_rounds": len(interaction),
                        "information_snapshots": len(information),
                        "cooperation_rounds": len(cooperation),
                        "persisted_round_trip": persisted == events,
                    },
                }
                rows.append(row)
                _append_jsonl(output / "episode-results.jsonl", row)
                status = row["verification"]["status"] if row["verification"] else "none"
                print(
                    f"[{len(rows)}/6] {provider} seed={seed} "
                    f"accept={row['acceptance_count']} contribution={contribution} "
                    f"verification={status}",
                    flush=True,
                )

    boundary_rows = [
        json.loads(line)
        for line in boundary_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if boundary_path.exists() else []
    provider_summary: dict[str, Any] = {}
    for provider in PROVIDERS:
        selected = [row for row in rows if row["provider"] == provider]
        provider_summary[provider] = {
            "episodes": len(selected),
            "accepted": sum(row["acceptance_count"] == 1 for row in selected),
            "rejected_or_ignored": sum(row["acceptance_count"] == 0 for row in selected),
            "fulfilled": sum(
                row["verification"] is not None
                and row["verification"]["status"] == "fulfilled"
                for row in selected
            ),
            "partial_betrayal": sum(
                row["verification"] is not None
                and row["verification"]["status"] == "partial_betrayal"
                for row in selected
            ),
            "betrayed": sum(
                row["verification"] is not None
                and row["verification"]["status"] == "betrayed"
                for row in selected
            ),
            "mean_contribution_cents": (
                sum(row["actual_contribution_cents"] for row in selected)
                // len(selected)
            ),
        }
    summary = {
        "schema_version": "cooperation-full-chain-real-summary-v1.0.0",
        "evidence_level": manifest["evidence_level"],
        "episodes": len(rows),
        "provider_summary": provider_summary,
        "successful_real_generation_boundaries": len(boundary_rows),
        "provider_request_count_including_schema_repairs": sum(
            1 + int(row["retry_count"]) for row in boundary_rows
        ),
        "all_economic_replays_passed": all(
            row["replay"]["economic_states"] == 3 for row in rows
        ),
        "all_interaction_replays_passed": all(
            row["replay"]["interaction_rounds"] == 2 for row in rows
        ),
        "all_information_replays_passed": all(
            row["replay"]["information_snapshots"] == 16 for row in rows
        ),
        "all_cooperation_replays_passed": all(
            row["replay"]["cooperation_rounds"] == 2 for row in rows
        ),
        "all_logs_round_trip": all(
            row["replay"]["persisted_round_trip"] for row in rows
        ),
        "usage": asdict(guard.actual),
        "reserved_usage": asdict(guard.reserved),
        "estimated_cost_cny_conservative": round(
            guard.actual.estimated_cost_microunits / 1_000_000, 6
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--authorize-real-model", action="store_true")
    args = parser.parse_args()
    if not args.authorize_real_model:
        raise RuntimeError("--authorize-real-model is required")
    result = asyncio.run(run(args.output, config_path=args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
