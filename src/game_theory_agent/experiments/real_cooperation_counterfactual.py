"""Repeated fixed-state real-model causal inputs for cooperation decisions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from game_theory_agent.agents import (
    AgentRuntime,
    DecisionContextBuilder,
    EpisodeMemory,
    load_persona_registry,
)
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.experiments.persona_pilot import (
    PROJECT_ROOT,
    _model_client,
    _observation,
)
from game_theory_agent.interaction import (
    CommunicationRoundLedger,
    CommunicationSubmission,
    MessageDraft,
)
from game_theory_agent.cooperation import SharedResilienceProposalDraft
from game_theory_agent.market import MarketEnv, load_market_config


CONDITIONS = (
    "no_message",
    "cooperation_proposal",
    "defection_statement",
    "high_credibility_proposal",
    "low_credibility_proposal",
)
REQUESTED_CONTRIBUTION_CENTS = 1_000_000


def _credibility(company_id: str, score: int) -> dict[str, Any]:
    return {
        "credibility_schema_version": "credibility-v1.0.0",
        "company_id": company_id,
        "verified_commitment_count": 1 if score != 500_000 else 0,
        "fulfilled_count": 1 if score > 500_000 else 0,
        "partial_betrayal_count": 0,
        "betrayed_count": 1 if score < 500_000 else 0,
        "total_promised_contribution_cents": (
            1_000_000 if score != 500_000 else 0
        ),
        "total_actual_capped_contribution_cents": (
            1_000_000 if score > 500_000 else 0
        ),
        "credibility_ppm": score,
    }


def _submission(condition: str) -> CommunicationSubmission:
    if condition == "no_message":
        return CommunicationSubmission()
    if condition == "defection_statement":
        return CommunicationSubmission(
            messages=[
                MessageDraft(
                    channel="private",
                    recipients=["company_B"],
                    speech_act="statement",
                    content="我不会为共享韧性贡献；你可独立决定。",
                )
            ]
        )
    if condition in {
        "cooperation_proposal",
        "high_credibility_proposal",
        "low_credibility_proposal",
    }:
        return CommunicationSubmission(
            messages=[
                MessageDraft(
                    channel="private",
                    recipients=["company_B"],
                    speech_act="proposal",
                    content="请在下一轮贡献1000000分共享韧性。",
                    cooperation_proposal=SharedResilienceProposalDraft(
                        target_round=2,
                        requested_contribution_cents=(
                            REQUESTED_CONTRIBUTION_CENTS
                        ),
                    ),
                )
            ]
        )
    raise ValueError(f"unknown condition: {condition}")


def build_condition_observation(
    base: dict[str, Any], *, condition: str
) -> dict[str, Any]:
    observation = json.loads(json.dumps(base))
    observation["communication_mode"] = "public_private"
    observation["cooperation_mode"] = "shared_resilience_v1"
    ledger = CommunicationRoundLedger(
        episode_id=observation["episode_id"],
        round_number=observation["round"],
        state_version=observation["state_version"],
        state_hash=observation["state_hash"],
        company_ids=("company_A", "company_B", "company_C", "company_D"),
        mode="public_private",
    )
    submission = _submission(condition)
    if submission.messages:
        ledger.submit("company_A", submission)
    view = ledger.close().views["company_B"]
    observation["communication_view"] = view.model_dump(mode="json")
    observation["communication_history"] = []
    proposals = [
        message.cooperation_proposal.model_dump(mode="json")
        for message in view.visible_messages
        if message.cooperation_proposal is not None
    ]
    score = (
        900_000
        if condition == "high_credibility_proposal"
        else 100_000
        if condition == "low_credibility_proposal"
        else 500_000
    )
    observation["cooperation"] = {
        "cooperation_view_schema_version": "cooperation-view-v1.0.0",
        "mode": "shared_resilience_v1",
        "round": observation["round"],
        "proposals_sent": [],
        "proposals_received": proposals,
        "pending_proposals_received": proposals,
        "responses": [],
        "active_commitments": [],
        "commitment_history": [],
        "public_credibility": {
            company_id: _credibility(
                company_id, score if company_id == "company_A" else 500_000
            )
            for company_id in ("company_A", "company_B", "company_C", "company_D")
        },
        "commitments_are_non_binding": True,
    }
    return observation


def _condition_order(repetition: int) -> tuple[str, ...]:
    shift = (repetition - 1) % len(CONDITIONS)
    return CONDITIONS[shift:] + CONDITIONS[:shift]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config_path = Path(
        os.environ.get(
            "MARKET_CONFIG_PATH", PROJECT_ROOT / "configs" / "market_v4.yaml"
        )
    )
    config = load_market_config(config_path)
    registry = load_persona_registry(config_path)
    profile = registry.get(args.persona)
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B", "company_C", "company_D"),
        episode_id=f"cooperation-counterfactual-{args.seed}",
        episode_seed=args.seed,
        market_model="balanced",
        max_rounds=5,
        cooperation_mode="shared_resilience_v1",
    )
    base = _observation(config, state, "company_B")
    base["shared_resilience"] = state.shared_resilience.to_dict()
    client = _model_client(
        args.provider,
        args.model,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    rows: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for condition in _condition_order(repetition):
            observation = build_condition_observation(base, condition=condition)
            runtime = AgentRuntime(
                f"cooperation-counterfactual-{args.provider}-B",
                "company_B",
                client,
                memory=EpisodeMemory(),
                context_builder=DecisionContextBuilder(
                    persona_profile=profile,
                    persona_registry=registry,
                    cooperation_history_mode="full",
                ),
            )
            result = await runtime.decide(
                observation, timeout_seconds=args.timeout
            )
            row: dict[str, Any] = {
                "repetition": repetition,
                "condition": condition,
                "call_order": len(rows) + 1,
                "state_hash": state.state_hash,
                "success": result.success,
                "error_code": result.error_code,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
            if result.success and result.decision is not None:
                requested = result.decision.requested_action.model_dump(mode="json")
                resolved = resolve_action_request(
                    config,
                    state,
                    "company_B",
                    requested,
                    source=f"cooperation-counterfactual:{condition}",
                )
                contribution = int(
                    resolved.action.shared_resilience_contribution_cents or 0
                )
                row.update(
                    {
                        "message_responses": [
                            item.model_dump(mode="json")
                            for item in result.decision.message_responses
                        ],
                        "requested_action": requested,
                        "final_action": resolved.action.to_dict(),
                        "contribution_cents": contribution,
                        "proposal_target_alignment_ppm": min(
                            1_000_000,
                            contribution
                            * 1_000_000
                            // REQUESTED_CONTRIBUTION_CENTS,
                        ),
                    }
                )
            rows.append(row)
    grouped = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in CONDITIONS
    }
    summary = {
        "experiment_schema_version": "real-cooperation-counterfactual-v1.0.0",
        "provider": args.provider,
        "model": args.model,
        "persona": args.persona,
        "seed": args.seed,
        "repetitions": args.repetitions,
        "conditions": list(CONDITIONS),
        "all_conditions_share_frozen_state": len(
            {row["state_hash"] for row in rows}
        )
        == 1,
        "all_calls_successful": all(row["success"] for row in rows),
        "condition_distributions": {
            condition: {
                "contribution_cents": [
                    row.get("contribution_cents") for row in condition_rows
                ],
                "proposal_target_alignment_ppm": [
                    row.get("proposal_target_alignment_ppm")
                    for row in condition_rows
                ],
                "decision_message_dispositions": [
                    response["disposition"]
                    for row in condition_rows
                    for response in row.get("message_responses", [])
                ],
            }
            for condition, condition_rows in grouped.items()
        },
        "causal_scope": (
            "Repeated provider calls on one frozen market state. High/low "
            "credibility are explicit trusted-history treatments; provider "
            "sampling remains stochastic."
        ),
        "rows": rows,
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("deepseek", "doubao"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--persona", default="balanced_v1")
    parser.add_argument("--seed", type=int, default=810)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["all_calls_successful"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
