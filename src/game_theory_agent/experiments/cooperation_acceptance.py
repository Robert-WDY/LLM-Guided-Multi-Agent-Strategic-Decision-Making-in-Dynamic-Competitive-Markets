"""Runnable Cooperation MVP v1 engineering acceptance."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path

from dotenv import dotenv_values
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ENV = dotenv_values(PROJECT_ROOT / ".env")
if _PROJECT_ENV.get("MARKET_CONFIG_PATH"):
    os.environ.setdefault(
        "MARKET_CONFIG_PATH", str(_PROJECT_ENV["MARKET_CONFIG_PATH"])
    )

from game_theory_agent.agents import AgentRuntime
from game_theory_agent.api import CONFIG, SESSIONS, app
from game_theory_agent.cooperation.replay import verify_cooperation_replay
from game_theory_agent.experiments.cooperation_metrics import (
    compute_cooperation_metrics,
)
from game_theory_agent.experiments.four_agent_acceptance import (
    _ControllerAdapter,
    _GatewayAdapter,
)
from game_theory_agent.interaction.replay import verify_interaction_replay
from game_theory_agent.information import verify_information_replay
from game_theory_agent.market import CompanyAction, MarketEnv
from game_theory_agent.market.models import MarketEvent
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.market.replay import verify_replay
from game_theory_agent.model_clients import MockModelClient
from game_theory_agent.orchestration import JsonlRoundEventLogger, RoundCoordinator


COMPANIES = ("company_A", "company_B", "company_C", "company_D")


def _actions(state: Any, contributions: dict[str, int]) -> dict[str, CompanyAction]:
    return {
        company_id: CompanyAction(
            action_id=f"{state.episode_id}:{state.round}:{company_id}",
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=10_000,
            shared_resilience_contribution_cents=contributions.get(company_id, 0),
        )
        for company_id in state.company_ids
    }


def _market_mechanism_checks(seed: int) -> dict[str, bool]:
    cooperate_env = MarketEnv(CONFIG)
    defect_env = MarketEnv(CONFIG)
    free_ride_env = MarketEnv(CONFIG)
    cooperate = cooperate_env.reset(
        COMPANIES,
        episode_id=f"mechanism-cooperate-{seed}",
        episode_seed=seed,
        max_rounds=5,
        cooperation_mode="shared_resilience_v1",
    )
    defect = defect_env.reset(
        COMPANIES,
        episode_id=f"mechanism-defect-{seed}",
        episode_seed=seed,
        max_rounds=5,
        cooperation_mode="shared_resilience_v1",
    )
    free_ride = free_ride_env.reset(
        COMPANIES,
        episode_id=f"mechanism-free-ride-{seed}",
        episode_seed=seed,
        max_rounds=5,
        cooperation_mode="shared_resilience_v1",
    )
    all_contributions = {company_id: 1_000_000 for company_id in COMPANIES}
    others_contribute = {
        company_id: 1_000_000 for company_id in COMPANIES if company_id != "company_A"
    }
    cooperate_after = cooperate_env.step(
        f"{cooperate.episode_id}:1:0", _actions(cooperate, all_contributions)
    ).state_after
    defect_after = defect_env.step(
        f"{defect.episode_id}:1:0", _actions(defect, {})
    ).state_after
    free_ride_after = free_ride_env.step(
        f"{free_ride.episode_id}:1:0", _actions(free_ride, others_contribute)
    ).state_after
    event = MarketEvent(
        event_id="acceptance-disaster",
        event_type="supply_disruption",
        severity="high",
        started_round=2,
        remaining_rounds=1,
        demand_multiplier_ppm=900_000,
        supply_cost_multiplier_ppm=1_700_000,
        capacity_multiplier_ppm=500_000,
        advertising_multiplier_ppm=700_000,
        service_penalty_ppm=250_000,
        reputation_penalty_ppm=150_000,
    )

    def with_event(state: Any) -> Any:
        changed = replace(state, active_market_events=(event,), state_hash="")
        return replace(changed, state_hash=state_hash(changed.to_dict()))

    protected = with_event(cooperate_after)
    unprotected = with_event(defect_after)
    cooperate_env.load_state(protected)
    defect_env.load_state(unprotected)
    protected_after = cooperate_env.step(
        f"{protected.episode_id}:2:1", _actions(protected, {})
    ).state_after
    unprotected_after = defect_env.step(
        f"{unprotected.episode_id}:2:1", _actions(unprotected, {})
    ).state_after
    return {
        "all_cooperate_pays_private_cost": all(
            defect_after.company(company_id).financial.round_profit_cents
            - cooperate_after.company(company_id).financial.round_profit_cents
            == 1_000_000
            for company_id in COMPANIES
        ),
        "all_cooperate_builds_public_resilience": (
            cooperate_after.shared_resilience.industry_resilience_ppm > 0
        ),
        "all_defect_builds_no_new_resilience": (
            defect_after.shared_resilience.industry_resilience_ppm == 0
        ),
        "free_rider_saves_private_cost": (
            free_ride_after.company("company_A").financial.round_profit_cents
            - cooperate_after.company("company_A").financial.round_profit_cents
            == 1_000_000
        ),
        "free_rider_receives_public_resilience": (
            free_ride_after.shared_resilience.industry_resilience_ppm > 0
        ),
        "public_resilience_reduces_disaster_loss": (
            sum(
                company.financial.round_profit_cents
                for company in protected_after.companies
            )
            > sum(
                company.financial.round_profit_cents
                for company in unprotected_after.companies
            )
        ),
    }


async def run(output: Path, *, seed: int = 20260821) -> dict:
    output = output.resolve()
    event_path = output / "round-events.jsonl"
    if event_path.exists() and event_path.stat().st_size:
        raise ValueError(f"refusing to append to non-empty log: {event_path}")
    output.mkdir(parents=True, exist_ok=True)
    token = f"cooperation-acceptance-{uuid.uuid4().hex}"
    os.environ["MARKET_CONTROLLER_TOKEN"] = token
    SESSIONS.clear()
    episode_id = f"cooperation-acceptance-{seed}"
    created_response = TestClient(app).post(
        "/api/episodes",
        headers={"X-Controller-Token": token},
        json={
            "episode_id": episode_id,
            "episode_seed": seed,
            "company_ids": list(COMPANIES),
            "max_rounds": 5,
            "market_model": "balanced",
            "information_mode": "perfect",
            "communication_mode": "public_private",
            "cooperation_mode": "shared_resilience_v1",
        },
    )
    created_response.raise_for_status()
    created = created_response.json()
    runtimes = {
        "company_A": AgentRuntime(
            "mock-company_A",
            "company_A",
            MockModelClient(
                model_name="mock-company_A",
                cooperation_proposal_receiver="company_B",
                cooperation_proposal_round=1,
                cooperation_proposal_target_round=2,
                cooperation_proposal_amount_cents=1_000_000,
            ),
        ),
        "company_B": AgentRuntime(
            "mock-company_B",
            "company_B",
            MockModelClient(
                model_name="mock-company_B",
                cooperation_response="accept",
                shared_resilience_contribution_cents=300_000,
                shared_resilience_contribution_rounds=(2,),
            ),
        ),
        **{
            company_id: AgentRuntime(
                f"mock-{company_id}",
                company_id,
                MockModelClient(model_name=f"mock-{company_id}"),
            )
            for company_id in ("company_C", "company_D")
        },
    }
    coordinator = RoundCoordinator(
        _ControllerAdapter(token),
        _GatewayAdapter(created["agent_tokens"]),
        runtimes,
        event_logger=JsonlRoundEventLogger(event_path),
    )
    rounds = await coordinator.run_episode(episode_id)
    events = [item.event for item in rounds]
    session = SESSIONS[episode_id]
    interaction = verify_interaction_replay(events)
    information = verify_information_replay(events, session.manifest)
    cooperation = verify_cooperation_replay(events, MarketEnv(CONFIG))
    economic = verify_replay(MarketEnv(CONFIG), session.manifest, session.transitions)
    loaded = list(JsonlRoundEventLogger(event_path).read_all())
    if loaded != events:
        raise RuntimeError("persisted round events differ from in-memory events")
    metrics = compute_cooperation_metrics(events)
    partial = [
        verification
        for record in cooperation
        for verification in record.verifications
        if verification.status == "partial_betrayal"
    ]
    round_three_a = next(
        trace
        for trace in events[2].traces
        if trace.company_id == "company_A"
    )
    company_b_memory = round_three_a.decision_context["cooperation"][
        "cooperation_memory"
    ]["company_B"]
    checks = {
        "five_rounds_complete": len(events) == 5 and events[-1].state_after["terminal"],
        "one_private_proposal": metrics["proposal_count"] == 1,
        "one_attributed_acceptance": metrics["acceptance_count"] == 1,
        "one_non_binding_commitment": metrics["commitment_count"] == 1,
        "actual_contribution_attributed": any(
            record.contribution_by_company_cents.get("company_B") == 300_000
            for record in cooperation
        ),
        "partial_betrayal_rebuilt": (
            len(partial) == 1 and partial[0].fulfillment_ratio_ppm == 300_000
        ),
        "company_benefit_attribution_complete": all(
            set(record.benefit_attribution_by_company) == set(COMPANIES)
            for record in cooperation
        ),
        "cooperation_memory_rebuilt": (
            company_b_memory["commitments_by_opponent"] == 1
            and company_b_memory["partial_betrayals_by_opponent"] == 1
            and company_b_memory["promised_by_opponent_cents"] == 1_000_000
            and company_b_memory["fulfilled_by_opponent_cents"] == 300_000
        ),
        "economic_replay_100_percent": (
            len(economic) == 6
            and economic[-1].state_hash == session.env.get_state().state_hash
        ),
        "interaction_replay_100_percent": len(interaction) == 5,
        "information_replay_100_percent": len(information) == 40,
        "cooperation_replay_100_percent": len(cooperation) == 5,
        "persisted_log_round_trip": loaded == events,
        **_market_mechanism_checks(seed),
    }
    summary = {
        "acceptance_schema_version": "cooperation-acceptance-v1.1.0",
        "episode_id": episode_id,
        "seed": seed,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "artifacts": {"round_events": "round-events.jsonl"},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="runs/cooperation-mvp-v1-acceptance"
    )
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    summary = asyncio.run(run(Path(args.output), seed=args.seed))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
