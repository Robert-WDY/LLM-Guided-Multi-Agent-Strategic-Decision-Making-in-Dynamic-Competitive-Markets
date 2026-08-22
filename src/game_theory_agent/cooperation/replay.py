"""Deterministic replay for the Shared Resilience cooperation protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from game_theory_agent.cooperation.contracts import (
    CooperativeBenefitAttribution,
    CooperationRoundRecord,
    apply_cooperation_history_mode,
)
from game_theory_agent.cooperation.ledger import (
    CooperationLedger,
    CooperationProtocolError,
)
from game_theory_agent.interaction.replay import verify_interaction_replay
from game_theory_agent.market.environment import MarketEnv
from game_theory_agent.market.models import MarketState


class CooperationReplayMismatchError(RuntimeError):
    """Recorded cooperation evidence cannot be rebuilt from authoritative input."""


def _fail(message: str) -> None:
    raise CooperationReplayMismatchError(message)


def _verify_context_view(
    raw_context: dict[str, Any] | None,
    authoritative: dict[str, Any],
    *,
    label: str,
) -> None:
    if raw_context is None:
        return
    history_mode = str(raw_context.get("cooperation_history_mode", "full"))
    if history_mode not in {"full", "none"}:
        _fail(f"{label} has an unknown cooperation history mode")
    expected = apply_cooperation_history_mode(
        authoritative,
        round_number=int(raw_context["meta"]["round"]),
        history_mode=history_mode,
    )
    if raw_context.get("cooperation") != expected:
        _fail(f"{label} cooperation view does not match authoritative ledger")


def verify_cooperation_replay(
    events: Sequence[Any],
    market_env: MarketEnv | None = None,
) -> tuple[CooperationRoundRecord, ...]:
    """Rebuild proposal→commitment→verification records for one episode.

    Model calls are intentionally not repeated. The replay authority is the
    recorded closed communication batch plus each settled final joint action.
    """

    records = list(events)
    if not records:
        return ()
    verify_interaction_replay(records)
    first = records[0]
    first_before = MarketState.from_dict(first.state_before)
    if first_before.shared_resilience is None:
        if any(getattr(event, "cooperation_round", None) is not None for event in records):
            _fail("off-mode event unexpectedly contains a cooperation record")
        return ()
    if first_before.round != 1:
        _fail("strict cooperation replay must start at round 1")

    ledger = CooperationLedger(
        mode="shared_resilience_v1",
        episode_id=first_before.episode_id,
        company_ids=first_before.company_ids,
        max_rounds=first_before.max_rounds,
    )
    rebuilt_records: list[CooperationRoundRecord] = []
    expected_round = 1
    for event in records:
        before = MarketState.from_dict(event.state_before)
        after = MarketState.from_dict(event.state_after)
        if before.episode_id != first_before.episode_id:
            _fail("cooperation replay cannot span episodes")
        if event.settled_round != expected_round or before.round != expected_round:
            _fail("cooperation rounds are not contiguous")
        if before.shared_resilience is None or after.shared_resilience is None:
            _fail("cooperation event is missing shared resilience market state")
        phase = getattr(event, "communication_phase", None)
        recorded = getattr(event, "cooperation_round", None)
        if phase is None or recorded is None:
            _fail("cooperation event is missing its close or round record")
        include_memory = (
            recorded.round_schema_version != "cooperation-round-v1.0.0"
        )
        preclose_views = {
            company_id: ledger.company_view(
                company_id,
                round_number=expected_round,
                include_memory=include_memory,
            )
            for company_id in before.company_ids
        }
        for generation in phase.generation_traces:
            if generation.generation_status == "disabled":
                continue
            _verify_context_view(
                generation.communication_context,
                preclose_views[generation.company_id],
                label=f"round {expected_round} communication context",
            )
        try:
            rebuilt_close = ledger.close_round(phase.closure)
        except CooperationProtocolError as exc:
            _fail(f"cooperation close cannot be rebuilt: {exc}")
        if rebuilt_close != recorded.close:
            _fail("recorded cooperation close does not match rebuilt close")
        postclose_views = {
            company_id: ledger.company_view(
                company_id,
                round_number=expected_round,
                include_memory=include_memory,
            )
            for company_id in before.company_ids
        }
        for trace in event.traces:
            authoritative_view = postclose_views[trace.company_id]
            if trace.observation is not None and trace.observation.get(
                "cooperation"
            ) != authoritative_view:
                _fail(
                    f"round {expected_round} observation cooperation view mismatch"
                )
            _verify_context_view(
                trace.decision_context,
                authoritative_view,
                label=f"round {expected_round} decision context",
            )

        contributions = {
            company_id: int(
                event.joint_action[company_id].get(
                    "shared_resilience_contribution_cents", 0
                )
                or 0
            )
            for company_id in before.company_ids
        }
        if contributions != recorded.contribution_by_company_cents:
            _fail("cooperation contribution attribution does not match final actions")
        if (
            after.shared_resilience.last_total_contribution_cents
            != sum(contributions.values())
            or dict(after.shared_resilience.last_contribution_by_company_cents)
            != contributions
        ):
            _fail("market shared resilience attribution does not match final actions")
        if (
            recorded.industry_resilience_before_ppm
            != before.shared_resilience.industry_resilience_ppm
            or recorded.industry_resilience_after_ppm
            != after.shared_resilience.industry_resilience_ppm
        ):
            _fail("cooperation record does not match public resilience state")
        attributions = recorded.benefit_attribution_by_company
        if recorded.round_schema_version == "cooperation-round-v1.1.0":
            if market_env is None:
                _fail(
                    "v1.1 cooperation replay requires a market environment "
                    "to rebuild public-benefit counterfactuals"
                )
            if set(attributions) != set(before.company_ids):
                _fail("cooperation benefit attribution is incomplete")
            shadow = market_env.counterfactual_without_public_resilience(
                before, event.joint_action
            )
            expected_attributions = {}
            for company_id, attribution in attributions.items():
                if attribution.actual_round_profit_cents != (
                    after.company(company_id).financial.round_profit_cents
                ):
                    _fail("benefit attribution does not match settled profit")
                if attribution.public_protection_received_ppm != (
                    recorded.public_protection_applied_ppm
                ):
                    _fail("benefit attribution protection binding mismatch")
                if recorded.public_protection_applied_ppm == 0 and (
                    attribution.counterfactual_profit_delta_cents != 0
                    or attribution.avoided_next_incident
                ):
                    _fail("zero public protection cannot create an attributed benefit")
                actual_incident = after.company(company_id).risk.active_incident
                shadow_incident = shadow.state_after.company(
                    company_id
                ).risk.active_incident
                expected_attributions[company_id] = (
                    CooperativeBenefitAttribution.from_counterfactual(
                        company_id=company_id,
                        current_contribution_cost_cents=contributions[company_id],
                        latest_source_contribution_cents=int(
                            dict(
                                before.shared_resilience.last_contribution_by_company_cents
                            ).get(company_id, 0)
                        ),
                        public_protection_received_ppm=(
                            recorded.public_protection_applied_ppm
                        ),
                        actual_round_profit_cents=(
                            after.company(
                                company_id
                            ).financial.round_profit_cents
                        ),
                        no_public_protection_round_profit_cents=(
                            shadow.state_after.company(
                                company_id
                            ).financial.round_profit_cents
                        ),
                        avoided_next_incident=(
                            actual_incident is None
                            and shadow_incident is not None
                        ),
                    )
                )
            if expected_attributions != attributions:
                _fail("public-benefit counterfactual attribution mismatch")
        rebuilt = ledger.settle_round(
            round_number=expected_round,
            final_actions=event.joint_action,
            industry_resilience_before_ppm=(
                before.shared_resilience.industry_resilience_ppm
            ),
            public_protection_applied_ppm=(
                recorded.public_protection_applied_ppm
            ),
            industry_resilience_after_ppm=(
                after.shared_resilience.industry_resilience_ppm
            ),
            benefit_attribution_by_company=attributions,
        )
        if rebuilt != recorded:
            _fail("commitment, fulfillment, or credibility reconstruction mismatch")
        rebuilt_records.append(rebuilt)
        expected_round += 1
    return tuple(rebuilt_records)


__all__ = [
    "CooperationReplayMismatchError",
    "verify_cooperation_replay",
]
