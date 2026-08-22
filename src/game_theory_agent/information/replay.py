"""Deterministic TrueState -> VisibilityPolicy -> Observation replay."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.information.contracts import (
    ObservationSnapshot,
    compute_observation_hash,
)
from game_theory_agent.market.models import MarketState
from game_theory_agent.market.protocols import state_hash


class InformationReplayMismatchError(RuntimeError):
    """A recorded Agent view cannot be reproduced from its true state."""


def _fail(message: str) -> None:
    raise InformationReplayMismatchError(
        f"information replay mismatch: {message}"
    )


OBSERVATION_V17_FIELDS = {
    "observation_schema_version",
    "episode_id",
    "round",
    "decision_round",
    "last_settled_round",
    "rounds_remaining",
    "state_version",
    "state_hash",
    "terminal",
    "episode_config",
    "information_mode",
    "visibility_policy_version",
    "visibility_policy",
    "belief_schema_version",
    "belief_hash",
    "belief_state",
    "opponent_model_hash",
    "opponent_model_state",
    "utility_inference_hash",
    "utility_inference_state",
    "communication_mode",
    "cooperation_mode",
    "public_state",
    "private_state",
    "market",
    "shared_resilience",
    "market_regime",
    "decision_support",
    "risk_signals",
    "active_market_events",
    "public_companies",
    "competitors",
    "public_history",
    "own_company",
    "company_analysis",
    "action_constraints",
    "communication_view",
    "communication_history",
    "cooperation",
    "repeated_game_strategy_hash",
    "repeated_game_strategy",
    "terminal_summary",
    "game_theory_advice",
    "observation_hash",
}


def _verify_public_extensions(observation: dict[str, Any]) -> None:
    if observation.get("information_mode") != "public":
        return
    if set(observation) - OBSERVATION_V17_FIELDS:
        _fail("public observation contains unregistered top-level fields")
    regime = observation.get("market_regime", {})
    if regime.get("capacity") != "unknown":
        _fail("public market regime disclosed private capacity classification")
    regime_metrics = regime.get("metrics", {})
    if {
        "average_capacity_utilization_ppm",
        "price_anchor_cents",
    } & set(regime_metrics):
        _fail("public market regime disclosed controller-only metrics")
    policy = observation["visibility_policy"]
    public_company_fields = set(policy["public_company_fields"])
    public_market_fields = set(policy["public_market_fields"])
    public_event_fields = set(policy["public_event_fields"])
    for history in observation.get("public_history", []):
        if set(history.get("market", {})) - public_market_fields:
            _fail("public history disclosed controller-only market fields")
        for company in history.get("public_companies", []):
            if set(company) != public_company_fields:
                _fail("public history disclosed private company fields")
        for event in history.get("active_events_during_round", []):
            if set(event) != public_event_fields:
                _fail("public history disclosed private event multipliers")


def verify_information_snapshot(
    true_state: MarketState,
    snapshot: ObservationSnapshot,
    *,
    decision_context: dict[str, Any] | None = None,
) -> ObservationSnapshot:
    if state_hash(true_state.to_dict()) != true_state.state_hash:
        _fail("recorded true state hash is invalid")
    if (
        snapshot.episode_id != true_state.episode_id
        or snapshot.round != true_state.round
        or snapshot.state_version != true_state.state_version
        or snapshot.state_hash != true_state.state_hash
    ):
        _fail(f"snapshot true-state binding mismatch for {snapshot.company_id}")
    if compute_observation_hash(snapshot.observation) != snapshot.observation_hash:
        _fail(f"observation hash mismatch for {snapshot.company_id}")
    try:
        expected = ObservationBuilder().build(
            true_state,
            snapshot.company_id,
            snapshot.information_mode,
            belief_state=snapshot.observation.get("belief_state"),
            belief_hash=snapshot.belief_hash,
            belief_schema_version=snapshot.belief_schema_version,
        )
    except (KeyError, ValueError) as exc:
        _fail(f"cannot rebuild view for {snapshot.company_id}: {exc}")

    observation = snapshot.observation
    _verify_public_extensions(observation)
    exact_fields = (
        "information_mode",
        "visibility_policy_version",
        "visibility_policy",
        "belief_schema_version",
        "belief_hash",
        "public_state",
        "private_state",
        "own_company",
        "competitors",
        "public_companies",
        "market",
        "shared_resilience",
        "risk_signals",
        "active_market_events",
    )
    for field_name in exact_fields:
        if observation.get(field_name) != expected[field_name]:
            _fail(
                f"{field_name} differs from visibility policy for "
                f"{snapshot.company_id}"
            )

    if decision_context is not None:
        meta = decision_context.get("meta", {})
        identity = decision_context.get("identity", {})
        if identity.get("company_id") != snapshot.company_id:
            _fail(f"decision context identity mismatch for {snapshot.company_id}")
        for field_name in (
            "episode_id",
            "round",
            "state_version",
            "state_hash",
            "information_mode",
            "observation_hash",
            "belief_hash",
            "opponent_model_hash",
            "utility_inference_hash",
            "repeated_game_strategy_hash",
        ):
            expected_value = observation.get(field_name)
            if meta.get(field_name) != expected_value:
                _fail(
                    f"decision context {field_name} mismatch for "
                    f"{snapshot.company_id}"
                )
        context_fields = (
            "market",
            "shared_resilience",
            "own_company",
            "competitors",
            "risk_signals",
            "active_market_events",
            "belief_state",
            "game_theory_advice",
            "opponent_model_state",
            "utility_inference_state",
            "repeated_game_strategy",
        )
        is_communication_context = str(
            decision_context.get("context_schema_version", "")
        ).startswith("communication-context-")
        for field_name in context_fields:
            # Communication generation intentionally receives a smaller
            # context than private economic decision generation. Strategic
            # advice enters the latter, not the cheap-talk prompt.
            if is_communication_context and field_name not in decision_context:
                continue
            if decision_context.get(field_name) != observation.get(field_name):
                _fail(
                    f"decision context {field_name} is not the recorded view "
                    f"for {snapshot.company_id}"
                )
    return snapshot


def verify_information_replay(
    events: Sequence[Any],
    manifest: Any | None = None,
) -> tuple[ObservationSnapshot, ...]:
    def expected_treatment(company_id: str) -> tuple[str, str] | None:
        if manifest is None:
            return None
        if hasattr(manifest, "information_mode_for"):
            mode = manifest.information_mode_for(company_id)
        else:
            mode = dict(
                getattr(manifest, "observer_information_modes", ())
            ).get(company_id, manifest.information_mode)
        policy_version = (
            "visibility-perfect-v1.0.0"
            if mode == "perfect"
            else "visibility-public-v2.0.0"
        )
        return mode, policy_version

    snapshots: list[ObservationSnapshot] = []
    for event in events:
        true_state = MarketState.from_dict(event.state_before)
        strict = event.event_schema_version in {
            "agent-round-event-v1.7.0",
            "agent-round-event-v1.8.0",
            "agent-round-event-v1.9.0",
        }
        public_state: dict[str, Any] | None = None
        phase = getattr(event, "communication_phase", None)
        if phase is not None:
            for generation in phase.generation_traces:
                raw_snapshot = getattr(
                    generation, "information_snapshot", None
                )
                requires_snapshot = (
                    strict
                    and generation.generation_status
                    not in {"disabled", "not_applicable"}
                )
                if raw_snapshot is None:
                    if requires_snapshot:
                        _fail(
                            "strict event is missing communication snapshot "
                            f"for {generation.company_id}"
                        )
                    continue
                snapshot = (
                    raw_snapshot
                    if isinstance(raw_snapshot, ObservationSnapshot)
                    else ObservationSnapshot.model_validate(raw_snapshot)
                )
                treatment = expected_treatment(snapshot.company_id)
                if treatment is not None and (
                    snapshot.information_mode != treatment[0]
                    or snapshot.visibility_policy_version != treatment[1]
                ):
                    _fail(
                        "communication snapshot treatment differs from manifest"
                    )
                if generation.observation_hash != snapshot.observation_hash:
                    _fail(
                        "communication observation hash mismatch for "
                        f"{generation.company_id}"
                    )
                verify_information_snapshot(
                    true_state,
                    snapshot,
                    decision_context=generation.communication_context,
                )
                current_public = snapshot.observation["public_state"]
                if public_state is None:
                    public_state = current_public
                elif current_public != public_state:
                    _fail("companies received inconsistent public state")
                snapshots.append(snapshot)
        for trace in event.traces:
            raw_snapshot = getattr(trace, "information_snapshot", None)
            if raw_snapshot is None:
                if strict and trace.observation is not None:
                    _fail(
                        f"strict event is missing snapshot for {trace.company_id}"
                    )
                continue
            snapshot = (
                raw_snapshot
                if isinstance(raw_snapshot, ObservationSnapshot)
                else ObservationSnapshot.model_validate(raw_snapshot)
            )
            treatment = expected_treatment(snapshot.company_id)
            if treatment is not None and (
                snapshot.information_mode != treatment[0]
                or snapshot.visibility_policy_version != treatment[1]
            ):
                _fail("decision snapshot treatment differs from manifest")
            if trace.observation != snapshot.observation:
                _fail(f"trace observation mismatch for {trace.company_id}")
            if trace.observation_hash != snapshot.observation_hash:
                _fail(f"trace observation hash mismatch for {trace.company_id}")
            verify_information_snapshot(
                true_state,
                snapshot,
                decision_context=trace.decision_context,
            )
            current_public = snapshot.observation["public_state"]
            if public_state is None:
                public_state = current_public
            elif current_public != public_state:
                _fail("companies received inconsistent public state")
            snapshots.append(snapshot)
    return tuple(snapshots)
