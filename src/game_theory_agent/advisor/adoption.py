"""Auditable comparison between non-binding advice and an LLM decision."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.market.protocols import sha256_hash


AdoptionStatus = Literal[
    "exact_action",
    "accepted_target",
    "partial",
    "rejected",
    "unavailable",
]


class AdvisorAdoptionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_schema_version: Literal["advisor-adoption-trace-v1.0.0"] = (
        "advisor-adoption-trace-v1.0.0"
    )
    advisor_mode: str
    advisor_candidate_id: str | None = None
    advisor_action: dict[str, Any]
    llm_requested_action: dict[str, Any]
    final_action: dict[str, Any]
    accepted: bool
    adoption_status: AdoptionStatus
    action_alignment_ppm: int = Field(ge=0, le=1_000_000)
    target_alignment_ppm: int = Field(ge=0, le=1_000_000)
    advisor_rank: int | None = Field(default=None, ge=1)
    chosen_candidate_id: str | None = None
    chosen_rank: int | None = Field(default=None, ge=1)
    agent_reason: str = Field(default="", max_length=1700)
    classification_method: Literal[
        "exact-economic-fields-and-target-direction-v1"
    ] = "exact-economic-fields-and-target-direction-v1"
    trace_hash: str

    @model_validator(mode="after")
    def validate_status_and_hash(self) -> "AdvisorAdoptionTrace":
        if self.accepted != (
            self.adoption_status in {"exact_action", "accepted_target"}
        ):
            raise ValueError("advisor adoption accepted/status mismatch")
        if self.trace_hash != compute_adoption_trace_hash(self):
            raise ValueError("advisor adoption trace hash mismatch")
        return self


def compute_adoption_trace_hash(
    trace: AdvisorAdoptionTrace | Mapping[str, Any],
) -> str:
    payload = (
        trace.model_dump(mode="json")
        if isinstance(trace, AdvisorAdoptionTrace)
        else dict(trace)
    )
    payload.pop("trace_hash", None)
    return sha256_hash(
        {
            "hash_protocol_version": "advisor-adoption-trace-hash-v1.0.0",
            "trace": payload,
        }
    )


def _economic_action(action: Mapping[str, Any]) -> dict[str, Any]:
    incident = action.get("incident_response")
    if not isinstance(incident, Mapping):
        incident = {}
    return {
        "price_cents": int(action.get("price_cents", 0)),
        "advertising_budget_cents": int(
            action.get("advertising_budget_cents", 0)
        ),
        "service_budget_cents": int(action.get("service_budget_cents", 0)),
        "capacity_investment_cents": int(
            action.get("capacity_investment_cents", 0)
        ),
        "resilience_budget_cents": int(
            action.get("resilience_budget_cents", 0)
        ),
        "shared_resilience_contribution_cents": int(
            action.get("shared_resilience_contribution_cents") or 0
        ),
        "incident_response_mode": str(
            action.get("incident_response_mode", incident.get("mode", "wait"))
        ),
        "repair_budget_cents": int(
            action.get("repair_budget_cents", incident.get("repair_budget_cents", 0))
        ),
    }


def _ranked_candidates(advice: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    raw_candidates = advice.get("candidate_actions")
    if not isinstance(raw_candidates, list):
        return []
    if advice.get("advisor_mode") in {
        "public_rollout_v3",
        "pareto_rollout_v4",
        "pareto_reliable_v5",
    }:
        rows = [item for item in raw_candidates if isinstance(item, Mapping)]
        decision = advice.get("pareto_decision")
        assessments = (
            decision.get("candidate_assessments", [])
            if isinstance(decision, Mapping)
            else []
        )
        assessment_by_id = {
            str(item["candidate_id"]): item
            for item in assessments
            if isinstance(item, Mapping) and item.get("candidate_id")
        }
        recommended = str(advice.get("recommended_candidate_id", ""))

        def ranking(item: Mapping[str, Any]) -> tuple[Any, ...]:
            candidate = item.get("candidate", {})
            candidate_id = str(candidate.get("candidate_id", ""))
            assessment = assessment_by_id.get(candidate_id, {})
            return (
                candidate_id == recommended,
                bool(assessment.get("eligible", False)),
                bool(assessment.get("pareto_frontier", False)),
                int(item.get("certainty_equivalent_value_cents", 0)),
                candidate_id,
            )

        ranked = sorted(rows, key=ranking, reverse=True)
        return [
            (
                str(item["candidate"]["candidate_id"]),
                _economic_action(item["candidate"]["action"]),
            )
            for item in ranked
        ]
    rows = [item for item in raw_candidates if isinstance(item, Mapping)]
    ranked = sorted(
        rows,
        key=lambda item: (
            int(item.get("expected_utility_proxy", 0)),
            int(item.get("worst_case_utility_proxy", 0)),
            str(item.get("action_label", "")),
        ),
        reverse=True,
    )
    return [
        (
            str(item.get("action_label", "unknown")),
            {"price_cents": int(item.get("price_cents", 0))},
        )
        for item in ranked
    ]


def build_advisor_adoption_trace(
    *,
    advice: Mapping[str, Any] | None,
    llm_requested_action: Mapping[str, Any],
    final_action: Mapping[str, Any],
    planner_output: Mapping[str, Any] | None,
) -> AdvisorAdoptionTrace | None:
    if not isinstance(advice, Mapping):
        return None
    advisor_mode = str(advice.get("advisor_mode", "unknown"))
    ranked = _ranked_candidates(advice)
    if advisor_mode in {
        "public_rollout_v3",
        "pareto_rollout_v4",
        "pareto_reliable_v5",
    }:
        candidate_id = str(advice.get("recommended_candidate_id", "")) or None
        advisor_action = _economic_action(
            advice.get("recommended_action", {})
            if isinstance(advice.get("recommended_action"), Mapping)
            else {}
        )
        baseline_action = next(
            (action for item_id, action in ranked if item_id == "maintain"),
            advisor_action,
        )
        candidate_rows = advice.get("candidate_actions", [])
        selected_row = next(
            (
                item
                for item in candidate_rows
                if isinstance(item, Mapping)
                and isinstance(item.get("candidate"), Mapping)
                and item["candidate"].get("candidate_id") == candidate_id
            ),
            {},
        )
        dimension_fields = {
            "price": "price_cents",
            "advertising": "advertising_budget_cents",
            "service": "service_budget_cents",
            "capacity": "capacity_investment_cents",
            "resilience": "resilience_budget_cents",
            "shared_resilience": "shared_resilience_contribution_cents",
            "repair": "repair_budget_cents",
        }
        changed_dimensions = [
            dimension_fields.get(str(field), str(field))
            for field in selected_row.get("candidate", {}).get(
                "changed_dimensions", []
            )
        ]
    else:
        candidate_id = str(advice.get("recommended_action", "")) or None
        advisor_action = {
            "price_cents": int(advice.get("recommended_price_cents", 0))
        }
        baseline_action = {
            "price_cents": int(advice.get("current_price_cents", 0))
        }
        changed_dimensions = ["price_cents"]

    requested = _economic_action(llm_requested_action)
    final = _economic_action(final_action)
    compared_fields = tuple(advisor_action)
    matches = sum(requested.get(field) == value for field, value in advisor_action.items())
    action_alignment = matches * 1_000_000 // max(1, len(compared_fields))

    target_matches = 0
    for field in changed_dimensions:
        expected_delta = int(advisor_action.get(field, 0)) - int(
            baseline_action.get(field, 0)
        )
        requested_delta = int(requested.get(field, 0)) - int(
            baseline_action.get(field, 0)
        )
        target_matches += (
            requested_delta == expected_delta
            or (expected_delta > 0 and requested_delta > 0)
            or (expected_delta < 0 and requested_delta < 0)
        )
    target_alignment = (
        target_matches * 1_000_000 // len(changed_dimensions)
        if changed_dimensions
        else action_alignment
    )
    exact = action_alignment == 1_000_000
    accepted_target = bool(changed_dimensions) and target_alignment == 1_000_000
    if exact:
        status: AdoptionStatus = "exact_action"
    elif accepted_target:
        status = "accepted_target"
    elif target_alignment > 0 or action_alignment >= 500_000:
        status = "partial"
    else:
        status = "rejected"

    chosen_candidate_id = next(
        (
            item_id
            for item_id, action in ranked
            if all(requested.get(field) == value for field, value in action.items())
        ),
        None,
    )
    chosen_rank = next(
        (
            index
            for index, (item_id, _action) in enumerate(ranked, start=1)
            if item_id == chosen_candidate_id
        ),
        None,
    )
    reason = ""
    if isinstance(planner_output, Mapping):
        reason = "；".join(
            str(planner_output.get(field, "")).strip()
            for field in ("strategy_summary", "situation_summary")
            if str(planner_output.get(field, "")).strip()
        )
    payload: dict[str, Any] = {
        "trace_schema_version": "advisor-adoption-trace-v1.0.0",
        "advisor_mode": advisor_mode,
        "advisor_candidate_id": candidate_id,
        "advisor_action": advisor_action,
        "llm_requested_action": requested,
        "final_action": final,
        "accepted": status in {"exact_action", "accepted_target"},
        "adoption_status": status,
        "action_alignment_ppm": action_alignment,
        "target_alignment_ppm": target_alignment,
        "advisor_rank": 1 if candidate_id else None,
        "chosen_candidate_id": chosen_candidate_id,
        "chosen_rank": chosen_rank,
        "agent_reason": reason,
        "classification_method": (
            "exact-economic-fields-and-target-direction-v1"
        ),
        "trace_hash": "pending",
    }
    payload["trace_hash"] = compute_adoption_trace_hash(payload)
    return AdvisorAdoptionTrace.model_validate(payload)
