"""Continuous action constraints, presets, and strict v4 validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.exceptions import ActionValidationError
from game_theory_agent.market.models import (
    CompanyAction,
    IncidentResponse,
    IncidentResponseMode,
    Level,
    MarketState,
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    action: CompanyAction | None
    errors: tuple[str, ...]

    def require_valid(self) -> CompanyAction:
        if not self.valid or self.action is None:
            raise ActionValidationError("; ".join(self.errors))
        return self.action


class PresetResolver:
    """Translate UI convenience levels into the canonical numeric action."""

    dimensions = {
        "price": "price_cents",
        "advertising": "advertising_budget_cents",
        "service": "service_budget_cents",
        "capacity": "capacity_investment_cents",
        "resilience": "resilience_budget_cents",
    }

    def __init__(self, config: MarketConfig) -> None:
        self.config = config

    def resolve(
        self,
        preset: Mapping[str, Any],
        *,
        action_id: str,
        episode_id: str,
        agent_id: str,
        round_number: int,
        state_version: int,
        strategy_summary: str = "",
    ) -> CompanyAction:
        unknown = set(preset) - set(self.dimensions)
        missing = set(self.dimensions) - set(preset)
        if missing or unknown:
            raise ActionValidationError(
                f"preset fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        values: dict[str, int] = {}
        table = self.config.mapping("action", "presets")
        for dimension, target in self.dimensions.items():
            try:
                level = Level(str(preset[dimension]))
                values[target] = int(table[level.value][target])
            except (KeyError, TypeError, ValueError) as exc:
                raise ActionValidationError(
                    f"{dimension} must be one of low, medium, high"
                ) from exc
        return CompanyAction(
            action_id=action_id,
            episode_id=episode_id,
            agent_id=agent_id,
            round=round_number,
            state_version=state_version,
            incident_response=IncidentResponse(),
            strategy_summary=strategy_summary,
            **values,
        )


class ActionValidator:
    required_fields = frozenset(
        {
            "action_id",
            "episode_id",
            "agent_id",
            "round",
            "state_version",
            "price_cents",
            "advertising_budget_cents",
            "service_budget_cents",
            "capacity_investment_cents",
            "resilience_budget_cents",
            "incident_response",
        }
    )
    optional_fields = frozenset(
        {"strategy_summary", "shared_resilience_contribution_cents"}
    )

    def __init__(self, config: MarketConfig) -> None:
        self.config = config

    def validate(
        self,
        raw_action: Any,
        *,
        state: MarketState,
        company_id: str,
    ) -> ValidationResult:
        parsed = raw_action
        if isinstance(raw_action, str):
            try:
                parsed = json.loads(raw_action)
            except json.JSONDecodeError as exc:
                return ValidationResult(False, None, (f"invalid JSON: {exc.msg}",))
        if isinstance(parsed, CompanyAction):
            parsed = parsed.to_dict()
        if not isinstance(parsed, Mapping):
            return ValidationResult(False, None, ("action must be a JSON object",))

        errors: list[str] = []
        actual = set(parsed)
        missing = self.required_fields - actual
        unknown = actual - self.required_fields - self.optional_fields
        if missing:
            errors.append(f"missing fields: {sorted(missing)}")
        if unknown:
            errors.append(f"unknown fields: {sorted(unknown)}")
        if errors:
            return ValidationResult(False, None, tuple(errors))

        for field in ("action_id", "episode_id", "agent_id"):
            if not isinstance(parsed.get(field), str) or not str(parsed[field]).strip():
                errors.append(f"{field} must be a non-empty string")
        for field in (
            "round",
            "state_version",
            "price_cents",
            "advertising_budget_cents",
            "service_budget_cents",
            "capacity_investment_cents",
            "resilience_budget_cents",
        ):
            if isinstance(parsed.get(field), bool) or not isinstance(
                parsed.get(field), int
            ):
                errors.append(f"{field} must be an integer")
        shared_raw = parsed.get("shared_resilience_contribution_cents")
        if shared_raw is not None and (
            isinstance(shared_raw, bool) or not isinstance(shared_raw, int)
        ):
            errors.append(
                "shared_resilience_contribution_cents must be an integer"
            )

        response_raw = parsed.get("incident_response")
        if not isinstance(response_raw, Mapping):
            errors.append("incident_response must be an object")
            response = IncidentResponse()
        else:
            response_unknown = set(response_raw) - {"mode", "repair_budget_cents"}
            if response_unknown:
                errors.append(
                    f"incident_response unknown fields: {sorted(response_unknown)}"
                )
            try:
                mode = IncidentResponseMode(response_raw.get("mode", "wait"))
            except ValueError:
                errors.append("incident_response.mode is invalid")
                mode = IncidentResponseMode.WAIT
            repair = response_raw.get("repair_budget_cents", 0)
            if isinstance(repair, bool) or not isinstance(repair, int):
                errors.append(
                    "incident_response.repair_budget_cents must be an integer"
                )
                repair = 0
            response = IncidentResponse(mode, repair)

        if errors:
            return ValidationResult(False, None, tuple(errors))

        action = CompanyAction(
            action_id=str(parsed["action_id"]),
            episode_id=str(parsed["episode_id"]),
            agent_id=str(parsed["agent_id"]),
            round=int(parsed["round"]),
            state_version=int(parsed["state_version"]),
            price_cents=int(parsed["price_cents"]),
            advertising_budget_cents=int(parsed["advertising_budget_cents"]),
            service_budget_cents=int(parsed["service_budget_cents"]),
            capacity_investment_cents=int(parsed["capacity_investment_cents"]),
            resilience_budget_cents=int(parsed["resilience_budget_cents"]),
            shared_resilience_contribution_cents=(
                int(shared_raw or 0)
                if state.shared_resilience is not None
                else (int(shared_raw) if shared_raw not in (None, 0) else None)
            ),
            incident_response=response,
            strategy_summary=str(parsed.get("strategy_summary", "")),
        )
        errors.extend(self._semantic_errors(action, state, company_id))
        return ValidationResult(
            not errors, action if not errors else None, tuple(errors)
        )

    def _semantic_errors(
        self,
        action: CompanyAction,
        state: MarketState,
        company_id: str,
    ) -> list[str]:
        errors: list[str] = []
        if action.episode_id != state.episode_id:
            errors.append("EPISODE_CONFLICT")
        if action.agent_id != company_id:
            errors.append("AGENT_CONFLICT")
        if action.round != state.round:
            errors.append("ROUND_CONFLICT")
        if action.state_version != state.state_version:
            errors.append("STATE_VERSION_CONFLICT")

        bounds = self.config.mapping("action", "bounds")
        for field in (
            "price_cents",
            "advertising_budget_cents",
            "service_budget_cents",
            "capacity_investment_cents",
            "resilience_budget_cents",
        ):
            value = getattr(action, field)
            low = int(bounds[field]["min"])
            high = int(bounds[field]["max"])
            if not low <= value <= high:
                errors.append(f"{field} must be in [{low}, {high}]")
        shared = action.shared_resilience_contribution_cents
        if state.shared_resilience is None:
            if shared not in (None, 0):
                errors.append("shared resilience contribution is disabled")
        else:
            shared_value = int(shared or 0)
            shared_bounds = bounds["shared_resilience_contribution_cents"]
            if not int(shared_bounds["min"]) <= shared_value <= int(
                shared_bounds["max"]
            ):
                errors.append(
                    "shared_resilience_contribution_cents must be in "
                    f"[{shared_bounds['min']}, {shared_bounds['max']}]"
                )

        repair_bounds = bounds["repair_budget_cents"]
        repair = action.incident_response.repair_budget_cents
        if not int(repair_bounds["min"]) <= repair <= int(repair_bounds["max"]):
            errors.append(
                f"repair_budget_cents must be in [{repair_bounds['min']}, {repair_bounds['max']}]"
            )

        company = state.company(company_id)
        incident = company.risk.active_incident
        if incident is None:
            if (
                action.incident_response.mode is not IncidentResponseMode.WAIT
                or repair != 0
            ):
                errors.append("repair is not allowed without an active incident")
        elif action.incident_response.mode is IncidentResponseMode.WAIT and repair != 0:
            errors.append("wait requires repair_budget_cents = 0")
        elif action.incident_response.mode is IncidentResponseMode.PARTIAL_REPAIR:
            if not 0 < repair < incident.remaining_repair_cents:
                errors.append(
                    "partial_repair must be positive and below remaining repair cost"
                )
        elif action.incident_response.mode is IncidentResponseMode.FULL_REPAIR:
            if repair != incident.remaining_repair_cents:
                errors.append("full_repair must equal the remaining repair cost")

        if state.rounds_remaining <= 1:
            if action.capacity_investment_cents != 0:
                errors.append("capacity investment is disabled in the last round")
            if action.resilience_budget_cents != 0:
                errors.append("resilience investment is disabled in the last round")
            if (action.shared_resilience_contribution_cents or 0) != 0:
                errors.append(
                    "shared resilience contribution is disabled in the last round"
                )
        if action.fixed_spend_cents > company.financial.cash_balance_cents:
            errors.append("BUDGET_EXCEEDED")
        return errors
