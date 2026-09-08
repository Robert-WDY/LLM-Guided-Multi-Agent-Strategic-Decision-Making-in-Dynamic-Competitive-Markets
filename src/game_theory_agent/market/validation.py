"""Continuous action constraints, presets, and strict v4 validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.exceptions import ActionValidationError
from game_theory_agent.market.models import (
    CompanyAction,
    CompanyOperatingStatus,
    IncidentResponse,
    IncidentResponseMode,
    Level,
    MarketState,
)


PPM = 1_000_000


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
        {
            "strategy_summary",
            "shared_resilience_contribution_cents",
            "threshold_project_contribution_cents",
            "mutual_aid_partner_company_id",
            "mutual_aid_capacity_offer_orders",
            "mutual_aid_capacity_request_orders",
            "price_coordination_partner_company_id",
            "price_coordination_target_cents",
            "primary_supplier_id",
            "backup_supplier_id",
            "primary_supplier_share_ppm",
            "procurement_quantity_orders", "contract_quantity_orders", "contract_duration_rounds", "contract_bid_cents",
        }
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
        project_raw = parsed.get("threshold_project_contribution_cents")
        if project_raw is not None and (
            isinstance(project_raw, bool) or not isinstance(project_raw, int)
        ):
            errors.append(
                "threshold_project_contribution_cents must be an integer"
            )
        mutual_partner_raw = parsed.get("mutual_aid_partner_company_id")
        if mutual_partner_raw is not None and (
            not isinstance(mutual_partner_raw, str)
            or not mutual_partner_raw.strip()
        ):
            errors.append(
                "mutual_aid_partner_company_id must be a non-empty string"
            )
        for field in (
            "mutual_aid_capacity_offer_orders",
            "mutual_aid_capacity_request_orders",
        ):
            value = parsed.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                errors.append(f"{field} must be an integer")
        coordination_partner_raw = parsed.get(
            "price_coordination_partner_company_id"
        )
        if coordination_partner_raw is not None and (
            not isinstance(coordination_partner_raw, str)
            or not coordination_partner_raw.strip()
        ):
            errors.append(
                "price_coordination_partner_company_id must be a non-empty string"
            )
        coordination_target_raw = parsed.get("price_coordination_target_cents")
        if coordination_target_raw is not None and (
            isinstance(coordination_target_raw, bool)
            or not isinstance(coordination_target_raw, int)
        ):
            errors.append("price_coordination_target_cents must be an integer")
        for field in ("primary_supplier_id", "backup_supplier_id"):
            value = parsed.get(field)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                errors.append(f"{field} must be a non-empty string")
        supplier_share_raw = parsed.get("primary_supplier_share_ppm")
        quantity = parsed.get("procurement_quantity_orders")
        if quantity is not None and type(quantity) is not int:
            errors.append("procurement_quantity_orders must be an integer")
        if supplier_share_raw is not None and (
            isinstance(supplier_share_raw, bool)
            or not isinstance(supplier_share_raw, int)
        ):
            errors.append("primary_supplier_share_ppm must be an integer")

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
            threshold_project_contribution_cents=(
                int(project_raw or 0)
                if state.strategic_market is not None
                and state.strategic_market.threshold_project is not None
                else (
                    int(project_raw)
                    if project_raw not in (None, 0)
                    else None
                )
            ),
            mutual_aid_partner_company_id=(
                str(mutual_partner_raw)
                if mutual_partner_raw is not None
                else None
            ),
            mutual_aid_capacity_offer_orders=(
                int(parsed.get("mutual_aid_capacity_offer_orders") or 0)
                if parsed.get("mutual_aid_capacity_offer_orders") is not None
                else None
            ),
            mutual_aid_capacity_request_orders=(
                int(parsed.get("mutual_aid_capacity_request_orders") or 0)
                if parsed.get("mutual_aid_capacity_request_orders") is not None
                else None
            ),
            price_coordination_partner_company_id=(
                str(coordination_partner_raw)
                if coordination_partner_raw is not None
                else None
            ),
            price_coordination_target_cents=(
                int(coordination_target_raw)
                if coordination_target_raw is not None
                else None
            ),
            primary_supplier_id=(
                str(parsed["primary_supplier_id"])
                if parsed.get("primary_supplier_id") is not None
                else None
            ),
            backup_supplier_id=(
                str(parsed["backup_supplier_id"])
                if parsed.get("backup_supplier_id") is not None
                else None
            ),
            primary_supplier_share_ppm=(
                int(parsed["primary_supplier_share_ppm"])
                if parsed.get("primary_supplier_share_ppm") is not None
                else None
            ),
            **{key:parsed.get(key) for key in ("contract_quantity_orders", "contract_duration_rounds", "contract_bid_cents")},
            procurement_quantity_orders=parsed.get("procurement_quantity_orders"),
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

        project = (
            state.strategic_market.threshold_project
            if state.strategic_market is not None
            else None
        )
        project_contribution = action.threshold_project_contribution_cents
        if project is None:
            if project_contribution not in (None, 0):
                errors.append("threshold project contribution is disabled")
        else:
            project_value = int(project_contribution or 0)
            project_bounds = bounds["threshold_project_contribution_cents"]
            if not int(project_bounds["min"]) <= project_value <= int(
                project_bounds["max"]
            ):
                errors.append(
                    "threshold_project_contribution_cents must be in "
                    f"[{project_bounds['min']}, {project_bounds['max']}]"
                )
            if project.status.value != "active" and project_value != 0:
                errors.append("threshold project no longer accepts contributions")
            if state.round > project.deadline_round and project_value != 0:
                errors.append("threshold project deadline has passed")

        mutual_enabled = bool(
            state.strategic_market is not None
            and state.strategic_market.mutual_aid_enabled
        )
        mutual_partner = action.mutual_aid_partner_company_id
        mutual_offer = int(action.mutual_aid_capacity_offer_orders or 0)
        mutual_request = int(action.mutual_aid_capacity_request_orders or 0)
        if not mutual_enabled:
            if mutual_partner is not None or mutual_offer or mutual_request:
                errors.append("mutual aid is disabled")
        else:
            for field, value in (
                ("mutual_aid_capacity_offer_orders", mutual_offer),
                ("mutual_aid_capacity_request_orders", mutual_request),
            ):
                mutual_bounds = bounds[field]
                if not int(mutual_bounds["min"]) <= value <= int(
                    mutual_bounds["max"]
                ):
                    errors.append(
                        f"{field} must be in "
                        f"[{mutual_bounds['min']}, {mutual_bounds['max']}]"
                    )
            if mutual_offer or mutual_request:
                if mutual_partner is None:
                    errors.append("mutual aid amount requires a partner")
                elif mutual_partner == company_id:
                    errors.append("mutual aid partner cannot be self")
                elif mutual_partner not in state.company_ids:
                    errors.append("mutual aid partner is unknown")
                elif (
                    state.strategic_market is not None
                    and mutual_partner
                    not in state.strategic_market.active_company_ids
                ):
                    errors.append("mutual aid partner has exited")
            elif mutual_partner is not None:
                errors.append("mutual aid partner requires an offer or request")

        coordination_enabled = bool(
            state.strategic_market is not None
            and state.strategic_market.price_coordination_enabled
        )
        coordination_partner = action.price_coordination_partner_company_id
        coordination_target = action.price_coordination_target_cents
        if not coordination_enabled:
            if coordination_partner is not None or coordination_target is not None:
                errors.append("price coordination is disabled")
        elif (coordination_partner is None) != (coordination_target is None):
            errors.append(
                "price coordination partner and target must be supplied together"
            )
        elif coordination_partner is not None and coordination_target is not None:
            if coordination_partner == company_id:
                errors.append("price coordination partner cannot be self")
            elif coordination_partner not in state.company_ids:
                errors.append("price coordination partner is unknown")
            elif (
                state.strategic_market is not None
                and coordination_partner
                not in state.strategic_market.active_company_ids
            ):
                errors.append("price coordination partner has exited")
            price_bounds = bounds["price_cents"]
            if not int(price_bounds["min"]) <= coordination_target <= int(
                price_bounds["max"]
            ):
                errors.append(
                    "price_coordination_target_cents must be in "
                    f"[{price_bounds['min']}, {price_bounds['max']}]"
                )

        supply_chain_enabled = state.supply_chain is not None
        for key,maximum in (("contract_quantity_orders",state.company(company_id).operations.base_capacity_orders),("contract_duration_rounds",5),("contract_bid_cents",100000)):
            value=getattr(action,key)
            if value is not None and (not self.config.data.get("supply_chain",{}).get("strategic_policy") or type(value) is not int or not 0 <= value <= maximum):
                errors.append("invalid supply contract field: "+key)
        quantity = action.procurement_quantity_orders
        if quantity is not None:
            if not self.config.data.get("autonomous_market") or type(quantity) is not int or not 0 <= quantity <= state.company(company_id).operations.base_capacity_orders:
                errors.append("procurement quantity requires autonomous market and must fit own capacity")
        primary_supplier = action.primary_supplier_id
        backup_supplier = action.backup_supplier_id
        primary_share = action.primary_supplier_share_ppm
        if not supply_chain_enabled:
            if (
                primary_supplier is not None
                or backup_supplier is not None
                or primary_share is not None
            ):
                errors.append("supply chain procurement is disabled")
        else:
            assert state.supply_chain is not None
            supplier_ids = set(state.supply_chain.supplier_ids)
            if primary_supplier is not None and primary_supplier not in supplier_ids:
                errors.append("primary supplier is unknown")
            if backup_supplier is not None and backup_supplier not in supplier_ids:
                errors.append("backup supplier is unknown")
            if (
                primary_supplier is not None
                and backup_supplier is not None
                and primary_supplier == backup_supplier
            ):
                errors.append("primary and backup suppliers must differ")
            if primary_share is not None:
                share_bounds = bounds["primary_supplier_share_ppm"]
                if not int(share_bounds["min"]) <= primary_share <= int(
                    share_bounds["max"]
                ):
                    errors.append(
                        "primary_supplier_share_ppm must be in [0, 1000000]"
                    )
            if backup_supplier is None and primary_share not in (None, PPM):
                errors.append(
                    "primary supplier share requires a distinct backup supplier"
                )

        repair_bounds = bounds["repair_budget_cents"]
        repair = action.incident_response.repair_budget_cents
        if not int(repair_bounds["min"]) <= repair <= int(repair_bounds["max"]):
            errors.append(
                f"repair_budget_cents must be in [{repair_bounds['min']}, {repair_bounds['max']}]"
            )

        company = state.company(company_id)
        if state.strategic_market is not None:
            lifecycle = state.strategic_market.lifecycle(company_id)
            if lifecycle.status is CompanyOperatingStatus.EXITED:
                if action.price_cents != company.commercial.price_cents:
                    errors.append("exited company must keep its frozen price")
                if action.fixed_spend_cents != 0:
                    errors.append("exited company cannot spend or contribute")
                if action.incident_response.mode is not IncidentResponseMode.WAIT:
                    errors.append("exited company cannot repair incidents")
                if action.mutual_aid_partner_company_id is not None:
                    errors.append(
                        "exited company cannot select a mutual aid partner"
                    )
                if (action.mutual_aid_capacity_offer_orders or 0) != 0:
                    errors.append("exited company cannot offer mutual aid")
                if (action.mutual_aid_capacity_request_orders or 0) != 0:
                    errors.append("exited company cannot request mutual aid")
                if action.price_coordination_partner_company_id is not None:
                    errors.append("exited company cannot coordinate price")
                if action.price_coordination_target_cents is not None:
                    errors.append("exited company cannot set coordination target")
                if action.primary_supplier_id is not None:
                    errors.append("exited company cannot select a supplier")
                if action.backup_supplier_id is not None:
                    errors.append("exited company cannot select a backup supplier")
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
            if (action.threshold_project_contribution_cents or 0) != 0:
                errors.append(
                    "threshold project contribution is disabled in the last round"
                )
        if action.fixed_spend_cents > company.financial.cash_balance_cents:
            errors.append("BUDGET_EXCEEDED")
        return errors
