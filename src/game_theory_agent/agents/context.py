"""Convert an Agent Gateway observation into a compact decision context."""

from __future__ import annotations

from typing import Any

from game_theory_agent.agents.contracts import (
    AgentIdentity,
    CommunicationContext,
    DecisionContext,
    DecisionMeta,
)
from game_theory_agent.agents.memory import EpisodeMemory
from game_theory_agent.agents.diagnostics import build_diagnostic_flags
from game_theory_agent.agents.personas import (
    PersonaProfile,
    PersonaRegistry,
    load_persona_registry,
)
from game_theory_agent.agents.plan_tracker import PlanTracker
from game_theory_agent.interaction.contracts import (
    CommunicationMode,
    CommunicationView,
    validate_communication_view_digest,
)
from game_theory_agent.cooperation.contracts import apply_cooperation_history_mode


class DecisionContextBuilder:
    def __init__(
        self,
        objective: str | None = None,
        plan_tracker: PlanTracker | None = None,
        persona_profile: PersonaProfile | None = None,
        persona_registry: PersonaRegistry | None = None,
        context_mode: str = "full",
        decision_support_version: str = "economic_v2",
        persona_semantics_version: str = "economic_v2",
        diagnostic_mode: str = "off",
        cooperation_history_mode: str = "full",
    ) -> None:
        if context_mode not in {"full", "state_only"}:
            raise ValueError("context_mode must be full or state_only")
        if decision_support_version not in {"legacy_v1", "economic_v2"}:
            raise ValueError("unsupported decision support version")
        if persona_semantics_version not in {"legacy_v1", "economic_v2"}:
            raise ValueError("unsupported persona semantics version")
        if diagnostic_mode not in {"off", "observe"}:
            raise ValueError("diagnostic_mode must be off or observe")
        if cooperation_history_mode not in {"full", "none"}:
            raise ValueError("cooperation_history_mode must be full or none")
        self.objective = objective
        self.plan_tracker = plan_tracker or PlanTracker()
        self.persona_profile = persona_profile
        self.persona_registry = persona_registry or load_persona_registry()
        self.context_mode = context_mode
        self.decision_support_version = decision_support_version
        self.persona_semantics_version = persona_semantics_version
        self.diagnostic_mode = diagnostic_mode
        self.cooperation_history_mode = cooperation_history_mode

    def _cooperation_for_context(
        self, observation: dict[str, Any]
    ) -> dict[str, Any] | None:
        raw = observation.get("cooperation")
        if raw is None:
            return None
        return apply_cooperation_history_mode(
            raw,
            round_number=int(observation["round"]),
            history_mode=self.cooperation_history_mode,
        )

    def _profile_for(self, own_company: dict[str, Any]) -> PersonaProfile:
        if self.persona_profile is not None:
            return self.persona_profile
        return self.persona_registry.get(
            str(
                own_company.get(
                    "persona", self.persona_registry.default_profile_id
                )
            )
        )

    @staticmethod
    def _meta(observation: dict[str, Any]) -> DecisionMeta:
        return DecisionMeta(
            episode_id=observation["episode_id"],
            round=int(observation["round"]),
            state_version=int(observation["state_version"]),
            state_hash=str(observation["state_hash"]),
            observation_hash=str(observation["observation_hash"]),
            belief_hash=(
                str(observation["belief_hash"])
                if observation.get("belief_hash") is not None
                else None
            ),
            opponent_model_hash=(
                str(observation["opponent_model_hash"])
                if observation.get("opponent_model_hash") is not None
                else None
            ),
            utility_inference_hash=(
                str(observation["utility_inference_hash"])
                if observation.get("utility_inference_hash") is not None
                else None
            ),
            game_theory_advice_hash=(
                str(observation["game_theory_advice"]["advice_hash"])
                if isinstance(observation.get("game_theory_advice"), dict)
                else None
            ),
            repeated_game_strategy_hash=(
                str(observation["repeated_game_strategy_hash"])
                if observation.get("repeated_game_strategy_hash") is not None
                else None
            ),
            rounds_remaining=int(observation["rounds_remaining"]),
            information_mode=str(observation["information_mode"]),
        )

    def build_communication(
        self,
        observation: dict[str, Any],
        company_id: str,
        memory: EpisodeMemory,
        communication_mode: CommunicationMode | None = None,
    ) -> CommunicationContext:
        """Build the pre-close context used only to author cheap-talk messages."""

        own_company = observation["own_company"]
        if own_company["company_id"] != company_id:
            raise ValueError("observation belongs to another company")
        raw_mode = communication_mode or observation.get(
            "communication_mode", "off"
        )
        if raw_mode not in {"off", "public_only", "public_private"}:
            raise ValueError("unsupported communication mode")
        mode: CommunicationMode = raw_mode
        profile = self._profile_for(own_company)
        memory_snapshot = memory.snapshot()
        recent_communication_views = self._resolve_communication_history(
            observation, company_id
        )
        if self.context_mode == "state_only":
            recent_communication_views = []
        competitors = list(observation["competitors"])
        recipient_ids = sorted(
            {
                str(item["company_id"])
                for item in competitors
                if item.get("company_id") and item["company_id"] != company_id
            }
        )
        action_bounds = dict(
            observation.get("action_constraints", {}).get("bounds", {})
        )
        claim_fields = (
            "price_cents",
            "advertising_budget_cents",
            "service_budget_cents",
            "capacity_investment_cents",
            "resilience_budget_cents",
            "shared_resilience_contribution_cents",
        )
        return CommunicationContext(
            context_mode=self.context_mode,
            cooperation_history_mode=self.cooperation_history_mode,
            communication_mode=mode,
            meta=self._meta(observation),
            identity=AgentIdentity(
                company_id=company_id,
                persona=profile.persona_id,
                objective=self.objective or profile.objective,
            ),
            persona_profile=profile,
            market=dict(observation["market"]),
            shared_resilience=observation.get("shared_resilience"),
            cooperation=self._cooperation_for_context(observation),
            belief_state=observation.get("belief_state"),
            market_regime=dict(observation["market_regime"]),
            decision_support=dict(observation["decision_support"]),
            own_company=own_company,
            competitors=competitors,
            risk_signals=list(observation.get("risk_signals", [])),
            active_market_events=list(
                observation.get("active_market_events", [])
            ),
            recent_communication_views=recent_communication_views,
            recent_rounds=(
                memory_snapshot["recent_rounds"]
                if self.context_mode == "full"
                else []
            ),
            rolling_summary=(
                memory_snapshot["rolling_summary"]
                if self.context_mode == "full"
                else {}
            ),
            current_plan=(
                memory_snapshot["current_plan"]
                if self.context_mode == "full"
                else None
            ),
            eligible_recipient_company_ids=recipient_ids,
            message_limits={
                "maximum_messages": 2,
                "maximum_public_messages": 1,
                "maximum_private_messages": (
                    1 if mode == "public_private" else 0
                ),
                "maximum_content_characters": 500,
            },
            action_claim_constraints={
                field_name: action_bounds[field_name]
                for field_name in claim_fields
                if field_name in action_bounds
            },
        )

    def build(
        self,
        observation: dict[str, Any],
        company_id: str,
        memory: EpisodeMemory,
        communication_view: CommunicationView | dict[str, Any] | None = None,
    ) -> DecisionContext:
        own_company = observation["own_company"]
        if own_company["company_id"] != company_id:
            raise ValueError("observation belongs to another company")
        persona_profile = self._profile_for(own_company)
        resolved_communication_view = self._resolve_communication_view(
            observation, company_id, communication_view
        )
        recent_communication_views = self._resolve_communication_history(
            observation, company_id
        )
        if self.context_mode == "state_only":
            recent_communication_views = []
        memory_snapshot = memory.snapshot()
        decision_support = dict(observation["decision_support"])
        if self.decision_support_version == "legacy_v1":
            legacy_fields = {
                "metrics_schema_version",
                "actual_unit_cost_cents",
                "fulfillment_cost_per_order_cents",
                "refund_rate_ppm",
                "current_unit_contribution_cents",
                "minimum_unit_contribution_cents",
                "minimum_safe_price_cents",
                "estimated_break_even_price_cents",
                "break_even_orders_at_current_price",
                "reference_orders",
                "fixed_overhead_cents",
                "protected_overhead_rounds",
                "minimum_cash_reserve_cents",
                "safe_discretionary_budget_cents",
                "maximum_discretionary_budget_cents",
                "cash_runway_milli_rounds",
                "consecutive_loss_rounds",
                "consecutive_profitable_rounds",
                "cash_drawdown_ppm",
                "strategic_phase",
                "price_below_variable_cost_floor",
                "plan_thresholds",
            }
            decision_support = {
                key: value
                for key, value in decision_support.items()
                if key in legacy_fields
            }
            decision_support["metrics_schema_version"] = (
                "decision-support-v1.0.0"
            )
        if self.context_mode == "full":
            current_plan = self.plan_tracker.evaluate(
                round_number=int(observation["round"]),
                decision_support=decision_support,
                rolling_summary=memory_snapshot["rolling_summary"],
                previous_plan=memory_snapshot["current_plan"],
                critical_events=memory_snapshot["critical_events"],
                risk_signals=list(observation.get("risk_signals", [])),
                active_incident=(
                    own_company.get("risk", {}).get("active_incident")
                ),
            )
            memory.set_current_plan(current_plan)
            recent_rounds = memory_snapshot["recent_rounds"]
            rolling_summary = memory_snapshot["rolling_summary"]
            critical_events = memory_snapshot["critical_events"]
        else:
            current_plan = None
            empty_memory = EpisodeMemory().snapshot()
            recent_rounds = []
            rolling_summary = empty_memory["rolling_summary"]
            critical_events = []
        diagnostic_flags = (
            build_diagnostic_flags(
                decision_support=decision_support,
                rolling_summary=rolling_summary,
                persona_profile=persona_profile,
                rounds_remaining=int(observation["rounds_remaining"]),
            )
            if self.diagnostic_mode == "observe"
            else []
        )
        return DecisionContext(
            context_mode=self.context_mode,
            cooperation_history_mode=self.cooperation_history_mode,
            decision_support_version=self.decision_support_version,
            persona_semantics_version=self.persona_semantics_version,
            diagnostic_mode=self.diagnostic_mode,
            meta=self._meta(observation),
            identity=AgentIdentity(
                company_id=company_id,
                persona=persona_profile.persona_id,
                objective=self.objective or persona_profile.objective,
            ),
            persona_profile=persona_profile,
            market=observation["market"],
            shared_resilience=observation.get("shared_resilience"),
            cooperation=self._cooperation_for_context(observation),
            belief_state=observation.get("belief_state"),
            opponent_model_state=observation.get("opponent_model_state"),
            utility_inference_state=observation.get("utility_inference_state"),
            game_theory_advice=observation.get("game_theory_advice"),
            repeated_game_strategy=observation.get("repeated_game_strategy"),
            market_regime=dict(observation["market_regime"]),
            decision_support=decision_support,
            diagnostic_flags=diagnostic_flags,
            own_company=own_company,
            competitors=list(observation["competitors"]),
            risk_signals=list(observation.get("risk_signals", [])),
            active_market_events=list(observation.get("active_market_events", [])),
            recent_rounds=recent_rounds,
            rolling_summary=rolling_summary,
            critical_events=critical_events,
            current_plan=current_plan,
            action_constraints=dict(observation["action_constraints"]),
            communication_view=resolved_communication_view,
            recent_communication_views=recent_communication_views,
        )

    @staticmethod
    def _resolve_communication_view(
        observation: dict[str, Any],
        company_id: str,
        supplied: CommunicationView | dict[str, Any] | None,
    ) -> CommunicationView | None:
        raw_view = (
            supplied
            if supplied is not None
            else observation.get("communication_view")
        )
        raw_mode = observation.get("communication_mode", "off")
        if raw_view is None:
            if raw_mode != "off":
                raise ValueError(
                    "a closed communication_view is required before decision"
                )
            return None
        view = (
            raw_view
            if isinstance(raw_view, CommunicationView)
            else CommunicationView.model_validate(raw_view)
        )
        expected = {
            "company_id": company_id,
            "episode_id": observation["episode_id"],
            "round": int(observation["round"]),
            "state_version": int(observation["state_version"]),
            "state_hash": str(observation["state_hash"]),
        }
        for field_name, expected_value in expected.items():
            if getattr(view, field_name) != expected_value:
                raise ValueError(
                    f"communication_view {field_name} does not match observation"
                )
        if view.status != "closed":
            raise ValueError("communication_view must be closed before decision")
        if view.mode != raw_mode:
            raise ValueError("communication_view mode does not match observation")
        validate_communication_view_digest(view)
        for message in view.visible_messages:
            if (
                message.episode_id != view.episode_id
                or message.round != view.round
                or message.state_version != view.state_version
                or message.state_hash != view.state_hash
            ):
                raise ValueError(
                    "visible message is not bound to the communication view state"
                )
            if message.channel == "private" and not (
                message.sender_company_id == company_id
                or company_id in message.recipients
            ):
                raise ValueError("communication_view contains a private-message leak")
        return view

    @staticmethod
    def _resolve_communication_history(
        observation: dict[str, Any],
        company_id: str,
    ) -> list[CommunicationView]:
        raw_history = observation.get("communication_history", [])
        if not isinstance(raw_history, list):
            raise ValueError("communication_history must be a list")
        if len(raw_history) > 3:
            raise ValueError("communication_history exceeds the three-round limit")
        mode = observation.get("communication_mode", "off")
        if mode == "off" and raw_history:
            raise ValueError("off mode cannot contain communication history")
        resolved: list[CommunicationView] = []
        prior_key: tuple[int, int] | None = None
        for raw_view in raw_history:
            view = (
                raw_view
                if isinstance(raw_view, CommunicationView)
                else CommunicationView.model_validate(raw_view)
            )
            if view.company_id != company_id:
                raise ValueError(
                    "communication_history contains another company's view"
                )
            if view.episode_id != observation["episode_id"]:
                raise ValueError("communication_history episode does not match")
            if view.status != "closed":
                raise ValueError("communication_history views must be closed")
            if view.mode != mode:
                raise ValueError("communication_history mode does not match")
            validate_communication_view_digest(view)
            if (
                view.round >= int(observation["round"])
                or view.state_version >= int(observation["state_version"])
            ):
                raise ValueError(
                    "communication_history cannot contain the current or future round"
                )
            key = (view.state_version, view.round)
            if prior_key is not None and key <= prior_key:
                raise ValueError(
                    "communication_history must be strictly chronological"
                )
            prior_key = key
            for message in view.visible_messages:
                if (
                    message.episode_id != view.episode_id
                    or message.round != view.round
                    or message.state_version != view.state_version
                    or message.state_hash != view.state_hash
                ):
                    raise ValueError(
                        "historical message is not bound to its communication view"
                    )
                if message.channel == "private" and not (
                    message.sender_company_id == company_id
                    or company_id in message.recipients
                ):
                    raise ValueError(
                        "communication_history contains a private-message leak"
                    )
            resolved.append(view)
        return resolved
