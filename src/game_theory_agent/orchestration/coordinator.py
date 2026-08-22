"""Round barrier: observe concurrently, decide concurrently, settle exactly once."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict

from game_theory_agent.agents.contracts import (
    AgentCommunicationResult,
    AgentDecisionResult,
)
from game_theory_agent.agents.contracts import SuccessCriteria
from game_theory_agent.agents.counterfactual import CounterfactualEvaluator
from game_theory_agent.agents.result_analyzer import ResultAnalyzer
from game_theory_agent.cooperation import (
    CooperationCloseRecord,
    CooperationRoundRecord,
)
from game_theory_agent.agents.runtime import AgentRuntime
from game_theory_agent.market.models import MarketState
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.interaction import (
    CommunicationClosure,
    CommunicationRoundLedger,
    CommunicationView,
    validate_communication_view_digest,
)
from game_theory_agent.information import (
    InformationReplayMismatchError,
    ObservationSnapshot,
    verify_information_snapshot,
)
from game_theory_agent.orchestration.clients import (
    AgentGatewayClient,
    ApiClientError,
    ControllerClient,
)
from game_theory_agent.orchestration.round_event import (
    AgentRoundTrace,
    CommunicationGenerationTrace,
    CommunicationPhaseRecord,
    JsonlRoundEventLogger,
    RoundEvent,
)


class StaleRoundError(RuntimeError):
    """Raised before settlement if participants did not see one frozen state."""


class CoordinatedRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settlement: dict[str, Any]
    event: RoundEvent


class RoundCoordinator:
    def __init__(
        self,
        controller: ControllerClient,
        gateway: AgentGatewayClient,
        runtimes: dict[str, AgentRuntime],
        *,
        result_analyzer: ResultAnalyzer | None = None,
        counterfactual_evaluator: CounterfactualEvaluator | None = None,
        event_logger: JsonlRoundEventLogger | None = None,
        decision_timeout_seconds: float = 45.0,
        communication_timeout_seconds: float = 30.0,
    ) -> None:
        if any(runtime.company_id != company_id for company_id, runtime in runtimes.items()):
            raise ValueError("runtime keys must match runtime.company_id")
        self.controller = controller
        self.gateway = gateway
        self.runtimes = dict(runtimes)
        self.result_analyzer = result_analyzer or ResultAnalyzer()
        self.counterfactual_evaluator = (
            counterfactual_evaluator or CounterfactualEvaluator()
        )
        self.event_logger = event_logger
        self.decision_timeout_seconds = decision_timeout_seconds
        self.communication_timeout_seconds = communication_timeout_seconds

    @staticmethod
    def _agent_type(model_name: str | None) -> str:
        if (model_name or "").startswith("mock"):
            return "mock"
        if (model_name or "").startswith("uniform-random"):
            return "random"
        return "model"

    @staticmethod
    def _closure_from_api(payload: dict[str, Any]) -> CommunicationClosure:
        raw_phase = payload.get("communication_phase", payload)
        if "closure" in raw_phase:
            return CommunicationClosure.model_validate(raw_phase["closure"])
        closure_payload = {
            field_name: raw_phase[field_name]
            for field_name in CommunicationClosure.model_fields
        }
        return CommunicationClosure.model_validate(closure_payload)

    def _communication_generation_traces(
        self,
        state_before: MarketState,
        communication_mode: str,
        results: dict[str, AgentCommunicationResult],
        acceptances: dict[str, dict[str, Any]],
        submission_errors: dict[str, str],
        information_snapshots: dict[str, ObservationSnapshot],
    ) -> list[CommunicationGenerationTrace]:
        traces: list[CommunicationGenerationTrace] = []
        for company_id in state_before.company_ids:
            result = results.get(company_id)
            information_snapshot = information_snapshots.get(company_id)
            if result is None:
                runtime = self.runtimes.get(company_id)
                traces.append(
                    CommunicationGenerationTrace(
                        company_id=company_id,
                        agent_id=(
                            runtime.agent_id
                            if runtime is not None
                            else "controller-rule"
                        ),
                        agent_type=(
                            self._agent_type(
                                getattr(
                                    runtime.model_client, "model_name", None
                                )
                            )
                            if runtime is not None
                            else "rule"
                        ),
                        generation_status=(
                            "disabled"
                            if communication_mode == "off"
                            else "not_applicable"
                        ),
                        observation_hash=None,
                        information_snapshot=None,
                        submission=None,
                        validation_errors=(
                            []
                            if communication_mode == "off"
                            else ["no communication runtime; implicit silence"]
                        ),
                    )
                )
                continue

            acceptance = acceptances.get(company_id)
            was_accepted = acceptance is not None
            if not was_accepted:
                generation_status = "invalid"
            elif not result.success:
                generation_status = "fallback"
            elif result.is_silence:
                generation_status = "silent"
            else:
                generation_status = "submitted"
            model_name = result.model_name or getattr(
                self.runtimes[company_id].model_client, "model_name", None
            )
            error_messages = []
            if company_id in submission_errors:
                error_messages.append(submission_errors[company_id])
            if result.error_code:
                error_messages.append(result.error_code)
            traces.append(
                CommunicationGenerationTrace(
                    company_id=company_id,
                    agent_id=result.agent_id,
                    agent_type=self._agent_type(model_name),
                    generation_status=generation_status,
                    observation_hash=(
                        information_snapshot.observation_hash
                        if information_snapshot is not None
                        else None
                    ),
                    information_snapshot=information_snapshot,
                    submission=(result.submission if was_accepted else None),
                    accepted_message_ids=(
                        list(acceptance.get("message_ids", []))
                        if acceptance is not None
                        else []
                    ),
                    model_name=result.model_name,
                    prompt_version=result.prompt_version,
                    raw_model_output=result.raw_response,
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    retry_count=result.retry_count,
                    validation_errors=error_messages,
                    error_code=(
                        result.error_code
                        or (
                            "COMMUNICATION_SUBMISSION_FAILED"
                            if not was_accepted
                            else None
                        )
                    ),
                    error_message=(
                        result.error_message
                        or submission_errors.get(company_id)
                    ),
                    **(
                        {
                            "communication_context": result.context.model_dump(
                                mode="json"
                            ),
                            "is_silence": result.is_silence,
                            "silence_reason": result.silence_reason,
                        }
                        if "communication_context"
                        in CommunicationGenerationTrace.model_fields
                        else {}
                    ),
                )
            )
        return traces

    async def run_round(self, episode_id: str) -> CoordinatedRound:
        episode = await self.controller.get_episode(episode_id)
        state_before = MarketState.from_dict(episode["state"])
        communication_mode = str(
            episode.get("manifest", {}).get("communication_mode", "off")
        )
        cooperation_mode = str(
            episode.get("manifest", {}).get("cooperation_mode", "off")
        )
        belief_mode = str(
            episode.get("manifest", {}).get("belief_mode", "off")
        )
        if state_before.terminal:
            raise RuntimeError("episode is terminal")
        unknown = set(self.runtimes) - set(state_before.company_ids)
        if unknown:
            raise ValueError(f"runtime companies are not in episode: {sorted(unknown)}")

        company_ids = list(self.runtimes)
        observations = await asyncio.gather(
            *(
                self.gateway.get_observation(episode_id, company_id)
                for company_id in company_ids
            )
        )
        communication_information_snapshots: dict[
            str, ObservationSnapshot
        ] = {}
        for company_id, observation in zip(company_ids, observations):
            if (
                observation["round"] != state_before.round
                or observation["state_version"] != state_before.state_version
                or observation["state_hash"] != state_before.state_hash
            ):
                raise StaleRoundError(
                    "Agent observations do not match the frozen controller state"
                )
            try:
                snapshot = ObservationSnapshot.from_observation(
                    observation, company_id
                )
                verify_information_snapshot(state_before, snapshot)
            except (ValueError, InformationReplayMismatchError) as exc:
                raise StaleRoundError(
                    "Agent observation violates the visibility policy"
                ) from exc
            communication_information_snapshots[company_id] = snapshot

        communication_results: dict[str, AgentCommunicationResult] = {}
        communication_acceptances: dict[str, dict[str, Any]] = {}
        communication_submission_errors: dict[str, str] = {}
        cooperation_close: CooperationCloseRecord | None = None
        if communication_mode == "off":
            communication_closure = CommunicationRoundLedger(
                episode_id=state_before.episode_id,
                round_number=state_before.round,
                state_version=state_before.state_version,
                state_hash=state_before.state_hash,
                company_ids=state_before.company_ids,
                mode="off",
            ).close()
            if cooperation_mode == "shared_resilience_v1":
                close_payload = await self.controller.close_communication(
                    episode_id,
                    state_before.round,
                    state_before.state_version,
                    state_before.state_hash,
                )
                communication_closure = self._closure_from_api(close_payload)
                raw_cooperation_close = close_payload.get("cooperation_close")
                if raw_cooperation_close is None:
                    raise StaleRoundError(
                        "cooperation close is missing from the no-communication barrier"
                    )
                cooperation_close = CooperationCloseRecord.model_validate(
                    raw_cooperation_close
                )
                if (
                    communication_closure.mode != "off"
                    or cooperation_close.episode_id != state_before.episode_id
                    or cooperation_close.round != state_before.round
                    or cooperation_close.state_version
                    != state_before.state_version
                    or cooperation_close.state_hash != state_before.state_hash
                    or cooperation_close.communication_transcript_hash
                    != communication_closure.transcript_hash
                ):
                    raise StaleRoundError(
                        "no-communication cooperation close does not match frozen state"
                    )
            decision_observations = observations
        else:
            generated = await asyncio.gather(
                *(
                    self.runtimes[company_id].communicate(
                        observation,
                        communication_mode=communication_mode,
                        timeout_seconds=self.communication_timeout_seconds,
                    )
                    for company_id, observation in zip(
                        company_ids, observations
                    )
                )
            )
            communication_results = {
                result.company_id: result for result in generated
            }

            async def submit_communication(
                result: AgentCommunicationResult,
            ) -> tuple[str, dict[str, Any] | Exception]:
                try:
                    payload = await self.gateway.submit_communication(
                        episode_id, result
                    )
                    return result.company_id, payload
                except Exception as exc:
                    return result.company_id, exc

            communication_submissions = await asyncio.gather(
                *(submit_communication(result) for result in generated)
            )
            for company_id, item in communication_submissions:
                if isinstance(item, Exception):
                    communication_submission_errors[company_id] = (
                        f"{type(item).__name__}: {item}"[:2000]
                    )
                else:
                    communication_acceptances[company_id] = item

            close_payload = await self.controller.close_communication(
                episode_id,
                state_before.round,
                state_before.state_version,
                state_before.state_hash,
            )
            communication_closure = self._closure_from_api(close_payload)
            if cooperation_mode == "shared_resilience_v1":
                raw_cooperation_close = close_payload.get("cooperation_close")
                if raw_cooperation_close is None:
                    raise StaleRoundError(
                        "cooperation close is missing from the closed barrier"
                    )
                cooperation_close = CooperationCloseRecord.model_validate(
                    raw_cooperation_close
                )
                if (
                    cooperation_close.episode_id != state_before.episode_id
                    or cooperation_close.round != state_before.round
                    or cooperation_close.state_version
                    != state_before.state_version
                    or cooperation_close.state_hash != state_before.state_hash
                    or cooperation_close.communication_transcript_hash
                    != communication_closure.transcript_hash
                ):
                    raise StaleRoundError(
                        "cooperation close does not match the frozen communication batch"
                    )
            if (
                communication_closure.mode != communication_mode
                or communication_closure.episode_id != state_before.episode_id
                or communication_closure.round != state_before.round
                or communication_closure.state_version
                != state_before.state_version
                or communication_closure.state_hash != state_before.state_hash
                or set(communication_closure.views)
                != set(state_before.company_ids)
            ):
                raise StaleRoundError(
                    "closed communication batch does not match the frozen state"
                )
            unexpected_submitters = set(
                communication_closure.submitted_company_ids
            ) - set(communication_results)
            if unexpected_submitters:
                raise StaleRoundError(
                    "closed communication batch contains submissions from "
                    "companies outside this coordinator run: "
                    f"{sorted(unexpected_submitters)}"
                )
            authoritative_ids = {
                company_id: [
                    message.message_id
                    for message in communication_closure.all_messages
                    if message.sender_company_id == company_id
                ]
                for company_id in company_ids
            }
            for company_id, result in communication_results.items():
                expected_ledger = CommunicationRoundLedger(
                    episode_id=state_before.episode_id,
                    round_number=state_before.round,
                    state_version=state_before.state_version,
                    state_hash=state_before.state_hash,
                    company_ids=state_before.company_ids,
                    mode=communication_mode,
                )
                expected_ids = [
                    message.message_id
                    for message in expected_ledger.submit(
                        company_id, result.submission
                    )
                ]
                was_authoritatively_submitted = (
                    company_id
                    in communication_closure.submitted_company_ids
                )
                acceptance = communication_acceptances.get(company_id)
                if not was_authoritatively_submitted:
                    if acceptance is not None:
                        raise StaleRoundError(
                            "communication acceptance is absent from closed batch"
                        )
                    continue
                if expected_ids != authoritative_ids[company_id]:
                    raise StaleRoundError(
                        "generated communication does not match closed batch"
                    )
                if acceptance is not None and list(
                    acceptance.get("message_ids", [])
                ) != expected_ids:
                    raise StaleRoundError(
                        "communication acceptance ids do not match closed batch"
                    )
                communication_acceptances[company_id] = {
                    **(acceptance or {}),
                    "message_ids": expected_ids,
                    "reconciled_from_closed_batch": acceptance is None,
                }
            decision_observations = await asyncio.gather(
                *(
                    self.gateway.get_observation(episode_id, company_id)
                    for company_id in company_ids
                )
            )
            for company_id, observation in zip(
                company_ids, decision_observations
            ):
                if (
                    observation["round"] != state_before.round
                    or observation["state_version"]
                    != state_before.state_version
                    or observation["state_hash"] != state_before.state_hash
                ):
                    raise StaleRoundError(
                        "post-communication observations changed market state"
                    )
                raw_view = observation.get("communication_view")
                if raw_view is None or raw_view.get("status") != "closed":
                    raise StaleRoundError(
                        "decision observation does not contain a closed communication view"
                    )
                expected_view = communication_closure.views[company_id]
                try:
                    view = CommunicationView.model_validate(raw_view)
                    validate_communication_view_digest(view)
                except ValueError as exc:
                    raise StaleRoundError(
                        "decision observation contains an invalid communication view"
                    ) from exc
                if view != expected_view:
                    raise StaleRoundError(
                        "decision observation communication view does not match close"
                    )

        decision_information_snapshots: dict[str, ObservationSnapshot] = {}
        for company_id, observation in zip(
            company_ids, decision_observations
        ):
            try:
                snapshot = ObservationSnapshot.from_observation(
                    observation, company_id
                )
                verify_information_snapshot(state_before, snapshot)
            except (ValueError, InformationReplayMismatchError) as exc:
                raise StaleRoundError(
                    "decision observation violates the visibility policy"
                ) from exc
            decision_information_snapshots[company_id] = snapshot

        generation_traces = self._communication_generation_traces(
            state_before,
            communication_mode,
            communication_results,
            communication_acceptances,
            communication_submission_errors,
            communication_information_snapshots,
        )
        communication_phase = CommunicationPhaseRecord.from_closure(
            communication_closure,
            generation_traces=generation_traces,
        )

        decisions_list = await asyncio.gather(
            *(
                self.runtimes[company_id].decide(
                    observation,
                    timeout_seconds=self.decision_timeout_seconds,
                )
                for company_id, observation in zip(
                    company_ids, decision_observations
                )
            )
        )
        decisions = {result.company_id: result for result in decisions_list}

        async def submit(
            result: AgentDecisionResult,
        ) -> tuple[str, dict[str, Any] | Exception]:
            try:
                return result.company_id, await self.gateway.submit_intent(
                    episode_id, result
                )
            except Exception as exc:  # each company has an independent fallback
                return result.company_id, exc

        submit_results = await asyncio.gather(
            *(submit(result) for result in decisions_list if result.success)
        )
        intent_ids: dict[str, str] = {}
        for company_id, item in submit_results:
            if isinstance(item, ApiClientError) and item.status_code == 409:
                raise StaleRoundError(
                    "Controller state changed while Agent intents were submitted"
                ) from item
            if isinstance(item, Exception):
                prior = decisions[company_id]
                decisions[company_id] = prior.model_copy(
                    update={
                        "success": False,
                        "error_code": "INTENT_SUBMISSION_FAILED",
                        "error_message": f"{type(item).__name__}: {item}"[:2000],
                        "fallback_required": True,
                    }
                )
            else:
                intent_ids[company_id] = item["intent_id"]
        for result in decisions.values():
            if not result.success:
                self.runtimes[result.company_id].memory.record_fallback()

        # MarketEnv owns the idempotency protocol and accepts this exact key only.
        step_id = f"{episode_id}:{state_before.round}:{state_before.state_version}"
        settlement = await self.controller.settle_agent_round(
            episode_id, step_id, intent_ids
        )
        state_after = MarketState.from_dict(settlement["state"])
        cooperation_round = None
        if cooperation_mode == "shared_resilience_v1":
            raw_cooperation_round = settlement.get("cooperation_round")
            if raw_cooperation_round is None:
                raise RuntimeError("cooperation settlement record is missing")
            cooperation_round = CooperationRoundRecord.model_validate(
                raw_cooperation_round
            )
            if (
                cooperation_round.close.round != state_before.round
                or cooperation_round.close.state_hash != state_before.state_hash
            ):
                raise RuntimeError(
                    "cooperation settlement does not match the market transition"
                )
        traces: list[AgentRoundTrace] = []
        final_actions: dict[str, dict[str, Any]] = {
            company_id: settlement["decision_resolutions"][company_id]["action"]
            for company_id in state_before.company_ids
        }

        for company_id in state_before.company_ids:
            resolution = settlement["decision_resolutions"][company_id]
            final_action = resolution["action"]
            result = decisions.get(company_id)
            information_snapshot = decision_information_snapshots.get(
                company_id
            )
            expected = (
                result.decision.plan.expected_outcome
                if (
                    result is not None
                    and result.success
                    and result.decision is not None
                )
                else None
            )
            success_criteria = (
                result.decision.plan.success_criteria
                if result is not None and result.decision is not None
                else None
            )
            if result is not None and result.decision is not None:
                plan_constraints = (result.context.current_plan or {}).get(
                    "constraints", {}
                )
                model_criteria = result.decision.plan.success_criteria
                plan_max_spend = int(
                    plan_constraints.get(
                        "maximum_discretionary_spend_cents",
                        model_criteria.maximum_fixed_spend_cents
                        if model_criteria.maximum_fixed_spend_cents is not None
                        else 0,
                    )
                )
                success_criteria = SuccessCriteria(
                    minimum_round_profit_cents=max(
                        0, model_criteria.minimum_round_profit_cents
                    ),
                    minimum_cash_reserve_cents=max(
                        int(
                            plan_constraints.get(
                                "minimum_cash_reserve_cents", 0
                            )
                        ),
                        model_criteria.minimum_cash_reserve_cents,
                    ),
                    maximum_fixed_spend_cents=min(
                        plan_max_spend,
                        model_criteria.maximum_fixed_spend_cents
                        if model_criteria.maximum_fixed_spend_cents is not None
                        else plan_max_spend,
                    ),
                    minimum_market_share_ppm=(
                        model_criteria.minimum_market_share_ppm
                    ),
                )
            counterfactual = (
                self.counterfactual_evaluator.evaluate(
                    state_before,
                    state_after,
                    final_actions,
                    company_id,
                )
                if company_id in self.runtimes
                else None
            )
            analysis = self.result_analyzer.analyze(
                state_before,
                state_after,
                company_id,
                expected,
                list(resolution["adjustments"]),
                success_criteria,
                counterfactual,
            )
            persona_profile = result.context.persona_profile if result else None
            persona_utility = (
                self.runtimes[company_id].assess_persona_utility(
                    state_before, state_after
                )
                if company_id in self.runtimes
                else None
            )
            if result is not None and result.success and result.decision is not None:
                model_name = result.model_name
                agent_type = self._agent_type(model_name)
                trace = AgentRoundTrace(
                    company_id=company_id,
                    agent_id=result.agent_id,
                    agent_type=agent_type,
                    decision_status="submitted",
                    observation_hash=information_snapshot.observation_hash,
                    observation=dict(information_snapshot.observation),
                    information_snapshot=information_snapshot,
                    decision_context=result.context.model_dump(mode="json"),
                    communication_view=communication_closure.views[company_id],
                    persona=result.context.persona,
                    persona_catalog_version=(
                        persona_profile.catalog_version
                        if persona_profile is not None
                        else None
                    ),
                    persona_profile_hash=(
                        persona_profile.profile_hash
                        if persona_profile is not None
                        else None
                    ),
                    persona_utility=persona_utility,
                    belief_before=information_snapshot.observation.get(
                        "belief_state"
                    ),
                    opponent_model=information_snapshot.observation.get(
                        "opponent_model_state"
                    ),
                    utility_inference=information_snapshot.observation.get(
                        "utility_inference_state"
                    ),
                    advisor_output=information_snapshot.observation.get(
                        "game_theory_advice"
                    ),
                    repeated_game_strategy=information_snapshot.observation.get(
                        "repeated_game_strategy"
                    ),
                    chosen_action=final_action,
                    counterfactual_results=(
                        {
                            "candidate_actions": information_snapshot.observation[
                                "game_theory_advice"
                            ].get(
                                "candidate_actions",
                                information_snapshot.observation[
                                    "game_theory_advice"
                                ].get("candidates", []),
                            )
                        }
                        if isinstance(
                            information_snapshot.observation.get(
                                "game_theory_advice"
                            ),
                            dict,
                        )
                        else None
                    ),
                    model_name=model_name,
                    prompt_version=result.prompt_version,
                    planner_output=result.decision.plan.model_dump(mode="json"),
                    message_responses=[
                        item.model_dump(mode="json")
                        for item in result.decision.message_responses
                    ],
                    raw_model_output=result.raw_response,
                    requested_action=result.decision.requested_action.model_dump(
                        mode="json"
                    ),
                    intent_id=intent_ids[company_id],
                    final_action=final_action,
                    resolution_source=resolution["source"],
                    resolution_adjustments=list(resolution["adjustments"]),
                    result_analysis=analysis,
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    retry_count=result.retry_count,
                    validation_errors=[
                        str(item.get("reason_code", "adjusted"))
                        for item in resolution["adjustments"]
                    ],
                )
                self.runtimes[company_id].memory.record(
                    result.decision,
                    final_action,
                    analysis,
                    persona_utility,
                )
            else:
                trace = AgentRoundTrace(
                    company_id=company_id,
                    agent_id=(result.agent_id if result else "controller-rule"),
                    agent_type="rule",
                    decision_status="fallback",
                    observation_hash=(
                        information_snapshot.observation_hash
                        if information_snapshot is not None
                        else None
                    ),
                    observation=(
                        dict(information_snapshot.observation)
                        if information_snapshot is not None
                        else None
                    ),
                    information_snapshot=information_snapshot,
                    decision_context=(
                        result.context.model_dump(mode="json")
                        if result is not None
                        else None
                    ),
                    communication_view=communication_closure.views[company_id],
                    persona=(result.context.persona if result else None),
                    persona_catalog_version=(
                        persona_profile.catalog_version
                        if persona_profile is not None
                        else None
                    ),
                    persona_profile_hash=(
                        persona_profile.profile_hash
                        if persona_profile is not None
                        else None
                    ),
                    persona_utility=persona_utility,
                    belief_before=(
                        information_snapshot.observation.get("belief_state")
                        if information_snapshot is not None
                        else None
                    ),
                    opponent_model=(
                        information_snapshot.observation.get(
                            "opponent_model_state"
                        )
                        if information_snapshot is not None
                        else None
                    ),
                    utility_inference=(
                        information_snapshot.observation.get(
                            "utility_inference_state"
                        )
                        if information_snapshot is not None
                        else None
                    ),
                    advisor_output=(
                        information_snapshot.observation.get(
                            "game_theory_advice"
                        )
                        if information_snapshot is not None
                        else None
                    ),
                    repeated_game_strategy=(
                        information_snapshot.observation.get(
                            "repeated_game_strategy"
                        )
                        if information_snapshot is not None
                        else None
                    ),
                    chosen_action=final_action,
                    counterfactual_results=(
                        {
                            "candidate_actions": information_snapshot.observation[
                                "game_theory_advice"
                            ].get(
                                "candidate_actions",
                                information_snapshot.observation[
                                    "game_theory_advice"
                                ].get("candidates", []),
                            )
                        }
                        if information_snapshot is not None
                        and isinstance(
                            information_snapshot.observation.get(
                                "game_theory_advice"
                            ),
                            dict,
                        )
                        else None
                    ),
                    model_name=(result.model_name if result else None),
                    prompt_version=(
                        result.prompt_version if result else None
                    ),
                    planner_output=(
                        result.decision.plan.model_dump(mode="json")
                        if result and result.decision
                        else None
                    ),
                    message_responses=(
                        [
                            item.model_dump(mode="json")
                            for item in result.decision.message_responses
                        ]
                        if result and result.decision
                        else []
                    ),
                    raw_model_output=(result.raw_response if result else ""),
                    requested_action=(
                        result.decision.requested_action.model_dump(mode="json")
                        if result and result.decision
                        else None
                    ),
                    final_action=final_action,
                    resolution_source=resolution["source"],
                    resolution_adjustments=list(resolution["adjustments"]),
                    result_analysis=analysis,
                    latency_ms=(result.latency_ms if result else 0),
                    error_code=(result.error_code if result else None),
                    error_message=(result.error_message if result else None),
                    retry_count=(result.retry_count if result else 0),
                    validation_errors=(
                        [str(result.error_code)]
                        if result is not None and result.error_code
                        else []
                    )
                    + [
                        str(item.get("reason_code", "adjusted"))
                        for item in resolution["adjustments"]
                    ],
                )
                if result is not None:
                    self.runtimes[company_id].memory.record_fallback_outcome(
                        result.decision,
                        final_action,
                        analysis,
                        result.error_code,
                        persona_utility,
                    )
            traces.append(trace)

        event = RoundEvent(
            event_id=f"{episode_id}:agent-round-{state_before.round:02d}",
            episode_id=episode_id,
            settled_round=state_before.round,
            state_before_hash=state_before.state_hash,
            state_after_hash=state_after.state_hash,
            joint_action_hash=sha256_hash(final_actions),
            state_before=state_before.to_dict(),
            state_after=state_after.to_dict(),
            joint_action=final_actions,
            phases=(
                (
                    [
                        "ROUND_OPEN",
                        "OBSERVATION_FROZEN",
                        "COMMUNICATION_CLOSED_NOOP",
                        "COMMITMENTS_GENERATED",
                        "AGENTS_DECIDING",
                        "INTENTS_ACCEPTED",
                        "ACTUAL_CONTRIBUTIONS_LOCKED",
                        "ACTION_LOCKED",
                        "ROUND_SETTLED",
                        "SHARED_RESILIENCE_UPDATED",
                        "COMMITMENTS_VERIFIED",
                        "CREDIBILITY_UPDATED",
                        "FEEDBACK_DISTRIBUTED",
                        "ROUND_LOGGED",
                    ]
                    if cooperation_mode == "shared_resilience_v1"
                    else [
                    "ROUND_OPEN",
                    "OBSERVATION_FROZEN",
                    "COMMUNICATION_CLOSED_NOOP",
                    "AGENTS_DECIDING",
                    "INTENTS_ACCEPTED",
                    "ACTION_LOCKED",
                    "ROUND_SETTLED",
                    "FEEDBACK_DISTRIBUTED",
                    "ROUND_LOGGED",
                    ]
                )
                if communication_mode == "off"
                else (
                    [
                        "ROUND_OPEN",
                        "OBSERVATION_FROZEN",
                        "COMMUNICATION_OPEN",
                        "AGENTS_COMMUNICATING",
                        "MESSAGES_ACCEPTED",
                        "PROPOSALS_PROCESSED",
                        "PROPOSAL_RESPONSES_PROCESSED",
                        "COMMUNICATION_CLOSED",
                        "COMMITMENTS_GENERATED",
                        "DECISION_OBSERVATIONS_FROZEN",
                        "AGENTS_DECIDING",
                        "INTENTS_ACCEPTED",
                        "ACTUAL_CONTRIBUTIONS_LOCKED",
                        "ACTION_LOCKED",
                        "ROUND_SETTLED",
                        "SHARED_RESILIENCE_UPDATED",
                        "COMMITMENTS_VERIFIED",
                        "CREDIBILITY_UPDATED",
                        "FEEDBACK_DISTRIBUTED",
                        "ROUND_LOGGED",
                    ]
                    if cooperation_mode == "shared_resilience_v1"
                    else [
                    "ROUND_OPEN",
                    "OBSERVATION_FROZEN",
                    "COMMUNICATION_OPEN",
                    "AGENTS_COMMUNICATING",
                    "MESSAGES_ACCEPTED",
                    "COMMUNICATION_CLOSED",
                    "DECISION_OBSERVATIONS_FROZEN",
                    "AGENTS_DECIDING",
                    "INTENTS_ACCEPTED",
                    "ACTION_LOCKED",
                    "ROUND_SETTLED",
                    "FEEDBACK_DISTRIBUTED",
                    "ROUND_LOGGED",
                    ]
                )
            ),
            random_draw_summary=dict(
                settlement["step_result"]["random_draw_summary"]
            ),
            step_result=dict(settlement["step_result"]),
            traces=traces,
            communication_phase=communication_phase,
            cooperation_round=cooperation_round,
        )
        if belief_mode != "off":
            observation_index = event.phases.index("OBSERVATION_FROZEN") + 1
            event.phases.insert(observation_index, "BELIEFS_FROZEN")
            feedback_index = event.phases.index("FEEDBACK_DISTRIBUTED")
            event.phases.insert(feedback_index, "BELIEFS_UPDATED")
        authoritative_hash = settlement["step_result"]["joint_action_hash"]
        if event.joint_action_hash != authoritative_hash:
            raise RuntimeError("controller joint action hash does not match RoundEvent")
        if self.event_logger is not None:
            self.event_logger.append(event)
        return CoordinatedRound(settlement=settlement, event=event)

    async def run_episode(
        self, episode_id: str, *, max_rounds: int | None = None
    ) -> tuple[CoordinatedRound, ...]:
        rounds: list[CoordinatedRound] = []
        while max_rounds is None or len(rounds) < max_rounds:
            episode = await self.controller.get_episode(episode_id)
            if episode["state"]["terminal"]:
                break
            rounds.append(await self.run_round(episode_id))
        return tuple(rounds)
