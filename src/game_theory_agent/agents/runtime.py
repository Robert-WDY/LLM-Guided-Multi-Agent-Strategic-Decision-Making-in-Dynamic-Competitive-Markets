"""A single Agent's read-decide-validate lifecycle."""

from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from game_theory_agent.agents.context import DecisionContextBuilder
from game_theory_agent.agents.contracts import (
    AgentCommunicationResult,
    AgentDecision,
    AgentDecisionResult,
    CommunicationContext,
    DecisionContext,
    MessageReferenceValidationError,
    validate_decision_message_references,
)
from game_theory_agent.agents.memory import EpisodeMemory
from game_theory_agent.agents.personas import (
    PersonaProfile,
    PersonaRegistry,
    PersonaUtilityAssessment,
    PersonaUtilityTracker,
    load_persona_registry,
)
from game_theory_agent.market.models import MarketState
from game_theory_agent.interaction.contracts import (
    CommunicationMode,
    CommunicationSubmission,
    CommunicationView,
)
from game_theory_agent.model_clients.base import ModelClient


class AgentRuntime:
    """Owns no market authority; it can only produce a validated intent."""

    def __init__(
        self,
        agent_id: str,
        company_id: str,
        model_client: ModelClient,
        *,
        memory: EpisodeMemory | None = None,
        context_builder: DecisionContextBuilder | None = None,
        persona_profile: PersonaProfile | None = None,
        persona_registry: PersonaRegistry | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.company_id = company_id
        self.model_client = model_client
        self.memory = memory or EpisodeMemory()
        registry = persona_registry or load_persona_registry()
        if (
            context_builder is not None
            and persona_profile is not None
            and context_builder.persona_profile != persona_profile
        ):
            raise ValueError(
                "persona_profile conflicts with the supplied context_builder"
            )
        self.context_builder = context_builder or DecisionContextBuilder(
            persona_profile=persona_profile,
            persona_registry=registry,
        )
        self.persona_registry = self.context_builder.persona_registry
        self._utility_tracker: PersonaUtilityTracker | None = None
        self._utility_episode_id: str | None = None

    async def communicate(
        self,
        observation: dict[str, object],
        *,
        communication_mode: CommunicationMode | None = None,
        timeout_seconds: float = 30.0,
    ) -> AgentCommunicationResult:
        """Generate one validated submission; every failure becomes silence."""

        context = self.context_builder.build_communication(
            observation,
            self.company_id,
            self.memory,
            communication_mode=communication_mode,
        )
        started = time.perf_counter()
        if context.communication_mode == "off":
            return AgentCommunicationResult(
                success=True,
                agent_id=self.agent_id,
                company_id=self.company_id,
                context=context,
                submission=CommunicationSubmission(),
                is_silence=True,
                silence_reason="communication_disabled",
                latency_ms=round((time.perf_counter() - started) * 1000),
            )

        generator = getattr(self.model_client, "generate_communication", None)
        if generator is None or not callable(generator):
            return self._communication_failure(
                context,
                started,
                "UNSUPPORTED_MODEL_CLIENT",
                "model client does not implement generate_communication",
                "unsupported_client",
            )
        try:
            generation = await asyncio.wait_for(
                generator(context), timeout=timeout_seconds
            )
            submission = CommunicationSubmission.model_validate(
                generation.parsed_output
            )
            self._validate_communication_submission(context, submission)
            latency_ms = generation.latency_ms or round(
                (time.perf_counter() - started) * 1000
            )
            return AgentCommunicationResult(
                success=True,
                agent_id=self.agent_id,
                company_id=self.company_id,
                context=context,
                submission=submission,
                is_silence=not submission.messages,
                silence_reason=(
                    "intentional" if not submission.messages else "not_silent"
                ),
                model_name=generation.model_name,
                prompt_version=generation.prompt_version,
                raw_response=generation.raw_response,
                latency_ms=latency_ms,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                retry_count=generation.retry_count,
            )
        except TimeoutError:
            return self._communication_failure(
                context,
                started,
                "COMMUNICATION_MODEL_TIMEOUT",
                "communication model timed out",
                "model_timeout",
            )
        except (ValidationError, ValueError) as exc:
            return self._communication_failure(
                context,
                started,
                "INVALID_COMMUNICATION_OUTPUT",
                str(exc),
                "invalid_model_output",
            )
        except Exception as exc:
            return self._communication_failure(
                context,
                started,
                "COMMUNICATION_MODEL_ERROR",
                f"{type(exc).__name__}: {exc}",
                "model_error",
            )

    async def decide(
        self,
        observation: dict[str, object],
        *,
        communication_view: CommunicationView | dict[str, object] | None = None,
        timeout_seconds: float = 45.0,
    ) -> AgentDecisionResult:
        context = self.context_builder.build(
            observation,
            self.company_id,
            self.memory,
            communication_view=communication_view,
        )
        if (
            self._utility_tracker is None
            or self._utility_episode_id != context.episode_id
            or self._utility_tracker.evaluator.profile.profile_hash
            != context.persona_profile.profile_hash
        ):
            self._utility_tracker = PersonaUtilityTracker(
                self.persona_registry.evaluator(context.persona_profile)
            )
            self._utility_episode_id = context.episode_id
        started = time.perf_counter()
        try:
            generation = await asyncio.wait_for(
                self.model_client.generate_decision(context),
                timeout=timeout_seconds,
            )
            decision = AgentDecision.model_validate(generation.parsed_output)
            validate_decision_message_references(
                decision, context.communication_view
            )
            latency_ms = generation.latency_ms or round(
                (time.perf_counter() - started) * 1000
            )
            return AgentDecisionResult(
                success=True,
                agent_id=self.agent_id,
                company_id=self.company_id,
                context=context,
                decision=decision,
                model_name=generation.model_name,
                prompt_version=generation.prompt_version,
                raw_response=generation.raw_response,
                latency_ms=latency_ms,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                retry_count=generation.retry_count,
            )
        except TimeoutError:
            return self._failure(context, started, "MODEL_TIMEOUT", "model timed out")
        except (ValidationError, MessageReferenceValidationError) as exc:
            return self._failure(
                context, started, "INVALID_MODEL_OUTPUT", str(exc)
            )
        except Exception as exc:  # provider failures are converted to rule fallback
            return self._failure(
                context,
                started,
                "MODEL_ERROR",
                f"{type(exc).__name__}: {exc}",
            )

    def _failure(
        self,
        context: DecisionContext,
        started: float,
        code: str,
        message: str,
    ) -> AgentDecisionResult:
        return AgentDecisionResult(
            success=False,
            agent_id=self.agent_id,
            company_id=self.company_id,
            context=context,
            latency_ms=round((time.perf_counter() - started) * 1000),
            error_code=code,
            error_message=message[:2000],
            fallback_required=True,
        )

    def _communication_failure(
        self,
        context: CommunicationContext,
        started: float,
        code: str,
        message: str,
        silence_reason: str,
    ) -> AgentCommunicationResult:
        return AgentCommunicationResult(
            success=False,
            agent_id=self.agent_id,
            company_id=self.company_id,
            context=context,
            submission=CommunicationSubmission(),
            is_silence=True,
            silence_reason=silence_reason,
            latency_ms=round((time.perf_counter() - started) * 1000),
            error_code=code,
            error_message=message[:2000],
            fallback_to_silence=True,
        )

    @staticmethod
    def _validate_communication_submission(
        context: CommunicationContext,
        submission: CommunicationSubmission,
    ) -> None:
        eligible = set(context.eligible_recipient_company_ids)
        constraints = context.action_claim_constraints
        for message in submission.messages:
            if (
                context.communication_mode == "public_only"
                and message.channel == "private"
            ):
                raise ValueError("private messages are disabled in public_only mode")
            if message.channel == "private" and message.recipients[0] not in eligible:
                raise ValueError("private recipient is not visible as an eligible peer")
            for claim in (message.own_action_claim, message.requested_peer_action):
                if claim is None:
                    continue
                for field_name, value in claim.model_dump().items():
                    if value is None or field_name not in constraints:
                        continue
                    bounds = constraints[field_name]
                    if not int(bounds["min"]) <= int(value) <= int(bounds["max"]):
                        raise ValueError(
                            f"{field_name} claim is outside current action bounds"
                        )

    @property
    def persona_profile(self) -> PersonaProfile:
        return self.context_builder.persona_profile or self.persona_registry.get()

    def persona_manifest(self) -> dict[str, object]:
        return self.persona_profile.manifest_dict()

    def assess_persona_utility(
        self,
        state_before: MarketState,
        state_after: MarketState,
    ) -> PersonaUtilityAssessment:
        if self._utility_tracker is None:
            profile = self.persona_profile
            self._utility_tracker = PersonaUtilityTracker(
                self.persona_registry.evaluator(profile)
            )
            self._utility_episode_id = state_before.episode_id
        if self._utility_episode_id != state_before.episode_id:
            raise ValueError("AgentRuntime utility tracker belongs to another episode")
        return self._utility_tracker.record(
            state_before, state_after, self.company_id
        )
