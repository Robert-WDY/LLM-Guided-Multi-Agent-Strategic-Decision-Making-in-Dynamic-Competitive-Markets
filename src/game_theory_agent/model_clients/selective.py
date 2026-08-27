"""Use a paid provider only on preregistered decision rounds."""

from __future__ import annotations

from game_theory_agent.agents.contracts import (
    AgentDecision,
    AgentRequestedAction,
    CommunicationContext,
    DecisionContext,
    ExpectedOutcome,
    ModelGeneration,
    StrategyPlan,
    SuccessCriteria,
)
from game_theory_agent.strategic_reliability.cost_guard import RealModelCostGuard

from .base import ModelClient


class FixedEconomicBaselineModelClient:
    """Condition-invariant unpaid policy for matched causal experiments."""

    async def generate_communication(
        self, context: CommunicationContext
    ) -> ModelGeneration:
        from game_theory_agent.interaction import CommunicationSubmission

        submission = CommunicationSubmission()
        return ModelGeneration(
            model_name="fixed-unpaid-baseline-v1",
            prompt_version="fixed-unpaid-communication-v1",
            parsed_output=submission.model_dump(mode="json"),
            raw_response=submission.model_dump_json(),
            latency_ms=0,
        )

    async def generate_decision(self, context: DecisionContext) -> ModelGeneration:
        price = int(context.own_company["commercial"]["price_cents"])
        decision = AgentDecision(
            plan=StrategyPlan(
                objective="保持共同状态直到预注册真实模型决策轮",
                situation_summary="非付费轮使用条件不变的零增量基线。",
                key_factors=["共同状态", "无战略处理"],
                strategy_summary="维持当前价格且不增加可选投入",
                expected_outcome=ExpectedOutcome(),
                success_criteria=SuccessCriteria(),
            ),
            requested_action=AgentRequestedAction(price_cents=price),
            confidence_ppm=1_000_000,
        )
        return ModelGeneration(
            model_name="fixed-unpaid-baseline-v1",
            prompt_version="fixed-unpaid-decision-v1",
            parsed_output=decision.model_dump(mode="json"),
            raw_response=decision.model_dump_json(),
            latency_ms=0,
        )


class SelectiveDecisionModelClient:
    def __init__(
        self,
        paid_client: ModelClient,
        fallback_client: ModelClient,
        *,
        paid_rounds: tuple[int, ...],
        cost_guard: RealModelCostGuard,
        reserved_prompt_tokens_per_call: int = 25_000,
        reserved_completion_tokens_per_call: int = 2_000,
        input_price_microunits_per_token: int = 1,
        output_price_microunits_per_token: int = 4,
    ) -> None:
        if not paid_rounds or min(paid_rounds) < 1:
            raise ValueError("paid_rounds must contain positive rounds")
        self.paid_client = paid_client
        self.fallback_client = fallback_client
        self.paid_rounds = frozenset(paid_rounds)
        self.cost_guard = cost_guard
        self.reserved_prompt_tokens_per_call = (
            reserved_prompt_tokens_per_call
        )
        self.reserved_completion_tokens_per_call = (
            reserved_completion_tokens_per_call
        )
        self.input_price_microunits_per_token = (
            input_price_microunits_per_token
        )
        self.output_price_microunits_per_token = (
            output_price_microunits_per_token
        )

    async def generate_communication(
        self, context: CommunicationContext
    ) -> ModelGeneration:
        return await self.fallback_client.generate_communication(context)

    async def generate_decision(self, context: DecisionContext) -> ModelGeneration:
        if context.round not in self.paid_rounds:
            return await self.fallback_client.generate_decision(context)
        reserved_cost = (
            self.reserved_prompt_tokens_per_call
            * self.input_price_microunits_per_token
            + self.reserved_completion_tokens_per_call
            * self.output_price_microunits_per_token
        )
        self.cost_guard.reserve(
            prompt_tokens=self.reserved_prompt_tokens_per_call,
            completion_tokens=self.reserved_completion_tokens_per_call,
            estimated_cost_microunits=reserved_cost,
        )
        result = await self.paid_client.generate_decision(context)
        prompt_tokens = int(result.input_tokens or 0)
        completion_tokens = int(result.output_tokens or 0)
        actual_cost = (
            prompt_tokens * self.input_price_microunits_per_token
            + completion_tokens * self.output_price_microunits_per_token
        )
        self.cost_guard.record_actual(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_microunits=actual_cost,
        )
        return result


class BudgetedModelClient:
    """Apply one shared fail-closed budget to communication and decisions."""

    def __init__(
        self,
        paid_client: ModelClient,
        *,
        cost_guard: RealModelCostGuard,
        reserved_prompt_tokens_per_call: int = 32_000,
        reserved_completion_tokens_per_call: int = 4_000,
        input_price_microunits_per_token: int = 2,
        output_price_microunits_per_token: int = 4,
    ) -> None:
        self.paid_client = paid_client
        self.cost_guard = cost_guard
        self.reserved_prompt_tokens_per_call = reserved_prompt_tokens_per_call
        self.reserved_completion_tokens_per_call = reserved_completion_tokens_per_call
        self.input_price_microunits_per_token = input_price_microunits_per_token
        self.output_price_microunits_per_token = output_price_microunits_per_token

    @property
    def model(self) -> str | None:
        return getattr(self.paid_client, "model", None)

    def _reserve(self) -> None:
        self.cost_guard.reserve(
            prompt_tokens=self.reserved_prompt_tokens_per_call,
            completion_tokens=self.reserved_completion_tokens_per_call,
            estimated_cost_microunits=(
                self.reserved_prompt_tokens_per_call
                * self.input_price_microunits_per_token
                + self.reserved_completion_tokens_per_call
                * self.output_price_microunits_per_token
            ),
        )

    def _record(self, result: ModelGeneration) -> ModelGeneration:
        prompt_tokens = int(result.input_tokens or 0)
        completion_tokens = int(result.output_tokens or 0)
        self.cost_guard.record_actual(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_microunits=(
                prompt_tokens * self.input_price_microunits_per_token
                + completion_tokens * self.output_price_microunits_per_token
            ),
        )
        return result

    async def generate_communication(
        self, context: CommunicationContext
    ) -> ModelGeneration:
        self._reserve()
        return self._record(await self.paid_client.generate_communication(context))

    async def generate_decision(self, context: DecisionContext) -> ModelGeneration:
        self._reserve()
        return self._record(await self.paid_client.generate_decision(context))
