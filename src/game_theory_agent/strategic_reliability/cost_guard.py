"""Fail-closed budget guard for future Stage 6 real-model experiments."""

from __future__ import annotations

from dataclasses import dataclass


class RealModelBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RealModelBudget:
    max_calls: int
    max_prompt_tokens: int
    max_completion_tokens: int
    max_estimated_cost_microunits: int
    currency: str = "CNY"

    def __post_init__(self) -> None:
        if min(
            self.max_calls,
            self.max_prompt_tokens,
            self.max_completion_tokens,
            self.max_estimated_cost_microunits,
        ) < 0:
            raise ValueError("real-model budget limits cannot be negative")


@dataclass(frozen=True, slots=True)
class RealModelUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_microunits: int = 0


class RealModelCostGuard:
    """Reserve estimated usage before every provider request.

    The guard does not contain provider prices because prices change.  A real
    experiment runner must calculate the estimate from a pinned price snapshot
    and pass it here.  Missing or zero price estimates should therefore be an
    explicit experiment decision, never an implicit network call.
    """

    def __init__(
        self,
        budget: RealModelBudget,
        *,
        explicitly_authorized: bool = False,
    ) -> None:
        self.budget = budget
        self.explicitly_authorized = explicitly_authorized
        self._reserved = RealModelUsage()
        self._actual = RealModelUsage()

    @property
    def reserved(self) -> RealModelUsage:
        return self._reserved

    @property
    def actual(self) -> RealModelUsage:
        return self._actual

    def reserve(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost_microunits: int,
    ) -> RealModelUsage:
        if not self.explicitly_authorized:
            raise RealModelBudgetExceeded(
                "real-model calls require explicit experiment authorization"
            )
        if min(prompt_tokens, completion_tokens, estimated_cost_microunits) < 0:
            raise ValueError("reserved model usage cannot be negative")
        proposed = RealModelUsage(
            calls=self._reserved.calls + 1,
            prompt_tokens=self._reserved.prompt_tokens + prompt_tokens,
            completion_tokens=(
                self._reserved.completion_tokens + completion_tokens
            ),
            estimated_cost_microunits=(
                self._reserved.estimated_cost_microunits
                + estimated_cost_microunits
            ),
        )
        self._validate(proposed)
        self._reserved = proposed
        return proposed

    def record_actual(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost_microunits: int,
    ) -> RealModelUsage:
        if min(prompt_tokens, completion_tokens, estimated_cost_microunits) < 0:
            raise ValueError("actual model usage cannot be negative")
        proposed = RealModelUsage(
            calls=self._actual.calls + 1,
            prompt_tokens=self._actual.prompt_tokens + prompt_tokens,
            completion_tokens=self._actual.completion_tokens + completion_tokens,
            estimated_cost_microunits=(
                self._actual.estimated_cost_microunits
                + estimated_cost_microunits
            ),
        )
        self._validate(proposed)
        if (
            proposed.calls > self._reserved.calls
            or proposed.prompt_tokens > self._reserved.prompt_tokens
            or proposed.completion_tokens > self._reserved.completion_tokens
            or proposed.estimated_cost_microunits
            > self._reserved.estimated_cost_microunits
        ):
            raise RealModelBudgetExceeded(
                "actual provider usage exceeds the amount reserved before execution"
            )
        self._actual = proposed
        return proposed

    def _validate(self, usage: RealModelUsage) -> None:
        limits = (
            ("calls", usage.calls, self.budget.max_calls),
            (
                "prompt_tokens",
                usage.prompt_tokens,
                self.budget.max_prompt_tokens,
            ),
            (
                "completion_tokens",
                usage.completion_tokens,
                self.budget.max_completion_tokens,
            ),
            (
                "estimated_cost_microunits",
                usage.estimated_cost_microunits,
                self.budget.max_estimated_cost_microunits,
            ),
        )
        for name, value, limit in limits:
            if value > limit:
                raise RealModelBudgetExceeded(
                    f"real-model {name} budget exceeded: {value} > {limit}"
                )
