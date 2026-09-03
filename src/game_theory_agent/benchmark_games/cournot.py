"""Integer two-player Cournot benchmark with closed-form reference points."""

from __future__ import annotations

from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.market.protocols import sha256_hash


PPM = 1_000_000


class CournotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "cournot-config-v1.0.0"
    demand_intercept: int = Field(default=30, gt=0)
    marginal_cost: int = Field(default=6, ge=0)
    maximum_quantity: int = Field(default=30, gt=0)

    @model_validator(mode="after")
    def validate_economics(self) -> "CournotConfig":
        if self.marginal_cost >= self.demand_intercept:
            raise ValueError("marginal_cost must be below demand_intercept")
        return self

    @property
    def config_hash(self) -> str:
        return sha256_hash(self.model_dump(mode="json"))


class CournotAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    company_id: str
    quantity: int = Field(ge=0)


class CournotOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "cournot-outcome-v1.0.0"
    price: int
    total_quantity: int
    quantities: dict[str, int]
    profits: dict[str, int]
    consumer_surplus_twice: int
    total_welfare_twice: int


class CournotAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "cournot-advice-v1.0.0"
    company_id: str
    belief_ppm: dict[int, int]
    recommended_quantity: int
    expected_profit_microunits: int
    expected_regret_microunits: int
    candidate_expected_profit_microunits: dict[int, int]
    information_boundary: str = (
        "online advice uses only a distribution over the opponent quantity; "
        "it never sees the opponent's simultaneous current action"
    )


class CournotRoundEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "cournot-round-event-v1.0.0"
    episode_id: str
    round: int = Field(gt=0)
    config: CournotConfig
    actions: tuple[CournotAction, CournotAction]
    outcome: CournotOutcome
    previous_event_hash: str | None = None
    event_hash: str

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        round_number: int,
        config: CournotConfig,
        actions: tuple[CournotAction, CournotAction],
        outcome: CournotOutcome,
        previous_event_hash: str | None = None,
    ) -> "CournotRoundEvent":
        payload = {
            "schema_version": "cournot-round-event-v1.0.0",
            "episode_id": episode_id,
            "round": round_number,
            "config": config.model_dump(mode="json"),
            "actions": [item.model_dump(mode="json") for item in actions],
            "outcome": outcome.model_dump(mode="json"),
            "previous_event_hash": previous_event_hash,
        }
        return cls(**payload, event_hash=sha256_hash(payload))


class CournotEnvironment:
    """Pure simultaneous settlement; there is no hidden market state."""

    def __init__(self, config: CournotConfig | None = None) -> None:
        self.config = config or CournotConfig()

    def settle(self, actions: tuple[CournotAction, CournotAction]) -> CournotOutcome:
        if len({item.company_id for item in actions}) != 2:
            raise ValueError("Cournot benchmark requires two distinct companies")
        if any(item.quantity > self.config.maximum_quantity for item in actions):
            raise ValueError("quantity exceeds maximum_quantity")
        quantities = {item.company_id: item.quantity for item in actions}
        total = sum(quantities.values())
        price = max(self.config.demand_intercept - total, 0)
        profits = {
            company_id: (price - self.config.marginal_cost) * quantity
            for company_id, quantity in quantities.items()
        }
        traded_quantity = min(total, self.config.demand_intercept)
        consumer_surplus_twice = (
            traded_quantity * (self.config.demand_intercept - price)
        )
        producer_surplus_twice = 2 * sum(profits.values())
        return CournotOutcome(
            price=price,
            total_quantity=total,
            quantities=quantities,
            profits=profits,
            consumer_surplus_twice=consumer_surplus_twice,
            total_welfare_twice=consumer_surplus_twice + producer_surplus_twice,
        )

    def profit(self, own_quantity: int, opponent_quantity: int) -> int:
        outcome = self.settle(
            (
                CournotAction(company_id="self", quantity=own_quantity),
                CournotAction(company_id="opponent", quantity=opponent_quantity),
            )
        )
        return outcome.profits["self"]

    def advise(
        self,
        *,
        company_id: str,
        belief_ppm: Mapping[int, int],
    ) -> CournotAdvice:
        normalized = _validate_belief(self.config, belief_ppm)
        candidate_values = {
            quantity: sum(
                probability * self.profit(quantity, opponent_quantity)
                for opponent_quantity, probability in normalized.items()
            )
            for quantity in range(self.config.maximum_quantity + 1)
        }
        best_value = max(candidate_values.values())
        recommended = min(
            quantity for quantity, value in candidate_values.items() if value == best_value
        )
        return CournotAdvice(
            company_id=company_id,
            belief_ppm=dict(normalized),
            recommended_quantity=recommended,
            expected_profit_microunits=best_value,
            expected_regret_microunits=0,
            candidate_expected_profit_microunits=candidate_values,
        )

    def regret(self, own_quantity: int, opponent_quantity: int) -> int:
        best = max(
            self.profit(candidate, opponent_quantity)
            for candidate in range(self.config.maximum_quantity + 1)
        )
        return best - self.profit(own_quantity, opponent_quantity)


def _validate_belief(
    config: CournotConfig, belief_ppm: Mapping[int, int]
) -> dict[int, int]:
    if not belief_ppm:
        raise ValueError("belief_ppm cannot be empty")
    normalized = {int(quantity): int(probability) for quantity, probability in belief_ppm.items()}
    if any(quantity < 0 or quantity > config.maximum_quantity for quantity in normalized):
        raise ValueError("belief quantity is outside the action space")
    if any(probability < 0 for probability in normalized.values()):
        raise ValueError("belief probabilities cannot be negative")
    if sum(normalized.values()) != PPM:
        raise ValueError("belief_ppm must sum to 1000000")
    return dict(sorted(normalized.items()))


def exact_best_response(opponent_quantity: int, config: CournotConfig | None = None) -> int:
    """Return the smallest maximizing integer quantity under the configured game."""

    env = CournotEnvironment(config)
    values = {
        quantity: env.profit(quantity, opponent_quantity)
        for quantity in range(env.config.maximum_quantity + 1)
    }
    best = max(values.values())
    return min(quantity for quantity, value in values.items() if value == best)
