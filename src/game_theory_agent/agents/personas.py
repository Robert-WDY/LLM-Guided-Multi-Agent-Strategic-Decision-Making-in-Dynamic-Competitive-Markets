"""Versioned Agent-side persona profiles and deterministic utility evaluation.

Persona profiles are experiment inputs. They never change market formulas or
execution guardrails; they only guide planning and evaluate realized outcomes.
"""

from __future__ import annotations

import os
from functools import lru_cache
from math import isqrt
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.market import MarketConfig, MarketState, load_market_config
from game_theory_agent.market.protocols import sha256_hash


PPM = 1_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PersonaUtilityWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profit: int = Field(ge=0, le=PPM)
    share: int = Field(ge=0, le=PPM)
    growth: int = Field(ge=0, le=PPM)
    stability: int = Field(ge=0, le=PPM)
    cash: int = Field(ge=0, le=PPM)
    reputation: int = Field(ge=0, le=PPM)
    resilience: int = Field(ge=0, le=PPM)
    # Reserved Research-MVP interfaces. Both capabilities are disabled today.
    social_welfare: int = Field(default=0, ge=0, le=PPM)
    cooperation_reputation: int = Field(default=0, ge=0, le=PPM)

    @model_validator(mode="after")
    def validate_total(self) -> "PersonaUtilityWeights":
        if sum(self.model_dump().values()) != PPM:
            raise ValueError("persona utility weights must sum to 1000000")
        return self


class PersonaTraits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    time_discount: int = Field(ge=0, le=PPM)
    risk_aversion: int = Field(ge=0, le=PPM)
    reciprocity: int = Field(ge=0, le=PPM)
    commitment_honesty: int = Field(ge=0, le=PPM)
    opportunism: int = Field(ge=0, le=PPM)


class PersonaProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_schema_version: Literal["persona-profile-v1.0.0"] = (
        "persona-profile-v1.0.0"
    )
    persona_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    catalog_version: str
    label: str = Field(min_length=1, max_length=120)
    objective: str = Field(min_length=1, max_length=500)
    utility_weights_ppm: PersonaUtilityWeights
    traits_ppm: PersonaTraits
    social_welfare_enabled: bool = False
    cooperation_enabled: bool = False

    @model_validator(mode="after")
    def validate_capabilities(self) -> "PersonaProfile":
        if not self.social_welfare_enabled and self.utility_weights_ppm.social_welfare:
            raise ValueError("social welfare weight requires its capability")
        if (
            not self.cooperation_enabled
            and self.utility_weights_ppm.cooperation_reputation
        ):
            raise ValueError("cooperation reputation weight requires its capability")
        return self

    @property
    def profile_hash(self) -> str:
        return sha256_hash(self.model_dump(mode="json"))

    def manifest_dict(self) -> dict[str, object]:
        return {
            **self.model_dump(mode="json"),
            "profile_hash": self.profile_hash,
        }


class PersonaUtilityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_schema_version: Literal["persona-utility-v1.0.0"] = (
        "persona-utility-v1.0.0"
    )
    persona_id: str
    persona_profile_hash: str
    component_scores_ppm: dict[str, int]
    weighted_contributions_ppm: dict[str, int]
    round_utility_ppm: int
    discount_multiplier_ppm: int = Field(ge=0, le=PPM)
    discounted_round_utility_ppm: int
    cumulative_discounted_utility_ppm: int
    terminal_enterprise_value_cents: int | None = None
    realized_incident_loss_cents: int = 0
    realized_unserved_contribution_loss_cents: int = 0
    realized_risk_adjusted_terminal_value_cents: int | None = None
    efficiency_and_risk_are_weighted_utility_components: bool = False
    social_welfare_available: bool = False
    cooperation_available: bool = False


class PersonaRegistry:
    """Validated persona catalogue derived from the episode market config."""

    def __init__(
        self,
        *,
        catalog_version: str,
        default_profile_id: str,
        profit_scale_cents: int,
        share_growth_scale_ppm: int,
        profiles: dict[str, PersonaProfile],
    ) -> None:
        if default_profile_id not in profiles:
            raise ValueError("default persona profile is unknown")
        if profit_scale_cents <= 0 or share_growth_scale_ppm <= 0:
            raise ValueError("persona normalization scales must be positive")
        self.catalog_version = catalog_version
        self.default_profile_id = default_profile_id
        self.profit_scale_cents = profit_scale_cents
        self.share_growth_scale_ppm = share_growth_scale_ppm
        self._profiles = dict(profiles)

    @classmethod
    def from_market_config(cls, config: MarketConfig) -> "PersonaRegistry":
        raw = config.mapping("persona_utilities")
        weights = config.mapping("persona_utilities", "weights_ppm")
        traits = config.mapping("persona_utilities", "traits_ppm")
        labels = config.mapping("persona_utilities", "labels")
        objectives = config.mapping("persona_utilities", "objectives")
        capabilities = config.mapping("persona_utilities", "capabilities")
        catalog_version = str(raw["schema_version"])
        profiles = {
            persona_id: PersonaProfile(
                persona_id=persona_id,
                catalog_version=catalog_version,
                label=str(labels[persona_id]),
                objective=str(objectives[persona_id]),
                utility_weights_ppm=PersonaUtilityWeights.model_validate(
                    dict(persona_weights)
                ),
                traits_ppm=PersonaTraits.model_validate(dict(traits[persona_id])),
                social_welfare_enabled=bool(capabilities["social_welfare"]),
                cooperation_enabled=bool(capabilities["cooperation"]),
            )
            for persona_id, persona_weights in weights.items()
        }
        return cls(
            catalog_version=catalog_version,
            default_profile_id=str(raw["default_profile_id"]),
            profit_scale_cents=int(raw["profit_scale_cents"]),
            share_growth_scale_ppm=int(raw["share_growth_scale_ppm"]),
            profiles=profiles,
        )

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def get(self, persona_id: str | None = None) -> PersonaProfile:
        selected = persona_id or self.default_profile_id
        try:
            return self._profiles[selected]
        except KeyError as exc:
            raise ValueError(
                f"unknown persona {selected!r}; expected one of {self.profile_ids}"
            ) from exc

    def evaluator(self, profile: PersonaProfile) -> "PersonaUtilityEvaluator":
        return PersonaUtilityEvaluator(
            profile,
            profit_scale_cents=self.profit_scale_cents,
            share_growth_scale_ppm=self.share_growth_scale_ppm,
        )


@lru_cache(maxsize=4)
def load_persona_registry(config_path: str | Path | None = None) -> PersonaRegistry:
    resolved = Path(
        config_path
        or os.environ.get(
            "MARKET_CONFIG_PATH", PROJECT_ROOT / "configs" / "market_v4.yaml"
        )
    ).resolve()
    return PersonaRegistry.from_market_config(load_market_config(resolved))


def _round_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


def _clip(value: int, low: int, high: int) -> int:
    return min(max(value, low), high)


def _population_std(values: tuple[int, ...]) -> int:
    if len(values) < 2:
        return 0
    count = len(values)
    total = sum(values)
    # Exact population variance around the rational mean, rounded down before
    # integer square root. No floating-point value enters research logs.
    scaled_squared_error = sum((value * count - total) ** 2 for value in values)
    variance = scaled_squared_error // (count**3)
    return isqrt(variance)


class PersonaUtilityEvaluator:
    def __init__(
        self,
        profile: PersonaProfile,
        *,
        profit_scale_cents: int,
        share_growth_scale_ppm: int,
    ) -> None:
        self.profile = profile
        self.profit_scale_cents = profit_scale_cents
        self.share_growth_scale_ppm = share_growth_scale_ppm

    def component_scores(
        self,
        state_before: MarketState,
        state_after: MarketState,
        company_id: str,
    ) -> dict[str, int]:
        before = state_before.company(company_id)
        after = state_after.company(company_id)
        profits = tuple(after.history.recent_profit_cents[-3:])
        stability = (
            PPM
            if len(profits) < 2
            else PPM
            - min(
                PPM,
                _round_div(_population_std(profits) * PPM, self.profit_scale_cents),
            )
        )
        initial_cash = max(
            1,
            before.financial.cash_balance_cents
            - before.financial.cumulative_profit_cents,
        )
        return {
            "profit": _clip(
                _round_div(
                    after.financial.round_profit_cents * PPM,
                    self.profit_scale_cents,
                ),
                -PPM,
                PPM,
            ),
            "share": _clip(after.commercial.market_share_ppm, 0, PPM),
            "growth": _clip(
                _round_div(
                    (
                        after.commercial.market_share_ppm
                        - before.commercial.market_share_ppm
                    )
                    * PPM,
                    self.share_growth_scale_ppm,
                ),
                -PPM,
                PPM,
            ),
            "stability": _clip(stability, 0, PPM),
            "cash": _clip(
                _round_div(
                    after.financial.cash_balance_cents * PPM,
                    2 * initial_cash,
                ),
                0,
                PPM,
            ),
            "reputation": _clip(after.brand.reputation_ppm, 0, PPM),
            "resilience": _clip(after.risk.resilience_ppm, 0, PPM),
            "social_welfare": 0,
            "cooperation_reputation": 0,
        }

    def evaluate(
        self,
        state_before: MarketState,
        state_after: MarketState,
        company_id: str,
        *,
        discount_multiplier_ppm: int = PPM,
        cumulative_before_ppm: int = 0,
    ) -> PersonaUtilityAssessment:
        scores = self.component_scores(state_before, state_after, company_id)
        weights = self.profile.utility_weights_ppm.model_dump()
        contributions = {
            name: _round_div(int(weights[name]) * score, PPM)
            for name, score in scores.items()
        }
        round_utility = _round_div(
            sum(int(weights[name]) * score for name, score in scores.items()), PPM
        )
        discounted = _round_div(round_utility * discount_multiplier_ppm, PPM)
        terminal_values = dict(state_after.terminal_enterprise_values_cents)
        after = state_after.company(company_id)
        sales = after.commercial.sales_orders
        realized_unit_contribution = (
            max(
                0,
                (
                    after.financial.round_revenue_cents
                    - after.financial.round_variable_cost_cents
                    - after.financial.round_operating_cost_cents
                )
                // sales,
            )
            if sales > 0
            else 0
        )
        unserved_loss = (
            after.commercial.attempted_unfulfilled_orders
            * realized_unit_contribution
        )
        incident_loss = after.financial.round_incident_cost_cents
        terminal_value = terminal_values.get(company_id)
        return PersonaUtilityAssessment(
            persona_id=self.profile.persona_id,
            persona_profile_hash=self.profile.profile_hash,
            component_scores_ppm=scores,
            weighted_contributions_ppm=contributions,
            round_utility_ppm=round_utility,
            discount_multiplier_ppm=discount_multiplier_ppm,
            discounted_round_utility_ppm=discounted,
            cumulative_discounted_utility_ppm=cumulative_before_ppm + discounted,
            terminal_enterprise_value_cents=terminal_value,
            realized_incident_loss_cents=incident_loss,
            realized_unserved_contribution_loss_cents=unserved_loss,
            realized_risk_adjusted_terminal_value_cents=(
                terminal_value - incident_loss - unserved_loss
                if terminal_value is not None
                else None
            ),
            social_welfare_available=self.profile.social_welfare_enabled,
            cooperation_available=self.profile.cooperation_enabled,
        )


class PersonaUtilityTracker:
    """Episode-local discounted utility accumulator for one Agent Runtime."""

    def __init__(self, evaluator: PersonaUtilityEvaluator) -> None:
        self.evaluator = evaluator
        self.discount_multiplier_ppm = PPM
        self.cumulative_discounted_utility_ppm = 0

    def record(
        self,
        state_before: MarketState,
        state_after: MarketState,
        company_id: str,
    ) -> PersonaUtilityAssessment:
        assessment = self.evaluator.evaluate(
            state_before,
            state_after,
            company_id,
            discount_multiplier_ppm=self.discount_multiplier_ppm,
            cumulative_before_ppm=self.cumulative_discounted_utility_ppm,
        )
        self.cumulative_discounted_utility_ppm = (
            assessment.cumulative_discounted_utility_ppm
        )
        self.discount_multiplier_ppm = _round_div(
            self.discount_multiplier_ppm
            * self.evaluator.profile.traits_ppm.time_discount,
            PPM,
        )
        return assessment
