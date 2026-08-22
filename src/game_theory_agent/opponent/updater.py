"""Deterministic rule-Bayesian updater over public market evidence."""

from __future__ import annotations

from collections import Counter
from typing import Mapping

from game_theory_agent.market.models import CompanyAction, MarketState
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.opponent.schema import (
    OpponentBehaviorProfile,
    OpponentModelState,
    OpponentStrategyModel,
    PublicStrategyEvidence,
    StrategyDistribution,
    StrategyType,
    compute_opponent_model_hash,
)


_STRATEGIES: tuple[StrategyType, ...] = (
    "growth",
    "profit",
    "defensive",
    "cooperative",
)


def _direction(previous: int, current: int) -> str:
    if current < previous:
        return "price_cut"
    if current > previous:
        return "price_raise"
    return "maintain"


def _ratio_ppm(successes: int, count: int) -> int:
    return (successes + 1) * 1_000_000 // (count + 2)


def _distribution(scores: Mapping[StrategyType, int]) -> StrategyDistribution:
    values = [int(scores[item]) for item in _STRATEGIES]
    total = sum(values)
    raw = [value * 1_000_000 for value in values]
    allocated = [value // total for value in raw]
    remaining = 1_000_000 - sum(allocated)
    order = sorted(
        range(len(values)),
        key=lambda index: (raw[index] % total, -index),
        reverse=True,
    )
    for index in order[:remaining]:
        allocated[index] += 1
    return StrategyDistribution(
        growth_ppm=allocated[0],
        profit_ppm=allocated[1],
        defensive_ppm=allocated[2],
        cooperative_ppm=allocated[3],
    )


def build_strategy_model(
    company_id: str, evidence: list[PublicStrategyEvidence]
) -> OpponentStrategyModel:
    """Build one deterministic public-evidence strategy distribution."""

    count = len(evidence)
    directions = Counter(item.price_direction for item in evidence)
    expansion = sum(
        item.market_share_delta_ppm > 0 or item.public_sales_orders > 3_500
        for item in evidence
    )
    risky = sum(
        item.price_direction == "price_cut"
        and item.reputation_delta_ppm <= 0
        for item in evidence
    )
    cooperative = sum(
        item.public_shared_resilience_contribution_cents > 0
        for item in evidence
    )
    scores: dict[StrategyType, int] = {
        strategy: 2 for strategy in _STRATEGIES
    }
    for item in evidence:
        if item.price_direction == "price_cut":
            scores["growth"] += 4
        elif item.price_direction == "price_raise":
            scores["profit"] += 4
        else:
            scores["profit"] += 2
            scores["defensive"] += 2
        if item.market_share_delta_ppm > 0:
            scores["growth"] += 3
        elif item.market_share_delta_ppm < 0:
            scores["defensive"] += 2
        if item.reputation_delta_ppm > 0:
            scores["defensive"] += 2
        elif item.reputation_delta_ppm < 0:
            scores["growth"] += 1
        if item.public_shared_resilience_contribution_cents > 0:
            scores["cooperative"] += 8
        else:
            scores["profit"] += 1
    latest = evidence[-1] if evidence else None
    return OpponentStrategyModel(
        opponent_company_id=company_id,
        evidence_count=count,
        latest_evidence_round=(latest.settled_round if latest else None),
        behavior_profile=OpponentBehaviorProfile(
            price_aggressiveness_ppm=_ratio_ppm(
                directions["price_cut"], count
            ),
            public_expansion_aggressiveness_ppm=_ratio_ppm(expansion, count),
            risk_tolerance_ppm=_ratio_ppm(risky, count),
            cooperation_tendency_ppm=_ratio_ppm(cooperative, count),
        ),
        strategy_distribution=_distribution(scores),
        confidence_ppm=count * 1_000_000 // (count + 3),
        public_evidence_ids=[item.evidence_id for item in evidence],
    )


class OpponentModelLedger:
    """Episode authority for public-only opponent strategy inference."""

    def __init__(self, *, episode_id: str, company_ids: tuple[str, ...]) -> None:
        if len(company_ids) < 2 or len(set(company_ids)) != len(company_ids):
            raise ValueError("opponent model requires unique companies")
        self.episode_id = episode_id
        self.company_ids = tuple(company_ids)
        self._evidence: dict[str, list[PublicStrategyEvidence]] = {
            company_id: [] for company_id in company_ids
        }
        self._settlement_hashes: dict[int, str] = {}

    @staticmethod
    def _model(
        company_id: str, evidence: list[PublicStrategyEvidence]
    ) -> OpponentStrategyModel:
        return build_strategy_model(company_id, evidence)

    def company_view(
        self, *, observer_company_id: str, round_number: int, state_version: int
    ) -> tuple[OpponentModelState, str]:
        if observer_company_id not in self.company_ids:
            raise KeyError(observer_company_id)
        state = OpponentModelState(
            episode_id=self.episode_id,
            observer_company_id=observer_company_id,
            prediction_target_round=round_number,
            state_version=state_version,
            public_evidence_through_round=max(0, round_number - 1),
            opponent_models={
                company_id: self._model(company_id, self._evidence[company_id])
                for company_id in self.company_ids
                if company_id != observer_company_id
            },
        )
        return state, compute_opponent_model_hash(state)

    def update_after_settlement(
        self,
        state_before: MarketState,
        state_after: MarketState,
        final_actions: Mapping[str, CompanyAction | Mapping[str, object]],
    ) -> tuple[PublicStrategyEvidence, ...]:
        if state_before.episode_id != self.episode_id:
            raise ValueError("opponent evidence belongs to another episode")
        if set(final_actions) != set(self.company_ids):
            raise ValueError("opponent evidence requires every company action")
        public_payload = {
            company_id: {
                "price_cents": int(
                    action.price_cents
                    if isinstance(action, CompanyAction)
                    else action["price_cents"]
                ),
                "shared_resilience_contribution_cents": int(
                    (
                        action.shared_resilience_contribution_cents
                        if isinstance(action, CompanyAction)
                        else action.get(
                            "shared_resilience_contribution_cents", 0
                        )
                    )
                    or 0
                ),
            }
            for company_id, action in final_actions.items()
        }
        settlement_hash = sha256_hash(
            {
                "episode_id": self.episode_id,
                "round": state_before.round,
                "state_before_hash": state_before.state_hash,
                "state_after_hash": state_after.state_hash,
                "public_actions": dict(sorted(public_payload.items())),
            }
        )
        prior = self._settlement_hashes.get(state_before.round)
        if prior is not None:
            if prior != settlement_hash:
                raise ValueError(
                    "opponent model round already contains different evidence"
                )
            return tuple(
                self._evidence[company_id][-1]
                for company_id in self.company_ids
            )
        created: list[PublicStrategyEvidence] = []
        for company_id in self.company_ids:
            before = state_before.company(company_id)
            after = state_after.company(company_id)
            settled_price = public_payload[company_id]["price_cents"]
            item = PublicStrategyEvidence(
                evidence_id=(
                    f"{self.episode_id}:round-{state_before.round}:"
                    f"{company_id}:public-strategy"
                ),
                episode_id=self.episode_id,
                settled_round=state_before.round,
                target_company_id=company_id,
                previous_price_cents=before.commercial.price_cents,
                settled_price_cents=settled_price,
                price_direction=_direction(
                    before.commercial.price_cents, settled_price
                ),
                market_share_delta_ppm=(
                    after.commercial.market_share_ppm
                    - before.commercial.market_share_ppm
                ),
                public_sales_orders=after.commercial.sales_orders,
                reputation_delta_ppm=(
                    after.brand.reputation_ppm - before.brand.reputation_ppm
                ),
                public_shared_resilience_contribution_cents=public_payload[
                    company_id
                ]["shared_resilience_contribution_cents"],
            )
            self._evidence[company_id].append(item)
            created.append(item)
        self._settlement_hashes[state_before.round] = settlement_hash
        return tuple(created)

    def evidence(self) -> tuple[PublicStrategyEvidence, ...]:
        return tuple(
            item
            for company_id in self.company_ids
            for item in self._evidence[company_id]
        )
