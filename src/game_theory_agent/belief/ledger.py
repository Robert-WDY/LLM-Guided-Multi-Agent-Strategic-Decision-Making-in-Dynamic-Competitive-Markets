"""Deterministic public-action belief updater and episode ledger."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from game_theory_agent.belief.contracts import (
    BeliefState,
    CommunicationPriceSignal,
    OpponentPriceBelief,
    PriceDirection,
    PriceDirectionDistribution,
    PublicPriceEvidence,
    compute_belief_hash,
    BELIEF_SCHEMA_VERSION,
    BELIEF_UPDATER_VERSION,
    SIGNAL_BELIEF_SCHEMA_VERSION,
    SIGNAL_BELIEF_UPDATER_VERSION,
)
from game_theory_agent.market.models import CompanyAction, MarketState
from game_theory_agent.market.protocols import sha256_hash


_DIRECTIONS: tuple[PriceDirection, ...] = (
    "price_cut",
    "maintain",
    "price_raise",
)


def classify_price_direction(
    previous_price_cents: int, settled_price_cents: int
) -> PriceDirection:
    if settled_price_cents < previous_price_cents:
        return "price_cut"
    if settled_price_cents > previous_price_cents:
        return "price_raise"
    return "maintain"


def _probabilities(counts: Mapping[PriceDirection, int]) -> PriceDirectionDistribution:
    posterior = [int(counts[item]) + 1 for item in _DIRECTIONS]
    total = sum(posterior)
    raw = [value * 1_000_000 for value in posterior]
    allocated = [value // total for value in raw]
    remaining = 1_000_000 - sum(allocated)
    # Largest-remainder allocation with a fixed direction-order tie break.
    order = sorted(
        range(len(_DIRECTIONS)),
        key=lambda index: (raw[index] % total, -index),
        reverse=True,
    )
    for index in order[:remaining]:
        allocated[index] += 1
    return PriceDirectionDistribution(
        price_cut_ppm=allocated[0],
        maintain_ppm=allocated[1],
        price_raise_ppm=allocated[2],
    )


def _weighted_probabilities(
    public_counts: Mapping[PriceDirection, int],
    signal_counts: Mapping[PriceDirection, int],
    reliability_ppm: int,
) -> PriceDirectionDistribution:
    scores = [
        (int(public_counts[item]) + 1) * 1_000_000
        + int(signal_counts[item]) * reliability_ppm
        for item in _DIRECTIONS
    ]
    total = sum(scores)
    raw = [value * 1_000_000 for value in scores]
    allocated = [value // total for value in raw]
    remaining = 1_000_000 - sum(allocated)
    order = sorted(
        range(len(_DIRECTIONS)),
        key=lambda index: (raw[index] % total, -index),
        reverse=True,
    )
    for index in order[:remaining]:
        allocated[index] += 1
    return PriceDirectionDistribution(
        price_cut_ppm=allocated[0],
        maintain_ppm=allocated[1],
        price_raise_ppm=allocated[2],
    )


class BeliefLedger:
    """Authority for all pre-decision belief views in one episode."""

    def __init__(
        self,
        *,
        episode_id: str,
        company_ids: tuple[str, ...],
        mode: str = "public_action_v1",
    ) -> None:
        if len(company_ids) < 2 or len(set(company_ids)) != len(company_ids):
            raise ValueError("belief ledger requires unique companies")
        self.episode_id = episode_id
        self.company_ids = tuple(company_ids)
        if mode not in {"public_action_v1", "public_action_signal_v2"}:
            raise ValueError(f"unsupported belief ledger mode: {mode}")
        self.mode = mode
        self._evidence: dict[str, list[PublicPriceEvidence]] = {
            company_id: [] for company_id in self.company_ids
        }
        self._settlement_hashes: dict[int, str] = {}
        self._claim_outcomes: dict[str, list[bool]] = {
            company_id: [] for company_id in self.company_ids
        }

    def _reliability_ppm(self, company_id: str) -> int:
        outcomes = self._claim_outcomes[company_id]
        # Beta(1,1) posterior mean: neutral before any verified claim.
        return (sum(outcomes) + 1) * 1_000_000 // (len(outcomes) + 2)

    def _visible_signals(
        self,
        *,
        observer_company_id: str,
        visible_messages: Iterable[Any],
        public_prices: Mapping[str, int],
    ) -> tuple[CommunicationPriceSignal, ...]:
        if self.mode != "public_action_signal_v2":
            return ()
        signals: list[CommunicationPriceSignal] = []
        for raw in visible_messages:
            message = (
                raw.model_dump(mode="json")
                if hasattr(raw, "model_dump")
                else dict(raw)
            )
            sender = str(message.get("sender_company_id", ""))
            if sender == observer_company_id or sender not in self.company_ids:
                continue
            claim = message.get("own_action_claim")
            if not isinstance(claim, Mapping) or claim.get("price_cents") is None:
                continue
            claimed_price = int(claim["price_cents"])
            signals.append(
                CommunicationPriceSignal(
                    message_id=str(message["message_id"]),
                    sender_company_id=sender,
                    observer_company_id=observer_company_id,
                    claimed_price_cents=claimed_price,
                    direction_relative_to_public_price=classify_price_direction(
                        int(public_prices[sender]), claimed_price
                    ),
                    historical_reliability_ppm=self._reliability_ppm(sender),
                )
            )
        return tuple(signals)

    def company_view(
        self,
        *,
        observer_company_id: str,
        round_number: int,
        state_version: int,
        visible_messages: Iterable[Any] = (),
        public_prices: Mapping[str, int] | None = None,
    ) -> tuple[BeliefState, str]:
        if observer_company_id not in self.company_ids:
            raise KeyError(observer_company_id)
        if self.mode == "public_action_signal_v2" and public_prices is None:
            raise ValueError("signal belief requires current public prices")
        signals = self._visible_signals(
            observer_company_id=observer_company_id,
            visible_messages=visible_messages,
            public_prices=public_prices or {},
        )
        opponent_beliefs: dict[str, OpponentPriceBelief] = {}
        for opponent in self.company_ids:
            if opponent == observer_company_id:
                continue
            evidence = self._evidence[opponent]
            counts: Counter[PriceDirection] = Counter(
                item.observed_direction for item in evidence
            )
            normalized = {direction: counts[direction] for direction in _DIRECTIONS}
            opponent_signals = [
                signal for signal in signals
                if signal.sender_company_id == opponent
            ]
            signal_counter: Counter[PriceDirection] = Counter(
                item.direction_relative_to_public_price
                for item in opponent_signals
            )
            signal_counts = {
                direction: signal_counter[direction] for direction in _DIRECTIONS
            }
            reliability = self._reliability_ppm(opponent)
            latest = evidence[-1] if evidence else None
            opponent_beliefs[opponent] = OpponentPriceBelief(
                opponent_company_id=opponent,
                prediction_target_round=round_number,
                evidence_count=len(evidence),
                latest_evidence_round=(
                    latest.settled_round if latest is not None else None
                ),
                latest_observed_direction=(
                    latest.observed_direction if latest is not None else None
                ),
                observed_counts=normalized,
                next_price_direction=(
                    _weighted_probabilities(
                        normalized, signal_counts, reliability
                    )
                    if self.mode == "public_action_signal_v2"
                    else _probabilities(normalized)
                ),
                signal_evidence_count=len(opponent_signals),
                signal_direction_counts=signal_counts,
                signal_reliability_ppm=(
                    reliability
                    if self.mode == "public_action_signal_v2"
                    else None
                ),
            )
        state = BeliefState(
            belief_schema_version=(
                SIGNAL_BELIEF_SCHEMA_VERSION
                if self.mode == "public_action_signal_v2"
                else BELIEF_SCHEMA_VERSION
            ),
            belief_mode=self.mode,
            updater_version=(
                SIGNAL_BELIEF_UPDATER_VERSION
                if self.mode == "public_action_signal_v2"
                else BELIEF_UPDATER_VERSION
            ),
            episode_id=self.episode_id,
            observer_company_id=observer_company_id,
            prediction_target_round=round_number,
            state_version=state_version,
            public_evidence_through_round=max(0, round_number - 1),
            evidence_scope=(
                "settled_public_prices_and_visible_non_binding_claims"
                if self.mode == "public_action_signal_v2"
                else "settled_public_prices_only"
            ),
            opponent_beliefs=opponent_beliefs,
            visible_communication_signals=list(signals),
        )
        return state, compute_belief_hash(state)

    def update_after_settlement(
        self,
        state_before: MarketState,
        final_actions: Mapping[str, CompanyAction | Mapping[str, object]],
        communication_messages: Iterable[Any] = (),
    ) -> tuple[PublicPriceEvidence, ...]:
        if state_before.episode_id != self.episode_id:
            raise ValueError("belief settlement belongs to another episode")
        if set(final_actions) != set(self.company_ids):
            raise ValueError("belief settlement must contain every company")
        canonical_prices = {
            company_id: int(
                action.price_cents
                if isinstance(action, CompanyAction)
                else action["price_cents"]
            )
            for company_id, action in final_actions.items()
        }
        canonical_messages = tuple(
            raw.model_dump(mode="json")
            if hasattr(raw, "model_dump")
            else dict(raw)
            for raw in communication_messages
        )
        settlement_hash = sha256_hash(
            {
                "episode_id": self.episode_id,
                "settled_round": state_before.round,
                "state_hash": state_before.state_hash,
                "settled_public_prices": dict(sorted(canonical_prices.items())),
                "communication_price_claims": [
                    {
                        "message_id": message.get("message_id"),
                        "sender_company_id": message.get("sender_company_id"),
                        "own_action_claim": message.get("own_action_claim"),
                    }
                    for message in canonical_messages
                    if message.get("own_action_claim") is not None
                ],
            }
        )
        prior = self._settlement_hashes.get(state_before.round)
        if prior is not None:
            if prior != settlement_hash:
                raise ValueError("belief round was already updated with different evidence")
            return tuple(
                self._evidence[company_id][-1] for company_id in self.company_ids
            )

        created: list[PublicPriceEvidence] = []
        for company_id in self.company_ids:
            previous = state_before.company(company_id).commercial.price_cents
            settled = canonical_prices[company_id]
            direction = classify_price_direction(previous, settled)
            evidence = PublicPriceEvidence(
                evidence_id=(
                    f"{self.episode_id}:round-{state_before.round}:"
                    f"{company_id}:public-price"
                ),
                episode_id=self.episode_id,
                settled_round=state_before.round,
                target_company_id=company_id,
                previous_public_price_cents=previous,
                settled_public_price_cents=settled,
                observed_direction=direction,
            )
            self._evidence[company_id].append(evidence)
            created.append(evidence)
        if self.mode == "public_action_signal_v2":
            for message in canonical_messages:
                sender = str(message.get("sender_company_id", ""))
                claim = message.get("own_action_claim")
                if sender not in self._claim_outcomes or not isinstance(
                    claim, Mapping
                ) or claim.get("price_cents") is None:
                    continue
                self._claim_outcomes[sender].append(
                    int(claim["price_cents"]) == canonical_prices[sender]
                )
        self._settlement_hashes[state_before.round] = settlement_hash
        return tuple(created)

    def evidence(self) -> tuple[PublicPriceEvidence, ...]:
        return tuple(
            item
            for company_id in self.company_ids
            for item in self._evidence[company_id]
        )

    def claim_reliability_ppm(self, company_id: str) -> int:
        if company_id not in self._claim_outcomes:
            raise KeyError(company_id)
        return self._reliability_ppm(company_id)
