import pytest
from pydantic import ValidationError

from game_theory_agent.agents.single.models import (
    DecisionCandidate,
    DecisionProposal,
    DecisionTrace,
    EconomicAction,
    EpisodeMemoryView,
    IntentDraft,
    PersonaInfluence,
    PersonaProfile,
    SnapshotKey,
)


def action(price_cents: int = 1_180) -> EconomicAction:
    return EconomicAction(
        price_cents=price_cents,
        advertising_budget_cents=240_000,
        service_budget_cents=160_000,
        capacity_investment_cents=0,
        resilience_budget_cents=80_000,
    )


def candidate(candidate_id: str) -> DecisionCandidate:
    return DecisionCandidate(
        candidate_id=candidate_id,
        label="平衡扩张",
        action=action(),
        evidence_paths=["own_company.financial.cash_balance_cents"],
        persona_influences=[
            PersonaInfluence(
                trait_key="risk_tolerance",
                direction="increase",
                affected_choice="advertising_budget_cents",
                summary="较高风险容忍提高扩张投入。",
            )
        ],
        tradeoffs=["提高份额但消耗现金"],
        expected_outcome="市场份额小幅提高",
    )


def test_persona_rejects_unknown_fields_and_out_of_range_traits():
    with pytest.raises(ValidationError):
        PersonaProfile(risk_tolerance=1.1)

    with pytest.raises(ValidationError):
        PersonaProfile(temperament="hidden")


def test_economic_action_rejects_negative_budgets():
    with pytest.raises(ValidationError):
        EconomicAction(price_cents=1_000, service_budget_cents=-1)


def test_decision_proposal_requires_exactly_three_candidates():
    with pytest.raises(ValidationError):
        DecisionProposal(
            candidates=[candidate("only")],
            selected_candidate_id="only",
            selection_reason_codes=["cash_protection"],
        )

    with pytest.raises(ValidationError):
        DecisionProposal(
            candidates=[candidate("a"), candidate("b")],
            selected_candidate_id="a",
            selection_reason_codes=["cash_protection"],
        )

    DecisionProposal(
        candidates=[candidate("a"), candidate("b"), candidate("c")],
        selected_candidate_id="a",
        selection_reason_codes=["cash_protection"],
    )
    with pytest.raises(ValidationError):
        DecisionProposal(
            candidates=[candidate(key) for key in ["a", "b", "c", "d"]],
            selected_candidate_id="a",
            selection_reason_codes=["cash_protection"],
        )


def test_candidate_readable_fields_have_bounded_lengths():
    with pytest.raises(ValidationError):
        DecisionCandidate(
            candidate_id="too-long",
            label="候" * 81,
            action=action(),
            expected_outcome="结果",
        )

    with pytest.raises(ValidationError):
        EconomicAction(price_cents=1_180, strategy_summary="策" * 241)


def test_decision_proposal_requires_selected_candidate_to_exist():
    with pytest.raises(ValidationError):
        DecisionProposal(
            candidates=[candidate("safe"), candidate("balanced"), candidate("growth")],
            selected_candidate_id="missing",
            selection_reason_codes=["share_growth"],
        )


def test_decision_proposal_returns_the_selected_candidate():
    proposal = DecisionProposal(
        candidates=[candidate("safe"), candidate("balanced"), candidate("growth")],
        selected_candidate_id="balanced",
        selection_reason_codes=["share_growth", "cash_protection"],
    )

    assert proposal.selected_candidate.candidate_id == "balanced"


def test_snapshot_key_requires_non_empty_state_hash_and_rejects_unknown_fields():
    key = SnapshotKey(
        episode_id="episode-1",
        company_id="company_A",
        round=2,
        state_version=4,
        state_hash="hash-123",
    )

    assert key.state_hash == "hash-123"

    with pytest.raises(ValidationError):
        SnapshotKey(
            episode_id="episode-1",
            company_id="company_A",
            round=2,
            state_version=4,
            state_hash="",
        )

    with pytest.raises(ValidationError):
        SnapshotKey(
            episode_id="episode-1",
            company_id="company_A",
            round=2,
            state_version=4,
            state_hash="hash-123",
            private_state="hidden",
        )


def test_episode_memory_view_keeps_a_bounded_recent_feedback_window():
    feedback = [
        {
            "settled_round": index,
            "own_action": {"price_cents": 1100 + index},
            "own_result": {"round_profit_cents": index * 100},
            "market": {"realized_demand_orders": 1000 + index},
            "active_events_during_round": [],
            "resolved_signal_outcomes": [],
        }
        for index in range(1, 6)
    ]

    memory = EpisodeMemoryView(recent_feedback=feedback)

    assert memory.history_limit == 2
    assert [item.settled_round for item in memory.recent_feedback] == [1, 2, 3, 4, 5]

    with pytest.raises(ValidationError):
        EpisodeMemoryView(recent_feedback=feedback + [feedback[-1] | {"settled_round": 6}])


def test_intent_draft_uses_typed_snapshot_and_action():
    key = SnapshotKey(
        episode_id="episode-1",
        company_id="company_A",
        round=2,
        state_version=4,
        state_hash="hash-123",
    )

    draft = IntentDraft(
        snapshot_key=key,
        agent_id="single-agent-company-A",
        action=action(),
        rationale="现金约束内保持增长。",
        expected_outcome="份额小幅提升。",
    )

    assert draft.snapshot_key == key
    assert draft.action.price_cents == 1180

    with pytest.raises(ValidationError):
        IntentDraft(
            snapshot_key=key.model_dump(),
            agent_id="single-agent-company-A",
            action={"price_cents": -1},
            rationale="非法动作",
            expected_outcome="失败",
        )


def test_decision_trace_accepts_legacy_json_and_defaults_new_cross_round_fields():
    legacy = DecisionTrace(
        trace_version="single-agent-trace-v1.0.0",
        episode_id="episode-1",
        company_id="company_A",
        round=1,
        state_version=0,
        status="accepted",
        model_id="openrouter/test",
        persona=PersonaProfile(),
        candidates=[candidate("balanced"), candidate("safe")],
        selected_candidate_id="balanced",
    )

    assert legacy.trace_version == "single-agent-trace-v1.0.0"
    assert legacy.memory_view is None
    assert legacy.strategy_reflection is None
    assert legacy.prepared_intent is None

    current = DecisionTrace(
        episode_id="episode-1",
        company_id="company_A",
        round=1,
        state_version=0,
        status="accepted",
        model_id="openrouter/test",
        persona=PersonaProfile(),
    )

    assert current.trace_version == "single-agent-trace-v1.2.0"
