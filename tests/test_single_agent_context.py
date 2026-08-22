from game_theory_agent.agents.single.context import (
    build_decision_context,
    build_deterministic_reflection,
)
from game_theory_agent.agents.single.gateway import GatewaySnapshot
from game_theory_agent.agents.single.models import (
    DecisionCandidate,
    DecisionTrace,
    EconomicAction,
    PersonaProfile,
)


def snapshot(round_number: int, history: list[dict] | None = None) -> GatewaySnapshot:
    return GatewaySnapshot(
        episode_id="episode-1",
        company_id="company_A",
        round=round_number,
        state_version=round_number - 1,
        state_hash=f"hash-{round_number}",
        observation={
            "episode_id": "episode-1",
            "company_id": "company_A",
            "round": round_number,
            "state_version": round_number - 1,
            "state_hash": f"hash-{round_number}",
            "own_company": {"financial": {"cash_balance_cents": 12_000_000}},
            "public_companies": [
                {"company_id": "company_A", "rank": 1},
                {
                    "company_id": "company_B",
                    "rank": 2,
                    "financial": {"cash_balance_cents": 99_000_000},
                },
            ],
            "public_history": history or [],
        },
        action_contract={
            "episode_id": "episode-1",
            "company_id": "company_A",
            "round": round_number,
            "state_version": round_number - 1,
            "bounds": {"price_cents": {"min": 800, "max": 2_000}},
        },
    )


def history_item(round_number: int, *, profit: int = 100, share: int = 220_000) -> dict:
    return {
        "settled_round": round_number,
        "own_action": {"price_cents": 1_100 + round_number},
        "own_result": {
            "round_profit_cents": profit,
            "market_share_ppm": share,
            "sales_orders": 100 + round_number,
        },
        "market": {
            "realized_demand_orders": 1_000 + round_number,
            "lost_after_stockout_orders": 0,
        },
        "active_events_during_round": [],
        "resolved_signal_outcomes": [],
    }


def trace(round_number: int, *, expected_outcome: str = "份额小幅提升") -> DecisionTrace:
    return DecisionTrace(
        episode_id="episode-1",
        company_id="company_A",
        round=round_number,
        state_version=round_number - 1,
        status="accepted",
        model_id="openrouter/test",
        persona=PersonaProfile(),
        candidates=[
            DecisionCandidate(
                candidate_id="balanced",
                label="平衡增长",
                action=EconomicAction(price_cents=1_180),
                expected_outcome=expected_outcome,
            ),
            DecisionCandidate(
                candidate_id="safe",
                label="现金保护",
                action=EconomicAction(price_cents=1_120),
                expected_outcome="现金保持稳定",
            ),
        ],
        selected_candidate_id="balanced",
    )


def test_first_round_context_has_empty_memory_and_baseline_reflection():
    context = build_decision_context(snapshot(1), [])

    assert context.snapshot_key.round == 1
    assert context.snapshot_key.state_hash == "hash-1"
    assert context.memory.recent_feedback == []
    assert context.reflection.source == "deterministic"
    assert context.reflection.summary
    assert "public_history" not in context.observation


def test_context_keeps_last_two_feedback_items_and_sanitizes_public_companies():
    history = [history_item(index) for index in range(1, 8)]

    context = build_decision_context(snapshot(8, history), [])

    assert [item.settled_round for item in context.memory.recent_feedback] == [6, 7]
    assert context.memory.history_limit == 2
    assert "public_history" not in context.observation
    assert "financial" not in context.observation["public_companies"][1]
    assert context.observation["own_company"]["financial"]["cash_balance_cents"] == 12_000_000


def test_context_aligns_previous_trace_with_last_feedback():
    context = build_decision_context(snapshot(2, [history_item(1)]), [trace(1)])

    assert context.memory.previous_selected_candidate_id == "balanced"
    assert context.memory.previous_expected_outcome == "份额小幅提升"
    assert "history_trace_mismatch" not in context.memory.diagnostic_codes
    assert context.reflection.source == "deterministic"


def test_context_marks_trace_history_mismatch_without_trusting_trace():
    context = build_decision_context(snapshot(3, [history_item(1), history_item(2)]), [trace(1)])

    assert context.memory.previous_selected_candidate_id is None
    assert context.memory.previous_expected_outcome is None
    assert "history_trace_mismatch" in context.memory.diagnostic_codes


def test_deterministic_reflection_uses_bounded_lesson_codes():
    context = build_decision_context(
        snapshot(3, [history_item(1, profit=-500, share=180_000), history_item(2, profit=800, share=210_000)]),
        [trace(2)],
    )

    reflection = build_deterministic_reflection(context.memory)

    assert reflection.source == "deterministic"
    assert "profit_positive" in reflection.lesson_codes
    assert "share_up" in reflection.lesson_codes
    assert reflection.evidence_paths
