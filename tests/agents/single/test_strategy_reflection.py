"""策略反思在首轮和后续轮都必须可审计。"""

from tests.test_single_agent_context import history_item, snapshot, trace

from game_theory_agent.agents.single.context import build_decision_context


def test_first_round_reflection_has_explicit_baseline_evidence():
    reflection = build_decision_context(snapshot(1), []).reflection

    assert reflection.source == "deterministic"
    assert reflection.lesson_codes == ["first_round_baseline"]
    assert reflection.adjustments
    assert reflection.summary
    assert reflection.evidence_paths == [
        "observation.own_company",
        "action_contract.bounds",
    ]


def test_later_round_reflection_remains_derived_from_real_history():
    reflection = build_decision_context(
        snapshot(2, [history_item(1, profit=500)]),
        [trace(1)],
    ).reflection

    assert reflection.source == "deterministic"
    assert "profit_positive" in reflection.lesson_codes
    assert any(path.startswith("memory.recent_feedback") for path in reflection.evidence_paths)
    assert reflection.summary
