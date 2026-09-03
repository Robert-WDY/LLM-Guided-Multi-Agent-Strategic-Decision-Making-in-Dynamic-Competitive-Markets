from __future__ import annotations

from game_theory_agent.experiments.canonical_validity_real import _cournot_prompt


def test_cournot_baseline_and_treatment_only_differ_by_advice() -> None:
    baseline, baseline_advice = _cournot_prompt(12, "persona_only")
    treatment, treatment_advice = _cournot_prompt(12, "exact_cournot_advisor")
    assert baseline_advice == treatment_advice
    assert '"advisor"' not in baseline
    assert '"advisor"' in treatment
    assert "current_opponent_action_visible\": false" in baseline


def test_cournot_advice_uses_classic_best_responses() -> None:
    expected = {6: 9, 8: 8, 12: 6}
    for opponent, quantity in expected.items():
        _prompt, advice = _cournot_prompt(opponent, "exact_cournot_advisor")
        assert advice["recommended_quantity"] == quantity
