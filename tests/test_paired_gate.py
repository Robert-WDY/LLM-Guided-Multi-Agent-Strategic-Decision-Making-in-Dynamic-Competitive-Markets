from types import SimpleNamespace
import pytest
from game_theory_agent.strategic_reliability.paired_gate import paired_deltas, build_paired_market_gate
from game_theory_agent.strategic_reliability import paired_gate
from game_theory_agent.strategic_reliability.reliable_planner import ParetoAbstentionGate, compute_abstention_gate_hash


def evaluation(gains, *, reverse=False):
    rows=[SimpleNamespace(scenario_index=i,scenario_seed=i+100,enterprise_value_cents=10000000*(i+1)+g,risk_adjusted_value_cents=10000000*(i+1)+g,persona_aligned_value_cents=10000000*(i+1)+g) for i,g in enumerate(gains)]
    return SimpleNamespace(outcomes=list(reversed(rows)) if reverse else rows)


def base_gate(confidence=800000):
    reasons=["top_candidates_overlap_forecast_uncertainty"]
    if confidence<700000: reasons.append("opponent_model_confidence_below_gate")
    data=dict(planner_decision_hash="decision",reliability_input_hash="input",planner_candidate_id="better",diagnostic_fallback_candidate_id="maintain",withheld_candidate_id="better",execution_disposition="defer_to_agent",safe_candidate_ids=["maintain","better"],excluded_candidates=[],opponent_model_confidence_ppm=confidence,top_two_value_gap_cents=0,forecast_uncertainty_cents=40000000,advisor_confidence_ppm=0,selected_passes_fallback_value_floor=True,selected_passes_fallback_worst_floor=True,selected_passes_fallback_persona_floor=True,should_abstain=True,abstain_reason_codes=reasons,marginal_investment_plan_hash="marginal")
    data=ParetoAbstentionGate.model_construct(**data).model_dump(mode="json")
    data["gate_hash"]=compute_abstention_gate_hash(data)
    return ParetoAbstentionGate.model_validate(data)


def gate(monkeypatch,gains,confidence=800000):
    monkeypatch.setattr(paired_gate,"build_abstention_gate",lambda **kw:base_gate(confidence))
    a,b=evaluation(gains),evaluation([0]*len(gains),reverse=True)
    a.candidate=SimpleNamespace(candidate_id="better");b.candidate=SimpleNamespace(candidate_id="maintain")
    plan=SimpleNamespace(evaluations=[a,b],baseline_candidate_id="maintain",plan_hash="plan")
    return build_paired_market_gate(plan=plan,decision=SimpleNamespace(recommended_candidate_id="better"),diagnostic_fallback_candidate_id="maintain")


def test_common_shock_cancels_without_requiring_unique_best_rank(monkeypatch):
    result=gate(monkeypatch,[300000]*5)
    assert result.execution_disposition=="recommend"
    assert result.forecast_uncertainty_cents==0
    assert result.top_two_value_gap_cents==300000


@pytest.mark.parametrize("gains,confidence", [([300000]*5,600000),([300000]*4,800000),([1000000]*4+[-1],800000),([100000]*5,800000)])
def test_low_history_missing_scenarios_tail_loss_and_small_gain_still_abstain(monkeypatch,gains,confidence):
    result=gate(monkeypatch,gains,confidence)
    assert result.execution_disposition=="defer_to_agent"
    assert result.effective_candidate_id is None


def test_pairing_uses_index_and_seed_and_refuses_mismatches():
    a,b=evaluation([300000]*5),evaluation([0]*5,reverse=True)
    assert all(r["enterprise_value_cents"]==300000 for r in paired_deltas(a,b))
    b.outcomes[0].scenario_seed=999
    with pytest.raises(ValueError,match="do not match"):paired_deltas(a,b)
