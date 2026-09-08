"""Versioned common-scenario improvement gate; does not claim unique optimality."""
from game_theory_agent.market.protocols import sha256_hash
from .reliable_planner import build_abstention_gate, ParetoAbstentionGate, compute_abstention_gate_hash

PAIRED_MARKET_GATE_POLICY = {
    "policy_version": "paired-baseline-dominance-v1.0.0",
    "minimum_opponent_confidence_ppm": 700000,
    "minimum_scenarios": 5,
    "minimum_expected_enterprise_value_gain_cents": 250000,
    "minimum_paired_enterprise_value_gain_cents": 0,
    "minimum_paired_risk_adjusted_gain_cents": 0,
    "minimum_paired_persona_gain_cents": 0,
    "comparators": ["maintain", "diagnostic_fallback"],
    "price_coordination_is_research_only": True,
    "optimality_claim": False,
}


def paired_deltas(selected, comparator):
    def keyed(evaluation):
        rows={(r.scenario_index,r.scenario_seed):r for r in evaluation.outcomes}
        if len(rows)!=len(evaluation.outcomes): raise ValueError("duplicate paired scenario")
        return rows
    left,right=keyed(selected),keyed(comparator)
    if left.keys()!=right.keys(): raise ValueError("paired scenarios do not match")
    return [{field:getattr(left[k],field)-getattr(right[k],field) for field in ("enterprise_value_cents","risk_adjusted_value_cents","persona_aligned_value_cents")} for k in sorted(left)]


def build_paired_market_gate(**kwargs):
    base=build_abstention_gate(**kwargs)
    plan,decision=kwargs["plan"],kwargs["decision"]
    evaluations={r.candidate.candidate_id:r for r in plan.evaluations}
    selected=evaluations[decision.recommended_candidate_id]
    comparators={plan.baseline_candidate_id,kwargs["diagnostic_fallback_candidate_id"]}
    comparisons={c:paired_deltas(selected,evaluations[c]) for c in sorted(comparators)}
    reasons=[r for r in base.abstain_reason_codes if r!="top_candidates_overlap_forecast_uncertainty"]
    if min(len(rows) for rows in comparisons.values())<5:
        reasons.append("insufficient_paired_scenarios")
    for comparator,rows in comparisons.items():
        if any(any(v<0 for v in row.values()) for row in rows):
            reasons.append(f"paired_downside_against:{comparator}")
        if sum(r["enterprise_value_cents"] for r in rows)<250000*len(rows):
            reasons.append(f"paired_gain_below_floor:{comparator}")
    if decision.recommended_candidate_id not in base.safe_candidate_ids:
        reasons.append("selected_candidate_not_gate_safe")
    reasons=sorted(set(reasons))
    payload=base.model_dump(mode="json")
    ev=[r["enterprise_value_cents"] for rows in comparisons.values() for r in rows]
    payload.update(reliability_input_hash=sha256_hash({"base_input":base.reliability_input_hash,"plan_hash":plan.plan_hash,"policy":PAIRED_MARKET_GATE_POLICY,"paired_deltas":comparisons}),top_two_value_gap_cents=max(0,min(ev)),forecast_uncertainty_cents=max(ev)-min(ev),advisor_confidence_ppm=base.opponent_model_confidence_ppm if not reasons else 0,should_abstain=bool(reasons),abstain_reason_codes=reasons,execution_disposition="defer_to_agent" if reasons else "recommend",effective_candidate_id=None if reasons else base.planner_candidate_id,withheld_candidate_id=base.planner_candidate_id if reasons else None)
    payload["gate_hash"]=compute_abstention_gate_hash(payload)
    return ParetoAbstentionGate.model_validate(payload)
