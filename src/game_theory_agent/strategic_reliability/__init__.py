"""Stage 6 strategic-decision reliability public API."""

from .contracts import (
    CandidateEconomicAction,
    RolloutCandidateEvaluation,
    RolloutScenarioOutcome,
    StrategicActionCandidate,
    StrategicReliabilityPlan,
    compute_reliability_plan_hash,
)
from .rollout import (
    AuthoritativeMarketRolloutEvaluator,
    generate_candidate_actions,
)
from .opponent_benchmark import (
    build_calibrated_opponent_state_v2,
    build_calibrated_strategy_model_v2,
    deterministic_dirichlet_weights,
    run_opponent_benchmark,
)
from .cost_guard import (
    RealModelBudget,
    RealModelBudgetExceeded,
    RealModelCostGuard,
    RealModelUsage,
)
from .public_contracts import (
    PublicForecastStateRecord,
    PublicRolloutCandidateSummary,
    PublicStrategicAdvice,
    compute_public_advice_hash,
)
from .public_rollout import (
    PublicMarketRolloutAdvisor,
    build_public_forecast_state,
    generate_public_overlay_candidates,
    generate_public_reliable_candidates,
    generate_public_marginal_candidates,
)
from .marginal_investment import (
    InvestmentMarginalAssessment,
    MarginalInvestmentPlan,
    build_marginal_investment_plan,
    compute_marginal_assessment_hash,
    compute_marginal_investment_plan_hash,
)
from .objective_calibration import (
    ObjectiveCalibrationDecision,
    ObjectiveCalibrationSpec,
    ObjectiveCandidateScore,
    StrategicObjectiveMode,
    calibrate_objective_decision,
    compute_objective_decision_hash,
)
from .pareto_planner import (
    PROMOTED_PARETO_SPEC,
    PROMOTION_EVIDENCE_SHA256,
    ParetoCandidateAssessment,
    ParetoPlannerDecision,
    ParetoPlannerSpec,
    ParetoSituation,
    compute_pareto_decision_hash,
    select_pareto_decision,
)
from .reliable_planner import (
    ExcludedCandidate,
    ParetoReliabilityGate,
    build_reliability_gate,
    compute_reliability_gate_hash,
)

__all__ = [
    "AuthoritativeMarketRolloutEvaluator",
    "CandidateEconomicAction",
    "RolloutCandidateEvaluation",
    "RolloutScenarioOutcome",
    "RealModelBudget",
    "RealModelBudgetExceeded",
    "RealModelCostGuard",
    "RealModelUsage",
    "PublicForecastStateRecord",
    "PublicMarketRolloutAdvisor",
    "PublicRolloutCandidateSummary",
    "PublicStrategicAdvice",
    "ObjectiveCalibrationDecision",
    "ObjectiveCalibrationSpec",
    "ObjectiveCandidateScore",
    "ParetoCandidateAssessment",
    "ParetoPlannerDecision",
    "ParetoPlannerSpec",
    "PROMOTED_PARETO_SPEC",
    "PROMOTION_EVIDENCE_SHA256",
    "ParetoSituation",
    "ParetoReliabilityGate",
    "ExcludedCandidate",
    "InvestmentMarginalAssessment",
    "MarginalInvestmentPlan",
    "StrategicActionCandidate",
    "StrategicObjectiveMode",
    "StrategicReliabilityPlan",
    "compute_reliability_plan_hash",
    "compute_public_advice_hash",
    "compute_objective_decision_hash",
    "compute_pareto_decision_hash",
    "compute_reliability_gate_hash",
    "calibrate_objective_decision",
    "select_pareto_decision",
    "build_reliability_gate",
    "build_marginal_investment_plan",
    "build_calibrated_strategy_model_v2",
    "build_calibrated_opponent_state_v2",
    "deterministic_dirichlet_weights",
    "generate_candidate_actions",
    "generate_public_overlay_candidates",
    "generate_public_reliable_candidates",
    "generate_public_marginal_candidates",
    "compute_marginal_assessment_hash",
    "compute_marginal_investment_plan_hash",
    "run_opponent_benchmark",
    "build_public_forecast_state",
]
