"""Minimal Bayesian Game Advisor public API."""

from game_theory_agent.advisor.advisor import (
    BayesianGameAdvisor,
    BayesianStrategyAdvisor,
)
from game_theory_agent.advisor.contracts import (
    ADVISOR_HASH_PROTOCOL_VERSION,
    ADVISOR_MODEL_VERSION,
    ADVISOR_SCHEMA_VERSION,
    AdvisorMode,
    BayesianCandidateEvaluation,
    PredictedOpponentResponse,
    GameTheoryAdvice,
    StrategicCandidateEvaluation,
    StrategicGameTheoryAdvice,
    compute_advice_hash,
    compute_strategic_advice_hash,
)
from game_theory_agent.advisor.replay import (
    AdvisorReplayMismatchError,
    verify_advisor_replay,
)
from game_theory_agent.advisor.adoption import (
    AdvisorAdoptionTrace,
    build_advisor_adoption_trace,
    compute_adoption_trace_hash,
)
from game_theory_agent.advisor.context_view import build_agent_advice_view

__all__ = [
    "ADVISOR_HASH_PROTOCOL_VERSION",
    "ADVISOR_MODEL_VERSION",
    "ADVISOR_SCHEMA_VERSION",
    "AdvisorMode",
    "AdvisorAdoptionTrace",
    "AdvisorReplayMismatchError",
    "BayesianCandidateEvaluation",
    "BayesianGameAdvisor",
    "BayesianStrategyAdvisor",
    "GameTheoryAdvice",
    "PredictedOpponentResponse",
    "StrategicCandidateEvaluation",
    "StrategicGameTheoryAdvice",
    "compute_advice_hash",
    "build_advisor_adoption_trace",
    "build_agent_advice_view",
    "compute_adoption_trace_hash",
    "compute_strategic_advice_hash",
    "verify_advisor_replay",
]
