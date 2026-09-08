from copy import deepcopy

from game_theory_agent.advisor import build_agent_advice_view
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.experiments.final_market_advisor_v9_validation import (
    CONFIG_PATH,
    FOCAL,
    HORIZON,
    SCENARIOS,
    _build_window,
)
from game_theory_agent.experiments.final_market_v9_real_llm_smoke import (
    _complete_observation,
)
from game_theory_agent.market import load_market_config
from game_theory_agent.strategic_reliability import PublicMarketRolloutAdvisor


def _advice(disposition: str) -> dict:
    return {
        "advice_schema_version": "public-strategic-market-advice-v9.0.0",
        "advisor_mode": "strategic_market_v9",
        "advice_hash": "sha256:full-artifact",
        "execution_disposition": disposition,
        "should_abstain": disposition == "defer_to_agent",
        "recommended_candidate_id": (
            None if disposition == "defer_to_agent" else "mutual_aid_request"
        ),
        "recommended_action": (
            None
            if disposition == "defer_to_agent"
            else {"mutual_aid_capacity_request_orders": 1000}
        ),
        "candidate_actions": [
            {
                "candidate": {
                    "candidate_id": "price_raise_small",
                    "action": {"price_cents": 11_500},
                },
                "certainty_equivalent_value_cents": 123,
            }
        ],
        "research_only_candidate_ids": ["price_coordination_honor"],
    }


def test_abstained_advice_is_not_an_agent_visible_intervention():
    source = _advice("defer_to_agent")
    original = deepcopy(source)
    view = build_agent_advice_view(source)

    assert source == original
    assert view is None


def test_released_advice_keeps_auditable_recommendation():
    source = _advice("recommend")
    view = build_agent_advice_view(source)

    assert view["recommended_candidate_id"] == "mutual_aid_request"
    assert view["candidate_actions"] == source["candidate_actions"]


def test_legacy_v7_abstention_contract_remains_visible():
    source = _advice("defer_to_agent")
    source["advisor_mode"] = "pareto_reliable_v7"

    assert build_agent_advice_view(source) == source


def test_abstention_and_no_advice_have_identical_agent_observation():
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    state, belief, opponent, observation = _build_window(config, 95104, "pivotal_project")
    advice = PublicMarketRolloutAdvisor(config).advise(
        observation=observation,
        company_id=FOCAL,
        persona_profile=personas.get("balanced_v1"),
        belief_state=belief,
        opponent_model=opponent,
        horizon_rounds=HORIZON,
        scenario_count=SCENARIOS,
        advisor_mode="strategic_market_v9",
    ).model_dump(mode="json")

    assert advice["execution_disposition"] == "defer_to_agent"
    baseline = _complete_observation(config, state, observation, advice=None)
    abstained = _complete_observation(config, state, observation, advice=advice)

    assert "game_theory_advice" not in abstained
    assert abstained == baseline
    assert abstained["observation_hash"] == baseline["observation_hash"]
