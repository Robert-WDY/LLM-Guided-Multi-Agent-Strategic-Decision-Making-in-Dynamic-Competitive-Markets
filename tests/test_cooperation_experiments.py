from game_theory_agent.cooperation.contracts import apply_cooperation_history_mode
from game_theory_agent.experiments.cooperation_real_multiseed import (
    CONDITIONS,
    build_plan,
    parse_seeds,
)
from game_theory_agent.experiments.real_cooperation_counterfactual import (
    CONDITIONS as COUNTERFACTUAL_CONDITIONS,
    _condition_order,
    build_condition_observation,
)
from game_theory_agent.experiments.persona_pilot import _observation
from game_theory_agent.market import MarketEnv


def test_real_cooperation_plan_uses_ten_common_seeds_and_three_conditions():
    seeds = parse_seeds(",".join(str(seed) for seed in range(101, 111)))
    plan = build_plan(seeds)

    assert len(plan) == 30
    assert {row["condition"] for row in plan} == set(CONDITIONS)
    for seed in seeds:
        rows = [row for row in plan if row["seed"] == seed]
        assert len(rows) == 3
        assert {row["condition"] for row in rows} == set(CONDITIONS)


def test_history_ablation_preserves_current_commitment_and_neutralizes_history():
    view = {
        "mode": "shared_resilience_v1",
        "proposals_sent": [],
        "proposals_received": [
            {
                "proposal_id": "proposal-current",
                "target_round": 3,
            },
            {
                "proposal_id": "proposal-old",
                "target_round": 2,
            },
        ],
        "pending_proposals_received": [],
        "responses": [{"proposal_id": "proposal-old"}],
        "active_commitments": [
            {
                "proposal_id": "proposal-current",
                "company_id": "company_B",
                "target_round": 3,
            }
        ],
        "commitment_history": [{"commitment_id": "past"}],
        "public_credibility": {
            "company_A": {
                "company_id": "company_A",
                "verified_commitment_count": 2,
                "fulfilled_count": 0,
                "partial_betrayal_count": 0,
                "betrayed_count": 2,
                "total_promised_contribution_cents": 2_000_000,
                "total_actual_capped_contribution_cents": 0,
                "credibility_ppm": 166_666,
            }
        },
        "cooperation_memory": {
            "company_A": {
                "memory_schema_version": "cooperation-memory-v1.0.0",
                "opponent_company_id": "company_A",
                "proposals_received": 3,
                "proposals_sent": 2,
                "accepted_by_self": 2,
                "accepted_by_opponent": 1,
                "commitments_by_opponent": 2,
                "fulfilled_by_opponent": 0,
                "partial_betrayals_by_opponent": 0,
                "betrayed_by_opponent": 2,
                "promised_by_opponent_cents": 2_000_000,
                "fulfilled_by_opponent_cents": 0,
                "credibility_ppm": 166_666,
                "history_is_neutralized": False,
            }
        },
    }

    ablated = apply_cooperation_history_mode(
        view, round_number=3, history_mode="none"
    )

    assert ablated["commitment_history"] == []
    assert ablated["responses"] == []
    assert [item["proposal_id"] for item in ablated["proposals_received"]] == [
        "proposal-current"
    ]
    assert ablated["active_commitments"] == view["active_commitments"]
    assert ablated["public_credibility"]["company_A"]["credibility_ppm"] == 500_000
    assert ablated["cooperation_memory"]["company_A"]["betrayed_by_opponent"] == 0
    assert ablated["cooperation_memory"]["company_A"]["credibility_ppm"] == 500_000
    assert ablated["cooperation_memory"]["company_A"]["history_is_neutralized"]
    assert view["public_credibility"]["company_A"]["credibility_ppm"] == 166_666


def test_real_counterfactual_conditions_share_state_and_vary_cooperation_inputs(config):
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B", "company_C", "company_D"),
        episode_id="cooperation-counterfactual-test",
        episode_seed=810,
        max_rounds=5,
        cooperation_mode="shared_resilience_v1",
    )
    base = _observation(config, state, "company_B")
    base["shared_resilience"] = state.shared_resilience.to_dict()
    observations = {
        condition: build_condition_observation(base, condition=condition)
        for condition in COUNTERFACTUAL_CONDITIONS
    }

    assert {item["state_hash"] for item in observations.values()} == {
        state.state_hash
    }
    assert observations["no_message"]["communication_view"][
        "visible_messages"
    ] == []
    assert observations["cooperation_proposal"]["cooperation"][
        "proposals_received"
    ]
    assert observations["defection_statement"]["cooperation"][
        "proposals_received"
    ] == []
    assert observations["high_credibility_proposal"]["cooperation"][
        "public_credibility"
    ]["company_A"]["credibility_ppm"] == 900_000
    assert observations["low_credibility_proposal"]["cooperation"][
        "public_credibility"
    ]["company_A"]["credibility_ppm"] == 100_000
    assert _condition_order(2)[0] == "cooperation_proposal"
