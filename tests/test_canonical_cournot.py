from __future__ import annotations

from game_theory_agent.benchmark_games import (
    CournotAction,
    CournotConfig,
    CournotEnvironment,
    CournotRoundEvent,
    exact_best_response,
    replay_cournot_events,
)


def test_classic_reference_points_and_no_unilateral_nash_gain() -> None:
    env = CournotEnvironment()
    assert env.settle((CournotAction(company_id="A", quantity=8), CournotAction(company_id="B", quantity=8))).profits == {"A": 64, "B": 64}
    assert env.settle((CournotAction(company_id="A", quantity=6), CournotAction(company_id="B", quantity=6))).profits == {"A": 72, "B": 72}
    assert env.settle((CournotAction(company_id="A", quantity=12), CournotAction(company_id="B", quantity=6))).profits == {"A": 72, "B": 36}
    assert exact_best_response(8) == 8
    assert all(env.profit(quantity, 8) <= env.profit(8, 8) for quantity in range(31))


def test_exact_advisor_matches_grid_best_response_for_every_opponent_quantity() -> None:
    env = CournotEnvironment()
    for opponent_quantity in range(31):
        advice = env.advise(company_id="A", belief_ppm={opponent_quantity: 1_000_000})
        assert advice.recommended_quantity == exact_best_response(opponent_quantity)
        assert advice.expected_regret_microunits == 0


def test_all_quantity_pairs_settle_to_the_formula() -> None:
    env = CournotEnvironment()
    for own in range(31):
        for opponent in range(31):
            outcome = env.settle((CournotAction(company_id="A", quantity=own), CournotAction(company_id="B", quantity=opponent)))
            price = max(30 - own - opponent, 0)
            assert outcome.price == price
            assert outcome.profits["A"] == (price - 6) * own
            assert outcome.profits["B"] == (price - 6) * opponent
            assert outcome.consumer_surplus_twice == min(own + opponent, 30) * (30 - price)


def test_replay_rebuilds_and_detects_tampering() -> None:
    config = CournotConfig()
    env = CournotEnvironment(config)
    actions = (CournotAction(company_id="A", quantity=8), CournotAction(company_id="B", quantity=8))
    first = CournotRoundEvent.create(episode_id="cournot-1", round_number=1, config=config, actions=actions, outcome=env.settle(actions))
    second = CournotRoundEvent.create(episode_id="cournot-1", round_number=2, config=config, actions=actions, outcome=env.settle(actions), previous_event_hash=first.event_hash)
    assert replay_cournot_events((first, second))
    tampered = second.model_copy(update={"event_hash": "sha256:" + "0" * 64})
    assert not replay_cournot_events((first, tampered))
