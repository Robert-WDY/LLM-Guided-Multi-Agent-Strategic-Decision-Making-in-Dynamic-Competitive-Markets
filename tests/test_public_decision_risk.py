from game_theory_agent.game_theory.public_price_guard import (
    PublicPriceGuard,
    public_features,
)

HISTORY = [[100, 100]] * 8


def example(seed, gain):
    return dict(
        seed=seed,
        features=public_features(HISTORY, 100, 27),
        action=2,
        gain=gain,
        sales_delta=1,
        exit_delta=0,
    )


def test_a_large_win_does_not_certify_frequent_losses():
    bank = {"native": [example(i, 100 if i == 15 else -1) for i in range(16)]}
    result = PublicPriceGuard(bank).explain("native", HISTORY, 100, 27, 2)
    assert result["action"] == 0
    assert result["losses"] == 15


def test_small_perfect_history_is_insufficient_but_supported_history_can_pass():
    for n, expected in ((16, 0), (42, 2), (60, 2)):
        bank = {"native": [example(i, 100) for i in range(n)]}
        result = PublicPriceGuard(bank).explain("native", HISTORY, 100, 27, 2)
        assert result["action"] == expected
        assert (result["risk_upper"] <= 0.1) == bool(expected)


def test_external_override_is_not_credited_to_managers_own_search_direction():
    from game_theory_agent.game_theory.cycle_pricing import feedback

    history = [[100, 100, 12, 100]] * 12 + [[104, 100, 11, 110]] * 4
    contaminated = {
        "direction": -1,
        "renewal_anchor": 100,
        "feedback_comparison_invalid": True,
    }
    target, direction = feedback(history, contaminated, 1, 1000)
    assert direction == -1 and target == 96.5
    assert feedback(history, {"direction": -1}, 1, 1000)[1] == 1
