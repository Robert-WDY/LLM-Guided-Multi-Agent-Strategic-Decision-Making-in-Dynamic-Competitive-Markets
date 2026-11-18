import re
from game_theory_agent.game_theory.price_language import legal_grammar

from game_theory_agent.game_theory.public_price_guard import (
    PublicPriceGuard,
    public_features,
)


PRICES = [[100, 99]] * 8


def bank(gain=1, exits=0, sales=1, seeds=range(48)):
    return {
        "m": [
            dict(
                seed=s,
                action=2,
                features=public_features(PRICES, 100, 27),
                gain=gain,
                exit_delta=exits,
                sales_delta=sales,
            )
            for s in seeds
        ]
    }


def test_risk_checks_require_independent_supported_terminal_outcomes():
    for examples in [bank(-1), bank(exits=1), bank(sales=-1), bank(seeds=[1] * 100)]:
        assert PublicPriceGuard(examples).assess("m", PRICES, 100, 27, 2) == 0
    guard = PublicPriceGuard(bank())
    assert guard.assess("m", PRICES, 100, 27, 2) == 2
    assert guard.assess("m", PRICES, 150, 27, 2) == 0
    assert guard.assess("m", PRICES, 100, 27, 1) == 0
    assert guard.assess("other", PRICES, 100, 27, 2) == 0


def test_quote_language_equals_executable_tick_set():
    for places, lo, hi in [
        (0, 0, 0),
        (0, 87, 124),
        (0, 123, 9876),
        (2, 1, 4),
        (2, 89, 124),
        (2, 999, 1001),
    ]:
        gbnf = legal_grammar(lo, hi, places).split("::=")[1]
        pattern = re.compile(
            re.sub(r'"([^"]*)"', lambda m: re.escape(m[1]), gbnf).replace(" ", "")
        )
        for tick in range(max(0, lo - 250), hi + 252):
            amount = f"{tick / (10**places):.{places}f}"
            assert bool(pattern.fullmatch(amount)) == (lo <= tick <= hi)
        assert not any(
            pattern.fullmatch(x) for x in ("01", "01.00", "-1", "1e2", "1.000")
        )


def test_failed_hike_expires_only_after_both_adverse_settlements():
    from game_theory_agent.game_theory.cycle_pricing import failed_trial_adjustment

    prior = [[100, 100, 10, 120]] * 4
    down = [[104, 99, 8, 115]] * 4
    assert failed_trial_adjustment(prior + down, 4) == 4
    assert failed_trial_adjustment(prior + down, -4) == 0
    assert failed_trial_adjustment(prior + [[104, 99, 11, 115]] * 4, 4) == 0
    assert failed_trial_adjustment(prior + [[104, 99, 8, 120]] * 4, 4) == 0
    assert failed_trial_adjustment(down, 4) == 0


def test_expiry_restores_saved_draft_with_the_real_feedback_direction():
    from game_theory_agent.game_theory.cycle_pricing import feedback

    prior = [[100, 100, 10, 50]] * 12
    for posted, step in [(104, 1), (96, -1)]:
        history = prior + [[posted, 101, 9, 60]] * 4
        original, direction = feedback(history, {"direction": 1}, 10, 200)
        restored, saved_direction = feedback(
            history, {"direction": 1, "renewal_anchor": 100}, 10, 200
        )
        assert direction == saved_direction == step
        assert original == posted * (1 + 0.035 * step)
        assert restored == 100 * (1 + 0.035 * step)
