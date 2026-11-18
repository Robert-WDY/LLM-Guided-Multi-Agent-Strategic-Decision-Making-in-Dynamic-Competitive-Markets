from pathlib import Path

import numpy as np
import pytest

from game_theory_agent.game_theory.renewal_review import (
    PublicRenewalReview,
    decisions,
    estimates,
)


def test_risk_is_conditional_on_executed_price_and_deducts_downside():
    # 8% unconditional loss among 40% effective interventions is a 20% execution risk.
    y = np.array([[10.0, 0.4, 0.08, 1.0, 12.0, 1.0, 0.0, 8.0]])
    score, risk = estimates(y)
    np.testing.assert_allclose(score, [[9.0, 4.0]])
    np.testing.assert_allclose(risk, [[0.2, 0.0]])
    assert decisions(y, [True], 0.1, 5).tolist() == [0]
    assert decisions(y, [True], 0.21, 5).tolist() == [1]
    assert decisions(y, [False], 1.0, 0).tolist() == [0]


def test_public_pretrade_review_requires_exact_prefix_and_preserves_draft():
    root = Path(__file__).resolve().parents[1]
    reviewer = PublicRenewalReview(
        root / "experiment-results/renewal-risk-v1/public-econ.npz"
    )
    history = [[1.1, 1.3, 1.5, 1.7]] * 22
    result = reviewer.review(history, 0, 23, 1.1)
    assert result["future_renewals"] == [27, 31] and result["contract_last_round"] == 26
    assert result["requested"] == 1.1 and result["disposition"] == "keep"
    with pytest.raises(ValueError):
        reviewer.review(history + [history[0]], 0, 23, 1.1)
    with pytest.raises(ValueError):
        reviewer.review(history, True, 23, 1.1)
    with pytest.raises(ValueError):
        reviewer.review(history, 0, 23, float("nan"))
    missing = reviewer.review([[None, 1.3, 1.5, 1.7]] * 22, 0, 23, 1.1)
    assert missing["candidate"] == "keep" and missing["alternatives"] == {}
