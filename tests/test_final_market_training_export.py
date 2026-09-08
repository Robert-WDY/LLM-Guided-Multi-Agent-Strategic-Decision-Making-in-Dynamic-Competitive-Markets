from __future__ import annotations

import pytest

from game_theory_agent.experiments.export_final_market_training_data import (
    _build_examples,
)


def _row(split: str, treatment: str, value: int):
    return {
        "split": split,
        "seed": 1,
        "resident_strategy_id": "resident",
        "focal_company_id": "company_A",
        "treatment_strategy_id": treatment,
        "focal_enterprise_value_cents": value,
        "episode": {
            "market_model": "balanced",
            "final_state_hash": f"sha256:{treatment}",
        },
    }


def test_training_export_prefers_higher_counterfactual_and_hides_opponent_type():
    examples = _build_examples(
        [
            _row("development", "balanced_competitor", 100),
            _row("development", "premium_defender", 120),
        ]
    )
    assert len(examples) == 1
    assert examples[0]["preferred_strategy_id"] == "premium_defender"
    assert "resident_strategy_id" not in examples[0]["input_context"]
    assert examples[0]["uses_hidden_opponent_state"] is False


def test_training_export_rejects_holdout_contamination():
    with pytest.raises(ValueError, match="holdout/test"):
        _build_examples(
            [
                _row("holdout", "balanced_competitor", 100),
                _row("holdout", "premium_defender", 120),
            ]
        )
