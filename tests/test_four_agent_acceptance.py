from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from game_theory_agent.experiments.four_agent_acceptance import (
    _active_round_evidence,
    _ensure_clean_round_event_log,
    _validate_acceptance_args,
)


def _args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "provider": "mock",
        "rounds": 20,
        "llm_count": 4,
        "communication_mode": "public_private",
        "mock_communication_scenario": "silence",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("provider", ["doubao", "deepseek"])
def test_active_scenario_is_mock_only(provider: str) -> None:
    with pytest.raises(ValueError, match="requires --provider mock"):
        _validate_acceptance_args(
            _args(provider=provider, mock_communication_scenario="public_price")
        )


def test_private_active_scenario_requires_two_llm_agents() -> None:
    with pytest.raises(ValueError, match="--llm-count >= 2"):
        _validate_acceptance_args(
            _args(llm_count=1, mock_communication_scenario="private_price")
        )


@pytest.mark.parametrize("scenario", ["public_price", "mixed"])
def test_public_active_scenario_requires_four_llm_agents(scenario: str) -> None:
    with pytest.raises(ValueError, match="--llm-count 4"):
        _validate_acceptance_args(
            _args(llm_count=3, mock_communication_scenario=scenario)
        )


def test_active_scenario_requires_exactly_twenty_rounds() -> None:
    with pytest.raises(ValueError, match="exactly 20 rounds"):
        _validate_acceptance_args(
            _args(rounds=10, mock_communication_scenario="public_price")
        )


def test_silence_scenario_preserves_non_mock_and_short_run_compatibility() -> None:
    _validate_acceptance_args(
        _args(
            provider="deepseek",
            rounds=5,
            llm_count=1,
            communication_mode="off",
        )
    )


def test_non_empty_round_event_log_is_never_reused(tmp_path: Path) -> None:
    path = tmp_path / "round-events.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to append"):
        _ensure_clean_round_event_log(path)

    empty_path = tmp_path / "empty.jsonl"
    empty_path.touch()
    _ensure_clean_round_event_log(empty_path)


def _mixed_event(round_number: int, *, company_b_price: int = 12_345) -> object:
    public_message = SimpleNamespace(
        sender_company_id="company_A",
        channel="public",
        message_id=f"public-{round_number}",
        recipients=[],
        speech_act="proposal",
        requested_peer_action=SimpleNamespace(price_cents=11_000),
    )
    private_message = SimpleNamespace(
        sender_company_id="company_A",
        channel="private",
        message_id=f"private-{round_number}",
        recipients=["company_B"],
        speech_act="proposal",
        requested_peer_action=SimpleNamespace(price_cents=12_345),
    )
    submission = SimpleNamespace(
        messages=[
            SimpleNamespace(channel="public"),
            SimpleNamespace(channel="private"),
        ]
    )
    generation = SimpleNamespace(
        company_id="company_A",
        generation_status="submitted",
        submission=submission,
        accepted_message_ids=[
            public_message.message_id,
            private_message.message_id,
        ],
    )
    phase = SimpleNamespace(
        closure=SimpleNamespace(
            all_messages=[public_message, private_message]
        ),
        generation_traces=[generation],
    )
    traces = [
        SimpleNamespace(
            company_id="company_A",
            communication_view=SimpleNamespace(
                visible_messages=[public_message, private_message]
            ),
            message_responses=[],
            requested_action={"price_cents": 10_000},
        ),
        SimpleNamespace(
            company_id="company_B",
            communication_view=SimpleNamespace(
                visible_messages=[public_message, private_message]
            ),
            message_responses=[
                {"message_id": public_message.message_id, "disposition": "accepted"},
                {"message_id": private_message.message_id, "disposition": "accepted"},
            ],
            requested_action={"price_cents": company_b_price},
        ),
        *[
            SimpleNamespace(
                company_id=company_id,
                communication_view=SimpleNamespace(
                    visible_messages=[public_message]
                ),
                message_responses=[
                    {
                        "message_id": public_message.message_id,
                        "disposition": "accepted",
                    }
                ],
                requested_action={"price_cents": 11_000},
            )
            for company_id in ("company_C", "company_D")
        ],
    ]
    return SimpleNamespace(
        settled_round=round_number,
        communication_phase=phase,
        traces=traces,
        joint_action={
            "company_A": {"price_cents": 10_000},
            "company_B": {"price_cents": company_b_price},
            "company_C": {"price_cents": 11_000},
            "company_D": {"price_cents": 11_000},
        },
    )


def test_mixed_active_contract_is_checked_for_every_round() -> None:
    events = [_mixed_event(round_number) for round_number in range(1, 21)]

    checks, evidence = _active_round_evidence(events, "mixed")

    assert all(checks.values())
    assert len(evidence) == 20
    assert all(row["a_actual_message_count"] == 2 for row in evidence)
    assert all(
        row["a_actual_channels"] == ["private", "public"]
        for row in evidence
    )
    assert all(row["a_generation_submitted"] for row in evidence)
    assert all(row["target_prices_met"] for row in evidence)


def test_one_bad_round_fails_active_target_price_contract() -> None:
    events = [_mixed_event(round_number) for round_number in range(1, 20)]
    events.append(_mixed_event(20, company_b_price=11_000))

    checks, evidence = _active_round_evidence(events, "mixed")

    assert checks["active_interaction_exactly_20_rounds"]
    assert not checks["active_target_final_price_every_round"]
    assert not checks["active_round_contract_every_round"]
    assert evidence[-1]["target_prices_met"] is False
