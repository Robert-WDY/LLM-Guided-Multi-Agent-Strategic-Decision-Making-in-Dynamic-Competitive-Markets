from __future__ import annotations

from types import SimpleNamespace

import pytest

from game_theory_agent.experiments.real_communication_smoke import (
    _event_interaction_evidence,
    build_matrix,
    parse_artifacts,
    parse_seeds,
)


def test_matrix_blocks_all_three_conditions_by_seed() -> None:
    matrix = build_matrix((810, 811))

    assert [(row["seed"], row["condition"]) for row in matrix] == [
        (810, "off"),
        (810, "public_only"),
        (810, "public_private"),
        (811, "off"),
        (811, "public_only"),
        (811, "public_private"),
    ]
    assert all(row["primary_experiment_unit"] == "paired_seed" for row in matrix)
    assert all(row["communication_is_non_binding"] for row in matrix)


def test_seed_parser_rejects_empty_and_duplicates() -> None:
    assert parse_seeds("810, 811") == (810, 811)
    with pytest.raises(ValueError, match="at least one"):
        parse_seeds(" , ")
    with pytest.raises(ValueError, match="unique"):
        parse_seeds("810,810")


def test_artifact_parser_supports_windows_paths_and_rejects_duplicates() -> None:
    artifacts = parse_artifacts(
        [r"810:off=C:\runs\off", r"810:public_private=C:\runs\on"]
    )

    assert artifacts[(810, "off")].name == "off"
    assert artifacts[(810, "public_private")].name == "on"
    with pytest.raises(ValueError, match="duplicate artifact"):
        parse_artifacts([r"810:off=C:\one", r"810:off=C:\two"])


def _event(*, response_message_id: str = "message-1") -> object:
    message = SimpleNamespace(
        message_id="message-1",
        sender_company_id="company_A",
        channel="private",
        speech_act="proposal",
    )
    phase = SimpleNamespace(
        closure=SimpleNamespace(all_messages=[message]),
        generation_traces=[
            SimpleNamespace(generation_status="submitted"),
            SimpleNamespace(generation_status="not_applicable"),
        ],
    )
    return SimpleNamespace(
        settled_round=1,
        communication_phase=phase,
        traces=[
            SimpleNamespace(
                company_id="company_B",
                communication_view=SimpleNamespace(visible_messages=[message]),
                message_responses=[
                    {
                        "message_id": response_message_id,
                        "disposition": "accepted",
                    }
                ],
                requested_action={"price_cents": 10_500},
                final_action={"price_cents": 10_500},
            )
        ],
    )


def test_interaction_evidence_records_complete_observable_chain() -> None:
    evidence = _event_interaction_evidence([_event()])

    assert evidence["message_count"] == 1
    assert evidence["channel_counts"] == {"private": 1}
    assert evidence["response_count"] == 1
    assert evidence["complete_response_chain_count"] == 1
    assert evidence["all_response_chains_complete"] is True
    assert evidence["hallucinated_message_reference_count"] == 0
    assert evidence["invisible_message_reference_count"] == 0


def test_interaction_evidence_detects_hallucinated_reference() -> None:
    evidence = _event_interaction_evidence(
        [_event(response_message_id="does-not-exist")]
    )

    assert evidence["hallucinated_message_reference_count"] == 1
    assert evidence["complete_response_chain_count"] == 0
    assert evidence["all_response_chains_complete"] is False
