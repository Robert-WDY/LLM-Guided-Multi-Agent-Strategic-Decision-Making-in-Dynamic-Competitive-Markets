from game_theory_agent.experiments.final_release_validation import (
    verify_artifact_hash,
)
from game_theory_agent.market.protocols import sha256_hash


def test_release_hash_verifier_supports_canonical_and_named_legacy_protocols():
    canonical = {"value": 7}
    canonical["result_hash"] = sha256_hash(canonical)
    assert verify_artifact_hash(canonical, "result")[0]

    legacy = {"value": 9, "result_hash": "pending"}
    legacy["result_hash"] = sha256_hash(legacy)
    assert verify_artifact_hash(legacy, "legacy_pending_result")[0]
    assert not verify_artifact_hash(legacy, "result")[0]


def test_release_hash_verifier_rejects_tampering():
    payload = {"value": 7}
    payload["result_hash"] = sha256_hash(payload)
    payload["value"] = 8
    assert not verify_artifact_hash(payload, "result")[0]
