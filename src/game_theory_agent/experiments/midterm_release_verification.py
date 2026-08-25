"""Verify the self-contained Strategic Research MVP v0.7 evidence snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from game_theory_agent.market.protocols import sha256_hash

from .stage66_failure_forensics import _events, _verify_episode


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_ROOT = PROJECT_ROOT / "midterm-release-v0.7"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def run(release_root: Path) -> dict[str, Any]:
    release_root = release_root.resolve()
    catalog_path = release_root / "EXPERIMENT_CATALOG.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_payload = dict(catalog)
    recorded_catalog_hash = str(catalog_payload.pop("catalog_hash"))
    rebuilt_catalog_hash = sha256_hash(
        {
            "protocol": "midterm-experiment-catalog-hash-v1.0.0",
            "catalog": catalog_payload,
        }
    )
    archive_rows = []
    for entry in catalog["experiments"]:
        root = (PROJECT_ROOT / entry["artifact_root"]).resolve()
        if root != release_root and release_root not in root.parents:
            raise ValueError("release experiment escaped the release directory")
        round_events = root / entry["round_events_path"]
        summary = root / entry["summary_path"]
        replay = _verify_episode(_events(root))
        archive_rows.append(
            {
                "experiment_id": entry["experiment_id"],
                "evidence_type": entry["evidence_type"],
                "round_events_sha256": _file_hash(round_events),
                "round_events_hash_matches_catalog": (
                    _file_hash(round_events) == entry["round_events_sha256"]
                ),
                "summary_sha256": _file_hash(summary),
                "summary_hash_matches_catalog": (
                    _file_hash(summary) == entry["summary_sha256"]
                ),
                "replay": replay,
            }
        )
    calibration_path = release_root / "selected-runs" / "forecast-calibration"
    calibration_summary = json.loads(
        (calibration_path / "summary.json").read_text(encoding="utf-8")
    )
    calibration_payload = dict(calibration_summary)
    calibration_hash = str(calibration_payload.pop("report_hash"))
    calibration_hash_rebuilt = sha256_hash(calibration_payload)
    all_archive_replay = all(
        all(
            row["replay"][key]
            for key in (
                "economic_replay",
                "interaction_replay",
                "information_replay",
                "belief_replay",
                "advisor_and_game_theory_replay",
                "adoption_trace_replay",
            )
        )
        and row["replay"]["hidden_state_leak_count"] == 0
        and row["round_events_hash_matches_catalog"]
        and row["summary_hash_matches_catalog"]
        for row in archive_rows
    )
    return {
        "verification_schema_version": "midterm-release-verification-v1.0.0",
        "release_root": str(release_root),
        "experiment_catalog_hash": recorded_catalog_hash,
        "experiment_catalog_hash_rebuilt": rebuilt_catalog_hash,
        "experiment_catalog_hash_replay": (
            recorded_catalog_hash == rebuilt_catalog_hash
        ),
        "archive_rows": archive_rows,
        "all_selected_archive_replays_passed": all_archive_replay,
        "forecast_calibration": {
            "report_hash": calibration_hash,
            "report_hash_rebuilt": calibration_hash_rebuilt,
            "report_hash_replay": calibration_hash == calibration_hash_rebuilt,
            "zero_token_holdout_gate_passed": calibration_summary[
                "zero_token_holdout_gate_passed"
            ],
            "real_llm_gate": calibration_summary["real_llm_gate"],
            "new_real_model_calls": calibration_summary["engineering_checks"][
                "new_real_model_calls"
            ],
        },
        "release_verification_passed": (
            recorded_catalog_hash == rebuilt_catalog_hash
            and all_archive_replay
            and calibration_hash == calibration_hash_rebuilt
            and not calibration_summary["zero_token_holdout_gate_passed"]
            and calibration_summary["real_llm_gate"]
            == "closed_no_real_llm_calls"
            and calibration_summary["engineering_checks"]["new_real_model_calls"]
            == 0
        ),
        "claim_boundary": (
            "release integrity and deterministic replay passed; the forecast "
            "prediction gate intentionally remains failed"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    args = parser.parse_args()
    result = run(args.release_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["release_verification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
