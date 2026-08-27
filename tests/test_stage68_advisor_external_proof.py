import pytest

from game_theory_agent.experiments.stage68_advisor_external_proof import (
    EXTERNAL_SOURCES,
    PROJECT_ROOT,
    build_preregistration,
    exact_two_sided_sign_test_ppm,
)


def test_stage68_preregistration_freezes_external_real_episode_pool():
    source_dir = PROJECT_ROOT / str(EXTERNAL_SOURCES[0]["directory"])
    if not (source_dir / "summary.json").is_file():
        pytest.skip("recorded external audit sources are not present")
    result = build_preregistration()

    assert result["source_episode_count"] == 6
    assert result["independent_seed_count"] == 4
    assert result["source_round_window_count"] == 45
    assert result["real_llm_gate"]["maximum_calls_if_open"] == 6
    assert result["profile_is_not_refit"] is True
    assert result["preregistration_hash"].startswith("sha256:")
    assert all(
        item["evidence_type"] == "REAL_RECORDED_LLM_EPISODE"
        for item in result["sources"]
    )


def test_exact_sign_test_uses_independent_clusters_not_round_windows():
    assert exact_two_sided_sign_test_ppm([1, 2, 3, 4]) == 125_000
    assert exact_two_sided_sign_test_ppm([1, 2, -3, -4]) == 1_000_000
    assert exact_two_sided_sign_test_ppm([0, 0]) is None
