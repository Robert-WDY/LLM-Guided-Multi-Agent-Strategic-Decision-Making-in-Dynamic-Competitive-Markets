from game_theory_agent.experiments.stage66_failure_forensics import (
    decompose_regret,
    economic_action,
    map_action_to_candidate,
)


def test_migrated_source_summary_resolves_without_rewriting_evidence(tmp_path, monkeypatch):
    from game_theory_agent.experiments import stage66_failure_forensics as forensic

    monkeypatch.setattr(forensic, "PROJECT_ROOT", tmp_path)
    local = tmp_path / "runs" / "recorded-pilot" / "summary.json"
    local.parent.mkdir(parents=True)
    local.write_text('{"frozen":true}\n', encoding="utf-8")
    original = local.read_bytes()
    for source in (
        "C:/Users/12204/Desktop/game-theory-agent/runs/recorded-pilot/summary.json",
        r"C:\Users\12204\Desktop\game-theory-agent\runs\recorded-pilot\summary.json",
    ):
        assert forensic._resolve_source_summary(source) == local
    assert local.read_bytes() == original


def test_source_summary_keeps_existing_paths_and_rejects_unrelated_relocation(tmp_path, monkeypatch):
    from pathlib import Path
    from game_theory_agent.experiments import stage66_failure_forensics as forensic

    monkeypatch.setattr(forensic, "PROJECT_ROOT", tmp_path)
    existing = tmp_path / "summary.json"
    existing.write_text("{}", encoding="utf-8")
    assert forensic._resolve_source_summary(str(existing)) == existing
    for source in (
        "C:/unrelated/runs/pilot/summary.json",
        "C:/Users/12204/Desktop/game-theory-agent/runs/../summary.json",
    ):
        assert forensic._resolve_source_summary(source) == Path(source)


def test_forensic_regret_decomposition_is_additive():
    result = decompose_regret(
        oracle_value=1_000,
        advisor_value=900,
        llm_value=850,
        final_value=825,
    )

    assert result == {
        "planner_regret_cents": 100,
        "adoption_regret_cents": 50,
        "execution_regret_cents": 25,
        "oracle_to_final_regret_cents": 175,
        "additive_identity_residual_cents": 0,
    }


def test_forensic_candidate_mapping_does_not_disguise_custom_action():
    candidates = {
        "maintain": economic_action({"price_cents": 10_000}),
        "price_cut_small": economic_action({"price_cents": 9_500}),
    }

    exact = map_action_to_candidate({"price_cents": 9_500}, candidates)
    custom = map_action_to_candidate(
        {"price_cents": 9_500, "advertising_budget_cents": 1}, candidates
    )

    assert exact["mapping_status"] == "exact_candidate"
    assert exact["candidate_id"] == "price_cut_small"
    assert custom["mapping_status"] == "custom_action"
    assert custom["candidate_id"] == "custom_action"
    assert custom["nearest_candidate_id"] == "price_cut_small"
    assert custom["different_field_count"] == 1
