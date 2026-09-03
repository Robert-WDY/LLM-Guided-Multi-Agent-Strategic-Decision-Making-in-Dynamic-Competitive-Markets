from pathlib import Path

from game_theory_agent.market import load_market_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v5_enables_cooperation_without_mutating_replay_stable_v4() -> None:
    v4 = load_market_config(PROJECT_ROOT / "configs" / "market_v4.yaml")
    v5 = load_market_config(
        PROJECT_ROOT / "configs" / "market_v5_cooperation.yaml"
    )
    v4_payload = v4.to_dict()
    v5_payload = v5.to_dict()

    assert v4.config_sha256 == (
        "sha256:d3139e6939023a81c430e5ab257d2a9808a57c07d1f01d14ddc922bb7b9d2f01"
    )
    assert v4_payload["persona_utilities"]["capabilities"]["cooperation"] is False
    assert v5_payload["persona_utilities"]["capabilities"]["cooperation"] is True
    assert v5.config_id == "market-v5-cooperation"
    assert v5.config_version == "market-v5.0.0"
    assert v5.environment_version == v4.environment_version
    assert (
        v5_payload["persona_utilities"]["weights_ppm"]
        == v4_payload["persona_utilities"]["weights_ppm"]
    )
    assert (
        v5_payload["persona_utilities"]["traits_ppm"]
        == v4_payload["persona_utilities"]["traits_ppm"]
    )
