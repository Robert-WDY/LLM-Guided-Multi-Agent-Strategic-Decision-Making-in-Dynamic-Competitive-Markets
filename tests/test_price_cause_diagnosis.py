from contextlib import contextmanager
import importlib.util
from pathlib import Path
import numpy as np
from game_theory_agent.game_theory.quote_panel import public_state
from game_theory_agent.game_theory.panel_advisor import PanelAdvisor

ROOT = Path(__file__).resolve().parents[1]


def test_public_panel_distinguishes_same_median_with_different_competition():
    a = np.tile([1.38, 1.25, 1.51, 1.51], (26, 1))
    b = np.tile([1.38, 1.51, 1.51, 2.03], (26, 1))
    assert np.median(a[-1, 1:]) == np.median(b[-1, 1:])
    assert public_state(a, 0, 27, 1.38) != public_state(b, 0, 27, 1.38)
    advisor = PanelAdvisor(ROOT / "experiment-results/price-causes-v1/panel-econ.npz")
    assert len(advisor.review(a, 0, 27, 1.38)["forecasts"]) == 3
    a[:, 1:] = np.nan
    assert advisor.review(a, 0, 27, 1.38)["forecasts"] == []


def test_cache_misses_share_the_cpu_server_and_never_request_acceleration(monkeypatch):
    path = ROOT / "scripts/deployment/cpu_server.py"
    spec = importlib.util.spec_from_file_location("pricing_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class Process:
        def poll(self):
            return None

    @contextmanager
    def start(name, port, **options):
        calls.append((name, port))
        try:
            yield Process()
        finally:
            calls.append("released")

    monkeypatch.setattr(module, "service", start)
    assert module.CPU_ARGS == [
        "--device",
        "none",
        "--gpu-layers",
        "0",
        "--no-kv-offload",
        "--no-op-offload",
    ]
    assert not module._active
    module.start_on_cache_miss("smol", 19943)
    module.start_on_cache_miss("smol", 19943)
    module._owned_services.close()
    assert calls == [("smol", 19943), "released"]
