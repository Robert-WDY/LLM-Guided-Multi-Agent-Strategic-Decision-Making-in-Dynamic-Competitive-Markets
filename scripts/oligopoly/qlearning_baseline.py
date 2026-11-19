"""Calvano et al. (2020) Q-learning sessions as the non-LLM reference for Delta."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from game_theory_agent.oligopoly.economics import LogitMarket  # noqa: E402
from game_theory_agent.oligopoly.qlearning import run_session  # noqa: E402

OUT = ROOT / "experiment-results" / "oligopoly-v1" / "qlearning" / "sessions.jsonl"


def one(seed: int) -> dict:
    return run_session(LogitMarket(), seed=seed)


if __name__ == "__main__":
    sessions = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["seed"] for l in OUT.read_text().splitlines() if l.strip()}
    seeds = [s for s in range(sessions) if s not in done]
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(one, seeds):
            with OUT.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(r, flush=True)
