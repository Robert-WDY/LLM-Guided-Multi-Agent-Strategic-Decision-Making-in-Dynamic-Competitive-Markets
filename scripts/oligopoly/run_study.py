"""Run a named cell grid of the v21 oligopoly study.

Usage:
    python scripts/oligopoly/run_study.py --phase pilot
    python scripts/oligopoly/run_study.py --phase main --max-tokens 40000000
    python scripts/oligopoly/run_study.py --phase main --offline   # replay from cache

Every episode is identified by (phase, cell, seed). Finished episodes are
skipped, so the command can be re-run after an interruption. LLM responses are
cached in experiment-results/oligopoly-v1/cache/<phase>.jsonl[.gz].
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_theory_agent.oligopoly.agents import LLMPricingAgent, RuleAgent  # noqa: E402
from game_theory_agent.oligopoly.economics import LogitMarket  # noqa: E402
from game_theory_agent.oligopoly.llm import (  # noqa: E402
    BudgetExceeded,
    OpenAICompatibleModel,
    ResponseCache,
    TokenBudget,
)
from game_theory_agent.oligopoly.runner import EpisodeSpec, run_episode  # noqa: E402
from grids import PHASES, MODELS  # noqa: E402

OUT = ROOT / "experiment-results" / "oligopoly-v1"
SCALES = (1.0, 3.2, 10.0)  # Fish, Gonczarowski & Shorrer (2024) price-scale design


def load_env() -> None:
    for name in (".env.local-keys", ".env"):
        path = ROOT / name
        if path.exists():
            for line in path.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["episode_id"] for l in path.read_text().splitlines() if l.strip()}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument("--max-tokens", type=int, default=20_000_000)
    ap.add_argument("--parallel", type=int, default=12, help="episodes in flight")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated cell names")
    args = ap.parse_args()
    load_env()

    cells = PHASES[args.phase]
    if args.only:
        keep = set(args.only.split(","))
        cells = [c for c in cells if c["cell"] in keep]
    cache = ResponseCache(OUT / "cache" / f"{args.phase}.jsonl")
    budget = TokenBudget(args.max_tokens)
    models: dict[str, OpenAICompatibleModel] = {}

    def model(alias: str) -> OpenAICompatibleModel:
        if alias not in models:
            provider, name, thinking = MODELS[alias]
            models[alias] = OpenAICompatibleModel(
                provider, name, cache=cache, budget=budget, thinking=thinking,
                concurrency=24, offline=args.offline,
            )
        return models[alias]

    phase_dir = OUT / (f"{args.phase}-replay" if args.offline else args.phase)
    summary_path = phase_dir / "summaries.jsonl"
    finished = done_ids(summary_path)
    jobs = []
    for cell in cells:
        for seed in cell["seeds"]:
            eid = f"{args.phase}-{cell['cell']}-s{seed}"
            if eid not in finished:
                jobs.append((cell, seed, eid))
    print(f"{len(jobs)} episodes to run ({len(finished)} already done)", flush=True)
    sem = asyncio.Semaphore(args.parallel)
    started = time.time()
    completed = 0

    async def one(cell, seed, eid):
        nonlocal completed
        async with sem:
            n = cell["n"]
            scale = SCALES[seed % len(SCALES)]
            market = LogitMarket(n=n, scale=scale)
            agents = []
            for i, firm in enumerate(cell["firms"]):
                if firm.startswith("rule:"):
                    agents.append(RuleAgent(market, firm.split(":", 1)[1]))
                else:
                    agents.append(
                        LLMPricingAgent(
                            model(firm),
                            persona=cell["personas"][i],
                            communication=cell.get("communication", False),
                            max_price=5 * scale,
                            episode_key=f"{cell['cell']}|seed{seed}",
                        )
                    )
            spec = EpisodeSpec(
                episode_id=eid, n=n, rounds=cell["rounds"], scale=scale,
                window=cell["window"],
                condition={k: v for k, v in cell.items() if k != "seeds"} | {"seed": seed},
            )
            summary = await run_episode(
                spec, agents,
                communication=cell.get("communication", False),
                advisor=cell.get("advisor"),
                out_dir=phase_dir / "episodes",
            )
            with summary_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
            completed += 1
            print(
                f"[{completed}/{len(jobs)} {time.time() - started:.0f}s tokens={budget.used}] "
                f"{eid} delta={summary['delta']:.3f} price_idx={summary['price_index']:.3f} "
                f"invalid={summary['invalid_rate']:.3f}",
                flush=True,
            )

    results = await asyncio.gather(*(one(*j) for j in jobs), return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    for e in errors[:5]:
        print("ERROR", type(e).__name__, str(e)[:300], flush=True)
    print(f"done: {completed} ok, {len(errors)} failed, tokens sent={budget.used}", flush=True)
    if args.offline:
        verify_replay(OUT / args.phase / "summaries.jsonl", summary_path)
    if any(isinstance(e, BudgetExceeded) for e in errors):
        sys.exit(3)


def verify_replay(original: Path, replay: Path) -> None:
    """Offline replay must reproduce every archived episode exactly."""

    def by_id(p):
        return {json.loads(l)["episode_id"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}

    a, b = by_id(original), by_id(replay)
    keys = ("delta", "price_index", "total_welfare", "mean_price_by_firm")
    mismatched = [e for e in a if e not in b or any(a[e][k] != b[e][k] for k in keys)]
    print(f"replay check: {len(a) - len(mismatched)}/{len(a)} episodes identical", flush=True)
    if mismatched:
        print("mismatched:", mismatched[:10], flush=True)
        sys.exit(4)


if __name__ == "__main__":
    asyncio.run(main())
