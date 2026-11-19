"""Run one repeated pricing episode and summarise it against Nash/monopoly."""

from __future__ import annotations

import asyncio
import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from game_theory_agent.oligopoly.agents import (
    FIRM_NAMES,
    MARKET_SIZE,
    Decision,
    Observation,
    RoundRecord,
    advisor_report,
)
from game_theory_agent.oligopoly.economics import LogitMarket


@dataclass
class EpisodeSpec:
    episode_id: str
    n: int
    rounds: int
    scale: float
    window: int
    condition: dict[str, Any] = field(default_factory=dict)


async def run_episode(
    spec: EpisodeSpec,
    agents: list[Any],
    *,
    communication: bool = False,
    advisor: str | None = None,
    opening: str | None = None,
    advisor_firms: list[int] | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """``opening`` = "nash" or "monopoly" fixes every firm's round-1 price."""
    if len(agents) != spec.n:
        raise ValueError("one agent per firm")
    market = LogitMarket(n=spec.n, scale=spec.scale)
    histories: list[list[RoundRecord]] = [[] for _ in range(spec.n)]
    messages: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for t in range(1, spec.rounds + 1):
        observations = []
        for i in range(spec.n):
            obs = Observation(
                firm=i,
                n=spec.n,
                round=t,
                marginal_cost=market.c * market.scale,
                history=histories[i],
                rival_messages={k: v for k, v in messages.items() if k != FIRM_NAMES[i]},
            )
            if advisor and (advisor_firms is None or i in advisor_firms):
                obs.advisor = advisor_report(market, obs, advisor)
            observations.append(obs)
        if t == 1 and opening is not None:
            fixed = {"nash": market.nash_price, "monopoly": market.monopoly_price}[opening]
            decisions = [Decision(price=fixed) for _ in range(spec.n)]
        else:
            decisions = await asyncio.gather(
                *(agent.decide(obs) for agent, obs in zip(agents, observations))
            )
        prices = [d.price for d in decisions]
        out = market.outcome(prices)
        for i in range(spec.n):
            histories[i].append(
                RoundRecord(
                    round=t,
                    own_price=prices[i],
                    rival_prices=[p for j, p in enumerate(prices) if j != i],
                    units=out["quantities"][i] * MARKET_SIZE,
                    profit=out["profits"][i] * MARKET_SIZE,
                )
            )
        if communication:
            messages = {
                FIRM_NAMES[i]: d.message for i, d in enumerate(decisions) if d.message
            }
        rows.append(
            {
                "round": t,
                "prices": prices,
                "quantities": out["quantities"],
                "profits": out["profits"],
                "consumer_surplus": out["consumer_surplus"],
                "total_welfare": out["total_welfare"],
                "valid": [d.valid for d in decisions],
                "messages": [d.message for d in decisions],
                "notes": [d.notes for d in decisions],
                "input_tokens": sum(d.input_tokens for d in decisions),
                "output_tokens": sum(d.output_tokens for d in decisions),
                "new_requests": sum(1 for d in decisions if d.raw is not None and not d.cached),
            }
        )
    summary = summarise(spec, market, rows, agents)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_dir / f"{spec.episode_id}.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({"spec": asdict(spec), "summary": summary}) + "\n")
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def summarise(
    spec: EpisodeSpec, market: LogitMarket, rows: list[dict[str, Any]], agents: list[Any]
) -> dict[str, Any]:
    tail = rows[-spec.window :]
    n = spec.n
    mean_price = [mean(r["prices"][i] for r in tail) for i in range(n)]
    mean_profit = [mean(r["profits"][i] for r in tail) for i in range(n)]
    avg_profit = mean(mean_profit)
    avg_price = mean(mean_price)
    bench = market.benchmarks()
    welfare = mean(r["total_welfare"] for r in tail)
    cs = mean(r["consumer_surplus"] for r in tail)
    return {
        **spec.condition,
        "episode_id": spec.episode_id,
        "n": n,
        "rounds": spec.rounds,
        "window": spec.window,
        "scale": spec.scale,
        "agents": [getattr(a, "kind", type(a).__name__) for a in agents],
        "mean_price": avg_price,
        "mean_price_by_firm": mean_price,
        "price_index": market.price_index(avg_price),
        "delta": market.collusion_index(avg_profit),
        "delta_by_firm": [market.collusion_index(p) for p in mean_profit],
        "consumer_surplus": cs,
        "total_welfare": welfare,
        "welfare_vs_nash": (welfare - bench["nash_welfare"])
        / (bench["monopoly_welfare"] - bench["nash_welfare"]),
        "cs_loss_vs_nash": (bench["nash_consumer_surplus"] - cs)
        / (bench["nash_consumer_surplus"] - bench["monopoly_consumer_surplus"]),
        "tail_price_sd": mean(pstdev([r["prices"][i] for r in tail]) for i in range(n))
        / market.scale,
        "cross_firm_price_gap": mean(max(r["prices"]) - min(r["prices"]) for r in tail)
        / market.scale,
        "invalid_rate": mean(1 - sum(r["valid"]) / n for r in rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "new_requests": sum(r["new_requests"] for r in rows),
        "price_path_mean": [mean(r["prices"]) / market.scale for r in rows],
        "benchmarks": bench,
    }
