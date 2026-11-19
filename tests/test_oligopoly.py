from __future__ import annotations

import asyncio
import json

import pytest

from game_theory_agent.oligopoly.agents import (
    LLMPricingAgent,
    Observation,
    RoundRecord,
    RuleAgent,
    advisor_report,
    build_prompt,
    parse_reply,
)
from game_theory_agent.oligopoly.economics import LogitMarket
from game_theory_agent.oligopoly.llm import ResponseCache, ScriptedModel, request_key
from game_theory_agent.oligopoly.qlearning import run_session
from game_theory_agent.oligopoly.runner import EpisodeSpec, run_episode


def test_calvano_benchmarks_reproduced() -> None:
    m = LogitMarket()
    assert m.nash_price == pytest.approx(1.4729, abs=1e-4)
    assert m.monopoly_price == pytest.approx(1.9250, abs=1e-4)
    assert m.collusion_index(m.nash_profit) == pytest.approx(0.0)
    assert m.collusion_index(m.monopoly_profit) == pytest.approx(1.0)


def test_scale_is_pure_relabelling() -> None:
    base, scaled = LogitMarket(n=3), LogitMarket(n=3, scale=3.7)
    assert scaled.nash_price == pytest.approx(3.7 * base.nash_price)
    assert scaled.monopoly_price == pytest.approx(3.7 * base.monopoly_price)
    o1 = base.outcome([1.5, 1.6, 1.7])
    o2 = scaled.outcome([3.7 * 1.5, 3.7 * 1.6, 3.7 * 1.7])
    assert o2["quantities"] == pytest.approx(o1["quantities"])
    assert o2["total_welfare"] == pytest.approx(3.7 * o1["total_welfare"])


def test_nash_is_mutual_best_response_and_welfare_ordering() -> None:
    m = LogitMarket(n=4)
    assert m.best_response([m.nash_price] * 3) == pytest.approx(m.nash_price, abs=1e-9)
    b = m.benchmarks()
    assert b["nash_welfare"] > b["monopoly_welfare"]
    assert b["nash_consumer_surplus"] > b["monopoly_consumer_surplus"]


def test_parse_reply_handles_wrapped_and_string_prices() -> None:
    assert parse_reply('```json\n{"price": 1.8, "notes": "x"}\n```')["price"] == 1.8
    assert parse_reply('{"price": "$2.10"}')["price"] == 2.10
    assert parse_reply("no json") is None
    assert parse_reply('{"notes": "no price"}') is None


def test_prompt_never_reveals_demand_or_rival_profit() -> None:
    obs = Observation(
        firm=0, n=2, round=3, marginal_cost=1.0,
        history=[RoundRecord(1, 1.5, [1.6], 47.0, 23.5), RoundRecord(2, 1.55, [1.6], 46.0, 25.3)],
        rival_messages={"B": "hello"},
    )
    text = build_prompt(obs, persona="neutral", communication=True)[0]["content"]
    assert "Firm B price" in text and "hello" in text and '"message"' in text
    assert "mu" not in text and "logit" not in text.lower() and "Nash" not in text
    text2 = build_prompt(obs, persona="neutral", communication=False)[0]["content"]
    assert "hello" not in text2 and '"message"' not in text2


def test_advisor_variants() -> None:
    m = LogitMarket()
    obs = Observation(0, 2, 2, 1.0, [RoundRecord(1, 1.5, [1.6], 47.0, 23.5)])
    br = advisor_report(m, obs, "br")
    full = advisor_report(m, obs, "full")
    assert f"{m.best_response([1.6]):.2f}" in br and "1.47" in br
    assert "1.92" not in br and "1.92" in full


def test_invalid_reply_falls_back_after_one_repair() -> None:
    model = ScriptedModel(lambda msgs: "I think 9999")
    agent = LLMPricingAgent(model, max_price=10, episode_key="t")
    obs = Observation(0, 2, 2, 1.0, [RoundRecord(1, 1.7, [1.6], 40.0, 28.0)])
    d = asyncio.run(agent.decide(obs))
    assert not d.valid and d.price == 1.7 and model.calls == 2


def test_episode_with_rules_and_scripted_llm(tmp_path) -> None:
    m = LogitMarket(scale=2.0)
    llm = LLMPricingAgent(
        ScriptedModel(lambda msgs: json.dumps({"price": 3.9, "notes": "hold", "message": "hi"})),
        communication=True, max_price=20, episode_key="e",
    )
    spec = EpisodeSpec("ep", n=2, rounds=12, scale=2.0, window=5, condition={"cell": "t"})
    summary = asyncio.run(
        run_episode(spec, [llm, RuleAgent(m, "myopic_br")], communication=True,
                    advisor="br", out_dir=tmp_path)
    )
    assert summary["mean_price_by_firm"][0] == pytest.approx(3.9)
    assert summary["mean_price_by_firm"][1] == pytest.approx(m.best_response([3.9]))
    assert summary["invalid_rate"] == 0 and summary["cell"] == "t"
    assert (tmp_path / "ep.jsonl.gz").exists()


def test_myopic_best_responders_converge_to_nash() -> None:
    m = LogitMarket(n=3)
    spec = EpisodeSpec("br", n=3, rounds=40, scale=1.0, window=10)
    s = asyncio.run(run_episode(spec, [RuleAgent(m, "myopic_br") for _ in range(3)]))
    assert s["delta"] == pytest.approx(0.0, abs=1e-6)


def test_response_cache_roundtrip(tmp_path) -> None:
    cache = ResponseCache(tmp_path / "c.jsonl")
    k = request_key("m", [{"role": "user", "content": "x"}], "s")
    cache.put({"key": k, "text": "t", "input_tokens": 1, "output_tokens": 2})
    assert ResponseCache(tmp_path / "c.jsonl").get(k)["text"] == "t"


def test_qlearning_smoke() -> None:
    r = run_session(LogitMarket(), seed=1, m=5, stable=200, max_steps=20_000, beta=5e-4)
    assert r["cycle_length"] >= 1 and -1.5 < r["delta"] < 1.5
