"""Demonstrate that the market expresses both cooperation and competition.

Cooperation: the ``cooperator`` portfolio contributes to the shared cold-chain
threshold project and the shared-resilience pool, so the public good actually
gets provisioned (the project succeeds) instead of silently failing.

Competition: the ``undercutter``/``value``/``margin`` portfolios spread prices
and shift market share, which is what drives concentration and exit pressure.

Run from the repository root:

    .venv/Scripts/python.exe -I examples/demo_coop_competition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game_theory_agent.market import MarketEnv, load_market_config  # noqa: E402
from game_theory_agent.market.actor_policies import company_action  # noqa: E402

CFG = "configs/market_v14_local.yaml"


def run(n, choices, seed=42, rounds=10):
    c = load_market_config(CFG)
    e = MarketEnv(c)
    s = e.reset(
        company_ids=[f"company_{chr(65+i)}" for i in range(n)],
        episode_id=f"demo-{seed}-{n}",
        episode_seed=seed,
        max_rounds=rounds,
        cooperation_mode="combined_v1",
    )
    prices = {}
    shares = {}
    statuses = []
    for _ in range(rounds):
        if s.terminal:
            break
        aa = {
            cid: company_action(c, s, cid, choices.get(cid, "balanced"))
            for cid in s.company_ids
        }
        s = e.step(f"{s.episode_id}:{s.round}:{s.state_version}", aa).state_after
        tp = s.strategic_market.threshold_project
        statuses.append(tp.status.value if tp else "none")
    for cid in s.company_ids:
        comp = s.company(cid)
        prices[cid] = comp.commercial.price_cents / 100
        shares[cid] = comp.commercial.market_share_ppm / 10000
    tp = s.strategic_market.threshold_project
    return dict(
        statuses=statuses,
        prices=prices,
        shares=shares,
        hhi=s.strategic_market.hhi_ppm / 10000,
        contribution=tp.accumulated_total_contribution_cents if tp else 0,
        required=tp.required_total_contribution_cents if tp else 0,
    )


def main() -> None:
    print("== 合作：4 家公司都选 cooperator ==")
    r = run(4, {cid: "cooperator" for cid in (f"company_{c}" for c in "ABCD")})
    print(f"  门槛项目状态序列: {r['statuses']}")
    print(f"  累计贡献 {r['contribution']/100:,.0f} 元 / 门槛 {r['required']/100:,.0f} 元")
    print(f"  -> 项目{'成功，公共冷链被建设' if 'succeeded' in r['statuses'] else '未成功'}\n")

    print("== 搭便车：只有 company_A 合作，其余 balanced ==")
    r = run(4, {"company_A": "cooperator"})
    print(f"  门槛项目状态序列: {r['statuses']}")
    print(f"  累计贡献 {r['contribution']/100:,.0f} 元 / 门槛 {r['required']/100:,.0f} 元")
    print(f"  -> 单一贡献者{'独自达标' if 'succeeded' in r['statuses'] else '不足以达标（集体行动困境）'}\n")

    print("== 竞争：undercutter / value / margin / balanced 各一 ==")
    r = run(4, {
        "company_A": "undercutter",
        "company_B": "value",
        "company_C": "margin",
        "company_D": "balanced",
    })
    for cid, p in sorted(r["prices"].items(), key=lambda kv: kv[1]):
        print(f"  {cid}: 价格 {p:,.0f} 元, 份额 {r['shares'][cid]:.1f}%")
    print(f"  HHI（集中度）={r['hhi']:.1f}%")
    print(f"  -> 价格出现明显分层，低价策略抢占份额（竞争可体现）")


if __name__ == "__main__":
    main()
