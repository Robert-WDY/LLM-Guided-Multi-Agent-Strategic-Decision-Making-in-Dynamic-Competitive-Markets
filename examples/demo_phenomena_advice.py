"""Show the game-theory phenomena advisor end to end.

For each market-coupled phenomenon the advisor:
1. extracts the empirical 2x2 payoff matrix by counterfactual rollouts,
2. classifies the game by the standard payoff ordering,
3. reports the incentive structure (parameters + calculation),
4. recommends an action, and
5. states the structural advice.

Run from the repository root:

    .venv/Scripts/python.exe -I examples/demo_phenomena_advice.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game_theory_agent.market import MarketEnv, MarketConfig, load_market_config  # noqa: E402
from game_theory_agent.game_theory.phenomena import advise, threshold_project_report  # noqa: E402

CFG = "configs/market_v14_local.yaml"


def show(report):
    print(f"\n  == {report['title']} ==")
    print(f"  现象: {report['kind']}")
    print(f"  参数: {report['parameters']}")
    R, S, T, P = (report["payoffs"][k] for k in "RSTP")
    print(f"  支付矩阵(元)  对手合作  对手背叛")
    print(f"    我合作      {R/100:>10,.0f}  {S/100:>10,.0f}")
    print(f"    我背叛      {T/100:>10,.0f}  {P/100:>10,.0f}")
    print(f"  计算: 背叛诱惑 T-R={report['defection_incentive']/100:,.0f} 元, "
          f"合作净价值 R-P={report['mutual_cooperation_value']/100:,.0f} 元")
    print(f"  建议: {report['recommended_action']}")
    print(f"  说明: {report['advice']}")


def main():
    c = load_market_config(CFG)
    e = MarketEnv(c)
    s = e.reset(company_ids=["company_A", "company_B"], episode_id="demo-ph",
                episode_seed=5, max_rounds=20, cooperation_mode="combined_v1")

    print("=== 默认市场：博弈论现象建议 ===")
    report = advise(c, s, "company_A", "company_B", horizon=5)
    for p in report["phenomena"]:
        show(p)
    print(f"\n  范围说明: {report['scope']}")

    print("\n\n=== 门槛降到可达水平后：门槛项目从僵局变为协调博弈 ===")
    base = load_market_config(CFG).to_dict()
    base["strategic_market"]["threshold_project"]["required_total_contribution_cents"] = 600_000
    c2 = MarketConfig.from_mapping(base)
    e2 = MarketEnv(c2)
    s2 = e2.reset(company_ids=["company_A", "company_B"], episode_id="demo-ph2",
                  episode_seed=5, max_rounds=20, cooperation_mode="combined_v1")
    show(threshold_project_report(c2, s2, "company_A", "company_B", horizon=5))


if __name__ == "__main__":
    main()
