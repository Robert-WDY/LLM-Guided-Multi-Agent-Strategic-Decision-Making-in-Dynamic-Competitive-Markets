"""Generate LaTeX tables and value macros for the final report from analysis.json.

python scripts/oligopoly/report_tables.py  -> report/final/generated/*.tex

The report cites numbers through \\V{key} (defined in generated/values.tex), so
every figure in the text comes from the same analysis output as the tables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = json.loads((ROOT / "experiment-results" / "oligopoly-v1" / "analysis.json").read_text())
OUT = Path(os.environ.get("REPORT_DIR", ROOT / "report" / "final")) / "generated"

LABEL = {
    "db-mini": "Doubao mini", "db-lite": "Doubao lite", "db-pro": "Doubao pro",
    "ds-flash": "DeepSeek Flash", "ds-pro": "DeepSeek Pro",
}


def f2(x: float) -> str:
    return f"{x:.2f}".replace("-", "$-$")


def ci(c) -> str:
    return f"[{f2(c[0])}, {f2(c[1])}]"


def pv(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}\\%"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = {c["cell"]: c for c in A["cells"]}
    vals: dict[str, str] = {}

    for name, c in cells.items():
        vals[f"{name}/delta"] = f2(c["delta_mean"])
        vals[f"{name}/deltaci"] = ci(c["delta_ci"])
        vals[f"{name}/median"] = f2(c["delta_median"])
        vals[f"{name}/rho"] = f2(c["price_index_mean"])
        vals[f"{name}/rhoci"] = ci(c["price_index_ci"])
        vals[f"{name}/above"] = pct(c["share_above_monopoly_price"])
        vals[f"{name}/aboveN"] = str(round(c["share_above_monopoly_price"] * c["episodes"]))
        vals[f"{name}/cs"] = f2(c["cs_loss_vs_nash"])
        vals[f"{name}/wl"] = f2(c["welfare_vs_nash"])
        vals[f"{name}/invalid"] = f"{100 * c['invalid_rate']:.1f}\\%"
        vals[f"{name}/n"] = str(c["episodes"])
        vals[f"{name}/belownash"] = str(c["episodes_with_firm_below_nash"])
        vals[f"{name}/gap"] = f2(c["price_gap_norm"])
    for k, t in A["tests"].items():
        key = "test/" + k.replace(" ", "_")
        vals[key + "/diff"] = f2(t["diff"])
        vals[key + "/ci"] = ci(t["ci"])
        vals[key + "/p"] = pv(t["p"])
        if "p_holm" in t:
            vals[key + "/holm"] = pv(t["p_holm"])
    for k, h in A["h1"].items():
        key = "h1/" + k.replace(" ", "_")
        vals[key + "/p"] = pv(h["p"])
        vals[key + "/pb"] = pv(h["p_bonferroni"])
    for m, e in A.get("e5", {}).items():
        for f in ("price_B_minus_A", "delta_A_minus_B", "delta_A", "delta_B", "delta_total"):
            vals[f"e5/{m}/{f}"] = f"{e[f]:.3f}".replace("-", "$-$") if f == "price_B_minus_A" else f2(e[f])
        vals[f"e5/{m}/b_above"] = str(e["episodes_B_above_A"])
        vals[f"e5/{m}/n"] = str(e["n"])
        vals[f"e5/{m}/price_ci"] = "[" + ", ".join(f"{x:.3f}" for x in e["price_B_minus_A_ci"]) + "]"
        vals[f"e5/{m}/delta_ci"] = ci(e["delta_A_minus_B_ci"])
        vals[f"e5/{m}/p_price"] = pv(e["p_price_one_sided"])
        vals[f"e5/{m}/p_delta"] = pv(e["p_delta_two_sided"])
    for k, e in A.get("e6", {}).items():
        if k.startswith("within_episode:"):
            name = k.split(":", 1)[1]
            vals[f"long/{name}/early"] = f2(e["delta_rounds_36_50"])
            vals[f"long/{name}/late"] = f2(e["delta_rounds_121_150"])
            vals[f"long/{name}/diff"] = f2(e["diff"])
            vals[f"long/{name}/ci"] = ci(e["diff_ci"])
            vals[f"long/{name}/rose"] = str(e["rose"])
            vals[f"long/{name}/fell"] = str(e["fell"])
            vals[f"long/{name}/n"] = str(e["n"])
        else:
            vals[f"long/{k}/escalated"] = pct(e["share_escalated_delta_gt_0.25"])
    for name, e in A.get("split", {}).items():
        vals[f"split/{name}/d0"] = f2(e["delta_firm0"])
        vals[f"split/{name}/d1"] = f2(e["delta_firm1"])
        vals[f"split/{name}/higher1"] = str(e["firm1_higher_profit"])
        vals[f"split/{name}/n"] = str(e["n"])
    for name, e in A.get("advisor_adoption", {}).items():
        vals[f"adopt/{name}/copy"] = pct(e["share_equal_to_reported_br"])
        vals[f"adopt/{name}/r1nash"] = pct(e["share_round1_at_nash"])
    r = A.get("reasoning", {})
    if r:
        vals["think/replies"] = f"{r['replies']:,}".replace(",", "{,}")
        vals["think/empty"] = str(r["empty"])
        vals["think/atcap"] = str(r["empty_at_cap"])
    fm = A.get("final_price_modes")
    if fm:
        vals["modes/total"] = f"{fm['firm_episodes']:,}".replace(",", "{,}")
        for i, (price, n) in enumerate(fm["top"][:5]):
            vals[f"modes/{i + 1}/price"] = f"{price:.2f}"
            vals[f"modes/{i + 1}/n"] = str(n)
    for m, e in A.get("own_gain", {}).items():
        for f in ("a_base", "a_advised", "diff", "b_base", "b_rival", "avg_base", "diff_vs_avg"):
            vals[f"own/{m}/{f}"] = f2(e[f])
        vals[f"own/{m}/ci_vs_avg"] = ci(e["ci_vs_avg"])
        vals[f"own/{m}/p_vs_avg"] = pv(e["p_vs_avg"])
        vals[f"own/{m}/ci"] = ci(e["ci"])
        vals[f"own/{m}/p"] = pv(e["p"])
    vals["min_delta"] = f2(A["min_delta"])
    ref = A["reference"]
    vals["q/delta"] = f"{ref['qlearning_delta_mean']:.3f}"
    vals["q/ci"] = "[" + ", ".join(f"{x:.2f}" for x in ref["qlearning_delta_ci"]) + "]"
    vals["q/n"] = str(ref["qlearning_sessions"])
    for name, m in A["mechanisms"].items():
        corr = m["opening_final_corr"]
        vals[f"mech/{name}/corr"] = "--" if corr is None else f2(corr)
        vals[f"mech/{name}/lastmove"] = f"{m['last_price_change_round_median']:.0f}"
        for k, v in m["notes_keyword_share"].items():
            vals[f"mech/{name}/{k}"] = pct(v)
    for name, m in A.get("message_prices", {}).items():
        vals[f"msg/{name}/withprice"] = pct(m["share_with_price"])
        vals[f"msg/{name}/own"] = pct(m["share_own_price"])
        vals[f"msg/{name}/announced"] = f2(m["own_price_index_median"])
        vals[f"msg/{name}/above"] = pct(m["own_above_monopoly_share"])
        vals[f"msg/{name}/n"] = f"{m['messages']:,}".replace(",", "{,}")

    # p-values for running text carry their relation: \V{.../pe} -> "= 0.016" or "< 0.001"
    for k in list(vals):
        if k.rsplit("/", 1)[-1] in {"p", "pb", "holm", "p_price", "p_delta"}:
            v = vals[k]
            vals[k + "e"] = "<0.001" if "<" in v else "=" + v
    lines = ["% generated by scripts/oligopoly/report_tables.py -- do not edit",
             "\\makeatletter",
             "\\newcommand{\\V}[1]{\\@ifundefined{v@#1}{\\textbf{??#1??}}{\\csname v@#1\\endcsname}}"]
    for k, v in sorted(vals.items()):
        lines.append(f"\\expandafter\\def\\csname v@{k}\\endcsname{{{v}}}")
    lines.append("\\makeatother")
    (OUT / "values.tex").write_text("\n".join(lines) + "\n")

    # Full cell table (appendix)
    order = sorted(cells, key=lambda n: (cells[n]["n"], n))
    rows = []
    for n in order:
        c = cells[n]
        rows.append(
            f"{n} & {c['episodes']} & {c['n']} & {f2(c['delta_mean'])} {ci(c['delta_ci'])} & "
            f"{f2(c['price_index_mean'])} {ci(c['price_index_ci'])} & {pct(c['share_above_monopoly_price'])} & "
            f"{f2(c['cs_loss_vs_nash'])} & {f2(c['welfare_vs_nash'])} & {100 * c['invalid_rate']:.1f} \\\\"
        )
    (OUT / "tab-all-cells.tex").write_text(
        "\\begin{longtable}{@{}lrrllrrrr@{}}\n"
        "\\caption{All conditions of the final study. $\\cindex$: collusion index; $\\pindex$: price index; "
        "$>\\pM$: share of episodes ending above the monopoly price; CS and W: consumer-surplus and "
        "total-welfare loss relative to Nash (1 = monopoly loss); Inv.: invalid-reply rate (\\%).}"
        "\\label{tab:allcells}\\\\\n\\toprule\n"
        "Condition & Ep. & $n$ & $\\cindex$ [95\\% CI] & $\\pindex$ [95\\% CI] & $>\\pM$ & CS & W & Inv. \\\\\n\\midrule\n\\endfirsthead\n"
        "\\toprule\nCondition & Ep. & $n$ & $\\cindex$ [95\\% CI] & $\\pindex$ [95\\% CI] & $>\\pM$ & CS & W & Inv. \\\\\n\\midrule\n\\endhead\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n")

    # Pre-specified family H2-H6
    hrows = []
    for k, t in A["tests"].items():
        if not k.startswith("H"):
            continue
        hyp, rest = k.split(" ", 1)
        model = "ds-flash" if "ds-flash" in rest else "db-lite"
        desc = {"H2": "cheap talk raises $\\cindex$", "H3": "BR advisor lowers $\\pindex$",
                "H4": "full advisor raises $\\cindex$", "H5a": "aggressive lowers $\\cindex$",
                "H5b": "cooperative raises $\\cindex$", "H6": "four firms lower $\\cindex$"}[hyp]
        verdict = "supported" if t["p_holm"] < 0.05 else ("opposite sign" if (t["diff"] > 0) != (t["alternative"] == "greater") and t["ci"][0] * t["ci"][1] > 0 else "not supported")
        hrows.append(f"{hyp} & {desc} & {LABEL[model]} & {f2(t['diff'])} {ci(t['ci'])} & {pv(t['p'])} & {pv(t['p_holm'])} & {verdict} \\\\")
    (OUT / "tab-family.tex").write_text(
        "\\begin{tabular}{@{}lllrrrl@{}}\n\\toprule\nHyp. & Prediction & Model & Difference [95\\% CI] & $p$ & Holm $p$ & Verdict \\\\\n\\midrule\n"
        + "\n".join(hrows) + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"{len(vals)} values, {len(rows)} cells, {len(hrows)} family tests")


if __name__ == "__main__":
    main()
