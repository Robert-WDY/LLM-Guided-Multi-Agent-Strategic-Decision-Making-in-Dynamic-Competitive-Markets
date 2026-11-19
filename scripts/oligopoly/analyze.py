"""Analyse oligopoly-v1 summaries: cell table, pre-specified tests, figures.

python scripts/oligopoly/analyze.py   -> experiment-results/oligopoly-v1/analysis.json
                                         experiment-results/oligopoly-v1/tables.md
                                         docs/figures/oligopoly-*.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiment-results" / "oligopoly-v1"
FIG = ROOT / "docs" / "figures"
PHASES = ("e1", "e2", "e3", "e4")
RNG = np.random.default_rng(20261119)
B = 10_000


def load() -> dict[str, list[dict]]:
    cells: dict[str, list[dict]] = defaultdict(list)
    for ph in PHASES:
        p = OUT / ph / "summaries.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    s = json.loads(line)
                    cells[s["cell"]].append(s)
    return cells


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def boot_diff_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mx = x[RNG.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    my = y[RNG.integers(0, len(y), size=(B, len(y)))].mean(axis=1)
    d = mx - my
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def perm_p(x: np.ndarray, y: np.ndarray, alternative: str) -> float:
    """Permutation p-value for mean(x) - mean(y); alternative greater/less/two."""

    obs = x.mean() - y.mean()
    pool = np.concatenate([x, y])
    stats = np.empty(B)
    for b in range(B):
        RNG.shuffle(pool)
        stats[b] = pool[: len(x)].mean() - pool[len(x) :].mean()
    if alternative == "greater":
        return float((np.sum(stats >= obs) + 1) / (B + 1))
    if alternative == "less":
        return float((np.sum(stats <= obs) + 1) / (B + 1))
    return float((np.sum(np.abs(stats) >= abs(obs)) + 1) / (B + 1))


def sign_flip_p(x: np.ndarray) -> float:
    """One-sided H0: mean <= 0, sign-flip randomisation test."""

    obs = x.mean()
    flips = RNG.choice([-1.0, 1.0], size=(B, len(x)))
    stats = (flips * np.abs(x)).mean(axis=1)
    return float((np.sum(stats >= obs) + 1) / (B + 1))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - k) * p))
        adj[name] = running
    return adj


def arr(cells, name, key="delta") -> np.ndarray:
    return np.array([s[key] for s in cells.get(name, [])], dtype=float)


def main() -> None:
    cells = load()
    table = []
    for name, rows in sorted(cells.items()):
        d = arr(cells, name)
        pi = arr(cells, name, "price_index")
        table.append(
            {
                "cell": name,
                "episodes": len(rows),
                "n": rows[0]["n"],
                "delta_mean": float(d.mean()),
                "delta_ci": boot_ci(d),
                "delta_median": float(np.median(d)),
                "price_index_mean": float(pi.mean()),
                "price_index_ci": boot_ci(pi),
                "share_above_monopoly_price": float(np.mean(pi > 1)),
                "welfare_vs_nash": float(arr(cells, name, "welfare_vs_nash").mean()),
                "cs_loss_vs_nash": float(arr(cells, name, "cs_loss_vs_nash").mean()),
                "invalid_rate": float(arr(cells, name, "invalid_rate").mean()),
                "tokens": int(sum(r["input_tokens"] + r["output_tokens"] for r in rows)),
            }
        )

    tests: dict[str, dict] = {}

    def two(name, a, b, alt, key="delta"):
        x, y = arr(cells, a, key), arr(cells, b, key)
        if len(x) and len(y):
            tests[name] = {
                "contrast": f"{a} - {b}", "metric": key, "alternative": alt,
                "diff": float(x.mean() - y.mean()), "ci": boot_diff_ci(x, y),
                "p": perm_p(x.copy(), y.copy(), alt), "n": [len(x), len(y)],
            }

    h1 = {}
    for m in ("db-mini", "db-lite", "db-pro", "ds-flash", "ds-pro"):
        x = arr(cells, f"{m}-base")
        if len(x):
            p = sign_flip_p(x)
            h1[m] = {"mean": float(x.mean()), "ci": boot_ci(x), "p": p,
                     "p_bonferroni": min(1.0, 5 * p), "n": len(x)}
    for m in ("ds-flash", "db-lite"):
        two(f"H2 comm>{m}", f"{m}-comm", f"{m}-base", "greater")
        two(f"H3 advisor-br<{m}", f"{m}-advisor-br", f"{m}-base", "less", "price_index")
        two(f"H4 advisor-full>{m}", f"{m}-advisor-full", f"{m}-base", "greater")
        two(f"H5a aggressive<{m}", f"{m}-aggressive", f"{m}-base", "less")
        two(f"H5b cooperative>{m}", f"{m}-cooperative", f"{m}-base", "greater")
        two(f"H6 n4<n2 {m}", f"{m}-n4-base", f"{m}-base", "less")
        two(f"X n4 comm-base {m}", f"{m}-n4-comm", f"{m}-n4-base", "two")
    two("X ds-flash vs BR rule", "ds-flash-vs-br", "ds-flash-base", "two")
    two("X heterogeneous vs ds-flash", "ds-flash-vs-db-lite", "ds-flash-base", "two")
    two("X heterogeneous vs db-lite", "ds-flash-vs-db-lite", "db-lite-base", "two")
    two("X ds-pro vs ds-flash", "ds-pro-base", "ds-flash-base", "two")
    two("X db-pro vs db-mini", "db-pro-base", "db-mini-base", "two")

    # E4 follow-up (PLAN addendum): from a forced Nash opening, do prices rise?
    for m in ("ds-flash", "db-lite", "ds-pro"):
        x = arr(cells, f"{m}-open-nash")
        if len(x):
            h1[f"{m} open-nash"] = {"mean": float(x.mean()), "ci": boot_ci(x), "p": sign_flip_p(x),
                                    "p_bonferroni": min(1.0, 3 * sign_flip_p(x)), "n": len(x)}
        xc = arr(cells, f"{m}-open-nash-comm")
        if len(xc):
            h1[f"{m} open-nash-comm"] = {"mean": float(xc.mean()), "ci": boot_ci(xc), "p": sign_flip_p(xc),
                                         "p_bonferroni": min(1.0, 3 * sign_flip_p(xc)), "n": len(xc)}
        two(f"E4 open-nash vs free opening {m}", f"{m}-open-nash", f"{m}-base", "two")
        two(f"E4 comm from Nash {m}", f"{m}-open-nash-comm", f"{m}-open-nash", "greater")
    two("E4 thinking vs no thinking (ds-flash)", "ds-flash-think-base", "ds-flash-base", "two")

    # Holm over pre-specified family H2-H6, pooling the two models by Fisher's
    # method is avoided: each model-specific test is a family member.
    family = {k: v["p"] for k, v in tests.items() if k.startswith("H")}
    for k, p in holm(family).items():
        tests[k]["p_holm"] = p

    q = [json.loads(l) for l in (OUT / "qlearning" / "sessions.jsonl").read_text().splitlines() if l.strip()]
    qd = np.array([r["delta"] for r in q])
    ref = {"qlearning_delta_mean": float(qd.mean()), "qlearning_delta_ci": boot_ci(qd),
           "qlearning_sessions": len(q)}

    result = {"cells": table, "h1": h1, "tests": tests, "reference": ref,
              "mechanisms": mechanisms(cells)}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=1))
    write_tables(result)
    figures(cells, ref)
    print((OUT / "tables.md").read_text())


KEYWORDS = {
    "collusion": ("collu", "tacit", "cartel"),
    "cooperation": ("cooperat", "coordinat", "mutual", "trust", "truce"),
    "price_war": ("price war", "price-war", "race to the bottom"),
    "retaliation": ("retaliat", "punish", "tit-for-tat", "tit for tat", "match any"),
    "undercut": ("undercut",),
    "equilibrium": ("equilibri", "nash"),
}


def episode_rows(s: dict) -> list[dict]:
    import gzip

    phase = s["episode_id"].split("-", 1)[0]
    path = OUT / phase / "episodes" / f"{s['episode_id']}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh][1:]


def mechanisms(cells) -> dict:
    out = {}
    for name, summaries in sorted(cells.items()):
        opening, final, lock, kw, msg_kw = [], [], [], defaultdict(list), defaultdict(list)
        for s in summaries:
            rows = episode_rows(s)
            b = s["benchmarks"]
            idx = lambda p: (p - b["nash_price"]) / (b["monopoly_price"] - b["nash_price"])  # noqa: E731
            opening.append(idx(np.mean(rows[0]["prices"])))
            final.append(s["price_index"])
            # lock-in: first round after which every firm's price stays within 1% of scale
            prices = np.array([r["prices"] for r in rows]) / s["scale"]
            moving = np.abs(np.diff(prices, axis=0)).max(axis=1) > 0.01
            last_move = int(np.nonzero(moving)[0].max()) + 2 if moving.any() else 1
            lock.append(last_move)
            notes = " ".join(" ".join(n or "" for n in r["notes"]) for r in rows).lower()
            msgs = " ".join(" ".join(m or "" for m in r["messages"]) for r in rows).lower()
            for k, words in KEYWORDS.items():
                kw[k].append(any(w in notes for w in words))
                if msgs.strip():
                    msg_kw[k].append(any(w in msgs for w in words))
        o, f = np.array(opening), np.array(final)
        out[name] = {
            "opening_final_corr": float(np.corrcoef(o, f)[0, 1]) if len(o) > 2 and o.std() > 0 and f.std() > 0 else None,
            "opening_price_index_mean": float(o.mean()),
            "last_price_change_round_median": float(np.median(lock)),
            "episodes_never_settling_share": float(np.mean(np.array(lock) > rows_len(summaries) - 5)),
            "notes_keyword_share": {k: float(np.mean(v)) for k, v in kw.items()},
            "message_keyword_share": {k: float(np.mean(v)) for k, v in msg_kw.items()} or None,
        }
    return out


def rows_len(summaries) -> int:
    return summaries[0]["rounds"]


def fmt_ci(ci) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def write_tables(r: dict) -> None:
    lines = ["| 格 | 局数 | 卖家 | Δ 均值 [95% CI] | Δ 中位数 | 价格指数 [95% CI] | 价格高于垄断价的比例 | 消费者剩余损失 | 总福利损失 | 无效回复率 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in r["cells"]:
        lines.append(
            f"| {c['cell']} | {c['episodes']} | {c['n']} | {c['delta_mean']:.2f} {fmt_ci(c['delta_ci'])} | "
            f"{c['delta_median']:.2f} | {c['price_index_mean']:.2f} {fmt_ci(c['price_index_ci'])} | "
            f"{c['share_above_monopoly_price']:.0%} | {c['cs_loss_vs_nash']:.2f} | {c['welfare_vs_nash']:.2f} | {c['invalid_rate']:.3f} |"
        )
    lines += ["", f"参照：Calvano Q-learning {r['reference']['qlearning_sessions']} 次会话 Δ = "
              f"{r['reference']['qlearning_delta_mean']:.3f} {fmt_ci(r['reference']['qlearning_delta_ci'])}。", ""]
    lines += ["| H1 模型 | 局数 | Δ 均值 [95% CI] | 单侧 p | Bonferroni p |", "|---|---|---|---|---|"]
    for m, h in r["h1"].items():
        lines.append(f"| {m} | {h['n']} | {h['mean']:.2f} {fmt_ci(h['ci'])} | {h['p']:.4f} | {h['p_bonferroni']:.4f} |")
    lines += ["", "| 检验 | 指标 | 方向 | 差值 [95% CI] | p | Holm p |", "|---|---|---|---|---|---|"]
    for k, t in r["tests"].items():
        lines.append(
            f"| {k} | {t['metric']} | {t['alternative']} | {t['diff']:+.2f} {fmt_ci(t['ci'])} | "
            f"{t['p']:.4f} | {t.get('p_holm', float('nan')):.4f} |"
        )
    lines += ["", "| 格 | 开局→终局价格指数相关 | 最后一次改价轮次（中位数） | 笔记提到合谋 | 提到合作/协调 | 提到价格战 | 提到报复 | 提到均衡/Nash | 消息提到合作/协调 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, m in r["mechanisms"].items():
        k = m["notes_keyword_share"]
        mk = (m["message_keyword_share"] or {}).get("cooperation")
        corr = "—" if m["opening_final_corr"] is None else f"{m['opening_final_corr']:.2f}"
        lines.append(
            f"| {name} | {corr} | {m['last_price_change_round_median']:.0f} | {k['collusion']:.0%} | "
            f"{k['cooperation']:.0%} | {k['price_war']:.0%} | {k['retaliation']:.0%} | {k['equilibrium']:.0%} | "
            f"{'—' if mk is None else f'{mk:.0%}'} |"
        )
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")


def figures(cells, ref) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    names = sorted(cells, key=lambda n: (cells[n][0]["n"], n))
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(names) + 1.2))
    for y, name in enumerate(names):
        d = arr(cells, name)
        ax.scatter(d, np.full(len(d), y) + RNG.uniform(-0.15, 0.15, len(d)), s=9, alpha=0.35, color="#4a6fa5")
        lo, hi = boot_ci(d)
        ax.plot([lo, hi], [y, y], color="#1d3557", lw=2)
        ax.plot(d.mean(), y, "o", color="#1d3557")
    ax.axvline(0, color="#888", lw=1, ls="--")
    ax.axvline(1, color="#888", lw=1, ls="--")
    ax.axvline(ref["qlearning_delta_mean"], color="#e76f51", lw=1.2, label="Q-learning (Calvano)")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Collusion index Δ (0 = Nash, 1 = joint monopoly)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-1.5, 1.3)
    fig.tight_layout()
    fig.savefig(FIG / "oligopoly-delta-by-cell.png", dpi=150)
    plt.close(fig)

    # welfare: price index vs total-welfare loss (both 0 at Nash, 1 at monopoly)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name in names:
        pi = arr(cells, name, "price_index").mean()
        wl = arr(cells, name, "welfare_vs_nash").mean()
        color = "#e76f51" if ("comm" in name or "cooperative" in name) else (
            "#2a9d8f" if "advisor-br" in name or "aggressive" in name else "#1d3557")
        ax.scatter(pi, wl, color=color, s=22)
        ax.annotate(name, (pi, wl), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.axhline(1, color="#888", ls="--", lw=1)
    ax.axvline(1, color="#888", ls="--", lw=1)
    ax.set_xlabel("price index (0 = Nash, 1 = monopoly price)")
    ax.set_ylabel("total-welfare loss (0 = Nash, 1 = monopoly loss)")
    fig.tight_layout()
    fig.savefig(FIG / "oligopoly-welfare.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, name in zip(axes, ("ds-flash-base", "db-lite-base")):
        for s in cells.get(name, []):
            b = s["benchmarks"]
            path = (np.array(s["price_path_mean"]) * s["scale"] - b["nash_price"]) / (
                b["monopoly_price"] - b["nash_price"])
            ax.plot(range(1, len(path) + 1), path, lw=0.8, alpha=0.6)
        ax.axhline(0, color="#888", ls="--", lw=1)
        ax.axhline(1, color="#888", ls="--", lw=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("round")
    axes[0].set_ylabel("price index (0 = Nash, 1 = monopoly)")
    axes[0].set_ylim(-1, 4)
    fig.tight_layout()
    fig.savefig(FIG / "oligopoly-price-paths.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
