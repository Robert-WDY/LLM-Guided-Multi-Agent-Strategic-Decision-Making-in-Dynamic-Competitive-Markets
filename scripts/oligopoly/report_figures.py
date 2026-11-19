"""Figures for the final report (PDF, print, light background).

python scripts/oligopoly/report_figures.py -> report/final/figures/*.pdf
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import arr, boot_ci, episode_rows, load  # noqa: E402

OUT = Path(os.environ.get("REPORT_DIR", ROOT / "report" / "final")) / "figures"
A = json.loads((ROOT / "experiment-results" / "oligopoly-v1" / "analysis.json").read_text())

# Validated categorical slots (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
})

GROUPS = [
    ("Base, five models", ["db-mini-base", "db-lite-base", "db-pro-base", "ds-flash-base", "ds-pro-base"]),
    ("Treatments, DeepSeek Flash", ["ds-flash-advisor-br", "ds-flash-advisor-full", "ds-flash-aggressive",
                                    "ds-flash-cooperative", "ds-flash-comm"]),
    ("Treatments, Doubao lite", ["db-lite-advisor-br", "db-lite-advisor-full", "db-lite-aggressive",
                                 "db-lite-cooperative", "db-lite-comm"]),
    ("Opening, horizon, adoption", ["ds-flash-open-nash", "ds-pro-open-nash", "db-lite-open-nash",
                                    "ds-flash-open-nash-long", "ds-flash-advisor-br-firm0",
                                    "db-lite-advisor-br-firm0"]),
    ("Structure and robustness", ["ds-flash-n4-base", "db-lite-n4-base", "ds-flash-vs-br",
                                  "ds-flash-vs-db-lite", "ds-flash-think-base", "ds-flash-base-alt",
                                  "db-lite-base-alt"]),
]
NAMES = {
    "db-mini-base": "Doubao mini", "db-lite-base": "Doubao lite", "db-pro-base": "Doubao pro",
    "ds-flash-base": "DeepSeek Flash", "ds-pro-base": "DeepSeek Pro",
    "advisor-br": "best-response advisor", "advisor-full": "full advisor", "aggressive": "aggressive persona",
    "cooperative": "cooperative persona", "comm": "cheap talk",
    "ds-flash-open-nash": "DS Flash, Nash opening", "ds-pro-open-nash": "DS Pro, Nash opening",
    "db-lite-open-nash": "Db lite, Nash opening", "ds-flash-open-nash-long": "DS Flash, Nash opening, 150 rds",
    "ds-flash-advisor-br-firm0": "DS Flash, advisor for one firm", "db-lite-advisor-br-firm0": "Db lite, advisor for one firm",
    "ds-flash-n4-base": "DS Flash, four firms", "db-lite-n4-base": "Db lite, four firms",
    "ds-flash-vs-br": "DS Flash vs best responder", "ds-flash-vs-db-lite": "DS Flash vs Db lite",
    "ds-flash-think-base": "DS Flash, reasoning mode", "ds-flash-base-alt": "DS Flash, paraphrased prompt",
    "db-lite-base-alt": "Db lite, paraphrased prompt",
}
RAISING = ("comm", "cooperative")
LOWERING = ("advisor-br", "aggressive")


PREVIEW = Path("/home/donxu/ddy/scratch") if "--preview" in sys.argv else None


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    if PREVIEW is not None:
        fig.savefig(PREVIEW / f"{stem}.png", dpi=150)


def label(name: str) -> str:
    if name in NAMES:
        return NAMES[name]
    for k in ("advisor-full", "advisor-br", "aggressive", "cooperative", "comm"):
        if name.endswith(k):
            return NAMES[k]
    return name


def color(name: str) -> str:
    if any(name.endswith(k) for k in RAISING):
        return ORANGE
    if any(name.endswith(k) for k in LOWERING):
        return AQUA
    return BLUE


def fig_delta(cells) -> None:
    rows = []
    for title, names in GROUPS:
        rows.append(("__title__", title))
        rows += [("cell", n) for n in names if n in cells]
    fig, ax = plt.subplots(figsize=(6.3, 0.19 * len(rows) + 0.7))
    rng = np.random.default_rng(1)
    y = 0
    ticks, ticklabels = [], []
    for kind, val in rows:
        if kind == "__title__":
            ax.text(-1.48, y, val, fontsize=8, fontweight="bold", color=INK, va="center")
        else:
            d = arr(cells, val)
            ax.scatter(np.clip(d, -1.45, 1.25), y + rng.uniform(-0.18, 0.18, len(d)), s=5, color=color(val),
                       alpha=0.35, linewidths=0)
            lo, hi = boot_ci(d)
            ax.plot([lo, hi], [y, y], color=INK, lw=1.4, solid_capstyle="round")
            ax.plot(d.mean(), y, "o", color=INK, ms=3.6)
            ticks.append(y)
            ticklabels.append(label(val))
        y -= 1
    for x in (0, 1):
        ax.axvline(x, color=INK2, lw=0.6, ls=(0, (3, 3)))
    q = A["reference"]["qlearning_delta_mean"]
    ax.axvline(q, color=INK2, lw=0.9)
    ax.text(q - 0.02, 0.6, "Q-learning", fontsize=7, color=INK2, ha="right")
    ax.text(0.02, 0.6, "Nash", fontsize=7, color=INK2)
    ax.text(1.02, 0.6, "monopoly", fontsize=7, color=INK2)
    ax.set_yticks(ticks, ticklabels, fontsize=7.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1.5, 1.3)
    ax.set_ylim(y + 0.3, 1.1)
    ax.set_xlabel("collusion index $\\Delta$ (0 = Nash, 1 = joint monopoly)")
    handles = [plt.Line2D([], [], ls="", marker="o", color=c, alpha=0.7, ms=4) for c in (BLUE, AQUA, ORANGE)]
    ax.legend(handles, ["base / structure", "advisor or aggressive persona", "cheap talk or cooperative persona"],
              loc="upper center", bbox_to_anchor=(0.45, -0.07 * 30 / len(rows)), ncol=3, fontsize=6.8,
              handletextpad=0.2)
    fig.tight_layout()
    save(fig, "delta-by-condition")
    plt.close(fig)


def fig_paths(cells) -> None:
    panels = [("ds-flash-base", "Free opening, 50 rounds"), ("ds-flash-open-nash", "Nash opening, 50 rounds"),
              ("ds-flash-open-nash-long", "Nash opening, 150 rounds")]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.3), sharey=True, gridspec_kw={"width_ratios": [1, 1, 2]})
    for ax, (name, title) in zip(axes, panels):
        paths = []
        for s in cells.get(name, []):
            b = s["benchmarks"]
            p = (np.array(s["price_path_mean"]) * s["scale"] - b["nash_price"]) / (b["monopoly_price"] - b["nash_price"])
            paths.append(p)
            ax.plot(range(1, len(p) + 1), p, color=BLUE, lw=0.6, alpha=0.28)
        if paths:
            ax.plot(range(1, len(paths[0]) + 1), np.mean(paths, axis=0), color=INK, lw=1.6)
        for v in (0, 1):
            ax.axhline(v, color=INK2, lw=0.6, ls=(0, (3, 3)))
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("round")
        ax.grid(axis="y", color=GRID, lw=0.4)
    axes[0].set_ylabel("price index $\\rho$")
    axes[0].set_ylim(-0.8, 3.0)
    axes[2].text(150, 1.06, "monopoly", fontsize=6.5, color=INK2, ha="right")
    axes[2].text(150, -0.22, "Nash", fontsize=6.5, color=INK2, ha="right")
    fig.tight_layout()
    save(fig, "price-paths")
    plt.close(fig)


def fig_welfare() -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    show = {  # name -> (label, dx, dy) in points
        "db-lite-comm": ("Db lite, cheap talk", -4, 4), "ds-flash-cooperative": ("DS Flash, cooperative", -4, 4),
        "db-lite-cooperative": ("Db lite, cooperative", -4, -10), "db-lite-n4-comm": ("Db lite, 4 firms, talk", -4, 4),
        "ds-flash-comm": ("DS Flash, cheap talk", 6, 0), "db-pro-base": ("Db pro, base", 6, -4),
        "ds-flash-n4-comm": ("DS Flash, 4 firms, talk", -6, 2), "advisor-label": None,
        "ds-flash-advisor-br": ("DS Flash, BR advisor", 8, -11),
    }
    for c in A["cells"]:
        name = c["cell"]
        x, y = c["price_index_mean"], c["welfare_vs_nash"]
        ax.scatter(x, y, s=22, color=color(name), edgecolors="white", linewidths=0.8, zorder=3)
        if show.get(name):
            text, dx, dy = show[name]
            ax.annotate(text, (x, y), fontsize=6, color=INK2, xytext=(dx, dy), textcoords="offset points",
                        ha="right" if dx < 0 else "left")
    for v in (0, 1):
        ax.axhline(v, color=INK2, lw=0.6, ls=(0, (3, 3)))
        ax.axvline(v, color=INK2, lw=0.6, ls=(0, (3, 3)))
    ax.set_xlabel("price index $\\rho$ (1 = monopoly price)")
    ax.set_ylabel("total-welfare loss (1 = monopoly loss)")
    ax.grid(color=GRID, lw=0.4)
    handles = [plt.Line2D([], [], ls="", marker="o", color=c, ms=5) for c in (BLUE, AQUA, ORANGE)]
    ax.legend(handles, ["base / structure", "advisor or aggressive", "cheap talk or cooperative"],
              loc="upper left", fontsize=6.8)
    fig.tight_layout()
    save(fig, "welfare")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load()
    fig_delta(cells)
    fig_paths(cells)
    fig_welfare()
    print(sorted(p.name for p in OUT.glob("*.pdf")))
