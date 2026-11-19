"""Show terminal outcomes from identical openings across four future draws."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from distribution_trials import OUT
from prices import ROOT
from run import read


if __name__ == "__main__":
    results = read(OUT / "results.json")["within_market"]
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.72, bottom=0.22, wspace=0.47)
    fig.text(
        0.09,
        0.92,
        "开局和第一次改价相同，后续四次抽样盈亏不同",
        fontsize=17,
        weight="bold",
        color="#192c3a",
    )
    fig.text(
        0.09, 0.85, "相对保留草稿、折现至第 60 轮的利润差", fontsize=10, color="#4f606d"
    )
    for ax, market, action, label, unit in (
        (axes[0], "econ", "cut", "零售降价 4%", "利润单位"),
        (axes[1], "native", "raise", "生鲜涨价 4%", "万模拟元"),
    ):
        d = results[market][action]["widest_flipping_example"]
        if d is None:
            ax.text(0.5, 0.5, "没有观察到盈亏翻转", ha="center", transform=ax.transAxes)
            continue
        values = np.asarray(d["outcomes"]) / (1e4 if market == "native" else 1)
        ax.bar(
            np.arange(4),
            values,
            width=0.53,
            color=np.where(values > 0, "#2d7186", "#c15030"),
            zorder=3,
        )
        ax.axhline(0, color="#4f606d", lw=1)
        ax.set_title(f"{label} · 种子 {d['seed']}", loc="left", color="#192c3a", pad=14)
        ax.set_ylabel(unit)
        ax.set_xticks(range(4), ["原抽样", "重抽1", "重抽2", "重抽3"])
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#b6c0c6")
        ax.grid(axis="y", color="#e1e7eb", lw=1, zorder=0)
        ax.tick_params(length=0, pad=5)
        ax.margins(y=0.24)
        for x, y in enumerate(values):
            offset = 3 if y > 0 else -3
            ax.annotate(
                f"{y:+.1f}",
                (x, y),
                (0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if y > 0 else "top",
            )
    fig.text(
        0.09,
        0.08,
        "每组展示观察到的盈亏翻转中范围最大的一个开局。仅重抽本店的后续经营选择。",
        fontsize=9,
        color="#4f606d",
    )
    fig.savefig(ROOT / "docs" / "distribution-examples.png", dpi=160)
