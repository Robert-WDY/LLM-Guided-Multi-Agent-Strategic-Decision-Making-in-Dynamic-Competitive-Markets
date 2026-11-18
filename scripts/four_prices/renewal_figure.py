"""Plot the worst executed public recommendation in the primary fresh-food cohort."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from renewal_trials import OUT
from run import read


def render():
    tail = read(OUT / "diagnostics.json")["native"]["tails"]["worst_accepted"]
    r = read(OUT / "confirm" / f"{tail['seed']}.json.gz")
    t, j = r["probe"], r["masks"]["1"][0]
    start = t - 1
    base, arm = (
        np.asarray(r["paths"][k]["rows"])
        for k in ("keep", ("audit_cut", "audit_raise")[tail["arm"] - 1])
    )
    time = np.arange(t, 61)
    curve = (
        np.cumsum(
            (arm[start:, j, 2] - base[start:, j, 2]) * 0.99 ** np.arange(len(time))
        )
        / 10000
    )
    assert np.isclose(curve[-1] * 10000, tail["total"])
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "font.size": 10,
        }
    )
    fig, (top, bot) = plt.subplots(2, 1, figsize=(8.3, 5.5), sharex=True)
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.12, right=0.95, hspace=0.37)
    direction = ("降价", "涨价")[tail["arm"] - 1]
    fig.text(
        0.12,
        0.93,
        f"一次{direction}后，亏损从第 {t + 12} 轮起拉大",
        fontsize=17,
        weight="bold",
        color="#192c3a",
    )
    fig.text(
        0.12,
        0.875,
        f"生鲜种子 {r['seed']} · 120 个新市场中按规则改价后亏损最大的一笔",
        color="#4f606d",
    )
    colors = ["#2d5e81", "#c24f2e"]
    for frame, label, color in zip(
        (base, arm), ("保留店长原报价", f"第 {t} 轮按建议{direction}"), colors
    ):
        price = np.where(frame[start:, j, 4] > 0, frame[start:, j, 0] / 100, np.nan)
        top.step(time, price, where="post", lw=2.2, label=label, color=color)
    top.set_ylabel("成交价（模拟元）")
    top.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    bot.plot([t - 1, *time], [0, *curve], color=colors[1], lw=2.3)
    bot.axhline(0, color="#52616b", lw=1)
    bot.scatter([t + 11, 60], [curve[11], curve[-1]], color=colors[1], s=23, zorder=4)
    bot.annotate(
        f"{curve[11]:.2f} 万",
        (t + 11, curve[11]),
        (15, -20),
        textcoords="offset points",
    )
    bot.annotate(
        f"{curve[-1]:.2f} 万",
        (60, curve[-1]),
        (-3, 19),
        ha="right",
        textcoords="offset points",
        weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
    )
    bot.set_ylabel("累计利润差（万模拟元）")
    bot.set_xlabel("经营轮次")
    bot.set_xlim(t - 1, 60.5)
    bot.set_xticks([*range(t, 58, 4), 60])
    for axis in (top, bot):
        axis.axvspan(t - 0.5, t + 3.5, color="#e9eff3", alpha=0.55, zorder=-2)
        axis.axvline(t + 11.5, color="#7b858b", ls="--", lw=1, zorder=-1)
        axis.grid(axis="y", color="#e0e5e8", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#b0bac0")
        axis.tick_params(length=0, pad=5)
    fig.text(
        0.12,
        0.047,
        "阴影为首次合同的四轮。虚线后，两次续约已结束。利润差按每轮 0.99 折现。",
        fontsize=9,
        color="#52616b",
    )
    dest = Path(__file__).resolve().parents[2] / "docs" / "renewal-tail.png"
    fig.savefig(dest, dpi=160, metadata={"Software": "game-theory-agent"})
    plt.close(fig)


if __name__ == "__main__":
    render()
