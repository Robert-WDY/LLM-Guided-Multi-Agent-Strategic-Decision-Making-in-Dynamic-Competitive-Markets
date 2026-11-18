"""Publication figure for the paired public-price replication."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from guard_study import OUTPUT
from run import read


def main():
    for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
        if name in font_manager.get_font_names():
            plt.rcParams["font.family"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    data = read(OUTPUT / "confirmation.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, m, unit, scale in zip(
        axes,
        ("native", "econ"),
        ("生鲜 · 万模拟元", "零售基准 · 模拟利润单位"),
        (10000, 1),
    ):
        rows = data["pooled-" + m]["install"]
        for method, label, color, offset in (
            ("inf", "旧建议", "#767676", 0.19),
            ("safe_inf", "筛选后的建议", "#176197", 0),
            ("safe_dir", "相同筛选＋简单估值", "#a65817", -0.19),
        ):
            triples = [rows[str(k)][method]["effect"] for k in (1, 2, 4)]
            means = [t[0] / scale for t in triples]
            ax.errorbar(
                means,
                [i + offset for i in (2, 1, 0)],
                xerr=[
                    [max(0, (t[0] - t[1]) / scale) for t in triples],
                    [max(0, (t[2] - t[0]) / scale) for t in triples],
                ],
                fmt="o",
                capsize=2,
                color=color,
                label=label,
            )
        ax.axvline(0, color="#444444", lw=1, ls="--")
        ax.set_yticks([2, 1, 0], ["装 1 家", "装 2 家", "装 4 家"])
        ax.set_ylim(-0.5, 2.65)
        ax.set_title(unit)
        ax.grid(axis="x", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("相对同店保留经营草稿的终局利润差")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle(
        "每家安装企业能多赚多少？\n每个主市场 108 个独立种子，点为均值，线为配对自举区间",
        fontsize=13,
    )
    fig.savefig(OUTPUT / "profit.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    main()
