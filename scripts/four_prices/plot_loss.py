"""Plot the most costly audited single contract in the native four-install condition."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from prices import OUT
from run import read


def main():
    v = read(OUT / "tail.json")["native-single-switch-4"][0]
    rows = read(OUT / f"native-{v['group']}.json.gz")
    r = next(d for d in rows if d["seed"] == v["seed"])
    zh = "Microsoft YaHei" in {f.name for f in font_manager.fontManager.ttflist}
    if zh:
        plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 6.8), sharex=True)
    start = v["start"]
    t = np.arange(start, 61)
    seat = v["seat"]
    for name, cn, en, color in (
        ("inf4", "执行这次建议", "Apply advice", "#a94336"),
        ("skip4", "跳过这一次", "Skip once", "#266991"),
    ):
        a = np.asarray(r["paths"][name]["rows"])[start - 1 :, seat]
        for ax, y in zip(
            axes, (a[:, 0] / 100, a[:, 1], np.cumsum(a[:, 2] * 0.99 ** np.arange(len(a))) / 10000)
        ):
            ax.step(t, y, where="mid", color=color, lw=2, label=cn if zh else en)
    for ax, cn, en in zip(
        axes,
        ("报价（元）", "销量（单）", "累计折现利润（万元）"),
        ("Price (yuan)", "Orders filled", "Discounted profit (10,000 yuan)"),
    ):
        ax.set_ylabel(cn if zh else en)
        ax.grid(axis="y", alpha=0.18)
        ax.axvspan(start - 0.4, start + 3.5, color="#e6cc7b", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", ncols=2, frameon=False)
    axes[-1].set_xlabel("经营轮次" if zh else "Market round")
    fig.suptitle(
        "只改一次价，后来的销量和利润怎样分开"
        if zh
        else "One early price revision changes subsequent sales and profits"
    )
    fig.tight_layout()
    fig.savefig(OUT / "worst-contract.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    main()
