"""Count up/down decisions and decompose every loss, alongside selected extreme trajectories."""

import numpy as np
from prices import MARKETS, GROUPS, OUT
from run import read, save, REVIEW
from saturation_report import discounted


def audit(m, rows, k):
    n_up = n_down = n_keep = 0
    cases = []
    whole = []
    for r in rows:
        path = r["paths"][f"inf{k}"]
        a = np.asarray(path["rows"])
        base = np.asarray(r["paths"]["keep"]["rows"])
        mask = r["masks"][str(k)]
        gain = discounted(a[:, :, 2] - base[:, :, 2])
        whole.extend(gain[mask].tolist())
        for d in path["decisions"]:
            if d["t"] not in REVIEW:
                continue
            for seat in mask:
                t = d["t"]
                sel = d["selected"][seat]
                if not a[t - 1, seat, 4] or a[t - 1, seat, 0] == d["original"][seat]:
                    n_keep += 1
                elif sel == 1:
                    n_down += 1
                elif sel == 2:
                    n_up += 1
                else:
                    raise ValueError("posted price without a selected change")
        seat = r["focal"][str(k)]
        t = r["probe"]
        b = np.asarray(r["paths"][f"skip{k}"]["rows"])
        if not np.any(a[t - 1 : t + 3, seat, 0] != b[t - 1 : t + 3, seat, 0]):
            continue
        selected = next(d["selected"][seat] for d in path["decisions"] if d["t"] == t)
        w = 0.99 ** np.arange(61 - t)
        delta = a[t - 1 :, seat, 2] - b[t - 1 :, seat, 2]
        volume = float(
            w
            @ ((a[t - 1 :, seat, 1] - b[t - 1 :, seat, 1]) * b[t - 1 :, seat, 0])
            / (100 if m == "native" else (1.0, 3.2, 10.0)[r["seed"] % 3])
        )
        cases.append((selected, float(w @ delta), float(w[:4] @ delta[:4]), volume))
    losses = [z for z in cases if z[1] < -1.0e-7]
    totals = sorted([-y for y in whole if y < -1.0e-7], reverse=True)
    return dict(
        renewals=dict(up=n_up, down=n_down, other=n_keep),
        by_direction={
            str(a): dict(
                changed=sum(z[0] == a for z in cases),
                losses=sum(z[0] == a and z[1] < -1.0e-7 for z in cases),
                mean=(
                    np.mean([z[1] for z in cases if z[0] == a])
                    if any(z[0] == a for z in cases)
                    else None
                ),
            )
            for a in (1, 2)
        },
        losing_probes=len(losses),
        fewer_sales_value=sum(z[3] < 0 for z in losses),
        short_positive_but_terminal_loss=sum(z[2] > 0 for z in losses),
        worst_five_pct_of_firms_share_of_total_loss=(
            sum(totals[: max(1, int(np.ceil(0.05 * len(whole))))]) / sum(totals) if totals else None
        ),
    )


if __name__ == "__main__":
    report = {}
    for m in MARKETS:
        rows = [r for g in GROUPS for r in read(OUT / f"{m}-{g}.json.gz")]
        report[m] = {k: audit(m, rows, k) for k in (1, 2, 4)}
    save(OUT / "loss-audit.json", report)
