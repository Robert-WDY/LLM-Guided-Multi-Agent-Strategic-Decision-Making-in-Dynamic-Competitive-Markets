"""Audit when realized losses accumulate after the recommendations actually selected."""

import argparse
import numpy as np

from cause_trials import ARMS
from renewal_trials import OUT
from run import read, save


def trace(record, name, terminal_gain):
    t, j = record["probe"], record["masks"]["1"][0]
    a, b = (np.asarray(record["paths"][k]["rows"]) for k in (name, "keep"))
    delta = (a[t - 1 :, j, 2] - b[t - 1 :, j, 2]) * 0.99 ** np.arange(61 - t)
    parts = [float(s.sum()) for s in (delta[:4], delta[4:12], delta[12:])]
    assert np.isclose(sum(parts), terminal_gain)
    return parts


def summarize(rows):
    negative = [p for p in rows if sum(p) < -1e-7]
    return dict(
        effective=len(rows),
        losing=len(negative),
        up_at_four_then_lost=sum(p[0] > 1e-7 for p in negative),
        up_at_twelve_then_lost=sum(sum(p[:2]) > 1e-7 for p in negative),
        most_loss_after_twelve=sum(-p[2] > -sum(p) / 2 for p in negative),
        mean_losing_parts=np.mean(negative, axis=0).tolist() if negative else None,
    )


def measure(phase):
    result = {}
    for m in ("econ", "native") if phase == "primary" else ("econ",):
        detail = read(
            OUT
            / (
                f"decisions-{m}.json.gz"
                if phase == "primary"
                else "precision-decisions.json.gz"
            )
        )
        rows = []
        for d in detail:
            key = "renewal" if phase == "primary" else "review"
            k = d["actions"][key]
            if not d["changed"][k]:
                continue
            name = ARMS[k] if phase == "primary" else "public_probe"
            archive = (
                OUT
                / ("confirm" if phase == "primary" else "precision")
                / f"{d['seed']}.json.gz"
            )
            rows.append(trace(read(archive), name, d["gains"][k]))
        result[m] = summarize(rows)
    save(OUT / f"timing-{phase}.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("primary", "precision"))
    measure(parser.parse_args().phase)
