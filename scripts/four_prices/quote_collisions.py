"""Explain settled price ties in the direct-number study using original model replies."""

import itertools
from open_price import OUTPUT
from run import read, save


def analyze(rows):
    pairs = ties = diff_raw = diff_submitted = 0
    for r in rows:
        b = r["paths"]["keep"]
        for d in b["decisions"]:
            t = d["t"]
            if t < 19:
                continue
            for n in range(t - 1, min(t + 3, 60)):
                settled = b["rows"][n]
                for i, j in itertools.combinations(range(4), 2):
                    if not settled[i][4] or not settled[j][4]:
                        continue
                    pairs += 1
                    if settled[i][0] != settled[j][0]:
                        continue
                    ties += 1
                    raw = [float(r["tape"][d["model_actions"][k]]["raw"]) for k in (i, j)]
                    diff_raw += int(raw[0] != raw[1])
                    diff_submitted += int(d["requested"][i] != d["requested"][j])
    return dict(
        active_pair_rounds=pairs,
        settled_ties=ties,
        different_raw_quotes_among_ties=diff_raw,
        different_submitted_quotes_among_ties=diff_submitted,
    )


if __name__ == "__main__":
    out = {
        m: analyze([read(f) for f in sorted(OUTPUT.glob(prefix + "*.json.gz"))])
        for m, prefix in (("econ", "183"), ("native", "184"))
    }
    save(OUTPUT / "collisions.json", out)
