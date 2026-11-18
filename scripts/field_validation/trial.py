"""Randomize market saturation and estimate net-margin ITT on a locked roster."""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import nct, t


KEYS = ("block_id", "market_id", "unit_id")
ASSIGN = (*KEYS, "saturation", "offer")
COSTS = ("refunds_cents", "cogs_cents", "fulfillment_cents", "fees_cents",
         "shrink_cents", "review_cents", "inference_cents", "other_cost_cents")
MONEY = ("gross_sales_cents", *COSTS)
COUNTS = ("eligible", "shown", "adopt_exact", "adopt_partial")


def assign(roster, seed):
    groups = defaultdict(lambda: defaultdict(list))
    units = set()
    markets = {}
    for row in roster:
        block, market, unit = (row[k] for k in KEYS)
        if not all((block, market, unit)) or unit in units:
            raise ValueError("Empty ID or duplicate unit")
        if markets.setdefault(market, block) != block:
            raise ValueError("A market belongs to multiple blocks")
        groups[block][market].append(unit)
        units.add(unit)
    if len(groups) < 2:
        raise ValueError("At least two independent blocks required")
    rng = random.Random(seed)
    result = []
    for block, bucket in sorted(groups.items()):
        if len(bucket) != 3:
            raise ValueError("Each matched block must contain three markets")
        levels = [0, 50, 100]
        rng.shuffle(levels)
        for market, level in zip(sorted(bucket), levels):
            members = sorted(bucket[market])
            if len(members) % 2:
                raise ValueError("Exact half-saturation requires even market size")
            selected = set(rng.sample(members, len(members) * level // 100))
            for unit in members:
                result.append(dict(zip(ASSIGN, (block, market, unit, level, int(unit in selected)))))
    return result


def interval(differences):
    n = len(differences)
    center = mean(differences)
    radius = t.ppf(0.975, n - 1) * stdev(differences) / math.sqrt(n)
    return {"blocks": n, "mean_currency_per_unit": center,
            "paired_block_ci95": [center - radius, center + radius]}


def count(value):
    s = str(value)
    if not s.isascii() or not s.isdigit():
        raise ValueError("Accounting amounts and counts must be nonnegative integers")
    return int(s)


def analyze(allocation, ledger, seed):
    # Recreate the assignment from the complete locked file; row order is irrelevant.
    expected = {row["unit_id"]: row for row in assign(allocation, seed)}
    for row in allocation:
        if any(str(row[k]) != str(expected[row["unit_id"]][k]) for k in ASSIGN):
            raise ValueError("Randomization mismatch")
    markets = defaultdict(list)
    adoption = defaultdict(lambda: {k: 0 for k in COUNTS})
    crossover = 0
    seen = set()
    for outcome in ledger:
        unit = outcome["unit_id"]
        if unit not in expected or unit in seen:
            raise ValueError("Extra or duplicate ledger unit")
        if outcome["reconciled"] != "1":
            raise ValueError("Missing or unsettled accounts; no completed estimate")
        seen.add(unit)
        a = expected[unit]
        m = {k: count(outcome[k]) for k in MONEY + COUNTS}
        if m["adopt_exact"] + m["adopt_partial"] > m["shown"] or m["shown"] > m["eligible"]:
            raise ValueError("Invalid adoption funnel")
        if a["offer"] == 0:
            crossover += m["shown"]  # Keep accidental exposure in its randomized arm.
        net = (m["gross_sales_cents"] - sum(m[k] for k in COSTS)) / 100
        markets[a["block_id"], a["saturation"]].append((a["offer"], net))
        totals = adoption[str(a["saturation"])]
        for key in COUNTS:
            totals[key] += m[key]
    if seen != set(expected):
        raise ValueError("Incomplete roster coverage; no completed estimate")
    contrasts = defaultdict(list)
    for block in sorted({k[0] for k in markets}):
        control = mean(net for _, net in markets[block, 0])
        full = mean(net for _, net in markets[block, 100])
        mixed = markets[block, 50]
        untreated = mean(net for offer, net in mixed if not offer)
        offered = mean(net for offer, net in mixed if offer)
        contrasts["full_vs_zero"].append(full - control)
        contrasts["untreated_half_vs_zero"].append(untreated - control)
        contrasts["offered_vs_unoffered_within_half"].append(offered - untreated)
    return {"unit": "currency per pre-enrolled unit over the fixed window",
            "weight": "equal market within each arm; equal weight per matched block",
            "estimates": {k: interval(v) for k, v in contrasts.items()},
            "adoption_counts_by_saturation": dict(adoption),
            "recommendations_shown_to_unoffered_units": crossover}


def power(effect, sd, blocks):
    critical = t.ppf(.975, blocks - 1)
    shift = effect * math.sqrt(blocks) / sd
    return float(nct.sf(critical, blocks - 1, shift) + nct.cdf(-critical, blocks - 1, shift))


def plan_power(effect, sd, target=.80, reserve=.0):
    if not 0 < target < 1 or not 0 <= reserve < 1 or min(effect, sd) <= 0:
        raise ValueError("Invalid power inputs")
    low, high = 1, 2
    while power(effect, sd, high) < target:
        high *= 2
        if high > 2**26:
            raise ValueError("Required sample exceeds supported range")
    while high - low > 1:
        midpoint = (low + high) // 2
        if power(effect, sd, midpoint) >= target:
            high = midpoint
        else:
            low = midpoint
    return {"net_itt_mde": effect, "paired_market_difference_sd": sd,
            "target_power": target, "complete_triplets": high,
            "attained_power": power(effect, sd, high),
            "planning_triplets_with_reserve": math.ceil(high / (1 - reserve))}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(value, path):
    output = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if path:
        Path(path).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assign")
    a.add_argument("roster")
    a.add_argument("output")
    a.add_argument("--seed", type=int, required=True)
    b = sub.add_parser("analyze")
    b.add_argument("allocation")
    b.add_argument("ledger")
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--allocation-sha256", required=True)
    b.add_argument("--output")
    p = sub.add_parser("power")
    p.add_argument("--net-itt", type=float, required=True)
    p.add_argument("--paired-sd", type=float, required=True)
    p.add_argument("--reserve", type=float, default=0)
    args = parser.parse_args()
    if args.command == "assign":
        assigned = assign(read_csv(args.roster), args.seed)
        with Path(args.output).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ASSIGN)
            writer.writeheader()
            writer.writerows(assigned)
        print(json.dumps({"seed": args.seed, "roster_sha256": digest(args.roster),
                          "allocation_sha256": digest(args.output), "units": len(assigned)}))
    elif args.command == "analyze":
        if digest(args.allocation) != args.allocation_sha256:
            raise ValueError("Locked allocation SHA-256 mismatch")
        result = analyze(read_csv(args.allocation), read_csv(args.ledger), args.seed)
        result["allocation_sha256"] = args.allocation_sha256
        result["ledger_sha256"] = digest(args.ledger)
        write_json(result, args.output)
    else:
        write_json(plan_power(args.net_itt, args.paired_sd, reserve=args.reserve), None)


if __name__ == "__main__":
    main()
