"""Offline A/A, known direct/spillover injection, and financial-ledger acceptance."""

import argparse
from copy import deepcopy
from random import Random

import numpy as np
from scipy.stats import t

from trial import COSTS, analyze, assign, plan_power, write_json


def simulate(allocation, active):
    rows = []
    for a in allocation:
        level, offer = a["saturation"], a["offer"]
        # Exactly two of four opportunities adopt. Expenses accrue on all offers.
        gross_gain = 1200 * offer + 400 * level // 100 if active else 0
        record = dict.fromkeys(COSTS, 0)
        record.update(unit_id=a["unit_id"], reconciled="1", eligible=4, shown=4 * offer,
                      adopt_exact=offer, adopt_partial=offer,
                      gross_sales_cents=10000 + gross_gain,
                      cogs_cents=2000, fulfillment_cents=500, refunds_cents=100,
                      fees_cents=100, shrink_cents=50, review_cents=100 * offer * active,
                      inference_cents=50 * offer * active)
        rows.append(record)
    return rows


def require_failure(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Invalid input accepted")


def check():
    members = [{"block_id": str(b), "market_id": f"{b}-{m}", "unit_id": f"{b}-{m}-{u}"}
               for b in range(36) for m in range(3) for u in range(4)]
    seed = 816
    allocation = assign(members, seed)
    shuffled = members[:]
    Random(1004).shuffle(shuffled)
    assert assign(shuffled, seed) == allocation
    aa = analyze(allocation, simulate(allocation, False), seed)
    ledger = simulate(allocation, True)
    active = analyze(allocation, ledger, seed)
    expected = {"full_vs_zero": 14.50, "untreated_half_vs_zero": 2.00,
                "offered_vs_unoffered_within_half": 10.50}
    for name, value in expected.items():
        assert aa["estimates"][name]["mean_currency_per_unit"] == 0
        assert active["estimates"][name]["mean_currency_per_unit"] == value
    require_failure(lambda: analyze(allocation, ledger[:-1], seed))
    require_failure(lambda: analyze(allocation, ledger + [ledger[0]], seed))
    unsettled = deepcopy(ledger)
    unsettled[0]["reconciled"] = "0"
    require_failure(lambda: analyze(allocation, unsettled, seed))
    swapped = deepcopy(allocation)
    swapped[0]["offer"] = 1 - int(swapped[0]["offer"])
    require_failure(lambda: analyze(swapped, ledger, seed))
    crossover = deepcopy(ledger)
    unoffered = next(i for i, a in enumerate(allocation) if not a["offer"])
    crossover[unoffered]["shown"] = 1
    assert analyze(allocation, crossover, seed)["recommendations_shown_to_unoffered_units"] == 1
    # Check interval coverage and power under independent Gaussian paired differences.
    design = plan_power(10, 25, reserve=.10)
    n = design["complete_triplets"]
    noise = np.random.default_rng(23971).normal(0, 25, size=(10000, n))
    averages = noise.mean(axis=1)
    radii = t.ppf(.975, n - 1) * noise.std(axis=1, ddof=1) / np.sqrt(n)
    false_positive = float((abs(averages) > radii).mean())
    achieved_power = float((abs(averages + 10) > radii).mean())
    assert .04 < false_positive < .06
    assert abs(achieved_power - design["attained_power"]) < .02
    return {"source": "synthetic QA only; zero firms, orders and model calls",
            "allocation_seed": seed, "units": len(members), "matched_market_triplets": 36,
            "aa": aa, "injected": active, "expected_injected_means": expected,
            "checks": ["roster-order invariance", "missing/duplicate/unsettled ledger rejected",
                       "allocation edit rejected", "accidental control exposure retained in ITT"],
            "illustrative_power_only": design,
            "gaussian_paired_difference_simulation": {"replications": len(noise),
                "seed": 23971, "null_rejection": false_positive, "power": achieved_power}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    write_json(check(), parser.parse_args().output)
