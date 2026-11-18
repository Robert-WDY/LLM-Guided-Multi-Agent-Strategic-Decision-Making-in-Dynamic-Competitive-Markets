"""Explain a frozen guard decision using only already posted prices and a draft."""

import argparse
import gzip
import json
import math
from pathlib import Path
from game_theory_agent.game_theory.public_price_guard import PublicPriceGuard

BANK = (
    Path(__file__).resolve().parents[2]
    / "experiment-results/price-guard-v1/bank.json.gz"
)
ACTIONS = ("keep", "cut", "raise")


def positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def review(proposal):
    market = proposal["market"]
    if market not in ("native", "econ", "appointments"):
        raise ValueError("Unknown market")
    rows = proposal["prices"]
    t, draft = proposal["round"], proposal["draft"]
    if type(t) is not int or not 1 <= t <= 60 or not positive(draft):
        raise ValueError("Expected a positive draft and round from 1 to 60")
    if len(rows) >= t or any(
        len(r) != 2 or not all(positive(v) for v in r) for r in rows
    ):
        raise ValueError(
            "Expected only settled [own price, rival median] pairs before this round"
        )
    candidate = ACTIONS.index(proposal["proposal"])
    bank = json.loads(gzip.decompress(BANK.read_bytes()))
    if t not in range(19, 56, 4):
        result = dict(action=0, reason="outside_review_round")
    else:
        result = PublicPriceGuard(bank).explain(
            "native" if market == "native" else "econ", rows, draft, t, candidate
        )
    action = result.pop("action")
    return dict(
        disposition=ACTIONS[action],
        candidate=ACTIONS[candidate],
        draft=draft,
        requested=draft * (1, 0.96, 1.04)[action],
        **result,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "proposal",
        type=Path,
        help="UTF-8 JSON containing market, round, prices, draft and proposal",
    )
    args = parser.parse_args()
    try:
        result = review(json.loads(args.proposal.read_text(encoding="utf8")))
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
