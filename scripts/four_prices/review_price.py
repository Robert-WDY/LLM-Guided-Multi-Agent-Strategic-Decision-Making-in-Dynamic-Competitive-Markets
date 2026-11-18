"""Explain a frozen guard decision using only already posted prices and a draft."""

import argparse
import gzip
import json
import math
from pathlib import Path
from game_theory_agent.game_theory.public_price_guard import PublicPriceGuard
from game_theory_agent.game_theory.panel_advisor import PanelAdvisor

BANK = (
    Path(__file__).resolve().parents[2]
    / "experiment-results/price-guard-v1/bank.json.gz"
)
CERTIFICATES = BANK.parents[1] / "price-causes-v1/certificates.json"
ACTIONS = ("keep", "cut", "raise")


def positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def review(proposal):
    market = proposal["market"]
    if market not in ("native", "econ", "appointments"):
        raise ValueError("Unknown market")
    if "panel" in proposal:
        if market not in ("native", "econ"):
            raise ValueError("No fitted four-seller model for this market")
        result = PanelAdvisor(CERTIFICATES.parent / f"panel-{market}.npz").review(
            proposal["panel"], proposal["seat"], proposal["round"], proposal["draft"]
        )
        certificate = (
            json.loads(CERTIFICATES.read_text(encoding="utf8"))[market]["forecast"]
            if CERTIFICATES.exists()
            else False
        )
        action = result["candidate"] if certificate else "keep"
        return dict(
            **result,
            disposition=action,
            qualified_on_calibration=certificate,
            requested=proposal["draft"]
            * {"keep": 1.0, "cut": 0.96, "raise": 1.04}[action],
        )
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
    if candidate and t in range(19, 56, 4):
        certificates = (
            json.loads(CERTIFICATES.read_text(encoding="utf8"))
            if CERTIFICATES.exists()
            else {}
        )
        certificate = (
            certificates.get(market, {}).get("external", {}).get(ACTIONS[candidate])
        )
        result["independent_calibration"] = certificate
        if result["action"] and not (certificate and certificate["passed"]):
            result.update(action=0, reason="independent_calibration")
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
    parser.add_argument(
        "--own-renewal",
        action="store_true",
        help="Preview the own-renewal risk estimate for a four-seller public panel",
    )
    args = parser.parse_args()
    try:
        proposal = json.loads(args.proposal.read_text(encoding="utf8"))
        if args.own_renewal:
            from game_theory_agent.game_theory.renewal_review import PublicRenewalReview

            market = proposal["market"]
            if market not in ("native", "econ"):
                raise ValueError("No own-renewal study for this market")
            weights = BANK.parents[1] / "renewal-risk-v1" / f"public-{market}.npz"
            result = PublicRenewalReview(weights).review(
                proposal["panel"],
                proposal["seat"],
                proposal["round"],
                proposal["draft"],
            )
        else:
            result = review(proposal)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
