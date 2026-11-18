"""Export the frozen two-stage model for offline, public-only shadow reviews."""

import numpy as np
from cause_trials import MARKETS
from export_panel import pack
from renewal_risk_study import fit
from renewal_trials import OUT
from run import read


if __name__ == "__main__":
    calibration = read(OUT / "freeze.json")
    for market in MARKETS:
        model = fit(market)
        data = {}
        for key, wrapped in (
            ("path", model.aux["renewal"]),
            ("economic", model.final["renewal"]),
        ):
            data.update(pack(key, wrapped.regressor_))
            data[key + "_mean"] = wrapped.transformer_.mean_
            data[key + "_scale"] = wrapped.transformer_.scale_
        data["gate"] = [
            calibration[market]["models"]["renewal"]["chosen"]["cutoff"],
            20000 if market == "native" else 5,
        ]
        np.savez_compressed(OUT / f"public-{market}.npz", **data)
