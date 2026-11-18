"""Read-only pricing review from public information before a four-round contract."""

import numpy as np
from .panel_advisor import Forest, LABELS
from .quote_panel import public_state, rivals


def estimates(values):
    by_arm = np.asarray(values).reshape(-1, 2, 4)
    gain, chance_changed, chance_lost, downside = np.moveaxis(by_arm, 2, 0)
    risk = np.divide(
        chance_lost, chance_changed, out=np.ones_like(gain), where=chance_changed > 1e-9
    )
    return gain - downside, np.clip(risk, 0, 1)


def decisions(values, enabled, cutoff, hurdle):
    score, risk = estimates(values)
    feasible = np.asarray(enabled)[:, None] & (score > hurdle) & (risk <= cutoff)
    return np.column_stack(
        (np.zeros(len(score)), np.where(feasible, score, -np.inf))
    ).argmax(axis=1)


class PublicRenewalReview:
    def __init__(self, archive):
        with np.load(archive, allow_pickle=False) as data:
            self.path = Forest(data, "path")
            self.economic = Forest(data, "economic")
            self.scaling = {
                k: (data[k + "_scale"], data[k + "_mean"]) for k in ("path", "economic")
            }
            self.cutoff, self.hurdle = map(float, data["gate"])

    def review(self, history, seat, t, draft):
        h = np.asarray(history, float)
        if type(t) is not int or t not in range(23, 44, 4) or h.shape != (t - 1, 4):
            raise ValueError(
                "Expected four public listings per previous round at a studied renewal"
            )
        if (
            type(seat) is not int
            or seat not in range(4)
            or type(draft) not in (int, float)
            or not np.isfinite(draft)
            or draft <= 0
            or np.isinf(h).any()
            or np.any(h <= 0)
        ):
            raise ValueError(
                "Expected a valid seat, positive draft and positive listings or null"
            )
        state = public_state(h, seat, t, draft)
        enabled = bool(np.isfinite(h[-1, seat]) and rivals(h, seat))
        scale, mean = self.scaling["path"]
        path = self.path.predict(state) * scale + mean
        scale, mean = self.scaling["economic"]
        y = self.economic.predict([*state, *path]) * scale + mean
        action = decisions(y, [enabled], self.cutoff, self.hurdle)[0]
        scores, risk = estimates(y)
        alternatives = {}
        if enabled:
            for i, arm in enumerate(("cut", "raise")):
                future = path.reshape(2, 2, 2)[i]
                alternatives[arm] = dict(
                    predicted_log_quote_difference=future[:, 0].tolist(),
                    predicted_listing_probability_difference=future[:, 1].tolist(),
                    estimated_terminal_gain=float(y[4 * i]),
                    expected_downside=float(y[4 * i + 3]),
                    penalized_gain=float(scores[0, i]),
                    loss_probability_if_executed=float(risk[0, i]),
                )
        return dict(
            candidate=LABELS[action],
            disposition="keep",
            requested=draft,
            contract_last_round=t + 3,
            future_renewals=[t + 4, t + 8],
            status="shadow_only" if enabled else "no_public_competition",
            cutoff=self.cutoff,
            alternatives=alternatives,
        )
