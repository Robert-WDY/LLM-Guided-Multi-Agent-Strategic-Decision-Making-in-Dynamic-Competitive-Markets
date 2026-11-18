"""Research estimates for four public posted quotes and a four-round price contract."""

from pathlib import Path
import numpy as np
from .quote_panel import public_state, rivals, seller_features

LABELS = ("keep", "cut", "raise")


class Forest:
    """Small, data-only export of a trained regression forest. No pickle on inference."""

    def __init__(self, archive, name):
        self.roots, self.feature, self.threshold, self.left, self.right, self.value = (
            archive[f"{name}_{key}"]
            for key in ("roots", "feature", "threshold", "left", "right", "value")
        )

    def predict(self, features):
        row = np.asarray(features, dtype=np.float32)
        indices = self.roots.copy()
        while True:
            active = self.left[indices] >= 0
            if not active.any():
                return self.value[indices].mean(axis=0)
            nodes = indices[active]
            indices[active] = np.where(
                row[self.feature[nodes]] <= self.threshold[nodes],
                self.left[nodes],
                self.right[nodes],
            )


class PanelAdvisor:
    def __init__(self, model_file):
        with np.load(Path(model_file), allow_pickle=False) as archive:
            self.quotes = Forest(archive, "quotes")
            self.direct = Forest(archive, "direct")
            self.value = Forest(archive, "value")

    def review(self, history, seat, t, draft):
        h = np.asarray(history, dtype=float)
        if (
            h.shape != (t - 1, 4)
            or not 19 <= t <= 55
            or (t - 19) % 4
            or not 0 <= seat < 4
        ):
            raise ValueError(
                "Expected four posted prices per settled round at a review date"
            )
        if (
            np.isinf(h).any()
            or np.any(h[np.isfinite(h)] <= 0)
            or not np.isfinite(draft)
            or draft <= 0
        ):
            raise ValueError(
                "Expected positive prices, or null for a seller without a quote"
            )
        if not np.isfinite(h[-1, seat]) or not rivals(h, seat):
            return dict(candidate="keep", reason="no_public_competition", forecasts=[])
        peers = rivals(h, seat)
        forecasts = []
        logs = [0.0] * 3
        for k, j in enumerate(peers):
            now, next_contract = self.quotes.predict(seller_features(h, j, t))
            logs[k] = float(now)
            last = h[-1, j]
            forecasts.append(
                dict(
                    seat=j,
                    last=float(last),
                    this_contract=float(last * np.exp(now)),
                    following_contract=float(last * np.exp(next_contract)),
                )
            )
        state = public_state(h, seat, t, draft)
        value = [0.0, *map(float, self.value.predict(state + logs))]
        direct = [0.0, *map(float, self.direct.predict(state))]
        return dict(
            candidate=LABELS[int(np.argmax(value))],
            simple_candidate=LABELS[int(np.argmax(direct))],
            estimated_terminal_gain=dict(zip(LABELS, value)),
            simple_terminal_gain=dict(zip(LABELS, direct)),
            forecasts=forecasts,
        )
