"""Public information at a simultaneous contract renewal. NaN means no posted quote."""

import numpy as np


def last_changes(history, seat):
    last = history[-1, seat]
    return [
        float(np.log(last / history[-1 - delay, seat]))
        if np.isfinite(history[-1 - delay, seat])
        else 0.0
        for delay in (4, 8, 12)
    ]


def rivals(history, seat):
    return sorted(
        (
            j
            for j in range(history.shape[1])
            if j != seat and np.isfinite(history[-1, j])
        ),
        key=lambda j: (history[-1, j], j),
    )


def seller_features(history, seat, t):
    last = history[-1, seat]
    peers = rivals(history, seat)
    current = [np.log(history[-1, j] / last) for j in peers]
    movement = [last_changes(history, j)[0] for j in peers]
    return [
        t / 60,
        np.log(last),
        *last_changes(history, seat),
        len(peers) / 3,
        *(
            [np.min(current), np.median(current), np.max(current), np.mean(movement)]
            if peers
            else [0.0] * 4
        ),
    ]


def public_state(history, seat, t, draft):
    if len(history) != t - 1 or len(history) < 13:
        raise ValueError("Need settled public prices through the previous round")
    if not np.isfinite(history[-1, seat]):
        return [0.0] * 20
    peers = rivals(history, seat)
    full = seller_features(history, seat, t) + [np.log(draft / history[-1, seat])]
    for j in peers:
        full.extend(
            [
                1.0,
                np.log(history[-1, j] / history[-1, seat]),
                last_changes(history, j)[0],
            ]
        )
    return full + [0.0] * (20 - len(full))
