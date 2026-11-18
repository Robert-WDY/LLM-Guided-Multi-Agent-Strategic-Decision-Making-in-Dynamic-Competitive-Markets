"""Public-history pricing. All times are one-based, inclusive settlement rounds."""
from statistics import mean, median
import numpy as np

from . import reliable_response as response

HOLD = 4
PROBES = (.88, 1., 1.12)
WARM = 2 * len(PROBES)
MODES = ("pricing_feedback",)
DEFAULT_GATE = float("inf")  # Experiment promotion requires an independent safety result.


def bounded(value, low, high):
    return min(high, max(low, float(value)))


def failed_trial_adjustment(history, previous_delta):
    """Manager's own settlement check at renewal after a temporary higher quote."""
    if previous_delta <= 0 or len(history) < 2 * HOLD:
        return 0.
    latest, prior = history[-HOLD:], history[-2 * HOLD:-HOLD]
    if (sum(row[2] for row in latest) < sum(row[2] for row in prior)
            and sum(row[3] for row in latest) < sum(row[3] for row in prior)):
        return previous_delta
    return 0.


def feedback(history, previous, low, high):
    """Perturb-and-observe from executed prices and mean settled cycle profits."""
    if len(history) == WARM:
        pairs = [history[i:i + 2] for i in range(0, WARM, 2)]
        best = max(pairs, key=lambda group: mean(row[3] for row in group))
        return mean(row[0] for row in best), 1
    latest = history[-HOLD:]
    prior = history[-2 * HOLD:-HOLD] if len(history) >= WARM + 2 * HOLD else None
    direction = previous["direction"]
    if prior:
        delta_price = mean(r[0] for r in latest) - mean(r[0] for r in prior)
        if abs(delta_price) > 1.e-7:
            direction = 1 if delta_price > 0 else -1
        if mean(r[3] for r in latest) < mean(r[3] for r in prior):
            direction *= -1
    price = previous.get("renewal_anchor", mean(r[0] for r in latest))
    trial = bounded(price * (1 + .035 * direction), low, high)
    if abs(trial - price) < 1.e-7:
        direction *= -1
        trial = bounded(price * (1 + .035 * direction), low, high)
    return trial, direction


def alternatives(history, draft, low, high, steps, mode):
    """Same ridge profit surface and draws. Only the response conditioning differs."""
    rows = np.asarray([row[:4] for row in history[-40:]], dtype=float)
    p, rival, quantity, profit = rows.T
    center = max(np.median(p), 1.e-8)
    p_rel, r_rel = p / center - 1., rival / center - 1.
    scale = max(1.e-8, float(np.mean(np.abs(profit))))
    design = np.column_stack((np.ones(len(p)), p_rel, p_rel**2, r_rel, p_rel*r_rel))
    weights = .94 ** np.arange(len(p)-1, -1, -1)
    reg = np.diag([1.e-6, .025, .004, .04, .02])
    coef = np.linalg.solve(design.T @ (design * weights[:, None]) + reg,
                           design.T @ (weights * profit / scale))
    residual = np.sqrt(np.average((profit / scale - design @ coef)**2, weights=weights))
    # Estimate the change in contribution from a second, demand-based surface.
    # Anchoring the unit margin to recent settlements keeps fixed costs local.
    volumes = np.maximum(quantity, .01)
    demand_x = design[:, [0, 1, 3]]
    demand_reg = np.diag([1.e-6, .012, .02])
    demand_coef = np.linalg.solve(demand_x.T @ (weights[:, None]*demand_x)+demand_reg,
                                  demand_x.T @ (weights*np.log(volumes)))
    margin = median((profit/volumes)[-4:])
    anchor = median(p[-4:])
    samples = [dict(features=(response.bucket(p[i-1]/p[i-2]-1), 0, 0),
                    change=rival[i]/rival[i-1]-1) for i in range(2, len(p))]
    model = response.fit(samples)
    kind = "conditional" if mode == "conditional" else "generic"
    predictions = [response.predict(model, (b, 0, 0), kind) for b in (-1, 0, 1)]
    probs = np.array([v["probabilities"] for v in predictions])
    changes = np.array([v["changes"] for v in predictions])
    width, paths = (15, 3) if mode == "wide" else (5, 9)
    candidates = np.clip(draft*(1 + np.linspace(-.08, .08, width)), low, high)
    ref = width//2
    moves = np.full(width, response.bucket(p[-1]/p[-2]-1)+1)
    peer = np.full((width, paths), rival[-1])
    x = candidates[:, None]/center-1
    rewards = np.zeros_like(peer)
    contribution = np.zeros_like(peer)
    # Deterministic stratified draws shared by all candidates.
    for j in range(steps):
        u = ((np.arange(paths)+.5)/paths + j*.381966) % 1
        cat = (u >= probs[moves, 0, None]).astype(int) + (u >= probs[moves, :2].sum(axis=1)[:, None])
        peer = np.clip(peer*(1+changes[moves[:, None], cat]), low, high)
        y = peer/center - 1
        rewards += (coef[0]+coef[1]*x+coef[2]*x*x+coef[3]*y+coef[4]*x*y)
        quantity_hat = np.exp(np.clip(demand_coef[0]+demand_coef[1]*x+demand_coef[2]*y, -8, 16))
        contribution += quantity_hat * (margin + candidates[:, None] - anchor) / scale
        moves = (np.array([response.bucket(v/p[-1]-1) for v in candidates])+1
                 if j == 0 else np.ones(width, dtype=int))
    gains = (rewards - rewards[ref]) / steps
    gains = np.minimum(gains, (contribution-contribution[ref])/steps)
    # Rank the best distinct alternative even when every alternative looks worse.
    distinct = np.flatnonzero(np.abs(candidates-draft) > 1.e-8)
    if not len(distinct):
        return draft, -1., width*paths*steps
    best = distinct[np.argmax(gains[distinct].mean(axis=1))]
    distance = max(0., candidates[best]-p.max(), p.min()-candidates[best])/center
    score = np.quantile(gains[best], .1) / max(.05, residual) - distance
    return float(candidates[best]), float(score), width*paths*steps


def decide(t, remaining, reference, low, high, history, previous=None, *, mode="feedback",
           gate=DEFAULT_GATE, original=None):
    """Called before settlement. History rows are own/rival prices, own sales and profit."""
    if remaining < 1 or not 0 < low <= high or t < 1:
        raise ValueError("invalid price bounds or clock")
    previous = previous or {}
    if previous and t <= previous["end"]:
        if t < previous["start"]:
            raise ValueError("price decision clock went backwards")
        return previous
    if t <= WARM:
        target = bounded(reference * PROBES[(t-1)//2], low, high)
        length, direction, score, work, proposed, draft = 2, 1, None, 0, target, target
    else:
        length = HOLD
        draft, direction = feedback(history, previous, low, high)
        if mode == "fixed":
            draft = previous["draft"] if t > WARM+1 else draft
        if original is not None:
            draft = bounded(original, low, high)
        target = proposed = draft
        score, work = None, 0
        if mode in ("generic", "conditional", "wide"):
            proposed, score, work = alternatives(history, draft, low, high, min(remaining,length), mode)
            if score > gate:
                target = proposed
        elif mode not in ("fixed", "feedback"):
            raise ValueError("unknown pricing policy")
    end = t + min(remaining, length)-1
    return dict(start=t, end=end, next_decision=end+1 if remaining>length else None,
                price=target, draft=draft, proposed=proposed, direction=direction,
                score=score, transitions=work)


def native_decision(config, state, actor, memory, mode):
    from ..market.actor_policies import observation
    obs = observation(state, actor)
    old = memory if memory and memory.get("episode") == state.episode_id else {}
    history = list(old.get("history", []))
    prices = obs["prices"]
    if old and obs["round"] > old["round"]:
        peers = [v for name,v in prices.items() if name != actor]
        history.append([prices[actor], median(peers), obs["sales"], obs["profit"]])
    reference = old.get("reference", prices[actor])
    bounds = config.mapping("action","bounds")["price_cents"]
    decision = decide(obs["round"], obs["remaining"], reference, bounds["min"], bounds["max"],
                      history, old.get("decision"), mode=mode.removeprefix("pricing_"))
    return decision, dict(episode=state.episode_id, reference=reference, history=history[-40:], round=obs["round"], decision=decision)


def apply_native_price(config, state, actor, base, price):
    from ..decisioning import resolve_action_request
    request = base.to_dict()
    request["price_cents"] = round(price)
    return resolve_action_request(config,state,actor,request,source="cycle-price",
                                  action_id=f"cycle:{state.round}:{actor}").action

