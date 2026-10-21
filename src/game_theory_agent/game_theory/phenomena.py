"""Market-coupled game-theory phenomena: classify, compute, and advise.

Each phenomenon is a symmetric 2x2 binary choice (cooperate / defect) whose
payoff matrix is extracted from the *actual market* by counterfactual rollouts,
then classified by the standard payoff ordering. The advice is a structural,
evidence-backed statement about the incentive structure — not a profit
guarantee and not a substitute for the economic reliability gates.

The focal player's matrix is laid out as::

               opponent C   opponent D
    focal C        R             S
    focal D        T             P

R = reward for mutual cooperation, S = sucker payoff, T = temptation to
defect, P = punishment for mutual defection.
"""

from __future__ import annotations

from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.market import MarketEnv, load_market_config


# ---------------------------------------------------------------------------
# 1. Classification: symmetric 2x2 game by payoff ordering
# ---------------------------------------------------------------------------

def classify_2x2(R, S, T, P):
    """Classify a symmetric 2x2 game and report its incentive structure.

    Classification uses best-response structure (with strict inequalities),
    which is robust to the T == P and R == P ties common in empirical payoffs.
    """
    defect_dominant = (T > R) and (P > S)
    cooperate_dominant = (R > T) and (S > P)
    if defect_dominant:
        kind = "prisoners_dilemma" if R > P else "deadlock"
    elif cooperate_dominant:
        kind = "harmony"
    elif (R > T) and (P > S):
        kind = "stag_hunt"  # two stable equilibria: mutual cooperate and mutual defect
    elif (T > R) and (S > P):
        kind = "chicken"    # anti-coordination: best to mismatch
    else:
        kind = "undefined"
    return dict(
        kind=kind,
        payoffs=dict(R=R, S=S, T=T, P=P),
        defection_incentive=T - R,          # temptation: defect beats cooperate if opponent cooperates
        defection_vs_defection=P - S,       # whether defect beats cooperate when opponent defects
        mutual_cooperation_value=R - P,     # collective gain of cooperation over mutual defection
        dominant_action=(
            "defect" if defect_dominant
            else "cooperate" if cooperate_dominant
            else None
        ),
        coordination_risk=(
            "two equilibria: cooperate only if the opponent cooperates"
            if kind == "stag_hunt" else None
        ),
    )


def _advice(kind):
    """One-line, phenomenon-labeled advice for the classified game."""
    return {
        "prisoners_dilemma": (
            "囚徒困境：单方面背叛有诱惑（T>R 且 P>S），但互相背叛差于互相合作（R>P）。"
            "一次性博弈应警惕成为唯一合作者；存在声誉/重复互动时才值得合作以维持 R。"
        ),
        "stag_hunt": (
            "猎鹿博弈：互相合作最优，但单独合作会吃亏（R>T 且 P>S）。"
            "只有在足够相信对手也会合作时才贡献，否则保守选择更安全。"
        ),
        "chicken": (
            "胆小鬼博弈：有人贡献时你搭便车更优（T>R），但无人贡献时你去贡献更优（S>P）。"
            "需要判断对手是否会让步；若都选择搭便车则公共项目失败，对双方都不利。"
        ),
        "harmony": (
            "和谐博弈：合作是严格占优（R>T 且 S>P）。直接合作即可。"
        ),
        "deadlock": (
            "僵局博弈：背叛占优且互相背叛不比合作差（T>R、P>S 且 P≥R）。"
            "当前参数下合作在经济上不划算，不应为了'显得合作'而贡献。"
        ),
        "undefined": "未识别为经典对称博弈，按一般收益比较处理。",
    }.get(kind, "未识别博弈结构。")


# ---------------------------------------------------------------------------
# 2. Empirical payoff extraction from the market
# ---------------------------------------------------------------------------

def _mutate_shared(action, choice, amount=1_000_000):
    action["shared_resilience_contribution_cents"] = amount if choice == "cooperate" else 0
    return action


def _mutate_threshold(action, choice, amount=2_000_000):
    action["threshold_project_contribution_cents"] = amount if choice == "cooperate" else 0
    return action


def binary_game(config, state, cid, opponent, mutator, *, horizon=3, seed=42):
    """Extract the empirical 2x2 payoff matrix for a binary choice.

    The focal company and one named opponent each choose cooperate/defect for
    the first round; everyone continues with the rule baseline afterwards. The
    focal payoff is its cumulative round profit over ``horizon`` rounds.
    """
    matrix = {}
    abbr = {"cooperate": "C", "defect": "D"}
    for focal in ("cooperate", "defect"):
        for opp in ("cooperate", "defect"):
            e = MarketEnv(config)
            e.load_state(state)
            s = state
            payoff = 0
            for _ in range(horizon):
                if s.terminal:
                    break
                aa = {}
                for c in s.company_ids:
                    req = build_rule_action(config, s, c).to_dict()
                    if c == cid:
                        req = mutator(req, focal)
                    elif c == opponent:
                        req = mutator(req, opp)
                    aa[c] = resolve_action_request(
                        config, s, c, req, source="phenomena",
                        action_id=f"ph:{s.round}:{c}:{focal}:{opp}",
                    ).action
                s = e.step(f"{s.episode_id}:{s.round}:{s.state_version}", aa).state_after
                payoff += s.company(cid).financial.round_profit_cents
            matrix[f"{abbr[focal]}/{abbr[opp]}"] = payoff
    R = matrix["C/C"]
    S = matrix["C/D"]
    T = matrix["D/C"]
    P = matrix["D/D"]
    return R, S, T, P, matrix


# ---------------------------------------------------------------------------
# 3. Phenomenon reports and advice
# ---------------------------------------------------------------------------

def shared_resilience_report(config, state, cid, opponent, *, horizon=3):
    """Shared resilience contribution as a public-good binary game."""
    R, S, T, P, matrix = binary_game(
        config, state, cid, opponent, _mutate_shared, horizon=horizon
    )
    cls = classify_2x2(R, S, T, P)
    cls["phenomenon"] = "shared_resilience"
    cls["title"] = "共享韧性公共品"
    cls["parameters"] = dict(horizon=horizon, contribution_cents=1_000_000)
    cls["matrix"] = matrix
    cls["advice"] = _advice(cls["kind"])
    cls["recommended_action"] = _recommend(cls)
    return cls


def threshold_project_report(config, state, cid, opponent, *, horizon=3):
    """Threshold cold-chain project as a coordination (stag-hunt) game."""
    R, S, T, P, matrix = binary_game(
        config, state, cid, opponent, _mutate_threshold, horizon=horizon
    )
    cls = classify_2x2(R, S, T, P)
    cls["phenomenon"] = "threshold_project"
    cls["title"] = "门槛冷链公共项目"
    cls["parameters"] = dict(horizon=horizon, contribution_cents=2_000_000)
    cls["matrix"] = matrix
    cls["advice"] = _advice(cls["kind"])
    cls["recommended_action"] = _recommend(cls)
    return cls


def _recommend(cls):
    """Map the classified game to a recommended binary action."""
    if cls["dominant_action"]:
        return cls["dominant_action"]
    if cls["kind"] == "stag_hunt":
        # Cooperate is best only under mutual trust; without that signal, the
        # safe action is defect. The advice text carries the coordination caveat.
        return "defect_if_uncertain"
    if cls["kind"] == "chicken":
        # Anti-coordination: free-ride if the opponent contributes, but step in
        # if the opponent defects and the public good is worth saving.
        return "free_ride_if_opponent_contributes"
    return "compare_payoffs"


def advise(config, state, cid, opponent, *, horizon=3):
    """Produce the full phenomenon-organized report for a focal company."""
    shared = shared_resilience_report(config, state, cid, opponent, horizon=horizon)
    threshold = threshold_project_report(config, state, cid, opponent, horizon=horizon)
    return dict(
        company_id=cid,
        opponent_id=opponent,
        horizon=horizon,
        phenomena=[shared, threshold],
        scope=(
            "固定状态、固定对手、单轮二值选择、多轮利润的经验反事实；"
            "识别的是激励结构，不是已验证的现实行为预测，也不替代经济可靠性门槛。"
        ),
    )


def _increased_cooperation(action, draft):
    """Whether the advisor raised a public-good contribution above the draft."""
    a = action or {}
    d = draft or {}
    return (
        (a.get("shared_resilience_contribution_cents") or 0)
        > (d.get("shared_resilience_contribution_cents") or 0)
        or (a.get("threshold_project_contribution_cents") or 0)
        > (d.get("threshold_project_contribution_cents") or 0)
    )


def price_sensitive(config):
    """Detect a price-sensitive market (roughly 2x the base price coefficient).

    In such markets the v20.1 holdout shows ``price_up`` recommendations have a
    short-term (3-round) gain but a longer-term (5-round) loss, so price hikes
    are flagged for caution rather than passed silently.
    """
    d = getattr(config, "data", {}) or {}
    segments = (d.get("consumer_choice") or {}).get("segments") or {}
    price_segment = segments.get("price_sensitive") or {}
    coeff = (price_segment.get("coefficients_ppm") or {}).get("price", 0)
    return int(coeff or 0) >= 2_000_000


def phenomena_aware_advise(config, state, request, *, horizon=3, base_advise=None):
    """Run the advisor, attach the game-theory diagnosis, and veto cooperation
    recommendations when the relevant public-good game is a deadlock.

    The veto is the decision-level use of the diagnosis: if the advisor raises
    a shared-resilience or threshold-project contribution while that public
    good is a deadlock (defection strictly dominates and cooperation is not
    worth it), revert to the agent's own draft and explain why.
    """
    from .empirical_advisor import advise as empirical_advise

    result = (base_advise or empirical_advise)(config, state, request)
    cid = getattr(request, "company_id", "company_A")
    opponent = None
    if len(state.company_ids) >= 2:
        opponent = state.company_ids[1] if state.company_ids[0] == cid else state.company_ids[0]
    if opponent is None:
        result["game_theory_phenomena"] = None
        return result

    report = advise(config, state, cid, opponent, horizon=horizon)
    result["game_theory_phenomena"] = report

    if result.get("disposition") == "recommend" and _increased_cooperation(
        result.get("action"), result.get("draft_action")
    ):
        for p in report["phenomena"]:
            if p["kind"] == "deadlock":
                result.update(
                    action=result["draft_action"],
                    recommended_id="agent_draft",
                    recommended_label="Agent原计划",
                    disposition="abstain",
                    rationale=(
                        f"博弈诊断：{p['title']}为僵局博弈（背叛严格占优且合作不划算），"
                        "已否决合作类建议，保留原计划。"
                    ),
                    game_theory_veto=dict(
                        phenomenon=p["phenomenon"],
                        kind=p["kind"],
                        reason="deadlock: cooperation is not game-theoretically justified",
                    ),
                )
                break

    # Price-sensitivity caution: a price hike in a price-elastic market is
    # short-term-profitable but long-term self-defeating (observed 3-round gain,
    # 5-round loss in the v20.1 holdout). Flag it for the caller; this is an
    # experimental signal, not a validated release gate.
    if price_sensitive(config) and result.get("recommended_id") == "price_up":
        result["price_sensitivity_caution"] = dict(
            market="price_sensitive",
            reason=(
                "价格敏感市场：提价在3轮内有利、5轮内丢份额转亏；"
                "建议谨慎采纳或优先考虑降价/保份额。"
            ),
        )
    return result
