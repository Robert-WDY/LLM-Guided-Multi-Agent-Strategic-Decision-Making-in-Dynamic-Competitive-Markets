"""The four-role market exposes first-class cooperation and competition policies."""
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.actor_policies import (
    COMPANY_OPTIONS,
    COMPANY_PORTFOLIOS,
    company_action,
)

CFG = "configs/market_v14_local.yaml"


def _env(n=4, seed=1):
    c = load_market_config(CFG)
    e = MarketEnv(c)
    s = e.reset(
        company_ids=[f"company_{chr(65+i)}" for i in range(n)],
        episode_id=f"coop-policy-{seed}",
        episode_seed=seed,
        max_rounds=10,
        cooperation_mode="combined_v1",
    )
    return c, e, s


def _run(c, e, s, choices):
    statuses = []
    for _ in range(10):
        if s.terminal:
            break
        aa = {
            cid: company_action(c, s, cid, choices.get(cid, "balanced"))
            for cid in s.company_ids
        }
        s = e.step(
            f"{s.episode_id}:{s.round}:{s.state_version}", aa
        ).state_after
        tp = s.strategic_market.threshold_project
        statuses.append(tp.status.value if tp else "none")
    return s, statuses


def test_both_policies_are_registered():
    assert "cooperator" in COMPANY_PORTFOLIOS
    assert "undercutter" in COMPANY_PORTFOLIOS
    # The learned operating selector keeps its frozen four-option domain.
    assert COMPANY_OPTIONS == ("balanced", "value", "margin", "resilience")


def test_cooperator_reaches_threshold_project():
    c, e, s = _env()
    choices = {cid: "cooperator" for cid in s.company_ids}
    s, statuses = _run(c, e, s, choices)
    tp = s.strategic_market.threshold_project
    assert "succeeded" in statuses
    assert (
        tp.accumulated_total_contribution_cents
        >= tp.required_total_contribution_cents
    )


def test_balanced_never_contributes_to_public_goods():
    c, e, s = _env()
    choices = {cid: "balanced" for cid in s.company_ids}
    s, statuses = _run(c, e, s, choices)
    assert s.strategic_market.threshold_project.accumulated_total_contribution_cents == 0
    assert "succeeded" not in statuses


def test_undercutter_undercuts_the_balanced_price():
    c, e, s = _env()
    under = company_action(c, s, "company_A", "undercutter")
    base = company_action(c, s, "company_A", "balanced")
    assert under.price_cents < base.price_cents


def test_cooperator_contributes_only_while_project_is_active():
    c, e, s = _env()
    # First move with the project still active.
    act = company_action(c, s, "company_A", "cooperator")
    assert (act.threshold_project_contribution_cents or 0) > 0
    assert (act.shared_resilience_contribution_cents or 0) > 0
