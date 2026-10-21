"""The market itself expresses canonical game-theoretic phenomena.

These tests pin the payoff structure that makes the market a game-theory
demonstration, not just a simulator:

- ``price_war``: mutual undercutting is a Bertrand / prisoner's dilemma —
  individually tempting, collectively destructive.
- ``stag_hunt``: the threshold project is a coordination game — all-cooperate
  succeeds, all-defect fails.
- ``free_rider``: shared resilience is a non-excludable public good — a
  defector benefits from another's contribution without paying.

All runs are deterministic (fixed seeds) and use the four-role portfolios.
"""
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.actor_policies import company_action

CFG = "configs/market_v14_local.yaml"


def _run(n, choices, seed, rounds=10):
    c = load_market_config(CFG)
    e = MarketEnv(c)
    s = e.reset(
        company_ids=[f"company_{chr(65+i)}" for i in range(n)],
        episode_id=f"phenomena-{seed}-{n}",
        episode_seed=seed,
        max_rounds=rounds,
        cooperation_mode="combined_v1",
    )
    total_profit = 0
    for _ in range(rounds):
        if s.terminal:
            break
        aa = {
            cid: company_action(c, s, cid, choices.get(cid, "balanced"))
            for cid in s.company_ids
        }
        s = e.step(f"{s.episode_id}:{s.round}:{s.state_version}", aa).state_after
        total_profit += sum(comp.financial.round_profit_cents for comp in s.companies)
    return c, s, total_profit


def _ids(n):
    return [f"company_{chr(65+i)}" for i in range(n)]


def test_price_war_destroys_total_profit():
    # Bertrand / prisoner's dilemma: everyone undercuts -> everyone loses money,
    # even though any single undercutter would gain share.
    _, _, profit_balanced = _run(4, {}, seed=7)
    _, _, profit_undercut = _run(4, {cid: "undercutter" for cid in _ids(4)}, seed=7)
    assert profit_undercut < profit_balanced
    assert profit_undercut < 0  # mutual defection is collectively ruinous


def test_single_undercutter_gains_share():
    # The temptation to defect: one undercutter wins share above a fair split.
    _, s_mixed, _ = _run(4, {"company_A": "undercutter"}, seed=7)
    share = s_mixed.company("company_A").commercial.market_share_ppm / 10000
    assert share > 100.0 / 4  # above the equal 25% split


def test_threshold_project_is_stag_hunt():
    # Coordination game: collective contribution provisions the public good,
    # universal defection does not.
    _, s_coop, _ = _run(4, {cid: "cooperator" for cid in _ids(4)}, seed=3)
    _, s_def, _ = _run(4, {}, seed=3)
    tp = s_coop.strategic_market.threshold_project
    assert tp.status.value == "succeeded"
    assert tp.accumulated_total_contribution_cents >= tp.required_total_contribution_cents
    assert s_def.strategic_market.threshold_project.accumulated_total_contribution_cents == 0
    assert s_def.strategic_market.threshold_project.status.value == "failed"


def test_free_rider_benefits_from_public_good_without_paying():
    # Shared resilience is non-excludable: company_B pays nothing but the
    # industry-wide protection rises from company_A's contribution, and the
    # defector ends the episode ahead of the contributor.
    _, s, _ = _run(2, {"company_A": "cooperator"}, seed=11)
    tp = s.strategic_market.threshold_project
    assert tp.accumulated_total_contribution_cents > 0  # A paid into the pool
    assert s.shared_resilience is not None
    assert s.shared_resilience.industry_resilience_ppm > 0  # public stock exists
    profit_a = s.company("company_A").financial.round_profit_cents
    profit_b = s.company("company_B").financial.round_profit_cents
    assert profit_b > profit_a  # defection is individually tempting
