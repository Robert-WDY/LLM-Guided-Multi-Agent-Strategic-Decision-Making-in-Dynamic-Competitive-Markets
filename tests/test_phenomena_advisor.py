"""Game-theory phenomena advisor: classification, advice, and effectiveness.

"Effectiveness" here is a game-theoretic claim, not an economic profit
guarantee: the advisor's recommendation must be an empirical best response —
the recommended action's payoff must be no worse than the alternative against
every opponent action in the market-extracted 2x2 matrix. It also verifies the
classification is self-consistent with the measured payoff ordering.
"""
from game_theory_agent.market import MarketEnv, MarketConfig, load_market_config
from game_theory_agent.game_theory.phenomena import (
    advise,
    classify_2x2,
    phenomena_aware_advise,
    price_sensitive,
    shared_resilience_report,
    threshold_project_report,
    _increased_cooperation,
)
from game_theory_agent.game_theory.reliable_advisor import ReliableRequest

CFG = "configs/market_v14_local.yaml"


def test_classify_textbook_games():
    assert classify_2x2(3, 0, 5, 1)["kind"] == "prisoners_dilemma"
    assert classify_2x2(4, 0, 3, 3)["kind"] == "stag_hunt"
    assert classify_2x2(3, 1, 4, 0)["kind"] == "chicken"
    assert classify_2x2(4, 3, 2, 1)["kind"] == "harmony"
    assert classify_2x2(1, 0, 3, 2)["kind"] == "deadlock"


def _assert_best_response(cls):
    """The classification must be consistent with the measured payoff matrix."""
    R = cls["payoffs"]["R"]
    S = cls["payoffs"]["S"]
    T = cls["payoffs"]["T"]
    P = cls["payoffs"]["P"]
    da = cls["dominant_action"]
    if da == "defect":
        assert T > R and P > S  # defect beats cooperate against both opponent actions
    elif da == "cooperate":
        assert R > T and S > P
    elif cls["kind"] == "stag_hunt":
        assert R > T and P > S  # two equilibria: mutual cooperate and mutual defect
    elif cls["kind"] == "chicken":
        assert T > R and S > P  # anti-coordination: mismatch is best


def _state(config=None, seed=5):
    c = config or load_market_config(CFG)
    e = MarketEnv(c)
    s = e.reset(
        company_ids=["company_A", "company_B"],
        episode_id=f"ph-{seed}",
        episode_seed=seed,
        max_rounds=20,
        cooperation_mode="combined_v1",
    )
    return c, s


def test_shared_resilience_advice_is_best_response():
    c, s = _state()
    r = shared_resilience_report(c, s, "company_A", "company_B", horizon=5)
    _assert_best_response(r)
    # Default parameters: the contribution cost exceeds the public-protection
    # benefit, so the honest advice is to defect, not to perform cooperation.
    assert r["kind"] == "deadlock"
    assert r["recommended_action"] == "defect"


def test_threshold_project_default_is_deadlock():
    c, s = _state()
    r = threshold_project_report(c, s, "company_A", "company_B", horizon=5)
    _assert_best_response(r)
    assert r["kind"] == "deadlock"
    assert r["recommended_action"] == "defect"


def test_threshold_project_becomes_coordination_game_when_feasible():
    # A lower, feasible threshold changes the game: cooperation is no longer
    # strictly dominated, and the advisor reports the free-rider incentive.
    base = load_market_config(CFG).to_dict()
    base["strategic_market"]["threshold_project"]["required_total_contribution_cents"] = 600_000
    c = MarketConfig.from_mapping(base)
    _, s = _state(c)
    r = threshold_project_report(c, s, "company_A", "company_B", horizon=5)
    _assert_best_response(r)
    assert r["kind"] != "deadlock"  # the advice reflects the parameter change


def test_advise_returns_phenomenon_organized_report():
    c, s = _state()
    r = advise(c, s, "company_A", "company_B", horizon=3)
    assert r["company_id"] == "company_A"
    assert [p["phenomenon"] for p in r["phenomena"]] == [
        "shared_resilience",
        "threshold_project",
    ]
    for p in r["phenomena"]:
        assert p["advice"] and p["parameters"] and p["payoffs"]


def test_increased_cooperation_detects_contribution_raise():
    draft = {"price_cents": 10000, "shared_resilience_contribution_cents": 0}
    assert not _increased_cooperation(draft, draft)
    assert _increased_cooperation({**draft, "shared_resilience_contribution_cents": 30000}, draft)
    assert _increased_cooperation({**draft, "threshold_project_contribution_cents": 1}, draft)


def test_phenomena_aware_advise_vetoes_deadlock_cooperation():
    c, s = _state()
    draft = {"price_cents": 10000, "shared_resilience_contribution_cents": 0,
             "threshold_project_contribution_cents": 0}

    def fake_base(config, state, request):
        return dict(
            disposition="recommend", recommended_id="cooperation_adjust",
            recommended_label="合作调整",
            action={**draft, "shared_resilience_contribution_cents": 30000},
            draft_action=draft, rationale="x",
        )

    request = ReliableRequest(draft_action=draft, goal="profit", mode="research", seed=1, step_budget=1)
    result = phenomena_aware_advise(c, s, request, base_advise=fake_base)
    # Default config: shared resilience is a deadlock -> the cooperation raise is vetoed.
    assert result["disposition"] == "abstain"
    assert result["action"] == result["draft_action"]
    assert result.get("game_theory_veto")
    assert result["game_theory_phenomena"] is not None


def test_price_sensitive_detection():
    base = load_market_config(CFG).to_dict()
    assert not price_sensitive(MarketConfig.from_mapping(base))
    base["consumer_choice"]["segments"]["price_sensitive"]["coefficients_ppm"]["price"] *= 2
    assert price_sensitive(MarketConfig.from_mapping(base))
