from dataclasses import replace

from fastapi.testclient import TestClient

from game_theory_agent.acceptance import evaluate_safety_matrix
from game_theory_agent.agents import CounterfactualEvaluator, PlanTracker
from game_theory_agent.agents.context import DecisionContextBuilder
from game_theory_agent.agents.memory import EpisodeMemory
from game_theory_agent.agents.contracts import ExpectedOutcome, SuccessCriteria
from game_theory_agent.agents.result_analyzer import ResultAnalyzer
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.economics import decision_support_metrics
from game_theory_agent.api import SESSIONS, agent_app, app
from game_theory_agent.market import MarketEnv
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.model_clients import UniformRandomIntentPolicy


def _request(**overrides):
    values = {
        "price_cents": 10_000,
        "advertising_budget_cents": 5_000_000,
        "service_budget_cents": 5_000_000,
        "capacity_investment_cents": 6_000_000,
        "resilience_budget_cents": 4_000_000,
        "incident_response": {"mode": "wait", "repair_budget_cents": 0},
    }
    values.update(overrides)
    return values


def test_controller_preserves_overhead_when_cash_is_critical(config, initial_state):
    company = initial_state.company("company_A")
    critical = replace(
        company,
        financial=replace(company.financial, cash_balance_cents=1_114_810),
    )
    state = replace(
        initial_state,
        companies=tuple(
            critical if item.company_id == "company_A" else item
            for item in initial_state.companies
        ),
    )

    resolved = resolve_action_request(
        config, state, "company_A", _request(), source="test"
    )

    assert resolved.action.fixed_spend_cents == 0
    assert {
        item.reason_code for item in resolved.adjustments
    } >= {"LIQUIDITY_RESERVE_PROTECTED"}


def test_controller_protects_minimum_unit_contribution(config, initial_state):
    stressed = replace(
        initial_state,
        market=replace(
            initial_state.market, actual_supply_cost_index_ppm=1_354_000
        ),
    )
    support = decision_support_metrics(config, stressed, "company_A")

    resolved = resolve_action_request(
        config,
        stressed,
        "company_A",
        _request(
            price_cents=7_600,
            advertising_budget_cents=0,
            service_budget_cents=0,
            capacity_investment_cents=0,
            resilience_budget_cents=0,
        ),
        source="test",
    )

    assert resolved.action.price_cents == support["minimum_safe_price_cents"]
    assert any(
        item.reason_code == "NEGATIVE_UNIT_MARGIN_PROTECTED"
        for item in resolved.adjustments
    )


def test_plan_tracker_enters_recovery_and_liquidity_phases(config, initial_state):
    company = initial_state.company("company_A")
    losing = replace(
        company,
        financial=replace(company.financial, cash_balance_cents=20_000_000),
        history=replace(company.history, recent_profit_cents=(-100_000, -200_000)),
    )
    state = replace(
        initial_state,
        companies=tuple(
            losing if item.company_id == "company_A" else item
            for item in initial_state.companies
        ),
    )
    support = decision_support_metrics(config, state, "company_A")
    plan = PlanTracker().evaluate(
        round_number=3,
        decision_support=support,
        rolling_summary={},
        previous_plan=None,
    )
    assert plan["phase"] == "profit_recovery"
    assert "do not lower price" in " ".join(plan["rules"])
    recovery_resolution = resolve_action_request(
        config,
        state,
        "company_A",
        _request(
            price_cents=7_500,
            advertising_budget_cents=0,
            service_budget_cents=0,
            capacity_investment_cents=0,
            resilience_budget_cents=0,
        ),
        source="test",
    )
    assert recovery_resolution.action.price_cents == company.commercial.price_cents
    assert any(
        item.reason_code == "RECOVERY_PRICE_FLOOR"
        for item in recovery_resolution.adjustments
    )

    one_recovery_profit = replace(
        losing,
        history=replace(
            losing.history,
            recent_profit_cents=(-100_000, -200_000, 50_000),
        ),
    )
    sticky_state = replace(
        state,
        companies=tuple(
            one_recovery_profit if item.company_id == "company_A" else item
            for item in state.companies
        ),
    )
    sticky_support = decision_support_metrics(config, sticky_state, "company_A")
    assert sticky_support["strategic_phase"] == "profit_recovery"

    two_recovery_profits = replace(
        one_recovery_profit,
        history=replace(
            one_recovery_profit.history,
            recent_profit_cents=(-200_000, 50_000, 60_000),
        ),
    )
    recovered_state = replace(
        sticky_state,
        companies=tuple(
            two_recovery_profits if item.company_id == "company_A" else item
            for item in sticky_state.companies
        ),
    )
    recovered_support = decision_support_metrics(
        config, recovered_state, "company_A"
    )
    assert recovered_support["strategic_phase"] == "growth"

    crisis_company = replace(
        losing, financial=replace(losing.financial, cash_balance_cents=5_000_000)
    )
    crisis_state = replace(
        state,
        companies=tuple(
            crisis_company if item.company_id == "company_A" else item
            for item in state.companies
        ),
    )
    crisis_support = decision_support_metrics(config, crisis_state, "company_A")
    crisis = PlanTracker().evaluate(
        round_number=4,
        decision_support=crisis_support,
        rolling_summary={},
        previous_plan=plan,
    )
    assert crisis["phase"] == "liquidity_crisis"
    assert crisis["constraints"]["maximum_discretionary_spend_cents"] == 0


def test_plan_tracker_preserves_plan_until_expiry_or_new_trigger(
    config, initial_state
):
    support = decision_support_metrics(config, initial_state, "company_A")
    tracker = PlanTracker()
    first = tracker.evaluate(
        round_number=1,
        decision_support=support,
        rolling_summary={},
        previous_plan=None,
    )
    second = tracker.evaluate(
        round_number=2,
        decision_support=support,
        rolling_summary={},
        previous_plan=first,
    )
    triggered = tracker.evaluate(
        round_number=3,
        decision_support=support,
        rolling_summary={},
        previous_plan=second,
        risk_signals=[{"signal_id": "risk-1", "event_type": "demand_shock"}],
    )

    assert first["replanned"] is True
    assert second["plan_id"] == first["plan_id"]
    assert second["replanned"] is False
    assert triggered["plan_id"] != first["plan_id"]
    assert triggered["replan_reason"].startswith("critical_event:risk:")


def test_uniform_random_policy_handles_zero_cash_and_is_reproducible():
    SESSIONS.clear()
    response = TestClient(app).post(
        "/api/episodes",
        json={
            "episode_id": "zero-cash-random",
            "episode_seed": 7,
            "company_ids": ["company_A", "company_B"],
            "max_rounds": 5,
            "information_mode": "public",
        },
    )
    assert response.status_code == 201
    session = SESSIONS["zero-cash-random"]
    state = session.env.get_state()
    company = state.company("company_A")
    zero = replace(
        company, financial=replace(company.financial, cash_balance_cents=0)
    )
    changed = replace(
        state,
        state_hash="",
        companies=tuple(
            zero if item.company_id == "company_A" else item
            for item in state.companies
        ),
    )
    changed = replace(changed, state_hash=state_hash(changed.to_dict()))
    session.env.load_state(changed)
    observation = TestClient(agent_app).get(
        "/v1/episodes/zero-cash-random/companies/company_A/observation"
    ).json()
    context = DecisionContextBuilder().build(
        observation, "company_A", EpisodeMemory()
    )
    policy = UniformRandomIntentPolicy(7)

    first = policy.sample(context)
    second = policy.sample(context)

    assert first == second
    assert first.price_cents >= context.decision_support["minimum_safe_price_cents"]
    assert first.advertising_budget_cents == 0
    assert first.service_budget_cents == 0
    assert first.capacity_investment_cents == 0
    assert first.resilience_budget_cents == 0


def test_counterfactual_holds_other_actions_and_randomness_fixed(
    config, initial_state, make_actions
):
    actions = make_actions(initial_state)
    env = MarketEnv(config)
    env.load_state(initial_state)
    actual = env.step(
        f"{initial_state.episode_id}:{initial_state.round}:{initial_state.state_version}",
        actions,
    )
    analysis = CounterfactualEvaluator(config).evaluate(
        initial_state,
        actual.state_after,
        {key: value.to_dict() for key, value in actions.items()},
        "company_A",
    )

    assert analysis["method"] == "same_state_same_seed_other_company_actions_fixed"
    assert len(analysis["alternatives"]) == 2
    assert all(
        item["invariant_results"] == ["all_passed"]
        for item in analysis["alternatives"]
    )


def test_forecast_accuracy_does_not_turn_a_loss_into_goal_success(
    config, initial_state, make_actions
):
    env = MarketEnv(config)
    env.load_state(initial_state)
    first = env.step(
        f"{initial_state.episode_id}:{initial_state.round}:{initial_state.state_version}",
        make_actions(initial_state, nonce="first"),
    )
    before = first.state_after
    actions = make_actions(
        before,
        nonce="loss",
        overrides={
            "company_A": {
                "price_cents": 7_500,
                "advertising_budget_cents": 5_000_000,
                "service_budget_cents": 5_000_000,
            }
        },
    )
    result = env.step(
        f"{before.episode_id}:{before.round}:{before.state_version}",
        actions,
    )
    analysis = ResultAnalyzer().analyze(
        before,
        result.state_after,
        "company_A",
        ExpectedOutcome(profit="down"),
        [],
        SuccessCriteria(minimum_round_profit_cents=0),
    )

    assert analysis.expectation_assessment.matches["profit"] is True
    assert analysis.goal_assessment.achieved is False
    assert "minimum_round_profit" in analysis.goal_assessment.violations


def test_acceptance_matrix_smoke(config):
    report = evaluate_safety_matrix(config, seed_count=1, rounds=5)

    assert report["total_episodes"] == 5
    assert report["completed_episodes"] == 5
    assert report["invariant_failures"] == 0
    assert report["negative_margin_executions"] == 0
    assert report["liquidity_reserve_violations"] == 0
    assert report["passed"] is True
