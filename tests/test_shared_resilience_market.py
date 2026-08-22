from dataclasses import replace

from game_theory_agent.market import CompanyAction, MarketEnv
from game_theory_agent.market.models import MarketEvent
from game_theory_agent.market.protocols import state_hash


COMPANIES = ("company_A", "company_B", "company_C", "company_D")


def _actions(state, contributions):
    return {
        company_id: CompanyAction(
            action_id=f"{state.episode_id}:{state.round}:{company_id}",
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=10_000,
            shared_resilience_contribution_cents=contributions.get(company_id, 0),
        )
        for company_id in state.company_ids
    }


def _with_disaster(state):
    event = MarketEvent(
        event_id="cooperation-test-disaster",
        event_type="supply_disruption",
        severity="high",
        started_round=state.round,
        remaining_rounds=1,
        demand_multiplier_ppm=900_000,
        supply_cost_multiplier_ppm=1_700_000,
        capacity_multiplier_ppm=500_000,
        advertising_multiplier_ppm=700_000,
        service_penalty_ppm=250_000,
        reputation_penalty_ppm=150_000,
    )
    changed = replace(state, active_market_events=(event,), state_hash="")
    return replace(changed, state_hash=state_hash(changed.to_dict()))


def test_all_cooperate_pays_now_and_builds_next_round_public_resilience(config):
    cooperative_env = MarketEnv(config)
    defect_env = MarketEnv(config)
    cooperative = cooperative_env.reset(
        COMPANIES,
        episode_id="all-cooperate",
        episode_seed=91,
        cooperation_mode="shared_resilience_v1",
        max_rounds=5,
    )
    defect = defect_env.reset(
        COMPANIES,
        episode_id="all-defect",
        episode_seed=91,
        cooperation_mode="shared_resilience_v1",
        max_rounds=5,
    )
    one_million = {company_id: 1_000_000 for company_id in COMPANIES}
    cooperative_after = cooperative_env.step(
        "all-cooperate:1:0", _actions(cooperative, one_million)
    ).state_after
    defect_after = defect_env.step(
        "all-defect:1:0", _actions(defect, {})
    ).state_after

    assert cooperative_after.shared_resilience is not None
    assert defect_after.shared_resilience is not None
    assert cooperative_after.shared_resilience.industry_resilience_ppm > 0
    assert defect_after.shared_resilience.industry_resilience_ppm == 0
    for company_id in COMPANIES:
        cooperative_company = cooperative_after.company(company_id)
        defect_company = defect_after.company(company_id)
        assert (
            defect_company.financial.round_profit_cents
            - cooperative_company.financial.round_profit_cents
            == 1_000_000
        )
    decayed = cooperative_env.step(
        "all-cooperate:2:1", _actions(cooperative_after, {})
    ).state_after
    assert decayed.shared_resilience is not None
    assert decayed.shared_resilience.industry_resilience_ppm < (
        cooperative_after.shared_resilience.industry_resilience_ppm
    )


def test_free_rider_saves_private_cost_but_receives_same_public_stock(config):
    contributing_env = MarketEnv(config)
    free_riding_env = MarketEnv(config)
    contributing = contributing_env.reset(
        COMPANIES,
        episode_id="a-contributes",
        episode_seed=92,
        cooperation_mode="shared_resilience_v1",
        max_rounds=5,
    )
    free_riding = free_riding_env.reset(
        COMPANIES,
        episode_id="a-free-rides",
        episode_seed=92,
        cooperation_mode="shared_resilience_v1",
        max_rounds=5,
    )
    all_contribute = {company_id: 1_000_000 for company_id in COMPANIES}
    only_others = {company_id: 1_000_000 for company_id in COMPANIES[1:]}
    contributing_after = contributing_env.step(
        "a-contributes:1:0", _actions(contributing, all_contribute)
    ).state_after
    free_riding_after = free_riding_env.step(
        "a-free-rides:1:0", _actions(free_riding, only_others)
    ).state_after

    assert (
        free_riding_after.company("company_A").financial.round_profit_cents
        - contributing_after.company("company_A").financial.round_profit_cents
        == 1_000_000
    )
    assert free_riding_after.shared_resilience is not None
    assert free_riding_after.shared_resilience.industry_resilience_ppm > 0
    assert (
        dict(free_riding_after.shared_resilience.last_contribution_by_company_cents)[
            "company_A"
        ]
        == 0
    )


def test_public_resilience_reduces_identical_next_round_disaster_loss(config):
    cooperative_env = MarketEnv(config)
    defect_env = MarketEnv(config)
    cooperative = cooperative_env.reset(
        COMPANIES,
        episode_id="protected",
        episode_seed=93,
        cooperation_mode="shared_resilience_v1",
        max_rounds=5,
    )
    defect = defect_env.reset(
        COMPANIES,
        episode_id="unprotected",
        episode_seed=93,
        cooperation_mode="shared_resilience_v1",
        max_rounds=5,
    )
    cooperative = cooperative_env.step(
        "protected:1:0",
        _actions(cooperative, {company_id: 1_000_000 for company_id in COMPANIES}),
    ).state_after
    defect = defect_env.step(
        "unprotected:1:0", _actions(defect, {})
    ).state_after
    cooperative = _with_disaster(cooperative)
    defect = _with_disaster(defect)
    cooperative_env.load_state(cooperative)
    defect_env.load_state(defect)

    cooperative_actions = _actions(cooperative, {})
    state_hash_before_shadow = cooperative_env.get_state().state_hash
    no_public_protection = (
        cooperative_env.counterfactual_without_public_resilience(
            cooperative, cooperative_actions
        )
    )
    assert cooperative_env.get_state().state_hash == state_hash_before_shadow

    protected_after = cooperative_env.step(
        "protected:2:1", cooperative_actions
    ).state_after
    unprotected_after = defect_env.step(
        "unprotected:2:1", _actions(defect, {})
    ).state_after
    protected_profit = sum(
        item.financial.round_profit_cents for item in protected_after.companies
    )
    unprotected_profit = sum(
        item.financial.round_profit_cents for item in unprotected_after.companies
    )
    assert protected_profit > unprotected_profit
    assert protected_after.market.lost_after_stockout_orders <= (
        unprotected_after.market.lost_after_stockout_orders
    )
    assert sum(
        protected_after.company(company_id).financial.round_profit_cents
        - no_public_protection.state_after.company(
            company_id
        ).financial.round_profit_cents
        for company_id in COMPANIES
    ) > 0
