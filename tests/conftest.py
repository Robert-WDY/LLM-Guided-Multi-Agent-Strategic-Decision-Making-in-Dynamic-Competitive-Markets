from pathlib import Path

import pytest

from game_theory_agent.market import CompanyAction, MarketEnv, load_market_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config():
    return load_market_config(ROOT / "configs" / "market_v4.yaml")


@pytest.fixture()
def env(config):
    return MarketEnv(config)


@pytest.fixture()
def initial_state(env):
    return env.reset(
        ("company_A", "company_B", "company_C", "company_D"), episode_seed=42
    )


@pytest.fixture()
def make_actions():
    def factory(state, *, nonce="base", overrides=None):
        overrides = overrides or {}
        result = {}
        for company_id in state.company_ids:
            values = {
                "price_cents": 10000,
                "advertising_budget_cents": 800000,
                "service_budget_cents": 800000,
                "capacity_investment_cents": 0,
                "resilience_budget_cents": 0,
            }
            values.update(overrides.get(company_id, {}))
            result[company_id] = CompanyAction(
                action_id=f"{state.episode_id}:{state.round}:{company_id}:{nonce}",
                episode_id=state.episode_id,
                agent_id=company_id,
                round=state.round,
                state_version=state.state_version,
                **values,
            )
        return result

    return factory
