import random

from game_theory_agent.market import CompanyAction, MarketEnv


def test_100_rule_driven_episodes_preserve_v4_invariants(config):
    company_ids = ("A", "B", "C", "D")
    for seed in range(100):
        policy_rng = random.Random(seed)
        env = MarketEnv(config)
        state = env.reset(company_ids, episode_id=f"episode-{seed}", episode_seed=seed)
        while not state.terminal:
            actions = {}
            for company_id in company_ids:
                last_round = state.rounds_remaining <= 1
                actions[company_id] = CompanyAction(
                    action_id=f"{state.episode_id}:{state.round}:{company_id}",
                    episode_id=state.episode_id,
                    agent_id=company_id,
                    round=state.round,
                    state_version=state.state_version,
                    price_cents=policy_rng.randrange(7500, 13001, 100),
                    advertising_budget_cents=policy_rng.randrange(
                        0, 1_000_001, 100_000
                    ),
                    service_budget_cents=policy_rng.randrange(0, 1_000_001, 100_000),
                    capacity_investment_cents=(
                        0 if last_round else policy_rng.randrange(0, 1_000_001, 100_000)
                    ),
                    resilience_budget_cents=(
                        0 if last_round else policy_rng.randrange(0, 500_001, 100_000)
                    ),
                )
            state = env.step(
                f"{state.episode_id}:{state.round}:{state.state_version}", actions
            ).state_after
        env.assert_invariants(state)
        assert all(
            company.financial.cash_balance_cents >= 0 for company in state.companies
        )
