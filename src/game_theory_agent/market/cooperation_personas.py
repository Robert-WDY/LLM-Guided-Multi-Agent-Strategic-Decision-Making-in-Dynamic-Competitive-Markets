"""Explicit public-history cooperation policies; no aliasing or hidden inputs."""
from dataclasses import replace

PERSONAS = ("cooperator", "free_rider", "retaliator")
VERSION = "public-contribution-reciprocity-v1.0.0"

def apply_cooperation_persona(config, state, company_id, action, persona):
    if persona not in PERSONAS:
        return action
    if not config.data.get("formal_cooperation_personas"):
        raise ValueError("formal cooperation personas require their versioned configuration")
    active = state.strategic_market.active_company_ids
    if company_id not in active:
        return action
    others = [a for a in state.last_joint_action if a.agent_id != company_id and a.agent_id in active]
    # Only public, settled contribution amounts are read; no other actor's cash.
    cooperative_history = not state.state_version or any(
        (a.shared_resilience_contribution_cents or 0)+(a.threshold_project_contribution_cents or 0)>0 for a in others)
    contribute = persona=="cooperator" or (persona=="retaliator" and cooperative_history)
    reason = "ongoing_public_contribution" if contribute else "private_saving" if persona=="free_rider" else "one_round_withhold_until_public_reciprocity"
    own = state.company(company_id)
    cash = max(0, own.financial.cash_balance_cents)
    reserve = max(config.integer("operating_costs","fixed_overhead_cents"), cash//2)
    envelope = min(200000, max(0,cash-reserve-action.fixed_spend_cents))
    if state.rounds_remaining<=1:
        envelope=0; reason="terminal_no_future_public_return"
    if not contribute: envelope=0
    project=state.strategic_market.threshold_project
    project_open=project is not None and project.status.value=="active" and state.round<=project.deadline_round
    threshold=envelope//2 if project_open else 0
    shared=envelope-threshold if state.shared_resilience is not None else 0
    return replace(action, shared_resilience_contribution_cents=shared if state.shared_resilience is not None else None,
                   threshold_project_contribution_cents=threshold if project is not None else None,
                   strategy_summary=(action.strategy_summary+f"; {VERSION}:{persona}:{reason}")[:500])
