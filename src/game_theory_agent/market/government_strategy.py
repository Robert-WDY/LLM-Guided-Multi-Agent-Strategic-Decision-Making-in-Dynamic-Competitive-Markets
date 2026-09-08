"""Budgeted policy portfolio; learning uses only completed public outcomes."""
import json
import math
from dataclasses import replace

OPTIONS = ("reserve", "enforce", "resilience", "consumer", "balanced")
PPM = 1_000_000


def unpack(value):
    return json.loads(value) if value else {}


def pack(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def select(memory):
    counts = memory.get("counts", {})
    total = sum(counts.values())
    for option in OPTIONS:
        if not counts.get(option):
            return option
    means = memory.get("means", {})
    return max(OPTIONS, key=lambda o: means[o] + math.sqrt(2 * math.log(total + 1) / counts[o]))


def plan(base, policy, resilience, memory, remaining, requested=None):
    option = requested or select(memory)
    if option not in OPTIONS:
        raise ValueError("unknown government policy")
    p = policy["strategic_policy"]
    enabled = policy["enabled"] and bool(base.active_company_ids)
    cash = base.opening_cash_cents if enabled else 0
    cases = min(policy["max_inspection_cases"], len(base.active_company_ids)//2,
                cash//policy["inspection_cost_per_case_cents"]) if option in ("enforce", "balanced") else 0
    inspection = cases * policy["inspection_cost_per_case_cents"]
    spend = min(p["max_program_cents"], (cash-inspection)*p["program_cash_share_ppm"]//PPM)
    grants = {cid: 0 for cid in base.active_company_ids}
    consumer = 0
    if remaining > 1 and option in ("resilience", "balanced") and grants:
        target = min(grants, key=lambda cid: (resilience[cid], cid))
        grants[target] = spend if option == "resilience" else spend//2
    if option in ("consumer", "balanced"):
        consumer = spend-sum(grants.values())
    audit = dict(version="budgeted-policy-learning-v1", option=option,
                 selection="external" if requested else "ucb_public_welfare",
                 opening_memory=memory, observed_resilience=resilience, remaining_rounds=remaining,
                 consumer_budget_cents=consumer, rebate_share_ppm=p["rebate_share_ppm"],
                 fine_multiplier_ppm=p["fine_multiplier_ppm"] if option == "enforce" else PPM,
                 grant_contract="reimburse_verified_current_resilience_expense",
                 consumer_contract="post_purchase_rebate_no_intrinsic_valuation_change")
    return replace(base, inspection_cases=cases, inspection_cost_cents=inspection,
                   detection_boost_ppm=policy["detection_boost_ppm"] if cases else 0,
                   support_by_company_cents=tuple(sorted(grants.items())),
                   reason=option, policy_version=audit["version"], strategic_policy=pack(audit))


def matched_support(decision, actions):
    caps = dict(decision.support_by_company_cents)
    if not decision.strategic_policy:
        return caps
    return {cid: min(value, actions[cid].resilience_budget_cents) for cid, value in caps.items()}


def rebates(decision, audits):
    if not decision or not decision.strategic_policy:
        return audits
    p = unpack(decision.strategic_policy)
    left = p["consumer_budget_cents"]
    result = {}
    # Lower disclosed budget cohorts first; deterministic rationing.
    for d in sorted(audits, key=lambda d: (d.unit_budget_cents, d.group_id)):
        value = min(left, (d.spending_cents-d.refund_cents)*p["rebate_share_ppm"]//PPM)
        left -= value
        result[d.group_id] = replace(d, government_rebate_cents=value)
    return tuple(result[d.group_id] for d in audits)


def learn(government, welfare):
    if not government.last_decision or not government.last_decision.strategic_policy:
        return government
    policy = unpack(government.last_decision.strategic_policy)
    memory = unpack(government.policy_memory)
    counts = dict(memory.get("counts", {})); means = dict(memory.get("means", {}))
    option = policy["option"]; count = counts.get(option, 0) + 1
    # Fixed monotone bounded normalization, rather than future-run statistics.
    reward = welfare / (abs(welfare) + 100_000_000)
    means[option] = means.get(option, 0.0) + (reward-means.get(option, 0.0))/count
    counts[option] = count
    return replace(government, policy_memory=pack(dict(counts=counts, means=means,
        last_reward_cents=welfare, observations=sum(counts.values()))))
