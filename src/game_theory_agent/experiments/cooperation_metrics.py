"""Research metrics for Shared Resilience cooperation events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def compute_cooperation_metrics(events: Sequence[Any]) -> dict[str, Any]:
    records = [event.cooperation_round for event in events if event.cooperation_round]
    proposals = [item for record in records for item in record.close.proposals_created]
    responses = [item for record in records for item in record.close.responses_recorded]
    commitments = [item for record in records for item in record.close.commitments_created]
    verifications = [item for record in records for item in record.verifications]
    acceptances = sum(item.response == "accept" for item in responses)
    promised = sum(item.promised_contribution_cents for item in verifications)
    fulfilled = sum(
        min(item.actual_contribution_cents, item.promised_contribution_cents)
        for item in verifications
    )
    company_rounds = sum(len(record.contribution_by_company_cents) for record in records)
    contributing_company_rounds = sum(
        amount > 0
        for record in records
        for amount in record.contribution_by_company_cents.values()
    )
    free_rider_company_rounds = sum(
        amount == 0 and record.total_contribution_cents > 0
        for record in records
        for amount in record.contribution_by_company_cents.values()
    )
    event_exposed = [
        event
        for event in events
        if event.state_before.get("active_market_events")
    ]
    attributions = [
        item
        for record in records
        for item in record.benefit_attribution_by_company.values()
    ]
    contributor_attributions = [
        item for item in attributions if item.latest_source_contribution_cents > 0
    ]
    free_rider_attributions = [
        item
        for record in records
        if record.public_protection_applied_ppm > 0
        for item in record.benefit_attribution_by_company.values()
        if item.latest_source_contribution_cents == 0
    ]
    contributor_profit_samples = [
        item.actual_round_profit_cents
        for item in contributor_attributions
    ]
    free_rider_profit_samples = [
        item.actual_round_profit_cents
        for item in free_rider_attributions
    ]
    contributor_cost = sum(
        item.latest_source_contribution_cents for item in contributor_attributions
    )
    contributor_delta = sum(
        item.counterfactual_profit_delta_cents
        for item in contributor_attributions
    )
    return {
        "metrics_schema_version": "cooperation-metrics-v1.1.0",
        "round_count": len(records),
        "proposal_count": len(proposals),
        "proposal_rate_ppm": (
            len(proposals) * 1_000_000 // company_rounds
            if company_rounds
            else 0
        ),
        "response_count": len(responses),
        "acceptance_count": acceptances,
        "peer_acceptance_rate_ppm": (
            acceptances * 1_000_000 // len(responses) if responses else 0
        ),
        "commitment_count": len(commitments),
        "commitment_rate_ppm": (
            len(commitments) * 1_000_000 // len(proposals) if proposals else 0
        ),
        "verification_count": len(verifications),
        "fulfillment_status_counts": {
            status: sum(item.status == status for item in verifications)
            for status in ("fulfilled", "partial_betrayal", "betrayed")
        },
        "amount_weighted_fulfillment_ppm": (
            fulfilled * 1_000_000 // promised if promised else 0
        ),
        "betrayal_rate_ppm": (
            sum(item.status != "fulfilled" for item in verifications)
            * 1_000_000
            // len(verifications)
            if verifications
            else 0
        ),
        "total_contribution_cents": sum(
            record.total_contribution_cents for record in records
        ),
        "contributing_company_round_rate_ppm": (
            contributing_company_rounds * 1_000_000 // company_rounds
            if company_rounds
            else 0
        ),
        "free_rider_company_round_rate_ppm": (
            free_rider_company_rounds * 1_000_000 // company_rounds
            if company_rounds
            else 0
        ),
        "mean_industry_resilience_after_ppm": (
            sum(record.industry_resilience_after_ppm for record in records)
            // len(records)
            if records
            else 0
        ),
        "mean_public_protection_applied_ppm": (
            sum(record.public_protection_applied_ppm for record in records)
            // len(records)
            if records
            else 0
        ),
        "market_total_round_profit_cents": sum(
            int(company["financial"]["round_profit_cents"])
            for event in events
            for company in event.state_after.get("companies", {}).values()
        ),
        "event_exposed_round_count": len(event_exposed),
        "event_exposed_total_profit_cents": sum(
            int(company["financial"]["round_profit_cents"])
            for event in event_exposed
            for company in event.state_after.get("companies", {}).values()
        ),
        "company_incident_observation_count": sum(
            company.get("risk", {}).get("active_incident") is not None
            for event in events
            for company in event.state_after.get("companies", {}).values()
        ),
        "cumulative_lost_after_stockout_orders": sum(
            int(event.state_after.get("market", {}).get(
                "lost_after_stockout_orders", 0
            ))
            for event in events
        ),
        "attributed_public_benefit_cents": sum(
            item.realized_avoided_loss_cents for item in attributions
        ),
        "public_protection_opportunity_cost_cents": sum(
            item.public_protection_opportunity_cost_cents
            for item in attributions
        ),
        "net_cooperative_cash_flow_cents": sum(
            item.net_cooperative_cash_flow_cents for item in attributions
        ),
        "individual_cooperative_roi_ppm": (
            contributor_delta * 1_000_000 // contributor_cost
            if contributor_cost
            else None
        ),
        "free_rider_advantage_cents": sum(
            item.free_rider_advantage_cents
            for item in free_rider_attributions
        ),
        "free_rider_mean_round_profit_cents": (
            sum(free_rider_profit_samples) // len(free_rider_profit_samples)
            if free_rider_profit_samples
            else None
        ),
        "contributor_mean_round_profit_cents": (
            sum(contributor_profit_samples) // len(contributor_profit_samples)
            if contributor_profit_samples
            else None
        ),
        "free_rider_short_term_profit_gap_cents": (
            sum(free_rider_profit_samples) // len(free_rider_profit_samples)
            - sum(contributor_profit_samples) // len(contributor_profit_samples)
            if free_rider_profit_samples and contributor_profit_samples
            else None
        ),
        "public_protection_avoided_next_incident_count": sum(
            item.avoided_next_incident for item in attributions
        ),
        "note": (
            "Market total profit is reported, but no full social-welfare system "
            "is enabled in Cooperation MVP v1."
        ),
    }


__all__ = ["compute_cooperation_metrics"]
