"""Minimize non-binding advice before it enters an LLM decision context."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def build_agent_advice_view(advice: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return only fields that are allowed to influence the Agent.

    Full candidate tables remain useful for evaluator audit, but exposing them
    after a reliable advisor has abstained creates an anchoring side channel.
    A released recommendation retains its auditable artifact.  An abstention
    returns ``None`` so the Agent receives exactly the same decision input as
    the no-advice baseline.  The full artifact belongs in evaluator/audit
    storage, not in the Agent-visible observation.
    """

    payload = deepcopy(dict(advice))
    if (
        payload.get("execution_disposition") != "defer_to_agent"
        or payload.get("advisor_mode") != "strategic_market_v9"
    ):
        return payload
    return None
