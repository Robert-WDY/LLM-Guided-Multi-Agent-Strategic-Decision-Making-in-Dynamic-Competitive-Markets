from copy import deepcopy

import pytest

from game_theory_agent.agents import load_persona_registry
from game_theory_agent.agents.single.gateway import GatewaySnapshot
from game_theory_agent.agents.single.models import (
    DecisionCandidate,
    DecisionProposal,
    DecisionTrace,
    EconomicAction,
    PersonaProfile,
)
from game_theory_agent.agents.single.provider import ProviderError, ProviderInvalidDecisionError, ProviderResult, TokenUsage
from game_theory_agent.agents.single.graph import normalize_evidence_paths
from game_theory_agent.agents.single.runtime import SingleAgentRuntime
from game_theory_agent.run_agents import PROJECT_ROOT


MODEL_ID = "nvidia/nemotron-3-super-120b-a12b:free"


def trace_persona_manifest() -> dict[str, object]:
    return load_persona_registry(PROJECT_ROOT / "configs" / "market_v4.yaml").get("balanced").manifest_dict()


def proposal(price_cents: int = 1180) -> DecisionProposal:
    def item(candidate_id: str, price: int) -> DecisionCandidate:
        return DecisionCandidate(
            candidate_id=candidate_id,
            label=candidate_id,
            action=EconomicAction(
                price_cents=price,
                advertising_budget_cents=200,
                service_budget_cents=100,
                resilience_budget_cents=50,
                strategy_summary="平衡增长与现金",
            ),
            evidence_paths=["own_company.financial.cash_balance_cents"],
            tradeoffs=["增长与现金"],
            expected_outcome="份额小幅提高",
        )

    return DecisionProposal(
        candidates=[item("safe", 1300), item("balanced", price_cents), item("growth", 1100)],
        selected_candidate_id="balanced",
        selection_reason_codes=["share_growth", "cash_protection"],
    )


def snapshot(*, terminal: bool = False) -> GatewaySnapshot:
    observation = {
        "episode_id": "episode-1",
        "company_id": "company_A",
        "round": 3,
        "state_version": 2,
        "state_hash": "abc",
        "terminal": terminal,
        "own_company": {"financial": {"cash_balance_cents": 10_000}},
        "public_companies": [],
        "public_history": [],
    }
    contract = {
        "episode_id": "episode-1",
        "company_id": "company_A",
        "round": 3,
        "state_version": 2,
        "constraints": {
            "cash_available_cents": 10_000,
            "bounds": {
                "price_cents": {"min": 800, "max": 2000},
                "advertising_budget_cents": {"min": 0, "max": 1000},
                "service_budget_cents": {"min": 0, "max": 1000},
                "capacity_investment_cents": {"min": 0, "max": 1000},
                "resilience_budget_cents": {"min": 0, "max": 1000},
                "repair_budget_cents": {"min": 0, "max": 1000},
            },
            "capacity_investment_enabled": True,
            "resilience_investment_enabled": True,
            "active_incident": None,
            "max_useful_repair_budget_cents": 0,
        },
    }
    return GatewaySnapshot(
        episode_id="episode-1",
        company_id="company_A",
        round=3,
        state_version=2,
        state_hash="abc",
        observation=observation,
        action_contract=contract,
    )


def snapshot_with_history() -> GatewaySnapshot:
    value = snapshot()
    value.observation["round"] = 4
    value.observation["state_version"] = 3
    value.observation["state_hash"] = "hash-4"
    value.observation["public_history"] = [
        {
            "settled_round": 3,
            "own_action": {"price_cents": 1180},
            "own_result": {
                "round_profit_cents": 500,
                "market_share_ppm": 230_000,
            },
            "market": {"realized_demand_orders": 1_200},
            "active_events_during_round": [],
            "resolved_signal_outcomes": [],
        }
    ]
    return GatewaySnapshot(
        episode_id="episode-1",
        company_id="company_A",
        round=4,
        state_version=3,
        state_hash="hash-4",
        observation=value.observation,
        action_contract=value.action_contract | {"round": 4, "state_version": 3},
    )


class FakeGateway:
    def __init__(self, value: GatewaySnapshot):
        self.value = value
        self.submissions = []

    def load_snapshot(self, episode_id: str, company_id: str) -> GatewaySnapshot:
        assert (episode_id, company_id) == ("episode-1", "company_A")
        return self.value

    def submit_intent(self, **kwargs):
        self.submissions.append(deepcopy(kwargs))
        return {"intent_id": "intent-1", "status": "accepted", "resolution": {"adjustments": []}}


class FakeProvider:
    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.calls = []

    def generate_decision(self, **kwargs) -> ProviderResult:
        self.calls.append(deepcopy(kwargs))
        selected = self.proposals.pop(0)
        if isinstance(selected, Exception):
            raise selected
        return ProviderResult(
            proposal=selected,
            model_id=MODEL_ID,
            usage=TokenUsage(total_tokens=100),
            latency_ms=12,
        )


class TraceCollector:
    def __init__(self):
        self.items = []
        self.prior = []

    def append(self, trace):
        self.items.append(trace)

    def read_company_before_round(self, episode_id, company_id, round_number, limit=5):
        return [
            item
            for item in self.prior
            if item.episode_id == episode_id
            and item.company_id == company_id
            and item.round < round_number
        ][-limit:]


def build_runtime(provider, gateway):
    traces = TraceCollector()
    return SingleAgentRuntime(provider=provider, gateway=gateway, trace_store=traces), traces


def test_graph_submits_one_valid_intent_and_records_explainable_trace():
    provider = FakeProvider([proposal()])
    gateway = FakeGateway(snapshot())
    runtime, traces = build_runtime(provider, gateway)

    result = runtime.decide_round(
        episode_id="episode-1",
        company_id="company_A",
        model_id=MODEL_ID,
        persona_manifest=trace_persona_manifest(),
    )

    assert result.status == "accepted"
    assert result.intent_id == "intent-1"
    assert len(provider.calls) == 1
    assert provider.calls[0]["context"].snapshot_key.state_hash == "abc"
    assert provider.calls[0]["context"].memory.recent_feedback == []
    assert provider.calls[0]["context"].reflection.source == "deterministic"
    assert provider.calls[0]["context"].reflection.summary
    assert "persona" not in provider.calls[0]
    assert len(gateway.submissions) == 1
    assert gateway.submissions[0]["action"].price_cents == 1180
    assert traces.items[0].selected_candidate_id == "balanced"
    assert traces.items[0].persona_manifest.persona_id == "balanced"
    assert traces.items[0].persona_manifest.profile_hash == trace_persona_manifest()["profile_hash"]
    assert traces.items[0].prepared_intent is not None
    assert traces.items[0].prepared_intent.snapshot_key.state_hash == "abc"
    assert traces.items[0].candidates[1].evidence_paths == [
        "observation.own_company.financial.cash_balance_cents"
    ]


def test_graph_passes_previous_round_feedback_to_provider_and_trace():
    provider = FakeProvider([proposal()])
    gateway = FakeGateway(snapshot_with_history())
    runtime, traces = build_runtime(provider, gateway)
    traces.prior.append(
        DecisionTrace(
            episode_id="episode-1",
            company_id="company_A",
            round=3,
            state_version=2,
            status="accepted",
            model_id=MODEL_ID,
            persona=PersonaProfile(),
            candidates=proposal().candidates,
            selected_candidate_id="balanced",
        )
    )

    result = runtime.decide_round(
        episode_id="episode-1",
        company_id="company_A",
        model_id=MODEL_ID,
    )

    assert result.status == "accepted"
    context = provider.calls[0]["context"]
    assert [item.settled_round for item in context.memory.recent_feedback] == [3]
    assert context.memory.previous_selected_candidate_id == "balanced"
    assert context.reflection.source == "deterministic"
    assert traces.items[0].memory_view.recent_feedback[0].settled_round == 3
    assert traces.items[0].strategy_reflection.source == "deterministic"


def test_graph_repairs_an_invalid_selected_action_once_before_submitting():
    provider = FakeProvider([proposal(price_cents=5000), proposal(price_cents=1180)])
    gateway = FakeGateway(snapshot())
    runtime, traces = build_runtime(provider, gateway)

    result = runtime.decide_round(
        episode_id="episode-1",
        company_id="company_A",
        model_id=MODEL_ID,
    )

    assert result.status == "accepted"
    assert len(provider.calls) == 2
    assert provider.calls[1]["repair_errors"] == ["price_cents_above_max"]
    assert len(gateway.submissions) == 1
    assert traces.items[0].repair_attempts == 1


def test_graph_uses_shared_repair_budget_for_provider_format_and_action_errors():
    provider = FakeProvider([ProviderError("invalid json"), proposal(price_cents=5000)])
    gateway = FakeGateway(snapshot())
    runtime, traces = build_runtime(provider, gateway)

    result = runtime.decide_round(
        episode_id="episode-1",
        company_id="company_A",
        model_id=MODEL_ID,
    )

    assert result.status == "no_intent"
    assert len(provider.calls) == 2
    assert provider.calls[1]["repair_errors"] == ["provider_output_invalid"]
    assert gateway.submissions == []
    assert traces.items[0].prepared_intent is None
    assert traces.items[0].validation_errors == ["price_cents_above_max"]


def test_graph_records_failed_provider_attempt_usage_latency_and_error_category():
    failure = lambda: ProviderInvalidDecisionError(
        "safe invalid output",
        code="truncated",
        latency_ms=37,
        usage=TokenUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        finish_reason="length",
    )
    provider = FakeProvider([failure(), failure()])
    runtime, traces = build_runtime(provider, FakeGateway(snapshot()))

    result = runtime.decide_round(
        episode_id="episode-1",
        company_id="company_A",
        model_id=MODEL_ID,
    )

    assert result.status == "no_intent"
    assert traces.items[0].latency_ms == 74
    assert traces.items[0].provider_usage["total_tokens"] == 36
    assert traces.items[0].provider_finish_reason == "length"
    assert traces.items[0].provider_error_category == "truncated"
    assert traces.items[0].provider_usage_available is True


def test_graph_reports_each_provider_attempt_without_blind_waiting():
    failure = ProviderInvalidDecisionError(
        "safe invalid output",
        code="truncated",
        latency_ms=37,
        usage=TokenUsage(total_tokens=18),
        finish_reason="length",
    )
    provider = FakeProvider([failure, proposal()])
    runtime, _ = build_runtime(provider, FakeGateway(snapshot()))
    events = []

    result = runtime.decide_round(
        episode_id="episode-1",
        company_id="company_A",
        model_id=MODEL_ID,
        progress_callback=lambda stage, details: events.append((stage, details)),
    )

    assert result.status == "accepted"
    provider_events = [item for item in events if item[0].startswith("provider_")]
    assert [item[0] for item in provider_events] == [
        "provider_request",
        "provider_error",
        "provider_request",
        "provider_response",
    ]
    assert provider_events[0][1]["attempt"] == 1
    assert provider_events[1][1] == {
        "attempt": 1,
        "error_category": "truncated",
        "finish_reason": "length",
        "usage_available": True,
        "total_tokens": 18,
        "latency_ms": 37,
    }
    assert provider_events[2][1]["attempt"] == 2
    assert provider_events[3][1]["status"] == "received"


def test_graph_returns_no_intent_after_the_single_repair_is_still_invalid():
    provider = FakeProvider([proposal(price_cents=5000), proposal(price_cents=6000)])
    gateway = FakeGateway(snapshot())
    runtime, traces = build_runtime(provider, gateway)

    result = runtime.decide_round(
        episode_id="episode-1",
        company_id="company_A",
        model_id=MODEL_ID,
    )

    assert result.status == "no_intent"
    assert result.intent_id is None
    assert len(provider.calls) == 2
    assert gateway.submissions == []
    assert traces.items[0].validation_errors == ["price_cents_above_max"]


def test_graph_does_not_reuse_previous_round_receipt_after_a_failed_round():
    provider = FakeProvider(
        [
            proposal(),
            proposal(price_cents=5000),
            proposal(price_cents=6000),
        ]
    )
    gateway = FakeGateway(snapshot())
    runtime, traces = build_runtime(provider, gateway)

    accepted = runtime.decide_round(
        episode_id="episode-1",
        company_id="company_A",
        model_id=MODEL_ID,
    )
    gateway.value = snapshot_with_history()
    failed = runtime.decide_round(
        episode_id="episode-1",
        company_id="company_A",
        model_id=MODEL_ID,
    )

    assert accepted.intent_id == "intent-1"
    assert failed.status == "no_intent"
    assert failed.intent_id is None
    assert traces.items[-1].intent_receipt is None
    assert traces.items[-1].prepared_intent is None


def test_runtime_rejects_unvalidated_or_forged_persona_trace_data():
    runtime, _ = build_runtime(FakeProvider([proposal()]), FakeGateway(snapshot()))

    with pytest.raises(ValueError):
        runtime.decide_round(
            episode_id="episode-1",
            company_id="company_A",
            model_id=MODEL_ID,
            persona_manifest={"api_key": "not-a-real-secret"},
        )

    forged = trace_persona_manifest() | {"profile_hash": "sha256:" + "0" * 64}
    with pytest.raises(ValueError, match="profile_hash"):
        runtime.decide_round(
            episode_id="episode-1",
            company_id="company_A",
            model_id=MODEL_ID,
            persona_manifest=forged,
        )


def test_graph_stops_cleanly_on_a_terminal_observation():
    provider = FakeProvider([proposal()])
    gateway = FakeGateway(snapshot(terminal=True))
    runtime, traces = build_runtime(provider, gateway)

    result = runtime.decide_round(
        episode_id="episode-1",
        company_id="company_A",
        model_id=MODEL_ID,
    )

    assert result.status == "terminal"
    assert provider.calls == []
    assert gateway.submissions == []
    assert traces.items[0].status == "terminal"


def test_evidence_paths_are_normalized_and_nonexistent_references_are_removed():
    value = proposal()
    value.candidates[1].evidence_paths = [
        "/visible_context/observation/own_company/financial/cash_balance_cents",
        "/visible_context/observation/decision_context/missing",
        "own_company.financial.cash_balance_cents",
    ]
    context = {
        "observation": snapshot().observation,
        "action_contract": snapshot().action_contract,
    }

    normalized = normalize_evidence_paths(value, context)

    assert normalized.candidates[1].evidence_paths == [
        "observation.own_company.financial.cash_balance_cents"
    ]
