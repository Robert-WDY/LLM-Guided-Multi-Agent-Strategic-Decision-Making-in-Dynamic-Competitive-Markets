"""Node implementations for the bounded single-agent decision workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .context import build_decision_context
from .gateway import GatewayError, StaleObservationError, SubmissionUnknownError
from .models import DecisionProposal, DecisionTrace, IntentDraft
from .provider import (
    ProviderError,
    ProviderInvalidDecisionError,
    ProviderResponseError,
    build_decision_prompts,
)


@dataclass(slots=True)
class SingleAgentNodes:
    deps: Any
    validate_action: Callable[[DecisionProposal, dict[str, Any]], list[str]]
    normalize_evidence_paths: Callable[[DecisionProposal, dict[str, Any]], DecisionProposal]

    def _report(self, stage: str, details: dict[str, Any] | None = None) -> None:
        progress = getattr(self.deps, "progress", None)
        if progress:
            progress.report(stage, details)

    def _cancelled(self) -> bool:
        progress = getattr(self.deps, "progress", None)
        return bool(progress and progress.cancelled())

    def load_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        self._report("load_snapshot")
        if self._cancelled():
            return {"status": "no_intent", "error_code": "job_cancelled"}
        try:
            snapshot = self.deps.gateway.load_snapshot(state["episode_id"], state["company_id"])
        except GatewayError:
            return {"status": "no_intent", "error_code": "gateway_read_failed"}
        if snapshot.observation.get("terminal"):
            return {"snapshot": snapshot, "status": "terminal"}
        return {"snapshot": snapshot, "status": "running"}

    def build_context(self, state: dict[str, Any]) -> dict[str, Any]:
        self._report("build_context")
        snapshot = state["snapshot"]
        prior_traces = self.deps.trace_store.read_company_before_round(
            snapshot.episode_id,
            snapshot.company_id,
            snapshot.round,
            limit=self.deps.history_limit,
        )
        decision_context = build_decision_context(
            snapshot,
            prior_traces,
            history_limit=self.deps.history_limit,
        )
        return {"decision_context": decision_context}

    def reflect_strategy(self, state: dict[str, Any]) -> dict[str, Any]:
        self._report("reflect_strategy")
        return {"strategy_reflection": state["decision_context"].reflection}

    def generate_candidates(self, state: dict[str, Any]) -> dict[str, Any]:
        self._report("generate_candidates")
        return self._call_provider(state, repair_errors=[])

    def validate(self, state: dict[str, Any]) -> dict[str, Any]:
        self._report("validate", {"repair_attempts": state.get("repair_attempts", 0)})
        errors = self.validate_action(
            state["proposal"],
            state["snapshot"].action_contract,
        )
        if errors and state.get("repair_attempts", 0) >= 1:
            return {"validation_errors": errors, "status": "no_intent"}
        return {"validation_errors": errors}

    def repair_decision(self, state: dict[str, Any]) -> dict[str, Any]:
        self._report("repair_decision", {"repair_attempts": 1})
        if state.get("repair_attempts", 0) >= 1:
            return {"status": "no_intent"}
        return self._call_provider(
            state,
            repair_errors=state.get("validation_errors", []),
            repair_attempts=state.get("repair_attempts", 0) + 1,
        )

    def prepare_intent(self, state: dict[str, Any]) -> dict[str, Any]:
        self._report("prepare_intent")
        if self._cancelled():
            return {"status": "no_intent", "error_code": "job_cancelled"}
        proposal = state["proposal"]
        selected = proposal.selected_candidate
        draft = IntentDraft(
            snapshot_key=state["decision_context"].snapshot_key,
            agent_id=f"single-agent-{state['company_id']}",
            action=selected.action,
            rationale="；".join(selected.tradeoffs or proposal.selection_reason_codes)[:500],
            expected_outcome=selected.expected_outcome,
        )
        return {"prepared_intent": draft}

    def submit_intent(self, state: dict[str, Any]) -> dict[str, Any]:
        self._report("submit_intent")
        if self._cancelled():
            return {"status": "no_intent", "error_code": "job_cancelled"}
        snapshot = state["snapshot"]
        draft = state["prepared_intent"]
        if draft.snapshot_key != state["decision_context"].snapshot_key:
            return {"status": "no_intent", "error_code": "stale_intent_draft"}
        try:
            receipt = self.deps.gateway.submit_intent(
                episode_id=state["episode_id"],
                company_id=state["company_id"],
                agent_id=draft.agent_id,
                round_number=snapshot.round,
                state_version=snapshot.state_version,
                observation_hash=str(
                    snapshot.observation.get("observation_hash")
                    or snapshot.state_hash
                ),
                action=draft.action,
                rationale=draft.rationale,
                expected_outcome=draft.expected_outcome,
            )
        except StaleObservationError:
            return {"status": "stale", "error_code": "stale_observation"}
        except SubmissionUnknownError:
            return {"status": "submission_unknown", "error_code": "submission_unknown"}
        except GatewayError:
            return {"status": "no_intent", "error_code": "submission_rejected"}
        return {"intent_receipt": receipt, "status": "accepted"}

    def finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        snapshot = state.get("snapshot")
        proposal = state.get("proposal")
        provider_result = state.get("provider_result")
        prior_usage = dict(state.get("provider_attempt_usage", {}))
        provider_usage = (
            provider_result.usage.model_dump(mode="python") if provider_result else {}
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            provider_usage[key] = int(provider_usage.get(key, 0)) + int(prior_usage.get(key, 0))
        status = state.get("status", "no_intent")
        trace = DecisionTrace(
            episode_id=state["episode_id"],
            company_id=state["company_id"],
            round=snapshot.round if snapshot else 1,
            state_version=snapshot.state_version if snapshot else 0,
            status=status,
            model_id=provider_result.model_id if provider_result else state["model_id"],
            persona_manifest=state.get("persona_manifest") or None,
            candidates=proposal.candidates if proposal else [],
            selected_candidate_id=proposal.selected_candidate_id if proposal else None,
            selection_reason_codes=proposal.selection_reason_codes if proposal else [],
            validation_errors=state.get("validation_errors", []),
            repair_attempts=state.get("repair_attempts", 0),
            provider_usage=provider_usage,
            latency_ms=(provider_result.latency_ms if provider_result else 0) + int(state.get("provider_attempt_latency_ms", 0)),
            provider_finish_reason=(provider_result.finish_reason if provider_result else None) or state.get("provider_finish_reason"),
            provider_error_category=state.get("provider_error_category"),
            provider_usage_available=(provider_result.usage_available if provider_result else False) or bool(state.get("provider_usage_available")),
            intent_receipt=state.get("intent_receipt"),
            memory_view=(
                state["decision_context"].memory if state.get("decision_context") else None
            ),
            strategy_reflection=state.get("strategy_reflection"),
            prepared_intent=state.get("prepared_intent"),
            prompt_audit=state.get("prompt_audit"),
            error_code=state.get("error_code"),
        )
        self.deps.trace_store.append(trace)
        self._report(
            "finalize",
            {
                "status": status,
                "repair_attempts": trace.repair_attempts,
                "total_tokens": trace.provider_usage.get("total_tokens", 0),
                "latency_ms": trace.latency_ms,
            },
        )
        return {"trace": trace, "status": status}

    def _call_provider(
        self,
        state: dict[str, Any],
        *,
        repair_errors: list[str],
        repair_attempts: int | None = None,
    ) -> dict[str, Any]:
        decision_context = state["decision_context"]
        attempt_number = (repair_attempts if repair_attempts is not None else state.get("repair_attempts", 0)) + 1
        prompt_audit = build_decision_prompts(
            decision_context,
            repair_errors,
            state.get("prompt_template"),
        )
        self._report(
            "provider_request",
            {
                "attempt": attempt_number,
                "model_id": state["model_id"],
                "repair": bool(repair_errors),
            },
        )

        def failure_update(exc: ProviderError) -> dict[str, Any]:
            prior = dict(state.get("provider_attempt_usage", {}))
            usage = getattr(exc, "usage", None)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                prior[key] = int(prior.get(key, 0)) + int(getattr(usage, key, 0) if usage else 0)
            return {
                "provider_attempt_usage": prior,
                "provider_attempt_latency_ms": int(state.get("provider_attempt_latency_ms", 0)) + int(getattr(exc, "latency_ms", 0)),
                "provider_finish_reason": getattr(exc, "finish_reason", None),
                "provider_error_category": getattr(exc, "code", "provider_error"),
                "provider_usage_available": bool(state.get("provider_usage_available")) or bool(usage and usage.total_tokens),
            }
        try:
            result = self.deps.provider.generate_decision(
                model_id=state["model_id"],
                context=decision_context,
                repair_errors=repair_errors,
                prompt_template=state.get("prompt_template"),
            )
        except ProviderInvalidDecisionError as exc:
            attempt = failure_update(exc)
            self._report(
                "provider_error",
                {
                    "attempt": attempt_number,
                    "error_category": exc.code,
                    "finish_reason": exc.finish_reason,
                    "usage_available": bool(exc.usage.total_tokens),
                    "total_tokens": exc.usage.total_tokens,
                    "latency_ms": exc.latency_ms,
                },
            )
            if repair_attempts is None and state.get("repair_attempts", 0) < 1:
                return {
                    "validation_errors": [getattr(exc, "code", "provider_output_invalid")],
                    "prompt_audit": prompt_audit,
                    **attempt,
                }
            return {
                "repair_attempts": repair_attempts or state.get("repair_attempts", 0),
                "status": "no_intent",
                "error_code": "repair_failed" if repair_attempts else "provider_failed",
                "prompt_audit": prompt_audit,
                **attempt,
            }
        except ProviderResponseError as exc:
            attempt = failure_update(exc)
            self._report(
                "provider_error",
                {
                    "attempt": attempt_number,
                    "error_category": exc.code,
                    "finish_reason": exc.finish_reason,
                    "usage_available": bool(exc.usage.total_tokens),
                    "total_tokens": exc.usage.total_tokens,
                    "latency_ms": exc.latency_ms,
                },
            )
            if repair_attempts is None and state.get("repair_attempts", 0) < 1:
                return {
                    "validation_errors": ["provider_request_failed"],
                    "prompt_audit": prompt_audit,
                    **attempt,
                }
            return {
                "repair_attempts": repair_attempts or state.get("repair_attempts", 0),
                "status": "no_intent",
                "error_code": "provider_failed",
                "prompt_audit": prompt_audit,
                **attempt,
            }
        except ProviderError as exc:
            attempt = failure_update(exc)
            self._report(
                "provider_error",
                {
                    "attempt": attempt_number,
                    "error_category": exc.code,
                    "finish_reason": exc.finish_reason,
                    "usage_available": bool(exc.usage.total_tokens),
                    "total_tokens": exc.usage.total_tokens,
                    "latency_ms": exc.latency_ms,
                },
            )
            if repair_attempts is None and state.get("repair_attempts", 0) < 1:
                return {
                    "validation_errors": ["provider_output_invalid"],
                    "prompt_audit": prompt_audit,
                    **attempt,
                }
            return {
                "repair_attempts": repair_attempts or state.get("repair_attempts", 0),
                "status": "no_intent",
                "error_code": "repair_failed" if repair_attempts else "provider_failed",
                "prompt_audit": prompt_audit,
                **attempt,
            }
        self._report(
            "provider_response",
            {
                "attempt": attempt_number,
                "status": "received",
                "finish_reason": result.finish_reason,
                "usage_available": result.usage_available,
                "total_tokens": result.usage.total_tokens,
                "latency_ms": result.latency_ms,
            },
        )
        normalized = self.normalize_evidence_paths(
            result.proposal,
            decision_context.model_dump(mode="python"),
        )
        if self._cancelled():
            return {
                "status": "no_intent",
                "error_code": "job_cancelled",
                "prompt_audit": prompt_audit,
            }
        update: dict[str, Any] = {
            "provider_result": result,
            "proposal": normalized,
            "prompt_audit": prompt_audit,
        }
        if repair_attempts is not None:
            update["repair_attempts"] = repair_attempts
        return update


def build_single_agent_nodes(
    deps: Any,
    *,
    validate_action: Callable[[DecisionProposal, dict[str, Any]], list[str]],
    normalize_evidence_paths: Callable[[DecisionProposal, dict[str, Any]], DecisionProposal],
) -> SingleAgentNodes:
    return SingleAgentNodes(
        deps=deps,
        validate_action=validate_action,
        normalize_evidence_paths=normalize_evidence_paths,
    )
