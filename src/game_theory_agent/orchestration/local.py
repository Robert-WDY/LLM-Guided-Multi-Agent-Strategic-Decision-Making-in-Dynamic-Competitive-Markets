"""In-process Controller/Gateway clients so hosted auto-run cannot deadlock on HTTP."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from game_theory_agent.agents.contracts import AgentDecisionResult
from game_theory_agent.orchestration.clients import ApiClientError

if TYPE_CHECKING:
    from game_theory_agent.agents.contracts import AgentCommunicationResult


CONTROLLER_DRIVEN: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "market_controller_driven", default=False
)


def run_coroutine_sync(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Run an async coordinator from a sync FastAPI route or pytest."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    def _runner() -> Any:
        return asyncio.run(coroutine)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner).result()


def _raise_api_error(exc: HTTPException) -> None:
    raise ApiClientError(int(exc.status_code), exc.detail) from exc


class LocalGatewayClient:
    """Call Agent Gateway handlers in-process under controller trust."""

    async def get_observation(
        self, episode_id: str, company_id: str
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_observation, episode_id, company_id)

    def _get_observation(self, episode_id: str, company_id: str) -> dict[str, Any]:
        from game_theory_agent.api import get_agent_observation

        token = CONTROLLER_DRIVEN.set(True)
        try:
            return get_agent_observation(episode_id, company_id, agent_token=None)
        except HTTPException as exc:
            _raise_api_error(exc)
            raise
        finally:
            CONTROLLER_DRIVEN.reset(token)

    async def submit_communication(
        self, episode_id: str, result: AgentCommunicationResult
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._submit_communication, episode_id, result
        )

    def _submit_communication(
        self, episode_id: str, result: AgentCommunicationResult
    ) -> dict[str, Any]:
        from game_theory_agent.api import (
            SubmitCommunicationRequest,
            submit_agent_communication,
        )

        context = result.context
        request = SubmitCommunicationRequest(
            round=context.round,
            state_version=context.state_version,
            state_hash=context.state_hash,
            submission=result.submission,
        )
        token = CONTROLLER_DRIVEN.set(True)
        try:
            return submit_agent_communication(
                episode_id, result.company_id, request, agent_token=None
            )
        except HTTPException as exc:
            _raise_api_error(exc)
            raise
        finally:
            CONTROLLER_DRIVEN.reset(token)

    async def submit_intent(
        self, episode_id: str, result: AgentDecisionResult
    ) -> dict[str, Any]:
        if not result.success or result.decision is None:
            raise ValueError("cannot submit an unsuccessful decision")
        return await asyncio.to_thread(self._submit_intent, episode_id, result)

    def _submit_intent(
        self, episode_id: str, result: AgentDecisionResult
    ) -> dict[str, Any]:
        from game_theory_agent.api import SubmitAgentIntentRequest, submit_agent_intent

        decision = result.decision
        request = SubmitAgentIntentRequest(
            agent_id=result.agent_id,
            company_id=result.company_id,
            round=result.context.round,
            state_version=result.context.state_version,
            observation_hash=result.context.meta.observation_hash,
            requested_action=decision.requested_action,
            rationale=decision.plan.situation_summary,
            expected_outcome=json.dumps(
                decision.plan.expected_outcome.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            ),
            communication_view_digest=(
                result.context.communication_view.view_digest
                if result.context.communication_view is not None
                else None
            ),
        )
        token = CONTROLLER_DRIVEN.set(True)
        try:
            return submit_agent_intent(episode_id, request, agent_token=None)
        except HTTPException as exc:
            _raise_api_error(exc)
            raise
        finally:
            CONTROLLER_DRIVEN.reset(token)


class LocalControllerClient:
    def __init__(self, controller_token: str, actor_choices: dict[str,str] | None = None) -> None:
        self._actor_choices = dict(actor_choices or {})
        self._token = controller_token

    async def create_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("hosted coordinator does not create episodes")

    async def get_episode(self, episode_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_episode, episode_id)

    def _get_episode(self, episode_id: str) -> dict[str, Any]:
        from game_theory_agent.api import get_state

        try:
            return get_state(episode_id)
        except HTTPException as exc:
            _raise_api_error(exc)
            raise

    async def settle_agent_round(
        self,
        episode_id: str,
        step_id: str,
        intent_ids: dict[str, str],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._settle_agent_round, episode_id, step_id, intent_ids
        )

    def _settle_agent_round(
        self,
        episode_id: str,
        step_id: str,
        intent_ids: dict[str, str],
    ) -> dict[str, Any]:
        from game_theory_agent.api import (
            SettleAgentRoundRequest,
            settle_agent_round,
        )

        request = SettleAgentRoundRequest(
            step_id=step_id, intent_ids=intent_ids, fallback="rule", actor_choices=self._actor_choices
        )
        try:
            return settle_agent_round(
                episode_id, request, controller_token=self._token
            )
        except HTTPException as exc:
            _raise_api_error(exc)
            raise

    async def close_communication(
        self,
        episode_id: str,
        round_number: int,
        state_version: int,
        state_hash: str,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._close_communication,
            episode_id,
            round_number,
            state_version,
            state_hash,
        )

    def _close_communication(
        self,
        episode_id: str,
        round_number: int,
        state_version: int,
        state_hash: str,
    ) -> dict[str, Any]:
        from game_theory_agent.api import (
            CloseCommunicationRequest,
            close_agent_communication,
        )

        request = CloseCommunicationRequest(
            round=round_number,
            state_version=state_version,
            state_hash=state_hash,
        )
        try:
            return close_agent_communication(
                episode_id, request, controller_token=self._token
            )
        except HTTPException as exc:
            _raise_api_error(exc)
            raise
