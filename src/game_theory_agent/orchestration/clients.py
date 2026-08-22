"""Async clients for the public Agent plane and protected control plane."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from game_theory_agent.agents.contracts import AgentDecisionResult

if TYPE_CHECKING:
    from game_theory_agent.agents.contracts import AgentCommunicationResult


class ApiClientError(RuntimeError):
    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(f"API request failed ({status_code}): {detail}")
        self.status_code = status_code
        self.detail = detail


class AgentGatewayClient(Protocol):
    async def get_observation(
        self, episode_id: str, company_id: str
    ) -> dict[str, Any]: ...

    async def submit_intent(
        self, episode_id: str, result: AgentDecisionResult
    ) -> dict[str, Any]: ...

    async def submit_communication(
        self, episode_id: str, result: "AgentCommunicationResult"
    ) -> dict[str, Any]: ...


class ControllerClient(Protocol):
    async def create_episode(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def get_episode(self, episode_id: str) -> dict[str, Any]: ...

    async def settle_agent_round(
        self,
        episode_id: str,
        step_id: str,
        intent_ids: dict[str, str],
    ) -> dict[str, Any]: ...

    async def close_communication(
        self,
        episode_id: str,
        round_number: int,
        state_version: int,
        state_hash: str,
    ) -> dict[str, Any]: ...


class _JsonHttpClient:
    def __init__(self, base_url: str, headers: dict[str, str] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Accept": "application/json", **(headers or {})}

    async def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._request, method, path, payload, extra_headers
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {**self.headers, **(extra_headers or {})}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(raw).get("detail", raw)
            except json.JSONDecodeError:
                detail = raw
            raise ApiClientError(exc.code, detail) from exc
        except URLError as exc:
            raise ApiClientError(0, str(exc.reason)) from exc


class HttpAgentGatewayClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8011",
        agent_tokens: dict[str, str] | None = None,
    ) -> None:
        self._http = _JsonHttpClient(base_url)
        self._agent_tokens = dict(agent_tokens or {})

    def set_agent_tokens(self, agent_tokens: dict[str, str]) -> None:
        self._agent_tokens = dict(agent_tokens)

    def _agent_headers(self, company_id: str) -> dict[str, str]:
        token = self._agent_tokens.get(company_id)
        return {"X-Agent-Token": token} if token else {}

    async def get_observation(
        self, episode_id: str, company_id: str
    ) -> dict[str, Any]:
        return await self._http.request(
            "GET",
            f"/v1/episodes/{episode_id}/companies/{company_id}/observation",
            extra_headers=self._agent_headers(company_id),
        )

    async def submit_communication(
        self, episode_id: str, result: "AgentCommunicationResult"
    ) -> dict[str, Any]:
        context = result.context
        return await self._http.request(
            "POST",
            (
                f"/v1/episodes/{episode_id}/companies/"
                f"{result.company_id}/communication/submissions"
            ),
            {
                "round": context.round,
                "state_version": context.state_version,
                "state_hash": context.state_hash,
                "submission": result.submission.model_dump(mode="json"),
            },
            extra_headers=self._agent_headers(result.company_id),
        )

    async def submit_intent(
        self, episode_id: str, result: AgentDecisionResult
    ) -> dict[str, Any]:
        if not result.success or result.decision is None:
            raise ValueError("cannot submit an unsuccessful decision")
        decision = result.decision
        return await self._http.request(
            "POST",
            f"/v1/episodes/{episode_id}/intents",
            {
                "agent_id": result.agent_id,
                "company_id": result.company_id,
                "round": result.context.round,
                "state_version": result.context.state_version,
                "observation_hash": result.context.meta.observation_hash,
                "requested_action": decision.requested_action.model_dump(
                    mode="json"
                ),
                "rationale": decision.plan.situation_summary,
                "expected_outcome": json.dumps(
                    decision.plan.expected_outcome.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "communication_view_digest": (
                    result.context.communication_view.view_digest
                    if result.context.communication_view is not None
                    else None
                ),
            },
            extra_headers=self._agent_headers(result.company_id),
        )


class HttpControllerClient:
    def __init__(
        self,
        controller_token: str,
        base_url: str = "http://127.0.0.1:8010",
    ) -> None:
        self._http = _JsonHttpClient(
            base_url, {"X-Controller-Token": controller_token}
        )

    async def create_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._http.request("POST", "/api/episodes", payload)

    async def get_episode(self, episode_id: str) -> dict[str, Any]:
        return await self._http.request("GET", f"/api/episodes/{episode_id}/state")

    async def settle_agent_round(
        self,
        episode_id: str,
        step_id: str,
        intent_ids: dict[str, str],
    ) -> dict[str, Any]:
        return await self._http.request(
            "POST",
            f"/api/v1/controller/episodes/{episode_id}/settle-agent-round",
            {"step_id": step_id, "intent_ids": intent_ids, "fallback": "rule"},
        )

    async def close_communication(
        self,
        episode_id: str,
        round_number: int,
        state_version: int,
        state_hash: str,
    ) -> dict[str, Any]:
        return await self._http.request(
            "POST",
            f"/api/v1/controller/episodes/{episode_id}/communication/close",
            {
                "round": round_number,
                "state_version": state_version,
                "state_hash": state_hash,
            },
        )
