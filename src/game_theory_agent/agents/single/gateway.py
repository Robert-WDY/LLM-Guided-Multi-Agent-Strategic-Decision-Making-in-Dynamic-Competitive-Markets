"""HTTP client for the untrusted Agent Gateway surface on port 8011."""

from __future__ import annotations

import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .models import EconomicAction


class GatewayError(RuntimeError):
    pass


class GatewayReadError(GatewayError):
    pass


class StaleObservationError(GatewayError):
    pass


class SubmissionUnknownError(GatewayError):
    pass


class GatewaySubmissionError(GatewayError):
    pass


class GatewaySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    company_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    state_hash: str = Field(min_length=1)
    observation: dict[str, Any]
    action_contract: dict[str, Any]


class AgentGatewayClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8011",
        client: httpx.Client | None = None,
        timeout_seconds: float = 10.0,
        read_retries: int = 2,
        retry_delay_seconds: float = 0.1,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._read_retries = max(0, read_retries)
        self._retry_delay_seconds = max(0.0, retry_delay_seconds)
        self._agent_tokens: dict[str, str] = {}

    def set_agent_tokens(self, tokens: dict[str, str]) -> None:
        """Install one-time Controller credentials without exposing them in traces."""

        self._agent_tokens = dict(tokens)

    def _headers(self, company_id: str) -> dict[str, str]:
        token = self._agent_tokens.get(company_id)
        return {"X-Agent-Token": token} if token else {}

    def load_snapshot(self, episode_id: str, company_id: str) -> GatewaySnapshot:
        prefix = f"/v1/episodes/{episode_id}/companies/{company_id}"
        observation = self._get_json(
            f"{prefix}/observation", company_id=company_id
        )
        action_contract = self._get_json(
            f"{prefix}/action-contract", company_id=company_id
        )
        observation_key = (
            observation.get("episode_id"),
            observation.get("company_id", company_id),
            observation.get("round"),
            observation.get("state_version"),
        )
        contract_key = (
            action_contract.get("episode_id", episode_id),
            action_contract.get("company_id", company_id),
            action_contract.get("round"),
            action_contract.get("state_version"),
        )
        if observation_key != contract_key:
            raise GatewayReadError("Gateway returned mismatched snapshot versions")
        try:
            return GatewaySnapshot(
                episode_id=str(observation_key[0]),
                company_id=str(observation_key[1]),
                round=int(observation_key[2]),
                state_version=int(observation_key[3]),
                state_hash=str(observation.get("state_hash") or ""),
                observation=observation,
                action_contract=action_contract,
            )
        except (TypeError, ValueError) as exc:
            raise GatewayReadError("Gateway returned an invalid snapshot") from exc

    def submit_intent(
        self,
        *,
        episode_id: str,
        company_id: str,
        agent_id: str,
        round_number: int,
        state_version: int,
        observation_hash: str,
        action: EconomicAction,
        rationale: str,
        expected_outcome: str,
    ) -> dict[str, Any]:
        payload = {
            "agent_id": agent_id,
            "company_id": company_id,
            "round": round_number,
            "state_version": state_version,
            "observation_hash": observation_hash,
            "requested_action": action.model_dump(mode="json"),
            "rationale": rationale,
            "expected_outcome": expected_outcome,
        }
        try:
            response = self._client.post(
                self._url(f"/v1/episodes/{episode_id}/intents"),
                json=payload,
                headers=self._headers(company_id),
            )
        except httpx.HTTPError as exc:
            raise SubmissionUnknownError("intent submission result is unknown") from exc
        if response.status_code == 409:
            raise StaleObservationError("observation became stale before submission")
        if response.status_code != 202:
            raise GatewaySubmissionError(
                f"Agent Gateway rejected intent with status {response.status_code}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise SubmissionUnknownError("intent receipt was not valid JSON") from exc
        if not isinstance(body, dict) or not body.get("intent_id"):
            raise SubmissionUnknownError("intent receipt was incomplete")
        return body

    def _get_json(self, path: str, *, company_id: str) -> dict[str, Any]:
        last_status: int | None = None
        for attempt in range(self._read_retries + 1):
            try:
                response = self._client.get(
                    self._url(path), headers=self._headers(company_id)
                )
                last_status = response.status_code
                if response.status_code == 200:
                    body = response.json()
                    if isinstance(body, dict):
                        return body
                    raise ValueError("response is not an object")
                if response.status_code < 500:
                    break
            except (httpx.HTTPError, ValueError):
                pass
            if attempt < self._read_retries and self._retry_delay_seconds:
                time.sleep(self._retry_delay_seconds)
        suffix = f" ({last_status})" if last_status is not None else ""
        raise GatewayReadError(f"Agent Gateway read failed{suffix}")

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"
