"""First-class contracts for auditable company-scoped observations."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.market.protocols import sha256_hash


OBSERVATION_HASH_PROTOCOL_VERSION = "observation-view-hash-v1.0.0"


class PublicState(BaseModel):
    """Versioned facts shared identically with every company in one round."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["public-state-v1.0.0"] = "public-state-v1.0.0"
    episode_id: str
    round: int = Field(ge=1)
    rounds_remaining: int = Field(ge=0)
    state_version: int = Field(ge=0)
    terminal: bool
    market: dict[str, Any]
    shared_resilience: dict[str, Any] | None = None
    risk_signals: list[dict[str, Any]] = Field(default_factory=list)
    active_market_events: list[dict[str, Any]] = Field(default_factory=list)
    companies: list[dict[str, Any]] = Field(default_factory=list)


class PrivateState(BaseModel):
    """The complete state of exactly one observing company."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["private-state-v1.0.0"] = "private-state-v1.0.0"
    company_id: str
    company: dict[str, Any]

    @model_validator(mode="after")
    def validate_company_scope(self) -> "PrivateState":
        if self.company.get("company_id") != self.company_id:
            raise ValueError("private state contains another company")
        return self


class ObservationEnvelope(BaseModel):
    """Strict company-scoped payload delivered by the Agent Gateway.

    Flexible economic sub-documents remain dictionaries because they are already
    versioned by the market protocol.  The envelope itself is closed, which
    prevents a new top-level field from bypassing the visibility policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_schema_version: Literal["agent-observation-v1.8.0"] = (
        "agent-observation-v1.8.0"
    )
    episode_id: str
    round: int = Field(ge=1)
    decision_round: int | None = Field(default=None, ge=1)
    last_settled_round: int = Field(ge=0)
    rounds_remaining: int = Field(ge=0)
    state_version: int = Field(ge=0)
    state_hash: str
    terminal: bool
    episode_config: dict[str, Any]
    information_mode: Literal["perfect", "public"]
    visibility_policy_version: str
    visibility_policy: dict[str, Any]
    belief_schema_version: str = "none"
    belief_hash: str | None = None
    belief_state: dict[str, Any] | None = None
    opponent_model_hash: str | None = None
    opponent_model_state: dict[str, Any] | None = None
    utility_inference_hash: str | None = None
    utility_inference_state: dict[str, Any] | None = None
    public_state: PublicState
    private_state: PrivateState
    communication_mode: Literal["off", "public_only", "public_private"] = "off"
    cooperation_mode: Literal["off", "shared_resilience_v1"] = "off"
    market: dict[str, Any]
    shared_resilience: dict[str, Any] | None = None
    market_regime: dict[str, Any]
    decision_support: dict[str, Any]
    risk_signals: list[dict[str, Any]] = Field(default_factory=list)
    active_market_events: list[dict[str, Any]] = Field(default_factory=list)
    public_companies: list[dict[str, Any]] = Field(default_factory=list)
    competitors: list[dict[str, Any]] = Field(default_factory=list)
    public_history: list[dict[str, Any]] = Field(default_factory=list)
    own_company: dict[str, Any]
    company_analysis: dict[str, Any]
    action_constraints: dict[str, Any]
    communication_view: dict[str, Any] | None = None
    communication_history: list[dict[str, Any]] = Field(default_factory=list)
    cooperation: dict[str, Any] | None = None
    repeated_game_strategy_hash: str | None = None
    repeated_game_strategy: dict[str, Any] | None = None
    terminal_summary: dict[str, Any] | None = None
    game_theory_advice: dict[str, Any] | None = None
    observation_hash: str

    @model_validator(mode="after")
    def validate_bindings(self) -> "ObservationEnvelope":
        if self.public_state.episode_id != self.episode_id:
            raise ValueError("public state episode binding mismatch")
        if self.public_state.round != self.round:
            raise ValueError("public state round binding mismatch")
        if self.public_state.state_version != self.state_version:
            raise ValueError("public state version binding mismatch")
        if self.private_state.company_id != self.own_company.get("company_id"):
            raise ValueError("private and own company scope mismatch")
        if self.private_state.company != self.own_company:
            raise ValueError("private state and own company differ")
        return self


def compute_observation_hash(observation: Mapping[str, Any]) -> str:
    payload = dict(observation)
    payload.pop("observation_hash", None)
    return sha256_hash(
        {
            "hash_protocol_version": OBSERVATION_HASH_PROTOCOL_VERSION,
            "visibility_policy_version": payload.get(
                "visibility_policy_version"
            ),
            "belief_schema_version": payload.get("belief_schema_version"),
            "observation": payload,
        }
    )


def seal_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(observation)
    sealed.pop("observation_hash", None)
    sealed["observation_hash"] = compute_observation_hash(sealed)
    if sealed.get("observation_schema_version") == "agent-observation-v1.8.0":
        # The Gateway contract is closed.  Builder-only projections deliberately
        # omit the outer envelope and are validated when the API assembles it.
        sealed = ObservationEnvelope.model_validate(sealed).model_dump(
            mode="json", exclude_unset=True
        )
    return sealed


class ObservationSnapshot(BaseModel):
    """Exact decision-time view plus its TrueState/policy binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_schema_version: Literal["observation-snapshot-v1.0.0"] = (
        "observation-snapshot-v1.0.0"
    )
    hash_protocol_version: Literal["observation-view-hash-v1.0.0"] = (
        OBSERVATION_HASH_PROTOCOL_VERSION
    )
    episode_id: str
    round: int
    state_version: int
    state_hash: str
    company_id: str
    information_mode: Literal["perfect", "public"]
    visibility_policy_version: str
    belief_schema_version: str = "none"
    belief_hash: str | None = None
    observation_hash: str
    observation: dict[str, Any]

    @model_validator(mode="after")
    def validate_bindings(self) -> "ObservationSnapshot":
        observation = self.observation
        bindings = {
            "episode_id": self.episode_id,
            "round": self.round,
            "state_version": self.state_version,
            "state_hash": self.state_hash,
            "information_mode": self.information_mode,
            "visibility_policy_version": self.visibility_policy_version,
            "belief_schema_version": self.belief_schema_version,
            "belief_hash": self.belief_hash,
            "observation_hash": self.observation_hash,
        }
        for field_name, expected in bindings.items():
            if observation.get(field_name) != expected:
                raise ValueError(
                    f"observation snapshot {field_name} binding mismatch"
                )
        private_state = observation.get("private_state")
        if not isinstance(private_state, dict):
            raise ValueError("observation snapshot private_state is missing")
        if private_state.get("company_id") != self.company_id:
            raise ValueError("observation snapshot company binding mismatch")
        expected_hash = compute_observation_hash(observation)
        if expected_hash != self.observation_hash:
            raise ValueError("observation snapshot hash mismatch")
        if self.belief_schema_version == "none":
            if observation.get("belief_state") is not None:
                raise ValueError("belief_state must be null before Belief MVP")
            if self.belief_hash is not None:
                raise ValueError("belief_hash must be null before Belief MVP")
        else:
            if not isinstance(observation.get("belief_state"), dict):
                raise ValueError("enabled belief_state must be an object")
            if not self.belief_hash:
                raise ValueError("enabled belief_state requires belief_hash")
            expected_belief_hash = sha256_hash(
                {
                    "hash_protocol_version": "belief-view-hash-v1.0.0",
                    "belief_schema_version": observation["belief_state"].get(
                        "belief_schema_version"
                    ),
                    "belief_state": observation["belief_state"],
                }
            )
            if expected_belief_hash != self.belief_hash:
                raise ValueError("belief_state hash mismatch")
        return self

    @classmethod
    def from_observation(
        cls, observation: Mapping[str, Any], company_id: str
    ) -> "ObservationSnapshot":
        payload = dict(observation)
        return cls(
            episode_id=str(payload["episode_id"]),
            round=int(payload["round"]),
            state_version=int(payload["state_version"]),
            state_hash=str(payload["state_hash"]),
            company_id=company_id,
            information_mode=str(payload["information_mode"]),
            visibility_policy_version=str(
                payload["visibility_policy_version"]
            ),
            belief_schema_version=str(payload["belief_schema_version"]),
            belief_hash=(
                str(payload["belief_hash"])
                if payload.get("belief_hash") is not None
                else None
            ),
            observation_hash=str(payload["observation_hash"]),
            observation=payload,
        )
