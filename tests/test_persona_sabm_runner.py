"""PersonaAgent 与 SABM 后端轮次执行器的最小接入测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from game_theory_agent.agents import load_persona_registry
from game_theory_agent.agents.single import PersonaAgent, SABMEpisodeRunner
from game_theory_agent.run_agents import (
    DEFAULT_OPENROUTER_SECRET,
    PROJECT_ROOT,
    _openrouter_agent_configs,
    _parser,
)


class RecordingRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def decide_round(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(status="accepted", intent_id="intent-A")


def test_persona_agent_binds_persona_prompt_and_runtime_identity():
    registry = load_persona_registry(PROJECT_ROOT / "configs" / "market_v4.yaml")
    profile = registry.get("balanced")
    runtime = RecordingRuntime()
    agent = PersonaAgent(
        company_id="company_A",
        model_id="nvidia/nemotron-nano-9b-v2:free",
        runtime=runtime,
        persona_profile=profile,
    )

    result = agent.decide_round("episode-1")

    assert result.intent_id == "intent-A"
    assert runtime.calls[0]["company_id"] == "company_A"
    assert runtime.calls[0]["model_id"] == "nvidia/nemotron-nano-9b-v2:free"
    assert runtime.calls[0]["persona_manifest"]["persona_id"] == "balanced"
    assert runtime.calls[0]["persona_manifest"]["profile_hash"] == profile.profile_hash
    prompt = runtime.calls[0]["prompt_template"]
    assert profile.label in prompt.system_prompt
    assert profile.objective in prompt.system_prompt
    assert agent.manifest()["agent_id"] == "single-agent-company_A"
    assert agent.manifest()["persona"]["persona_id"] == "balanced"


class SequenceAgent:
    def __init__(self, company_id: str, results: list[object]) -> None:
        self.company_id = company_id
        self._results = iter(results)

    def decide_round(self, episode_id: str):
        return next(self._results)


class RecordingController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    async def get_episode(self, episode_id):
        return {
            "state": {
                "episode_id": episode_id,
                "round": len(self.calls) + 1,
                "state_version": len(self.calls),
                "terminal": False,
            }
        }

    async def settle_agent_round(self, episode_id, step_id, intent_ids):
        self.calls.append((episode_id, step_id, intent_ids))
        return {"state": {"round": len(self.calls) + 1, "terminal": False}}


def test_sabm_runner_submits_only_accepted_intents_and_leaves_rule_fallbacks():
    controller = RecordingController()
    runner = SABMEpisodeRunner(
        controller=controller,
        agents={
            "company_A": SequenceAgent(
                "company_A",
                [
                    SimpleNamespace(status="accepted", intent_id="intent-A1"),
                    SimpleNamespace(status="no_intent", intent_id=None),
                ],
            ),
            "company_B": SequenceAgent(
                "company_B",
                [
                    SimpleNamespace(status="no_intent", intent_id=None),
                    SimpleNamespace(status="accepted", intent_id="intent-B2"),
                ],
            ),
        },
    )

    settlements = asyncio.run(runner.run_episode("episode-1", rounds=2))

    assert len(settlements) == 2
    assert controller.calls == [
        ("episode-1", "episode-1:1:0", {"company_A": "intent-A1"}),
        ("episode-1", "episode-1:2:1", {"company_B": "intent-B2"}),
    ]


def test_run_agents_defaults_to_openrouter_sabm_backend(monkeypatch):
    monkeypatch.delenv("AGENT_PROVIDER", raising=False)

    args = _parser().parse_args([])

    assert args.provider == "openrouter"
    assert args.openrouter_secret == DEFAULT_OPENROUTER_SECRET
    assert args.openrouter_secret == PROJECT_ROOT / "secrets" / "open_router-api_key.env"


def test_openrouter_episode_manifest_uses_persona_agent_identity():
    registry = load_persona_registry(PROJECT_ROOT / "configs" / "market_v4.yaml")

    configs = _openrouter_agent_configs(
        ("company_A",),
        {"company_A": registry.get("balanced")},
        "nvidia/nemotron-nano-9b-v2:free",
        decision_timeout=45.0,
    )

    assert configs["company_A"]["agent_id"] == "single-agent-company_A"
    assert configs["company_A"]["provider"] == "openrouter"
    assert configs["company_A"]["model_name"] == "nvidia/nemotron-nano-9b-v2:free"
    assert configs["company_A"]["persona"]["persona_id"] == "balanced"
