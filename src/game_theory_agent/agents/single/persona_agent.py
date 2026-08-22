"""PersonaAgent 类与 SABM 多公司节点流轮次执行器。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from game_theory_agent.agents.personas import PersonaProfile

from .models import DEFAULT_SYSTEM_PROMPT, PromptTemplate


@dataclass(slots=True)
class PersonaAgent:
    """把 Persona 身份、OpenRouter 模型与单公司 SABM 图绑定在一起。"""

    company_id: str
    model_id: str
    runtime: Any
    persona_profile: PersonaProfile

    @property
    def agent_id(self) -> str:
        return f"single-agent-{self.company_id}"

    def manifest(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "provider": "openrouter",
            "model_name": self.model_id,
            "persona": self.persona_profile.manifest_dict(),
        }

    def decide_round(self, episode_id: str, *, progress_callback: Any = None) -> Any:
        return self.runtime.decide_round(
            episode_id=episode_id,
            company_id=self.company_id,
            model_id=self.model_id,
            persona_manifest=self.persona_profile.manifest_dict(),
            prompt_template=self._prompt_template(),
            progress_callback=progress_callback,
        )

    def _prompt_template(self) -> PromptTemplate:
        persona = self.persona_profile.manifest_dict()
        persona_json = json.dumps(persona, ensure_ascii=False, sort_keys=True)
        return PromptTemplate(
            system_prompt=(
                f"{DEFAULT_SYSTEM_PROMPT}\n\n"
                f"当前 Persona：{self.persona_profile.label}。"
                f"经营目标：{self.persona_profile.objective}。"
                "必须在市场事实和动作约束内体现该 Persona；Persona 配置如下："
                f"{persona_json}"
            )
        )


@dataclass(frozen=True, slots=True)
class SABMRoundResult:
    round_number: int
    accepted_intent_ids: dict[str, str]
    decisions: dict[str, Any]
    settlement: dict[str, Any]


class SABMEpisodeRunner:
    """并发运行各 PersonaAgent，并把缺失 intent 留给 Controller 规则回退。"""

    def __init__(self, *, controller: Any, agents: dict[str, PersonaAgent]) -> None:
        self.controller = controller
        self.agents = dict(agents)

    async def run_episode(
        self, episode_id: str, *, rounds: int
    ) -> list[SABMRoundResult]:
        results: list[SABMRoundResult] = []
        for _ in range(rounds):
            episode = await self.controller.get_episode(episode_id)
            state = episode["state"]
            if bool(state.get("terminal")):
                break
            round_number = int(state["round"])
            state_version = int(state["state_version"])
            decisions = await self._decide_round(episode_id)
            intent_ids = {
                company_id: str(decision.intent_id)
                for company_id, decision in decisions.items()
                if not isinstance(decision, Exception)
                and getattr(decision, "status", None) == "accepted"
                and getattr(decision, "intent_id", None)
            }
            settlement = await self.controller.settle_agent_round(
                episode_id,
                f"{episode_id}:{round_number}:{state_version}",
                intent_ids,
            )
            results.append(
                SABMRoundResult(
                    round_number=round_number,
                    accepted_intent_ids=intent_ids,
                    decisions=decisions,
                    settlement=settlement,
                )
            )
            if bool(settlement.get("state", {}).get("terminal")):
                break
        return results

    async def _decide_round(self, episode_id: str) -> dict[str, Any]:
        company_ids = tuple(self.agents)
        outcomes = await asyncio.gather(
            *(
                asyncio.to_thread(self.agents[company_id].decide_round, episode_id)
                for company_id in company_ids
            ),
            return_exceptions=True,
        )
        return dict(zip(company_ids, outcomes, strict=True))
