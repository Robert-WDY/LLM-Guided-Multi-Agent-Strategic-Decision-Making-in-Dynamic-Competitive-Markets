"""CLI for a bounded Doubao/Mock Agent episode against the local API process."""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from game_theory_agent.agents import (
    AgentRuntime,
    PersonaProfile,
    PersonaRegistry,
    load_persona_registry,
)
from game_theory_agent.agents.single.persona_agent import (
    PersonaAgent,
    SABMEpisodeRunner,
)
from game_theory_agent.agents.single.service import build_agent_lab_runtime
from game_theory_agent.model_clients import (
    DeepSeekModelClient,
    DoubaoModelClient,
    MockModelClient,
    UniformRandomModelClient,
)
from game_theory_agent.orchestration import (
    HttpAgentGatewayClient,
    HttpControllerClient,
    JsonlRoundEventLogger,
    RoundCoordinator,
)


DEFAULT_COMPANIES = ("company_A", "company_B", "company_C", "company_D")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_OPENROUTER_SECRET = PROJECT_ROOT / "secrets" / "open_router-api_key.env"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run model Agents through the protected market loop."
    )
    parser.add_argument(
        "--provider",
        choices=("openrouter", "doubao", "deepseek", "mock"),
        default=os.getenv("AGENT_PROVIDER", "openrouter"),
    )
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, choices=(5, 10, 15, 20), default=5)
    parser.add_argument(
        "--market-model",
        choices=(
            "random",
            "balanced",
            "value_oriented",
            "quality_oriented",
            "service_oriented",
        ),
        default="balanced",
    )
    parser.add_argument(
        "--information-mode", choices=("perfect", "public"), default="perfect"
    )
    parser.add_argument(
        "--communication-mode",
        choices=("off", "public_only", "public_private"),
        default="off",
    )
    parser.add_argument(
        "--agent-companies",
        default="company_A",
        help="Comma-separated companies controlled by model Agents.",
    )
    parser.add_argument(
        "--persona",
        default=os.getenv("AGENT_PERSONA", "balanced"),
        help="Default persona profile for model-controlled companies.",
    )
    parser.add_argument(
        "--persona-map",
        default="",
        help=(
            "Optional comma-separated overrides such as "
            "company_A=selfish_long_term,company_B=conservative."
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--openrouter-secret",
        type=Path,
        default=DEFAULT_OPENROUTER_SECRET,
        help="git-crypt protected OpenRouter API key file.",
    )
    parser.add_argument(
        "--sabm-output-root",
        type=Path,
        default=PROJECT_ROOT / "~outputs-intermediate" / "agent-runs",
        help="SABM checkpoints and decision traces.",
    )
    parser.add_argument(
        "--opponent-policy",
        choices=("controller-rule", "uniform-random"),
        default="controller-rule",
    )
    parser.add_argument("--opponent-seed", type=int, default=None)
    parser.add_argument("--controller-url", default="http://127.0.0.1:8010")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8011")
    parser.add_argument("--decision-timeout", type=float, default=45.0)
    parser.add_argument("--communication-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _persona_assignments(
    raw: str,
    selected: tuple[str, ...],
    registry: PersonaRegistry,
    default_persona: str,
) -> dict[str, PersonaProfile]:
    default_profile = registry.get(default_persona)
    assignments = {company_id: default_profile for company_id in selected}
    if not raw.strip():
        return assignments
    for item in raw.split(","):
        if "=" not in item:
            raise SystemExit(
                "--persona-map entries must use company_id=persona_id"
            )
        company_id, persona_id = (part.strip() for part in item.split("=", 1))
        if company_id not in assignments:
            raise SystemExit(
                "--persona-map may only override --agent-companies; "
                f"unexpected {company_id!r}"
            )
        try:
            assignments[company_id] = registry.get(persona_id)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    return assignments


def _model_name(model_client: object) -> str:
    for field in ("model", "model_name"):
        value = getattr(model_client, field, None)
        if isinstance(value, str) and value:
            return value
    return type(model_client).__name__


def _openrouter_agent_configs(
    selected: tuple[str, ...],
    persona_assignments: dict[str, PersonaProfile],
    model_id: str,
    *,
    decision_timeout: float,
) -> dict[str, dict[str, object]]:
    """生成 Controller 可审计的 PersonaAgent manifest。"""

    return {
        company_id: {
            "agent_id": f"single-agent-{company_id}",
            "provider": "openrouter",
            "model_name": model_id,
            "persona": persona_assignments[company_id].manifest_dict(),
            "decision_timeout_seconds": decision_timeout,
        }
        for company_id in selected
    }


async def _run_openrouter_episode(
    args: argparse.Namespace,
    *,
    selected: tuple[str, ...],
    persona_assignments: dict[str, PersonaProfile],
    episode_id: str,
    controller: HttpControllerClient,
) -> int:
    """通过 PersonaAgent/SABM 图运行后端回合，不扩展公共 API。"""

    model_id = args.model or DEFAULT_OPENROUTER_MODEL
    agent_configs = _openrouter_agent_configs(
        selected,
        persona_assignments,
        model_id,
        decision_timeout=args.decision_timeout,
    )
    created = await controller.create_episode(
        {
            "episode_id": episode_id,
            "episode_seed": args.seed,
            "company_ids": list(DEFAULT_COMPANIES),
            "market_model": args.market_model,
            "max_rounds": args.rounds,
            "information_mode": args.information_mode,
            "communication_mode": "off",
            "agent_configs": agent_configs,
        }
    )
    agent_tokens = created.get("agent_tokens", {})
    agents: dict[str, PersonaAgent] = {}
    for company_id in selected:
        bundle = build_agent_lab_runtime(
            episode_id=f"{episode_id}-{company_id}",
            secret_path=args.openrouter_secret,
            output_root=args.sabm_output_root,
            gateway_base_url=args.gateway_url,
            provider_timeout_seconds=args.decision_timeout,
        )
        bundle.gateway.set_agent_tokens(agent_tokens)
        agents[company_id] = PersonaAgent(
            company_id=company_id,
            model_id=model_id,
            runtime=bundle.runtime,
            persona_profile=persona_assignments[company_id],
        )

    results = await SABMEpisodeRunner(
        controller=controller,
        agents=agents,
    ).run_episode(episode_id, rounds=args.rounds)
    submitted = 0
    for result in results:
        submitted += len(result.accepted_intent_ids)
        state = result.settlement["state"]
        print(
            f"R{result.round_number:02d} settled "
            f"state_version={state['state_version']} "
            f"submitted_intents={len(result.accepted_intent_ids)}/{len(selected)} "
            f"hash={state['state_hash']}"
        )
    print(f"Episode complete: {episode_id}")
    print(f"SABM output root: {args.sabm_output_root.resolve()}")
    expected = len(results) * len(selected)
    print(f"Primary model intents accepted: {submitted}/{expected}; fallbacks: {expected - submitted}")
    return 0 if submitted else 2


async def _run(args: argparse.Namespace) -> int:
    token = os.getenv("MARKET_CONTROLLER_TOKEN")
    if not token:
        raise SystemExit("MARKET_CONTROLLER_TOKEN is not set")
    selected = tuple(
        item.strip() for item in args.agent_companies.split(",") if item.strip()
    )
    if not selected or set(selected) - set(DEFAULT_COMPANIES):
        raise SystemExit(
            "--agent-companies must contain company_A through company_D"
        )
    if args.provider == "doubao" and not os.getenv("ARK_API_KEY"):
        raise SystemExit("ARK_API_KEY is not set")
    if args.provider == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    if args.provider == "openrouter":
        if not args.openrouter_secret.is_file():
            raise SystemExit(f"OpenRouter secret file does not exist: {args.openrouter_secret}")
        if args.communication_mode != "off":
            raise SystemExit("OpenRouter SABM backend currently requires --communication-mode off")
        if args.opponent_policy != "controller-rule":
            raise SystemExit("OpenRouter SABM backend currently requires --opponent-policy controller-rule")

    registry = load_persona_registry(PROJECT_ROOT / "configs" / "market_v4.yaml")
    try:
        persona_assignments = _persona_assignments(
            args.persona_map,
            selected,
            registry,
            args.persona,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    episode_id = args.episode_id or f"agent-smoke-{uuid.uuid4()}"
    controller = HttpControllerClient(token, args.controller_url)
    if args.provider == "openrouter":
        return await _run_openrouter_episode(
            args,
            selected=selected,
            persona_assignments=persona_assignments,
            episode_id=episode_id,
            controller=controller,
        )

    gateway = HttpAgentGatewayClient(args.gateway_url)

    runtimes: dict[str, AgentRuntime] = {}
    for company_id in selected:
        if args.provider == "doubao":
            model_client = DoubaoModelClient(model=args.model)
        elif args.provider == "deepseek":
            model_client = DeepSeekModelClient(model=args.model)
        else:
            model_client = MockModelClient()
        runtimes[company_id] = AgentRuntime(
            agent_id=f"{args.provider}-planner-{company_id}",
            company_id=company_id,
            model_client=model_client,
            persona_profile=persona_assignments[company_id],
            persona_registry=registry,
        )
    if args.opponent_policy == "uniform-random":
        opponent_seed = args.seed if args.opponent_seed is None else args.opponent_seed
        for company_id in DEFAULT_COMPANIES:
            if company_id not in runtimes:
                runtimes[company_id] = AgentRuntime(
                    agent_id=f"uniform-random-{company_id}",
                    company_id=company_id,
                    model_client=UniformRandomModelClient(opponent_seed),
                    persona_profile=registry.get("none"),
                    persona_registry=registry,
                )

    agent_configs = {
        company_id: {
            "agent_id": runtime.agent_id,
            "provider": (
                args.provider if company_id in selected else "uniform-random"
            ),
            "model_name": _model_name(runtime.model_client),
            "persona": runtime.persona_manifest(),
            "decision_timeout_seconds": args.decision_timeout,
        }
        for company_id, runtime in runtimes.items()
    }
    created = await controller.create_episode(
        {
            "episode_id": episode_id,
            "episode_seed": args.seed,
            "company_ids": list(DEFAULT_COMPANIES),
            "market_model": args.market_model,
            "max_rounds": args.rounds,
            "information_mode": args.information_mode,
            "communication_mode": args.communication_mode,
            "agent_configs": agent_configs,
        }
    )
    gateway.set_agent_tokens(created.get("agent_tokens", {}))

    output = args.output or Path("runs") / episode_id / "agent-rounds.jsonl"
    coordinator = RoundCoordinator(
        controller,
        gateway,
        runtimes,
        event_logger=JsonlRoundEventLogger(output),
        decision_timeout_seconds=args.decision_timeout,
        communication_timeout_seconds=args.communication_timeout,
    )
    results = await coordinator.run_episode(episode_id)
    total_submitted = 0
    selected_submitted = 0
    for result in results:
        state = result.settlement["state"]
        submitted = sum(
            trace.decision_status == "submitted" for trace in result.event.traces
        )
        submitted_by_selected = sum(
            trace.decision_status == "submitted"
            and trace.company_id in selected
            for trace in result.event.traces
        )
        total_submitted += submitted
        selected_submitted += submitted_by_selected
        print(
            f"R{result.event.settled_round:02d} settled "
            f"state_version={state['state_version']} "
            f"submitted_intents={submitted}/{len(runtimes)} "
            f"messages={len(result.event.communication_phase.closure.all_messages) if result.event.communication_phase else 0} "
            f"hash={state['state_hash']}"
        )
    print(f"Episode complete: {episode_id}")
    print(f"RoundEvent log: {output.resolve()}")
    expected_submissions = len(results) * len(runtimes)
    expected_selected_submissions = len(results) * len(selected)
    print(
        f"All intents accepted: {total_submitted}/{expected_submissions}; "
        f"fallbacks: {expected_submissions - total_submitted}"
    )
    print(
        f"Primary model intents accepted: "
        f"{selected_submitted}/{expected_selected_submissions}; "
        f"fallbacks: {expected_selected_submissions - selected_submitted}"
    )
    if args.provider != "mock" and selected_submitted == 0:
        print(
            f"{args.provider} connectivity/decision smoke test failed; "
            "inspect RoundEvent errors."
        )
        return 2
    return 0


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
