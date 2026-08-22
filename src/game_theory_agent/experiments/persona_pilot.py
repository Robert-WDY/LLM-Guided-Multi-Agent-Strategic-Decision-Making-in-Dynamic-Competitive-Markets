"""Small frozen-context experiment for testing Persona effects on decisions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from dotenv import load_dotenv

from game_theory_agent.agents import (
    AgentRuntime,
    EpisodeMemory,
    MarketRegimeEvaluator,
    ObservationBuilder,
    PersonaProfile,
    load_persona_registry,
)
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.economics import decision_support_metrics
from game_theory_agent.market import (
    CompanyIncident,
    MarketConfig,
    MarketEnv,
    MarketState,
    load_market_config,
)
from game_theory_agent.market.protocols import sha256_hash, state_hash
from game_theory_agent.information import seal_observation
from game_theory_agent.model_clients import (
    DeepSeekModelClient,
    DoubaoModelClient,
    MockModelClient,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PERSONAS = (
    "balanced",
    "aggressive",
    "conservative",
    "selfish_long_term",
    "profit_myopic",
)
DEFAULT_SCENARIOS = ("normal", "risk_warning", "financial_stress")
SUPPORTED_SCENARIOS = DEFAULT_SCENARIOS + ("capacity_pressure", "active_incident")
DEFAULT_ABLATIONS = ("full",)
SUPPORTED_ABLATIONS = (
    "full",
    "label_only",
    "objective_only",
    "weights_only",
    "traits_only",
)
ACTION_FIELDS = (
    "price_cents",
    "advertising_budget_cents",
    "service_budget_cents",
    "capacity_investment_cents",
    "resilience_budget_cents",
)


@dataclass(frozen=True, slots=True)
class FrozenScenario:
    scenario_id: str
    scenario_class: str
    market_seed: int
    state: MarketState
    observation: dict[str, Any]


def _rehash(state: MarketState) -> MarketState:
    without_hash = replace(state, state_hash="")
    return replace(without_hash, state_hash=state_hash(without_hash.to_dict()))


def _observation(
    config: MarketConfig,
    state: MarketState,
    company_id: str = "company_A",
) -> dict[str, Any]:
    env = MarketEnv(config)
    env.load_state(state)
    views = ObservationBuilder().build(state, company_id, "perfect")
    observation = {
        "observation_schema_version": "agent-observation-v1.8.0",
        "episode_id": state.episode_id,
        "round": state.round,
        "decision_round": None if state.terminal else state.round,
        "last_settled_round": min(state.state_version, state.max_rounds),
        "rounds_remaining": state.rounds_remaining,
        "state_version": state.state_version,
        "state_hash": state.state_hash,
        "terminal": state.terminal,
        "episode_config": {
            "max_rounds": state.max_rounds,
            "market_model_id": state.market.market_model_id,
            "information_mode": "perfect",
            "communication_mode": "off",
            "cooperation_mode": "off",
            "belief_mode": "off",
            "advisor_mode": "off",
        },
        **views,
        "communication_mode": "off",
        "cooperation_mode": "off",
        "market_regime": MarketRegimeEvaluator(config).evaluate(state),
        "decision_support": decision_support_metrics(config, state, company_id),
        "action_constraints": env.get_action_constraints(
            company_id, state.state_version
        ),
        "company_analysis": {},
    }
    return seal_observation(observation)


def _normal_state(config: MarketConfig, seed: int = 42) -> MarketState:
    return MarketEnv(config).reset(
        episode_id=(
            "persona-pilot-normal"
            if seed == 42
            else f"persona-pilot-normal-{seed}"
        ),
        episode_seed=seed,
        market_model="balanced",
        max_rounds=5,
    )


def _risk_state(config: MarketConfig, seed: int = 0) -> MarketState:
    for candidate_seed in range(seed, seed + 2_000):
        state = MarketEnv(config).reset(
            episode_id=f"persona-pilot-risk-{candidate_seed}",
            episode_seed=candidate_seed,
            market_model="balanced",
            max_rounds=5,
        )
        if state.risk_signals:
            return state
    raise RuntimeError("unable to find an initial risk-warning scenario")


def _financial_stress_state(config: MarketConfig, seed: int = 42) -> MarketState:
    state = _normal_state(config, seed)
    company = state.company("company_A")
    stressed = replace(
        company,
        financial=replace(
            company.financial,
            cash_balance_cents=12_000_000,
            round_profit_cents=-6_000_000,
            cumulative_profit_cents=-18_000_000,
        ),
        history=replace(
            company.history,
            recent_profit_cents=(-6_000_000, -6_000_000),
        ),
    )
    return _rehash(
        replace(
            state,
            episode_id=(
                "persona-pilot-financial-stress"
                if seed == 42
                else f"persona-pilot-financial-stress-{seed}"
            ),
            round=3,
            rounds_remaining=3,
            state_version=2,
            risk_signals=(),
            companies=tuple(
                stressed if item.company_id == "company_A" else item
                for item in state.companies
            ),
        )
    )


def _capacity_pressure_state(config: MarketConfig, seed: int = 42) -> MarketState:
    state = _normal_state(config, seed)
    company = state.company("company_A")
    pressured = replace(
        company,
        commercial=replace(
            company.commercial,
            attempted_unfulfilled_orders=700,
        ),
        operations=replace(
            company.operations,
            effective_capacity_orders=3_500,
            capacity_utilization_ppm=980_000,
        ),
        brand=replace(
            company.brand,
            last_attempted_unfulfilled_rate_ppm=200_000,
        ),
        history=replace(
            company.history,
            recent_profit_cents=(1_000_000, 900_000),
        ),
    )
    return _rehash(
        replace(
            state,
            episode_id=f"persona-pilot-capacity-pressure-{seed}",
            round=3,
            rounds_remaining=3,
            state_version=2,
            companies=tuple(
                pressured if item.company_id == "company_A" else item
                for item in state.companies
            ),
        )
    )


def _active_incident_state(config: MarketConfig, seed: int = 42) -> MarketState:
    state = _normal_state(config, seed)
    company = state.company("company_A")
    incident = CompanyIncident(
        incident_id=f"persona-pilot-incident-{seed}",
        incident_type="warehouse_equipment_failure",
        severity="high",
        started_round=2,
        remaining_rounds=2,
        repair_required_cents=1_000_000,
        accumulated_repair_cents=0,
        capacity_multiplier_ppm=200_000,
        advertising_multiplier_ppm=1_000_000,
        service_penalty_ppm=100_000,
        reputation_penalty_ppm=50_000,
        refund_rate_ppm=20_000,
    )
    affected = replace(
        company,
        risk=replace(company.risk, active_incident=incident),
        history=replace(
            company.history,
            recent_profit_cents=(1_200_000, -800_000),
        ),
    )
    return _rehash(
        replace(
            state,
            episode_id=f"persona-pilot-active-incident-{seed}",
            round=3,
            rounds_remaining=3,
            state_version=2,
            companies=tuple(
                affected if item.company_id == "company_A" else item
                for item in state.companies
            ),
        )
    )
def build_scenarios(
    config: MarketConfig,
    *,
    scenario_classes: tuple[str, ...] = DEFAULT_SCENARIOS,
    market_seeds: tuple[int, ...] = (42,),
) -> tuple[FrozenScenario, ...]:
    scenarios: list[FrozenScenario] = []
    multiple = len(market_seeds) > 1
    for seed in market_seeds:
        states = {
            "normal": _normal_state(config, seed),
            "risk_warning": _risk_state(
                config, 0 if seed == 42 and not multiple else seed
            ),
            "financial_stress": _financial_stress_state(config, seed),
            "capacity_pressure": _capacity_pressure_state(config, seed),
            "active_incident": _active_incident_state(config, seed),
        }
        for scenario_class in scenario_classes:
            state = states[scenario_class]
            scenario_id = (
                f"{scenario_class}_seed_{seed}" if multiple else scenario_class
            )
            scenarios.append(
                FrozenScenario(
                    scenario_id,
                    scenario_class,
                    seed,
                    state,
                    _observation(config, state),
                )
            )
    return tuple(scenarios)


def _context_hash_without_persona(context: dict[str, Any]) -> str:
    payload = json.loads(json.dumps(context, ensure_ascii=False))
    payload.pop("persona_profile", None)
    identity = payload.get("identity", {})
    identity.pop("persona", None)
    identity.pop("objective", None)
    return sha256_hash(payload)


def _model_client(
    provider: str,
    model: str | None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> object:
    if provider == "mock":
        return MockModelClient()
    if provider == "doubao":
        return DoubaoModelClient(
            model=model, temperature=temperature, top_p=top_p
        )
    if provider == "deepseek":
        return DeepSeekModelClient(
            model=model, temperature=temperature, top_p=top_p
        )
    raise ValueError(f"unsupported provider: {provider}")


def _rotated(values: tuple[str, ...], offset: int) -> tuple[str, ...]:
    shift = offset % len(values)
    return values[shift:] + values[:shift]


def _ablation_profile(
    target: PersonaProfile,
    baseline: PersonaProfile,
    mode: str,
) -> PersonaProfile:
    if mode == "full":
        return target
    if mode not in SUPPORTED_ABLATIONS:
        raise ValueError(f"unsupported ablation: {mode}")
    payload = baseline.model_dump(mode="json")
    payload.update(
        {
            "persona_id": f"ablation_{mode}",
            "catalog_version": f"{target.catalog_version}-ablation-v1",
            "label": target.label if mode == "label_only" else "实验人格",
            "objective": (
                target.objective
                if mode == "objective_only"
                else "在统一安全约束内，根据给定实验字段做经营决策。"
            ),
            "utility_weights_ppm": (
                target.utility_weights_ppm.model_dump(mode="json")
                if mode == "weights_only"
                else baseline.utility_weights_ppm.model_dump(mode="json")
            ),
            "traits_ppm": (
                target.traits_ppm.model_dump(mode="json")
                if mode == "traits_only"
                else baseline.traits_ppm.model_dump(mode="json")
            ),
        }
    )
    return PersonaProfile.model_validate(payload)


async def _run_decision(
    *,
    provider: str,
    model_client: object,
    profile: PersonaProfile,
    target_persona_id: str,
    ablation_mode: str,
    registry: Any,
    config: MarketConfig,
    scenario: FrozenScenario,
    repetition: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    runtime = AgentRuntime(
        agent_id=(
            f"persona-pilot-{provider}-{target_persona_id}-{ablation_mode}"
        ),
        company_id="company_A",
        model_client=model_client,
        memory=EpisodeMemory(),
        persona_profile=profile,
        persona_registry=registry,
    )
    result = await runtime.decide(
        scenario.observation,
        timeout_seconds=timeout_seconds,
    )
    base = {
        "experiment_schema_version": "persona-pilot-row-v1.0.0",
        "provider": provider,
        "scenario_id": scenario.scenario_id,
        "scenario_class": scenario.scenario_class,
        "market_seed": scenario.market_seed,
        "repetition": repetition,
        "state_hash": scenario.state.state_hash,
        "persona_id": target_persona_id,
        "condition_id": (
            target_persona_id
            if ablation_mode == "full"
            else f"{target_persona_id}:{ablation_mode}"
        ),
        "ablation_mode": ablation_mode,
        "persona_profile": profile.model_dump(mode="json"),
        "persona_profile_hash": profile.profile_hash,
        "success": result.success,
        "model_name": result.model_name,
        "prompt_version": result.prompt_version,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "error_code": result.error_code,
        "error_message": result.error_message,
    }
    context_payload = result.context.model_dump(mode="json")
    base["context_hash_without_persona"] = _context_hash_without_persona(
        context_payload
    )
    if not result.success or result.decision is None:
        return base
    requested = result.decision.requested_action.model_dump(mode="json")
    resolution = resolve_action_request(
        config,
        scenario.state,
        "company_A",
        requested,
        source=f"persona-pilot:{target_persona_id}:{ablation_mode}",
        action_id=(
            f"persona-pilot:{scenario.scenario_id}:{target_persona_id}:"
            f"{ablation_mode}:{repetition}"
        ),
    )
    base.update(
        {
            "plan": result.decision.plan.model_dump(mode="json"),
            "confidence_ppm": result.decision.confidence_ppm,
            "requested_action": requested,
            "final_action": resolution.action.to_dict(),
            "resolution_adjustments": [
                item.to_dict() for item in resolution.adjustments
            ],
        }
    )
    return base


def _action_vector(action: dict[str, Any], config: MarketConfig) -> tuple[float, ...]:
    bounds = config.mapping("action", "bounds")
    values: list[float] = []
    for field in ACTION_FIELDS:
        low = int(bounds[field]["min"])
        high = int(bounds[field]["max"])
        values.append((int(action[field]) - low) / max(1, high - low))
    repair = action.get("incident_response", {})
    repair_high = int(bounds["repair_budget_cents"]["max"])
    values.append(int(repair.get("repair_budget_cents", 0)) / repair_high)
    mode_score = {"wait": 0.0, "partial_repair": 0.5, "full_repair": 1.0}
    values.append(mode_score.get(str(repair.get("mode", "wait")), 0.0))
    return tuple(values)


def _distance(left: Iterable[float], right: Iterable[float]) -> float:
    pairs = tuple(zip(left, right))
    return sum(abs(a - b) for a, b in pairs) / max(1, len(pairs))


def _mean_vector(vectors: list[tuple[float, ...]]) -> tuple[float, ...]:
    return tuple(mean(values) for values in zip(*vectors))


def summarize(
    rows: list[dict[str, Any]], config: MarketConfig
) -> dict[str, Any]:
    successful = [row for row in rows if row["success"] and row.get("final_action")]
    context_integrity: dict[str, bool] = {}
    for scenario_id in sorted({row["scenario_id"] for row in rows}):
        hashes = {
            row["context_hash_without_persona"]
            for row in rows
            if row["scenario_id"] == scenario_id
        }
        context_integrity[scenario_id] = len(hashes) == 1

    grouped: dict[tuple[str, str], list[tuple[float, ...]]] = {}
    action_means: dict[str, dict[str, dict[str, float]]] = {}
    for row in successful:
        condition_id = row.get("condition_id", row["persona_id"])
        key = (row["scenario_id"], condition_id)
        grouped.setdefault(key, []).append(
            _action_vector(row["final_action"], config)
        )
        scenario = action_means.setdefault(row["scenario_id"], {})
        persona = scenario.setdefault(condition_id, {})
        for field in ACTION_FIELDS:
            persona.setdefault(field, 0.0)
            persona[field] += float(row["final_action"][field])
        persona.setdefault("sample_count", 0.0)
        persona["sample_count"] += 1
    for scenario in action_means.values():
        for persona in scenario.values():
            count = persona.pop("sample_count")
            for field in tuple(persona):
                persona[field] = round(persona[field] / count, 2)

    within: list[float] = []
    for vectors in grouped.values():
        within.extend(_distance(a, b) for a, b in combinations(vectors, 2))
    between: list[float] = []
    for scenario_id in sorted({key[0] for key in grouped}):
        means = [
            _mean_vector(vectors)
            for (scenario, _persona), vectors in grouped.items()
            if scenario == scenario_id
        ]
        between.extend(_distance(a, b) for a, b in combinations(means, 2))
    within_mean = mean(within) if within else 0.0
    between_mean = mean(between) if between else 0.0
    signal_ratio = between_mean / within_mean if within_mean > 0 else None
    return {
        "summary_schema_version": "persona-pilot-summary-v1.0.0",
        "row_count": len(rows),
        "successful_decisions": len(successful),
        "failed_decisions": len(rows) - len(successful),
        "context_integrity": context_integrity,
        "all_contexts_frozen": all(context_integrity.values()),
        "mean_within_persona_action_distance": round(within_mean, 6),
        "mean_between_persona_action_distance": round(between_mean, 6),
        "persona_signal_ratio": (
            round(signal_ratio, 4) if signal_ratio is not None else None
        ),
        "action_means": action_means,
        "fallback_or_error_counts": {
            code: sum(row.get("error_code") == code for row in rows)
            for code in sorted(
                {str(row.get("error_code")) for row in rows if row.get("error_code")}
            )
        },
    }


async def run(args: argparse.Namespace) -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config_path = Path(
        os.environ.get("MARKET_CONFIG_PATH", PROJECT_ROOT / "configs" / "market_v4.yaml")
    )
    config = load_market_config(config_path)
    registry = load_persona_registry(config_path)
    persona_ids = tuple(
        item.strip() for item in args.personas.split(",") if item.strip()
    )
    ablation_modes = tuple(
        item.strip() for item in args.ablations.split(",") if item.strip()
    )
    unknown_ablations = sorted(set(ablation_modes) - set(SUPPORTED_ABLATIONS))
    if unknown_ablations:
        raise ValueError("unknown ablations: " + ", ".join(unknown_ablations))
    baseline_profile = registry.get("none")
    conditions = tuple(
        (persona_id, mode)
        for persona_id in persona_ids
        for mode in ablation_modes
        if persona_id != "none" or mode == "full"
    )
    profiles = {
        (persona_id, mode): _ablation_profile(
            registry.get(persona_id), baseline_profile, mode
        )
        for persona_id, mode in conditions
    }
    scenario_classes = tuple(
        item.strip() for item in args.scenarios.split(",") if item.strip()
    )
    unknown_scenarios = sorted(set(scenario_classes) - set(SUPPORTED_SCENARIOS))
    if unknown_scenarios:
        raise ValueError("unknown scenarios: " + ", ".join(unknown_scenarios))
    market_seeds = tuple(
        int(item.strip()) for item in args.market_seeds.split(",") if item.strip()
    )
    if not market_seeds or any(seed < 0 or seed >= 1 << 64 for seed in market_seeds):
        raise ValueError("market seeds must be uint64 integers")
    scenarios = build_scenarios(
        config,
        scenario_classes=scenario_classes,
        market_seeds=market_seeds,
    )
    output = args.output or (
        PROJECT_ROOT
        / "runs"
        / f"persona-pilot-{args.provider}-{uuid.uuid4().hex[:8]}"
    )
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "decisions.jsonl"
    rows: list[dict[str, Any]] = []
    client = _model_client(
        args.provider, args.model, args.temperature, args.top_p
    )
    manifest = {
        "manifest_schema_version": "persona-pilot-manifest-v2.0.0",
        "provider": args.provider,
        "model": args.model,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "provider_seed": None,
        },
        "personas": list(persona_ids),
        "ablations": list(ablation_modes),
        "market_seeds": list(market_seeds),
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "scenario_class": scenario.scenario_class,
                "market_seed": scenario.market_seed,
                "state": scenario.state.to_dict(),
            }
            for scenario in scenarios
        ],
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    call_index = 0
    condition_keys = tuple(f"{persona_id}|{mode}" for persona_id, mode in conditions)
    for repetition in range(1, args.repetitions + 1):
        for scenario_index, scenario in enumerate(scenarios):
            order = _rotated(condition_keys, repetition + scenario_index)
            for condition_key in order:
                persona_id, ablation_mode = condition_key.split("|", 1)
                call_index += 1
                print(
                    f"[{call_index}/{len(scenarios) * len(conditions) * args.repetitions}] "
                    f"{scenario.scenario_id} / {persona_id} / {ablation_mode}",
                    flush=True,
                )
                row = await _run_decision(
                    provider=args.provider,
                    model_client=client,
                    profile=profiles[(persona_id, ablation_mode)],
                    target_persona_id=persona_id,
                    ablation_mode=ablation_mode,
                    registry=registry,
                    config=config,
                    scenario=scenario,
                    repetition=repetition,
                    timeout_seconds=args.timeout,
                )
                rows.append(row)
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")
    summary = summarize(rows, config)
    summary.update(
        {
            "experiment_id": output.name,
            "provider": args.provider,
            "personas": list(persona_ids),
            "ablations": list(ablation_modes),
            "market_seeds": list(market_seeds),
            "sampling": manifest["sampling"],
            "scenarios": [scenario.scenario_id for scenario in scenarios],
            "repetitions": args.repetitions,
            "config_id": config.config_id,
            "config_version": config.config_version,
            "config_sha256": config.config_sha256,
        }
    )
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"OUTPUT_DIR={output.resolve()}", flush=True)
    return 0 if summary["failed_decisions"] == 0 else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=("mock", "doubao", "deepseek"), default="mock"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--personas", default=",".join(DEFAULT_PERSONAS))
    parser.add_argument("--ablations", default=",".join(DEFAULT_ABLATIONS))
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--market-seeds", default="42")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
