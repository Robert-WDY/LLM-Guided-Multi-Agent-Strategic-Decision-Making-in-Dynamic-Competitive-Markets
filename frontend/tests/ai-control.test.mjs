import assert from "node:assert/strict";
import test from "node:test";

import {
  AI_MODEL_OPTIONS,
  DEFAULT_AGENTS,
  agentsForEntry,
  buildEpisodeAgentConfigs,
  mergeBackendAgentState,
  aiFlowAgents,
} from "../app/lab-model.ts";

const MODEL_IDS = [
  "nvidia/nemotron-3-super-120b-a12b:free",
  "nvidia/nemotron-3-ultra-550b-a55b:free",
];

test("control configuration exposes only human and fixed AI models", () => {
  assert.deepEqual(AI_MODEL_OPTIONS.map((option) => option.id), MODEL_IDS);
  assert.ok(AI_MODEL_OPTIONS.every((option) => option.label && !/openrouter/i.test(option.label)));
  assert.deepEqual(new Set(DEFAULT_AGENTS.map((agent) => agent.driver)), new Set(["human", "model"]));
  assert.equal(DEFAULT_AGENTS[0].driver, "human");
  assert.ok(DEFAULT_AGENTS.slice(1).every((agent) => agent.driver === "model"));
});

test("observer and research entries start with AI while participant keeps company A human", () => {
  const participant = agentsForEntry("participant");
  const observer = agentsForEntry("observer");
  const research = agentsForEntry("research");

  assert.equal(participant[0].driver, "human");
  assert.ok(participant.slice(1).every((agent) => agent.driver === "model"));
  assert.ok(observer.every((agent) => agent.driver === "model"));
  assert.ok(research.every((agent) => agent.driver === "model"));
});

test("episode agent configs preserve each selected model and PersonaAgent identity", () => {
  const agents = agentsForEntry("research").map((agent, index) => ({
    ...agent,
    model: MODEL_IDS[index % MODEL_IDS.length],
  }));

  const configs = buildEpisodeAgentConfigs(agents);

  for (const [index, agent] of agents.entries()) {
    assert.equal(configs[agent.companyId].agent_type, "model");
    assert.equal(configs[agent.companyId].model, MODEL_IDS[index % MODEL_IDS.length]);
    assert.equal(configs[agent.companyId].agent_id, `single-agent-${agent.companyId}`);
    assert.equal(configs[agent.companyId].persona_name, agent.persona);
  }
});

test("new backend episode has no fabricated settlement delta before the first round", () => {
  const configs = agentsForEntry("participant");
  const companies = Object.fromEntries(configs.map((agent) => [agent.companyId, {
    financial: { cash_balance_cents: 20_000_000, round_profit_cents: 0 },
    commercial: { price_cents: 10_000, market_share_ppm: 250_000 },
    risk: { resilience_ppm: 150_000 },
  }]));

  const created = mergeBackendAgentState({ companies }, configs);

  assert.ok(created.every((agent) => agent.shareDelta === 0));
  assert.ok(created.every((agent) => agent.profit === 0));
});

test("AI node flow excludes human-controlled companies", () => {
  const runtime = mergeBackendAgentState({
    companies: Object.fromEntries(agentsForEntry("participant").map((agent) => [agent.companyId, {
      financial: { cash_balance_cents: 20_000_000, round_profit_cents: 0 },
      commercial: { price_cents: 10_000, market_share_ppm: 250_000 },
      risk: { resilience_ppm: 150_000 },
    }])),
  }, agentsForEntry("participant"));

  assert.deepEqual(aiFlowAgents(runtime).map((agent) => agent.companyId), ["company_B", "company_C", "company_D"]);
});
