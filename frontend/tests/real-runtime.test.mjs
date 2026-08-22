import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { agentsForEntry } from "../app/lab-model.ts";
import { buildRealRuntimeAgents } from "../app/real-runtime.ts";

test("真实模式投影不依赖演示数据，页面不回退到旧合并器", () => {
  const realRuntimeSource = readFileSync(new URL("../app/real-runtime.ts", import.meta.url), "utf8");
  const pageSource = readFileSync(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.doesNotMatch(realRuntimeSource, /\bDEMO_/);
  assert.doesNotMatch(pageSource, /import\s*\{[^}]*mergeBackendAgentState/);
});

const state = {
  market: {
    realized_demand_orders: 321,
    average_paid_price_cents: 8765,
    actual_supply_cost_index_ppm: 1_123_456,
    market_sentiment_ppm: 654_321,
  },
  companies: Object.fromEntries(["A", "B", "C", "D"].map((shortName, index) => [`company_${shortName}`, {
    financial: { cash_balance_cents: 10_000_000 + index, round_profit_cents: 0 },
    commercial: { price_cents: 8_000 + index, market_share_ppm: 250_000 },
    operations: { effective_capacity_orders: 1_000 + index, actual_unit_cost_cents: 4_000 + index },
    brand: { service_quality_ppm: 500_000, reputation_ppm: 600_000 },
    risk: { resilience_ppm: 200_000 },
    history: { last_action: null, recent_profit_cents: [], recent_market_share_ppm: [] },
  }])),
};

for (const entry of ["participant", "observer", "research"]) {
  test(`${entry} real runtime contains only backend values before AI runs`, () => {
    const agents = buildRealRuntimeAgents({ state, configs: agentsForEntry(entry) });

    assert.equal(agents[0].cash, 10_000_000);
    assert.equal(agents[0].observation.public[0].value, "321 单");
    assert.equal(agents[0].observation.private[1].value, "1000 单");
    assert.equal(agents[0].decision.summary, "");
    assert.equal(agents[0].plan.goal, "");
    assert.deepEqual(agents[0].beliefs, []);
    assert.equal(agents[0].actionAvailable, false);
    assert.equal(agents[0].dataSource, "backend_initial_state");
    assert.doesNotMatch(JSON.stringify(agents), /12,480|提高风险调整|降低价格并暂缓|供应冲击信号/);
  });
}

test("real runtime uses only actual resolution after settlement", () => {
  const agents = buildRealRuntimeAgents({
    state,
    configs: agentsForEntry("observer"),
    resolutions: { company_A: { action: { price_cents: 7777, advertising_budget_cents: 1234 } } },
    previous: buildRealRuntimeAgents({ state, configs: agentsForEntry("observer") }),
  });

  assert.equal(agents[0].action.price, 7777);
  assert.equal(agents[0].action.advertising, 1234);
  assert.equal(agents[0].actionAvailable, true);
  assert.equal(agents[0].dataSource, "controller_settlement");
});
