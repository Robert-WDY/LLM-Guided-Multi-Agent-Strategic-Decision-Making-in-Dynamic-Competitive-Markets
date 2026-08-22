import assert from "node:assert/strict";
import test from "node:test";

import { advanceDemoRound, DEMO_AGENTS } from "../app/lab-model.ts";

const baseAction = { price: 9800, advertising: 200000, contribution: 200000 };

test("demo market conserves total share and does not force monotonic trajectories", () => {
  let agents = DEMO_AGENTS;
  const directions = DEMO_AGENTS.map(() => []);

  for (let round = 1; round <= 5; round += 1) {
    agents = advanceDemoRound(agents, round, baseAction);
    const totalShare = agents.reduce((sum, agent) => sum + agent.share, 0);
    assert.ok(Math.abs(totalShare - 100) < 1e-9, `round ${round} share total was ${totalShare}`);
    agents.forEach((agent, index) => directions[index].push(Math.sign(agent.shareDelta)));
  }

  assert.ok(
    directions.some((trajectory) => trajectory.includes(-1) && trajectory.includes(1)),
    "at least one company should both gain and lose share across rounds",
  );
});

test("human price choice changes share and profit instead of replaying fixed deltas", () => {
  const lowPrice = advanceDemoRound(DEMO_AGENTS, 1, { ...baseAction, price: 8000 });
  const highPrice = advanceDemoRound(DEMO_AGENTS, 1, { ...baseAction, price: 12000 });

  assert.ok(lowPrice[0].share > highPrice[0].share);
  assert.notEqual(lowPrice[0].profit, highPrice[0].profit);
});

test("opponent judgment starts without evidence and updates only after settlement", () => {
  assert.ok(DEMO_AGENTS.every((agent) => agent.beliefs.every((belief) => belief.evidence.length === 0)));
  const afterFirstSettlement = advanceDemoRound(DEMO_AGENTS, 1, baseAction);
  assert.ok(afterFirstSettlement.every((agent) => agent.beliefs.every((belief) => belief.evidence.length === 2)));
  assert.ok(afterFirstSettlement[0].beliefs.some((belief) => belief.nextAction.cut !== 33));
});
