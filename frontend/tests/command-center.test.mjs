import assert from "node:assert/strict";
import test from "node:test";

import {
  createScreen,
  PHASES,
  SCREEN_MAX_AGENTS,
  SCREEN_MIN_AGENTS,
  SCREEN_ROUNDS,
  screenReducer,
  tick,
} from "../app/command-center-model.ts";

function runRounds(state, rounds) {
  let s = state;
  for (let i = 0; i < rounds * PHASES.length; i += 1) s = tick(s);
  return s;
}

test("screen clamps agent count to the supported roster", () => {
  assert.equal(createScreen(1).agents.length, SCREEN_MIN_AGENTS);
  assert.equal(createScreen(99).agents.length, SCREEN_MAX_AGENTS);
  assert.equal(createScreen(6).agents.length, 6);
});

test("every settled round conserves 100% share and rounds advance without looping", () => {
  for (const count of [2, 4, 7, 10]) {
    let s = createScreen(count);
    assert.equal(s.round, 1);
    const seen = [];
    for (let i = 0; i < SCREEN_ROUNDS * PHASES.length; i += 1) {
      s = tick(s);
      if (PHASES[s.phase].id === "settle") {
        const total = s.agents.reduce((sum, a) => sum + a.share, 0);
        assert.ok(Math.abs(total - 100) < 1e-6, `count ${count} round ${s.round} total ${total}`);
        seen.push(s.round);
      }
    }
    assert.deepEqual(seen, Array.from({ length: SCREEN_ROUNDS }, (_, i) => i + 1));
  }
});

test("after the final round a fresh episode starts at round 1", () => {
  const s = runRounds(createScreen(4), SCREEN_ROUNDS);
  assert.equal(s.episode, 2);
  assert.equal(s.round, 1);
  assert.equal(s.history.length, 1);
  assert.ok(s.feed.some((item) => item.verb === "新局"));
});

test("first round advice does not fabricate opponent knowledge", () => {
  const s = createScreen(5);
  assert.ok(s.plan.advice.every((a) => a.phenomenon === "信息不足"));
});

test("messages only connect agents on the roster and agreements record actual behaviour", () => {
  let s = createScreen(8);
  const ids = new Set(s.agents.map((a) => a.companyId));
  let messages = 0;
  for (let r = 0; r < 10; r += 1) {
    for (const m of s.plan.messages) {
      assert.ok(ids.has(m.from));
      assert.ok(m.to === null || ids.has(m.to));
      assert.notEqual(m.from, m.to);
    }
    messages += s.plan.messages.length;
    s = runRounds(s, 1);
  }
  assert.ok(messages > 0);
  assert.equal(s.totals.messages, messages);
  const linkTotal = Object.values(s.links).reduce((sum, row) => sum + Object.values(row).reduce((a, b) => a + b, 0), 0);
  assert.equal(linkTotal, messages);
  const settled = s.agreements.filter((a) => a.round < s.round);
  assert.ok(settled.length > 0);
  for (const ag of settled) {
    assert.ok(ag.outcome);
    for (const id of [ag.proposer, ag.partner]) {
      const act = ag.actual[id];
      assert.equal(act.kept, act.price >= ag.terms.minPrice && act.contribution >= ag.terms.minContribution);
    }
  }
  assert.ok(s.totals.kept + s.totals.broken > 0);
});

test("inventory flow balances every round", () => {
  let s = createScreen(5);
  for (let r = 0; r < 8; r += 1) {
    for (const o of s.plan.ops) {
      assert.equal(o.stockStart, s.inventory[o.companyId]);
      assert.equal(o.sold + o.shortage, o.orders);
      assert.equal(o.stockStart + o.procured - o.sold - o.spoiled, o.stockEnd);
      assert.ok(o.procured <= o.capacity && o.stockEnd >= 0);
    }
    s = runRounds(s, 1);
  }
});

test("beliefs stay unknown in round 1 and use public evidence afterwards", () => {
  let s = createScreen(4);
  assert.ok(s.plan.beliefs.every((b) => b.targetId === null && b.probability === null));
  s = runRounds(s, 1);
  assert.ok(s.plan.beliefs.some((b) => b.targetId && b.evidence.length > 0));
});

test("simulation is deterministic and reset changes the roster", () => {
  const a = runRounds(createScreen(6), 7);
  const b = runRounds(createScreen(6), 7);
  assert.deepEqual(a.agents.map((x) => x.share), b.agents.map((x) => x.share));
  const reset = screenReducer(a, { type: "reset", count: 3 });
  assert.equal(reset.agents.length, 3);
  assert.equal(reset.round, 1);
  assert.equal(reset.phase, 0);
});

test("advice phenomena go beyond the uninformed default once history exists", () => {
  let s = createScreen(10);
  const kinds = new Set();
  for (let r = 0; r < SCREEN_ROUNDS - 1; r += 1) {
    s = runRounds(s, 1);
    s.plan.advice.forEach((a) => kinds.add(a.phenomenon));
  }
  assert.ok(kinds.size >= 3, `only saw ${[...kinds].join(",")}`);
});

test("advice impact is a same-market counterfactual and totals accumulate it", () => {
  let s = createScreen(6);
  let adopted = 0;
  let missed = 0;
  for (let r = 0; r < 12; r += 1) {
    const { advice, impacts } = s.plan;
    assert.equal(impacts.length, advice.length);
    impacts.forEach((imp, i) => {
      assert.equal(imp.adopted, advice[i].adopted);
      assert.equal(imp.comparedWith, imp.adopted ? "keep" : "follow");
      assert.ok(Number.isFinite(imp.profitDelta));
      if (imp.adopted) adopted += imp.profitDelta;
      else missed += imp.profitDelta;
    });
    s = runRounds(s, 1);
  }
  assert.equal(s.totals.adoptedGain, adopted);
  assert.equal(s.totals.missedGain, missed);
  assert.ok(s.lastOps.length === s.agents.length);
});

test("an override that repeats the actual action leaves the market unchanged", async () => {
  const { advanceDemoRound, DEMO_AGENTS } = await import("../app/lab-model.ts");
  const base = { price: 9800, advertising: 200000, contribution: 200000 };
  const plain = advanceDemoRound(DEMO_AGENTS, 1, base);
  const same = advanceDemoRound(DEMO_AGENTS, 1, base, { company_B: { price: plain[1].price } });
  assert.deepEqual(same.map((a) => a.share), plain.map((a) => a.share));
});

test("agents do not all make the same choice: beliefs, prices and advice uptake diverge", () => {
  for (const count of [4, 6, 10]) {
    let s = createScreen(count);
    const beliefTexts = new Set();
    let spreadRounds = 0;
    let mixedAdoption = 0;
    for (let r = 1; r < SCREEN_ROUNDS; r += 1) {
      s.plan.beliefs.forEach((b) => beliefTexts.add(`${b.move}:${b.probability}`));
      const adopted = s.plan.advice.filter((a) => a.adopted).length;
      if (r > 1 && adopted > 0 && adopted < s.plan.advice.length) mixedAdoption += 1;
      s = runRounds(s, 1);
      const prices = s.agents.map((a) => a.price);
      if (Math.max(...prices) - Math.min(...prices) >= 200) spreadRounds += 1;
    }
    assert.ok(beliefTexts.size >= 8, `count ${count}: only ${beliefTexts.size} distinct beliefs`);
    assert.ok(spreadRounds >= 10, `count ${count}: prices spread ≥ ¥2 in only ${spreadRounds} rounds`);
    assert.ok(mixedAdoption >= 10, `count ${count}: mixed adoption in only ${mixedAdoption} rounds`);
  }
});

test("decisions are driven by persona, advice and agreements and the trace explains them", () => {
  let s = createScreen(8);
  for (let r = 0; r < 12; r += 1) {
    s.plan.decisions.forEach((d, i) => {
      assert.equal(d.followed, s.plan.advice[i].adopted);
      assert.ok(d.reasons.length > 0 || s.round === 1);
      const act = s.plan.actions[i];
      if (d.honoured === true) assert.ok(act.price >= d.floor);
      if (d.honoured === false) assert.ok(act.price < d.floor && act.contribution === 0);
    });
    s = runRounds(s, 1);
  }
});
