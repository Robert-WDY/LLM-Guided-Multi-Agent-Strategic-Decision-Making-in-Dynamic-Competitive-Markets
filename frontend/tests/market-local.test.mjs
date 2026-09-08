import assert from "node:assert/strict";
import test from "node:test";
import { readFile, writeFile, unlink } from "node:fs/promises";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
const source = await readFile(new URL("../app/market-local.tsx", import.meta.url), "utf8");
const path = new URL(`./.market-local-render-${process.pid}.mjs`, import.meta.url);
await writeFile(path, ts.transpileModule(source, { compilerOptions: { jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText);
const { LocalMarketLedger } = await import(path.href);
await unlink(path);
const render = state => renderToStaticMarkup(React.createElement(LocalMarketLedger, { state }));
const {restoreTheoryParameters}=await import('../app/theory-parameters.ts');
const advisorSource=await readFile(new URL('../app/advisor-v16.tsx',import.meta.url),'utf8');
const advisorPath=new URL(`./.advisor-render-${process.pid}.mjs`,import.meta.url);
await writeFile(advisorPath,ts.transpileModule(advisorSource,{compilerOptions:{jsx:ts.JsxEmit.ReactJSX,target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText);
const {AdvisorResult}=await import(advisorPath.href);await unlink(advisorPath);
const advisorFixture=JSON.parse(await readFile(new URL('./fixtures/advisor-v16.json',import.meta.url),'utf8'));
const robustFixture=JSON.parse(await readFile(new URL('./fixtures/advisor-v17.json',import.meta.url),'utf8'));
test("legacy state does not invent four-actor accounts", () => assert.equal(render({}), ""));
test("four-actor ledger renders actual signed budgets, materials and voluntary choice", () => {
  const html = render({
    government: { cash_cents: 12000, cumulative_net_cents: -3000, round_fines_cents: 0, last_decision: { applies_round: 2, observed_stockout_orders: 9, observed_hhi_ppm: 300000, inspection_cases: 1, opening_cash_cents: 15000, inspection_cost_cents: 1000, support_by_company_cents: { A: 2000 } } },
    companies: { A: { financial: { cash_balance_cents: -12345 } } },
    supply_chain: { suppliers: { S: { supplier_id: "S", label: "供应商 S", base_capacity_orders: 50, account: { cash_cents: 25000, receipts_cents: 1234, production_cost_cents: 1000 }, investment_decision: { reason: "payback_horizon_insufficient", added_capacity_orders: 0 } } }, last_procurement_outcomes: { A: { company_id: "A", material: { payment_cents: 1234, used_orders: 7, wasted_orders: 3, payments_by_supplier: { S: 1234 } } } } },
    consumer_decisions: [{ group_id: "群体一", unit_budget_cents: 9999, demand_orders: 20, voluntary_no_purchase_orders: 4, stockout_orders: 9, spending_cents: 12000, refund_cents: 250, purchases_by_company: { A: 7 } }],
  });
  for (const phrase of [/四方自主市场/, /-¥123\.45/, /12\.34/, /7 件/, /3 件/, /7 \/ 4 \/ 9/, /2\.50/, /剩余经营期不足/]) assert.match(html, phrase);
});

test('actual advisor response renders goals, evidence and limits',()=>{
 const html=renderToStaticMarkup(React.createElement(AdvisorResult,{value:advisorFixture}));
 for(const phrase of ['社会福利（本公司单边行动）','扩张','为什么选择它','不是置信区间','独立市场执行对照','可执行动作'])assert.ok(html.includes(phrase),phrase);
});
test('incomplete multiplayer checks never render a Nash certificate',()=>{
 const value=structuredClone(advisorFixture);value.diagnostics={scope:'partial',recommendation_check:{complete:false,restricted_pure_nash:null,deviation_gains:{},profile:{company_A:'baseline'}}};value.backtest=null;
 const html=renderToStaticMarkup(React.createElement(AdvisorResult,{value}));
 assert.ok(html.includes('未完整检查，不能判断均衡'));assert.ok(!html.includes('各方没有获利偏离'));
});
test('v17 renders response selection, cold start and preference sensitivity',()=>{
 const html=renderToStaticMarkup(React.createElement(AdvisorResult,{value:robustFixture}));
 for(const text of ['对手反击怎样影响最终建议','无历史时依赖先验','目标权重是否敏感','无历史评估','未来覆盖保证'])assert.ok(html.includes(text),text);
 assert.ok(html.includes('已完成配置的反击搜索'));
});
test('v17 incomplete response clearly abstains and does not invent sensitivity',()=>{
 const value=structuredClone(robustFixture);value.v17.response_complete=false;value.v17.sensitivity=null;
 const html=renderToStaticMarkup(React.createElement(AdvisorResult,{value}));
 assert.ok(html.includes('本次退回基线'));assert.ok(html.includes('未生成权重稳健性结论'));
});
test('saved theory requests restore API defaults and explicit overrides',()=>{
 assert.deepEqual(restoreTheoryParameters('repeated',{}),{row:'tit_for_tat',column:'cooperate',rounds:50,noise:0,discount:.95,seed:1});
 const p=restoreTheoryParameters('learning',{rounds:2999,seed:42});assert.equal(p.game,'matching_pennies');assert.equal(p.rounds,2999);assert.equal(p.seed,42);
 assert.deepEqual(restoreTheoryParameters('matrix',{}),{game:'prisoners_dilemma'});
});
