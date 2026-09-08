import assert from "node:assert/strict";
import test from "node:test";
import { readFile, writeFile, unlink } from "node:fs/promises";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

const source = await readFile(new URL("../app/market-v10.tsx", import.meta.url), "utf8");
const path = new URL(`./.market-v10-render-${process.pid}.mjs`, import.meta.url);
await writeFile(path, ts.transpileModule(source, { compilerOptions: { jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText);
const components = await import(path.href);
await unlink(path);
const render = (component, props) => renderToStaticMarkup(React.createElement(component, props));

test("v10 welfare renders signed amounts and never substitutes welfare proxy", () => {
  const html = render(components.V10Results, { state: { state_version: 1, welfare_accounting: { round_government_net_budget_cents: -12345, cumulative_government_net_budget_cents: -12345, round_total_economic_welfare_cents: 10000 } } });
  assert.match(html, /政府净预算/); assert.match(html, /123\.45/); assert.match(html, /总经济福利/);
  assert.match(html, /合成研究参数/);
});
test("procurement has real supplier choices and no backup means all-primary", () => {
  const html = render(components.ProcurementControls, { action: { price: 9800, advertising: 0, contribution: 0 }, change() {}, disabled: false, state: { state_version: 0, supply_chain: { suppliers: { economy_supplier: { supplier_id: "economy_supplier", label: "经济供应商", unit_price_cents: 1800 }, resilient_supplier: { supplier_id: "resilient_supplier", label: "韧性供应商", unit_price_cents: 2400 } } } } });
  assert.match(html, /经济供应商/); assert.match(html, /韧性供应商/); assert.match(html, /100%/); assert.match(html, /不分散采购/);
});
test("saved experiments separate interrupted recovery from continuation", () => {
  const html = render(components.SavedExperiments, { episodes: [], busy: false, currentId: "saved", recovery: true, message: "", configHash: "", onRefresh() {}, onSave() {}, onRestore() {}, onExport() {}, onRecover() {} });
  assert.match(html, /不会重试模型/); assert.match(html, /结束中断轮/); assert.match(html, /保存当前实验/);
});
test("supplier audit separates posted next-round quote from settled price", () => {
  const html = render(components.V10Results, { state: { state_version: 1, supply_chain: { suppliers: { economy: { supplier_id: "economy", label: "供应商", unit_price_cents: 1890, last_settled_unit_price_cents: 1800, available_capacity_orders: 3300, reliability_ppm: 780000, disrupted: true, quote_decision: { observed_round: 1, applies_round: 2, previous_price_cents: 1800, quoted_price_cents: 1890, observed_sales_orders: 11000, observed_capacity_orders: 11000, utilization_ppm: 1000000, reason: "high_utilization" } } }, last_procurement_outcomes: {} } } });
  assert.match(html, /第 2 轮起生效/); assert.match(html, /最近结算单价/);
  assert.match(html, /18\.90/); assert.match(html, /18\.00/);
  assert.match(html, /不读取下一轮冲击/); assert.match(html, /上调报价/);
});
