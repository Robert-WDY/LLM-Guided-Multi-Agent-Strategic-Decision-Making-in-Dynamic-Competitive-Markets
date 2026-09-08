"use client";

type Investment = { observed_round: number; applies_round: number; investment_cents: number; added_capacity_orders: number; reason: string; observed_requested_orders?: number; observed_unit_margin_cents?: number; remaining_rounds?: number };
type Account = { opening_cash_cents: number; cash_cents: number; receipts_cents: number; production_cost_cents: number; investment_cents?: number };
type Material = { payment_cents: number; used_orders: number; wasted_orders: number; payments_by_supplier: Record<string, number> };
type GovernmentDecision = { observed_round: number; applies_round: number; opening_cash_cents: number; observed_stockout_orders: number; observed_hhi_ppm: number; inspection_cases: number; inspection_cost_cents: number; detection_boost_ppm: number; support_by_company_cents: Record<string, number>; reason: string };
type ConsumerChoice = { group_id: string; unit_budget_cents: number; demand_orders: number; voluntary_no_purchase_orders: number; stockout_orders: number; spending_cents: number; refund_cents: number; purchases_by_company: Record<string, number> };
export type LocalMarketState = {
  companies?: Record<string, { financial: { cash_balance_cents: number; round_government_support_cents?: number }; operations?: { base_capacity_orders?: number } }>;
  government?: { cash_cents: number; cumulative_net_cents: number; round_fines_cents: number; round_matched_support_cents?:number; round_consumer_rebate_cents?:number; last_decision: GovernmentDecision | null } | null;
  consumer_decisions?: ConsumerChoice[];
  supply_chain?: { suppliers: Record<string, { supplier_id: string; label: string; base_capacity_orders?: number; account?: Account; investment_decision?: Investment }>; last_procurement_outcomes?: Record<string, { company_id: string; material?: Material }> } | null;
};
const money = (n: number) => new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", maximumFractionDigits: 2 }).format(n / 100);
const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);
const investmentReasons: Record<string, string> = {
  funded_excess_demand_payback: "超额订单与预计回本均满足，扩充产能",
  temporary_disruption_not_capacity_shortage: "供给暂时中断，等待恢复",
  no_excess_orders: "现有基础产能可覆盖订单",
  payback_horizon_insufficient: "剩余经营期不足以保守回收投资",
  preserve_capacity: "使用率尚未达到扩产条件",
  terminal_no_investment: "本局结束，不再投资",
  disabled: "本实验关闭扩产策略",
  cash_or_capacity_limit: "保留现金或已达到产能上限",
  expand_high_utilization: "使用率较高，预算内扩产",
};

export function LocalMarketLedger({ state }: { state: LocalMarketState }) {
  if (!state.government) return null;
  const government = state.government;
  const decision = government.last_decision;
  const suppliers = Object.values(state.supply_chain?.suppliers ?? {});
  const consumers = state.consumer_decisions ?? [];
  const materials = Object.values(state.supply_chain?.last_procurement_outcomes ?? {}).filter(o => o.material);
  const companyCash = sum(Object.values(state.companies ?? {}).map(c => c.financial.cash_balance_cents));
  const support = state.government?.round_matched_support_cents ?? sum(Object.values(decision?.support_by_company_cents ?? {}));
  const invoices = sum(materials.map(o => o.material!.payment_cents));
  const receipts = sum(suppliers.map(s => s.account?.receipts_cents ?? 0));
  return <section className="card local-market-ledger" aria-label="四方自主市场">
    <div className="mini-panel-head"><span>四方自主市场</span><b>本机模拟 · 逐轮结算</b></div>
    <div className="v10-metric-grid">
      <article><h3>企业</h3><strong>{money(companyCash)}</strong><p>企业现金合计；采购量、价格与投入由各企业决策。</p></article>
      <article><h3>消费者</h3><strong>{sum(consumers.map(c => sum(Object.values(c.purchases_by_company)))) .toLocaleString()} 单</strong><p>自愿不买 {sum(consumers.map(c => c.voluntary_no_purchase_orders)).toLocaleString()} 单 · 缺货 {sum(consumers.map(c => c.stockout_orders)).toLocaleString()} 单</p></article>
      <article><h3>供应商</h3><strong>{money(sum(suppliers.map(s => s.account?.cash_cents ?? 0)))}</strong><p>现金合计；独立报价，在超额订单和回本条件满足时扩产。</p></article>
      <article><h3>政府</h3><strong>{money(government.cash_cents)}</strong><p>可用预算；本轮检查支出 {money(decision?.inspection_cost_cents ?? 0)}，经营支持 {money(support)}。</p></article>
    </div>
    <details open><summary>政府 · 最近决策与预算</summary>{decision ? <>
      <p>第 {decision.applies_round} 轮依据上一轮缺货 {decision.observed_stockout_orders.toLocaleString()} 单、集中度 {(decision.observed_hhi_ppm / 10000).toFixed(1)}% 决定：安排 {decision.inspection_cases} 份检查额度，向活跃企业发放支持 {money(support)}。</p>
      <p>期初 {money(decision.opening_cash_cents)} + 罚款收入 {money(government.round_fines_cents)} − 检查 {money(decision.inspection_cost_cents)} − 支持 {money(support)} − 消费者返还 {money(government.round_consumer_rebate_cents??0)} = 期末 {money(government.cash_cents)}。</p>
      <p>支持授权上限：{Object.entries(decision.support_by_company_cents).map(([id, value]) => `${id}：${money(value)}`).join(" · ") || "本轮没有经营支持。"}；实际支出 {money(support)}。</p>
    </> : <p>首次结算后显示政府决策。初始预算为外生资金，不自动补充。</p>}</details>
    <details open><summary>供应商 · 扩产与现金</summary><div className="v10-metric-grid">{suppliers.map(s => <article key={s.supplier_id}>
      <h3>{s.label}</h3><strong>基础产能 {(s.base_capacity_orders ?? 0).toLocaleString()} 单</strong>
      {s.account && <p>收款 {money(s.account.receipts_cents)} − 生产成本 {money(s.account.production_cost_cents)} − 投资 {money(s.account.investment_cents ?? 0)}；余额 {money(s.account.cash_cents)}。</p>}
      {s.investment_decision ? <><p>{investmentReasons[s.investment_decision.reason] ?? s.investment_decision.reason}。</p>{s.investment_decision.added_capacity_orders > 0 && <p>新增 {s.investment_decision.added_capacity_orders} 单产能，从第 {s.investment_decision.applies_round} 轮生效。</p>}
        {s.investment_decision.observed_requested_orders != null && <p>自身订单 {s.investment_decision.observed_requested_orders.toLocaleString()} 单 · 已结算单位毛利 {money(s.investment_decision.observed_unit_margin_cents ?? 0)} · 剩余 {s.investment_decision.remaining_rounds} 轮。</p>}</> : <p>首次结算后决定是否投资。</p>}
    </article>)}</div></details>
    <details open><summary>原料 · 付款、消耗与报损</summary>
      <p>企业付款 {money(invoices)} = 供应商收款 {money(receipts)}。原料按本轮易耗品处理；采购后未用于内部履约的部分报损，加工成本另计。</p>
      {materials.length > 0 ? <div className="v10-table-scroll"><table><caption>最近一轮原料账</caption><thead><tr><th>企业</th><th>采购付款</th><th>内部履约消耗</th><th>期末报损</th><th>收款方</th></tr></thead><tbody>{materials.map(o => <tr key={o.company_id}><td>{o.company_id}</td><td>{money(o.material!.payment_cents)}</td><td>{o.material!.used_orders.toLocaleString()} 件</td><td>{o.material!.wasted_orders.toLocaleString()} 件</td><td>{Object.entries(o.material!.payments_by_supplier).map(([id, value]) => `${id} ${money(value)}`).join(" / ") || "无采购"}</td></tr>)}</tbody></table></div> : <p>尚未结算。</p>}
    </details>
    <details><summary>消费者 · 预算与选择明细</summary><p>每轮进入新的消费群体，每人最多购买一单，预算按支付意愿配置；在预算范围内按价格、服务、品牌等效用选择，也可不买。退款返回消费者。</p>
      {consumers.length > 0 && <div className="v10-table-scroll"><table><caption>最近一轮消费群体选择</caption><thead><tr><th>群体</th><th>每人预算</th><th>人数</th><th>购买 / 不买 / 缺货</th><th>支付 / 退款</th></tr></thead><tbody>{consumers.map(c => <tr key={c.group_id}><td>{c.group_id}</td><td>{money(c.unit_budget_cents)}</td><td>{c.demand_orders}</td><td>{sum(Object.values(c.purchases_by_company))} / {c.voluntary_no_purchase_orders} / {c.stockout_orders}</td><td>{money(c.spending_cents)} / {money(c.refund_cents)}</td></tr>)}</tbody></table></div>}
    </details>
  </section>;
}
