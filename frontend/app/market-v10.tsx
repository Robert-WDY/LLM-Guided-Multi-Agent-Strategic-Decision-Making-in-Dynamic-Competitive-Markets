"use client";

import type { AgentConfig, LabConfig } from "./lab-model";
import type { LocalMarketState } from "./market-local";

export type HumanAction = { price: number; advertising: number; contribution: number; primarySupplier?: string; backupSupplier?: string; primaryShare?: number; procurementQuantity?: number; advanced?:Record<string,string|number>; publicMessage?:string; privateMessage?:string; messageRecipient?:string };
export type SavedUi = { entryMode?: "participant" | "observer" | "research"; config?: Partial<LabConfig>; agents?: AgentConfig[] };
export type CheckpointMeta = { episode_id: string; round: number; terminal: boolean; saved_at: string; config_sha256: string; recovery_required: boolean };
type QuoteDecision = { policy_version: string; observed_round: number; applies_round: number; previous_price_cents: number; quoted_price_cents: number; observed_sales_orders: number; observed_capacity_orders: number; utilization_ppm: number; reason: string };
type Supplier = { supplier_id: string; label: string; unit_price_cents: number; reliability_ppm: number; available_capacity_orders: number; disrupted: boolean; last_settled_unit_price_cents?: number; quote_decision?: QuoteDecision };
const quoteReason: Record<string, string> = { high_utilization: "供给使用率高，上调报价", low_utilization: "供给使用率低，下调报价", target_band: "使用率在目标区间，维持报价", unavailable: "上一轮无可供容量，维持报价" };
type Procurement = { company_id: string; primary_supplier_id: string; backup_supplier_id: string | null; primary_supplier_share_ppm: number; requested_orders: number; fulfilled_orders: number; unfulfilled_orders: number; weighted_unit_input_price_cents: number };
export type V10State = LocalMarketState & {
  state_version: number;
  supply_chain?: { suppliers: Record<string, Supplier>; last_procurement_outcomes: Record<string, Procurement> } | null;
  welfare_accounting?: Record<string, number | string> | null;
};
const money = (cents: number) => new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", maximumFractionDigits: 2 }).format(cents / 100);

export function ProcurementControls({ action, change, state, disabled, companyId }: { action: HumanAction; change: (action: HumanAction) => void; state: V10State; disabled: boolean; companyId?: string }) {
  const suppliers = Object.values(state.supply_chain?.suppliers ?? {});
  if (!suppliers.length) return null;
  const primary = action.primarySupplier ?? "economy_supplier";
  return <fieldset className="procurement-controls" disabled={disabled}>
    <legend>采购组合</legend>
    {state.government && <label>采购数量<input type="number" min={0} max={companyId ? state.companies?.[companyId]?.operations?.base_capacity_orders : undefined} value={action.procurementQuantity ?? ""} placeholder="留空按自身产能请求" onChange={e => change({ ...action, procurementQuantity: e.target.value === "" ? undefined : Math.max(0, Math.floor(Number(e.target.value))) })} /><small>实际请求受采购前现金限制；未使用原料本轮报损。</small></label>}
    <p>主、备供应商按比例同时下单；不是缺货后自动补单。</p>
    <label>主供应商<select value={primary} onChange={e => change({ ...action, primarySupplier: e.target.value, backupSupplier: action.backupSupplier === e.target.value ? "" : action.backupSupplier })}>{suppliers.map(s => <option key={s.supplier_id} value={s.supplier_id}>{s.label} · {money(s.unit_price_cents)}/单</option>)}</select></label>
    <label>备供应商<select value={action.backupSupplier ?? ""} onChange={e => change({ ...action, backupSupplier: e.target.value })}><option value="">不分散采购</option>{suppliers.filter(s => s.supplier_id !== primary).map(s => <option key={s.supplier_id} value={s.supplier_id}>{s.label}</option>)}</select></label>
    <label>主供应商占比 · {action.backupSupplier ? action.primaryShare ?? 50 : 100}%<input type="range" min={0} max={100} step={5} disabled={!action.backupSupplier} value={action.backupSupplier ? action.primaryShare ?? 50 : 100} onChange={e => change({ ...action, primaryShare: Number(e.target.value) })} /></label>
  </fieldset>;
}

export function V10Results({ state }: { state: V10State }) {
  const suppliers = Object.values(state.supply_chain?.suppliers ?? {});
  const outcomes = Object.values(state.supply_chain?.last_procurement_outcomes ?? {});
  const welfare = state.welfare_accounting;
  const metrics = [
    ["consumer_surplus", "消费者剩余"], ["downstream_producer_surplus", "下游企业利润"], ["upstream_producer_surplus", "上游供应商利润"],
    ["government_net_budget", "政府净预算"], ["total_economic_welfare", "总经济福利"],
  ];
  return <section className="card v10-results" aria-label="采购与经济福利">
    <div className="mini-panel-head"><span>采购与经济福利</span><b>后端结算数据</b></div>
    <p>供应商容量对应当前决策状态；采购结果和当期福利对应最近一次已完成结算。金额为模拟市场人民币。</p>
    {suppliers.length > 0 && <div className="v10-metric-grid">{suppliers.map(s => <article key={s.supplier_id}><h3>{s.label}</h3><strong>{money(s.unit_price_cents)} / 单</strong><p>可供 {s.available_capacity_orders.toLocaleString()} 单 · 可靠性 {(s.reliability_ppm / 10000).toFixed(0)}%</p><b>{s.disrupted ? "本轮供应中断" : "正常供给"}</b>{s.quote_decision && <details><summary>自主报价依据 · 第 {s.quote_decision.applies_round} 轮起生效</summary><p>第 {s.quote_decision.observed_round} 轮已结算销量 {s.quote_decision.observed_sales_orders} / 可供 {s.quote_decision.observed_capacity_orders} 单，使用率 {(s.quote_decision.utilization_ppm / 10000).toFixed(1)}%。{quoteReason[s.quote_decision.reason] ?? s.quote_decision.reason}。</p><p>{money(s.quote_decision.previous_price_cents)} → {money(s.quote_decision.quoted_price_cents)} / 单（受成本底价和最高报价约束）</p><p>基于上一轮经营结果的确定性策略；不读取下一轮冲击，也不调用模型。</p></details>}{s.last_settled_unit_price_cents != null && <p>最近结算单价 {money(s.last_settled_unit_price_cents)}</p>}</article>)}</div>}
    {outcomes.length > 0 ? <div className="v10-table-scroll"><table><caption>最近一轮采购结算</caption><thead><tr><th>企业</th><th>主 / 备供应商</th><th>主占比</th><th>需求 / 到货 / 缺货</th><th>平均进价</th></tr></thead><tbody>{outcomes.map(o => <tr key={o.company_id}><td>{o.company_id}</td><td>{o.primary_supplier_id} / {o.backup_supplier_id ?? "无"}</td><td>{o.primary_supplier_share_ppm / 10000}%</td><td>{o.requested_orders} / {o.fulfilled_orders} / {o.unfulfilled_orders}</td><td>{money(o.weighted_unit_input_price_cents)}</td></tr>)}</tbody></table></div> : <p>尚未发生采购结算。</p>}
    {welfare && state.state_version > 0 ? <><div className="v10-metric-grid">{metrics.map(([key, label]) => <article key={key}><h3>{label}</h3><strong>{money(Number(welfare[`round_${key}_cents`] ?? 0))}</strong><p>累计 {money(Number(welfare[`cumulative_${key}_cents`] ?? 0))}</p></article>)}</div><p>当期缺货外部性 {money(Number(welfare.round_stockout_externality_cents ?? 0))} · 退出外部性 {money(Number(welfare.round_business_exit_externality_cents ?? 0))} · 服务连续性 {Number(welfare.round_service_continuity_orders ?? 0)} 单</p><p>总福利 = 消费者剩余 + 上下游利润 + 政府净预算 − 外部性成本。它与旧版福利代理指标含义不同。需求估值与人格权重仍是合成研究参数。</p></> : <p>首次结算后显示福利核算。</p>}
  </section>;
}

export function SavedExperiments({ episodes, busy, currentId, recovery, onRefresh, onSave, onRestore, onExport, onRecover, message, configHash }: {
  episodes: CheckpointMeta[]; busy: boolean; currentId: string; recovery: boolean; message: string; configHash: string;
  onRefresh: () => void; onSave: () => void; onRestore: (id: string) => void; onExport: () => void; onRecover: () => void;
}) {
  return <section className="card saved-experiments" aria-label="保存与恢复">
    <div className="checkpoint-toolbar"><div><h2>我的实验 · 保存与恢复</h2><p>完成操作后自动落盘，可在关闭浏览器或重启后端后继续。</p></div><div className="checkpoint-actions"><button type="button" disabled={busy} onClick={onRefresh}>读取存档</button><button type="button" disabled={busy || !currentId} onClick={onSave}>保存当前实验</button><button type="button" disabled={busy || !currentId} onClick={onExport}>导出实验 JSON</button></div></div>
    {message && <p role="status">{message}</p>}
    {recovery && <div role="alert"><p>上次执行中断，当前实验已暂停。恢复会保留已接受动作，用规则补齐缺失动作并结算该轮，不会重试模型。</p><button type="button" disabled={busy} onClick={onRecover}>结束中断轮（缺失动作使用规则）</button></div>}
    {episodes.length > 0 && <details open={!currentId}><summary>已保存 {episodes.length} 个实验</summary><ul className="checkpoint-list">{episodes.map(e => <li key={e.episode_id}><div><strong>{e.episode_id}</strong><small>{e.terminal ? "已结束" : `第 ${e.round} 轮待决策`} · {new Date(e.saved_at).toLocaleString("zh-CN")}{e.recovery_required ? " · 执行曾中断" : ""}{configHash && e.config_sha256 !== configHash ? " · 需切换原配置" : ""}</small></div><button type="button" disabled={busy || Boolean(configHash && e.config_sha256 !== configHash)} onClick={() => onRestore(e.episode_id)}>恢复实验</button></li>)}</ul></details>}
  </section>;
}

export function BackendEvidence({ evidence, onRefresh, busy }: { evidence: Record<string, unknown> | null; onRefresh: () => void; busy: boolean }) {
  return <section className="card v10-results"><h2>当前实验的已记录证据</h2><p>研究控制台可查看结算、通信与协调器记录；不以演示信念或策略填充缺失记录。记录包含各公司的控制台信息，请在导出前确认研究用途。</p><button type="button" disabled={busy} onClick={onRefresh}>刷新真实记录</button>{evidence ? <details open><summary>结算与执行记录</summary><pre className="evidence-json">{JSON.stringify(evidence, null, 2)}</pre></details> : <p>点击刷新读取当前实验记录。</p>}</section>;
}
