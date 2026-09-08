"use client";
import {RecordedFields} from "./research-live";
type Ledger = {inventory_orders:number;debt_cents:number;lender_cash_cents:number;bankrupt:boolean;contracts:Record<string,unknown>[];audit:Record<string,unknown>|null};
export type StrategicSupply = {supply_chain?:{suppliers:Record<string,{label:string;strategic_ledger?:Ledger}>}|null};
const money=(v:number)=>new Intl.NumberFormat("zh-CN",{style:"currency",currency:"CNY"}).format(v/100);
export function SupplierStrategies({state}:{state:StrategicSupply}) {
  const suppliers=Object.entries(state.supply_chain?.suppliers??{}).filter(([,s])=>s.strategic_ledger);
  if(!suppliers.length)return null;
  return <section className="card research-live" aria-label="供应商合同与财务"><h2>供应商合同、库存与债务</h2><p>企业采购仍预付；供应商库存跨轮保存并发生损耗。贷款来自有限资金的债权方，利息和坏账在双方分别记录。</p>{suppliers.map(([id,s])=>{const d=s.strategic_ledger!;return <section key={id}><h3>{s.label} · {d.bankrupt?"已破产，停止供货":"经营中"}</h3><p>库存 {d.inventory_orders} 件 · 未偿本金 {money(d.debt_cents)} · 债权方剩余现金 {money(d.lender_cash_cents)}</p><h4>锁价与优先供货合同</h4>{d.contracts.length?<div className="v10-table-scroll"><table><thead><tr><th>企业</th><th>每轮预留</th><th>单价</th><th>生效回合</th></tr></thead><tbody>{d.contracts.map((c,i)=><tr key={i}><td>{String(c.company_id)}</td><td>{Number(c.quantity_orders)}</td><td>{money(Number(c.unit_price_cents))}</td><td>{Number(c.start_round)}—{Number(c.end_round)}</td></tr>)}</tbody></table></div>:<p>无有效或本轮结束合同。</p>}<details><summary>议价、履约与资金核对</summary><RecordedFields value={d.audit}/></details></section>;})}</section>;
}
