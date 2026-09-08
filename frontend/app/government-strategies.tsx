"use client";
import {RecordedFields} from "./research-live";
type Government={cash_cents:number;round_consumer_rebate_cents?:number;round_matched_support_cents?:number;policy_memory?:string;last_decision?:{strategic_policy?:string;inspection_cost_cents:number}};
const labels:Record<string,string>={reserve:"保留预算",enforce:"加强执法",resilience:"定向韧性报销",consumer:"消费者返还",balanced:"组合政策"};
const money=(v:number)=>new Intl.NumberFormat("zh-CN",{style:"currency",currency:"CNY"}).format(v/100);
export function GovernmentStrategies({state}:{state:{government?:Government|null}}){
 const g=state.government;if(!g?.last_decision?.strategic_policy)return null;
 const p=JSON.parse(g.last_decision.strategic_policy),m=JSON.parse(g.policy_memory||"{}");
 return <section className="card research-live" aria-label="政府自主政策"><h2>政府自主政策 · {labels[p.option]||p.option}</h2><p>剩余预算 {money(g.cash_cents)} · 本轮执法 {money(g.last_decision.inspection_cost_cents)} · 罚款倍数 {p.fine_multiplier_ppm/1000000}</p><p>韧性实报实销 {money(g.round_matched_support_cents||0)} · 消费者返还 {money(g.round_consumer_rebate_cents||0)}</p><p>补贴在双方账目抵消。消费返还按成交后净支出计算，优先低预算群体；当前模型未引入家庭跨期储蓄，因此返还本身不会提高总福利。</p><p>已学习 {m.observations||0} 轮；自动策略依据已结算福利进行探索和选择，小样本均值不代表因果效果或长期最优。</p><details><summary>政策授权、目标与学习记录</summary><RecordedFields value={{policy:p,memory:m}}/></details></section>;
}
