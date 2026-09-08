"use client";
import { useEffect, useState } from "react";

type Row = Record<string, unknown>;
const obj = (value: unknown): Row => value && typeof value === "object" && !Array.isArray(value) ? value as Row : {};
const arr = (value: unknown): Row[] => Array.isArray(value) ? value.map(obj) : [];
const str = (value: unknown) => value == null ? "未记录" : String(value);
const labels: Record<string,string> = {public_state:"公开市场",private_state:"自身私有信息",self_state:"自身信息",market:"市场",companies:"企业",financial:"财务",commercial:"经营",operations:"运营",risk:"风险",brand:"品牌与服务",history:"历史",belief_before:"决策前信念",opponent_model:"对手模型",utility_inference:"效用推断",advisor_output:"策略建议",advisor_adoption:"建议采纳",repeated_game_strategy:"重复博弈",plan:"行动计划",situation_summary:"局势判断",strategy_summary:"策略说明",key_factors:"依据",expected_outcome:"预期结果",confidence_ppm:"置信度",price_cents:"价格",cash_balance_cents:"现金",round_profit_cents:"本轮利润",advertising_budget_cents:"广告投入",service_budget_cents:"服务投入",capacity_investment_cents:"扩产投入",resilience_budget_cents:"韧性投入",shared_resilience_contribution_cents:"共同韧性投入",threshold_project_contribution_cents:"阈值项目投入",procurement_quantity_orders:"采购数量",primary_supplier_id:"主供应商",backup_supplier_id:"备用供应商",primary_supplier_share_ppm:"主供应商份额",incident_response:"事故处理",mode:"模式",reason:"原因",reasons:"原因",message:"内容",status:"状态",recommendation:"建议",recommendations:"建议",evidence:"证据",assumptions:"假设",action:"动作",target_company_id:"目标企业",company_id:"企业",cash_delta_cents:"现金变化",sales_orders:"销量",market_share_ppm:"市场份额",result_analysis:"结果复盘",observed_effects:"实际影响",success_criteria_met:"目标完成情况",resolution_adjustments:"执行调整",cooperation:"合作",commitments:"承诺",outcomes:"履约结果",profit_cents:"利润",repair_budget_cents:"维修投入"};
function display(key: string, value: unknown) {
  if (value == null) return "未记录";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "number" && key.endsWith("_cents")) return new Intl.NumberFormat("zh-CN",{style:"currency",currency:"CNY"}).format(value/100);
  if (typeof value === "number" && key.endsWith("_ppm")) return `${(value/10000).toFixed(2)}%`;
  return String(value);
}
export function RecordedFields({value,depth=0}:{value:unknown;depth?:number}) {
  if (value == null) return <p className="compact-empty">此项未记录或未启用。</p>;
  if (typeof value !== "object") return <span>{String(value)}</span>;
  const entries = Array.isArray(value) ? value.map((v,i)=>[String(i+1),v] as const) : Object.entries(obj(value));
  if (!entries.length) return <p className="compact-empty">无记录。</p>;
  return <dl className="research-fields">{entries.map(([key,v]) => <div key={key}><dt>{labels[key] ?? key}</dt><dd>{v != null && typeof v === "object" ? <details open={depth<1}><summary>查看{Array.isArray(v) ? ` ${v.length} 项` : "明细"}</summary><RecordedFields value={v} depth={depth+1}/></details> : display(key,v)}</dd></div>)}</dl>;
}

export function ResearchLive({episodeId,apiUrl,token,view}:{episodeId:string;apiUrl:string;token:string;view:string}) {
  const [data,setData]=useState<Row|null>(null);
  const [selectedRound,setSelectedRound]=useState(0);
  const [company,setCompany]=useState("");
  const [revision,setRevision]=useState(0);
  const [error,setError]=useState("");
  const [loading,setLoading]=useState(true);
  useEffect(()=>{
    const controller=new AbortController();
    fetch(`${apiUrl}/v1/controller/episodes/${episodeId}/research-view${selectedRound?`?round_number=${selectedRound}`:""}`,{headers:token?{"X-Controller-Token":token}:{},signal:controller.signal})
      .then(async response=>{const payload=obj(await response.json());if(!response.ok)throw new Error(typeof payload.detail==="string"?payload.detail:`HTTP ${response.status}`);if(!controller.signal.aborted){setData(payload);setError("");}})
      .catch(e=>{if(!controller.signal.aborted)setError(e instanceof Error?e.message:"读取失败");})
      .finally(()=>{if(!controller.signal.aborted)setLoading(false);});
    return ()=>controller.abort();
  },[episodeId,apiUrl,token,selectedRound,revision]);
  const detail=obj(data?.detail), before=obj(detail.state_before), after=obj(detail.state_after);
  const companies=Object.keys(obj(after.companies));
  const companyId=companies.includes(company)?company:companies[0]??"";
  const trace=arr(detail.traces).find(t=>t.company_id===companyId);
  const messages=arr(detail.messages).filter(m=>m.channel==="public"||m.sender_company_id===companyId||(Array.isArray(m.recipients)&&m.recipients.includes(companyId)));
  const title=({observatory:"智能体观察",communication:"通信与履约",strategy:"信念与策略",replay:"逐轮回放"} as Record<string,string>)[view];
  return <section className="card research-live" aria-label={title}>
    <div className="checkpoint-toolbar"><div><h2>{title}</h2><p>来自本局已保存记录。观察页展示选中企业当时的输入；回放页是研究控制台全量账目。</p></div><button disabled={loading} onClick={()=>setRevision(v=>v+1)}>刷新本局记录</button></div>
    <div className="research-selectors"><label>结算回合<select aria-label="结算回合" value={selectedRound||Number(data?.selected_round)||0} onChange={e=>setSelectedRound(Number(e.target.value))}><option value={0}>最近一轮</option>{Array.isArray(data?.rounds)&&data.rounds.map(v=><option key={Number(v)} value={Number(v)}>第 {Number(v)} 轮</option>)}</select></label><label>查看企业<select aria-label="查看企业" value={companyId} onChange={e=>setCompany(e.target.value)}>{companies.map(id=><option key={id}>{id}</option>)}</select></label></div>
    {error&&<p role="alert">{error}</p>}{loading&&<p role="status">正在读取已保存记录…</p>}
    {!data?.detail&&!loading&&<p>尚未结算。完成一轮后即可查看。</p>}
    {data?.detail!=null&&<>
      {!detail.trace_available&&<p role="status">这个历史回合没有保存完整智能体轨迹；仍可查看实际结算，不补造信念或建议。</p>}
      {view==="observatory"&&<><p>决策来源：{str(trace?.model_name??trace?.resolution_source)} · 状态：{str(trace?.decision_status)}</p><p>观察校验：{str(trace?.observation_hash)}</p><RecordedFields value={trace?.observation}/>{trace?.observation==null&&<p>该席位未记录模型观察。规则结算数据请在回放中查看。</p>}</>}
      {view==="communication"&&<><h3>选中企业可见消息</h3>{messages.length?messages.map((m,i)=><article className="research-message" key={str(m.message_id??i)}><b>{str(m.sender_company_id??m.sender_id)} → {m.channel==="public"?"公开":Array.isArray(m.recipients)?m.recipients.join(", "):"未记录"}</b><p>{str(m.content??m.text)}</p></article>):<p>本轮没有该企业可见的消息；沉默不等于达成协议。</p>}<h3>承诺与实际履约</h3><RecordedFields value={detail.cooperation}/></>}
      {view==="strategy"&&<>{["plan","belief_before","opponent_model","utility_inference","advisor_output","advisor_adoption","repeated_game_strategy"].map(key=><details key={key} open={key==="plan"||key==="advisor_output"}><summary>{labels[key]}</summary><RecordedFields value={key==="plan"?trace?.planner_output:trace?.[key]}/></details>)}<h3>最终动作与执行调整</h3><RecordedFields value={obj(detail.final_actions)[companyId]}/><RecordedFields value={trace?.resolution_adjustments}/></>}
      {view==="replay"&&<><h3>企业经营轨迹</h3><div className="v10-table-scroll"><table><caption>{companyId} · 各轮真实结算</caption><thead><tr><th>回合</th><th>价格</th><th>销量</th><th>本轮利润</th><th>现金</th></tr></thead><tbody>{arr(data.timeline).map(row=>{const c=obj(obj(row.companies)[companyId]);return <tr key={Number(row.round)}><td>{Number(row.round)}</td><td>{display("price_cents",c.price_cents)}</td><td>{display("sales_orders",c.sales_orders)}</td><td>{display("profit_cents",c.profit_cents)}</td><td>{display("cash_cents",c.cash_cents)}</td></tr>;})}</tbody></table></div><div className="research-comparison"><section><h3>本轮结算前</h3><RecordedFields value={obj(before.companies)[companyId]}/></section><section><h3>本轮结算后</h3><RecordedFields value={obj(after.companies)[companyId]}/></section></div><h3>结果复盘</h3><RecordedFields value={trace?.result_analysis}/><details><summary>结算一致性检查</summary><RecordedFields value={detail.invariants}/></details></>}
    </>}
  </section>;
}
