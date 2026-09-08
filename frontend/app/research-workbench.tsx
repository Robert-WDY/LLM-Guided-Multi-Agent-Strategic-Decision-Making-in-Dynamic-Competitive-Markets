"use client";
import {useCallback,useEffect,useState} from "react";
import {AdvisorResult} from "./advisor-v16";
import {RecordedFields} from "./research-live";
type Mode="robust_company"|"robust_profit"|"robust_welfare"|"advisor_company"|"advisor_profit"|"advisor_welfare"|"rule"|"learning"|"neural"|"model"|"theory_best_response"|"theory_maximin"|"theory_tft"|"theory_grim"|"theory_wsls";
type Variant={label:string;modes:Record<string,Mode>;fixed:Record<string,string>;overrides:Record<string,number>};
type Job={id:string;status:string;created:string;error?:string;payload:{variants:Variant[]};progress:Record<string,unknown>;results:Record<string,unknown>[];statistics:{groups:{variant:string;n:number;mean_welfare_cents:number;std_welfare_cents:number|null;mean_company_profit_cents:number;mean_supplier_profit_cents:number;mean_consumer_surplus_cents:number}[];paired_comparisons:Record<string,unknown>[]}};
const otherActors=["economy_supplier","resilient_supplier","consumers","government"];
const names:Record<string,string>={company_A:"企业 A",company_B:"企业 B",company_C:"企业 C",company_D:"企业 D",economy_supplier:"经济供应商",resilient_supplier:"韧性供应商",consumers:"消费者群体",government:"政府"};
const controls:Record<Mode,string>={robust_company:"v17反击感知：公司阶段",robust_profit:"v17反击感知：公司利润",robust_welfare:"v17反击感知：社会福利",advisor_company:"目标建议：公司阶段",advisor_profit:"目标建议：公司利润",advisor_welfare:"目标建议：社会福利",rule:"规则",learning:"共同学习",neural:"已训练神经策略",model:"真实模型",theory_best_response:"公开预测最佳回应",theory_maximin:"公开预测保底策略",theory_tft:"公开价格互惠",theory_grim:"价格触发持续竞争",theory_wsls:"收益赢留输换"};
const statusNames:Record<string,string>={queued:"排队中",running:"运行中",complete:"已完成",cancelled:"已暂停",interrupted:"服务中断",failed:"失败"};
const money=(c:number)=>new Intl.NumberFormat("zh-CN",{style:"currency",currency:"CNY"}).format(c/100);
type TheoryDecision={mode:string;signal:string;public_price_change:number;selected_option:string;criterion?:string;scope?:string;belief_counts?:Record<string,number>;payoff_table?:{option:string;scenario_profits_cents:number[];expected_profit_cents:number;worst_profit_cents:number}[]};
type AuditRow={caseName:string;round:number;actor:string;decision:TheoryDecision};
export function ResearchWorkbench({apiUrl,token}:{apiUrl:string;token:string}){
 const [companyCount,setCompanyCount]=useState(4);
 const actors=[...Array.from({length:companyCount},(_,i)=>`company_${String.fromCharCode(65+i)}`),...otherActors];
 const [jobs,setJobs]=useState<Job[]>([]),[error,setError]=useState(""),[busy,setBusy]=useState(false);
 const [seeds,setSeeds]=useState("260101,260102"),[rounds,setRounds]=useState(10),[mode,setMode]=useState<Mode>("rule");
 const [modes,setModes]=useState<Record<string,Mode>>({}),[parameter,setParameter]=useState(""),[values,setValues]=useState(""),[compare,setCompare]=useState(true),[authorize,setAuthorize]=useState(false);
 const [model,setModel]=useState("deepseek-v4-flash");
 const [selected,setSelected]=useState<string[]>([]);
 const [advisorAudits,setAdvisorAudits]=useState<Record<string,{caseName:string;round:number;actor:string;decision:unknown}[]>>({});
 const [audits,setAudits]=useState<Record<string,AuditRow[]>>({});
 const comparisonRows=selected.flatMap(id=>{
   const job=jobs.find(j=>j.id===id);return job?.statistics.groups.map(g=>({id,label:`${id.slice(0,8)} · ${g.variant}`,group:g,rows:job.results.filter(r=>r.variant===g.variant)}))||[];
 });
 const call=useCallback(async(path:string,body?:unknown)=>{
   const r=await fetch(`${apiUrl}/v1/controller/workbench/${path}`,{method:body===undefined?"GET":"POST",headers:{"Content-Type":"application/json",...(token?{"X-Controller-Token":token}:{})},...(body===undefined?{}:{body:JSON.stringify(body)})});
   const data=await r.json() as {detail?:unknown;jobs:Job[]};if(!r.ok)throw new Error(typeof data.detail==="string"?data.detail:JSON.stringify(data.detail));return data;
 },[apiUrl,token]);
 const refresh=useCallback(async()=>{try{const data=await call("jobs");setJobs(data.jobs);}catch(e){setError(String(e));}},[call]);
 useEffect(()=>{const initial=setTimeout(()=>void refresh(),0);const timer=setInterval(()=>void refresh(),2000);return()=>{clearTimeout(initial);clearInterval(timer);};},[refresh]);
 const chosen=Object.fromEntries(actors.map(a=>[a,modes[a]||mode])) as Record<string,Mode>;
 const paid=Object.values(chosen).includes("model");
 async function submit(){
   setBusy(true);setError("");
   try{
    const parsed=seeds.split(/[,，\s]+/).filter(Boolean).map(Number);
    if(!parsed.length||parsed.some(s=>!Number.isSafeInteger(s)||s<0))throw new Error("种子应为非负整数，用逗号分隔。");
    const variants:Variant[]=[];
    if(compare)variants.push({label:"规则基线",modes:{},fixed:{},overrides:{}});
    const params=parameter?values.split(/[,，\s]+/).filter(Boolean).map(Number):[null];
    if(!params.length||params.some(v=>v!==null&&(!Number.isSafeInteger(v)||v<0)))throw new Error("参数应为非负整数。");
    for(const value of params)variants.push({label:value===null?"所选四方策略":`${parameter}=${value}`,modes:chosen,fixed:{},overrides:value===null?{}:{[parameter]:value}});
    await call("jobs",{request_id:crypto.randomUUID(),seeds:parsed,rounds,company_count:companyCount,variants,authorize_real:authorize,model_name:model});await refresh();
   }catch(e){setError(String(e));}finally{setBusy(false);}
 }
 async function control(id:string,action:string){try{await call(`jobs/${id}/${action}`,{});await refresh();}catch(e){setError(String(e));}}
 async function readAudit(id:string){try{const response=await fetch(`${apiUrl}/v1/controller/workbench/jobs/${id}/export`,{headers:token?{"X-Controller-Token":token}:{}});if(!response.ok)throw new Error(await response.text());const data=await response.json() as {cases:Record<string,{decisions:{decisions:Record<string,{observation:{round:number};theory_decision?:TheoryDecision}>}[]}>};const rows:AuditRow[]=[];for(const [caseName,c] of Object.entries(data.cases))for(const round of c.decisions)for(const [actor,d] of Object.entries(round.decisions))if(d.theory_decision)rows.push({caseName,round:d.observation.round,actor,decision:d.theory_decision});setAudits({...audits,[id]:rows.slice(-12)});}catch(e){setError(String(e));}}
 async function readAdvisorAudit(id:string){try{const response=await fetch(`${apiUrl}/v1/controller/workbench/jobs/${id}/export`,{headers:token?{"X-Controller-Token":token}:{}});if(!response.ok)throw new Error(await response.text());const data=await response.json() as {cases:Record<string,{decisions:{decisions:Record<string,{observation:{round:number};advisor_decision?:unknown}>}[]}>};const rows:{caseName:string;round:number;actor:string;decision:unknown}[]=[];for(const [caseName,c] of Object.entries(data.cases))for(const round of c.decisions)for(const [actor,d] of Object.entries(round.decisions))if(d.advisor_decision)rows.push({caseName,round:d.observation.round,actor,decision:d.advisor_decision});setAdvisorAudits({...advisorAudits,[id]:rows.slice(-5)});}catch(e){setError(String(e));}}
 async function download(id:string){try{const response=await fetch(`${apiUrl}/v1/controller/workbench/jobs/${id}/export`,{headers:token?{"X-Controller-Token":token}:{}});if(!response.ok)throw new Error(await response.text());const url=URL.createObjectURL(await response.blob());const a=document.createElement("a");a.href=url;a.download=`market-experiment-${id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){setError(String(e));}}
 return <section className="card research-live workbench" aria-label="四方批量实验工作台"><h1>四方批量实验工作台</h1><p>配置策略与参数，对每个种子成对比较。单队列逐局运行，完成回合自动落盘；关闭页面后继续。真实模型使用受约束的策略选项，各主体只读取自身和公开信息。</p>
 <div className="workbench-controls"><label>实验种子<input value={seeds} onChange={e=>setSeeds(e.target.value)}/></label><label>每局回合数<input type="number" min="5" max="60" value={rounds} onChange={e=>setRounds(Number(e.target.value))}/></label>
 <label>公司数量<input type="number" min="2" max="10" value={companyCount} onChange={e=>setCompanyCount(Math.max(2,Math.min(10,Math.trunc(Number(e.target.value))||2)))}/></label><button onClick={()=>setModes({...modes,...Object.fromEntries(actors.filter(a=>a.startsWith("company_")).map(a=>[a,"robust_profit" as Mode]))})}>全部企业使用v17利润建议</button>
 <label>四方默认控制<select value={mode} onChange={e=>setMode(e.target.value as Mode)}>{Object.entries(controls).filter(([k])=>!k.startsWith('theory_')&&!k.startsWith('advisor_')&&!k.startsWith('robust_')).map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
 {actors.map(a=><label key={a}>{names[a]||`企业 ${a.slice(-1)}`}控制<select value={modes[a]||""} onChange={e=>setModes({...modes,[a]:e.target.value as Mode})}><option value="">跟随默认</option>{Object.entries(controls).filter(([k])=>a.startsWith('company_')||!k.startsWith('theory_')&&!k.startsWith('advisor_')&&!k.startsWith('robust_')).map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>)}
 <label>扫描参数<select value={parameter} onChange={e=>setParameter(e.target.value)}><option value="">不扫描参数</option><option value="supplier_cash">供应商初始现金（分）</option><option value="inventory_target">供应商库存目标（ppm）</option><option value="government_program">政府每轮项目预算（分）</option><option value="government_cash">政府初始现金（分）</option></select></label>
 {parameter&&<label>参数值列表<input value={values} placeholder="例如 100000,20000000" onChange={e=>setValues(e.target.value)}/></label>}
 <label><input type="checkbox" checked={compare} onChange={e=>setCompare(e.target.checked)}/>加入同种子规则基线</label>
 {paid&&<><label>模型供应商<select value={model} onChange={e=>setModel(e.target.value)}><option value="deepseek-v4-flash">DeepSeek v4 Flash</option><option value="doubao-seed-2-0-lite-260215">豆包 Seed 2.0 Lite</option></select></label><label><input type="checkbox" checked={authorize} onChange={e=>setAuthorize(e.target.checked)}/>允许本批真实调用，共享原 10 元累计上限</label><p>失败和未知调用保留额度。剩余预算不足会停止；神经策略和共同学习不收取模型调用费用。</p></>}
 </div><p>神经策略已训练并通过留出实验，但未优于规则福利基线；可用于研究对照。修改经济参数后需重新训练，系统会拒绝不兼容的权重。</p>
 <p>v17企业建议每轮读取此前公开价格，预测1轮、2情景、最多8候选、1个非基线入围者，最多500至700步。各企业同时依据回合开始状态决策，完成回合后才更新历史。v16目标建议每轮按最新合法观察重新规划；默认预测2轮、2个情景、最多8个候选，计算上限200步。公司阶段自动更新。详细多人诊断请使用博弈实验室。实际动作和每轮依据随实验保存。</p>
 <button disabled={busy||(paid&&!authorize)} onClick={()=>void submit()}>加入实验队列</button> <button onClick={()=>void refresh()}>刷新队列</button>
 {error&&<p role="alert">{error}</p>}
 <h2>实验队列与比较</h2>{!jobs.length&&<p>还没有批量实验。</p>}
 {comparisonRows.length>0&&<section><h3>跨批次比较</h3><p>按勾选顺序，以第一组为基线；差值仅配对相同种子和轮数。参数和模型差异仍需按实验记录解释。</p><div className="v10-table-scroll"><table><thead><tr><th>批次与处理组</th><th>样本数</th><th>平均福利</th><th>配对数</th><th>配对福利差</th></tr></thead><tbody>{comparisonRows.map(row=>{
   const paired=row.rows.flatMap(r=>{const b=comparisonRows[0].rows.find(b=>b.seed===r.seed&&b.rounds===r.rounds&&(b.company_count||4)===(r.company_count||4));return b?[Number(r.welfare_cents)-Number(b.welfare_cents)]:[];});
   return <tr key={row.label}><td>{row.label}</td><td>{row.group.n}</td><td>{money(row.group.mean_welfare_cents)}</td><td>{paired.length}</td><td>{paired.length?money(paired.reduce((a,b)=>a+b,0)/paired.length):"无可比种子"}</td></tr>;
 })}</tbody></table></div></section>}
 {jobs.map(j=><section key={j.id} className="card"><h3>{statusNames[j.status]||j.status} · {j.id.slice(0,8)}</h3><label><input type="checkbox" checked={selected.includes(j.id)} onChange={e=>setSelected(e.target.checked?[...selected,j.id]:selected.filter(id=>id!==j.id))}/>加入跨批次比较 {j.id.slice(0,8)}</label><RecordedFields value={j.progress}/>{j.error&&<p role="alert">{j.error}</p>}
 {["running","queued"].includes(j.status)&&<button onClick={()=>void control(j.id,"cancel")}>安全暂停</button>}
 {["interrupted","cancelled","failed"].includes(j.status)&&<button onClick={()=>void control(j.id,"resume")}>从已存回合恢复</button>}
 {j.status==="failed"&&j.error?.startsWith("ActorModelValidationError:")&&<button onClick={()=>void control(j.id,"retry-schema")}>已完成但格式错误：再尝试一次（保留原费用）</button>}
 {j.status==="failed"&&!j.payload.variants.some(v=>Object.values(v.modes).includes("model"))&&<button onClick={()=>void control(j.id,"rebuild-free")}>保留故障文件并重新计算免费批次</button>}
 <button onClick={()=>void download(j.id)}>导出全部轨迹与模型记录</button>
 {j.payload.variants.some(v=>Object.values(v.modes).some(m=>m.startsWith('theory_')))&&<button onClick={()=>void readAudit(j.id)}>查看策略决策依据</button>}
 {j.payload.variants.some(v=>Object.values(v.modes).some(m=>m.startsWith('advisor_')||m.startsWith('robust_')))&&<button onClick={()=>void readAdvisorAudit(j.id)}>查看目标建议依据</button>}
 {advisorAudits[j.id]&&<section aria-label="已执行目标建议"><h4>最近5条已执行目标建议</h4>{!advisorAudits[j.id].length&&<p>尚无已执行建议。</p>}{advisorAudits[j.id].map((r,i)=><details key={i} open={i===advisorAudits[j.id].length-1}><summary>{r.caseName} · 第{r.round}轮 · {names[r.actor]||`企业 ${r.actor.slice(-1)}`}</summary><AdvisorResult value={r.decision}/></details>)}</section>}
 {audits[j.id]&&<section aria-label="市场策略决策依据"><h4>最近12条已记录策略决策</h4>{!audits[j.id].length&&<p>尚无已记录策略决策。</p>}{audits[j.id].map((r,i)=><details key={i} open={i===audits[j.id].length-1}><summary>{r.caseName} · 第{r.round}轮 · {names[r.actor]||`企业 ${r.actor.slice(-1)}`} · {controls[r.decision.mode as Mode]} → {r.decision.selected_option}</summary><p>公开价格信号：{r.decision.signal}；对手平均价格变化 {(100*r.decision.public_price_change).toFixed(2)}%。实际选择 {r.decision.selected_option}。</p>{r.decision.belief_counts&&<p>对手降价/基线/提价情景计数：{['value','balanced','margin'].map(k=>r.decision.belief_counts?.[k]).join(' / ')}。{r.decision.criterion==='expected_profit_cents'?'按经验频率加权的预测利润选择。':'按三个情景中的最低预测利润选择。'}</p>}{r.decision.payoff_table&&<div className="v10-table-scroll"><table><thead><tr><th>自身选项</th><th>对手降价</th><th>对手基线</th><th>对手提价</th><th>期望利润</th><th>最低利润</th></tr></thead><tbody>{r.decision.payoff_table.map(p=><tr key={p.option}><td>{p.option}{p.option===r.decision.selected_option?' ✓':''}</td>{p.scenario_profits_cents.map((v,k)=><td key={k}>{money(v)}</td>)}<td>{money(p.expected_profit_cents)}</td><td>{money(p.worst_profit_cents)}</td></tr>)}</tbody></table></div>}<p>{r.decision.scope||'根据公开价格历史及自身利润更新状态；不读取对手私有现金或人格。这是有限启发式，不是市场均衡证明。'}</p></details>)}</section>}
 {!!j.statistics.groups.length&&<div className="v10-table-scroll"><table><thead><tr><th>处理组</th><th>完成局数</th><th>平均福利</th><th>福利标准差</th><th>企业利润</th><th>供应商利润</th><th>消费者剩余</th></tr></thead><tbody>{j.statistics.groups.map(g=><tr key={g.variant}><td>{g.variant}</td><td>{g.n}</td><td>{money(g.mean_welfare_cents)}</td><td>{g.std_welfare_cents===null?"样本不足":money(g.std_welfare_cents)}</td><td>{money(g.mean_company_profit_cents)}</td><td>{money(g.mean_supplier_profit_cents)}</td><td>{money(g.mean_consumer_surplus_cents)}</td></tr>)}</tbody></table></div>}
 <details><summary>同种子差值与完成结果</summary><RecordedFields value={{paired:j.statistics.paired_comparisons,results:j.results}}/></details><p>统计来自已完成局；中断或失败局不计入均值。小样本差异不能解释为现实市场最优。</p></section>)}</section>;
}
