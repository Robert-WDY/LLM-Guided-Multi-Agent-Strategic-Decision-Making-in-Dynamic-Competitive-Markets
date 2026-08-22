"use client";

/* eslint-disable jsx-a11y/no-noninteractive-element-interactions, react/no-unescaped-entities */

import { useEffect, useMemo, useState } from "react";
import {
  AgentConfig,
  AgentRuntimeView,
  advanceDemoRound,
  DEFAULT_AGENTS,
  DEMO_AGENTS,
  DEMO_MESSAGES,
  DEMO_REPLAY,
  LabConfig,
  PERSONAS,
  PersonaKey,
  PROFIT_SERIES,
  ViewId,
} from "./lab-model";

const API_URL = process.env.NEXT_PUBLIC_MARKET_API_URL ?? "http://localhost:8010/api";

type EntryMode = "participant" | "observer" | "research";

const NAV_ITEMS: Array<{ id: ViewId; index: string; label: string; hint: string }> = [
  { id: "live", index: "01", label: "实时现场", hint: "回合进行" },
  { id: "observatory", index: "02", label: "智能体观察", hint: "输入与判断" },
  { id: "communication", index: "03", label: "通信记录", hint: "消息" },
  { id: "strategy", index: "04", label: "信念与策略", hint: "博弈分析" },
  { id: "market", index: "05", label: "市场结果", hint: "经营结果" },
  { id: "replay", index: "06", label: "回合记录", hint: "过程重建" },
  { id: "report", index: "07", label: "实验报告", hint: "已完成" },
];

const ENTRY_META: Record<EntryMode, { label: string; eyebrow: string }> = {
  participant: { label: "个人体验", eyebrow: "参与者" },
  observer: { label: "观察模式", eyebrow: "观察者" },
  research: { label: "研究控制台", eyebrow: "研究员" },
};

const PERSONA_WEIGHT_LABELS: Record<string, string> = { profit: "利润", growth: "增长", risk: "风险控制", cash: "现金安全", social: "共同收益" };
const PERSONA_TRAIT_LABELS: Record<string, string> = { timeDiscount: "长期重视程度", riskAversion: "风险厌恶", cooperation: "合作倾向", opportunism: "机会主义倾向" };
const STRATEGY_LABELS: Record<string, string> = { growth: "增长导向", profit: "利润导向", defensive: "防御导向" };
const ACTION_LABELS: Record<string, string> = { cut: "降价", maintain: "维持", raise: "提价" };
const PLAN_STATUS_LABELS: Record<string, string> = { done: "已完成", active: "进行中", queued: "待处理" };

const DRIVER_MODELS: Record<AgentConfig["driver"], string> = {
  human: "Human",
  doubao: "Doubao Seed 2.0 Lite",
  deepseek: "DeepSeek V3",
  rule: "Deterministic Rule",
};

const MARKET_LABELS: Record<LabConfig["marketType"], string> = {
  balanced: "均衡市场",
  high_demand: "高需求市场",
  supply_crisis: "供应紧张市场",
  disaster: "高灾害风险场景",
  public_goods: "公共品合作场景",
};

type BackendCompany = {
  financial: { cash_balance_cents: number; round_profit_cents: number };
  commercial: { price_cents: number; market_share_ppm: number };
  risk: { resilience_ppm: number };
};

type BackendEpisode = {
  state: {
    episode_id: string;
    episode_seed: number;
    round: number;
    max_rounds: number;
    state_hash: string;
    companies: Record<string, BackendCompany>;
  };
  agent_tokens?: Record<string, string>;
};

type RuntimeMode = "draft" | "demo" | "backend";

function formatMoney(cents: number, compact = false) {
  if (compact) return `¥${new Intl.NumberFormat("zh-CN", { notation: "compact", maximumFractionDigits: 1 }).format(cents / 100)}`;
  return new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", maximumFractionDigits: 0 }).format(cents / 100);
}

function formatPercent(value: number) {
  return `${value.toFixed(1)}%`;
}

function Toggle({ checked, label, onChange }: { checked: boolean; label: string; onChange: (value: boolean) => void }) {
  return <button type="button" className={`switch${checked ? " on" : ""}`} role="switch" aria-checked={checked} aria-label={label} onClick={() => onChange(!checked)}><i /></button>;
}

function StatusDot({ tone = "ok" }: { tone?: "ok" | "warn" | "off" }) {
  return <i className={`status-dot ${tone}`} aria-hidden="true" />;
}

function AppShell({ active, setActive, children, runtimeMode, backendOnline, episodeId, entryMode, completed, onHome }: { active: ViewId; setActive: (view: ViewId) => void; children: React.ReactNode; runtimeMode: RuntimeMode; backendOnline: boolean | null; episodeId: string; entryMode: EntryMode; completed: boolean; onHome: () => void }) {
  if (active === "home") return <main className="landing-shell">{children}</main>;
  const nav = NAV_ITEMS.find((item) => item.id === active);
  if (active === "setup") return <main className="configuration-shell"><header className="simple-workspace-header"><button type="button" onClick={onHome}>← 返回主页</button><div><span>{ENTRY_META[entryMode].eyebrow} · 环境配置</span><strong>{ENTRY_META[entryMode].label}</strong></div><b>博弈实验室</b></header>{children}</main>;
  const allowed = entryMode === "participant" ? ["live", "replay", "report"] : entryMode === "observer" ? ["live", "observatory", "communication", "market", "replay", "report"] : ["live", "observatory", "communication", "strategy", "market", "replay", "report"];
  const visibleNavigation = NAV_ITEMS.filter((item) => allowed.includes(item.id) && (item.id !== "report" || completed));
  return <main className="lab-shell">
    <aside className="lab-sidebar">
      <button className="lab-brand" type="button" onClick={onHome}><span className="brand-glyph">博弈</span><div><strong>多智能体博弈实验室</strong><small>返回主入口</small></div></button>
      <div className="workspace-identity"><span>{ENTRY_META[entryMode].eyebrow}</span><strong>{ENTRY_META[entryMode].label}</strong><button type="button" onClick={() => setActive("setup")}>修改环境</button></div>
      <nav aria-label="当前模式导航">{visibleNavigation.map((item) => <button key={item.id} type="button" className={active === item.id ? "active" : ""} onClick={() => setActive(item.id)}><span>{item.index}</span><div><strong>{item.label}</strong><small>{item.hint}</small></div><i>›</i></button>)}</nav>
      <div className="sidebar-run-card"><span>当前会话</span><strong>{episodeId || "尚未创建"}</strong><div><StatusDot tone={runtimeMode === "backend" ? "ok" : runtimeMode === "demo" ? "warn" : "off"} />{runtimeMode === "backend" ? "后端真实运行" : runtimeMode === "demo" ? "交互演示" : "尚未开始"}</div></div>
      <div className="sidebar-system"><div><span>市场接口</span><b>{backendOnline === null ? "检查中" : backendOnline ? "在线" : "离线"}</b></div><div><span>回合状态</span><b>{completed ? "已完成" : "进行中"}</b></div></div>
    </aside>
    <section className="lab-main"><header className="lab-topbar"><div><span className="topbar-kicker">{ENTRY_META[entryMode].eyebrow} · {nav?.hint}</span><h1>{nav?.label}</h1></div><div className="topbar-meta"><span><StatusDot tone={backendOnline ? "ok" : "warn"} />{backendOnline ? "后端已连接" : "当前为演示模式"}</span><button type="button" onClick={onHome}>返回主页</button></div></header>{children}</section>
  </main>;
}

function LandingView({ choose, resume, hasSession }: { choose: (mode: EntryMode) => void; resume: () => void; hasSession: boolean }) {
  return <div className="landing-page"><header className="landing-header"><div className="landing-logo"><span>博弈</span><div><strong>多智能体博弈实验室</strong><small>多智能体博弈实验平台</small></div></div><div className="landing-status"><StatusDot /><span>本地研究环境</span></div></header><section className="landing-hero"><span>选择进入方式</span><h1>从一个清楚的入口<br />进入多智能体博弈。</h1><p>先选择你要扮演的角色，再配置市场环境。观察、参与和研究不会混在同一个界面里。</p></section><section className="entry-grid"><button type="button" className="entry-card participant" onClick={() => choose("participant")}><i>01</i><span>参与决策</span><h2>个人体验</h2><p>作为公司 A 做价格、投入与合作决策，观察其他智能体如何响应。</p><b>配置并进入 <em>→</em></b></button><button type="button" className="entry-card observer" onClick={() => choose("observer")}><i>02</i><span>观察过程</span><h2>观察实验</h2><p>不参与决策，跟随回合查看市场、消息、智能体判断和结算结果。</p><b>配置并进入 <em>→</em></b></button><button type="button" className="entry-card research" onClick={() => choose("research")}><i>03</i><span>研究分析</span><h2>研究控制台</h2><p>配置处理组、回合重建和完整研究工具；实验完成后才生成报告。</p><b>配置并进入 <em>→</em></b></button></section>{hasSession && <section className="resume-session"><div><span>当前会话</span><strong>已有一个未关闭的本地会话</strong></div><button type="button" onClick={resume}>继续当前会话 →</button></section>}<footer className="landing-footer"><span>后端市场环境是权威结果来源</span><span>不展示模型隐藏思维过程</span><span>交互演示不等于研究证据</span></footer></div>;
}

function SectionHead({ eyebrow, title, description, action }: { eyebrow: string; title: string; description?: string; action?: React.ReactNode }) {
  return <div className="section-head"><div><span>{eyebrow}</span><h2>{title}</h2>{description && <p>{description}</p>}</div>{action}</div>;
}

function SetupView({ config, setConfig, agents, setAgents, editingAgent, setEditingAgent, start, busy, notice, backendOnline, entryMode }: { config: LabConfig; setConfig: React.Dispatch<React.SetStateAction<LabConfig>>; agents: AgentConfig[]; setAgents: React.Dispatch<React.SetStateAction<AgentConfig[]>>; editingAgent: string | null; setEditingAgent: (id: string | null) => void; start: (forceDemo: boolean) => void; busy: boolean; notice: string; backendOnline: boolean | null; entryMode: EntryMode }) {
  const editor = agents.find((agent) => agent.companyId === editingAgent) ?? null;
  const patchAgent = (companyId: string, patch: Partial<AgentConfig>) => setAgents((current) => current.map((agent) => agent.companyId === companyId ? { ...agent, ...patch } : agent));
  function removeAgent(companyId: string) { setAgents((current) => current.filter((agent) => agent.companyId !== companyId)); if (editingAgent === companyId) setEditingAgent(null); }
  function addAgent() { if (agents.length < 4) setAgents((current) => [...current, { ...DEFAULT_AGENTS[agents.length] }]); }
  return <div className="view-pad setup-view">
    <section className="setup-hero"><div><span>{ENTRY_META[entryMode].eyebrow} · 环境</span><h2>配置环境</h2><p>当前入口：<b>{ENTRY_META[entryMode].label}</b>。只需确认市场与智能体；进入后会打开独立界面，随时可以返回主页。</p></div><div className="setup-readiness"><span>进入前检查</span><ul><li className="done">共同随机种子已固定</li><li className="done">{agents.length} 个智能体已分配</li><li className={config.controllerToken ? "done" : "warn"}>{config.controllerToken ? "控制器已授权" : "当前将使用演示环境"}</li></ul></div></section>
    <div className="setup-grid">
      <section className="card market-config-card"><SectionHead eyebrow="01 · 市场" title="市场配置" description="进行对照实验时，保持随机种子和市场不变，只改变需要研究的处理条件。" /><div className="segmented-label">信息可见范围</div><div className="segmented three">{(["perfect", "public", "imperfect"] as const).map((mode) => <button key={mode} type="button" className={config.informationMode === mode ? "active" : ""} onClick={() => setConfig((current) => ({ ...current, informationMode: mode }))}><b>{mode === "perfect" ? "完全信息" : mode === "public" ? "公共信息" : "不完全信息"}</b><small>{mode === "perfect" ? "查看完整市场状态" : mode === "public" ? "只能查看授权信息" : "需要形成对手判断"}</small></button>)}</div>
        <label className="field"><span>市场类型</span><select value={config.marketType} onChange={(event) => setConfig((current) => ({ ...current, marketType: event.target.value as LabConfig["marketType"] }))}>{Object.entries(MARKET_LABELS).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
        <div className="field-row"><label className="field"><span>回合数</span><select value={config.rounds} onChange={(event) => setConfig((current) => ({ ...current, rounds: Number(event.target.value) as LabConfig["rounds"] }))}>{[5, 10, 15, 20].map((rounds) => <option value={rounds} key={rounds}>{rounds} 回合</option>)}</select></label><label className="field"><span>共同随机种子</span><input type="number" value={config.seed} min={0} onChange={(event) => setConfig((current) => ({ ...current, seed: Math.max(0, Number(event.target.value) || 0) }))} /></label></div>
        <div className="capability-list"><div><span><b>智能体通信</b><small>公开消息与点对点私信</small></span><Toggle checked={config.communication} label="启用通信" onChange={(value) => setConfig((current) => ({ ...current, communication: value }))} /></div><div><span><b>共享抗冲击投入</b><small>当前唯一合作机制</small></span><Toggle checked={config.cooperation} label="启用合作" onChange={(value) => setConfig((current) => ({ ...current, cooperation: value, communication: value || current.communication }))} /></div><div><span><b>博弈分析辅助</b><small>对手判断 → 效用推断 → 策略建议</small></span><Toggle checked={config.gameTheory} label="启用博弈论增强" onChange={(value) => setConfig((current) => ({ ...current, gameTheory: value }))} /></div></div>
      </section>
      <section className="card agent-config-card"><SectionHead eyebrow="02 · 智能体" title="智能体构成" description="点击人格标签可以查看研究参数。" action={<button className="ghost-button" type="button" disabled={agents.length >= 4} onClick={addAgent}>＋ 添加智能体</button>} /><div className="agent-config-list">{agents.map((agent) => <article key={agent.companyId} style={{ "--agent": agent.color } as React.CSSProperties}><span className="agent-letter">{agent.shortName}</span><div className="agent-identity"><strong>{agent.companyName}</strong><small>{agent.companyId}</small></div><label><span>控制方式</span><select value={agent.driver} onChange={(event) => { const driver = event.target.value as AgentConfig["driver"]; patchAgent(agent.companyId, { driver, model: DRIVER_MODELS[driver] }); }}><option value="human">人类参与者</option><option value="doubao">豆包模型</option><option value="deepseek">深度求索模型</option><option value="rule">确定性规则</option></select></label><label><span>人格</span><button type="button" className="persona-chip" onClick={() => setEditingAgent(agent.companyId)}>{PERSONAS[agent.persona].label}<i>↗</i></button></label><div className="agent-flags"><span>{agent.information === "full" ? "完整信息" : agent.information === "public" ? "公共信息" : "不完全信息"}</span><span className={agent.communication ? "on" : ""}>通信</span><span className={agent.gameTheory ? "on" : ""}>博弈辅助</span></div>{agents.length > 2 && <button className="remove-agent" type="button" aria-label={`移除 ${agent.companyName}`} onClick={() => removeAgent(agent.companyId)}>×</button>}</article>)}</div></section>
    </div>
    <section className="launch-bar"><div className="controller-field"><span>本地控制器令牌</span><input type="password" placeholder="只保存在当前浏览器内存；高级后端实验必填" value={config.controllerToken} onChange={(event) => setConfig((current) => ({ ...current, controllerToken: event.target.value }))} /></div><div className="launch-status"><StatusDot tone={backendOnline ? "ok" : "warn"} /><span>{notice}</span></div><button className="secondary-launch" type="button" disabled={busy} onClick={() => start(true)}>进入交互演示</button><button className="primary-launch" type="button" disabled={busy} onClick={() => start(false)}>{busy ? "创建中…" : "创建真实实验"}<i>→</i></button></section>
    {editor && <div className="drawer-backdrop" role="presentation" onMouseDown={() => setEditingAgent(null)}><aside className="persona-drawer" role="dialog" aria-modal="true" aria-label="人格配置" onMouseDown={(event) => event.stopPropagation()}><div className="drawer-head"><div><span>人格配置</span><h2>{editor.companyName}</h2></div><button type="button" aria-label="关闭" onClick={() => setEditingAgent(null)}>×</button></div><label className="field"><span>人格预设</span><select value={editor.persona} onChange={(event) => patchAgent(editor.companyId, { persona: event.target.value as PersonaKey })}>{Object.values(PERSONAS).map((persona) => <option key={persona.key} value={persona.key}>{persona.label}</option>)}</select></label><p className="persona-summary">{PERSONAS[editor.persona].summary}</p><div className="persona-section"><span>目标权重</span>{Object.entries(PERSONAS[editor.persona].weights).map(([key, value]) => <div className="persona-slider" key={key}><label><b>{PERSONA_WEIGHT_LABELS[key]}</b><span>{value}%</span></label><input type="range" value={value} min={0} max={100} readOnly /></div>)}</div><div className="persona-section"><span>行为倾向</span>{Object.entries(PERSONAS[editor.persona].traits).map(([key, value]) => <div className="persona-slider" key={key}><label><b>{PERSONA_TRAIT_LABELS[key]}</b><span>{value}%</span></label><input type="range" value={value} min={0} max={100} readOnly /></div>)}</div><div className="drawer-note"><b>研究边界</b><p>前端展示的是版本化人格参数，不是模型内部心理状态；自定义参数写入后也必须由后端实验清单固化。</p></div></aside></div>}
  </div>;
}

function MetricCard({ label, value, note, accent }: { label: string; value: string; note: string; accent?: boolean }) { return <article className={`metric-card${accent ? " accent" : ""}`}><span>{label}</span><strong>{value}</strong><small>{note}</small></article>; }

function LiveView({ agents, round, maxRounds, nextRound, runtimeMode, notice, completed, interactive }: { agents: AgentRuntimeView[]; round: number; maxRounds: number; nextRound: (action: { price: number; advertising: number; contribution: number }) => void; runtimeMode: RuntimeMode; notice: string; completed: boolean; interactive: boolean }) {
  const [humanAction, setHumanAction] = useState({ price: 9800, advertising: 200000, contribution: 200000 });
  const [processStep, setProcessStep] = useState(0);
  const [processCompanyId, setProcessCompanyId] = useState(agents[0].companyId);
  const phase = ["接收信息", "交流", "形成判断", "制定策略", "做出决策", "市场结算"];
  const human = agents[0];
  const processAgent = agents.find((agent) => agent.companyId === processCompanyId) ?? agents[0];
  const shareTotal = agents.reduce((sum, agent) => sum + agent.share, 0);
  const averagePrice = agents.reduce((sum, agent) => sum + agent.price * agent.share, 0) / Math.max(1, shareTotal);
  const publicState = human.observation.public;
  const visibleMessages = [
    { route: "B → PUBLIC", text: round % 2 === 0 ? "我们会守住当前价格区间。" : "本轮将优先扩大市场份额。", trust: "UNVERIFIED" },
    { route: `${round % 3 === 0 ? "C" : "D"} → A`, text: round % 3 === 0 ? "建议下一轮共同提高韧性贡献。" : "供应冲击可能高于公开预测。", trust: "PRIVATE · UNVERIFIED" },
  ];
  const processMessages = processAgent.companyId === human.companyId ? visibleMessages : [
    { route: `${processAgent.companyId.slice(-1)} → PUBLIC`, text: processAgent.decision.summary, trust: "UNVERIFIED" },
    { route: `A → ${processAgent.companyId.slice(-1)}`, text: "请说明下一轮是否会改变公开价格。", trust: "PRIVATE · UNVERIFIED" },
  ];
  const hasSettledRound = agents.some((agent) => agent.shareDelta !== 0 || agent.profit !== 0);
  const hasHistoricalEvidence = round > 1 || completed;
  const moveToNextRound = () => { nextRound(humanAction); setProcessStep(0); };
  return <div className="view-pad live-view"><section className="live-command card"><div className="round-orbit"><span>当前决策回合</span><strong>{round}</strong><small>/ {maxRounds}</small><i style={{ "--progress": `${maxRounds === 1 ? 100 : Math.min(100, ((round - 1) / (maxRounds - 1)) * 100)}%` } as React.CSSProperties} /></div><div className="phase-track">{phase.map((item, index) => <div className={completed || index < processStep ? "done" : index === processStep ? "active" : ""} key={item}><i>{completed || index < processStep ? "✓" : index + 1}</i><span>{item}</span></div>)}</div><div className="live-actions"><span>{runtimeMode === "backend" ? "后端真实回合" : "前端交互演示"}<small>{notice}</small></span><b className={completed ? "round-complete" : "round-open"}>{completed ? "本次体验已完成" : `第 ${round} 回合进行中`}</b></div></section>
    <div className="live-scope-banner"><div><StatusDot tone={runtimeMode === "backend" ? "ok" : "warn"} /><span><b>{runtimeMode === "backend" ? "权威后端状态" : "交互演示状态"}</b> · 消息是未验证陈述；对手私有信息不会进入你的事实状态。</span></div><strong>市场份额合计 {shareTotal.toFixed(1)}%</strong></div>
    <div className="metrics-row"><MetricCard label="市场需求" value={publicState[0]?.value ?? "—"} note="最近一次结算产生的订单需求" /><MetricCard label="市场平均价格" value={formatMoney(averagePrice)} note="按照各公司当前市场份额加权" /><MetricCard label="行业抗冲击能力" value={publicState[3]?.value ?? "—"} note="越高表示灾害或供应冲击造成的损失越小" accent /><MetricCard label="份额校验" value={`${shareTotal.toFixed(1)}%`} note="四家公司市场份额必须合计为 100%" /></div>
    <section className="agent-board clean">{agents.map((agent, index) => <article className={`runtime-agent-card${interactive && index === 0 ? " human" : ""}`} key={agent.companyId} style={{ "--agent": agent.color } as React.CSSProperties}><div className="runtime-agent-head"><span>{agent.companyId.slice(-1)}</span><div><strong>{agent.companyName}</strong><small>{agent.persona}</small></div><b>{interactive && index === 0 ? "你" : "智能体"}</b></div><div className="agent-primary-result"><span>市场份额 <i title="该公司获得的全部市场订单比例。四家公司合计为 100%。">?</i></span><strong>{formatPercent(agent.share)}</strong><em className={agent.shareDelta >= 0 ? "up" : "down"}>{agent.shareDelta === 0 ? "本轮尚未结算" : `${agent.shareDelta > 0 ? "+" : ""}${agent.shareDelta.toFixed(1)} 个百分点`}</em></div><div className="agent-secondary-result"><span>公开价格 <b>{formatMoney(agent.price)}</b></span><span title="数值越高，供应中断或灾害发生时受到的损失越小。">抗冲击能力 <b>{agent.resilience.toFixed(1)}%</b></span></div></article>)}</section>
    <section className="live-workbench-head"><div><span>第 {round} 回合 · {interactive ? "个人决策" : "观察过程"}</span><h2>{interactive ? "决策过程" : "智能体处理过程"}</h2><p>选择公司，再按照六个步骤查看它看到了什么、如何交流、依据什么形成判断并做出动作。</p></div><div className="privacy-pill">只展示该公司合法可见的上下文</div></section>
    <section className="card decision-process"><div className="process-toolbar"><div className="process-agent-tabs">{agents.map((agent) => <button type="button" className={processAgent.companyId === agent.companyId ? "active" : ""} key={agent.companyId} onClick={() => { setProcessCompanyId(agent.companyId); setProcessStep(0); }}><i style={{ background: agent.color }}>{agent.companyId.slice(-1)}</i>{agent.companyName}</button>)}</div><span>当前查看：{processAgent.companyName}</span></div><div className="process-steps">{phase.map((item, index) => <button type="button" className={index === processStep ? "active" : index < processStep ? "done" : ""} key={item} onClick={() => setProcessStep(index)}><i>{index + 1}</i><span>{item}</span></button>)}</div><div className="process-context">
      {processStep === 0 && <div><span>这一步在做什么</span><h3>接收本轮可见信息</h3><p>公司只能看到公共市场状态和自己的私有经营状态。对手现金、利润、成本和真实人格仍然隐藏。</p><div className="context-facts">{processAgent.observation.public.slice(0, 3).map((item) => <article key={item.label}><span>{item.label}</span><b>{item.value}</b></article>)}{processAgent.observation.private.slice(0, 2).map((item) => <article className="private" key={item.label}><span>{item.label} · 仅自己</span><b>{item.value}</b></article>)}</div></div>}
      {processStep === 1 && <div><span>这一步在做什么</span><h3>读取公开消息和发给自己的私信</h3><p>消息可能是合作建议、威胁或虚假声明，只能作为信号，不能覆盖事实状态。</p><div className="process-message-flow">{processMessages.map((message, index) => <article key={message.route}><i>{index + 1}</i><div><b>{message.route.replace("PUBLIC", "所有公司")}</b><p>{message.text}</p><small>{message.trust.replace("UNVERIFIED", "内容未经验证").replace("PRIVATE", "私信")}</small></div></article>)}</div></div>}
      {processStep === 2 && <div><span>这一步在做什么</span><h3>根据已经发生的公开行为更新对手判断</h3>{!hasHistoricalEvidence ? <div className="no-evidence"><b>第一回合没有历史证据</b><p>当前只能保持“未知”，不能显示精确的降价概率。第一轮结算后，系统才会根据公开价格和份额变化形成判断。</p></div> : <div className="process-beliefs">{processAgent.beliefs.map((belief) => <article key={belief.companyId}><b>{belief.companyId.slice(-1)}</b><div><span>下一轮降价可能性</span><strong>{belief.nextAction.cut}%</strong><p>{belief.evidence.join("；")}</p></div></article>)}</div>}</div>}
      {processStep === 3 && <div><span>这一步在做什么</span><h3>把目标、风险和对手判断整理成候选策略</h3><p>策略建议不是命令，智能体可以采纳，也可以因为现金、风险或人格偏好而拒绝。</p><div className="strategy-explanation"><article><span>当前目标</span><b>{processAgent.plan.goal}</b></article><article><span>系统建议价格</span><b>{formatMoney(processAgent.advisor.recommendedPrice)}</b></article><article><span>最重要的风险</span><b>{processAgent.decision.factors[2]}</b></article></div></div>}
      {processStep === 4 && <div><span>这一步在做什么</span><h3>综合上下文后提交本轮动作</h3><p>{processAgent.decision.summary}</p><div className="final-action-context"><span>价格 <b>{formatMoney(processAgent.action.price)}</b></span><span>广告 <b>{formatMoney(processAgent.action.advertising, true)}</b></span><span>服务投入 <b>{formatMoney(processAgent.action.service, true)}</b></span><span>共享抗冲击投入 <b>{formatMoney(processAgent.action.contribution, true)}</b></span></div></div>}
      {processStep === 5 && <div><span>这一步在做什么</span><h3>四家公司动作锁定后统一结算</h3><p>市场份额是相对竞争结果：一家公司的份额增加，必然由其他公司份额减少来平衡。</p><div className="settlement-explanation"><b>{processAgent.companyName}</b><span>份额变化 {processAgent.shareDelta >= 0 ? "+" : ""}{processAgent.shareDelta.toFixed(1)} 个百分点</span><span>本轮价格 {formatMoney(processAgent.action.price)}</span></div></div>}
    </div><div className="process-next"><span>步骤 {processStep + 1} / {phase.length}</span><button type="button" disabled={processStep >= phase.length - 1} onClick={() => setProcessStep((current) => Math.min(phase.length - 1, current + 1))}>查看下一步 →</button></div></section>
    <div className="compact-evidence-row"><section className="card compact-messages"><div><span>本轮消息</span><small>始终可见 · 内容未验证</small></div>{visibleMessages.map((message) => <article key={message.route}><b>{message.route.replace("PUBLIC", "所有公司")}</b><p>{message.text}</p></article>)}</section><section className="card compact-opponent-view"><div><span>对手判断</span><small>只依据已经结算的公开历史</small></div>{!hasHistoricalEvidence ? <p className="compact-empty">第一回合尚无历史行为，暂不判断。</p> : human.beliefs.map((belief) => <article key={belief.companyId}><b>{belief.companyId.slice(-1)}</b><span>降价可能性</span><strong>{belief.nextAction.cut}%</strong></article>)}</section></div>
    <div className="live-main-action-grid">{interactive ? <section className="card live-action-panel primary-task"><div className="mini-panel-head"><span>你的本轮决策</span><b>{completed ? "已锁定" : "可以修改"}</b></div><h3>第 {round} 回合动作</h3><div className="live-action-controls"><label><span>价格 <b>{formatMoney(humanAction.price)}</b></span><small>价格更低通常有利于份额，但会压缩每笔订单利润。</small><input disabled={completed} type="range" min={8000} max={12000} step={50} value={humanAction.price} onChange={(event) => setHumanAction((current) => ({ ...current, price: Number(event.target.value) }))} /></label><label><span>广告预算 <b>{formatMoney(humanAction.advertising, true)}</b></span><small>用于吸引订单，当期需要支付成本。</small><input disabled={completed} type="range" min={0} max={2000000} step={100000} value={humanAction.advertising} onChange={(event) => setHumanAction((current) => ({ ...current, advertising: Number(event.target.value) }))} /></label><label><span>共享抗冲击投入 <b>{formatMoney(humanAction.contribution, true)}</b></span><small>当期支付成本，未来全行业都可能获得保护。</small><input disabled={completed} type="range" min={0} max={2000000} step={100000} value={humanAction.contribution} onChange={(event) => setHumanAction((current) => ({ ...current, contribution: Number(event.target.value) }))} /></label></div><button className="settle-round-button" type="button" disabled={completed} onClick={moveToNextRound}>{runtimeMode === "backend" ? "请求后端推进回合" : round === maxRounds ? "提交并结算最终回合" : `提交第 ${round} 回合并进入下一轮`}<i>→</i></button></section> : <section className="card observer-round-panel primary-task"><div className="mini-panel-head"><span>观察推进</span><b>不控制任何公司</b></div><h3>观察第 {round} 回合</h3><p>系统使用四个智能体各自的策略完成联合动作与市场结算。</p><button className="settle-round-button" type="button" disabled={completed} onClick={moveToNextRound}>{round === maxRounds ? "观察最终回合结算" : "观察下一回合"}<i>→</i></button></section>}
      <section className="card live-settlement-panel"><div className="mini-panel-head"><span>最近一次市场结算</span><b>{hasSettledRound ? "已有结果" : "等待第一轮"}</b></div><h3>{hasSettledRound ? `第 ${Math.max(1, round - (completed ? 0 : 1))} 回合结果` : "尚无已结算回合"}</h3><div className="settlement-table"><div><span>公司</span><span>价格</span><span>份额变化</span><span>利润</span></div>{agents.map((agent, index) => <article key={agent.companyId}><b style={{ color: agent.color }}>{agent.companyId.slice(-1)}</b><span>{formatMoney(agent.action.price)}</span><em className={agent.shareDelta >= 0 ? "up" : "down"}>{agent.shareDelta === 0 ? "—" : `${agent.shareDelta > 0 ? "+" : ""}${agent.shareDelta.toFixed(1)} 个百分点`}</em><strong>{index === 0 ? formatMoney(agent.profit, true) : "未公开"}</strong></article>)}</div><p className="settlement-proof">四家公司市场份额合计为 <b>{shareTotal.toFixed(1)}%</b>。对手利润属于私有信息，因此不向参与者展示。</p></section></div>
  </div>;
}

type ObservatoryTab = "observation" | "belief" | "planning" | "decision";

function ObservatoryView({ agents }: { agents: AgentRuntimeView[] }) {
  const [companyId, setCompanyId] = useState(agents[0].companyId);
  const [tab, setTab] = useState<ObservatoryTab>("observation");
  const agent = agents.find((item) => item.companyId === companyId) ?? agents[0];
  return <div className="view-pad observatory-view"><section className="observatory-header card"><div><span>智能体范围审计</span><h2>这个智能体在这一轮究竟知道什么？</h2><p>可见信息、自己的私有信息、隐藏信息和概率判断被严格分开。这里不展示模型隐藏思维过程。</p></div><div className="agent-selector">{agents.map((item) => <button type="button" className={item.companyId === companyId ? "active" : ""} style={{ "--agent": item.color } as React.CSSProperties} key={item.companyId} onClick={() => setCompanyId(item.companyId)}><i>{item.companyId.slice(-1)}</i><span>{item.companyName}<small>{item.persona}</small></span></button>)}</div></section>
    <div className="observatory-layout"><aside className="agent-dossier card"><div className="dossier-avatar" style={{ background: agent.color }}>{agent.companyId.slice(-1)}</div><span>当前智能体</span><h2>{agent.companyName}</h2><p>{agent.driver}<br />人格 · {agent.persona}</p><dl><div><dt>输入信息校验值</dt><dd>{agent.observationHash}</dd></div><div><dt>回合</dt><dd>05</dd></div><div><dt>状态版本</dt><dd>4</dd></div><div><dt>输入状态</dt><dd className="good">已冻结</dd></div></dl><div className="privacy-rule"><b>信息边界</b><p>对手现金、利润、成本、事故细节和真实人格不可见。</p></div></aside>
      <section className="cognition-panel card"><div className="tabbar">{(["observation", "belief", "planning", "decision"] as const).map((item) => <button type="button" className={tab === item ? "active" : ""} key={item} onClick={() => setTab(item)}>{item === "observation" ? "可见信息" : item === "belief" ? "对手判断" : item === "planning" ? "行动计划" : "决策摘要"}</button>)}</div>
        {tab === "observation" && <div className="observation-grid"><div className="visibility-column visible"><h3><i>✓</i> 公共状态 <span>所有公司可见</span></h3>{agent.observation.public.map((item) => <article key={item.label}><span>{item.label}</span><strong>{item.value}</strong></article>)}</div><div className="visibility-column private"><h3><i>●</i> 自己的私有状态 <span>仅自己可见</span></h3>{agent.observation.private.map((item) => <article key={item.label}><span>{item.label}</span><strong>{item.value}</strong></article>)}</div><div className="visibility-column hidden"><h3><i>⌁</i> 对手状态 <span>已隐藏</span></h3>{agent.observation.hidden.map((item) => <article key={item.label}><span>{item.label}</span><strong>{item.value}</strong></article>)}</div><div className="untrusted-note"><b>未验证消息边界</b><p>通信中的“现金充足”“准备扩产”等声明只作为未验证信号进入对手判断，绝不会写回事实状态。</p></div></div>}
        {tab === "belief" && <div className="belief-grid">{agent.beliefs.map((belief) => <article key={belief.companyId}><div className="belief-head"><span>{belief.companyId.slice(-1)}</span><div><strong>{belief.companyId}</strong><small>基于公开证据的概率判断</small></div></div><h4>策略倾向</h4>{Object.entries(belief.strategy).map(([key, value]) => <div className="belief-bar" key={key}><span>{STRATEGY_LABELS[key]}</span><i><b style={{ width: `${value}%` }} /></i><em>{value}%</em></div>)}<h4>下一轮价格动作</h4>{Object.entries(belief.nextAction).map(([key, value]) => <div className="belief-bar compact" key={key}><span>{ACTION_LABELS[key]}</span><i><b style={{ width: `${value}%` }} /></i><em>{value}%</em></div>)}<details><summary>判断依据 · {belief.evidence.length} 条</summary><ul>{belief.evidence.map((item) => <li key={item}>{item}</li>)}</ul></details></article>)}</div>}
        {tab === "planning" && <div className="planning-view"><div className="plan-hero"><span>当前目标</span><h3>{agent.plan.goal}</h3><p>计划考虑未来 {agent.plan.horizon} 个回合</p></div><div className="plan-steps">{agent.plan.subgoals.map((goal, index) => <article className={goal.status} key={goal.label}><i>{goal.status === "done" ? "✓" : index + 1}</i><span>{goal.label}<small>{PLAN_STATUS_LABELS[goal.status]}</small></span></article>)}</div><div className="replan-triggers"><span>重新规划条件</span>{agent.plan.triggers.map((trigger) => <p key={trigger}>↳ {trigger}</p>)}</div></div>}
        {tab === "decision" && <div className="decision-summary-view"><div className="no-cot"><span>可审计决策摘要</span><b>不展示隐藏思维过程</b></div><article><span>当前局面</span><p>{agent.decision.situation}</p></article><article><span>主要依据</span><ol>{agent.decision.factors.map((factor) => <li key={factor}>{factor}</li>)}</ol></article><article><span>最终决定</span><p className="decision-highlight">{agent.decision.summary}</p></article><article><span>预期结果</span><p>{agent.decision.expected}</p></article></div>}
      </section></div>
  </div>;
}

function CommunicationView() {
  const [filter, setFilter] = useState<"all" | "public" | "private">("all");
  const messages = DEMO_MESSAGES.filter((message) => filter === "all" || message.channel === filter);
  const filterLabels = { all: "全部", public: "公开消息", private: "私信" };
  const kindLabels: Record<string, string> = { statement: "声明", proposal: "提议", commitment: "承诺", threat: "威胁", signal: "信号", response: "回应" };
  return <div className="view-pad communication-view"><SectionHead eyebrow="交流、提议与承诺" title="通信记录" description="公开消息、私信、提议与履约结果位于同一条时间线。所有消息都不具有强制约束力，也不一定真实。" action={<div className="filter-pills">{(["all", "public", "private"] as const).map((item) => <button type="button" className={filter === item ? "active" : ""} onClick={() => setFilter(item)} key={item}>{filterLabels[item]}</button>)}</div>} /><div className="communication-layout"><section className="message-stream card"><div className="stream-head"><span>第 5 回合 · 通信已关闭</span><b>{messages.length} 条消息</b></div>{messages.map((message) => <article className={message.channel} key={message.id}><div className="message-route"><i>{message.sender.slice(-1)}</i><span>{message.sender}{message.recipient ? ` → ${message.recipient}` : " → 所有公司"}<small>{message.channel === "public" ? "公开" : "私信"} · {kindLabels[message.kind]}</small></span><time>第 {message.round} 回合</time></div><blockquote>{message.text}</blockquote><div className="message-meta"><span>可见公司：{message.visibility.join("、")}</span>{message.status && <b className={message.status}>{message.status === "accepted" ? "已接受" : message.status === "rejected" ? "已拒绝" : "部分违约"}</b>}</div></article>)}</section><aside className="communication-audit card"><span>消息可见范围</span><h3>谁能看见什么</h3><div className="matrix"><div><i /><b>A</b><b>B</b><b>C</b><b>D</b></div>{DEMO_MESSAGES.slice(0, 4).map((message) => <div key={message.id}><span>{message.sender.slice(-1)}{message.recipient ? `→${message.recipient.slice(-1)}` : "→全部"}</span>{["A", "B", "C", "D"].map((company) => <i className={message.visibility.includes(company) ? "seen" : "hidden"} key={company}>{message.visibility.includes(company) ? "✓" : "–"}</i>)}</div>)}</div><div className="audit-callout"><b>未发现可见性泄漏</b><p>非接收者看不到私信的编号、数量、时间或正文。</p></div><div className="commitment-chain"><span>承诺履约过程</span><ol><li><i>01</i>提出投入 10,000 元</li><li><i>02</i>C 接受提议</li><li><i>03</i>实际投入 3,000 元</li><li className="bad"><i>04</i>履约 30%，属于部分违约</li></ol></div></aside></div></div>;
}

function StrategyView({ agents }: { agents: AgentRuntimeView[] }) {
  const [companyId, setCompanyId] = useState(agents[1].companyId);
  const agent = agents.find((item) => item.companyId === companyId) ?? agents[1];
  const pipeline = ["可见信息", "动作概率判断", "对手策略判断", "效用目标推断", "策略建议", "最终动作"];
  return <div className="view-pad strategy-view"><SectionHead eyebrow="对手判断 → 效用推断 → 策略建议" title="博弈策略分析" description="每一层都记录输入来源；策略建议不具有强制约束力，最终决定仍由智能体提交。" action={<select value={companyId} onChange={(event) => setCompanyId(event.target.value)}>{agents.map((item) => <option value={item.companyId} key={item.companyId}>{item.companyName}</option>)}</select>} /><section className="strategy-pipeline">{pipeline.map((item, index) => <div key={item} className={index === 4 ? "focus" : ""}><i>{index + 1}</i><span>{item}<small>{index < 4 ? "已验证输入" : index === 4 ? "仅供参考" : "智能体提交"}</small></span>{index < 5 && <b>→</b>}</div>)}</section><div className="strategy-grid"><section className="card opponent-model-card"><span>对手策略判断</span><h3>对手策略倾向</h3>{agent.beliefs.map((belief) => <article key={belief.companyId}><div><b>{belief.companyId}</b><small>{belief.evidence.length} 条公开依据</small></div><div className="strategy-stack" title="增长导向、利润导向、防御导向"><i style={{ width: `${belief.strategy.growth}%` }} /><i style={{ width: `${belief.strategy.profit}%` }} /><i style={{ width: `${belief.strategy.defensive}%` }} /></div><em>增长导向 {belief.strategy.growth}%</em></article>)}<div className="stack-legend"><span><i />增长</span><span><i />利润</span><span><i />防御</span></div></section><section className="card utility-card"><span>效用目标推断</span><h3>推断偏好，不等于真实人格</h3><div className="utility-radar"><svg viewBox="0 0 220 180" role="img" aria-label="效用权重雷达图"><polygon points="110,12 204,72 168,168 52,168 16,72" className="radar-grid" /><polygon points={`110,${100 - agent.utility.profit} ${110 + agent.utility.growth * 1.5},78 ${145 + agent.utility.risk / 2},142 66,142 42,78`} className="radar-data" /><text x="110" y="9">利润</text><text x="184" y="66">增长</text><text x="174" y="176">风险</text><text x="16" y="176">现金</text><text x="0" y="66">共同收益</text></svg></div><div className="utility-values"><span>利润 <b>{agent.utility.profit}%</b></span><span>增长 <b>{agent.utility.growth}%</b></span><span>风险 <b>{agent.utility.risk}%</b></span></div></section><section className="card advisor-card"><span>博弈策略建议</span><h3>有限动作候选</h3><div className="candidate-table"><div><span>动作</span><span>效用评分</span><span>风险</span></div>{agent.advisor.candidates.map((candidate, index) => <article className={index === 0 ? "recommended" : ""} key={candidate.action}><span>{candidate.action}{index === 0 && <b>系统建议</b>}</span><strong>{candidate.utility.toFixed(2)}</strong><em>{candidate.risk.toFixed(2)}</em></article>)}</div><div className="advisor-result"><span>建议价格<strong>{formatMoney(agent.advisor.recommendedPrice)}</strong></span><i>→</i><span>最终价格<strong>{formatMoney(agent.action.price)}</strong></span><b className={agent.advisor.adopted ? "adopted" : "rejected"}>{agent.advisor.adopted ? "已采纳" : "未采纳"}</b></div><p>是否有效必须使用真实市场结果评价；内部评分不能证明建议提高了利润。</p></section></div></div>;
}

function LineChart() {
  const colors: Record<string, string> = { A: "#21b99a", B: "#ef765d", C: "#6286eb", D: "#a777e3" };
  const keys = ["A", "B", "C", "D"] as const;
  const path = (key: typeof keys[number]) => PROFIT_SERIES.map((point, index) => `${index ? "L" : "M"}${30 + index * 112},${190 - point[key] * 52}`).join(" ");
  return <svg className="line-chart" viewBox="0 0 510 220" role="img" aria-label="公司利润趋势"><g className="chart-grid">{[30, 80, 130, 180].map((y) => <line key={y} x1="28" y1={y} x2="485" y2={y} />)}</g>{keys.map((key) => <g key={key}><path d={path(key)} fill="none" stroke={colors[key]} strokeWidth="3" />{PROFIT_SERIES.map((point, index) => <circle key={point.round} cx={30 + index * 112} cy={190 - point[key] * 52} r="4" fill={colors[key]} />)}</g>)}</svg>;
}

function MarketView({ agents }: { agents: AgentRuntimeView[] }) {
  const gradients = agents.map((agent, index) => `${agent.color} ${agents.slice(0, index).reduce((sum, item) => sum + item.share, 0)}% ${agents.slice(0, index + 1).reduce((sum, item) => sum + item.share, 0)}%`).join(",");
  return <div className="view-pad market-view"><SectionHead eyebrow="市场结果与公共状态" title="市场面板" description="这里只展示后端市场状态和结算派生指标，不在前端复制权威市场公式。" /><div className="metrics-row"><MetricCard label="市场需求" value="12,480" note="实际产生的订单数" /><MetricCard label="市场平均价格" value="¥98.20" note="成交订单加权价格" /><MetricCard label="行业抗冲击能力" value="62.4%" note="越高表示冲击损失越小" accent /><MetricCard label="消费者信任" value="71.0%" note="消费者对市场的总体信心" /></div><div className="market-grid"><section className="card chart-panel"><div className="card-head"><div><span>利润变化</span><h3>单轮利润轨迹</h3></div><div className="chart-legend">{agents.map((agent) => <span key={agent.companyId}><i style={{ background: agent.color }} />{agent.companyId.slice(-1)}</span>)}</div></div><LineChart /><div className="chart-axis">{PROFIT_SERIES.map((point) => <span key={point.round}>第 {point.round} 轮</span>)}</div></section><section className="card share-panel"><div className="card-head"><div><span>市场份额</span><h3>竞争格局</h3></div></div><div className="share-ring" style={{ background: `conic-gradient(${gradients})` }}><i><strong>100%</strong><span>全部市场</span></i></div><div className="share-list">{agents.map((agent) => <div key={agent.companyId}><i style={{ background: agent.color }} /><span>{agent.companyName}</span><strong>{formatPercent(agent.share)}</strong></div>)}</div></section><section className="card investment-panel"><div className="card-head"><div><span>投入结构</span><h3>资源配置对照</h3></div></div><div className="investment-table"><div><span>公司</span><span>广告</span><span>服务</span><span>产能</span><span>抗冲击投入</span></div>{agents.map((agent) => <article key={agent.companyId}><b style={{ color: agent.color }}>{agent.companyId.slice(-1)}</b>{[agent.action.advertising, agent.action.service, agent.action.capacity, agent.action.resilience].map((value, index) => <span key={index}><i style={{ width: `${Math.min(100, value / 8000)}%` }} /><em>{formatMoney(value, true)}</em></span>)}</article>)}</div></section></div></div>;
}

function ReplayView() {
  const [selected, setSelected] = useState(4);
  const step = DEMO_REPLAY[selected];
  return <div className="view-pad replay-view"><SectionHead eyebrow="确定性输入重建" title="回合重建" description="按照市场状态 → 可见信息 → 对手判断 → 通信 → 决策 → 动作 → 结果重建过程；不会重新调用模型生成语义。" action={<div className="replay-selectors"><select aria-label="实验"><option>第 1001 号随机种子实验</option></select><select aria-label="回合"><option>第 5 回合</option></select><select aria-label="智能体"><option>全部智能体</option></select></div>} /><div className="replay-layout"><section className="replay-timeline card">{DEMO_REPLAY.map((item, index) => <button type="button" key={`${item.phase}-${index}`} className={`${item.tone}${selected === index ? " selected" : ""}`} onClick={() => setSelected(index)}><i>{index + 1}</i><div><span>{item.phase} · {item.agent}</span><strong>{item.title}</strong><small>{item.detail}</small></div><em>{item.hash ?? "—"}</em></button>)}</section><section className="replay-inspector card"><div className="inspector-head"><span>过程详情</span><b>{step.phase}</b></div><div className="trace-summary"><span>{step.agent}</span><h3>{step.title}</h3><p>{step.detail}</p></div><div className="json-view"><span>{`{`}</span><p><i>"round"</i>: <b>5</b>,</p><p><i>"phase"</i>: <em>"{step.phase.toLowerCase().replace(" ", "_")}"</em>,</p><p><i>"company_scope"</i>: <em>"{step.agent}"</em>,</p><p><i>"source_hash"</i>: <em>"{step.hash ?? "derived"}"</em>,</p><p><i>"replay_match"</i>: <b>true</b></p><span>{`}`}</span></div><div className="replay-checks"><div><StatusDot />经济状态重建 <b>100%</b></div><div><StatusDot />交互过程重建 <b>100%</b></div><div><StatusDot />可见信息重建 <b>100%</b></div><div><StatusDot />博弈分析重建 <b>100%</b></div></div></section></div></div>;
}

function ReportView({ agents, runtimeMode }: { agents: AgentRuntimeView[]; runtimeMode: RuntimeMode }) {
  const profitWinner = [...agents].sort((a, b) => b.profit - a.profit)[0];
  const shareWinner = [...agents].sort((a, b) => b.share - a.share)[0];
  const averageResilience = agents.reduce((sum, agent) => sum + agent.resilience, 0) / agents.length;
  const totalShare = agents.reduce((sum, agent) => sum + agent.share, 0);
  return <div className="view-pad report-view"><section className="report-hero"><div><span>实验已完成 · 最终总结</span><h2>本次实验已经完成。</h2><p>{runtimeMode === "demo" ? "这是交互演示的回合总结，不是大模型实验结论。" : "报告来自当前后端实验；研究结论仍需使用共同随机种子进行对照。"}</p></div><button type="button">导出当前结果 <i>↗</i></button></section><div className="report-scorecard"><article><span>利润最高</span><strong style={{ color: profitWinner.color }}>{profitWinner.companyName}</strong><small>{formatMoney(profitWinner.profit)}</small></article><article><span>市场份额最高</span><strong style={{ color: shareWinner.color }}>{shareWinner.companyName}</strong><small>{formatPercent(shareWinner.share)}</small></article><article><span>行业平均抗冲击能力</span><strong>{averageResilience.toFixed(1)}%</strong><small>四家公司平均值</small></article><article><span>市场份额合计</span><strong>{totalShare.toFixed(1)}%</strong><small>必须等于 100%</small></article></div><div className="report-grid clean-report"><section className="card report-findings"><span>本次运行说明了什么</span><h3>当前实验结果</h3><article><i>01</i><div><strong>市场竞争已经完成</strong><p>价格和投入共同影响份额与利润，四家公司份额总和保持 100%。</p></div><b>已观察到</b></article><article><i>02</i><div><strong>份额不等于利润</strong><p>份额领先者与利润领先者可能不是同一家公司，需要同时解释竞争和经营结果。</p></div><b>描述性结果</b></article><article className="warn"><i>03</i><div><strong>{runtimeMode === "demo" ? "不能作为研究证据" : "单个实验不能形成因果结论"}</strong><p>{runtimeMode === "demo" ? "正式结论必须来自后端市场环境、真实智能体、回合事件和过程重建。" : "至少需要共同随机种子、固定处理条件和配对统计。"}</p></div><b>证据边界</b></article></section><section className="card agent-report"><span>智能体最终结果</span><h3>公司结果</h3>{agents.map((agent) => <article key={agent.companyId}><i style={{ background: agent.color }}>{agent.companyId.slice(-1)}</i><div><strong>{agent.persona}</strong><p>{agent.decision.summary}</p></div><span><b>{formatPercent(agent.share)}</b><small>市场份额</small></span><span><b>{formatMoney(agent.profit, true)}</b><small>利润</small></span></article>)}</section></div></div>;
}

function hydrateAgents(payload: BackendEpisode, configs: AgentConfig[]) {
  return DEMO_AGENTS.map((agent) => {
    const state = payload.state.companies[agent.companyId];
    const config = configs.find((item) => item.companyId === agent.companyId);
    if (!state || !config) return agent;
    return { ...agent, persona: PERSONAS[config.persona].label, driver: config.model, cash: state.financial.cash_balance_cents, profit: state.financial.round_profit_cents, share: state.commercial.market_share_ppm / 10_000, price: state.commercial.price_cents, resilience: state.risk.resilience_ppm / 10_000 };
  });
}

function hydrateDemoAgents(configs: AgentConfig[]) {
  return DEMO_AGENTS.map((agent) => {
    const config = configs.find((item) => item.companyId === agent.companyId);
    if (!config) return agent;
    return {
      ...agent,
      driver: config.model,
      persona: PERSONAS[config.persona].label,
      action: { ...agent.action },
      observation: { public: [...agent.observation.public], private: [...agent.observation.private], hidden: [...agent.observation.hidden] },
    };
  });
}

export default function Home() {
  const [active, setActive] = useState<ViewId>("home");
  const [entryMode, setEntryMode] = useState<EntryMode>("participant");
  const [config, setConfig] = useState<LabConfig>({ informationMode: "public", marketType: "balanced", rounds: 20, seed: 20260821, communication: true, cooperation: false, gameTheory: true, controllerToken: "" });
  const [agents, setAgents] = useState<AgentConfig[]>(DEFAULT_AGENTS.map((agent) => ({ ...agent })));
  const [runtimeAgents, setRuntimeAgents] = useState<AgentRuntimeView[]>(DEMO_AGENTS);
  const [editingAgent, setEditingAgent] = useState<string | null>(null);
  const [runtimeMode, setRuntimeMode] = useState<RuntimeMode>("draft");
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("正在检查 Market API；研究演示始终可用。");
  const [episodeId, setEpisodeId] = useState("");
  const [round, setRound] = useState(1);
  const [demoCompleted, setDemoCompleted] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_URL}/health`, { signal: controller.signal }).then((response) => { setBackendOnline(response.ok); setNotice((current) => current.startsWith("研究演示") ? current : response.ok ? "Market API 在线，可创建真实 Episode。" : "Market API 不可用，可载入演示。"); }).catch(() => { setBackendOnline(false); setNotice((current) => current.startsWith("研究演示") ? current : "Market API 离线；可载入有明确标识的研究演示。"); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [active]);

  const marketModel = useMemo(() => ({ balanced: "balanced", high_demand: "quality_oriented", supply_crisis: "value_oriented", disaster: "service_oriented", public_goods: "balanced" } as const)[config.marketType], [config.marketType]);

  async function startExperiment(forceDemo: boolean) {
    if (forceDemo || !backendOnline) { setRuntimeMode("demo"); setEpisodeId(`demo-${config.seed}`); setRound(1); setDemoCompleted(false); setRuntimeAgents(hydrateDemoAgents(agents)); setNotice("研究演示从第 1 回合开始；所有示例字段均标记为 DEMO，不代表一次真实模型调用。"); setActive("live"); return; }
    const protectedMode = config.communication || config.cooperation || config.gameTheory || config.informationMode !== "perfect";
    if (protectedMode && !config.controllerToken) { setNotice("高级实验需要本地 Controller Token。可填写 Token，或先载入研究演示。"); return; }
    setBusy(true);
    try {
      const response = await fetch(`${API_URL}/episodes`, { method: "POST", headers: { "Content-Type": "application/json", ...(config.controllerToken ? { "X-Controller-Token": config.controllerToken } : {}) }, body: JSON.stringify({ episode_seed: config.seed, company_ids: agents.map((agent) => agent.companyId), personas: Object.fromEntries(agents.map((agent) => [agent.companyId, agent.persona.startsWith("aggressive") ? "aggressive" : agent.persona.startsWith("risk") ? "conservative" : "balanced"])), agent_configs: Object.fromEntries(agents.map((agent) => [agent.companyId, { agent_id: `${agent.driver}-${agent.companyId}`, agent_type: agent.driver === "rule" ? "rule" : agent.driver === "human" ? "human" : "model", model: agent.model, persona_name: agent.persona }])), game_mode: agents.some((agent) => agent.driver === "human") && !protectedMode ? "single_company" : "market", player_company_id: agents.find((agent) => agent.driver === "human")?.companyId ?? null, market_model: marketModel, max_rounds: config.rounds, information_mode: config.informationMode === "perfect" ? "perfect" : "public", communication_mode: config.communication ? "public_private" : "off", cooperation_mode: config.cooperation ? "shared_resilience_v1" : "off", belief_mode: config.gameTheory ? "public_action_v1" : "off", opponent_model_mode: config.gameTheory ? "public_strategy_v1" : "off", utility_inference_mode: config.gameTheory ? "strategy_utility_v1" : "off", advisor_mode: config.gameTheory ? "bayesian_strategy_v2" : "off", repeated_game_mode: "off" }) });
      const payload = await response.json() as BackendEpisode & { detail?: string };
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `HTTP ${response.status}`);
      setRuntimeMode("backend"); setEpisodeId(payload.state.episode_id); setRound(Math.max(1, payload.state.round)); setDemoCompleted(false); setRuntimeAgents(hydrateAgents(payload, agents)); setNotice("真实 Episode 已创建；高级回合推进等待 Coordinator intents/communication barrier。仅创建不等于模型已运行。"); setActive("live");
    } catch (error) { setNotice(error instanceof Error ? `创建失败：${error.message}` : "创建失败；状态未改变。"); } finally { setBusy(false); }
  }

  function nextRound(humanAction: { price: number; advertising: number; contribution: number }) {
    if (runtimeMode === "backend") { setNotice("高级真实 Episode 必须由 Coordinator 完成通信、Intent 与 Settlement；前端不会绕过屏障直接 Step。"); return; }
    if (demoCompleted) return;
    const settledRound = round;
    setRuntimeAgents((current) => advanceDemoRound(current, settledRound, humanAction));
    if (settledRound >= config.rounds) {
      setDemoCompleted(true);
      setNotice(`第 ${settledRound} 回合已结算，Episode 完成；不会循环回第 1 回合。`);
      return;
    }
    setRound(settledRound + 1);
    setNotice(`第 ${settledRound} 回合已联合结算；现在进入第 ${settledRound + 1} 回合决策。演示结果不作为研究证据。`);
  }

  function chooseEntry(mode: EntryMode) {
    setEntryMode(mode);
    setRuntimeMode("draft");
    setEpisodeId("");
    setRound(1);
    setDemoCompleted(false);
    setRuntimeAgents(DEMO_AGENTS);
    setAgents(DEFAULT_AGENTS.map((agent, index) => mode === "observer" ? { ...agent, driver: index === 3 ? "rule" : index === 1 ? "doubao" : "deepseek", model: index === 3 ? DRIVER_MODELS.rule : index === 1 ? DRIVER_MODELS.doubao : DRIVER_MODELS.deepseek } : { ...agent }));
    setNotice(`${ENTRY_META[mode].label}：请先确认环境，然后进入独立界面。`);
    setActive("setup");
  }

  const view = active === "home" ? <LandingView choose={chooseEntry} resume={() => setActive("live")} hasSession={Boolean(episodeId)} /> : active === "setup" ? <SetupView config={config} setConfig={setConfig} agents={agents} setAgents={setAgents} editingAgent={editingAgent} setEditingAgent={setEditingAgent} start={(forceDemo) => void startExperiment(forceDemo)} busy={busy} notice={notice} backendOnline={backendOnline} entryMode={entryMode} /> : active === "live" ? <LiveView agents={runtimeAgents} round={round} maxRounds={config.rounds} nextRound={nextRound} runtimeMode={runtimeMode} notice={notice} completed={demoCompleted} interactive={entryMode === "participant"} /> : active === "observatory" ? <ObservatoryView agents={runtimeAgents} /> : active === "communication" ? <CommunicationView /> : active === "strategy" ? <StrategyView agents={runtimeAgents} /> : active === "market" ? <MarketView agents={runtimeAgents} /> : active === "replay" ? <ReplayView /> : <ReportView agents={runtimeAgents} runtimeMode={runtimeMode} />;
  return <AppShell active={active} setActive={setActive} runtimeMode={runtimeMode} backendOnline={backendOnline} episodeId={episodeId} entryMode={entryMode} completed={demoCompleted} onHome={() => setActive("home")}>{view}</AppShell>;
}
