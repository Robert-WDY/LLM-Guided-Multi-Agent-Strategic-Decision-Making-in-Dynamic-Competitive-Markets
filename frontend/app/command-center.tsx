"use client";

import { useEffect, useMemo, useReducer, useRef, useState, type CSSProperties, type ReactNode } from "react";
import type { AgentRuntimeView } from "./lab-model";
import {
  agentName,
  createScreen,
  hhi,
  moveLabel,
  PHASES,
  PHENOMENA,
  phaseId,
  SCREEN_MAX_AGENTS,
  SCREEN_MIN_AGENTS,
  SCREEN_ROUNDS,
  screenReducer,
  strategyLabel,
  termsText,
  yuan,
  type FeedItem,
  type PhaseId,
  type ScreenMessage,
  type ScreenState,
} from "./command-center-model";
import { PixelAvatar } from "./pixel-avatar";
import "./command-center.css";

/* ---------- constants ---------- */

const SPEEDS = [
  { label: "慢", ms: 5200 },
  { label: "中", ms: 3400 },
  { label: "快", ms: 1900 },
];

/** Soft pastel palette that reads well on deep indigo; replaces the lab's saturated seat colours here. */
const SCREEN_COLORS = ["#7ce3d0", "#ffab91", "#8fb8ff", "#c7a6ff", "#ff8fab", "#ffd36e", "#8fe3a2", "#7fd6f5", "#f5a3d6", "#b8c4ff"];

const DRIVER_TAG: Record<string, string> = { human: "代席", doubao: "LLM", deepseek: "LLM", rule: "规则" };

const KIND_COLOR: Record<ScreenMessage["kind"], string> = {
  proposal: "#7fd6f5",
  response: "#8fe3a2",
  threat: "#ff8fab",
  signal: "#ffd36e",
  statement: "#c7a6ff",
};
const KIND_LABEL: Record<ScreenMessage["kind"], string> = {
  proposal: "提议",
  response: "回应",
  threat: "威胁",
  signal: "信号",
  statement: "声明",
};
const PHENOMENON_COLOR: Record<string, string> = {
  信息不足: "#6c7399",
  价格战: "#ff8fab",
  搭便车: "#ffd36e",
  份额流失: "#ffab91",
  价格敏感: "#c7a6ff",
  协调均衡: "#8fe3a2",
};

type TabId = "strategy" | "talk" | "ops" | "market";
const TABS: Array<{ id: TabId; label: string; en: string }> = [
  { id: "strategy", label: "战略建议与影响", en: "STRATEGY" },
  { id: "talk", label: "交流与协议", en: "TALK" },
  { id: "ops", label: "操作", en: "OPS" },
  { id: "market", label: "市场与库存", en: "MARKET" },
];
const TAB_FOR_PHASE: Record<PhaseId, TabId> = {
  observe: "market",
  communicate: "talk",
  belief: "strategy",
  advise: "strategy",
  act: "ops",
  settle: "strategy",
};

const HUB = { x: 500, y: 318 };

/* ---------- helpers ---------- */

function layout(count: number) {
  return Array.from({ length: count }, (_, i) => {
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / count;
    const cos = Math.cos(angle);
    return { x: HUB.x + 400 * cos, y: HUB.y + 200 * Math.sin(angle), side: cos < -0.2 ? ("right" as const) : ("left" as const) };
  });
}

function curve(a: { x: number; y: number }, b: { x: number; y: number }) {
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  const cx = mx + (HUB.x - mx) * 0.35;
  const cy = my + (HUB.y - my) * 0.35;
  return `M${a.x.toFixed(1)},${a.y.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${b.x.toFixed(1)},${b.y.toFixed(1)}`;
}

const money = (cents: number) => {
  const value = cents / 100;
  if (Math.abs(value) >= 10_000) return `¥${(value / 10_000).toFixed(1)}万`;
  return `¥${Math.round(value).toLocaleString("zh-CN")}`;
};
const num = (v: number) => v.toLocaleString("zh-CN");
const signedMoney = (cents: number) => `${cents > 0 ? "+" : cents < 0 ? "−" : "±"}${money(Math.abs(cents))}`;
const signedPp = (v: number) => `${v > 0 ? "+" : v < 0 ? "−" : "±"}${Math.abs(v).toFixed(1)}pp`;
const pad2 = (v: number) => String(v).padStart(2, "0");

function useColors(state: ScreenState) {
  return useMemo(
    () => Object.fromEntries(state.agents.map((a, i) => [a.companyId, SCREEN_COLORS[i % SCREEN_COLORS.length]])),
    [state.agents],
  );
}

function Who({ state, id, colors }: { state: ScreenState; id: string | null; colors: Record<string, string> }) {
  if (!id) return <span className="cc-who all">全体</span>;
  const index = state.agents.findIndex((a) => a.companyId === id);
  return (
    <span className="cc-who" style={{ color: colors[id] }}>
      <PixelAvatar index={index} color={colors[id]} size={18} blink={false} />
      {agentName(state.agents, id)}
    </span>
  );
}

/* ---------- header ---------- */

function Clock() {
  const [now, setNow] = useState("--:--:--");
  useEffect(() => {
    const update = () => setNow(new Date().toLocaleTimeString("zh-CN", { hour12: false }));
    const first = window.setTimeout(update, 0);
    const timer = window.setInterval(update, 1000);
    return () => {
      window.clearTimeout(first);
      window.clearInterval(timer);
    };
  }, []);
  return <time className="cc-clock">{now}</time>;
}

function Header(props: {
  state: ScreenState;
  paused: boolean;
  setPaused: (v: boolean) => void;
  speed: number;
  setSpeed: (v: number) => void;
  onStep: () => void;
  onCount: (n: number) => void;
  onFullscreen: () => void;
  fullscreen: boolean;
  onExit?: () => void;
}) {
  const { state } = props;
  const progress = ((state.round - 1) * PHASES.length + state.phase + 1) / (SCREEN_ROUNDS * PHASES.length);
  return (
    <header className="cc-header">
      <div className="cc-brand">
        <PixelAvatar index={0} color={SCREEN_COLORS[0]} size={44} />
        <div>
          <h1>GAME OPS<span>博弈指挥大屏</span></h1>
          <p>交互演示数据 · 非研究证据 · {state.agents.length} 个智能体 · 1 个共享市场</p>
        </div>
      </div>
      <div className="cc-progress" aria-label={`第 ${state.episode} 局进度`}>
        <span>EP {pad2(state.episode)}</span>
        <div className="cc-progress-bar"><i style={{ width: `${(progress * 100).toFixed(1)}%` }} /></div>
        <b role="status">ROUND {pad2(state.round)} / {SCREEN_ROUNDS}</b>
        <em>{PHASES[state.phase].label}</em>
      </div>
      <div className="cc-controls">
        <label className="cc-count">
          <span>智能体</span>
          <select aria-label="智能体数量" value={state.agents.length} onChange={(e) => props.onCount(Number(e.target.value))}>
            {Array.from({ length: SCREEN_MAX_AGENTS - SCREEN_MIN_AGENTS + 1 }, (_, i) => i + SCREEN_MIN_AGENTS).map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
        <div className="cc-seg" role="group" aria-label="播放速度">
          {SPEEDS.map((item, index) => (
            <button key={item.label} type="button" aria-pressed={props.speed === index} onClick={() => props.setSpeed(index)}>{item.label}</button>
          ))}
        </div>
        <button type="button" onClick={() => props.setPaused(!props.paused)}>{props.paused ? "▶ 继续" : "❚❚ 暂停"}</button>
        <button type="button" onClick={props.onStep} disabled={!props.paused} title="暂停时逐阶段推进">单步</button>
        <button type="button" onClick={props.onFullscreen}>{props.fullscreen ? "退出全屏" : "全屏"}</button>
        {props.onExit && <button type="button" className="cc-exit" onClick={props.onExit}>返回实验室</button>}
        <Clock />
      </div>
    </header>
  );
}

/* ---------- agent strip ---------- */

function AgentStrip({ state, colors, focus, setFocus }: { state: ScreenState; colors: Record<string, string>; focus: string; setFocus: (id: string) => void }) {
  const phase = phaseId(state);
  const speaking = new Set(state.plan.messages.map((m) => m.from));
  return (
    <section className="cc-cards" aria-label="智能体状态">
      {state.agents.map((agent, index) => {
        const color = colors[agent.companyId];
        const status =
          phase === "communicate" && speaking.has(agent.companyId) ? "speaking"
            : phase === "belief" ? "thinking"
              : phase === "advise" ? "advising"
                : phase === "act" ? "acting"
                  : phase === "settle" ? "settled" : "observing";
        const statusText = { speaking: "交流中", thinking: "判断中", advising: "听取建议", acting: "提交动作", settled: "已结算", observing: "观察中" }[status];
        const line =
          phase === "act" || phase === "settle" ? state.plan.actions[index].label
            : phase === "advise" ? `建议 · ${state.plan.advice[index].headline}`
              : phase === "belief" ? state.plan.beliefs[index].text
                : `${agent.persona} · 现金 ${money(agent.cash)}`;
        return (
          <button
            type="button"
            key={agent.companyId}
            className={`cc-card ${focus === agent.companyId ? "is-focus" : ""}`}
            style={{ "--agent": color } as CSSProperties}
            onClick={() => setFocus(focus === agent.companyId ? "" : agent.companyId)}
            aria-pressed={focus === agent.companyId}
          >
            <span className="cc-card-avatar"><PixelAvatar index={index} color={color} size={36} /></span>
            <span className="cc-card-body">
              <span className="cc-card-top">
                <strong>{agent.companyName}</strong>
                <em>{DRIVER_TAG[state.configs[index].driver]}</em>
              </span>
              <span className="cc-card-metrics">
                <b>{agent.share.toFixed(1)}<small>%</small></b>
                <span>¥{yuan(agent.price)}</span>
                <span className={agent.shareDelta > 0 ? "up" : agent.shareDelta < 0 ? "down" : ""}>
                  {agent.shareDelta > 0 ? "▲" : agent.shareDelta < 0 ? "▼" : "·"}{Math.abs(agent.shareDelta).toFixed(1)}
                </span>
              </span>
              <span className="cc-card-foot">
                <span>{line}</span>
                <i className={`cc-status ${status}`}>{statusText}</i>
              </span>
            </span>
          </button>
        );
      })}
    </section>
  );
}

/* ---------- arena ---------- */

type Impact = ScreenState["plan"]["impacts"][number];

function ImpactLine({ impact }: { impact: Impact }) {
  const good = impact.profitDelta >= 0;
  return (
    <span className={`cc-impact ${good ? "good" : "bad"}`}>
      <i>{impact.adopted ? "采纳带来" : "若采纳"}</i>
      利润 {signedMoney(impact.profitDelta)} · 份额 {signedPp(impact.shareDelta)}
      {impact.resilienceDelta !== 0 && <> · 韧性 {impact.resilienceDelta > 0 ? "+" : "−"}{Math.abs(impact.resilienceDelta).toFixed(1)}</>}
    </span>
  );
}

function AdviceCard({ side, phase, advice, impact }: {
  side: "left" | "right";
  phase: PhaseId;
  advice: ScreenState["plan"]["advice"][number];
  impact: Impact;
}) {
  const decided = phase === "act" || phase === "settle";
  return (
    <aside className={`cc-advice side-${side} ${decided ? (advice.adopted ? "adopted" : "rejected") : ""}`} style={{ "--ph": PHENOMENON_COLOR[advice.phenomenon] } as CSSProperties}>
      <span>
        <em>{advice.phenomenon}</em>
        {decided ? <strong>{advice.adopted ? "✓ 接受建议" : "✗ 拒绝建议"}</strong> : <strong className="pending">建议中…</strong>}
      </span>
      <b>{advice.headline}</b>
      {phase === "advise" && <small>{advice.detail}</small>}
      {phase === "settle" && <ImpactLine impact={impact} />}
    </aside>
  );
}

function Arena({ state, colors, focus, setFocus }: { state: ScreenState; colors: Record<string, string>; focus: string; setFocus: (id: string) => void }) {
  const phase = phaseId(state);
  const points = useMemo(() => layout(state.agents.length), [state.agents.length]);
  const indexOf = useMemo(() => new Map(state.agents.map((a, i) => [a.companyId, i])), [state.agents]);
  const showLines = phase === "communicate" || phase === "belief";
  const showAdvice = phase === "advise" || phase === "act" || phase === "settle";
  const firstMessage = new Map<string, ScreenMessage>();
  for (const m of state.plan.messages) if (!firstMessage.has(m.from)) firstMessage.set(m.from, m);
  const avg = state.agents.reduce((s, a) => s + a.price, 0) / state.agents.length;

  const bubble = (agent: AgentRuntimeView, index: number): { kind: string; tag: string; text: string } | null => {
    if (phase === "communicate") {
      const m = firstMessage.get(agent.companyId);
      return m ? { kind: m.kind, tag: KIND_LABEL[m.kind], text: m.text } : null;
    }
    if (phase === "belief") {
      const b = state.plan.beliefs[index];
      return { kind: "belief", tag: "判断", text: b.targetId ? `${agentName(state.agents, b.targetId)} 可能${moveLabel(b.move)} ${b.probability}%` : "对手未知" };
    }
    if (phase === "act" || phase === "settle") {
      const action = state.plan.actions[index];
      const why = state.plan.decisions[index]?.reasons[0];
      return { kind: action.priceDelta < 0 ? "cut" : action.priceDelta > 0 ? "raise" : "hold", tag: "动作", text: why && phase === "act" ? `${action.label}｜${why}` : action.label };
    }
    return null;
  };

  return (
    <section className="cc-arena" aria-label="智能体博弈现场">
      <div className="cc-arena-head">
        <span>ARENA · 博弈现场</span>
        <ol className="cc-phases" aria-label="回合阶段">
          {PHASES.map((p, i) => (
            <li key={p.id} className={i === state.phase ? "is-now" : i < state.phase ? "is-done" : ""}>
              <i>{i + 1}</i>{p.label}
            </li>
          ))}
        </ol>
      </div>
      <div className="cc-stage">
        <svg className="cc-links" viewBox="0 0 1000 600" preserveAspectRatio="none" aria-hidden="true">
          <ellipse cx={HUB.x} cy={HUB.y} rx="400" ry="200" className="cc-orbit" />
          <ellipse cx={HUB.x} cy={HUB.y} rx="290" ry="140" className="cc-orbit thin" />
          {points.map((p, i) => <line key={i} x1={p.x} y1={p.y} x2={HUB.x} y2={HUB.y} className="cc-spoke" />)}
          {showLines && state.plan.messages.map((m) => {
            const from = indexOf.get(m.from);
            if (from === undefined) return null;
            const to = m.to ? indexOf.get(m.to) : undefined;
            const d = curve(points[from], to === undefined ? HUB : points[to]);
            const dim = focus && focus !== m.from && focus !== m.to;
            const color = m.kind === "response" && m.status === "rejected" ? KIND_COLOR.threat : KIND_COLOR[m.kind];
            return (
              <g key={m.id} className={`cc-link ${dim ? "is-dim" : ""}`} style={{ "--link": color } as CSSProperties}>
                <path d={d} className="cc-link-glow" />
                <path d={d} className="cc-link-path" />
                <circle r="4" className="cc-link-dot">
                  <animateMotion dur="2.8s" repeatCount="indefinite" path={d} />
                </circle>
              </g>
            );
          })}
        </svg>

        <div className="cc-hub">
          <div className="cc-hub-title"><span>SHARED MARKET</span><b>共享市场 · R{pad2(state.round)}</b></div>
          <div className="cc-hub-stats">
            <span><small>成交均价</small>¥{(avg / 100).toFixed(1)}</span>
            <span><small>本轮需求</small>{state.agents[0]?.observation.public[0]?.value ?? "—"}</span>
            <span><small>集中度 HHI</small>{hhi(state.agents)}</span>
          </div>
          <div className="cc-hub-bars" aria-hidden="true">
            {state.agents.map((a) => (
              <i key={a.companyId} style={{ height: `${Math.min(100, a.share * 2.6)}%`, background: colors[a.companyId] }} />
            ))}
          </div>
        </div>

        {state.agents.map((agent, index) => {
          const p = points[index];
          const b = bubble(agent, index);
          const advice = state.plan.advice[index];
          const color = colors[agent.companyId];
          const dim = focus && focus !== agent.companyId;
          return (
            <div
              key={agent.companyId}
              className={`cc-node ${dim ? "is-dim" : ""} ${focus === agent.companyId ? "is-focus" : ""}`}
              style={{ left: `${p.x / 10}%`, top: `${p.y / 6}%`, "--agent": color } as CSSProperties}
            >
              {b && (
                <div key={`${state.seq}-${b.kind}`} className={`cc-bubble k-${b.kind}`}>
                  <em>{b.tag}</em>{b.text}
                </div>
              )}
              <button
                type="button"
                className="cc-node-body"
                onClick={() => setFocus(focus === agent.companyId ? "" : agent.companyId)}
                aria-pressed={focus === agent.companyId}
                aria-label={`${agent.companyName}，份额 ${agent.share.toFixed(1)}%`}
              >
                <PixelAvatar index={index} color={color} size={54} />
              </button>
              <span className="cc-node-name">{agent.companyName}</span>
              {phase === "settle" && agent.shareDelta !== 0 && (
                <span className={`cc-delta ${agent.shareDelta > 0 ? "up" : "down"}`}>
                  {agent.shareDelta > 0 ? "+" : ""}{agent.shareDelta.toFixed(1)}pp
                </span>
              )}
              {showAdvice && advice && (
                <AdviceCard
                  key={`${state.round}-${phase}`}
                  side={p.side}
                  phase={phase}
                  advice={advice}
                  impact={state.plan.impacts[index]}
                />
              )}
            </div>
          );
        })}
      </div>
      <div className="cc-legend" aria-label="连线图例">
        {(Object.keys(KIND_LABEL) as ScreenMessage["kind"][]).map((k) => (
          <span key={k}><i style={{ background: KIND_COLOR[k] }} />{KIND_LABEL[k]}</span>
        ))}
        <span className="cc-legend-note">连到中心 = 公开广播 · 消息未经验证、不具约束力 · 点击头像聚焦</span>
      </div>
    </section>
  );
}

/* ---------- detail panels ---------- */

function Table({ head, rows, className }: { head: ReactNode[]; rows: ReactNode[][]; className?: string }) {
  return (
    <div className={`cc-table ${className ?? ""}`}>
      <table>
        <thead><tr>{head.map((h, i) => <th key={i}>{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

function StrategyPanel({ state, colors, focus }: { state: ScreenState; colors: Record<string, string>; focus: string }) {
  const previous = state.phase < 3 && state.lastPlan !== null;
  const plan = previous && state.lastPlan ? state.lastPlan : state.plan;
  const revealed = previous || state.phase >= 3;
  const acted = previous || state.phase >= 4;
  const settled = previous || state.phase >= 5;
  const t = state.totals;
  const rows = state.agents
    .map((agent, i) => ({ agent, i }))
    .filter(({ agent }) => !focus || focus === agent.companyId)
    .map(({ agent, i }) => {
      const b = plan.beliefs[i];
      const a = plan.advice[i];
      return [
        <Who key="w" state={state} id={agent.companyId} colors={colors} />,
        <span key="b" className="cc-belief">
          {b.targetId ? (
            <>
              <b>{agentName(state.agents, b.targetId)}</b> {strategyLabel(b.strategy)} · 下轮{moveLabel(b.move)}
              <span className="cc-prob"><i style={{ width: `${b.probability}%` }} />{b.probability}%</span>
              <small>{b.evidence[1] ?? b.evidence[0]}</small>
            </>
          ) : <span className="cc-muted">未知（无公开历史）</span>}
        </span>,
        revealed ? <span key="p" className="cc-pill" style={{ "--ph": PHENOMENON_COLOR[a.phenomenon] } as CSSProperties}>{a.phenomenon}</span> : <span key="p" className="cc-muted">待生成</span>,
        revealed ? <span key="a" className="cc-advice-text"><b>{a.headline}</b><small>{a.detail}</small></span> : <span key="a" className="cc-muted">—</span>,
        acted ? <span key="r" className={a.adopted ? "cc-ok" : "cc-bad"}>{a.adopted ? "接受" : "拒绝"}<small>{plan.actions[i].label}</small><small className="cc-why">{plan.decisions[i].reasons.join(" · ")}</small></span> : <span key="r" className="cc-muted">待提交</span>,
        settled ? <ImpactLine key="i" impact={plan.impacts[i]} /> : <span key="i" className="cc-muted">结算后计算</span>,
      ];
    });
  const history = state.adviceHistory.slice(-12);
  return (
    <div className="cc-tab-body">
      <div className="cc-kpi-row">
        <div><small>建议采纳率</small><b>{t.adviceTotal ? Math.round((t.adviceAdopted / t.adviceTotal) * 100) : 0}%</b></div>
        <div><small>采纳建议累计带来利润</small><b className={t.adoptedGain >= 0 ? "cc-ok" : "cc-bad"}>{signedMoney(t.adoptedGain)}</b></div>
        <div><small>拒绝建议累计错失利润</small><b className={t.missedGain > 0 ? "cc-warn" : "cc-ok"}>{signedMoney(t.missedGain)}</b></div>
        <div><small>建议利润非负占比</small><b>{t.adviceTotal ? Math.round((t.helpful / t.adviceTotal) * 100) : 0}%</b></div>
      </div>
      <h3>{previous ? `上一轮 R${pad2(plan.round)} 已结算结果` : `本轮 R${pad2(plan.round)}`} <small>{previous ? "本轮建议生成后自动切换" : PHASES[state.phase].label}</small></h3>
      <Table className="wide" head={["智能体", "对手判断（依据公开证据）", "博弈现象", "战略建议", "是否接受 · 实际动作", "建议带来的影响（本轮）"]} rows={rows} />
      <p className="cc-note">影响 = 同一演示市场、其他企业动作不变时，采纳建议与不采纳的结果差。已接受：实际 − 维持上轮动作；已拒绝：照建议执行 − 实际。仅为演示反事实，非因果研究结论。</p>
      <div className="cc-sub">
        <h3>现象热力图 <small>最近 {history.length} 轮 · 每格为该轮诊断现象，空心 = 未采纳</small></h3>
        <div className="cc-heat" style={{ gridTemplateColumns: `90px repeat(${Math.max(1, history.length)}, 1fr)` }}>
          {state.agents.map((agent) => (
            <div key={agent.companyId} className="cc-heat-row">
              <span style={{ color: colors[agent.companyId] }}>{agent.companyName}</span>
              {history.map((h) => {
                const item = h.items.find((x) => x.companyId === agent.companyId);
                const c = item ? PHENOMENON_COLOR[item.phenomenon] : "transparent";
                return <i key={h.round} title={`R${h.round} ${item?.phenomenon ?? ""}`} className={item?.adopted ? "" : "hollow"} style={{ "--ph": c } as CSSProperties} />;
              })}
            </div>
          ))}
          {history.length === 0 && <p className="cc-muted">首轮结算后开始记录。</p>}
        </div>
        <div className="cc-ph-legend">
          {PHENOMENA.map((p) => <span key={p}><i style={{ background: PHENOMENON_COLOR[p] }} />{p}</span>)}
        </div>
      </div>
    </div>
  );
}

function TalkPanel({ state, colors, focus }: { state: ScreenState; colors: Record<string, string>; focus: string }) {
  const visible = state.phase >= 1;
  const messages = visible ? state.plan.messages.filter((m) => !focus || m.from === focus || m.to === focus) : [];
  const agreements = [...state.agreements].reverse().filter((a) => !focus || a.proposer === focus || a.partner === focus).slice(0, 8);
  const ids = state.agents.map((a) => a.companyId);
  const max = Math.max(1, ...Object.values(state.links).flatMap((row) => Object.values(row)));
  return (
    <div className="cc-tab-body cols">
      <div className="cc-col">
        <h3>本轮对话 <small>R{pad2(state.round)} · {messages.length} 条</small></h3>
        <ul className="cc-chat">
          {!visible && <li className="cc-muted">交流阶段尚未开始。</li>}
          {visible && messages.length === 0 && <li className="cc-muted">本轮无消息。</li>}
          {messages.map((m) => (
            <li key={m.id} style={{ "--k": KIND_COLOR[m.kind] } as CSSProperties}>
              <div><Who state={state} id={m.from} colors={colors} /><span className="cc-arrow">→</span><Who state={state} id={m.to} colors={colors} /><em>{KIND_LABEL[m.kind]}</em></div>
              <p>{m.text}</p>
            </li>
          ))}
        </ul>
      </div>
      <div className="cc-col">
        <h3>协议公式与实际执行 <small>守约 {state.totals.kept} · 违约 {state.totals.broken}</small></h3>
        <ul className="cc-deals">
          {agreements.length === 0 && <li className="cc-muted">尚无被接受的提议。</li>}
          {agreements.map((ag) => (
            <li key={ag.id} className={ag.outcome ?? "pending"}>
              <div className="cc-deal-head">
                <span>R{pad2(ag.round)}</span>
                <Who state={state} id={ag.proposer} colors={colors} /><span className="cc-arrow">⇄</span><Who state={state} id={ag.partner} colors={colors} />
                <em>{ag.outcome === "kept" ? "双方守约" : ag.outcome === "broken" ? "双方违约" : ag.outcome === "partial" ? "单方违约" : "待结算"}</em>
              </div>
              <code>{termsText(ag.terms)}</code>
              {ag.actual && (
                <div className="cc-deal-actual">
                  {[ag.proposer, ag.partner].map((id) => {
                    const act = ag.actual![id];
                    return (
                      <span key={id} className={act.kept ? "cc-ok" : "cc-bad"}>
                        {agentName(state.agents, id)}：¥{yuan(act.price)} · 共建 ¥{num(yuan(act.contribution))} {act.kept ? "✓" : "✗"}
                      </span>
                    );
                  })}
                </div>
              )}
            </li>
          ))}
        </ul>
      </div>
      <div className="cc-col narrow">
        <h3>通信矩阵 <small>累计条数 · 行=发送</small></h3>
        <div className="cc-matrix" style={{ gridTemplateColumns: `52px repeat(${ids.length + 1}, 1fr)` }}>
          <span />
          {ids.map((id, i) => <span key={id} className="cc-matrix-h" style={{ color: colors[id] }}>{String.fromCharCode(65 + i)}</span>)}
          <span className="cc-matrix-h">广播</span>
          {ids.map((from, i) => (
            <div key={from} className="cc-matrix-row">
              <span className="cc-matrix-h" style={{ color: colors[from] }}>{String.fromCharCode(65 + i)} {state.agents[i].companyName.slice(0, 2)}</span>
              {[...ids, "ALL"].map((to) => {
                const v = state.links[from]?.[to] ?? 0;
                return <i key={to} style={{ "--v": v / max } as CSSProperties}>{from === to ? "" : v || ""}</i>;
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Meter({ value, max, color }: { value: number; max: number; color: string }) {
  return <span className="cc-meter"><i style={{ width: `${Math.min(100, (value / Math.max(1, max)) * 100)}%`, background: color }} /></span>;
}

function OpsPanel({ state, colors, focus }: { state: ScreenState; colors: Record<string, string>; focus: string }) {
  const acted = state.phase >= 4;
  const ops = state.plan.ops.filter((o) => !focus || o.companyId === focus);
  const maxAd = Math.max(1, ...state.plan.ops.map((o) => o.advertising));
  const maxSvc = Math.max(1, ...state.plan.ops.map((o) => o.service));
  const rows = ops.map((o) => {
    const i = state.agents.findIndex((a) => a.companyId === o.companyId);
    const before = state.agents[i];
    const c = colors[o.companyId];
    const show = (v: ReactNode) => (acted ? v : <span className="cc-muted">待提交</span>);
    return [
      <Who key="w" state={state} id={o.companyId} colors={colors} />,
      show(<span className="cc-num">¥{yuan(before.price)} → <b>¥{yuan(o.price)}</b></span>),
      show(<span className="cc-bar-cell"><Meter value={o.advertising} max={maxAd} color={c} />{money(o.advertising)}</span>),
      show(<span className="cc-bar-cell"><Meter value={o.service} max={maxSvc} color={c} />{money(o.service)}</span>),
      show(<span className="cc-num">{money(o.capacityInvest)}</span>),
      show(<span className="cc-num">{money(o.resilience)}</span>),
      show(<span className={o.contribution === 0 ? "cc-bad" : "cc-num"}>{o.contribution === 0 ? "停止" : money(o.contribution)}</span>),
      show(<span className="cc-num">¥{yuan(o.unitCost)}</span>),
    ];
  });
  return (
    <div className="cc-tab-body">
      <Table className="wide" head={["智能体", "定价", "广告投放", "服务投入", "扩产投资", "韧性预算", "公共共建", "单位成本"]} rows={rows} />
      <p className="cc-note">动作在“提交动作”阶段同时揭示；各方只看到公开信息，提交前互不可见。</p>
    </div>
  );
}

function StockTable({ state, colors, focus }: { state: ScreenState; colors: Record<string, string>; focus: string }) {
  const settled = state.phase >= 5;
  const useCurrent = settled || state.lastOps.length === 0;
  const source = useCurrent ? state.plan.ops : state.lastOps;
  const roundLabel = useCurrent ? state.round : state.round - 1;
  const ops = source.filter((o) => !focus || o.companyId === focus);
  const shown = !useCurrent || settled;
  const rows = ops.map((o) => {
    const c = colors[o.companyId];
    const util = Math.round((o.procured / Math.max(1, o.capacity)) * 100);
    const show = (v: ReactNode) => (shown ? v : <span className="cc-muted">待结算</span>);
    return [
      <Who key="w" state={state} id={o.companyId} colors={colors} />,
      <span key="s" className="cc-num">{num(o.stockStart)}</span>,
      <span key="p" className="cc-bar-cell"><Meter value={o.procured} max={o.capacity} color={c} />{num(o.procured)} <small>{util}%</small></span>,
      show(<span className="cc-num">{num(o.orders)}</span>),
      show(<span className="cc-num">{num(o.sold)}</span>),
      show(<span className={o.shortage ? "cc-bad" : "cc-num"}>{num(o.shortage)}</span>),
      show(<span className={o.spoiled ? "cc-warn" : "cc-num"}>{num(o.spoiled)}</span>),
      show(<b className="cc-num">{num(o.stockEnd)}</b>),
    ];
  });
  return (
    <div className="cc-col wide-col">
      <h3>内部库存 <small>R{pad2(roundLabel)} · 累计缺货 {num(state.totals.shortage)} · 累计损耗 {num(state.totals.spoiled)}</small></h3>
      <Table head={["智能体", "期初", "采购·产能", "订单", "成交", "缺货", "损耗", "期末"]} rows={rows} />
      <p className="cc-note">演示示意：采购 = 上轮需求 × 1.08 − 期初，受产能约束；未售损耗 15%；期初 + 采购 − 成交 − 损耗 = 期末。不反馈到利润。</p>
    </div>
  );
}

function LineChart({ state, colors, pick, label, format }: {
  state: ScreenState;
  colors: Record<string, string>;
  pick: (h: ScreenState["history"][number], id: string) => number;
  label: string;
  format: (v: number) => string;
}) {
  const W = 460, H = 150, padX = 38, padY = 16;
  const points = state.history.filter((h) => h.round > 0 || label === "份额");
  const values = points.flatMap((h) => state.agents.map((a) => pick(h, a.companyId)));
  const lo = label === "份额" ? 0 : Math.floor(Math.min(...values, Infinity) - 2);
  const hi = Math.max(...values, lo + 1) * (label === "份额" ? 1.1 : 1) + (label === "份额" ? 0 : 2);
  const x = (r: number) => padX + (r / SCREEN_ROUNDS) * (W - padX - 8);
  const y = (v: number) => H - padY - ((v - lo) / (hi - lo)) * (H - padY * 2);
  return (
    <figure className="cc-chart">
      <figcaption>{label}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label}随回合变化`}>
        {[0, 0.5, 1].map((t) => {
          const v = lo + (hi - lo) * t;
          return (
            <g key={t} className="cc-axis">
              <line x1={padX} x2={W - 8} y1={y(v)} y2={y(v)} />
              <text x={padX - 4} y={y(v) + 3} textAnchor="end">{format(v)}</text>
            </g>
          );
        })}
        {points.length > 0 && state.agents.map((a) => (
          <polyline key={a.companyId} fill="none" stroke={colors[a.companyId]} strokeWidth="1.8" strokeLinejoin="round"
            points={points.map((h) => `${x(h.round).toFixed(1)},${y(pick(h, a.companyId)).toFixed(1)}`).join(" ")} />
        ))}
        {points.length === 0 && <text x={W / 2} y={H / 2} textAnchor="middle" className="cc-axis-empty">首轮结算后开始绘制</text>}
      </svg>
    </figure>
  );
}

function MarketPanel({ state, colors, focus }: { state: ScreenState; colors: Record<string, string>; focus: string }) {
  const ranked = [...state.agents].sort((a, b) => b.share - a.share);
  return (
    <div className="cc-tab-body cols">
      <div className="cc-col charts">
        <LineChart state={state} colors={colors} label="份额" pick={(h, id) => h.shares[id] ?? 0} format={(v) => `${v.toFixed(0)}%`} />
        <LineChart state={state} colors={colors} label="价格" pick={(h, id) => (h.prices[id] ?? 0) / 100} format={(v) => `¥${v.toFixed(0)}`} />
      </div>
      <div className="cc-col">
        <h3>排行榜 <small>按份额 · 每轮总和 100%</small></h3>
        <Table
          head={["#", "智能体", "份额", "本轮利润", "现金"]}
          rows={ranked.map((a, i) => [
            <b key="i" className="cc-rank">{i + 1}</b>,
            <Who key="w" state={state} id={a.companyId} colors={colors} />,
            <span key="s" className="cc-bar-cell"><Meter value={a.share} max={ranked[0].share} color={colors[a.companyId]} />{a.share.toFixed(1)}%</span>,
            <span key="p" className={a.profit < 0 ? "cc-bad" : "cc-ok"}>{money(a.profit)}</span>,
            <span key="c" className="cc-num">{money(a.cash)}</span>,
          ])}
        />
      </div>
      <StockTable state={state} colors={colors} focus={focus} />
    </div>
  );
}

function DetailPanel({ state, colors, focus, tab, setTab, auto, setAuto }: {
  state: ScreenState;
  colors: Record<string, string>;
  focus: string;
  tab: TabId;
  setTab: (t: TabId) => void;
  auto: boolean;
  setAuto: (v: boolean) => void;
}) {
  return (
    <section className="cc-panel cc-detail" aria-label="详细面板">
      <div className="cc-tabs" role="tablist" aria-label="面板切换">
        {TABS.map((t) => (
          <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}>
            <small>{t.en}</small>{t.label}
          </button>
        ))}
        <label className="cc-auto">
          <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
          跟随阶段自动切换
        </label>
        {focus && <span className="cc-focus-tag">仅显示：{agentName(state.agents, focus)}</span>}
      </div>
      <div key={tab} className="cc-tab-wrap" role="tabpanel">
        {tab === "strategy" && <StrategyPanel state={state} colors={colors} focus={focus} />}
        {tab === "talk" && <TalkPanel state={state} colors={colors} focus={focus} />}
        {tab === "ops" && <OpsPanel state={state} colors={colors} focus={focus} />}
        {tab === "market" && <MarketPanel state={state} colors={colors} focus={focus} />}
      </div>
    </section>
  );
}

/* ---------- side column ---------- */

function Overview({ state, colors }: { state: ScreenState; colors: Record<string, string> }) {
  const r = 70;
  const circumference = 2 * Math.PI * r;
  const arcs = state.agents.map((a, i) => ({
    a,
    len: (a.share / 100) * circumference,
    offset: state.agents.slice(0, i).reduce((s, b) => s + (b.share / 100) * circumference, 0),
  }));
  const totalProfit = state.agents.reduce((s, a) => s + a.profit, 0);
  const adopt = state.totals.adviceTotal ? (state.totals.adviceAdopted / state.totals.adviceTotal) * 100 : 0;
  const resolved = state.totals.kept + state.totals.broken;
  return (
    <section className="cc-panel cc-overview">
      <h2>OVERVIEW <span>市场总览</span></h2>
      <div className="cc-overview-body">
        <svg viewBox="0 0 180 180" role="img" aria-label="各企业市场份额环形图">
          <circle cx="90" cy="90" r="86" className="cc-ring-bg" />
          <g transform="rotate(-90 90 90)">
            {arcs.map(({ a, len, offset }) => (
              <circle key={a.companyId} cx="90" cy="90" r={r} fill="none" stroke={colors[a.companyId]} strokeWidth="16"
                strokeDasharray={`${Math.max(0, len - 2)} ${circumference}`} strokeDashoffset={-offset} className="cc-ring-arc" />
            ))}
          </g>
          <g className="cc-sweep"><path d="M90 90 L90 6 A84 84 0 0 1 149 30 Z" /></g>
          <text x="90" y="88" textAnchor="middle" className="cc-ring-num">{state.agents.length}</text>
          <text x="90" y="106" textAnchor="middle" className="cc-ring-cap">AGENTS</text>
        </svg>
        <dl className="cc-kpis">
          <div><dt>本轮总利润</dt><dd className={totalProfit < 0 ? "neg" : ""}>{money(totalProfit)}</dd></div>
          <div><dt>建议采纳率</dt><dd>{adopt.toFixed(0)}%</dd></div>
          <div><dt>承诺履约率</dt><dd>{resolved ? `${((state.totals.kept / resolved) * 100).toFixed(0)}%` : "—"}</dd></div>
          <div><dt>累计消息</dt><dd>{state.totals.messages}</dd></div>
        </dl>
      </div>
    </section>
  );
}

const TONE_CLASS: Record<FeedItem["tone"], string> = { info: "t-info", talk: "t-talk", advice: "t-advice", action: "t-action", result: "t-result", alert: "t-alert" };

function LiveFeed({ state, colors }: { state: ScreenState; colors: Record<string, string> }) {
  return (
    <section className="cc-panel cc-feed">
      <h2>LIVE FEED <span>事件流</span><small>E{state.episode} · R{state.round} · {PHASES[state.phase].en}</small></h2>
      <ol>
        {state.feed.slice(0, 30).map((item) => (
          <li key={item.id} className={TONE_CLASS[item.tone]}>
            <time>R{pad2(item.round)}</time>
            <b style={item.companyId ? { color: colors[item.companyId] } : undefined}>{item.companyId ? agentName(state.agents, item.companyId) : "系统"}</b>
            <em>{item.verb}</em>
            <span>{item.text}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

/* ---------- root ---------- */

export function CommandCenter({ onExit, initialAgents = 6 }: { onExit?: () => void; initialAgents?: number }) {
  const [state, dispatch] = useReducer(screenReducer, initialAgents, createScreen);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [focus, setFocus] = useState("");
  const [fullscreen, setFullscreen] = useState(false);
  const [manualTab, setManualTab] = useState<TabId>("market");
  const [auto, setAuto] = useState(true);
  const rootRef = useRef<HTMLDivElement>(null);
  const colors = useColors(state);
  const tab = auto ? TAB_FOR_PHASE[phaseId(state)] : manualTab;

  useEffect(() => {
    if (paused) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) dispatch({ type: "tick" });
    }, SPEEDS[speed].ms);
    return () => window.clearInterval(timer);
  }, [paused, speed]);

  useEffect(() => {
    const onChange = () => setFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void rootRef.current?.requestFullscreen?.();
  };
  const focusValid = state.agents.some((a) => a.companyId === focus) ? focus : "";

  return (
    <div className="cc-root" ref={rootRef}>
      <Header
        state={state}
        paused={paused}
        setPaused={setPaused}
        speed={speed}
        setSpeed={setSpeed}
        onStep={() => dispatch({ type: "tick" })}
        onCount={(count) => { setFocus(""); dispatch({ type: "reset", count }); }}
        onFullscreen={toggleFullscreen}
        fullscreen={fullscreen}
        onExit={onExit}
      />
      <AgentStrip state={state} colors={colors} focus={focusValid} setFocus={setFocus} />
      <div className="cc-main">
        <Arena state={state} colors={colors} focus={focusValid} setFocus={setFocus} />
        <div className="cc-side">
          <Overview state={state} colors={colors} />
          <LiveFeed state={state} colors={colors} />
        </div>
      </div>
      <DetailPanel
        state={state}
        colors={colors}
        focus={focusValid}
        tab={tab}
        setTab={(t) => { setManualTab(t); setAuto(false); }}
        auto={auto}
        setAuto={(v) => { if (!v) setManualTab(tab); setAuto(v); }}
      />
      <footer className="cc-footer">
        <span>UI-ONLY DEMO MARKET · 份额每轮归一化为 100% · 回合 1→{SCREEN_ROUNDS} 不循环，结束后开启新局</span>
        <span>建议为前端启发式，不替代后端 Advisor · 真实结果以后端 MarketEnv 与 Replay 为准</span>
      </footer>
    </div>
  );
}
