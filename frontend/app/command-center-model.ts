/**
 * Pure state machine behind the 博弈指挥大屏 (command-center screen).
 *
 * It wraps the deterministic UI-only `advanceDemoRound` market and adds the
 * presentation layer the big screen needs: per-phase reveal, generated
 * messages and agreements, belief summaries, heuristic strategy advice,
 * an illustrative inventory flow and a live feed. Everything here is
 * interaction demo data, never research evidence.
 */
import {
  AGENT_TEMPLATES,
  advanceDemoRound,
  buildRuntimeAgents,
  demoDemand,
  demoUnitCost,
  PERSONAS,
  type AgentConfig,
  type AgentRuntimeView,
} from "./lab-model.ts";

export const SCREEN_ROUNDS = 20;
export const SCREEN_MIN_AGENTS = 2;
export const SCREEN_MAX_AGENTS = AGENT_TEMPLATES.length;
export const FEED_LIMIT = 80;

export const PHASES = [
  { id: "observe", label: "接收信息", en: "OBSERVE" },
  { id: "communicate", label: "语言交流", en: "TALK" },
  { id: "belief", label: "对手判断", en: "BELIEF" },
  { id: "advise", label: "战略建议", en: "ADVISE" },
  { id: "act", label: "提交动作", en: "ACT" },
  { id: "settle", label: "联合结算", en: "SETTLE" },
] as const;

export type PhaseId = (typeof PHASES)[number]["id"];

export type MessageKind = "proposal" | "response" | "threat" | "signal" | "statement";

export type AgreementTerms = { minPrice: number; minContribution: number };

export type ScreenMessage = {
  id: string;
  round: number;
  from: string;
  to: string | null;
  kind: MessageKind;
  text: string;
  status?: "accepted" | "rejected";
  terms?: AgreementTerms;
};

export type Phenomenon = "信息不足" | "价格战" | "搭便车" | "份额流失" | "价格敏感" | "协调均衡";

export const PHENOMENA: Phenomenon[] = ["信息不足", "价格战", "搭便车", "份额流失", "价格敏感", "协调均衡"];

export type ScreenAdvice = {
  companyId: string;
  phenomenon: Phenomenon;
  headline: string;
  detail: string;
  recommendedPrice: number;
  adopted: boolean;
};

export type PredictedMove = "cut" | "maintain" | "raise";

export type BeliefSummary = {
  companyId: string;
  /** Most threatening opponent, or null before any public evidence exists. */
  targetId: string | null;
  move: PredictedMove | null;
  probability: number | null;
  strategy: "growth" | "profit" | "defensive" | null;
  evidence: string[];
  text: string;
};

export type Agreement = {
  id: string;
  round: number;
  proposer: string;
  partner: string;
  terms: AgreementTerms;
  /** Filled at settlement: what each side actually did. */
  actual?: Record<string, { price: number; contribution: number; kept: boolean }>;
  outcome?: "kept" | "broken" | "partial";
};

export type OpsRecord = {
  companyId: string;
  capacity: number;
  stockStart: number;
  procured: number;
  orders: number;
  sold: number;
  shortage: number;
  spoiled: number;
  stockEnd: number;
  unitCost: number;
  price: number;
  advertising: number;
  service: number;
  capacityInvest: number;
  resilience: number;
  contribution: number;
};

/**
 * Effect of adopting the advice, measured in the same demo market with all other
 * agents' actions held fixed: (world where the agent adopts) − (world where it does not).
 * Adopted: actual − "keep last round's action". Not adopted: "follow advice" − actual.
 */
export type AdviceImpact = {
  companyId: string;
  adopted: boolean;
  comparedWith: "keep" | "follow";
  profitDelta: number;
  shareDelta: number;
  resilienceDelta: number;
};

/** Why an agent chose its action: the weighed inputs, shown on the screen. */
export type DecisionTrace = {
  companyId: string;
  /** Price the persona would pick on its own before advice and agreements. */
  instinctPrice: number;
  /** Probability (0–100) of following the advice, from persona + track record. */
  trust: number;
  followed: boolean;
  /** Agreement price floor the agent was bound to, if any. */
  floor: number | null;
  honoured: boolean | null;
  reasons: string[];
};

export type ActionSummary = {
  companyId: string;
  price: number;
  priceDelta: number;
  advertising: number;
  contribution: number;
  label: string;
};

export type FeedTone = "info" | "talk" | "advice" | "action" | "result" | "alert";

export type FeedItem = {
  id: string;
  episode: number;
  round: number;
  phase: PhaseId;
  companyId: string | null;
  verb: string;
  text: string;
  tone: FeedTone;
};

export type HistoryPoint = {
  round: number;
  avgPrice: number;
  totalProfit: number;
  hhi: number;
  demand: number;
  shares: Record<string, number>;
  prices: Record<string, number>;
  profits: Record<string, number>;
  cash: Record<string, number>;
};

export type AdviceRecord = { round: number; items: Array<Pick<ScreenAdvice, "companyId" | "phenomenon" | "adopted">> };

export type RoundPlan = {
  round: number;
  messages: ScreenMessage[];
  agreements: Agreement[];
  beliefs: BeliefSummary[];
  advice: ScreenAdvice[];
  impacts: AdviceImpact[];
  decisions: DecisionTrace[];
  actions: ActionSummary[];
  ops: OpsRecord[];
  next: AgentRuntimeView[];
  feed: FeedItem[];
};

export type ScreenTotals = {
  messages: number;
  adviceTotal: number;
  adviceAdopted: number;
  kept: number;
  broken: number;
  spoiled: number;
  shortage: number;
  /** Sum of profit deltas from adopted advice (cents). */
  adoptedGain: number;
  /** Sum of profit deltas the agents would have gained by adopting rejected advice (cents). */
  missedGain: number;
  /** Rounds × agents where adopting was at least as profitable as not adopting. */
  helpful: number;
};

export type ScreenState = {
  episode: number;
  seed: number;
  configs: AgentConfig[];
  agents: AgentRuntimeView[];
  /** Round currently being played (1-based). */
  round: number;
  history: HistoryPoint[];
  adviceHistory: AdviceRecord[];
  agreements: Agreement[];
  /** Cumulative message counts: from → (to | "ALL") → count. */
  links: Record<string, Record<string, number>>;
  inventory: Record<string, number>;
  /** Inventory flow of the most recently settled round. */
  lastOps: OpsRecord[];
  /** Public price history per company (cents), oldest first. */
  priceHistory: Record<string, number[]>;
  /** Running score in [-1, 1] of how well following advice has paid off per company. */
  trackRecord: Record<string, number>;
  totals: ScreenTotals;
  plan: RoundPlan;
  /** Fully settled plan of the previous round, kept so panels never go blank. */
  lastPlan: RoundPlan | null;
  phase: number;
  feed: FeedItem[];
  seq: number;
};

export type ScreenAction = { type: "tick" } | { type: "reset"; count: number };

/** Small deterministic integer hash so every episode replays identically. */
export function mix(...values: number[]): number {
  let h = 2166136261;
  for (const value of values) {
    h ^= value | 0;
    h = Math.imul(h, 16777619);
    h ^= h >>> 13;
  }
  return Math.abs(h);
}

export const yuan = (cents: number) => Math.round(cents / 100);

export function agentName(agents: AgentRuntimeView[], id: string | null): string {
  if (!id) return "全体";
  return agents.find((a) => a.companyId === id)?.companyName ?? id;
}

function averagePrice(agents: AgentRuntimeView[]): number {
  return agents.reduce((sum, a) => sum + a.price, 0) / Math.max(1, agents.length);
}

export function hhi(agents: AgentRuntimeView[]): number {
  return Math.round(agents.reduce((sum, a) => sum + a.share * a.share, 0));
}

function historyPoint(round: number, agents: AgentRuntimeView[]): HistoryPoint {
  const by = (pick: (a: AgentRuntimeView) => number) => Object.fromEntries(agents.map((a) => [a.companyId, pick(a)]));
  return {
    round,
    avgPrice: Math.round(averagePrice(agents)),
    totalProfit: agents.reduce((sum, a) => sum + a.profit, 0),
    hhi: hhi(agents),
    demand: round === 0 ? 0 : demoDemand(round),
    shares: by((a) => a.share),
    prices: by((a) => a.price),
    profits: by((a) => a.profit),
    cash: by((a) => a.cash),
  };
}

function personaOf(config: AgentConfig) {
  return PERSONAS[config.persona];
}

/** Seat A has no human on the big screen, so it follows a deterministic rule schedule. */
function seatAAction(round: number, seed: number, agent: AgentRuntimeView) {
  const swings = [0, -200, 150, -100, 250, -50];
  return {
    price: 9800 + swings[(round + seed) % swings.length],
    advertising: 200_000 + ((round + seed) % 3) * 100_000,
    contribution: (round + seed) % 5 === 0 ? 0 : agent.action.contribution || 200_000,
  };
}

const MOVE_LABEL: Record<PredictedMove, string> = { cut: "降价", maintain: "维持", raise: "提价" };
const STRATEGY_LABEL = { growth: "增长型", profit: "利润型", defensive: "防御型" } as const;

export function moveLabel(move: PredictedMove | null) {
  return move ? MOVE_LABEL[move] : "未知";
}
export function strategyLabel(strategy: BeliefSummary["strategy"]) {
  return strategy ? STRATEGY_LABEL[strategy] : "未知";
}

/**
 * Belief about one rival from public evidence only (its last two public prices and
 * share change), filtered through the observer's own suspicion (opportunism and
 * risk aversion make an observer expect more aggressive rivals).
 */
function beliefAbout(
  observer: AgentConfig,
  rival: AgentRuntimeView,
  rivalHistory: number[],
): { cut: number; maintain: number; raise: number } {
  const traits = PERSONAS[observer.persona].traits;
  const last = rivalHistory.at(-1) ?? rival.price;
  const prev = rivalHistory.at(-2) ?? last;
  const momentum = Math.max(-1, Math.min(1, (last - prev) / 250));
  const shareSignal = Math.max(-1, Math.min(1, rival.shareDelta / 3));
  const suspicion = (traits.opportunism + traits.riskAversion) / 200 - 0.4;
  let cut = 34 - momentum * 22 + shareSignal * 8 + suspicion * 30;
  let raise = 30 + momentum * 22 - shareSignal * 6 - suspicion * 20;
  cut = Math.max(6, Math.min(88, cut));
  raise = Math.max(6, Math.min(88, raise));
  let maintain = Math.max(6, 100 - cut - raise);
  const total = cut + raise + maintain;
  cut = Math.round((cut / total) * 100);
  raise = Math.round((raise / total) * 100);
  maintain = 100 - cut - raise;
  return { cut, maintain, raise };
}

function summarizeBelief(
  config: AgentConfig,
  agent: AgentRuntimeView,
  agents: AgentRuntimeView[],
  round: number,
  priceHistory: Record<string, number[]>,
): BeliefSummary {
  if (round === 1) {
    return {
      companyId: agent.companyId,
      targetId: null,
      move: null,
      probability: null,
      strategy: null,
      evidence: [],
      text: "尚无公开历史，对手判断保持未知",
    };
  }
  // Watch the most direct competitor: closest public price, then the biggest share mover.
  const rivals = agents.filter((a) => a.companyId !== agent.companyId);
  const target = [...rivals].sort(
    (a, b) =>
      Math.abs(a.price - agent.price) - Math.abs(b.price - agent.price) ||
      Math.abs(b.shareDelta) - Math.abs(a.shareDelta),
  )[0];
  const history = priceHistory[target.companyId] ?? [target.price];
  const p = beliefAbout(config, target, history);
  const moves: PredictedMove[] = ["cut", "maintain", "raise"];
  const move = moves.reduce((best, m) => (p[m] > p[best] ? m : best), "maintain" as PredictedMove);
  const strategy =
    target.shareDelta >= 1 || target.action.advertising >= 400_000 ? "growth"
      : target.price >= averagePrice(agents) + 100 ? "profit"
        : target.shareDelta <= -1 ? "defensive" : "profit";
  const prev = history.at(-2);
  const evidence = [
    prev !== undefined
      ? `公开价格 ¥${yuan(prev)} → ¥${yuan(target.price)}`
      : `公开价格 ¥${yuan(target.price)}`,
    `公开份额变化 ${target.shareDelta >= 0 ? "+" : ""}${target.shareDelta.toFixed(1)}pp`,
  ];
  return {
    companyId: agent.companyId,
    targetId: target.companyId,
    move,
    probability: p[move],
    strategy,
    evidence,
    text: `判断 ${agentName(agents, target.companyId)} 下轮${MOVE_LABEL[move]} ${p[move]}%`,
  };
}

function adviseAgent(agent: AgentRuntimeView, agents: AgentRuntimeView[], round: number): Omit<ScreenAdvice, "adopted"> {
  const avg = averagePrice(agents);
  const id = agent.companyId;
  if (round === 1) {
    return {
      companyId: id,
      phenomenon: "信息不足",
      headline: "先观察对手首轮动作",
      detail: "尚无历史行为，对手判断保持均匀分布，不伪造精确概率。",
      recommendedPrice: agent.price,
    };
  }
  if (agent.price < avg - 120) {
    const rec = Math.round(avg / 50) * 50;
    return {
      companyId: id,
      phenomenon: "价格战",
      headline: `回调至 ¥${yuan(rec)}`,
      detail: `低于均价 ¥${yuan(avg - agent.price)}；对手跟进后份额收益被抵消，形成囚徒困境。`,
      recommendedPrice: rec,
    };
  }
  const contributors = agents.filter((a) => a.companyId !== id && a.action.contribution > 0).length;
  if (agent.action.contribution === 0 && contributors > 0) {
    return {
      companyId: id,
      phenomenon: "搭便车",
      headline: "恢复公共韧性投入",
      detail: `另有 ${contributors} 家在投入；共享保护非排他，持续搭便车会触发对方停止合作。`,
      recommendedPrice: agent.price,
    };
  }
  if (agent.shareDelta <= -1.5) {
    return {
      companyId: id,
      phenomenon: "份额流失",
      headline: `小幅降价至 ¥${yuan(agent.price - 150)}`,
      detail: `份额下滑 ${Math.abs(agent.shareDelta).toFixed(1)}pp，建议用服务投入代替深度降价。`,
      recommendedPrice: agent.price - 150,
    };
  }
  if (agent.price > avg + 150) {
    return {
      companyId: id,
      phenomenon: "价格敏感",
      headline: "暂停继续提价",
      detail: "价格敏感段短期利润可能在 5 轮内转亏，建议维持当前价位。",
      recommendedPrice: agent.price,
    };
  }
  return {
    companyId: id,
    phenomenon: "协调均衡",
    headline: `维持 ¥${yuan(agent.price)}`,
    detail: "接近当前最佳回应，单方偏离的预期收益为负。",
    recommendedPrice: agent.price,
  };
}

function actionLabel(before: AgentRuntimeView, after: AgentRuntimeView): string {
  const delta = after.price - before.price;
  const price =
    delta < 0
      ? `降价 ¥${yuan(-delta)} → ¥${yuan(after.price)}`
      : delta > 0
        ? `提价 ¥${yuan(delta)} → ¥${yuan(after.price)}`
        : `维持 ¥${yuan(after.price)}`;
  const ad =
    after.action.advertising !== before.action.advertising
      ? ` · 广告${after.action.advertising > before.action.advertising ? "+" : "-"}`
      : "";
  const contribution = after.action.contribution === 0 ? " · 停止共建" : "";
  return `${price}${ad}${contribution}`;
}

export function termsText(terms: AgreementTerms): string {
  return `价格 ≥ ¥${yuan(terms.minPrice)} ∧ 共建 ≥ ¥${yuan(terms.minContribution).toLocaleString("zh-CN")}`;
}

function generateMessages(
  configs: AgentConfig[],
  agents: AgentRuntimeView[],
  round: number,
  seed: number,
): { messages: ScreenMessage[]; agreements: Agreement[] } {
  const messages: ScreenMessage[] = [];
  const agreements: Agreement[] = [];
  const limit = Math.max(2, Math.ceil(configs.length * 0.7));
  const speakers = configs
    .map((config, index) => ({ config, index, roll: mix(seed, round, index) % 100 }))
    .filter((item) => item.config.communication && item.roll < 70)
    .sort((a, b) => a.roll - b.roll)
    .slice(0, limit);

  for (const { config, index } of speakers) {
    const self = agents[index];
    const others = agents.filter((_, i) => i !== index);
    if (!others.length) break;
    const target = others[mix(seed, round, index, 7) % others.length];
    const persona = personaOf(config);
    const price = yuan(self.price);
    const id = `r${round}-${index}`;
    if (persona.traits.opportunism >= 60) {
      const threat = mix(seed, round, index, 3) % 2 === 0;
      messages.push({
        id,
        round,
        from: self.companyId,
        to: threat ? target.companyId : null,
        kind: threat ? "threat" : "signal",
        text: threat ? `若你继续降价，我方将跟进至 ¥${price - 3}。` : "本轮扩大广告投放，产能充足（未经验证）。",
      });
    } else if (persona.traits.cooperation >= 25) {
      const terms = { minPrice: self.price - 100, minContribution: 150_000 };
      messages.push({
        id,
        round,
        from: self.companyId,
        to: target.companyId,
        kind: "proposal",
        text: `提议：${termsText(terms)}`,
        terms,
      });
      const targetConfig = configs.find((c) => c.companyId === target.companyId);
      const opportunism = targetConfig ? personaOf(targetConfig).traits.opportunism : 50;
      const accepted = mix(seed, round, index, 11) % 100 >= opportunism;
      messages.push({
        id: `${id}-re`,
        round,
        from: target.companyId,
        to: self.companyId,
        kind: "response",
        text: accepted ? "接受，条件是你方同样履约。" : "拒绝，本轮保留定价自由。",
        status: accepted ? "accepted" : "rejected",
      });
      if (accepted) {
        agreements.push({ id, round, proposer: self.companyId, partner: target.companyId, terms });
      }
    } else {
      messages.push({ id, round, from: self.companyId, to: null, kind: "statement", text: "本轮维持定价，重点关注供应成本。" });
    }
  }
  return { messages, agreements };
}

/**
 * Illustrative inventory flow. Orders come from the demo settlement; procurement
 * is decided before demand is known (forecast × 1.08 − stock) and 15% of leftover
 * fresh stock spoils. It is presentation only and does not feed back into profit.
 */
function planOps(
  agents: AgentRuntimeView[],
  next: AgentRuntimeView[],
  round: number,
  inventory: Record<string, number>,
): OpsRecord[] {
  const n = agents.length;
  const baseDemand = 12_000;
  const demand = demoDemand(round);
  const forecastDemand = round === 1 ? baseDemand : demoDemand(round - 1);
  return agents.map((agent, index) => {
    const after = next[index];
    const capacity = Math.round((baseDemand / n) * 1.35 + after.action.capacity / 1000);
    const stockStart = inventory[agent.companyId] ?? 0;
    const forecast = (forecastDemand * agent.share) / 100;
    const procured = Math.max(0, Math.min(capacity, Math.round(forecast * 1.08 - stockStart)));
    const available = stockStart + procured;
    const orders = Math.round((demand * after.share) / 100);
    const sold = Math.min(orders, available);
    const leftover = available - sold;
    const spoiled = Math.round(leftover * 0.15);
    return {
      companyId: agent.companyId,
      capacity,
      stockStart,
      procured,
      orders,
      sold,
      shortage: orders - sold,
      spoiled,
      stockEnd: leftover - spoiled,
      unitCost: [6200, 6000, 6500, 6300][index % 4] + Math.floor(index / 4) * 40,
      price: after.price,
      advertising: after.action.advertising,
      service: after.action.service,
      capacityInvest: after.action.capacity,
      resilience: after.action.resilience,
      contribution: after.action.contribution,
    };
  });
}

type PersonaTraits = (typeof PERSONAS)[keyof typeof PERSONAS];

/** What this persona would do left alone, reacting to its belief about the rival. */
function instinct(
  persona: PersonaTraits,
  agent: AgentRuntimeView,
  belief: BeliefSummary,
  avg: number,
  roll: number,
  costCents: number,
): { price: number; reasons: string[] } {
  const w = persona.weights;
  const t = persona.traits;
  // Each driver records the direction it pushes the price (−1 down, +1 up, 0 neutral).
  const drivers: Array<{ text: string; dir: number }> = [];
  // Growth/opportunistic personas push below the market, profit/risk personas above.
  const stance = (w.profit + w.risk + w.cash - w.growth * 1.6 - t.opportunism * 0.5) / 100;
  let target = avg + stance * 420;
  if (belief.move === "cut") {
    const response = t.opportunism >= 55 ? -260 : t.riskAversion >= 55 ? -40 : t.cooperation >= 50 ? 30 : -140;
    target += response * ((belief.probability ?? 50) / 60);
    drivers.push({ text: response <= -140 ? "预期对手降价、抢先跟进" : response >= 0 ? "预期对手降价、不跟进价格战" : "预期对手降价、小幅防守", dir: Math.sign(response) });
  } else if (belief.move === "raise") {
    const response = t.opportunism >= 55 ? -120 : 120;
    target += response * ((belief.probability ?? 50) / 60);
    drivers.push({ text: response < 0 ? "预期对手提价、压价抢量" : "预期对手提价、同步提价", dir: Math.sign(response) });
  }
  if (agent.shareDelta <= -1.5 && w.growth >= 20) {
    target -= 160;
    drivers.push({ text: "份额下滑、看重增长", dir: -1 });
  }
  drivers.push(
    stance > 0.15 ? { text: "人设偏利润/稳健", dir: 1 } : stance < -0.15 ? { text: "人设偏增长/激进", dir: -1 } : { text: "人设均衡", dir: 0 },
  );
  // Inertia: risk-averse personas move slowly from last round's price.
  const inertia = 0.25 + t.riskAversion / 200;
  target = agent.price * inertia + target * (1 - inertia);
  target += ((roll % 9) - 4) * 45; // idiosyncratic noise, deterministic by seed
  const floor = costCents + 2200;
  const price = Math.max(floor, Math.min(12_000, Math.round(target / 50) * 50));
  const move = price < agent.price ? `倾向降价 ¥${yuan(agent.price - price)}` : price > agent.price ? `倾向提价 ¥${yuan(price - agent.price)}` : "倾向维持";
  const dir = Math.sign(price - agent.price);
  const shown = drivers.filter((d) => dir === 0 || d.dir === dir || d.dir === 0).map((d) => d.text);
  return { price, reasons: [`${move}（${(shown.length ? shown : ["惯性与随机扰动"]).join("，")}）`] };
}

function decide(
  config: AgentConfig,
  index: number,
  agent: AgentRuntimeView,
  agents: AgentRuntimeView[],
  belief: BeliefSummary,
  advice: Omit<ScreenAdvice, "adopted">,
  agreements: Agreement[],
  trackRecord: number,
  round: number,
  seed: number,
): { action: Partial<AgentRuntimeView["action"]>; trace: DecisionTrace } {
  const persona = PERSONAS[config.persona];
  const t = persona.traits;
  const avg = averagePrice(agents);
  const roll = mix(seed, round, index, 101);
  const { price: instinctPrice, reasons } = instinct(persona, agent, belief, avg, roll, demoUnitCost(index));

  // Trust in the advisor: persona base, bent by how well advice has paid off so far.
  const base = 30 + t.riskAversion * 0.35 + t.cooperation * 0.2 - t.opportunism * 0.3 + (config.gameTheory ? 15 : 0);
  const trust = Math.round(Math.max(8, Math.min(92, base + trackRecord * 18)));
  const followed = round > 1 && roll % 100 < trust;

  let price = instinctPrice;
  let contribution = agent.action.contribution;
  if (followed) {
    if (advice.phenomenon === "搭便车") contribution = Math.max(200_000, contribution);
    else price = advice.recommendedPrice;
    reasons.push(`采纳建议（信任度 ${trust}%）`);
  } else if (round > 1) {
    reasons.push(`未采纳建议（信任度 ${trust}%），按自身倾向`);
  }

  // Agreements: honour or defect depending on opportunism.
  const mine = agreements.filter((ag) => ag.proposer === agent.companyId || ag.partner === agent.companyId);
  let floor: number | null = null;
  let honoured: boolean | null = null;
  if (mine.length) {
    floor = Math.max(...mine.map((ag) => ag.terms.minPrice));
    const minContribution = Math.max(...mine.map((ag) => ag.terms.minContribution));
    honoured = mix(seed, round, index, 211) % 100 >= t.opportunism * 0.8;
    if (honoured) {
      price = Math.max(price, floor);
      contribution = Math.max(contribution, minContribution);
      reasons.push(`遵守协议：价格 ≥ ¥${yuan(floor)}`);
    } else {
      price = Math.min(price, floor - 150);
      contribution = 0;
      reasons.push("背离协议，暗中压价并停止共建");
    }
  }

  // Free-riding temptation for opportunists when not bound by an agreement.
  if (!mine.length && !(followed && advice.phenomenon === "搭便车") && t.opportunism >= 70 && roll % 3 === 0) {
    contribution = 0;
    reasons.push("搭便车：暂停公共共建");
  } else if (contribution === 0 && t.cooperation >= 40) {
    contribution = 200_000;
    reasons.push("合作倾向：恢复公共共建");
  }

  // Put the factor that actually fixed the price first; it is what the screen shows beside the agent.
  const decisive = reasons.findIndex((r) => r.startsWith("遵守协议") || r.startsWith("背离协议"));
  const followIdx = reasons.findIndex((r) => r.startsWith("采纳建议"));
  const lead = decisive >= 0 ? decisive : followed && advice.phenomenon !== "搭便车" ? followIdx : -1;
  if (lead > 0) reasons.unshift(...reasons.splice(lead, 1));
  const advertising = Math.max(
    0,
    Math.round((150_000 + persona.weights.growth * 9_000 + (agent.shareDelta < 0 ? 100_000 : 0) + ((roll >> 3) % 3) * 50_000) / 50_000) * 50_000,
  );
  const service = Math.max(0, Math.round((persona.weights.risk * 6_000 + persona.weights.social * 4_000) / 50_000) * 50_000);
  return {
    action: { price, contribution, advertising, service },
    trace: { companyId: agent.companyId, instinctPrice, trust, followed, floor, honoured, reasons },
  };
}

export function planRound(
  configs: AgentConfig[],
  agents: AgentRuntimeView[],
  round: number,
  seed: number,
  episode: number,
  inventory: Record<string, number> = {},
  priceHistory: Record<string, number[]> = {},
  trackRecord: Record<string, number> = {},
): RoundPlan {
  const { messages, agreements } = generateMessages(configs, agents, round, seed);
  const beliefs = agents.map((agent, i) => summarizeBelief(configs[i], agent, agents, round, priceHistory));
  const baseAdvice = agents.map((agent) => adviseAgent(agent, agents, round));
  const decided = agents.map((agent, i) =>
    decide(configs[i], i, agent, agents, beliefs[i], baseAdvice[i], agreements, trackRecord[agent.companyId] ?? 0, round, seed),
  );
  const decisions = decided.map((d) => d.trace);
  const joint = Object.fromEntries(agents.map((a, i) => [a.companyId, decided[i].action]));
  const seatA = seatAAction(round, seed, agents[0]);
  const next = advanceDemoRound(agents, round, seatA, joint);
  const advice = baseAdvice.map((a, i) => ({ ...a, adopted: decisions[i].followed }));
  const impacts: AdviceImpact[] = agents.map((agent, index) => {
    const a = advice[index];
    const actual = next[index];
    const override = a.adopted
      ? { price: agent.price, contribution: agent.action.contribution }
      : {
          price: a.recommendedPrice,
          contribution: a.phenomenon === "搭便车" ? Math.max(200_000, actual.action.contribution) : actual.action.contribution,
        };
    const alt = advanceDemoRound(agents, round, seatA, { ...joint, [agent.companyId]: { ...joint[agent.companyId], ...override } })[index];
    const [adoptWorld, otherWorld] = a.adopted ? [actual, alt] : [alt, actual];
    return {
      companyId: agent.companyId,
      adopted: a.adopted,
      comparedWith: a.adopted ? "keep" : "follow",
      profitDelta: adoptWorld.profit - otherWorld.profit,
      shareDelta: Math.round((adoptWorld.share - otherWorld.share) * 10) / 10,
      resilienceDelta: Math.round((adoptWorld.resilience - otherWorld.resilience) * 10) / 10,
    };
  });
  const actions = agents.map((before, index) => ({
    companyId: before.companyId,
    price: next[index].price,
    priceDelta: next[index].price - before.price,
    advertising: next[index].action.advertising,
    contribution: next[index].action.contribution,
    label: actionLabel(before, next[index]),
  }));
  const ops = planOps(agents, next, round, inventory);

  const feed: FeedItem[] = [];
  const push = (item: Omit<FeedItem, "id" | "episode" | "round">) =>
    feed.push({ ...item, id: `e${episode}-r${round}-${feed.length}`, episode, round });
  push({ phase: "observe", companyId: null, verb: "开局", text: `第 ${round} 回合：${agents.length} 家企业接收公开市场信息`, tone: "info" });
  const verbs: Record<MessageKind, string> = { proposal: "提议", response: "回应", threat: "威胁", signal: "信号", statement: "声明" };
  for (const m of messages) {
    push({
      phase: "communicate",
      companyId: m.from,
      verb: m.kind === "response" ? (m.status === "accepted" ? "接受" : "拒绝") : verbs[m.kind],
      text: `→ ${agentName(agents, m.to)}：${m.text}`,
      tone: m.kind === "threat" ? "alert" : "talk",
    });
  }
  for (const b of beliefs) {
    if (b.targetId) push({ phase: "belief", companyId: b.companyId, verb: "判断", text: b.text, tone: "info" });
  }
  if (beliefs.every((b) => !b.targetId)) {
    push({ phase: "belief", companyId: null, verb: "判断", text: "首轮无公开历史，所有对手判断保持未知", tone: "info" });
  }
  for (const a of advice) {
    push({
      phase: "advise",
      companyId: a.companyId,
      verb: a.phenomenon,
      text: a.headline,
      tone: a.phenomenon === "价格战" || a.phenomenon === "搭便车" ? "alert" : "advice",
    });
  }
  actions.forEach((a, i) =>
    push({ phase: "act", companyId: a.companyId, verb: "提交", text: `${a.label}｜${decisions[i].reasons[0] ?? "按自身倾向"}`, tone: "action" }),
  );
  const leader = [...next].sort((a, b) => b.share - a.share)[0];
  push({
    phase: "settle",
    companyId: leader.companyId,
    verb: "结算",
    text: `领跑份额 ${leader.share.toFixed(1)}% · 均价 ¥${yuan(averagePrice(next))} · HHI ${hhi(next)}`,
    tone: "result",
  });
  return { round, messages, agreements, beliefs, advice, impacts, decisions, actions, ops, next, feed };
}

export function createScreen(count: number, episode = 1): ScreenState {
  const size = Math.max(SCREEN_MIN_AGENTS, Math.min(SCREEN_MAX_AGENTS, Math.round(count)));
  const configs = AGENT_TEMPLATES.slice(0, size);
  const agents = buildRuntimeAgents(configs);
  const seed = mix(episode, size) % 997;
  const inventory = Object.fromEntries(agents.map((a) => [a.companyId, 0]));
  const priceHistory = Object.fromEntries(agents.map((a) => [a.companyId, [a.price]]));
  const plan = planRound(configs, agents, 1, seed, episode, inventory, priceHistory, {});
  return {
    episode,
    seed,
    configs,
    agents,
    round: 1,
    history: [historyPoint(0, agents)],
    adviceHistory: [],
    agreements: [],
    links: {},
    inventory,
    lastOps: [],
    priceHistory,
    trackRecord: {},
    totals: { messages: 0, adviceTotal: 0, adviceAdopted: 0, kept: 0, broken: 0, spoiled: 0, shortage: 0, adoptedGain: 0, missedGain: 0, helpful: 0 },
    plan,
    lastPlan: null,
    phase: 0,
    feed: plan.feed.filter((item) => item.phase === "observe"),
    seq: 0,
  };
}

function addLinks(links: ScreenState["links"], messages: ScreenMessage[]) {
  const next: ScreenState["links"] = Object.fromEntries(Object.entries(links).map(([k, v]) => [k, { ...v }]));
  for (const m of messages) {
    const row = (next[m.from] ??= {});
    const key = m.to ?? "ALL";
    row[key] = (row[key] ?? 0) + 1;
  }
  return next;
}

function settle(state: ScreenState): ScreenState {
  const { plan } = state;
  const actionOf = new Map(plan.next.map((a) => [a.companyId, a]));
  let kept = 0;
  let broken = 0;
  const agreements = state.agreements.map((ag) => {
    if (ag.outcome || ag.round !== plan.round) return ag;
    const actual: NonNullable<Agreement["actual"]> = {};
    for (const id of [ag.proposer, ag.partner]) {
      const after = actionOf.get(id);
      const price = after?.price ?? 0;
      const contribution = after?.action.contribution ?? 0;
      const ok = price >= ag.terms.minPrice && contribution >= ag.terms.minContribution;
      if (ok) kept += 1;
      else broken += 1;
      actual[id] = { price, contribution, kept: ok };
    }
    const keptCount = Object.values(actual).filter((x) => x.kept).length;
    const outcome = keptCount === 2 ? "kept" : keptCount === 0 ? "broken" : "partial";
    return { ...ag, actual, outcome } as Agreement;
  });
  const inventory = Object.fromEntries(plan.ops.map((o) => [o.companyId, o.stockEnd]));
  return {
    ...state,
    agents: plan.next,
    inventory,
    lastOps: plan.ops,
    priceHistory: Object.fromEntries(
      plan.next.map((a) => [a.companyId, [...(state.priceHistory[a.companyId] ?? []), a.price].slice(-6)]),
    ),
    trackRecord: Object.fromEntries(
      plan.impacts.map((imp) => {
        const prevScore = state.trackRecord[imp.companyId] ?? 0;
        // Agents learn from the outcome they observed: adopted advice paid off, or rejected advice would have.
        const signal = Math.max(-1, Math.min(1, imp.profitDelta / 150_000));
        return [imp.companyId, Math.round((prevScore * 0.6 + signal * 0.4) * 100) / 100];
      }),
    ),
    history: [...state.history, historyPoint(plan.round, plan.next)].slice(-SCREEN_ROUNDS - 1),
    adviceHistory: [
      ...state.adviceHistory,
      { round: plan.round, items: plan.advice.map(({ companyId, phenomenon, adopted }) => ({ companyId, phenomenon, adopted })) },
    ].slice(-SCREEN_ROUNDS),
    agreements: agreements.slice(-60),
    totals: {
      ...state.totals,
      adviceTotal: state.totals.adviceTotal + plan.advice.length,
      adviceAdopted: state.totals.adviceAdopted + plan.advice.filter((a) => a.adopted).length,
      kept: state.totals.kept + kept,
      broken: state.totals.broken + broken,
      spoiled: state.totals.spoiled + plan.ops.reduce((s, o) => s + o.spoiled, 0),
      shortage: state.totals.shortage + plan.ops.reduce((s, o) => s + o.shortage, 0),
      adoptedGain: state.totals.adoptedGain + plan.impacts.filter((i) => i.adopted).reduce((s, i) => s + i.profitDelta, 0),
      missedGain: state.totals.missedGain + plan.impacts.filter((i) => !i.adopted).reduce((s, i) => s + i.profitDelta, 0),
      helpful: state.totals.helpful + plan.impacts.filter((i) => i.profitDelta >= 0).length,
    },
  };
}

function reveal(state: ScreenState, phase: number): FeedItem[] {
  const id = PHASES[phase].id;
  const items = state.plan.feed.filter((item) => item.phase === id);
  return [...items.reverse(), ...state.feed].slice(0, FEED_LIMIT);
}

/** Advance one phase. Settlement commits numbers; after it a new round (or episode) begins. */
export function tick(state: ScreenState): ScreenState {
  const seq = state.seq + 1;
  if (state.phase < PHASES.length - 1) {
    const phase = state.phase + 1;
    const base = { ...state, phase, seq, feed: reveal(state, phase) };
    const id = PHASES[phase].id;
    if (id === "settle") return settle(base);
    if (id === "communicate") {
      return {
        ...base,
        agreements: [...state.agreements, ...state.plan.agreements],
        links: addLinks(state.links, state.plan.messages),
        totals: { ...state.totals, messages: state.totals.messages + state.plan.messages.length },
      };
    }
    return base;
  }
  if (state.round >= SCREEN_ROUNDS) {
    const fresh = createScreen(state.configs.length, state.episode + 1);
    const banner: FeedItem = {
      id: `e${fresh.episode}-start`,
      episode: fresh.episode,
      round: 1,
      phase: "observe",
      companyId: null,
      verb: "新局",
      text: `第 ${state.episode} 局 ${SCREEN_ROUNDS} 回合完成，开启第 ${fresh.episode} 局（回合不循环）`,
      tone: "result",
    };
    return { ...fresh, seq, feed: [...fresh.feed, banner, ...state.feed].slice(0, FEED_LIMIT) };
  }
  const round = state.round + 1;
  const plan = planRound(state.configs, state.agents, round, state.seed, state.episode, state.inventory, state.priceHistory, state.trackRecord);
  const next = { ...state, round, plan, lastPlan: state.plan, phase: 0, seq };
  return { ...next, feed: reveal(next, 0) };
}

export function screenReducer(state: ScreenState, action: ScreenAction): ScreenState {
  if (action.type === "reset") return createScreen(action.count, state.episode + 1);
  return tick(state);
}

export function phaseId(state: ScreenState): PhaseId {
  return PHASES[state.phase].id;
}
