export type ViewId =
  | "home"
  | "setup"
  | "live"
  | "observatory"
  | "communication"
  | "strategy"
  | "market"
  | "replay"
  | "report";

export type PersonaKey =
  | "balanced_v1"
  | "aggressive_v1_extreme"
  | "risk_guarded_v1"
  | "selfish_long_term_v1"
  | "cooperator"
  | "free_rider"
  | "retaliator";

export type PersonaProfile = {
  key: PersonaKey;
  label: string;
  summary: string;
  weights: {
    profit: number;
    growth: number;
    risk: number;
    cash: number;
    social: number;
  };
  traits: {
    timeDiscount: number;
    riskAversion: number;
    cooperation: number;
    opportunism: number;
  };
};

export type AgentConfig = {
  companyId: string;
  shortName: string;
  companyName: string;
  color: string;
  driver: "human" | "doubao" | "deepseek" | "rule";
  model: string;
  persona: PersonaKey;
  information: "full" | "public" | "imperfect";
  communication: boolean;
  gameTheory: boolean;
};

export type LabConfig = {
  informationMode: "perfect" | "public" | "imperfect";
  marketType: "balanced" | "high_demand" | "supply_crisis" | "disaster" | "public_goods";
  rounds: 5 | 10 | 15 | 20;
  seed: number;
  communication: boolean;
  cooperation: boolean;
  gameTheory: boolean;
  controllerToken: string;
};

export type Belief = {
  companyId: string;
  strategy: { growth: number; profit: number; defensive: number };
  nextAction: { cut: number; maintain: number; raise: number };
  evidence: string[];
};

export type AgentRuntimeView = {
  companyId: string;
  companyName: string;
  color: string;
  persona: string;
  driver: string;
  cash: number;
  profit: number;
  share: number;
  shareDelta: number;
  price: number;
  resilience: number;
  observationHash: string;
  observation: {
    public: Array<{ label: string; value: string }>;
    private: Array<{ label: string; value: string }>;
    hidden: Array<{ label: string; value: string }>;
  };
  beliefs: Belief[];
  plan: {
    goal: string;
    horizon: number;
    subgoals: Array<{ label: string; status: "done" | "active" | "queued" }>;
    triggers: string[];
  };
  decision: {
    situation: string;
    factors: string[];
    summary: string;
    expected: string;
  };
  action: {
    price: number;
    advertising: number;
    service: number;
    capacity: number;
    resilience: number;
    contribution: number;
  };
  advisor: {
    recommendedPrice: number;
    adopted: boolean;
    candidates: Array<{ action: string; utility: number; risk: number }>;
  };
  utility: { profit: number; growth: number; risk: number };
};

export type CommunicationRecord = {
  id: string;
  round: number;
  sender: string;
  recipient: string | null;
  channel: "public" | "private";
  kind: "statement" | "proposal" | "commitment" | "threat" | "signal" | "response";
  text: string;
  visibility: string[];
  status?: "accepted" | "rejected" | "partial_betrayal";
};

export type ReplayStep = {
  round: number;
  phase: string;
  agent: string;
  title: string;
  detail: string;
  hash?: string;
  tone: "neutral" | "signal" | "action" | "result";
};

export const PERSONAS: Record<PersonaKey, PersonaProfile> = {
  balanced_v1: {
    key: "balanced_v1",
    label: "均衡经营",
    summary: "平衡利润、增长、现金和风险的长期基线。",
    weights: { profit: 30, growth: 22, risk: 18, cash: 18, social: 12 },
    traits: { timeDiscount: 92, riskAversion: 45, cooperation: 30, opportunism: 40 },
  },
  aggressive_v1_extreme: {
    key: "aggressive_v1_extreme",
    label: "激进增长",
    summary: "以份额与扩张为主，容忍价格战和现金波动。",
    weights: { profit: 24, growth: 43, risk: 8, cash: 8, social: 17 },
    traits: { timeDiscount: 85, riskAversion: 15, cooperation: 10, opportunism: 70 },
  },
  risk_guarded_v1: {
    key: "risk_guarded_v1",
    label: "风险防御",
    summary: "优先现金安全、韧性和风险调整后的长期价值。",
    weights: { profit: 27, growth: 10, risk: 31, cash: 22, social: 10 },
    traits: { timeDiscount: 96, riskAversion: 70, cooperation: 25, opportunism: 20 },
  },
  selfish_long_term_v1: {
    key: "selfish_long_term_v1",
    label: "长期自利",
    summary: "只在合作能改善自身长期回报时参与合作。",
    weights: { profit: 38, growth: 13, risk: 13, cash: 28, social: 8 },
    traits: { timeDiscount: 95, riskAversion: 35, cooperation: 10, opportunism: 60 },
  },
  cooperator: {
    key: "cooperator",
    label: "合作共赢",
    summary: "重视公共韧性、互惠和可持续的共同收益。",
    weights: { profit: 23, growth: 14, risk: 20, cash: 13, social: 30 },
    traits: { timeDiscount: 96, riskAversion: 42, cooperation: 88, opportunism: 10 },
  },
  free_rider: {
    key: "free_rider",
    label: "搭便车者",
    summary: "偏好享受公共收益，同时尽量避免私人贡献成本。",
    weights: { profit: 42, growth: 18, risk: 13, cash: 22, social: 5 },
    traits: { timeDiscount: 72, riskAversion: 25, cooperation: 8, opportunism: 88 },
  },
  retaliator: {
    key: "retaliator",
    label: "报复者",
    summary: "初始愿意合作，发现背叛后迅速转入对等反制。",
    weights: { profit: 26, growth: 13, risk: 21, cash: 14, social: 26 },
    traits: { timeDiscount: 91, riskAversion: 44, cooperation: 70, opportunism: 35 },
  },
};

export const DEFAULT_AGENTS: AgentConfig[] = [
  { companyId: "company_A", shortName: "A", companyName: "青禾速配", color: "#4ee0bd", driver: "human", model: "Human", persona: "balanced_v1", information: "public", communication: true, gameTheory: false },
  { companyId: "company_B", shortName: "B", companyName: "橙选到家", color: "#ff8468", driver: "doubao", model: "Doubao Seed 2.0 Lite", persona: "aggressive_v1_extreme", information: "public", communication: true, gameTheory: true },
  { companyId: "company_C", shortName: "C", companyName: "蓝仓鲜送", color: "#7196ff", driver: "deepseek", model: "DeepSeek V3", persona: "selfish_long_term_v1", information: "public", communication: true, gameTheory: true },
  { companyId: "company_D", shortName: "D", companyName: "紫藤优鲜", color: "#b38cff", driver: "rule", model: "Deterministic Rule", persona: "risk_guarded_v1", information: "public", communication: false, gameTheory: false },
];

const commonPublic = [
  { label: "实现需求", value: "12,480 单" },
  { label: "市场成交均价", value: "¥98.20" },
  { label: "供应成本指数", value: "1.084" },
  { label: "行业共享韧性", value: "62.4%" },
];

function runtime(agent: AgentConfig, index: number): AgentRuntimeView {
  const prices = [9800, 9600, 10100, 9900];
  const shares = [27.5, 26, 24.5, 22];
  const profits = [0, 0, 0, 0];
  return {
    companyId: agent.companyId,
    companyName: agent.companyName,
    color: agent.color,
    persona: PERSONAS[agent.persona].label,
    driver: agent.model,
    cash: 24_800_000 - index * 1_420_000,
    profit: profits[index],
    share: shares[index],
    shareDelta: 0,
    price: prices[index],
    resilience: 68 - index * 4,
    observationHash: `sha256:${["8e31b4d9", "112fa4c0", "7232ec18", "c9e04bd1"][index]}…`,
    observation: {
      public: commonPublic,
      private: [
        { label: "本公司现金", value: `¥${((24_800_000 - index * 1_420_000) / 10000).toFixed(0)}k` },
        { label: "有效产能", value: `${4_200 - index * 180} 单` },
        { label: "单位履约成本", value: `¥${(62 + index * 1.7).toFixed(1)}` },
      ],
      hidden: [
        { label: "对手现金", value: "Hidden" },
        { label: "对手真实成本", value: "Hidden" },
        { label: "对手 Persona", value: "Belief only" },
      ],
    },
    beliefs: DEFAULT_AGENTS.filter((item) => item.companyId !== agent.companyId).map((opponent) => ({
      companyId: opponent.companyId,
      strategy: { growth: 33, profit: 34, defensive: 33 },
      nextAction: { cut: 33, maintain: 34, raise: 33 },
      evidence: [],
    })),
    plan: {
      goal: index === 1 ? "在不触发现金警戒线的前提下扩大份额" : "提高风险调整后的长期企业价值",
      horizon: 3,
      subgoals: [
        { label: "校准价格与边际利润", status: "done" },
        { label: "观察竞争者价格响应", status: "active" },
        { label: "根据需求更新投入", status: "queued" },
      ],
      triggers: ["现金低于 ¥10M", "竞争者连续两轮降价", "灾难概率超过 35%"],
    },
    decision: {
      situation: index === 1 ? "竞争者维持价格，当前产能利用率处于高位。" : "市场需求稳定，但下一轮存在供应冲击信号。",
      factors: ["单位利润空间", "竞争价格响应", "未来事故风险"],
      summary: index === 1 ? "降低价格并暂缓非关键投入。" : "维持价格，提高服务与韧性预算。",
      expected: index === 1 ? "短期份额提高，利润与韧性承压。" : "短期利润略降，长期风险损失下降。",
    },
    action: { price: prices[index], advertising: index === 1 ? 500_000 : 200_000, service: index === 1 ? 0 : 300_000, capacity: index === 2 ? 600_000 : 0, resilience: index === 1 ? 0 : 250_000, contribution: index === 3 ? 0 : 200_000 },
    advisor: {
      recommendedPrice: index === 1 ? 8600 : prices[index] - 200,
      adopted: index === 1,
      candidates: [
        { action: "大幅降价", utility: 0.76, risk: 0.68 },
        { action: "维持当前价格", utility: 0.64, risk: 0.31 },
        { action: "提高价格", utility: 0.42, risk: 0.24 },
      ],
    },
    utility: { profit: 42 - index * 3, growth: 38 + index * 2, risk: 20 + index },
  };
}

export const DEMO_AGENTS = DEFAULT_AGENTS.map(runtime);

export type DemoHumanAction = {
  price: number;
  advertising: number;
  contribution: number;
};

/**
 * A deterministic UI-only market step. It is intentionally smaller than
 * MarketEnv and must never be used as research evidence. The important UI
 * invariants are that actions affect outcomes and market shares sum to 100%.
 */
export function advanceDemoRound(
  agents: AgentRuntimeView[],
  settledRound: number,
  humanAction: DemoHumanAction,
): AgentRuntimeView[] {
  const pricePatterns = [
    [0, 0, 0, 0],
    [-250, 180, -120, 260],
    [180, -160, 220, -80],
    [80, 240, -200, 140],
  ];
  const shockPatterns = [
    [1.04, 0.96, 1.01, 0.99],
    [0.97, 1.08, 0.94, 1.01],
    [1.01, 0.93, 1.09, 0.98],
    [0.95, 1.02, 0.98, 1.08],
    [1.07, 0.97, 1.02, 0.94],
  ];
  const pattern = pricePatterns[settledRound % pricePatterns.length];
  const shocks = shockPatterns[(settledRound - 1) % shockPatterns.length];
  const nextActions = agents.map((agent, index) => {
    if (index === 0) {
      return { ...agent.action, price: humanAction.price, advertising: humanAction.advertising, contribution: humanAction.contribution };
    }
    const anchor = [9800, 9500, 10100, 9900][index];
    return {
      ...agent.action,
      price: Math.max(8000, Math.min(12000, anchor + pattern[index])),
      advertising: Math.max(0, agent.action.advertising + ((settledRound + index) % 3 - 1) * 100_000),
      service: Math.max(0, agent.action.service + ((settledRound * index) % 3 - 1) * 100_000),
      contribution: index === 3 && settledRound % 2 === 0 ? 0 : agent.action.contribution,
    };
  });
  const averagePrice = nextActions.reduce((sum, action) => sum + action.price, 0) / nextActions.length;
  const scores = agents.map((agent, index) => {
    const action = nextActions[index];
    const pricePull = Math.pow(averagePrice / action.price, 2.6);
    const investmentPull = 1 + action.advertising / 3_500_000 + action.service / 5_000_000 + action.capacity / 9_000_000;
    const continuity = 0.82 + agent.share / 140;
    return Math.max(0.05, pricePull * investmentPull * continuity * shocks[index]);
  });
  const scoreTotal = scores.reduce((sum, value) => sum + value, 0);
  const roundedShares = scores.map((score) => Math.round((score / scoreTotal) * 1000) / 10);
  roundedShares[roundedShares.length - 1] = Math.round((100 - roundedShares.slice(0, -1).reduce((sum, value) => sum + value, 0)) * 10) / 10;
  const realizedDemand = 11_600 + ((settledRound * 811) % 1_900);
  const unitCosts = [6200, 6000, 6500, 6300];

  return agents.map((agent, index) => {
    const action = nextActions[index];
    const share = roundedShares[index];
    const shareDelta = Math.round((share - agent.share) * 10) / 10;
    const fulfilledOrders = realizedDemand * share / 100;
    const operatingProfit = Math.round(fulfilledOrders * (action.price - unitCosts[index]));
    const investmentCost = action.advertising + action.service + action.capacity + action.resilience + action.contribution;
    const profit = operatingProfit - investmentCost;
    const cash = agent.cash + profit;
    const resilience = Math.max(20, Math.min(95, agent.resilience + action.contribution / 250_000 - 0.35));
    const priceDirection = action.price < agent.price ? "降价" : action.price > agent.price ? "提价" : "维持价格";
    const beliefs = agent.beliefs.map((belief) => {
      const opponentIndex = agents.findIndex((item) => item.companyId === belief.companyId);
      const opponentBefore = agents[opponentIndex];
      const opponentAction = nextActions[opponentIndex];
      const direction = opponentAction.price < opponentBefore.price ? "cut" : opponentAction.price > opponentBefore.price ? "raise" : "maintain";
      const nextAction = direction === "cut" ? { cut: 58, maintain: 28, raise: 14 } : direction === "raise" ? { cut: 16, maintain: 29, raise: 55 } : { cut: 24, maintain: 55, raise: 21 };
      const opponentShareDelta = Math.round((roundedShares[opponentIndex] - opponentBefore.share) * 10) / 10;
      const growth = Math.max(18, Math.min(65, 36 + Math.round(opponentAction.advertising / 100_000) + Math.max(0, Math.round(opponentShareDelta * 2))));
      const defensive = Math.max(15, Math.min(48, 32 + Math.round(opponentAction.service / 150_000) - Math.max(0, Math.round(opponentShareDelta))));
      return {
        ...belief,
        strategy: { growth, profit: Math.max(10, 100 - growth - defensive), defensive },
        nextAction,
        evidence: [
          `第 ${settledRound} 回合公开价格：${(opponentBefore.price / 100).toFixed(0)} 元 → ${(opponentAction.price / 100).toFixed(0)} 元`,
          `第 ${settledRound} 回合公开份额变化：${opponentShareDelta >= 0 ? "+" : ""}${opponentShareDelta.toFixed(1)} 个百分点`,
        ],
      };
    });
    return {
      ...agent,
      cash,
      profit,
      share,
      shareDelta,
      price: action.price,
      resilience,
      action,
      beliefs,
      observationHash: `demo:r${settledRound + 1}:${agent.companyId.slice(-1)}:${Math.round(share * 10)}`,
      observation: {
        ...agent.observation,
        public: [
          { label: "实现需求", value: `${realizedDemand.toLocaleString("zh-CN")} 单` },
          { label: "市场成交均价", value: `¥${(averagePrice / 100).toFixed(2)}` },
          { label: "市场份额总和", value: "100.0%" },
          { label: "行业共享韧性", value: `${(agents.reduce((sum, item) => sum + item.resilience, 0) / agents.length).toFixed(1)}%` },
        ],
        private: [
          { label: "本公司现金", value: `¥${(cash / 100).toLocaleString("zh-CN", { maximumFractionDigits: 0 })}` },
          { label: "本轮利润", value: `¥${(profit / 100).toLocaleString("zh-CN", { maximumFractionDigits: 0 })}` },
          { label: "份额变化", value: `${shareDelta >= 0 ? "+" : ""}${shareDelta.toFixed(1)}pp` },
        ],
      },
      decision: {
        ...agent.decision,
        situation: `第 ${settledRound} 回合需求 ${realizedDemand.toLocaleString("zh-CN")} 单，市场份额发生重新分配。`,
        summary: `${priceDirection}，并按 Persona 调整投入。`,
        expected: `份额 ${shareDelta >= 0 ? "增加" : "减少"} ${Math.abs(shareDelta).toFixed(1)}pp；本轮利润 ${profit >= 0 ? "为正" : "为负"}。`,
      },
    };
  });
}

export const DEMO_MESSAGES: CommunicationRecord[] = [
  { id: "msg-r5-a-1", round: 5, sender: "company_A", recipient: null, channel: "public", kind: "statement", text: "我们计划维持稳定定价，并提高行业韧性投入。", visibility: ["A", "B", "C", "D"] },
  { id: "msg-r5-b-1", round: 5, sender: "company_B", recipient: "company_C", channel: "private", kind: "proposal", text: "建议双方下一轮各贡献 ¥10,000 用于共享韧性。", visibility: ["B", "C"], status: "accepted" },
  { id: "msg-r5-c-1", round: 5, sender: "company_C", recipient: "company_B", channel: "private", kind: "response", text: "接受，但仅在你保持贡献的条件下履约。", visibility: ["B", "C"], status: "accepted" },
  { id: "msg-r4-d-1", round: 4, sender: "company_D", recipient: null, channel: "public", kind: "signal", text: "当前现金足以支持扩产。该声明未经验证。", visibility: ["A", "B", "C", "D"] },
  { id: "msg-r3-b-1", round: 3, sender: "company_B", recipient: "company_A", channel: "private", kind: "commitment", text: "承诺本轮贡献 ¥10,000。", visibility: ["A", "B"], status: "partial_betrayal" },
];

export const DEMO_REPLAY: ReplayStep[] = [
  { round: 5, phase: "真实市场状态", agent: "系统", title: "市场状态冻结", detail: "需求 12,480 · 供应成本指数 1.084 · 状态版本 4", hash: "sha256:41bd9e…", tone: "neutral" },
  { round: 5, phase: "可见信息", agent: "A", title: "生成公司范围信息", detail: "4 个公共字段 · 3 个自己的私有字段 · 对手私有字段已隐藏", hash: "sha256:8e31b4…", tone: "signal" },
  { round: 5, phase: "对手判断", agent: "A", title: "更新动作概率", detail: "B 降价可能性 42% → 64% · 3 条公开依据", hash: "belief:920fd1…", tone: "signal" },
  { round: 5, phase: "通信", agent: "B → C", title: "私密合作提议", detail: "只有 B 和 C 可见 · 不具有强制约束力 · 共享抗冲击投入", hash: "view:6a110c…", tone: "signal" },
  { round: 5, phase: "策略建议", agent: "B", title: "近似最佳回应", detail: "建议价格 86 元 · 预期效用评分 0.76", tone: "action" },
  { round: 5, phase: "最终动作", agent: "B", title: "智能体采纳建议", detail: "价格 86 元 · 服务投入 0 元 · 抗冲击投入 0 元", hash: "action:f106e2…", tone: "action" },
  { round: 5, phase: "市场结果", agent: "系统", title: "联合动作结算", detail: "B 份额增加 4.1 个百分点 · B 利润减少 12,400 元 · 重建一致", hash: "sha256:ffe08a…", tone: "result" },
];

export const PROFIT_SERIES = [
  { round: 1, A: 1.4, B: 1.8, C: 1.2, D: 1.5 },
  { round: 2, A: 1.7, B: 1.5, C: 1.9, D: 1.6 },
  { round: 3, A: 2.0, B: 1.3, C: 2.2, D: 1.7 },
  { round: 4, A: 2.1, B: 1.0, C: 2.5, D: 1.8 },
  { round: 5, A: 2.18, B: 1.24, C: 2.65, D: 1.78 },
];
