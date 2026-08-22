// 真实 Episode 的唯一前端投影；禁止从 DEMO 常量补齐缺失字段。
import { PERSONAS } from "./lab-model.ts";
import type { AgentConfig, AgentRuntimeView } from "./lab-model.ts";

type RealCompanyState = {
  financial: { cash_balance_cents: number; round_profit_cents: number };
  commercial: { price_cents: number; market_share_ppm: number };
  operations?: { effective_capacity_orders?: number; actual_unit_cost_cents?: number };
  brand?: { service_quality_ppm?: number; reputation_ppm?: number };
  risk: { resilience_ppm: number };
  history?: { last_action?: Record<string, unknown> | null; recent_profit_cents?: number[]; recent_market_share_ppm?: number[] };
};

type RealMarketState = {
  realized_demand_orders?: number;
  average_paid_price_cents?: number;
  actual_supply_cost_index_ppm?: number;
  market_sentiment_ppm?: number;
};

export type RealBackendState = {
  market?: RealMarketState;
  companies: Record<string, RealCompanyState>;
};

type Resolution = { action?: Record<string, number> };

export function buildRealRuntimeAgents({ state, configs, previous, resolutions = {}, observations = {} }: {
  state: RealBackendState;
  configs: AgentConfig[];
  previous?: AgentRuntimeView[];
  resolutions?: Record<string, Resolution>;
  observations?: Record<string, Record<string, unknown>>;
}): AgentRuntimeView[] {
  const companies = Object.values(state.companies);
  const industryResilience = companies.length
    ? companies.reduce((sum, item) => sum + item.risk.resilience_ppm, 0) / companies.length / 10_000
    : 0;
  return configs.flatMap((config) => {
    const company = state.companies[config.companyId];
    if (!company) return [];
    const prior = previous?.find((item) => item.companyId === config.companyId);
    const share = company.commercial.market_share_ppm / 10_000;
    const resolution = resolutions[config.companyId];
    const action = resolution?.action;
    const market = state.market ?? {};
    const operation = company.operations ?? {};
    const observation = observations[config.companyId] ?? {};
    return [{
      companyId: config.companyId,
      companyName: config.companyName,
      color: config.color,
      persona: PERSONAS[config.persona].label,
      driver: config.driver === "human" ? "人类参与者" : config.model,
      dataSource: action ? "controller_settlement" : "backend_initial_state",
      actionAvailable: Boolean(action),
      cash: company.financial.cash_balance_cents,
      profit: company.financial.round_profit_cents,
      share,
      shareDelta: prior ? Math.round((share - prior.share) * 10) / 10 : 0,
      price: company.commercial.price_cents,
      resilience: company.risk.resilience_ppm / 10_000,
      observationHash: typeof observation.observation_hash === "string" ? observation.observation_hash : "",
      observation: {
        public: [
          { label: "实现需求", value: market.realized_demand_orders === undefined ? "后端未返回" : `${market.realized_demand_orders} 单` },
          { label: "市场成交均价", value: market.average_paid_price_cents === undefined ? "后端未返回" : `¥${(market.average_paid_price_cents / 100).toFixed(2)}` },
          { label: "供应成本指数", value: market.actual_supply_cost_index_ppm === undefined ? "后端未返回" : (market.actual_supply_cost_index_ppm / 1_000_000).toFixed(3) },
          { label: "行业平均抗冲击能力", value: `${industryResilience.toFixed(1)}%` },
        ],
        private: [
          { label: "本公司现金", value: `¥${(company.financial.cash_balance_cents / 100).toLocaleString("zh-CN")}` },
          { label: "有效产能", value: operation.effective_capacity_orders === undefined ? "后端未返回" : `${operation.effective_capacity_orders} 单` },
          { label: "单位履约成本", value: operation.actual_unit_cost_cents === undefined ? "后端未返回" : `¥${(operation.actual_unit_cost_cents / 100).toFixed(2)}` },
        ],
        hidden: [],
      },
      beliefs: [],
      plan: { goal: "", horizon: 0, subgoals: [], triggers: [] },
      decision: { situation: "", factors: [], summary: "", expected: "" },
      action: {
        price: action?.price_cents ?? 0,
        advertising: action?.advertising_budget_cents ?? 0,
        service: action?.service_budget_cents ?? 0,
        capacity: action?.capacity_investment_cents ?? 0,
        resilience: action?.resilience_budget_cents ?? 0,
        contribution: action?.shared_resilience_contribution_cents ?? 0,
      },
      advisor: { recommendedPrice: 0, adopted: false, candidates: [] },
      utility: { profit: 0, growth: 0, risk: 0 },
    }];
  });
}
