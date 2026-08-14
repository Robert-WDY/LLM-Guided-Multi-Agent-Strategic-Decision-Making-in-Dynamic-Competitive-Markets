"use client";

import { useEffect, useMemo, useState } from "react";

type Bound = { min: number; max: number };
type Incident = {
  incident_id: string;
  incident_type: string;
  severity: string;
  remaining_rounds: number;
  repair_required_cents: number;
  accumulated_repair_cents: number;
};
type Company = {
  company_id: string;
  persona: string;
  financial: {
    cash_balance_cents: number;
    round_revenue_cents: number;
    round_variable_cost_cents: number;
    round_fixed_spend_cents: number;
    round_incident_cost_cents: number;
    round_operating_cost_cents: number;
    round_profit_cents: number;
    cumulative_profit_cents: number;
    capacity_book_value_cents: number;
  };
  commercial: {
    price_cents: number;
    market_share_ppm: number;
    potential_demand_orders: number;
    sales_orders: number;
    attempted_unfulfilled_orders: number;
    orders_received_from_redistribution: number;
    orders_lost_after_redistribution: number;
  };
  operations: {
    base_capacity_orders: number;
    effective_capacity_orders: number;
    financial_capacity_orders: number;
    capacity_utilization_ppm: number;
    base_unit_cost_cents: number;
    actual_unit_cost_cents: number;
  };
  brand: {
    brand_awareness_ppm: number;
    service_quality_ppm: number;
    reputation_ppm: number;
    last_attempted_unfulfilled_rate_ppm: number;
  };
  risk: { resilience_ppm: number; active_incident: Incident | null };
};
type RiskSignal = {
  signal_id: string;
  event_type: string;
  target_round: number;
  estimated_probability_ppm: number;
  severity: string;
  lead_time_rounds: number;
};
type MarketEvent = {
  event_id: string;
  event_type: string;
  severity: string;
  remaining_rounds: number;
  demand_multiplier_ppm: number;
  supply_cost_multiplier_ppm: number;
  capacity_multiplier_ppm: number;
};
type MarketSnapshot = {
  base_demand_orders: number;
  realized_demand_orders: number;
  no_purchase_orders: number;
  lost_after_stockout_orders: number;
  market_sentiment_ppm: number;
  base_supply_cost_index_ppm: number;
  actual_supply_cost_index_ppm: number;
  average_paid_price_cents: number;
  market_model_id: string;
  market_model_label: string;
  market_model_description: string;
  demand_bias_ppm: number;
  price_anchor_cents: number;
  price_band_cents: number;
};
type MarketState = {
  episode_id: string;
  episode_seed: number;
  round: number;
  rounds_remaining: number;
  state_version: number;
  terminal: boolean;
  max_rounds: number;
  market: MarketSnapshot;
  risk_signals: RiskSignal[];
  active_market_events: MarketEvent[];
  company_order: string[];
  companies: Record<string, Company>;
  terminal_enterprise_values_cents: Record<string, number>;
  state_hash: string;
};
type ActionConstraints = {
  schema_version: string;
  cash_available_cents: number;
  bounds: Record<string, Bound>;
  capacity_investment_enabled: boolean;
  resilience_investment_enabled: boolean;
  active_incident: Incident | null;
  max_useful_repair_budget_cents: number;
  mandatory_operating_costs: {
    fixed_overhead_cents: number;
    fulfillment_cost_per_order_cents: number;
    description: string;
  };
};
type EpisodePayload = {
  state: MarketState;
  action_constraints: Record<string, ActionConstraints>;
  action_presets: Record<string, Record<string, number>>;
  game_mode: GameMode;
  player_company_id: string | null;
  market_model_options: Record<string, { label: string; description: string }>;
  episode_options: {
    round_options: number[];
    default_rounds: number;
    seed: { min: number; max: number; random_supported: boolean; fixed_supported: boolean; note: string };
    market_models: Record<string, { label: string; description: string }>;
  };
  company_analysis?: CompanyAnalysis;
};
type GameMode = "market" | "single_company";
type SeedMode = "random" | "fixed";
type MarketModel = "random" | "balanced" | "value_oriented" | "quality_oriented" | "service_oriented";
type AnalysisFactor = {
  key: string;
  label: string;
  value_ppm: number;
  status: "healthy" | "watch" | "risk";
  summary: string;
};
type Recommendation = {
  priority: "critical" | "high" | "medium" | "normal";
  dimension: string;
  title: string;
  rationale: string;
};
type CompanyAnalysis = {
  company_id: string;
  round: number;
  health_score: number;
  health_label: string;
  factors: AnalysisFactor[];
  recommendations: Recommendation[];
  decision_context: {
    margin_per_order_cents: number;
    fulfillment_cost_per_order_cents: number;
    relative_price_cents: number;
    capacity_buffer_orders: number;
    rounds_remaining: number;
  };
};
type RetrospectiveRound = {
  round: number;
  verdict: string;
  action: Record<string, number | string | Record<string, number | string>>;
  profit_cents: number;
  operating_cost_cents: number;
  cash_cents: number;
  enterprise_value_cents: number;
  enterprise_value_delta_cents: number;
  market_share_ppm: number;
  share_delta_ppm: number;
  sales_orders: number;
  effective_capacity_orders: number;
  stockout_orders: number;
  market_demand_orders: number;
  market_demand_delta_orders: number;
  supply_cost_index_ppm: number;
  supply_cost_delta_ppm: number;
  outside_option_orders: number;
  average_paid_price_cents: number;
  awareness_ppm: number;
  reputation_ppm: number;
  resilience_ppm: number;
  active_events: string[];
  reasons: string[];
  state_after_hash: string;
};
type ValueBreakdown = {
  cash_cents: number;
  capacity_book_value_cents: number;
  total_assets_cents: number;
  capacity_salvage_cents: number;
  awareness_value_cents: number;
  service_value_cents: number;
  reputation_value_cents: number;
  resilience_value_cents: number;
  enterprise_value_cents: number;
};
type RankingRow = { rank: number; company_id: string; value_cents: number; breakdown: ValueBreakdown };
type TrendPoint = {
  round: number;
  enterprise_value_cents: number;
  total_assets_cents: number;
  cash_cents: number;
  cumulative_profit_cents: number;
  market_share_ppm: number;
};
type Retrospective = {
  status: "complete" | "in_progress";
  player_company_id: string;
  outcome: string;
  headline: string;
  rank: number;
  asset_rank: number;
  company_count: number;
  terminal_value_cents: number;
  market_model: { id: string; label: string; description: string; demand_bias_ppm: number; price_anchor_cents: number; price_band_cents: number };
  rankings: { composite: RankingRow[]; total_assets: RankingRow[] };
  ranking_methodology: { composite: string; total_assets: string };
  component_comparison: Array<{ key: string; label: string; own_value_cents: number; leader_value_cents: number; gap_cents: number }>;
  rank_explanation: string[];
  trend_series: Array<{ company_id: string; points: TrendPoint[] }>;
  summary: {
    initial_cash_cents: number;
    final_cash_cents: number;
    cumulative_profit_cents: number;
    initial_share_ppm: number;
    final_share_ppm: number;
    final_reputation_ppm: number;
    final_resilience_ppm: number;
    profitable_rounds: number;
    stockout_rounds: number;
  };
  success_reasons: string[];
  failure_reasons: string[];
  turning_point_rounds: number[];
  rounds: RetrospectiveRound[];
  methodology: string;
};
type Draft = {
  price_cents: number;
  advertising_budget_cents: number;
  service_budget_cents: number;
  capacity_investment_cents: number;
  resilience_budget_cents: number;
  incident_mode: "wait" | "partial_repair" | "full_repair";
  repair_budget_cents: number;
};
type HistoryPoint = {
  settledRound: number;
  state: MarketState;
  market: MarketSnapshot;
  signalOutcomes: Array<{ signal: RiskSignal; realized: boolean }>;
};

const API_URL = process.env.NEXT_PUBLIC_MARKET_API_URL ?? "http://localhost:8010/api";
const DEFAULT_ROUNDS = 10;
const COMPANY_META = [
  { id: "company_A", shortName: "A", name: "青禾速配", color: "#0f766e" },
  { id: "company_B", shortName: "B", name: "橙选到家", color: "#e86445" },
  { id: "company_C", shortName: "C", name: "蓝仓鲜送", color: "#4c6fff" },
  { id: "company_D", shortName: "D", name: "紫藤优鲜", color: "#8b5cf6" },
];
const EVENT_LABELS: Record<string, string> = {
  extreme_weather: "极端天气",
  supply_chain_shock: "供应链冲击",
  regional_logistics_disruption: "区域物流中断",
  festival_demand_surge: "节庆需求激增",
  platform_system_outage: "平台系统故障",
  warehouse_equipment_failure: "仓储设备故障",
  cold_chain_incident: "冷链事故",
};
const SEVERITY_LABELS: Record<string, string> = {
  low: "低",
  medium: "中",
  high: "高",
};

function metaFor(id: string) {
  return COMPANY_META.find((item) => item.id === id) ?? {
    id,
    shortName: id.slice(-1).toUpperCase(),
    name: id,
    color: "#64748b",
  };
}

function money(cents: number) {
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency: "CNY",
    maximumFractionDigits: 0,
  }).format(cents / 100);
}

function compactMoney(cents: number) {
  return `¥${new Intl.NumberFormat("zh-CN", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(cents / 100)}`;
}

function percent(ppm: number, digits = 1) {
  return `${(ppm / 10_000).toFixed(digits)}%`;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "请求失败" }));
    throw new Error(body.detail ?? `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function draftsFrom(payload: EpisodePayload): Record<string, Draft> {
  const preset = payload.action_presets.medium;
  return Object.fromEntries(
    payload.state.company_order.map((companyId) => [
      companyId,
      {
        price_cents: preset.price_cents,
        advertising_budget_cents: preset.advertising_budget_cents,
        service_budget_cents: preset.service_budget_cents,
        capacity_investment_cents: preset.capacity_investment_cents,
        resilience_budget_cents: preset.resilience_budget_cents,
        incident_mode: "wait",
        repair_budget_cents: 0,
      } satisfies Draft,
    ]),
  );
}

function RangeControl({
  label,
  value,
  bound,
  step,
  disabled,
  format,
  timing,
  impact,
  usage,
  onChange,
}: {
  label: string;
  value: number;
  bound: Bound;
  step: number;
  disabled?: boolean;
  format: (value: number) => string;
  timing: string;
  impact: string;
  usage: string;
  onChange: (value: number) => void;
}) {
  const progress = ((value - bound.min) / Math.max(1, bound.max - bound.min)) * 100;
  return (
    <div className={`range-control${disabled ? " disabled" : ""}`}>
      <span><i>{label}</i><strong>{format(value)}</strong></span>
      <input
        type="range"
        min={bound.min}
        max={bound.max}
        step={step}
        value={value}
        disabled={disabled}
        aria-label={label}
        style={{ "--range-progress": `${progress}%` } as React.CSSProperties}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <div className="control-explainer">
        <span>{timing}</span>
        <p>{impact}</p>
        <small>{usage}</small>
      </div>
    </div>
  );
}

function CompactDecisionControl({
  label,
  value,
  bound,
  step,
  disabled,
  format,
  timing,
  impact,
  usage,
  onChange,
}: {
  label: string;
  value: number;
  bound: Bound;
  step: number;
  disabled?: boolean;
  format: (value: number) => string;
  timing: string;
  impact: string;
  usage: string;
  onChange: (value: number) => void;
}) {
  const progress = ((value - bound.min) / Math.max(1, bound.max - bound.min)) * 100;
  return <div className={`command-control${disabled ? " disabled" : ""}`}>
    <div className="command-control-head"><strong>{label}</strong><span>{timing}</span><b>{format(value)}</b></div>
    <input type="range" min={bound.min} max={bound.max} step={step} value={value} disabled={disabled} aria-label={label} style={{ "--range-progress": `${progress}%` } as React.CSSProperties} onChange={(event) => onChange(Number(event.target.value))} />
    <details><summary>{impact}</summary><p>{usage}</p></details>
  </div>;
}

function PreviousRoundBrief({ state, companies, settledMarket }: { state: MarketState; companies: Company[]; settledMarket?: MarketSnapshot }) {
  const settledRound = state.round - 1;
  const isInitial = settledRound === 0;
  const totalSales = companies.reduce((sum, company) => sum + company.commercial.sales_orders, 0);
  const totalRevenue = companies.reduce((sum, company) => sum + company.financial.round_revenue_cents, 0);
  const averageListedPrice = Math.round(companies.reduce((sum, company) => sum + company.commercial.price_cents, 0) / Math.max(1, companies.length));
  const minimumPrice = Math.min(...companies.map((company) => company.commercial.price_cents));
  const maximumPrice = Math.max(...companies.map((company) => company.commercial.price_cents));
  const market = settledMarket ?? state.market;
  const demand = isInitial ? state.market.base_demand_orders : market.realized_demand_orders;
  return (
    <section className="previous-round-brief" aria-label="上一轮公开市场信息">
      <div className="brief-head">
        <div><span className="eyebrow">PUBLIC INFORMATION · DECISION BASIS</span><h3>{isInitial ? "初始公开市场基线" : `上一轮（R${settledRound}）结算快照`}</h3></div>
        <p>{isInitial ? "第一轮尚未发生交易，下列需求为初始基准。" : `这些是进入 Round ${state.round} 决策前所有公司都能看到的已结算信息。`}</p>
      </div>
      <div className="brief-metrics">
        <article><span>{isInitial ? "需求基准" : "全市场实现需求"}</span><strong>{demand.toLocaleString("zh-CN")}<i>单</i></strong><small>进入消费者选择阶段，不等于成交销量</small></article>
        <article><span>公司成交总量</span><strong>{isInitial ? "—" : totalSales.toLocaleString("zh-CN")}<i>{isInitial ? "" : "单"}</i></strong><small>四家公司最终成功履约的订单合计</small></article>
        <article><span>平均挂牌价</span><strong>{money(averageListedPrice)}</strong><small>公司公开报价的简单平均；区间 {money(minimumPrice)}–{money(maximumPrice)}</small></article>
        <article><span>实际成交均价</span><strong>{isInitial || !market.average_paid_price_cents ? "—" : money(market.average_paid_price_cents)}</strong><small>按实际成交订单计算的市场平均支付价格</small></article>
        <article><span>市场销售额</span><strong>{isInitial ? "—" : compactMoney(totalRevenue)}</strong><small>各公司本轮营业收入合计</small></article>
        <article><span>{isInitial ? "当前供应成本指数" : `R${settledRound} 供应成本指数`}</span><strong>{(market.actual_supply_cost_index_ppm / 1_000_000).toFixed(3)}</strong><small>与标题回合使用同一结算时点</small></article>
      </div>
      <div className="public-company-strip">
        {companies.map((company) => {
          const meta = metaFor(company.company_id);
          return <article key={company.company_id}><i style={{ background: meta.color }}>{meta.shortName}</i><div><strong>{meta.name}</strong><small>报价 {money(company.commercial.price_cents)} · 份额 {percent(company.commercial.market_share_ppm)} · 销量 {isInitial ? "尚未成交" : `${company.commercial.sales_orders.toLocaleString("zh-CN")} 单`}</small></div></article>;
        })}
      </div>
      <div className="brief-foot">
        <p><b>流失口径：</b>未购买 {market.no_purchase_orders.toLocaleString("zh-CN")} 单是消费者选择 Outside Option；缺货后流失 {market.lost_after_stockout_orders.toLocaleString("zh-CN")} 单是转售后仍未履约。</p>
        <p><b>公开风险：</b>{state.active_market_events.length ? state.active_market_events.map((event) => EVENT_LABELS[event.event_type] ?? event.event_type).join("、") : "当前无已激活市场事件"}；{state.risk_signals.length ? `${state.risk_signals.length} 条未来风险预警` : "无未来预警"}。</p>
      </div>
    </section>
  );
}

function ProfitChart({ history, companyIds }: { history: HistoryPoint[]; companyIds: string[] }) {
  const maxMagnitude = Math.max(
    1,
    ...history.flatMap((point) =>
      companyIds.map((companyId) =>
        Math.abs(point.state.companies[companyId].financial.round_profit_cents),
      ),
    ),
  );
  return (
    <div className="chart-card">
      <div className="section-heading compact">
        <div><span className="eyebrow">PROFIT / ROUND</span><h3>后端利润轨迹</h3></div>
        <span className="chart-unit">CNY</span>
      </div>
      <div className="profit-chart" aria-label="各公司后端单轮利润柱状图">
        {history.length === 0 ? (
          <div className="chart-empty"><span>01</span><p>提交第一组联合动作后，真实利润轨迹会从这里生长。</p></div>
        ) : history.map((point) => (
          <div className="round-column" key={point.settledRound}>
            <div className="bar-cluster">
              {companyIds.map((companyId) => {
                const profit = point.state.companies[companyId].financial.round_profit_cents;
                return <div
                  key={companyId}
                  className={`profit-bar${profit < 0 ? " loss" : ""}`}
                  style={{
                    height: `${Math.max(4, (Math.abs(profit) / maxMagnitude) * 100)}%`,
                    background: metaFor(companyId).color,
                  }}
                  title={`${metaFor(companyId).name}: ${money(profit)}`}
                />;
              })}
            </div>
            <span>R{point.settledRound}</span>
          </div>
        ))}
      </div>
      <div className="legend-row">
        {companyIds.map((companyId) => <span key={companyId}><i style={{ background: metaFor(companyId).color }} />{metaFor(companyId).shortName}</span>)}
      </div>
    </div>
  );
}

function MarketShareHistory({ history, companyIds }: { history: HistoryPoint[]; companyIds: string[] }) {
  return (
    <div className="statistics-card share-history-card">
      <div className="section-heading compact"><div><span className="eyebrow">MARKET SHARE</span><h3>逐轮市场份额</h3></div><span className="chart-unit">成交订单占比</span></div>
      <p className="statistics-note">每一行是一轮结算后的公司成交份额；它反映竞争位置，不包含未购买和缺货后流失订单。</p>
      <div className="share-history-table">
        {history.length === 0 ? <div className="statistics-empty">完成第一轮后显示份额变化</div> : history.map((point) => <article key={point.settledRound}>
          <span>R{String(point.settledRound).padStart(2, "0")}</span>
          <div>{companyIds.map((companyId) => <i key={companyId} style={{ width: `${point.state.companies[companyId].commercial.market_share_ppm / 10_000}%`, background: metaFor(companyId).color }} title={`${metaFor(companyId).name} ${percent(point.state.companies[companyId].commercial.market_share_ppm)}`} />)}</div>
          <small>{companyIds.map((companyId) => `${metaFor(companyId).shortName} ${percent(point.state.companies[companyId].commercial.market_share_ppm, 0)}`).join(" · ")}</small>
        </article>)}
      </div>
      <div className="legend-row">{companyIds.map((companyId) => <span key={companyId}><i style={{ background: metaFor(companyId).color }} />{metaFor(companyId).name}</span>)}</div>
    </div>
  );
}

function RevenueTable({ companies, settledRound }: { companies: Company[]; settledRound: number }) {
  const totalRevenue = companies.reduce((sum, company) => sum + company.financial.round_revenue_cents, 0);
  return (
    <div className="statistics-card revenue-card">
      <div className="section-heading compact"><div><span className="eyebrow">SALES REVENUE</span><h3>{settledRound ? `R${settledRound} 公司销售额` : "公司销售额"}</h3></div><span className="chart-unit">CNY</span></div>
      <p className="statistics-note">销售额采用后端实际营业收入；客单收入为销售额 ÷ 成交订单，不是公司挂牌价。</p>
      <div className="data-table revenue-table">
        <div className="data-table-head"><span>公司</span><span>成交销量</span><span>销售额</span><span>客单收入</span><span>销售额占比</span></div>
        {companies.map((company) => {
          const meta = metaFor(company.company_id);
          const revenueSharePpm = totalRevenue > 0 ? Math.round(company.financial.round_revenue_cents / totalRevenue * 1_000_000) : 0;
          const revenuePerOrder = company.commercial.sales_orders > 0 ? Math.round(company.financial.round_revenue_cents / company.commercial.sales_orders) : 0;
          return <div className="data-table-row" key={company.company_id}>
            <span><i style={{ background: meta.color }}>{meta.shortName}</i><b>{meta.name}</b></span>
            <span>{settledRound ? `${company.commercial.sales_orders.toLocaleString("zh-CN")} 单` : "—"}</span>
            <span>{settledRound ? money(company.financial.round_revenue_cents) : "—"}</span>
            <span>{settledRound && revenuePerOrder ? money(revenuePerOrder) : "—"}</span>
            <span>{settledRound ? percent(revenueSharePpm) : "—"}</span>
          </div>;
        })}
        <div className="data-table-total"><span>市场合计</span><span>{settledRound ? `${companies.reduce((sum, company) => sum + company.commercial.sales_orders, 0).toLocaleString("zh-CN")} 单` : "—"}</span><strong>{settledRound ? money(totalRevenue) : "—"}</strong></div>
      </div>
    </div>
  );
}

function MarketStatisticsTable({ history }: { history: HistoryPoint[] }) {
  return (
    <div className="statistics-card market-statistics-card">
      <div className="section-heading compact"><div><span className="eyebrow">MARKET STATISTICS</span><h3>逐轮市场统计</h3></div><span className="chart-unit">已结算公开数据</span></div>
      <div className="metric-definition"><b>实现需求</b><span>本轮进入消费者选择阶段的订单总量</span><b>成交总量</b><span>各公司最终履约订单之和</span><b>未购买</b><span>消费者主动选择 Outside Option</span><b>缺货流失</b><span>产能与转售后仍未履约</span></div>
      <div className="market-statistics-scroll">
        <table>
          <thead><tr><th>回合</th><th>实现需求（单）</th><th>成交总量（单）</th><th>市场销售额</th><th>成交均价</th><th>未购买（单）</th><th>缺货流失（单）</th><th>供应成本指数</th></tr></thead>
          <tbody>{history.length === 0 ? <tr><td colSpan={8}>完成第一轮后显示统计数据</td></tr> : history.map((point) => {
            const pointCompanies = point.state.company_order.map((id) => point.state.companies[id]);
            const totalSales = pointCompanies.reduce((sum, company) => sum + company.commercial.sales_orders, 0);
            const totalRevenue = pointCompanies.reduce((sum, company) => sum + company.financial.round_revenue_cents, 0);
            return <tr key={point.settledRound}><td>R{String(point.settledRound).padStart(2, "0")}</td><td>{point.market.realized_demand_orders.toLocaleString("zh-CN")}</td><td>{totalSales.toLocaleString("zh-CN")}</td><td>{money(totalRevenue)}</td><td>{money(point.market.average_paid_price_cents)}</td><td>{point.market.no_purchase_orders.toLocaleString("zh-CN")}</td><td>{point.market.lost_after_stockout_orders.toLocaleString("zh-CN")}</td><td>{(point.market.actual_supply_cost_index_ppm / 1_000_000).toFixed(3)}</td></tr>;
          })}</tbody>
        </table>
      </div>
    </div>
  );
}

function CompanyCockpit({
  state,
  company,
  analysis,
}: {
  state: MarketState;
  company: Company;
  analysis: CompanyAnalysis | null;
}) {
  const meta = metaFor(company.company_id);
  const ranking = state.company_order
    .map((id) => state.companies[id])
    .sort((a, b) => b.commercial.market_share_ppm - a.commercial.market_share_ppm);
  return (
    <section className="company-cockpit" style={{ "--player": meta.color } as React.CSSProperties}>
      <div className="cockpit-heading">
        <div className="player-identity"><span>{meta.shortName}</span><div><small>YOUR COMPANY</small><h2>{meta.name}</h2><p>你负责当前公司的全部资源配置；其他公司由带 Seed 随机风格的规则程序决策。</p></div></div>
        <div className="health-score"><span>经营健康度</span><strong>{analysis?.health_score ?? "—"}</strong><small>/ 100 · {analysis?.health_label ?? "分析中"}</small></div>
      </div>
      <div className="cockpit-metrics">
        <article><span>现金</span><strong>{compactMoney(company.financial.cash_balance_cents)}</strong><small>累计利润 {compactMoney(company.financial.cumulative_profit_cents)}</small></article>
        <article><span>市场份额</span><strong>{percent(company.commercial.market_share_ppm)}</strong><small>当前排名 #{ranking.findIndex((item) => item.company_id === company.company_id) + 1}</small></article>
        <article><span>单位价差</span><strong>{money(analysis?.decision_context.margin_per_order_cents ?? 0)}</strong><small>报价减当前单位成本</small></article>
        <article><span>履约余量</span><strong>{(analysis?.decision_context.capacity_buffer_orders ?? 0).toLocaleString("zh-CN")}</strong><small>有效产能 {company.operations.effective_capacity_orders.toLocaleString("zh-CN")}</small></article>
      </div>
      <div className="cockpit-body">
        <div className="health-factors">
          <div className="mini-heading"><span>STATE DIAGNOSIS</span><h3>公司状态体检</h3></div>
          {analysis?.factors.map((factor) => <article key={factor.key} className={`factor-row ${factor.status}`}>
            <div><strong>{factor.label}</strong><small>{factor.summary}</small></div>
            <span><i style={{ width: `${Math.min(100, factor.value_ppm / 10_000)}%` }} /></span>
            <em>{factor.status === "healthy" ? "稳健" : factor.status === "watch" ? "观察" : "风险"}</em>
          </article>)}
        </div>
        <div className="decision-brief">
          <div className="mini-heading"><span>DECISION BRIEF</span><h3>本轮决策提示</h3></div>
          <p className="brief-disclaimer">提示来自当前状态与已知约束，不替你选择策略。</p>
          <div className="recommendation-list">
            {analysis?.recommendations.map((item, index) => <article key={`${item.dimension}-${index}`} className={item.priority}>
              <span>{index + 1}</span><div><strong>{item.title}</strong><p>{item.rationale}</p></div>
            </article>)}
          </div>
        </div>
        <div className="competitor-watch">
          <div className="mini-heading"><span>COMPETITOR WATCH</span><h3>竞争位置</h3></div>
          {ranking.map((item, index) => {
            const itemMeta = metaFor(item.company_id);
            return <article key={item.company_id} className={item.company_id === company.company_id ? "player" : ""}><b>#{index + 1}</b><i style={{ background: itemMeta.color }}>{itemMeta.shortName}</i><div><strong>{itemMeta.name}</strong><small>{money(item.commercial.price_cents)} · 现金 {compactMoney(item.financial.cash_balance_cents)}</small></div><span>{percent(item.commercial.market_share_ppm)}</span></article>;
          })}
        </div>
      </div>
    </section>
  );
}

function TerminalTrendChart({ retrospective }: { retrospective: Retrospective }) {
  const width = 920;
  const height = 280;
  const padding = { left: 66, right: 24, top: 22, bottom: 36 };
  const values = retrospective.trend_series.flatMap((series) => series.points.map((point) => point.enterprise_value_cents));
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const spread = Math.max(1, rawMax - rawMin);
  const minimum = Math.max(0, rawMin - spread * 0.12);
  const maximum = rawMax + spread * 0.12;
  const maxRounds = retrospective.rounds.length || DEFAULT_ROUNDS;
  const x = (round: number) => padding.left + (round / maxRounds) * (width - padding.left - padding.right);
  const y = (value: number) => padding.top + ((maximum - value) / Math.max(1, maximum - minimum)) * (height - padding.top - padding.bottom);
  const gridValues = Array.from({ length: 5 }, (_, index) => minimum + ((maximum - minimum) * index) / 4).reverse();
  return <section className="terminal-trend-card">
    <div className="terminal-section-head"><div><span>VALUE TREND</span><h3>综合价值折线图</h3></div><p>从 R0 初始状态到 R10 终局，采用与综合榜相同的估值口径。</p></div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="四家公司逐轮综合价值折线图">
      {gridValues.map((value) => <g key={value}><line x1={padding.left} x2={width - padding.right} y1={y(value)} y2={y(value)} /><text x={padding.left - 10} y={y(value) + 3} textAnchor="end">{compactMoney(Math.round(value))}</text></g>)}
      {Array.from({ length: maxRounds + 1 }, (_, round) => <text key={round} x={x(round)} y={height - 10} textAnchor="middle">R{round}</text>)}
      {retrospective.trend_series.map((series) => {
        const meta = metaFor(series.company_id);
        const points = series.points.map((point) => `${x(point.round)},${y(point.enterprise_value_cents)}`).join(" ");
        return <g key={series.company_id} className={series.company_id === retrospective.player_company_id ? "player-line" : ""}>
          <polyline points={points} style={{ stroke: meta.color }} />
          {series.points.map((point) => <circle key={point.round} cx={x(point.round)} cy={y(point.enterprise_value_cents)} r={series.company_id === retrospective.player_company_id ? 4 : 2.5} style={{ fill: meta.color }}><title>{meta.name} R{point.round}: {money(point.enterprise_value_cents)}</title></circle>)}
        </g>;
      })}
    </svg>
    <div className="terminal-chart-legend">{retrospective.trend_series.map((series) => <span key={series.company_id}><i style={{ background: metaFor(series.company_id).color }} />{metaFor(series.company_id).name}</span>)}</div>
  </section>;
}

function TerminalRankings({ retrospective }: { retrospective: Retrospective }) {
  const boards = [
    { key: "composite" as const, eyebrow: "COMPOSITE VALUE", title: "综合价值榜", rows: retrospective.rankings.composite, method: retrospective.ranking_methodology.composite },
    { key: "total_assets" as const, eyebrow: "TOTAL ASSETS", title: "总资产榜", rows: retrospective.rankings.total_assets, method: retrospective.ranking_methodology.total_assets },
  ];
  return <div className="terminal-rankings">{boards.map((board) => <section key={board.key} className="terminal-ranking-card">
    <div className="terminal-section-head"><div><span>{board.eyebrow}</span><h3>{board.title}</h3></div></div>
    <p className="ranking-method">{board.method}</p>
    <div>{board.rows.map((row) => { const meta = metaFor(row.company_id); return <article key={row.company_id} className={row.company_id === retrospective.player_company_id ? "player" : ""}>
      <b>#{row.rank}</b><i style={{ background: meta.color }}>{meta.shortName}</i><div><strong>{meta.name}</strong><small>{board.key === "composite" ? `现金 ${compactMoney(row.breakdown.cash_cents)} · 长期价值 ${compactMoney(row.value_cents - row.breakdown.cash_cents)}` : `现金 ${compactMoney(row.breakdown.cash_cents)} · 产能账面 ${compactMoney(row.breakdown.capacity_book_value_cents)}`}</small></div><span>{compactMoney(row.value_cents)}</span>
    </article>; })}</div>
  </section>)}</div>;
}

function MarketRetrospective({ retrospective }: { retrospective: Retrospective }) {
  return (
    <section className="retrospective">
      <div className="retro-hero">
        <div><span className="eyebrow">MARKET RETROSPECTIVE</span><h2>市场回溯：{retrospective.outcome}</h2><p>{retrospective.headline}</p></div>
        <div className="retro-rank"><span>终局排名</span><strong>#{retrospective.rank}</strong><small>/ {retrospective.company_count} · 企业价值 {compactMoney(retrospective.terminal_value_cents)}</small></div>
      </div>
      <div className="retro-summary">
        <article><span>累计利润</span><strong>{money(retrospective.summary.cumulative_profit_cents)}</strong></article>
        <article><span>最终现金</span><strong>{compactMoney(retrospective.summary.final_cash_cents)}</strong></article>
        <article><span>最终份额</span><strong>{percent(retrospective.summary.final_share_ppm)}</strong></article>
        <article><span>盈利回合</span><strong>{retrospective.summary.profitable_rounds} / {retrospective.rounds.length}</strong></article>
      </div>
      <div className="retro-market-model"><span>本局市场模型</span><strong>{retrospective.market_model.label}</strong><p>{retrospective.market_model.description} 需求偏差 {percent(retrospective.market_model.demand_bias_ppm, 1)}，价格锚点 {money(retrospective.market_model.price_anchor_cents)} ± {money(retrospective.market_model.price_band_cents)}。</p></div>
      <TerminalRankings retrospective={retrospective} />
      <TerminalTrendChart retrospective={retrospective} />
      <section className="rank-explanation">
        <div className="terminal-section-head"><div><span>WHY THIS RANK</span><h3>为什么得到这个结果</h3></div><p>将玩家与综合榜冠军逐项对比。</p></div>
        <div className="rank-explanation-grid"><div>{retrospective.rank_explanation.map((reason) => <p key={reason}>{reason}</p>)}</div><div className="component-comparison">{retrospective.component_comparison.map((item) => <article key={item.key}><span>{item.label}</span><b>{compactMoney(item.own_value_cents)}</b><small className={item.gap_cents > 0 ? "negative" : "positive"}>{item.gap_cents > 0 ? `落后 ${compactMoney(item.gap_cents)}` : `领先 ${compactMoney(-item.gap_cents)}`}</small></article>)}</div></div>
      </section>
      <div className="retro-reasons">
        <div className="success"><span>WHAT WORKED</span><h3>为什么成功</h3>{retrospective.success_reasons.length ? retrospective.success_reasons.map((reason) => <p key={reason}>+ {reason}</p>) : <p>没有识别到持续的正向证据。</p>}</div>
        <div className="failure"><span>VALUE LEAKS</span><h3>为什么失利</h3>{retrospective.failure_reasons.length ? retrospective.failure_reasons.map((reason) => <p key={reason}>− {reason}</p>) : <p>没有识别到显著的价值损失环节。</p>}</div>
      </div>
      <div className="retro-timeline">
        <div className="section-heading"><div><span className="eyebrow">ROUND BY ROUND</span><h2>状态是如何变化的</h2></div><p>{retrospective.methodology}</p></div>
        {retrospective.rounds.map((round) => <article key={round.round} className={retrospective.turning_point_rounds.includes(round.round) ? "turning-point" : ""}>
          <div className="retro-round"><span>R{String(round.round).padStart(2, "0")}</span><strong>{round.verdict}</strong>{retrospective.turning_point_rounds.includes(round.round) && <small>关键转折</small>}</div>
          <div className="retro-stats"><span>利润 <b className={round.profit_cents < 0 ? "negative" : "positive"}>{money(round.profit_cents)}</b></span><span>综合价值变化 <b className={round.enterprise_value_delta_cents < 0 ? "negative" : "positive"}>{round.enterprise_value_delta_cents >= 0 ? "+" : ""}{compactMoney(round.enterprise_value_delta_cents)}</b></span><span>份额 <b>{percent(round.market_share_ppm)} ({round.share_delta_ppm >= 0 ? "+" : ""}{percent(round.share_delta_ppm)})</b></span><span>销量 / 产能 <b>{round.sales_orders.toLocaleString("zh-CN")} / {round.effective_capacity_orders.toLocaleString("zh-CN")}</b></span></div>
          <div className="retro-market"><span>同轮市场</span><p>总需求 {round.market_demand_orders.toLocaleString("zh-CN")} 单（{round.market_demand_delta_orders >= 0 ? "+" : ""}{round.market_demand_delta_orders.toLocaleString("zh-CN")}） · 成交均价 {money(round.average_paid_price_cents)} · 供应成本指数 {(round.supply_cost_index_ppm / 1_000_000).toFixed(3)}（{round.supply_cost_delta_ppm >= 0 ? "+" : ""}{(round.supply_cost_delta_ppm / 1_000_000).toFixed(3)}） · 场外流失 {round.outside_option_orders.toLocaleString("zh-CN")} 单 · 事件 {round.active_events.length ? round.active_events.map((event) => EVENT_LABELS[event] ?? event).join(" / ") : "无"}</p></div>
          <div className="retro-action"><span>决策与成本</span><p>价格 {money(Number(round.action.price_cents))} · 广告 {compactMoney(Number(round.action.advertising_budget_cents))} · 服务 {compactMoney(Number(round.action.service_budget_cents))} · 产能 {compactMoney(Number(round.action.capacity_investment_cents))} · 韧性 {compactMoney(Number(round.action.resilience_budget_cents))} · 常规运营 {compactMoney(round.operating_cost_cents)}</p></div>
          <ul>{round.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
          <code>{round.state_after_hash.slice(0, 30)}…</code>
        </article>)}
      </div>
    </section>
  );
}

export default function Home() {
  const [gameMode, setGameMode] = useState<GameMode>("single_company");
  const [playerCompanyId, setPlayerCompanyId] = useState("company_A");
  const [companyCount, setCompanyCount] = useState(4);
  const [state, setState] = useState<MarketState | null>(null);
  const [constraints, setConstraints] = useState<Record<string, ActionConstraints>>({});
  const [presets, setPresets] = useState<EpisodePayload["action_presets"]>({});
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [companyAnalysis, setCompanyAnalysis] = useState<CompanyAnalysis | null>(null);
  const [retrospective, setRetrospective] = useState<Retrospective | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState("正在连接 MARKET_ENV_V4");
  const [error, setError] = useState<string | null>(null);
  const [seedMode, setSeedMode] = useState<SeedMode>("random");
  const [fixedSeed, setFixedSeed] = useState(42);
  const [activeSeed, setActiveSeed] = useState<number | null>(null);
  const [marketModel, setMarketModel] = useState<MarketModel>("random");
  const [marketModelOptions, setMarketModelOptions] = useState<Record<string, { label: string; description: string }>>({});
  const [selectedRounds, setSelectedRounds] = useState(DEFAULT_ROUNDS);
  const [roundOptions, setRoundOptions] = useState<number[]>([5, 10, 15, 20]);

  async function createEpisode(
    count = companyCount,
    mode: GameMode = gameMode,
    player = playerCompanyId,
    model: MarketModel = marketModel,
    rounds = selectedRounds,
  ) {
    setLoading(true);
    setError(null);
    try {
      const randomValues = new Uint32Array(1);
      crypto.getRandomValues(randomValues);
      const episodeSeed = seedMode === "fixed" ? fixedSeed : randomValues[0];
      const payload = await api<EpisodePayload>("/episodes", {
        method: "POST",
        body: JSON.stringify({
          episode_seed: episodeSeed,
          company_ids: COMPANY_META.slice(0, count).map((company) => company.id),
          game_mode: mode,
          player_company_id: mode === "single_company" ? player : null,
          market_model: model,
          max_rounds: rounds,
        }),
      });
      setState(payload.state);
      setConstraints(payload.action_constraints);
      setPresets(payload.action_presets);
      setMarketModelOptions(payload.market_model_options);
      setRoundOptions(payload.episode_options.round_options);
      setSelectedRounds(payload.state.max_rounds);
      setDrafts(draftsFrom(payload));
      setHistory([]);
      setCompanyAnalysis(payload.company_analysis ?? null);
      setRetrospective(null);
      setActiveSeed(payload.state.episode_seed);
      setNotice(mode === "single_company" ? "你的公司已就绪 · 对手由随机规则程序控制" : "后端已连接 · 等待联合动作");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法连接市场后端");
      setNotice("后端连接失败");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    // The initial remote synchronization intentionally initializes component state.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void createEpisode(4, "single_company", "company_A");
    // Initial connection only; resets are explicit user actions.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const companyIds = useMemo(
    () => state?.company_order ?? COMPANY_META.slice(0, companyCount).map((item) => item.id),
    [state, companyCount],
  );
  const companies = useMemo(
    () => state ? companyIds.map((id) => state.companies[id]) : [],
    [state, companyIds],
  );
  const leader = useMemo(
    () => [...companies].sort((a, b) => b.commercial.market_share_ppm - a.commercial.market_share_ppm)[0],
    [companies],
  );
  const topProfit = useMemo(
    () => [...companies].sort((a, b) => b.financial.round_profit_cents - a.financial.round_profit_cents)[0],
    [companies],
  );

  function updateDraft(companyId: string, field: keyof Draft, value: Draft[keyof Draft]) {
    setDrafts((current) => ({
      ...current,
      [companyId]: { ...current[companyId], [field]: value },
    }));
    setNotice("动作已编辑 · 尚未锁定");
  }

  function changeIncidentMode(companyId: string, mode: Draft["incident_mode"]) {
    const maximum = constraints[companyId].max_useful_repair_budget_cents;
    updateDraft(companyId, "incident_mode", mode);
    updateDraft(
      companyId,
      "repair_budget_cents",
      mode === "full_repair" ? maximum : mode === "partial_repair" ? Math.max(1, Math.floor(maximum / 2)) : 0,
    );
  }

  function randomizeActions() {
    if (!state) return;
    setDrafts((current) => Object.fromEntries(companyIds.map((companyId) => {
      const companyConstraints = constraints[companyId];
      const randomWithin = (field: string, step: number) => {
        const bound = companyConstraints.bounds[field];
        const slots = Math.floor((bound.max - bound.min) / step);
        return bound.min + Math.floor(Math.random() * (slots + 1)) * step;
      };
      return [companyId, {
        ...current[companyId],
        price_cents: randomWithin("price_cents", 50),
        advertising_budget_cents: randomWithin("advertising_budget_cents", 100_000),
        service_budget_cents: randomWithin("service_budget_cents", 100_000),
        capacity_investment_cents: companyConstraints.capacity_investment_enabled ? randomWithin("capacity_investment_cents", 100_000) : 0,
        resilience_budget_cents: companyConstraints.resilience_investment_enabled ? randomWithin("resilience_budget_cents", 100_000) : 0,
      }];
    })));
    setNotice("已生成一组可行动作 · 尚未提交");
  }

  async function commitRound() {
    if (!state || state.terminal || submitting) return;
    setSubmitting(true);
    setError(null);
    setNotice(gameMode === "single_company" ? `正在结算你的 Round ${state.round} 决策` : `正在锁定 Round ${state.round} 联合动作`);
    const jointAction = Object.fromEntries(companyIds.map((companyId) => {
      const draft = drafts[companyId];
      return [companyId, {
        action_id: crypto.randomUUID(),
        episode_id: state.episode_id,
        agent_id: companyId,
        round: state.round,
        state_version: state.state_version,
        price_cents: draft.price_cents,
        advertising_budget_cents: draft.advertising_budget_cents,
        service_budget_cents: draft.service_budget_cents,
        capacity_investment_cents: draft.capacity_investment_cents,
        resilience_budget_cents: draft.resilience_budget_cents,
        incident_response: {
          mode: draft.incident_mode,
          repair_budget_cents: draft.repair_budget_cents,
        },
        strategy_summary: "frontend numeric action",
      }];
    }));
    try {
      const endpoint = gameMode === "single_company" ? `/episodes/${state.episode_id}/player-steps` : `/episodes/${state.episode_id}/steps`;
      const requestBody = gameMode === "single_company"
        ? {
            step_id: `${state.episode_id}:${state.round}:${state.state_version}`,
            player_action: jointAction[playerCompanyId],
          }
        : {
            step_id: `${state.episode_id}:${state.round}:${state.state_version}`,
            joint_action: jointAction,
          };
      const payload = await api<{
        state: MarketState;
        settled_market: MarketSnapshot;
        action_constraints: Record<string, ActionConstraints>;
        step_result: { settled_round: number };
        company_analysis?: CompanyAnalysis;
        retrospective?: Retrospective;
      }>(endpoint, {
        method: "POST",
        body: JSON.stringify(requestBody),
      });
      setState(payload.state);
      setConstraints(payload.action_constraints);
      const signalOutcomes = state.risk_signals
        .filter((signal) => signal.target_round === payload.state.round)
        .map((signal) => ({
          signal,
          realized: payload.state.active_market_events.some((event) => event.event_id.includes(signal.signal_id)),
        }));
      setHistory((current) => [...current, { settledRound: payload.step_result.settled_round, state: payload.state, market: payload.settled_market, signalOutcomes }]);
      setCompanyAnalysis(payload.company_analysis ?? null);
      setRetrospective(payload.retrospective ?? null);
      setDrafts((current) => Object.fromEntries(companyIds.map((companyId) => [
        companyId,
        {
          ...current[companyId],
          capacity_investment_cents: payload.state.rounds_remaining <= 1 ? 0 : current[companyId].capacity_investment_cents,
          resilience_budget_cents: payload.state.rounds_remaining <= 1 ? 0 : current[companyId].resilience_budget_cents,
          incident_mode: "wait",
          repair_budget_cents: 0,
        },
      ])));
      setNotice(payload.state.terminal ? "Episode 已完成 · 市场回溯已生成" : gameMode === "single_company" ? `Round ${payload.step_result.settled_round} 完成 · 对手已同步行动` : `Round ${payload.step_result.settled_round} 已由后端结算`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "联合动作提交失败");
      setNotice("动作未提交 · 状态未改变");
    } finally {
      setSubmitting(false);
    }
  }

  async function changeCount(count: number) {
    setCompanyCount(count);
    await createEpisode(count);
  }

  async function changeMode(mode: GameMode) {
    setGameMode(mode);
    const count = mode === "single_company" ? 4 : companyCount;
    if (mode === "single_company") setCompanyCount(4);
    await createEpisode(count, mode, playerCompanyId);
  }

  async function changePlayer(companyId: string) {
    setPlayerCompanyId(companyId);
    await createEpisode(4, "single_company", companyId);
  }

  async function changeMarketModel(model: MarketModel) {
    setMarketModel(model);
    await createEpisode(
      gameMode === "single_company" ? 4 : companyCount,
      gameMode,
      playerCompanyId,
      model,
      selectedRounds,
    );
  }

  async function changeRounds(rounds: number) {
    setSelectedRounds(rounds);
    await createEpisode(
      gameMode === "single_company" ? 4 : companyCount,
      gameMode,
      playerCompanyId,
      marketModel,
      rounds,
    );
  }

  function applyPreset(level: "low" | "medium" | "high") {
    if (!presets[level] || !state) return;
    const preset = presets[level];
    setDrafts((current) => Object.fromEntries(companyIds.map((companyId) => [companyId, {
      ...current[companyId],
      price_cents: preset.price_cents,
      advertising_budget_cents: preset.advertising_budget_cents,
      service_budget_cents: preset.service_budget_cents,
      capacity_investment_cents: state.rounds_remaining <= 1 ? 0 : preset.capacity_investment_cents,
      resilience_budget_cents: state.rounds_remaining <= 1 ? 0 : preset.resilience_budget_cents,
    }])));
    setNotice(`已应用 ${level.toUpperCase()} 参数锚点 · 尚未提交`);
  }

  const settledRound = state ? state.round - 1 : 0;
  const maxRounds = state?.max_rounds ?? selectedRounds;
  const progress = (settledRound / maxRounds) * 100;
  const backendOnline = Boolean(state && !error);
  const editableCompanyIds = gameMode === "single_company" ? [playerCompanyId] : companyIds;
  const playerCompany = state?.companies[playerCompanyId] ?? null;

  if (gameMode === "single_company") {
    const draft = drafts[playerCompanyId];
    const playerConstraints = constraints[playerCompanyId];
    const averageListedPrice = companies.length ? Math.round(companies.reduce((sum, company) => sum + company.commercial.price_cents, 0) / companies.length) : 0;
    const totalSales = companies.reduce((sum, company) => sum + company.commercial.sales_orders, 0);
    const totalRevenue = companies.reduce((sum, company) => sum + company.financial.round_revenue_cents, 0);
    const fixedSpend = draft ? draft.advertising_budget_cents + draft.service_budget_cents + draft.capacity_investment_cents + draft.resilience_budget_cents + draft.repair_budget_cents : 0;
    const rankedCompanies = [...companies].sort((a, b) => b.commercial.market_share_ppm - a.commercial.market_share_ppm);
    const incident = playerCompany?.risk.active_incident ?? null;
    const recentHistory = [...history].slice(-5).reverse();
    const lastSettledMarket = history[history.length - 1]?.market;
    return <main className={`command-shell${state?.terminal ? " terminal" : ""}`}>
      <header className="command-topbar">
        <div className="command-brand"><span>FM</span><div><strong>FRESH MARKET LAB</strong><small>单公司经营控制台</small></div></div>
        <div className="command-mode"><button type="button" className="active">单公司经营</button><button type="button" onClick={() => void changeMode("market")}>市场全景</button></div>
        <div className="command-company-picker"><span>你的公司</span>{COMPANY_META.map((company) => <button type="button" key={company.id} title={company.name} className={playerCompanyId === company.id ? "active" : ""} style={{ "--pick": company.color } as React.CSSProperties} onClick={() => void changePlayer(company.id)}>{company.shortName}</button>)}</div>
        <div className="command-seed"><select aria-label="市场模型" title={state?.market.market_model_description} value={marketModel} onChange={(event) => void changeMarketModel(event.target.value as MarketModel)}><option value="random">随机市场</option>{Object.entries(marketModelOptions).map(([id, option]) => <option key={id} value={id}>{option.label}</option>)}</select><select aria-label="轮数" value={selectedRounds} onChange={(event) => void changeRounds(Number(event.target.value))}>{roundOptions.map((rounds) => <option key={rounds} value={rounds}>{rounds} 轮</option>)}</select><select aria-label="Seed 模式" value={seedMode} onChange={(event) => setSeedMode(event.target.value as SeedMode)}><option value="random">随机 Seed</option><option value="fixed">固定 Seed</option></select>{seedMode === "fixed" && <input aria-label="固定 Seed" type="number" min={0} max={4_294_967_295} value={fixedSeed} onChange={(event) => setFixedSeed(Math.max(0, Math.min(4_294_967_295, Number(event.target.value) || 0)))} />}<small>{state?.market.market_model_label ?? "—"} · Seed {activeSeed ?? "—"}</small></div>
        <div className="command-round"><span>ROUND</span><strong>{state?.terminal ? maxRounds : state?.round ?? 1}</strong><small>/ {maxRounds}</small><i><b style={{ width: `${progress}%` }} /></i></div>
        <div className={`command-connection${backendOnline ? "" : " offline"}`}><i />{backendOnline ? "市场模型在线" : "正在连接"}</div>
      </header>

      <section className="command-ticker" aria-label="决策所需公开市场摘要">
        <article><span>{settledRound ? `R${settledRound} 实现需求` : "初始需求基准"}</span><strong>{state ? (settledRound && lastSettledMarket ? lastSettledMarket.realized_demand_orders : state.market.base_demand_orders).toLocaleString("zh-CN") : "—"}<i> 单</i></strong><small title={state?.market.market_model_description}>{state?.market.market_model_description ?? "进入消费者选择，非成交量"}</small></article>
        <article><span>成交总量</span><strong>{settledRound ? totalSales.toLocaleString("zh-CN") : "—"}<i>{settledRound ? " 单" : ""}</i></strong><small>最终成功履约订单</small></article>
        <article><span>平均挂牌价</span><strong>{averageListedPrice ? money(averageListedPrice) : "—"}</strong><small>公开报价简单平均</small></article>
        <article><span>{settledRound ? `R${settledRound} 成交均价` : "实际成交均价"}</span><strong>{settledRound && lastSettledMarket?.average_paid_price_cents ? money(lastSettledMarket.average_paid_price_cents) : "—"}</strong><small>按成交订单加权</small></article>
        <article><span>市场销售额</span><strong>{settledRound ? compactMoney(totalRevenue) : "—"}</strong><small>公司营业收入合计</small></article>
        <article><span>未购买 / 缺货流失</span><strong>{lastSettledMarket ? `${lastSettledMarket.no_purchase_orders.toLocaleString("zh-CN")} / ${lastSettledMarket.lost_after_stockout_orders.toLocaleString("zh-CN")}` : "—"}</strong><small>上一轮 Outside Option / 未履约</small></article>
        <article><span>{state?.terminal ? "终局供应成本" : `R${state?.round ?? 1} 当前供应成本`}</span><strong>{state ? (state.market.actual_supply_cost_index_ppm / 1_000_000).toFixed(3) : "—"}</strong><small>含当前已激活事件，供本轮决策</small></article>
      </section>

      {state?.terminal && retrospective ? <div className="command-terminal-view"><div className="command-terminal-actions"><button type="button" onClick={() => void createEpisode(4, "single_company", playerCompanyId)}>再经营一次</button><span>Episode 已结束，以下是完整市场回溯。</span></div><MarketRetrospective retrospective={retrospective} /></div> : <div className="command-layout">
        <section className="command-company-panel command-panel">
          <div className="command-panel-head"><div><span>01 · COMPANY</span><h2>公司状态</h2></div><button type="button" onClick={() => void createEpisode(4, "single_company", playerCompanyId)}>重新开始</button></div>
          {playerCompany ? <>
            <div className="command-company-identity" style={{ "--player": metaFor(playerCompanyId).color } as React.CSSProperties}><i>{metaFor(playerCompanyId).shortName}</i><div><strong>{metaFor(playerCompanyId).name}</strong><small>{playerCompany.persona}</small></div><b>{companyAnalysis?.health_score ?? "—"}<small>健康度</small></b></div>
            <div className="command-company-kpis"><article><span>现金</span><strong>{compactMoney(playerCompany.financial.cash_balance_cents)}</strong><small>累计利润 {compactMoney(playerCompany.financial.cumulative_profit_cents)}</small></article><article><span>市场份额</span><strong>{percent(playerCompany.commercial.market_share_ppm)}</strong><small>排名 #{rankedCompanies.findIndex((company) => company.company_id === playerCompanyId) + 1}</small></article><article><span>单笔贡献</span><strong>{money(companyAnalysis?.decision_context.margin_per_order_cents ?? 0)}</strong><small>报价 − 商品 − 履约成本</small></article><article><span>产能利用率</span><strong>{percent(playerCompany.operations.capacity_utilization_ppm, 0)}</strong><small>余量 {companyAnalysis?.decision_context.capacity_buffer_orders.toLocaleString("zh-CN") ?? "—"} 单</small></article></div>
            <div className="command-diagnosis"><div className="command-subhead"><span>状态体检</span><small>当前公开状态</small></div>{companyAnalysis?.factors.map((factor) => <article key={factor.key}><div><strong>{factor.label}</strong><small>{factor.summary}</small></div><span><i className={factor.status} style={{ width: `${Math.min(100, factor.value_ppm / 10_000)}%` }} /></span><em>{factor.status === "healthy" ? "稳健" : factor.status === "watch" ? "观察" : "风险"}</em></article>)}</div>
            <div className="command-advice"><div className="command-subhead"><span>本轮提示</span><small>规则证据，不代替决策</small></div>{companyAnalysis?.recommendations.slice(0, 3).map((item, index) => <article key={`${item.dimension}-${index}`}><b>{index + 1}</b><div><strong>{item.title}</strong><p>{item.rationale}</p></div></article>)}</div>
          </> : <div className="command-loading">正在载入公司状态…</div>}
        </section>

        <section className="command-decision-panel command-panel">
          <div className="command-panel-head"><div><span>02 · DECISION</span><h2>Round {state?.round ?? 1} 资源配置</h2></div><div className="command-presets"><button type="button" onClick={() => applyPreset("low")}>低投入</button><button type="button" onClick={() => applyPreset("medium")}>中投入</button><button type="button" onClick={() => applyPreset("high")}>高投入</button></div></div>
          <div className="command-decision-summary"><p>先确定价格定位，再分配获客、履约和风险资源。市场还会扣除固定运营费与逐单履约费。</p><div><span>主动投入 / 固定运营</span><strong>{money(fixedSpend)}</strong><small>{playerConstraints ? `+ ${money(playerConstraints.mandatory_operating_costs.fixed_overhead_cents)}，另 ${money(playerConstraints.mandatory_operating_costs.fulfillment_cost_per_order_cents)}/单` : "/ 可用 —"}</small></div></div>
          {draft && playerConstraints ? <div className="command-controls">
            <CompactDecisionControl label="价格" value={draft.price_cents} bound={playerConstraints.bounds.price_cents} step={50} format={money} timing="当轮生效" impact="影响消费者选择与每单毛利" usage={`上一轮平均挂牌价为 ${money(averageListedPrice)}。低价通常争取份额，高价需要品牌、服务或产能支撑。`} onChange={(value) => updateDraft(playerCompanyId, "price_cents", value)} />
            <CompactDecisionControl label="广告" value={draft.advertising_budget_cents} bound={playerConstraints.bounds.advertising_budget_cents} step={100_000} format={money} timing="当轮 + 跨轮" impact="提升吸引力并沉淀品牌知名度" usage="适合份额或知名度偏弱时使用；投入先占用现金，同轮不保证完全回收。" onChange={(value) => updateDraft(playerCompanyId, "advertising_budget_cents", value)} />
            <CompactDecisionControl label="服务" value={draft.service_budget_cents} bound={playerConstraints.bounds.service_budget_cents} step={100_000} format={money} timing="当轮 + 跨轮" impact="改善消费者选择、服务质量与声誉" usage="适合支撑溢价或修复声誉；如果根因是产能不足，仅增加服务不能消除缺货。" onChange={(value) => updateDraft(playerCompanyId, "service_budget_cents", value)} />
            <CompactDecisionControl label="产能" value={draft.capacity_investment_cents} bound={playerConstraints.bounds.capacity_investment_cents} step={100_000} disabled={!playerConstraints.capacity_investment_enabled} format={money} timing="下一轮生效" impact="提高后续有效产能，期末保留资产价值" usage="高利用率或持续缺货时使用。本轮先承担成本，最后一轮因来不及形成运营收益而禁用。" onChange={(value) => updateDraft(playerCompanyId, "capacity_investment_cents", value)} />
            <CompactDecisionControl label="韧性" value={draft.resilience_budget_cents} bound={playerConstraints.bounds.resilience_budget_cents} step={100_000} disabled={!playerConstraints.resilience_investment_enabled} format={money} timing="保护后续轮次" impact="缓冲未来市场事件与公司事故损失" usage="出现未来风险预警时更有价值；不能抵消本轮已经激活的事件。" onChange={(value) => updateDraft(playerCompanyId, "resilience_budget_cents", value)} />
            {incident ? <div className="command-incident"><div><strong>事故维修</strong><small>{EVENT_LABELS[incident.incident_type] ?? incident.incident_type} · 待修 {money(playerConstraints.max_useful_repair_budget_cents)} · 本轮仍保留残余影响</small></div>{(["wait", "partial_repair", "full_repair"] as const).map((mode) => <button type="button" key={mode} className={draft.incident_mode === mode ? "active" : ""} onClick={() => changeIncidentMode(playerCompanyId, mode)}>{mode === "wait" ? "等待" : mode === "partial_repair" ? "部分维修" : "完全维修"}</button>)}</div> : <div className="command-no-incident">✓ 当前无公司事故</div>}
          </div> : <div className="command-loading">正在载入动作约束…</div>}
          <div className="command-submit"><div><span>{fixedSpend > (playerConstraints?.cash_available_cents ?? Infinity) ? "预算超出可用现金" : notice}</span><small>提交后，3 个规则对手同步决策并由后端统一结算。</small></div><button type="button" onClick={() => void commitRound()} disabled={!state || loading || submitting || state.terminal || fixedSpend > (playerConstraints?.cash_available_cents ?? 0)}>{submitting ? "结算中…" : `提交 Round ${state?.round ?? 1}`}<i>→</i></button></div>
        </section>

        <aside className="command-market-panel command-panel">
          <div className="command-panel-head"><div><span>03 · MARKET</span><h2>公开市场情报</h2></div><small>{history.length} / {maxRounds} 已结算</small></div>
          <div className="command-share"><div className="command-subhead"><span>当前市场份额</span><small>成交订单占比</small></div><div>{companies.map((company) => <i key={company.company_id} style={{ width: `${company.commercial.market_share_ppm / 10_000}%`, background: metaFor(company.company_id).color }} title={`${metaFor(company.company_id).name} ${percent(company.commercial.market_share_ppm)}`} />)}</div></div>
          <div className="command-public-table"><div><span>公司</span><span>报价</span><span>份额</span><span>销量</span><span>销售额</span></div>{rankedCompanies.map((company) => { const meta = metaFor(company.company_id); return <article key={company.company_id} className={company.company_id === playerCompanyId ? "player" : ""}><span><i style={{ background: meta.color }}>{meta.shortName}</i>{meta.name}</span><span>{money(company.commercial.price_cents)}</span><strong>{percent(company.commercial.market_share_ppm)}</strong><span>{settledRound ? company.commercial.sales_orders.toLocaleString("zh-CN") : "—"}</span><span>{settledRound ? compactMoney(company.financial.round_revenue_cents) : "—"}</span></article>; })}</div>
          <div className="command-risk-block"><div className="command-subhead"><span>事件与风险</span><small>{state?.active_market_events.length ?? 0} 激活 · {state?.risk_signals.length ?? 0} 预警</small></div><div>{state?.active_market_events.map((event) => <article key={event.event_id} className="active"><b>!</b><span><strong>{EVENT_LABELS[event.event_type] ?? event.event_type}</strong><small>{SEVERITY_LABELS[event.severity]}强度 · 剩余 {event.remaining_rounds} 轮 · 需求 {percent(event.demand_multiplier_ppm, 0)} · 成本 {percent(event.supply_cost_multiplier_ppm, 0)} · 产能 {percent(event.capacity_multiplier_ppm, 0)}</small></span></article>)}{state?.risk_signals.map((signal) => <article key={signal.signal_id}><b>↗</b><span><strong>{EVENT_LABELS[signal.event_type] ?? signal.event_type}</strong><small>R{signal.target_round} · 概率 {percent(signal.estimated_probability_ppm, 0)}</small></span></article>)}{recentHistory[0]?.signalOutcomes.map(({ signal, realized }) => <article key={`outcome-${signal.signal_id}`} className={realized ? "active" : ""}><b>{realized ? "✓" : "×"}</b><span><strong>{EVENT_LABELS[signal.event_type] ?? signal.event_type}</strong><small>R{signal.target_round} 预警{realized ? "已兑现" : "未发生"}</small></span></article>)}{!state?.active_market_events.length && !state?.risk_signals.length && !recentHistory[0]?.signalOutcomes.length && <p>当前无已激活事件或未来风险预警</p>}</div></div>
          <div className="command-history"><div className="command-subhead"><span>最近回合</span><small>需求 → 成交 → 流失</small></div>{recentHistory.length ? recentHistory.map((point) => { const pointCompanies = point.state.company_order.map((id) => point.state.companies[id]); const pointSales = pointCompanies.reduce((sum, company) => sum + company.commercial.sales_orders, 0); return <article key={point.settledRound}><b>R{point.settledRound}</b><span><strong>{point.state.market.realized_demand_orders.toLocaleString("zh-CN")}</strong><small>实现需求</small></span><span><strong>{pointSales.toLocaleString("zh-CN")}</strong><small>成交</small></span><span><strong>{(point.state.market.no_purchase_orders + point.state.market.lost_after_stockout_orders).toLocaleString("zh-CN")}</strong><small>总流失</small></span></article>; }) : <p>完成第一轮后显示回合统计</p>}</div>
          <div className="command-definition"><b>实现需求</b> = 进入消费者选择的全市场订单；<b>成交</b> = 最终成功履约；<b>流失</b> = 未购买 + 缺货后未履约。</div>
        </aside>
      </div>}
      {error && <div className="command-error">{error}</div>}
    </main>;
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><span>FM</span></div><div><strong>FRESH MARKET LAB</strong><span>动态随机市场实验台</span></div></div>
        <div className="header-status">
          <span className={`status-pill${backendOnline ? "" : " offline"}`}><i />{backendOnline ? "后端实时连接" : "等待后端"}</span>
          <span className="version">MARKET_ENV_V4.0</span>
        </div>
      </header>

      <section className="mode-dock" aria-label="体验模式">
        <div className="mode-switch">
          <button type="button" className={gameMode === "single_company" ? "active" : ""} onClick={() => void changeMode("single_company")}><span>01</span><strong>单公司经营</strong><small>你决策 · 随机规则对手行动</small></button>
          <button type="button" className={gameMode === "market" ? "active" : ""} onClick={() => void changeMode("market")}><span>02</span><strong>市场全景</strong><small>配置全部公司联合动作</small></button>
        </div>
        {gameMode === "single_company" && <div className="player-picker"><span>选择你的公司</span>{COMPANY_META.map((company) => <button type="button" key={company.id} className={playerCompanyId === company.id ? "active" : ""} style={{ "--pick": company.color } as React.CSSProperties} onClick={() => void changePlayer(company.id)}>{company.shortName}<small>{company.name}</small></button>)}</div>}
      </section>

      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">{gameMode === "single_company" ? "MANAGEMENT SIMULATION" : "SEEDED DYNAMIC MARKET"}</span>
          <h1>{gameMode === "single_company" ? <>经营一家公司，<br />承担每个选择。</> : <>决策会留下痕迹，<br />市场也会改变。</>}</h1>
          <p>{gameMode === "single_company" ? "分析你的现金、产能、品牌和风险状态，只提交一家公司的决策。竞争者会自主响应，十轮结束后用完整市场轨迹解释成败。" : "连续配置价格、广告、服务、产能与韧性。后端统一处理消费者选择、现金约束、缺货转售、风险事件和公司事故，前端只展示真实的状态转移。"}</p>
        </div>
        <div className="round-panel">
          <div className="round-number"><span>SETTLED</span><strong>{String(settledRound).padStart(2, "0")}</strong><small>/ {maxRounds}</small></div>
          <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
          <div className="round-state"><span className="pulse-dot" />{notice}</div>
          {error && <div className="error-banner">{error}</div>}
        </div>
      </section>

      <section className="metric-strip" aria-label="市场概览">
        <article><span>{settledRound ? "上一轮全市场实现需求" : "初始需求基准"}</span><strong>{state ? (settledRound ? state.market.realized_demand_orders : state.market.base_demand_orders).toLocaleString("zh-CN") : "—"}<i> 单</i></strong><small>{settledRound ? "进入消费者选择，不等于成交量" : "尚未进入首轮交易"}</small></article>
        <article><span>未购买 / 缺货后流失</span><strong>{state ? `${state.market.no_purchase_orders.toLocaleString("zh-CN")} / ${state.market.lost_after_stockout_orders.toLocaleString("zh-CN")}` : "—"}</strong><small>单位：单 · 两类流失口径不同</small></article>
        <article><span>当前领先</span><strong style={{ color: leader ? metaFor(leader.company_id).color : undefined }}>{leader ? `${metaFor(leader.company_id).shortName} · ${percent(leader.commercial.market_share_ppm)}` : "—"}</strong><small>{leader ? metaFor(leader.company_id).name : "等待状态"}</small></article>
        <article><span>最高单轮利润</span><strong>{topProfit && settledRound ? compactMoney(topProfit.financial.round_profit_cents) : "—"}</strong><small>{topProfit && settledRound ? metaFor(topProfit.company_id).name : "等待首轮结果"}</small></article>
      </section>

      {gameMode === "single_company" && state && playerCompany && <CompanyCockpit state={state} company={playerCompany} analysis={companyAnalysis} />}

      <section className="workspace-grid">
        <div className="market-board panel">
          <div className="section-heading">
            <div><span className="eyebrow">MARKET STATE</span><h2>成交份额与经营状态</h2></div>
            {gameMode === "market" && <div className="company-count" role="group" aria-label="公司数量">
              {[2, 3, 4].map((count) => <button type="button" key={count} className={companyCount === count ? "active" : ""} onClick={() => void changeCount(count)} disabled={loading}>{count} 家</button>)}
            </div>}
          </div>
          <div className="share-stack" aria-label="后端市场份额堆叠图">
            {companies.map((company) => {
              const meta = metaFor(company.company_id);
              return <div key={company.company_id} style={{ width: `${company.commercial.market_share_ppm / 10_000}%`, background: meta.color }}>{company.commercial.market_share_ppm > 150_000 && <span>{meta.shortName} {percent(company.commercial.market_share_ppm)}</span>}</div>;
            })}
          </div>
          <div className="company-table dynamic-table">
            <div className="table-head"><span>公司</span><span>价格</span><span>成交份额</span><span>销量 / 产能</span><span>利润 / 现金</span></div>
            {companies.map((company) => {
              const meta = metaFor(company.company_id);
              return <div className="table-row" key={company.company_id}>
                <span className="company-name"><i style={{ background: meta.color }}>{meta.shortName}</i><b>{meta.name}<small>声誉 {percent(company.brand.reputation_ppm, 0)} · 韧性 {percent(company.risk.resilience_ppm, 0)}</small></b></span>
                <span>{money(company.commercial.price_cents)}<small>成本 {money(company.operations.actual_unit_cost_cents)}</small></span>
                <span><b>{percent(company.commercial.market_share_ppm)}</b><em><i style={{ width: `${company.commercial.market_share_ppm / 10_000}%`, background: meta.color }} /></em></span>
                <span>{company.commercial.sales_orders.toLocaleString("zh-CN")} / {company.operations.effective_capacity_orders.toLocaleString("zh-CN")}<small>利用率 {percent(company.operations.capacity_utilization_ppm, 0)}</small></span>
                <span className={company.financial.round_profit_cents < 0 ? "negative" : "positive"}>{settledRound ? money(company.financial.round_profit_cents) : "—"}<small>现金 {compactMoney(company.financial.cash_balance_cents)}</small></span>
              </div>;
            })}
          </div>
          <div className="formula-note"><span>BACKEND SINGLE SOURCE</span><p>Segment Choice → Outside Option → Capacity / Cash → Redistribution → State Hash</p><code>{state?.state_hash ? `${state.state_hash.slice(0, 24)}…` : "等待 State Hash"}</code></div>
        </div>

        <aside className="risk-panel panel">
          <div className="section-heading compact"><div><span className="eyebrow">RISK DESK</span><h3>风险与事件</h3></div><span className="event-count">R{state?.round ?? 1}</span></div>
          <div className="risk-section">
            <span className="risk-label">ACTIVE EVENTS</span>
            {state?.active_market_events.length ? state.active_market_events.map((event) => <article className="risk-item active" key={event.event_id}><i>!</i><div><strong>{EVENT_LABELS[event.event_type] ?? event.event_type}</strong><p>{SEVERITY_LABELS[event.severity]}强度 · 剩余 {event.remaining_rounds} 轮</p></div></article>) : <p className="risk-empty">当前无重大市场事件</p>}
          </div>
          <div className="risk-section">
            <span className="risk-label">EARLY WARNINGS</span>
            {state?.risk_signals.length ? state.risk_signals.map((signal) => <article className="risk-item" key={signal.signal_id}><i>↗</i><div><strong>{EVENT_LABELS[signal.event_type] ?? signal.event_type}</strong><p>R{signal.target_round} · 概率 {percent(signal.estimated_probability_ppm, 0)} · {SEVERITY_LABELS[signal.severity]}</p></div></article>) : <p className="risk-empty">暂无未来风险预警</p>}
          </div>
          <div className="risk-section">
            <span className="risk-label">COMPANY INCIDENTS</span>
            {companies.some((company) => company.risk.active_incident) ? companies.filter((company) => company.risk.active_incident).map((company) => <article className="risk-item incident" key={company.company_id}><i style={{ background: metaFor(company.company_id).color }}>{metaFor(company.company_id).shortName}</i><div><strong>{EVENT_LABELS[company.risk.active_incident!.incident_type]}</strong><p>剩余 {company.risk.active_incident!.remaining_rounds} 轮 · 可主动维修</p></div></article>) : <p className="risk-empty">所有公司运营正常</p>}
          </div>
          <div className="index-grid"><span>市场情绪<strong>{state ? (state.market.market_sentiment_ppm / 1_000_000).toFixed(3) : "—"}</strong></span><span>供应成本指数<strong>{state ? (state.market.actual_supply_cost_index_ppm / 1_000_000).toFixed(3) : "—"}</strong></span></div>
        </aside>
      </section>

      <section className="action-studio">
        <div className="section-heading studio-heading">
          <div><span className="eyebrow">{gameMode === "single_company" ? "YOUR DECISION" : "NUMERIC JOINT ACTION"}</span><h2>Round {state?.terminal ? "—" : state?.round ?? 1} {gameMode === "single_company" ? `${metaFor(playerCompanyId).name}决策` : "连续动作配置"}</h2><p>{gameMode === "single_company" ? "提交后，三个规则对手会基于同一 State 同步行动；你无法在看到对手动作后反悔。" : "所有金额由后端以整数分校验；联合动作锁定后只执行一次。"}</p></div>
          <div className="studio-tools"><button className="text-button" type="button" onClick={() => applyPreset("low")}>低投入锚点</button><button className="text-button" type="button" onClick={() => applyPreset("medium")}>中投入锚点</button><button className="text-button" type="button" onClick={() => applyPreset("high")}>高投入锚点</button><button className="text-button" type="button" onClick={randomizeActions}>生成可行动作</button><label className="studio-seed"><select aria-label="市场模型" value={marketModel} onChange={(event) => void changeMarketModel(event.target.value as MarketModel)}><option value="random">随机市场</option>{Object.entries(marketModelOptions).map(([id, option]) => <option key={id} value={id}>{option.label}</option>)}</select><select aria-label="轮数" value={selectedRounds} onChange={(event) => void changeRounds(Number(event.target.value))}>{roundOptions.map((rounds) => <option key={rounds} value={rounds}>{rounds} 轮</option>)}</select><select aria-label="Seed 模式" value={seedMode} onChange={(event) => setSeedMode(event.target.value as SeedMode)}><option value="random">随机 Seed</option><option value="fixed">固定 Seed</option></select>{seedMode === "fixed" && <input aria-label="固定 Seed" type="number" min={0} max={4_294_967_295} value={fixedSeed} onChange={(event) => setFixedSeed(Math.max(0, Math.min(4_294_967_295, Number(event.target.value) || 0)))} />}<span>{state?.market.market_model_label ?? "—"} · {selectedRounds}轮 · Seed {activeSeed ?? "—"}</span></label><button className="text-button" type="button" onClick={() => void createEpisode(companyCount, gameMode, playerCompanyId)}>重新开始</button></div>
        </div>
        {state && !state.terminal && <PreviousRoundBrief state={state} companies={companies} settledMarket={history[history.length - 1]?.market} />}
        {gameMode === "single_company" && <div className="ai-opponent-note"><span>3 SEEDED RULE PROGRAMS</span><p>对手不使用 Agent；每局随机分配价值、溢价、增长或谨慎风格，再结合现金、份额、产能、预警和事故执行规则动作。</p></div>}
        <div className={`action-grid${gameMode === "single_company" ? " player-action-grid" : ""}`}>
          {state && editableCompanyIds.map((companyId) => {
            const meta = metaFor(companyId);
            const company = state.companies[companyId];
            const draft = drafts[companyId];
            const companyConstraints = constraints[companyId];
            if (!draft || !companyConstraints) return null;
            const incident = company.risk.active_incident;
            const fixedSpend = draft.advertising_budget_cents + draft.service_budget_cents + draft.capacity_investment_cents + draft.resilience_budget_cents + draft.repair_budget_cents;
            const averageListedPrice = Math.round(companies.reduce((sum, item) => sum + item.commercial.price_cents, 0) / Math.max(1, companies.length));
            return <article className="action-card numeric-card" key={companyId} style={{ "--company": meta.color } as React.CSSProperties}>
              <div className="action-card-head"><span className="company-badge">{meta.shortName}</span><div><strong>{meta.name}</strong><small>现金 {money(company.financial.cash_balance_cents)}</small></div><span className={`ready${fixedSpend > companyConstraints.cash_available_cents ? " invalid" : ""}`}>{fixedSpend > companyConstraints.cash_available_cents ? "OVER" : "VALID"}</span></div>
              <div className="decision-method"><strong>参数怎么用</strong><p>先用价格确定“份额还是毛利”的定位，再在现金约束内分配品牌、履约和风险投入。结果由竞争者动作与随机市场共同决定，不是确定性预测。</p></div>
              <RangeControl label="价格" value={draft.price_cents} bound={companyConstraints.bounds.price_cents} step={50} format={money} timing="当轮生效" impact="同时影响消费者选择、成交份额和每单毛利。" usage={`参考上一轮市场平均挂牌价 ${money(averageListedPrice)}：降低价格通常更利于争取订单，提高价格则需要品牌与服务支撑。`} onChange={(value) => updateDraft(companyId, "price_cents", value)} />
              <RangeControl label="广告投入" value={draft.advertising_budget_cents} bound={companyConstraints.bounds.advertising_budget_cents} step={100_000} format={money} timing="当轮 + 跨轮" impact="提高本轮消费者吸引力，并沉淀为后续品牌知名度。" usage="知名度或份额偏弱时可加大；注意广告先占用现金，不能保证在同一轮完全回收。" onChange={(value) => updateDraft(companyId, "advertising_budget_cents", value)} />
              <RangeControl label="服务投入" value={draft.service_budget_cents} bound={companyConstraints.bounds.service_budget_cents} step={100_000} format={money} timing="当轮 + 跨轮" impact="改善本轮选择吸引力，并影响服务质量与后续声誉。" usage="适合支持溢价、修复低声誉；若缺货持续存在，仅加服务投入不能替代产能。" onChange={(value) => updateDraft(companyId, "service_budget_cents", value)} />
              <RangeControl label="产能投资" value={draft.capacity_investment_cents} bound={companyConstraints.bounds.capacity_investment_cents} step={100_000} disabled={!companyConstraints.capacity_investment_enabled} format={money} timing="下一轮生效" impact="提高后续有效产能并形成期末产能资产，本轮先承担固定支出。" usage="利用率接近上限或频繁缺货时使用；最后一轮因来不及形成运营收益而禁用。" onChange={(value) => updateDraft(companyId, "capacity_investment_cents", value)} />
              <RangeControl label="韧性投入" value={draft.resilience_budget_cents} bound={companyConstraints.bounds.resilience_budget_cents} step={100_000} disabled={!companyConstraints.resilience_investment_enabled} format={money} timing="保护后续轮次" impact="提升风险准备，用于缓冲未来市场事件或公司事故的经营损失。" usage="出现未来风险预警时更有价值；不能抵消本轮已经激活的市场事件。" onChange={(value) => updateDraft(companyId, "resilience_budget_cents", value)} />
              {incident ? <div className="repair-control"><span><i>事故响应</i><strong>待修 {money(companyConstraints.max_useful_repair_budget_cents)}</strong></span><p>维修在本轮销售前执行：等待可保留现金但继续承受事故影响；部分或完全维修会立即占用现金并降低运营损失。</p><div>{(["wait", "partial_repair", "full_repair"] as const).map((mode) => <button type="button" key={mode} className={draft.incident_mode === mode ? "active" : ""} onClick={() => changeIncidentMode(companyId, mode)}>{mode === "wait" ? "等待" : mode === "partial_repair" ? "部分" : "完全"}</button>)}</div></div> : <div className="no-incident">✓ 当前无公司事故；若后续发生事故，这里会出现等待、部分维修和完全维修选项。</div>}
              <div className="spend-summary"><span>固定支出</span><strong>{money(fixedSpend)}</strong><small>占现金 {percent(Math.round((fixedSpend / Math.max(1, companyConstraints.cash_available_cents)) * 1_000_000), 0)}</small></div>
            </article>;
          })}
        </div>
        <div className="commit-bar">
          <div><span>{state ? `${gameMode === "single_company" ? "1 PLAYER ACTION" : `${companyIds.length}/${companyIds.length} ACTIONS`} READY · STATE V${state.state_version}` : "CONNECTING"}</span><small>{gameMode === "single_company" ? "提交时后端补全规则对手动作，再统一 Action Lock" : "后端将验证 Budget、Round、State Version 与 Action Idempotency"}</small></div>
          <button className="commit-button" type="button" onClick={() => void commitRound()} disabled={!state || loading || submitting || state.terminal}>
            <span>{state?.terminal ? "Episode 已完成" : submitting ? "后端结算中…" : `锁定并提交 Round ${state?.round ?? 1}`}</span><i>→</i>
          </button>
        </div>
      </section>

      <section className="analytics-grid">
        <ProfitChart history={history} companyIds={companyIds} />
        <div className="history-card">
          <div className="section-heading compact"><div><span className="eyebrow">STATE TRANSITIONS</span><h3>回合轨迹</h3></div><span className="event-count">{history.length} EVENTS</span></div>
          <div className="event-list">
            {history.length === 0 ? <div className="event-empty"><span>等待 State<sub>1</sub> → State<sub>2</sub></span><p>实现需求指进入消费者选择的全市场订单，并不等于成交销量。</p></div> : [...history].reverse().map((point) => {
              const pointCompanies = point.state.company_order.map((id) => point.state.companies[id]);
              const pointLeader = [...pointCompanies].sort((a, b) => b.commercial.market_share_ppm - a.commercial.market_share_ppm)[0];
              const totalSales = pointCompanies.reduce((sum, company) => sum + company.commercial.sales_orders, 0);
              return <article key={point.settledRound}><span className="event-round">R{String(point.settledRound).padStart(2, "0")}</span><div><strong>{metaFor(pointLeader.company_id).name} 领跑 · 全市场实现需求 {point.state.market.realized_demand_orders.toLocaleString("zh-CN")} 单</strong><p>实际成交 {totalSales.toLocaleString("zh-CN")} 单 · 未购买 {point.state.market.no_purchase_orders.toLocaleString("zh-CN")} 单 · 缺货流失 {point.state.market.lost_after_stockout_orders.toLocaleString("zh-CN")} 单 · 领先份额 {percent(pointLeader.commercial.market_share_ppm)}</p></div><span className="event-ok">VALID</span></article>;
            })}
          </div>
        </div>
      </section>

      <section className="statistics-suite">
        <div className="section-heading statistics-heading"><div><span className="eyebrow">PUBLIC MARKET DATA</span><h2>市场份额、销售额与统计数据</h2><p>所有表格使用每轮后端结算结果，不用前端估算替代真实状态。</p></div><span className="event-count">{history.length} / {maxRounds} ROUNDS</span></div>
        <div className="statistics-grid">
          <MarketShareHistory history={history} companyIds={companyIds} />
          <RevenueTable companies={companies} settledRound={settledRound} />
          <MarketStatisticsTable history={history} />
        </div>
      </section>

      {gameMode === "single_company" && retrospective && <MarketRetrospective retrospective={retrospective} />}

      <footer><span>FRESH MARKET LAB · ENGINEERING MVP V4</span><p>单公司分析与回溯只使用已记录状态和规则证据，不虚构严格因果。</p></footer>
    </main>
  );
}
