// 报告导出只包含页面已展示的汇总字段，不导出 Agent 私有观察、凭据或模型原始响应。
interface VisibleAgentSummary {
  companyId: string;
  companyName: string;
  persona: string;
  share: number;
  profit: number;
  resilience: number;
}

export function buildReportExport({ runtimeMode, agents }: {
  runtimeMode: "draft" | "demo" | "backend";
  agents: VisibleAgentSummary[];
}) {
  const payload = {
    export_version: "market-agents-visible-report-v1.0.0",
    runtime_mode: runtimeMode,
    evidence_boundary: runtimeMode === "demo"
      ? "DEMO_ONLY_NOT_RESEARCH_EVIDENCE"
      : runtimeMode === "backend"
        ? "SINGLE_EPISODE_DESCRIPTIVE_RESULT"
        : "NO_EPISODE",
    agents: agents.map((agent) => ({
      company_id: agent.companyId,
      company_name: agent.companyName,
      persona_label: agent.persona,
      market_share_percent: agent.share,
      round_profit_cents: agent.profit,
      resilience_percent: agent.resilience,
    })),
  };
  return {
    filename: `market-agents-${runtimeMode}-report-${new Date().toISOString().slice(0, 10)}.json`,
    content: `${JSON.stringify(payload, null, 2)}\n`,
  };
}

export function reportDownloadHref(content: string): string {
  return `data:application/json;charset=utf-8,${encodeURIComponent(content)}`;
}
