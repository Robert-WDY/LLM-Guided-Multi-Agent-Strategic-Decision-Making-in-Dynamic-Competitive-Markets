import assert from "node:assert/strict";
import test from "node:test";

import { buildReportExport, reportDownloadHref } from "../app/report-export.ts";

test("report export contains visible summaries and an explicit evidence boundary", () => {
  const result = buildReportExport({
    runtimeMode: "demo",
    agents: [{ companyId: "company_A", companyName: "青禾市场", persona: "稳健经营", share: 26.5, profit: 123400, resilience: 71 }],
  });
  const payload = JSON.parse(result.content);

  assert.match(result.filename, /^market-agents-demo-report-/);
  assert.equal(payload.evidence_boundary, "DEMO_ONLY_NOT_RESEARCH_EVIDENCE");
  assert.deepEqual(payload.agents[0], {
    company_id: "company_A",
    company_name: "青禾市场",
    persona_label: "稳健经营",
    market_share_percent: 26.5,
    round_profit_cents: 123400,
    resilience_percent: 71,
  });
  assert.doesNotMatch(result.content, /api[_-]?key|hidden_reasoning/i);
  assert.match(reportDownloadHref(result.content), /^data:application\/json;charset=utf-8,/);
});
