import assert from "node:assert/strict";
import test from "node:test";

import { buildSabmNodeFlow, SABM_NODE_DEFINITIONS } from "../app/sabm-node-flow.ts";

test("SABM visualization exposes the eight real backend nodes in order", () => {
  assert.deepEqual(
    SABM_NODE_DEFINITIONS.map((node) => node.key),
    [
      "load_snapshot",
      "build_context",
      "reflect_strategy",
      "generate_candidates",
      "validate / repair_decision",
      "prepare_intent",
      "submit_intent",
      "finalize",
    ],
  );
  assert.equal(SABM_NODE_DEFINITIONS[3].ai, true);
  assert.ok(SABM_NODE_DEFINITIONS.every((node) => node.stages.length > 0));
});

test("not-yet-run topology reports absence of execution instead of static evidence", () => {
  const flow = buildSabmNodeFlow({ runtimeMode: "backend", execution: null });

  assert.equal(flow.sourceLabel, "尚未运行");
  assert.equal(flow.traceConnected, false);
  assert.ok(flow.nodes.every((node) => node.status === "waiting"));
  assert.ok(flow.nodes.every((node) => node.detail === "尚未运行"));
});

test("backend topology renders actual execution events and trace metrics", () => {
  const flow = buildSabmNodeFlow({
    runtimeMode: "backend",
    execution: {
      company_id: "company_B",
      model_id: "nvidia/nemotron-3-super-120b-a12b:free",
      fallback_used: false,
      events: [
        { stage: "load_snapshot", details: {} },
        { stage: "build_context", details: {} },
        { stage: "reflect_strategy", details: {} },
        { stage: "generate_candidates", details: {} },
        { stage: "provider_request", details: { attempt: 1, model_id: "nvidia/nemotron-3-super-120b-a12b:free", repair: false } },
        { stage: "provider_response", details: { attempt: 1, status: "received", finish_reason: "stop", usage_available: true, total_tokens: 437, latency_ms: 1260 } },
        { stage: "validate", details: { repair_attempts: 0 } },
        { stage: "prepare_intent", details: {} },
        { stage: "submit_intent", details: {} },
        { stage: "finalize", details: { status: "accepted", repair_attempts: 0, total_tokens: 437, latency_ms: 1260 } },
      ],
      trace: {
        status: "accepted",
        repair_attempts: 0,
        provider_usage: { prompt_tokens: 311, completion_tokens: 126, total_tokens: 437 },
        latency_ms: 1260,
        provider_finish_reason: "stop",
        provider_error_category: null,
        error_code: null,
        selected_candidate_id: "balanced",
        selection_reason_codes: ["profit_guard", "share_growth"],
        validation_errors: [],
        memory_view: { history_limit: 2, recent_feedback: [], diagnostic_codes: ["first_round"] },
        strategy_reflection: { source: "deterministic", summary: "首轮保持价格纪律", adjustments: ["控制广告预算"] },
        prompt_audit: { system_prompt: "你是激进增长型市场公司。", user_prompt: "实现需求 12480，提交三项候选。" },
        candidates: [
          { candidate_id: "balanced", label: "稳健增长", action: { price_cents: 9700 }, evidence_paths: ["market.realized_demand_orders"], tradeoffs: ["利润与份额平衡"], expected_outcome: "份额小幅增长" },
          { candidate_id: "defensive", label: "防御", action: { price_cents: 10000 }, evidence_paths: [], tradeoffs: [], expected_outcome: "保持利润" },
          { candidate_id: "growth", label: "增长", action: { price_cents: 9500 }, evidence_paths: [], tradeoffs: [], expected_outcome: "扩大份额" },
        ],
        prepared_intent: { agent_id: "single-agent-company_B", action: { price_cents: 9700 }, rationale: "采用稳健增长候选" },
        intent_receipt: { intent_id: "intent-B-1", accepted: true },
      },
    },
  });

  assert.equal(flow.traceConnected, true);
  assert.ok(flow.nodes.every((node) => node.status === "done"));
  assert.match(flow.nodes[3].detail, /437/);
  assert.match(flow.nodes[3].detail, /"latency_ms": 1260/);
  assert.match(flow.nodes[4].detail, /"repair_attempts": 0/);
  assert.match(flow.nodes[0].detail, /history_limit/);
  assert.match(flow.nodes[1].detail, /实现需求 12480/);
  assert.match(flow.nodes[2].detail, /首轮保持价格纪律/);
  assert.match(flow.nodes[3].detail, /稳健增长/);
  assert.match(flow.nodes[3].detail, /profit_guard/);
  assert.match(flow.nodes[5].detail, /single-agent-company_B/);
  assert.match(flow.nodes[6].detail, /intent-B-1/);
  assert.doesNotMatch(JSON.stringify(flow), /OpenRouter/i);
});

test("failed execution marks the failing node and reports actual fallback", () => {
  const flow = buildSabmNodeFlow({
    runtimeMode: "backend",
    execution: {
      company_id: "company_C",
      model_id: "nvidia/nemotron-nano-9b-v2:free",
      fallback_used: true,
      events: [
        { stage: "load_snapshot", details: {} },
        { stage: "build_context", details: {} },
        { stage: "reflect_strategy", details: {} },
        { stage: "generate_candidates", details: {} },
        { stage: "provider_error", details: { attempt: 1, error_category: "provider_request_failed", latency_ms: 800 } },
        { stage: "repair_decision", details: { repair_attempts: 1 } },
        { stage: "provider_error", details: { attempt: 2, error_category: "provider_request_failed", latency_ms: 700 } },
        { stage: "finalize", details: { status: "no_intent", repair_attempts: 1, total_tokens: 0, latency_ms: 1500 } },
      ],
      trace: {
        status: "no_intent",
        repair_attempts: 1,
        provider_usage: { total_tokens: 0 },
        latency_ms: 1500,
        provider_finish_reason: null,
        provider_error_category: "provider_request_failed",
        error_code: "provider_failed",
        selected_candidate_id: null,
      },
    },
  });

  assert.equal(flow.traceConnected, true);
  assert.equal(flow.nodes[3].status, "error");
  assert.match(flow.boundary, /规则回退：是/);
  assert.match(flow.nodes[7].detail, /provider_failed/);
});

test("live provider request marks the generation node current with real waiting data", () => {
  const flow = buildSabmNodeFlow({
    runtimeMode: "backend",
    execution: {
      company_id: "company_B",
      model_id: "model-b",
      fallback_used: false,
      events: [
        { stage: "load_snapshot", details: {} },
        { stage: "build_context", details: {} },
        { stage: "reflect_strategy", details: {} },
        { stage: "generate_candidates", details: {} },
        { stage: "provider_request", details: { attempt: 1, repair: false } },
      ],
      live_progress: {
        company_id: "company_B",
        model_id: "model-b",
        status: "running",
        current_stage: "provider_request",
        events: [],
        started_at_ms: 1000,
        updated_at_ms: 2400,
        elapsed_ms: 1400,
        provider_attempts: 1,
        provider_waiting: true,
        total_tokens: null,
        provider_latency_ms: null,
        finish_reason: null,
        fallback_used: null,
        error_category: null,
      },
      trace: {
        status: "running",
        repair_attempts: 0,
        provider_usage: {},
        latency_ms: 0,
        provider_finish_reason: null,
        provider_error_category: null,
        error_code: null,
        selected_candidate_id: null,
        candidates: [],
      },
    },
  });

  assert.equal(flow.nodes[0].status, "done");
  assert.equal(flow.nodes[3].status, "current");
  assert.equal(flow.nodes[4].status, "waiting");
  assert.match(flow.nodes[3].detail, /等待结构化结果/);
  assert.match(flow.nodes[3].detail, /"provider_attempts": 1/);
  assert.equal(flow.sourceLabel, "实时运行中");
});

test("live provider response completes generation without inventing the next node", () => {
  const flow = buildSabmNodeFlow({
    runtimeMode: "backend",
    execution: {
      company_id: "company_C",
      model_id: "model-c",
      fallback_used: false,
      events: [
        { stage: "generate_candidates", details: {} },
        { stage: "provider_request", details: { attempt: 1 } },
        { stage: "provider_response", details: { total_tokens: 222, latency_ms: 900 } },
      ],
      live_progress: {
        company_id: "company_C",
        model_id: "model-c",
        status: "running",
        current_stage: "provider_response",
        events: [],
        started_at_ms: 1000,
        updated_at_ms: 1900,
        elapsed_ms: 900,
        provider_attempts: 1,
        provider_waiting: false,
        total_tokens: 222,
        provider_latency_ms: 900,
        finish_reason: "stop",
        fallback_used: null,
        error_category: null,
      },
      trace: {
        status: "running",
        repair_attempts: 0,
        provider_usage: { total_tokens: 222 },
        latency_ms: 900,
        provider_finish_reason: "stop",
        provider_error_category: null,
        error_code: null,
        selected_candidate_id: null,
        candidates: [],
      },
    },
  });

  assert.equal(flow.nodes[3].status, "done");
  assert.equal(flow.nodes[4].status, "waiting");
  assert.match(flow.nodes[3].detail, /222/);
});
