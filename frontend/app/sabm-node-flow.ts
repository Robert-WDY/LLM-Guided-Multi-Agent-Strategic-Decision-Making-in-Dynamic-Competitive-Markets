// 纵向布局参考本地 ~member-2-v1（固定快照 7cbace1）；节点详情只呈现本次运行事件。
import type { ManagedCompanyProgress } from "./managed-round-progress.ts";
export type SabmRuntimeMode = "draft" | "demo" | "backend";
export type SabmNodeStatus = "waiting" | "current" | "done" | "error";

export const SABM_NODE_DEFINITIONS = [
  { key: "load_snapshot", label: "读取快照", stages: ["load_snapshot"] },
  { key: "build_context", label: "构建上下文", stages: ["build_context"] },
  { key: "reflect_strategy", label: "策略反思", stages: ["reflect_strategy"] },
  { key: "generate_candidates", label: "生成候选", stages: ["generate_candidates", "provider_request", "provider_response", "provider_error"], ai: true },
  { key: "validate / repair_decision", label: "校验 / 修复", stages: ["validate", "repair_decision"] },
  { key: "prepare_intent", label: "准备意图", stages: ["prepare_intent"] },
  { key: "submit_intent", label: "提交意图", stages: ["submit_intent"] },
  { key: "finalize", label: "完成审计", stages: ["finalize"] },
] as const;

export interface SabmExecution {
  company_id: string;
  model_id: string;
  fallback_used: boolean;
  events: Array<{ stage: string; details: Record<string, unknown> }>;
  live_progress?: ManagedCompanyProgress;
  trace: {
    status: string;
    repair_attempts: number;
    provider_usage: Record<string, number>;
    latency_ms: number;
    provider_finish_reason: string | null;
    provider_error_category: string | null;
    error_code: string | null;
    selected_candidate_id: string | null;
    provider_usage_available?: boolean;
    selection_reason_codes?: string[];
    validation_errors?: string[];
    memory_view?: Record<string, unknown> | null;
    strategy_reflection?: Record<string, unknown> | null;
    prompt_audit?: { system_prompt?: string; user_prompt?: string } | null;
    candidates?: Array<Record<string, unknown>>;
    prepared_intent?: Record<string, unknown> | null;
    intent_receipt?: Record<string, unknown> | null;
  };
}

export interface SabmNodeView {
  key: string;
  number: string;
  label: string;
  summary: string;
  detail: string;
  status: SabmNodeStatus;
  ai: boolean;
}

export function buildSabmNodeFlow({ runtimeMode, execution = null }: {
  runtimeMode: SabmRuntimeMode;
  execution?: SabmExecution | null;
}) {
  const seenStages = new Set(execution?.events.map((event) => event.stage) ?? []);
  const running = execution?.trace.status === "running";
  const currentStage = execution?.live_progress?.current_stage ?? execution?.events.at(-1)?.stage ?? null;
  const currentNodeIndex = SABM_NODE_DEFINITIONS.findIndex((node) => currentStage !== null && node.stages.includes(currentStage as never));
  const latestProviderEvent = execution?.events.findLast((event) => event.stage.startsWith("provider_"));
  const providerFailed = latestProviderEvent?.stage === "provider_error"
    && execution?.trace.status !== "accepted";
  const nodes: SabmNodeView[] = SABM_NODE_DEFINITIONS.map((node, index) => {
    const ran = node.stages.some((stage) => seenStages.has(stage));
    const status: SabmNodeStatus = node.key === "generate_candidates" && providerFailed
      ? "error"
      : !ran ? "waiting"
      : running && index === currentNodeIndex && !["provider_response", "finalize"].includes(currentStage ?? "")
        ? "current"
        : "done";
    return {
      key: node.key,
      number: String(index + 1).padStart(2, "0"),
      label: node.label,
      summary: ran ? "本轮真实事件已记录" : "等待本轮真实事件",
      detail: execution ? nodeDetail(node.key, node.stages, execution) : "尚未运行",
      status,
      ai: "ai" in node && node.ai === true,
    };
  });

  return {
    nodes,
    traceConnected: Boolean(execution),
    contextLabel: execution ? `AI 决策节点流 · ${execution.company_id} · ${execution.model_id}` : "AI 决策节点流",
    sourceLabel: running ? "实时运行中" : execution ? "真实运行数据" : runtimeMode === "draft" ? "等待创建实验" : "尚未运行",
    boundary: execution
      ? running
        ? `状态：运行中 · 当前节点：${currentStage ?? "等待首个事件"} · 数据来自本轮后端实时事件。`
        : `状态：${execution.trace.status} · 规则回退：${execution.fallback_used ? "是" : "否"} · 数据来自本轮后端事件与终态 trace。`
      : "当前公司尚无真实决策事件；所有节点保持未运行，不显示示例值。",
  };
}

function nodeDetail(key: string, stages: readonly string[], execution: SabmExecution): string {
  const events = execution.events.filter((event) => stages.includes(event.stage));
  if (!events.length) return "尚未运行";
  if (execution.trace.status === "running") {
    const live = execution.live_progress;
    if (key === "generate_candidates") {
      return traceJson({
        status: live?.provider_waiting ? "模型已接收请求，等待结构化结果" : "模型调用事件已返回",
        provider_attempts: live?.provider_attempts ?? events.filter((event) => event.stage === "provider_request").length,
        elapsed_ms: live?.elapsed_ms ?? null,
        total_tokens: live?.total_tokens ?? null,
        provider_latency_ms: live?.provider_latency_ms ?? null,
        finish_reason: live?.finish_reason ?? null,
        error_category: live?.error_category ?? null,
        events,
      });
    }
    return traceJson({
      current_stage: live?.current_stage ?? events.at(-1)?.stage ?? null,
      elapsed_ms: live?.elapsed_ms ?? null,
      events,
    });
  }
  if (key === "load_snapshot") {
    return traceJson(execution.trace.memory_view, "本轮未生成记忆视图");
  }
  if (key === "build_context") {
    return execution.trace.prompt_audit?.user_prompt ?? "本轮未生成实际决策输入";
  }
  if (key === "reflect_strategy") {
    return traceJson(execution.trace.strategy_reflection, "本轮未生成策略反思");
  }
  if (key === "generate_candidates") {
    const requests = events.filter((event) => event.stage === "provider_request").length;
    return traceJson({
      model: execution.model_id,
      provider_requests: requests,
      system_prompt: execution.trace.prompt_audit?.system_prompt ?? null,
      user_prompt: execution.trace.prompt_audit?.user_prompt ?? null,
      candidates: execution.trace.candidates ?? [],
      selected_candidate_id: execution.trace.selected_candidate_id,
      selection_reason_codes: execution.trace.selection_reason_codes ?? [],
      provider_usage: execution.trace.provider_usage_available === false ? "unavailable" : execution.trace.provider_usage,
      finish_reason: execution.trace.provider_finish_reason,
      latency_ms: execution.trace.latency_ms,
      provider_error_category: execution.trace.provider_error_category,
    });
  }
  if (key === "validate / repair_decision") {
    return traceJson({
      validation_events: events.map((event) => event.stage),
      validation_errors: execution.trace.validation_errors ?? [],
      repair_attempts: execution.trace.repair_attempts,
      provider_error_category: execution.trace.provider_error_category,
    });
  }
  if (key === "prepare_intent") {
    return traceJson(execution.trace.prepared_intent, "本轮未生成待提交意图");
  }
  if (key === "submit_intent") {
    return traceJson(execution.trace.intent_receipt, "本轮未生成提交回执");
  }
  if (key === "finalize") {
    return traceJson({
      status: execution.trace.status,
      selected_candidate_id: execution.trace.selected_candidate_id,
      error_code: execution.trace.error_code,
      fallback_used: execution.fallback_used,
    });
  }
  return `事件：${events.map((event) => event.stage).join(" → ")}\n状态：已执行`;
}

function traceJson(value: unknown, empty = "未生成"): string {
  return value === undefined || value === null ? empty : JSON.stringify(value, null, 2);
}
