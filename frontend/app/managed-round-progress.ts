// 真实管理回合的轮询与临时 execution 投影；不得生成模型未返回的内容。
import type { SabmExecution } from "./sabm-node-flow.ts";

export type ManagedRoundProgressStatus = "idle" | "running" | "settling" | "completed" | "failed";
export type ManagedCompanyProgressStatus = "queued" | "running" | "completed" | "fallback" | "failed";

export interface ManagedProgressEvent {
  stage: string;
  details: Record<string, unknown>;
  occurred_at_ms: number;
}

export interface ManagedCompanyProgress {
  company_id: string;
  model_id: string;
  status: ManagedCompanyProgressStatus;
  current_stage: string | null;
  events: ManagedProgressEvent[];
  started_at_ms: number | null;
  updated_at_ms: number;
  elapsed_ms: number;
  provider_attempts: number;
  provider_waiting: boolean;
  total_tokens: number | null;
  provider_latency_ms: number | null;
  finish_reason: string | null;
  fallback_used: boolean | null;
  error_category: string | null;
}

export interface ManagedRoundProgress {
  episode_id: string;
  round: number | null;
  state_version: number | null;
  status: ManagedRoundProgressStatus;
  started_at_ms: number | null;
  updated_at_ms: number | null;
  elapsed_ms: number;
  companies: Record<string, ManagedCompanyProgress>;
  error_category: string | null;
}

export async function pollManagedRoundProgress({
  fetchProgress,
  onProgress,
  onUnavailable,
  signal,
  intervalMs = 500,
}: {
  fetchProgress: (signal?: AbortSignal) => Promise<ManagedRoundProgress>;
  onProgress: (progress: ManagedRoundProgress) => void;
  onUnavailable?: () => void;
  signal?: AbortSignal;
  intervalMs?: number;
}): Promise<void> {
  while (!signal?.aborted) {
    try {
      const progress = await fetchProgress(signal);
      onProgress(progress);
      if (progress.status === "completed" || progress.status === "failed") return;
    } catch (error) {
      if (signal?.aborted || (error instanceof DOMException && error.name === "AbortError")) return;
      onUnavailable?.();
    }
    await waitForNextPoll(intervalMs, signal);
  }
}

export function progressToPartialExecutions(
  progress: ManagedRoundProgress | null,
  modelByCompany: Record<string, string>,
): Record<string, SabmExecution> {
  if (!progress) return {};
  return Object.fromEntries(Object.entries(progress.companies).map(([companyId, company]) => [
    companyId,
    {
      company_id: companyId,
      model_id: company.model_id || modelByCompany[companyId] || "",
      fallback_used: company.fallback_used === true,
      events: company.events.map((event) => ({ stage: event.stage, details: event.details })),
      live_progress: company,
      trace: {
        status: company.status === "completed" ? "accepted" : company.status === "fallback" ? "no_intent" : "running",
        repair_attempts: 0,
        provider_usage: company.total_tokens === null ? {} : { total_tokens: company.total_tokens },
        latency_ms: company.provider_latency_ms ?? 0,
        provider_finish_reason: company.finish_reason,
        provider_error_category: company.error_category,
        error_code: company.error_category,
        selected_candidate_id: null,
        provider_usage_available: company.total_tokens !== null,
        selection_reason_codes: [],
        validation_errors: [],
        memory_view: null,
        strategy_reflection: null,
        prompt_audit: null,
        candidates: [],
        prepared_intent: null,
        intent_receipt: null,
      },
    },
  ]));
}

function waitForNextPoll(milliseconds: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.resolve();
  return new Promise((resolve) => {
    const timeout = setTimeout(done, milliseconds);
    signal?.addEventListener("abort", done, { once: true });
    function done() {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", done);
      resolve();
    }
  });
}
