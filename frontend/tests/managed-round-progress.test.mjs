import assert from "node:assert/strict";
import test from "node:test";

import {
  pollManagedRoundProgress,
  progressToPartialExecutions,
} from "../app/managed-round-progress.ts";

function progress(status, company = {}) {
  return {
    episode_id: "ep-1",
    round: 1,
    state_version: 0,
    status,
    started_at_ms: 1_000,
    updated_at_ms: 1_100,
    elapsed_ms: 100,
    error_category: null,
    companies: {
      company_B: {
        company_id: "company_B",
        model_id: "model-b",
        status: "running",
        current_stage: "provider_request",
        events: [
          { stage: "load_snapshot", details: {}, occurred_at_ms: 1_010 },
          { stage: "provider_request", details: { attempt: 1, repair: false }, occurred_at_ms: 1_020 },
        ],
        started_at_ms: 1_010,
        updated_at_ms: 1_100,
        elapsed_ms: 90,
        provider_attempts: 1,
        provider_waiting: true,
        total_tokens: null,
        provider_latency_ms: null,
        finish_reason: null,
        fallback_used: null,
        error_category: null,
        ...company,
      },
    },
  };
}

test("polling publishes real snapshots and stops at a terminal status", async () => {
  const snapshots = [progress("running"), progress("settling"), progress("completed")];
  const seen = [];
  await pollManagedRoundProgress({
    fetchProgress: async () => snapshots.shift(),
    onProgress: (value) => seen.push(value.status),
    intervalMs: 1,
  });
  assert.deepEqual(seen, ["running", "settling", "completed"]);
});

test("polling reports a temporary network failure and continues", async () => {
  let calls = 0;
  let unavailable = 0;
  const seen = [];
  await pollManagedRoundProgress({
    fetchProgress: async () => {
      calls += 1;
      if (calls === 1) throw new Error("temporary network error");
      return progress("completed");
    },
    onProgress: (value) => seen.push(value.status),
    onUnavailable: () => { unavailable += 1; },
    intervalMs: 1,
  });
  assert.equal(unavailable, 1);
  assert.deepEqual(seen, ["completed"]);
});

test("aborted polling performs no request", async () => {
  const controller = new AbortController();
  controller.abort();
  let calls = 0;
  await pollManagedRoundProgress({
    fetchProgress: async () => { calls += 1; return progress("running"); },
    onProgress: () => {},
    signal: controller.signal,
    intervalMs: 1,
  });
  assert.equal(calls, 0);
});

test("progress maps only real events into partial executions", () => {
  const executions = progressToPartialExecutions(progress("running"), {
    company_B: "configured-model-b",
  });
  assert.deepEqual(executions.company_B.events.map((event) => event.stage), ["load_snapshot", "provider_request"]);
  assert.equal(executions.company_B.model_id, "model-b");
  assert.equal(executions.company_B.trace.status, "running");
  assert.equal(executions.company_B.live_progress.provider_waiting, true);
  assert.equal(executions.company_B.trace.candidates.length, 0);
});
