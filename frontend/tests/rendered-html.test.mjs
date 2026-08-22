import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the multi-agent game research dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>多智能体博弈实验室/);
  assert.match(html, /多智能体博弈实验室/);
  assert.match(html, /多智能体博弈实验平台/);
  assert.match(html, /从一个清楚的入口/);
  assert.match(html, /个人体验/);
  assert.match(html, /观察实验/);
  assert.match(html, /研究控制台/);
  assert.match(html, /配置并进入/);
  assert.doesNotMatch(html, /当前模式导航/);
  assert.doesNotMatch(html, /实验报告 REPORT/);
  assert.doesNotMatch(html, /Chain of Thought.*真实推理过程/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/);
});
