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

test("server-renders the market laboratory", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Fresh Market Lab/);
  assert.match(html, /FRESH MARKET LAB/);
  assert.match(html, /MARKET_ENV_V4/);
  assert.match(html, /单公司经营/);
  assert.match(html, /市场全景/);
  assert.match(html, /规则对手/);
  assert.match(html, /单公司经营控制台/);
  assert.match(html, /公司状态/);
  assert.match(html, /公开市场情报/);
  assert.match(html, /资源配置/);
  assert.match(html, /实现需求/);
  assert.match(html, /成交总量/);
  assert.match(html, /市场模型在线|正在连接/);
  assert.doesNotMatch(html, /MVP_MARKET_V1|Attractiveness → Softmax/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/);
});
