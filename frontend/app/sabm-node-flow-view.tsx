import { buildSabmNodeFlow, SabmExecution, SabmRuntimeMode } from "./sabm-node-flow.ts";

export function SabmNodeFlow({ runtimeMode, companyName, driverLabel, execution }: {
  runtimeMode: SabmRuntimeMode;
  companyName: string;
  driverLabel: string;
  execution?: SabmExecution | null;
}) {
  const flow = buildSabmNodeFlow({ runtimeMode, execution });

  return <section className="card sabm-flow" aria-label="PersonaAgent SABM 节点流">
    <header className="sabm-flow-head">
      <div><span>PersonaAgent · SABM</span><h3>AI 后端节点流</h3><p>{companyName} · 当前驱动 {driverLabel} · 纵向展示本轮真实节点事件与审计指标</p></div>
      <b className={execution?.trace.status === "running" ? "current" : execution ? "done" : "waiting"}>{flow.sourceLabel}</b>
    </header>
    <div className="sabm-node-grid">
      {flow.nodes.map((node, index) => {
        return <div className="sabm-node-wrap" key={node.key}>
          <article className={`sabm-node ${node.status}${node.ai ? " ai" : ""}`}>
            <div className="sabm-node-title"><i>{node.status === "done" ? "✓" : node.number}</i><span><strong>{node.label}{node.ai ? <em>AI</em> : null}</strong><small>{node.key}</small></span><b>{node.status === "done" ? "已执行" : node.status === "current" ? "执行中" : node.status === "error" ? "错误" : "尚未运行"}</b></div>
            <p>{node.summary}</p>
            <pre>{node.detail}</pre>
          </article>
          {index < flow.nodes.length - 1 ? <span className="sabm-connector" aria-hidden="true">↓</span> : null}
        </div>;
      })}
    </div>
    <p className="sabm-boundary"><b>运行证据</b>{flow.boundary}</p>
  </section>;
}
