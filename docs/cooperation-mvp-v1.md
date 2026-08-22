# Cooperation MVP v1：Shared Resilience Contribution

## 范围

本阶段只加入一个公共品机制：`shared_resilience_v1`。它允许公司为行业共享韧性支付真实成本，形成下一轮所有公司都能使用的公共保护。

本阶段没有联合定价、共享物流、转账、联盟、产能共享或完整社会福利系统。承诺始终 `binding=false`，不会替 Agent 执行动作。

有通信合作条件使用：

```json
{
  "communication_mode": "public_private",
  "cooperation_mode": "shared_resilience_v1"
}
```

默认 `cooperation_mode=off`，旧状态与旧动作的序列化形状保持不变。

因果基线允许 `communication_mode=off` 与 `cooperation_mode=shared_resilience_v1` 同时使用：Agent 可以贡献，但不能发起 Proposal 或生成新 Commitment。`public_only` 不支持本合作协议，因为 Proposal 必须是一对一私信。

## 每轮流程

现有通信协议每轮只有一个同步波次，所以 v1 使用跨轮回应：

```text
Round t Observation
  -> private Proposal（target_round > t）
  -> Communication Close
  -> Private Decision
  -> Actual Contribution
  -> Joint Action / Market Update

Round t+1 Observation
  -> private Acceptance / Rejection
  -> Communication Close
  -> non-binding Commitment
  -> Private Decision
  -> Actual Contribution
  -> Shared Resilience Update
  -> Fulfillment / Betrayal Verification
  -> Credibility Update
```

同一波次回应本轮新提议会被拒绝。自由文本中的“同意”不会生成 Commitment；必须使用结构化 `cooperation_response`。

## 数据与权限

- `cooperation_proposal`：只能出现在一对一私密 `proposal` 消息中；Controller 生成可信 `proposal_id`、发送者、轮次和状态绑定。
- `cooperation_response`：只能由提议接收者在后续轮次私密回复原提议者。
- `Commitment`：只由合法 `accept` 生成，且永远非绑定。
- `shared_resilience_contribution_cents`：唯一会真实扣款的合作动作。
- `CommitmentVerification`：按最终执行动作计算 `fulfilled / partial_betrayal / betrayed`。
- `CredibilityRecord`：按承诺金额加权更新；新公司以 50% 中性先验开始。

提议、回应和承诺只对参与双方可见；其他公司看不到对应消息 ID、数量或内容。经核验后的公司可信度是公共信息。

## 市场公式

贡献是当期固定支出。总贡献更新下一轮行业韧性：

```text
next_industry_resilience
  = retention × current_industry_resilience
  + input_weight × saturation(total_contribution)
```

下一轮公共保护与公司自身韧性合并，用于降低：

- 市场事件造成的供应成本上升；
- 产能、广告、服务和声誉冲击；
- 新公司事故的发生概率和严重程度。

公共保护对所有公司相同，因此未贡献者也会受益，形成真实搭便车动机。无新增贡献时，已有行业韧性按 retention 衰减。

## 三层 Replay

- Economic Replay：用最终联合动作重放市场，验证状态 Hash。
- Interaction Replay：重建公开/私密消息可见性和每个 Agent 的通信输入。
- Cooperation Replay：从关闭消息和最终动作重新生成 Proposal、Response、Commitment、履约状态和 Credibility。

LLM 本身不会重新调用；可回放的是它看到的输入、记录的输出和 Controller 的确定性处理。

## 当前验收

运行：

```powershell
$env:PYTHONPATH='src'
python -m game_theory_agent.experiments.cooperation_acceptance `
  --output runs/cooperation-mvp-v1-acceptance-20260821-v3 `
  --seed 20260821
```

该验收执行 4 Mock × 5 轮：A 在第 1 轮向 B 私密提议第 2 轮贡献 100 万分，B 接受但实际只贡献 30 万分。硬门槛包括：

- 私密提议、接受和承诺归属正确；
- Commitment Close 前后市场 Hash 不变；
- B 最终贡献归属为 30 万分；
- 履约率为 30%，状态为 `partial_betrayal`；
- 三类 Replay 全部通过；
- JSONL 落盘后重新读取与内存事件完全一致。

市场机制另由确定性测试覆盖：All Cooperate 当期支付成本并建立公共韧性、One Free-rides 节省私人成本但获得公共保护、All Defect 不建立新韧性且已有韧性衰减；相同灾难下有公共韧性的市场损失更低。

真实 LLM 的 2 LLM + 2 Rule、10 Seeds × 10 Rounds 配对实验已于 2026-08-21 完成；30/30 Episode 通过三类 Replay，总 Token 为 17,985,985。入口为：

```powershell
$env:PYTHONPATH='src'
python -m game_theory_agent.experiments.cooperation_real_multiseed `
  --provider deepseek `
  --model <固定模型版本> `
  --seeds 101,102,103,104,105,106,107,108,109,110 `
  --rounds 10 `
  --output runs/cooperation-real-multiseed
```

三组条件使用相同 Seeds：`action_only`（无通信但可贡献）、`communication_no_history`（通信与贡献，历史可信度中性化）、`communication_with_history`（通信、贡献与完整承诺历史）。统计单位是配对 Seed；每个 Episode 都强制三类 Replay 通过，利润或贡献方向不作为工程通过条件。结果位于 `runs/cooperation-real-multiseed-20260821/summary.json`。

固定状态消息反事实入口会轮换调用顺序并重复真实模型调用：

```powershell
$env:PYTHONPATH='src'
python -m game_theory_agent.experiments.real_cooperation_counterfactual `
  --provider deepseek `
  --model <固定模型版本> `
  --repetitions 10 `
  --output runs/real-cooperation-counterfactual
```

它固定 State、Seed、Persona 和动作护栏，只改变 No Message、Cooperation Proposal、Defection Statement、High-credibility Proposal 与 Low-credibility Proposal，报告贡献分布、目标金额对齐率和决策阶段消息 disposition。协议级 ProposalResponse 接受率应以 30-Episode 配对实验中的真实结构化 Response 为准；不能用一次调用或普通文本 disposition 替代。

## Cooperation Research v2：P0 修复状态

2026-08-21 已开始修复三个研究缺口，当前完成第一批底层能力：

- 每轮用相同状态、动作和组件随机源运行“继承公共韧性归零”的只读影子结算；
- `CooperativeBenefitAttribution` 按公司记录实际利润、无公共保护利润、利润差、避免损失、机会成本、上一轮贡献来源、Free Rider 免费收益和 Individual Cooperative ROI；
- 当前轮贡献与上一轮保护来源分开记录，当前贡献仍只从下一轮开始产生保护；
- `Cooperation Replay` v1.1 必须使用相同市场环境重新运行影子结算，不能只相信日志中的收益数字；
- 公司级 `cooperation_memory` 从权威 Ledger 按对手派生，包含提议、接受、承诺、履约、背离、金额和可信度；
- `cooperation_history_mode=none` 同时清空详细历史、聚合记忆并将可信度中性化；
- Mock 通信策略可在同一 Proposal 下根据提议者可信度阈值确定性接受或拒绝；
- 合作指标新增公共收益、Free Rider Advantage、净合作现金流、Individual ROI、Free Rider 与贡献者当期利润差和避免事故数。

新的确定性验收产物位于：

- `runs/cooperation-research-v2-p0-acceptance-20260821-v2/summary.json`
- `runs/cooperation-research-v2-p0-acceptance-20260821-v2/round-events.jsonl`

该 5 轮验收已通过三类 Replay，并重建 30% 部分履约、公司级收益归因和第三轮对手合作记忆。它记录到 146,064 分公共保护收益全部由未支付上一轮贡献的公司获得，而上一轮贡献者本样本 ROI 为 0；这是当前 Seed 的真实机制结果，说明“存在公共收益”不等于“贡献者私人回报已经足够”。

长期 Free Rider 机制矩阵也已完成。`runs/cooperation-research-v2-free-rider-matrix-20260821/summary.json` 使用 5 个共同 Seed，在固定 15,000 分价格、每轮高供应冲击下分别运行 All Cooperate、One Free Rider 和 All Defect 各 10 轮。所有工程门槛通过：Free Rider 首轮精确节省 500,000 分、仍获得平均 28,980,335.6 分公共保护收益；All Defect 相对 All Cooperate 平均多损失 9,414 个缺货订单，市场累计利润合计低 116,658,503.4 分。A 相对同组贡献者平均多赚 6,806,061.4 分，但相对 A 自己参与 All Cooperate 的长期利润平均少 290,362.4 分，说明短期搭便车优势与长期公共品损害同时存在。

尚未完成的 v2 内容包括：One Betrayer 的多轮共同 Seed 与公司位置轮换矩阵、真实模型高/中/低可信度重复实验，以及合作专用人格的结构化效用语义与消融。
