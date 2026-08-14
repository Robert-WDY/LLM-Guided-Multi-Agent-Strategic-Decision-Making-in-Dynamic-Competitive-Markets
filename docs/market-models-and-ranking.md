# 市场模型、终局榜单与回溯协议

## 1. 创建 Episode 时选择市场模型

`POST /api/episodes` 增加 `market_model`：

```json
{
  "episode_seed": 42,
  "company_ids": ["company_A", "company_B", "company_C", "company_D"],
  "game_mode": "single_company",
  "player_company_id": "company_A",
  "market_model": "random",
  "max_rounds": 10
}
```

允许值：

| 值 | 含义 |
| --- | --- |
| `random` | 根据 Seed 从下列模型中选择，并在需求偏差 ±2.5%、价格锚点 ±2.5 元内扰动 |
| `balanced` | 价格、品牌和服务共同影响选择 |
| `value_oriented` | 价格敏感消费者占 60%，需求略高，合理价格带更窄 |
| `quality_oriented` | 品牌敏感消费者占 48%；知名度与历史声誉是主要品质信任信号，广告/品牌积累的需求回报更强 |
| `service_oriented` | 服务敏感消费者占 45%；当期服务质量、履约稳定性和历史缺货惩罚更强，单纯品牌曝光作用较弱 |

实际选中的模型通过 `state.market` 公开：

- `market_model_id`、`market_model_label`、`market_model_description`；
- `demand_bias_ppm`；
- `price_anchor_cents` 与 `price_band_cents`。
- `utility_*_multiplier_ppm`：价格、知名度、服务、声誉和历史缺货在该模型中的效用倍率。

品质与服务不再只靠消费者分群比例区分。消费者效用还会乘入模型专属因子：品质市场强化 `awareness/reputation`，弱化当期 `service`；服务市场强化 `service/prior_stockout`，弱化 `awareness`。在同 Seed、同公司状态的配对动作中，品牌集中投入应在品质市场获得更多潜在订单，而服务集中投入应在服务市场获得更多潜在订单。该方向由自动化测试固定。

可选轮数为 `5 / 10 / 15 / 20`。终局发生在第 `max_rounds` 轮结算后，状态中的协议回合为 `max_rounds+1`；风险信号生成、末轮长期投资禁用、回溯折线长度都随该值变化。

规则对手每轮基于市场价格锚点、上一轮实际成交均价、单位成本和逐单履约成本重新报价，并限制在公开合理价格带附近。该规则允许成本冲击下的提价，但不允许上一轮报价自动成为下一轮硬下限。

## 2. 同轮市场快照

Step 响应中的两个市场对象有不同职责：

- `state.market`：进入下一决策轮时的当前公开条件；
- `settled_market`：刚结算回合的实现需求、未购买、缺货流失、成交均价、情绪和供应成本。

Agent Gateway 的 `public_history[].market` 与 `settled_market` 同口径。`actual_supply_cost_index_ppm` 只乘入 `active_events_during_round` 中列出的事件，不会提前包含下一轮刚兑现的事件。

## 3. 终局榜单

回溯和 Agent `terminal_summary.rankings` 同时返回两张榜：

### 综合价值榜 `composite`

```text
综合价值 = 现金
         + 产能账面价值 × 产能残值率
         + 品牌知名度价值
         + 服务价值
         + 声誉价值
         + 韧性价值
```

该榜是默认比赛结果，体现经营现金与长期能力的合计。

### 总资产榜 `total_assets`

```text
总资产 = 现金 + 产能账面价值
```

该榜只显示可确认的财务和产能账面资产，不包含品牌、服务、声誉与韧性的估值。

每个排名条目包含 `value_cents` 和 `breakdown`，可直接解释两张榜出现不同名次的原因。

## 4. 回溯解释与折线数据

`GET /api/episodes/{episode_id}/retrospective` 在 Episode 完成后返回：

- `rankings`：两张终局榜；
- `component_comparison`：玩家与综合榜冠军在现金、产能残值、品牌、服务、声誉和韧性上的差距；
- `rank_explanation`：根据上述差距生成的主要排名原因；
- `trend_series`：每家公司 R0～R`max_rounds` 的综合价值、总资产、现金、累计利润和份额；
- `rounds[].enterprise_value_delta_cents`：该回合综合价值变化；
- `rounds[].market_*`：与该回合事件严格对齐的市场结果；
- `turning_point_rounds`：按综合价值变化、份额变化和市场事件选出的三个关键回合。

回溯只对规则中可观测的传导关系做解释。例如“相对低价与份额上升同时出现”属于机制证据，不会被表述为排除其他变量后的严格因果结论。
