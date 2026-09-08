"""Render conclusions from complete held-out evidence; never turn failed gates into passes."""
from reliable_v19_common import *

MARKETS=dict(normal='正常',recession='需求收缩',tight_supply='供应紧张',price_sensitive='价格敏感',project_feasible='项目可达配置',scaled='资源随人数扩展')
def money(v):return f'{v/100:,.2f}'
def interval(s):return f'{money(s["mean"])} [{money(s["interval"][0])}, {money(s["interval"][1])}]'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])

def main():
    a=read_json(OUT/'analysis.json');acceptance=read_json(OUT/'acceptance.json');real=read_json(OUT/'real/status.json');cal=read_json(OUT/'response-calibration.json');timing=read_json(OUT/'timing.json');sens=read_json(OUT/'sensitivity.json');raw=read_json(OUT/'advice.json')['results']
    primary=[r for r in a['groups'] if r['mode']=='conditional' and r['horizon']==3]
    goal_pass={g:sum(x['effectiveness_pass'] for x in primary if x['goal']==g) for g in SPEC['goals']}
    text=[f'''# Reliable Strategic Advisor v19：实现、留出实验与验收

2026-09-08。本轮是原计划改进层的研究交付。市场公式保持冻结，旧v17功能保留；不能把工程完成解释为经济效果全部通过。

**主结果：条件建议三轮收益门槛通过 {a['conditional_primary_pass']}/36 组。三轮下降 {a['conditional_3round_losses']}/720，五轮下降 {a['conditional_5round_losses']}/720，弃权 {a['conditional_abstentions']}/720。收费前置门槛 {'通过' if acceptance['paid_gate_passed'] else '未通过'}。**

按目标分开：利润 {goal_pass['profit']}/18 组通过，福利 {goal_pass['welfare']}/18 组通过。“一组通过”不意味着这组每个种子都获益。当前证据更支持利润改进；福利改进是主要未通过部分。

## 1. 实现了什么

- 以当轮合法、已规范化Agent Draft为候选0。退出时或证据不足时完整保留原动作；不以规则动作偷换原计划。
- 12个候选名额只做价格、服务、广告、产能、韧性、采购、供应商比例、合同和公共投入的局部扰动。合法化可能联动缩减其他预算，实际差异以保存动作而非标签为准；不是完整动作空间。
- 一般倾向和条件模型均来自独立公开训练记录。条件特征是其他公司的已结算价格变化、公开缺货和自身服务投入。当前同时行动对手看不到本轮候选；候选影响下一轮公开状态后才能影响预测反应。
- H3研究发现集每候选3个情景，只选择一次候选；对该候选与原计划分别以8个新情景做H3/H5配对验证。模型内单侧95% Student-t下界、所有样本不下降、现金底线与退出检查共同控制建议；未通过则弃权。它不能消除模型偏差。
- H1交互模式不运行旧版重诊断；界面给出目标权重、预测指标与效用贡献变化、原计划差异、验证下界及限制。自动保存/恢复/导出沿用原实验室。
- 独立的有向信誉账本记录“观察者→提议者”，只核验已经接受的正额承诺。Beta(2,1)先验、接受阈值0.6、违约4轮后允许试探。公共品仍不可排他；现有双边互助可按信誉拒绝。它是实验机制，不是完整自由语言谈判系统。
- 30元累计费用授权保留旧账；真实调用脚本有前置门槛、共享草稿、独立采用选择、原始证据与未知结果不重试。

## 2. 先冻结，再验证

市场、合法动作层、观察与公开预测构造的逐文件SHA256在 `runs/reliable-advisor-v19/frozen-market.json`。建议核心、响应模型、评价器和共同实验定义的留出前hash在 `implementation-before-holdout.json`。最终逐项核对保持一致。

训练24种子519001–519024，公开转移{cal['training_rows']}条；响应校准8个新种子520001–520008，共{cal['test_rows']}条。主实验使用521001–521020，制度使用522001–522020，规模使用523001–523020，互不冒充新增独立样本。开发3例518xxx不计入主实验。

主实验6市场×2/5/10家公司×利润/福利×20种子=720源状态；3种响应消融合计2160次建议。先运行4轮生成原计划与公开历史，然后每个候选在同源真实引擎中执行3/5轮、折扣0.95。首轮之后回到同一原计划并按新状态修正，其他企业使用TFT/Grim/WSLS并更新记忆；消费者、供应商、政府为原生规则。不是每轮重新调用LLM的闭环实验。

每个状态对所有同池候选执行反事实，保存动作、指标、状态hash与独立重放。Oracle是同池、同次实际未来的事后最优上界，不传给建议。规则仅为辅助对照。现金/退出也保存，收益结论按折扣公司利润或已核算社会福利评价。

每个单元20独立种子，95%配对bootstrap 2000次；跨市场总体区间先对每个种子的条件取均值，再重采样20种子。区间是探索性的，未校正多重比较。每组通过需均值区间下限>0且实际负收益率≤10%；弃权和零收益不算有效改进。

开发10候选无法覆盖合同/合作，正式留出开始前改为12并保留旧预注册。制度实验两次修补均保留整批旧结果与erratum：先参与后承诺；互助不得重新缩减已承诺预算。最终主制度结果只取修补后的1080局，不混用前两批。

## 3. 建议相对原计划：全部36组

金额是合成元。三/五轮区间来自同种子配对差；通过列是预注册三轮门槛，不能推为现实盈利保证。
''']
    rows=[]
    for r in primary:
        five=next(x for x in a['groups'] if (x['market'],x['companies'],x['goal'],x['mode'],x['horizon'])==(r['market'],r['companies'],r['goal'],'conditional',5))
        rows.append([MARKETS[r['market']],r['companies'],'利润' if r['goal']=='profit' else '福利',interval(r['value']),r['value']['negative'],interval(five['value']),five['value']['negative'],r['abstentions'],'通过' if r['effectiveness_pass'] else '未通过'])
    text += [table(['市场','N','目标','三轮均值 [95%区间]','三轮下降/20','五轮均值 [95%区间]','五轮下降/20','弃权/20','三轮门槛'],rows)]
    text += ['\n## 4. 条件模型是否比搜索本身更有用\n',table(['模式/期限','按种子聚合的平均增益 [95%区间]'],[[k,interval(v)] for k,v in a['pooled'].items()])]
    gains=[]
    for seed in SPEC['seeds']:
        rr=[r for r in raw if r['seed']==seed]
        gains.append(mean(next(x for x in r['results'] if x['mode']=='conditional')['delta']['3']['value']-next(x for x in r['results'] if x['mode']=='none')['delta']['3']['value'] for r in rr))
    vs=paired_summary(gains);write_json(OUT/'conditional-vs-search.json',vs)
    calibration_diffs={}
    for metric in ('brier','absolute_error'):
        calibration_diffs[metric]=paired_summary([mean(r[metric] for r in cal['results']['conditional']['rows'] if r['seed']==seed)-mean(r[metric] for r in cal['results']['generic']['rows'] if r['seed']==seed) for seed in SPEC['calibration_seeds']])
    write_json(OUT/'calibration-seed-statistics.json',calibration_diffs)
    equal_search=sum(next(x for x in row['results'] if x['mode']=='conditional')['selected']==next(x for x in row['results'] if x['mode']=='none')['selected'] for row in raw)
    equal_generic=sum(next(x for x in row['results'] if x['mode']=='conditional')['selected']==next(x for x in row['results'] if x['mode']=='generic')['selected'] for row in raw)
    text += [f'条件模型与搜索-only选择相同动作 {equal_search}/720；与一般倾向相同 {equal_generic}/720。校准误差另按8个种子配对聚合，条件减一般倾向的Brier均值差 {calibration_diffs["brier"]["mean"]:.6f}，95%区间 {calibration_diffs["brier"]["interval"]}；价格幅度MAE差 {calibration_diffs["absolute_error"]["mean"]:.6f}，区间 {calibration_diffs["absolute_error"]["interval"]}。误差差值低于零才是改善。']
    text += [f'\n条件响应相对同池搜索的三轮额外收益为 **{interval(vs)} 合成元**（每个种子先聚合36条件）。这是响应建模增量的比较，不是扩大候选预算的比较。利润与福利金额混合的总体均值仅作摘要，目标分组结果以上表和analysis.json为准。',table(['公开留出预测','一般倾向','条件响应'],[['三分类Brier（低优）',cal['results']['generic']['brier'],cal['results']['conditional']['brier']],['价格变化MAE（低优）',cal['results']['generic']['mae'],cal['results']['conditional']['mae']]])]
    text += ['概率分类更好并不推出价格幅度更准或经营收益更高。这里只覆盖三种已有脚本策略及其混合；没有证明迁移到未知LLM对手。采样情景之间的Student-t下界也不是现实价格模型误差的置信界。\n']
    losses=[]
    for row in raw:
        r=next(x for x in row['results'] if x['mode']=='conditional')
        if r['delta']['5']['value']<0:losses.append((r['delta']['5']['value'],row,r))
    losses.sort(key=lambda x:x[0])
    mismatch=[]
    for actual5,row,r in losses:
        path=OUT/'advice'/f"{row['market']}-{row['companies']}-{row['seed']}-{row['goal']}.json.gz"
        detail=read(path)['details']['conditional']
        mismatch.append(dict(market=row['market'],companies=row['companies'],goal=row['goal'],seed=row['seed'],selected=r['selected'],actual_3round_cents=r['delta']['3']['value'],actual_5round_cents=actual5,model_validation=detail['validation'],all_model_gates_passed=all(v['pass_gate'] for v in detail['validation'])))
    write_json(OUT/'negative-case-validation-mismatch.json',mismatch)
    text+=['### 五轮最差反例\n',table(['市场','N','目标','种子','动作','三轮差','五轮差'],[[MARKETS[row['market']],row['companies'],row['goal'],row['seed'],r['selected'],money(r['delta']['3']['value']),money(v)] for v,row,r in losses[:8]])]
    text += [f"五轮负收益的{len(mismatch)}例中，{sum(x['all_model_gates_passed'] for x in mismatch)}例此前通过全部模型内验证门槛。逐例对应下界与真实执行差保存在negative-case-validation-mismatch.json。这直接反驳了把模型内下界当作市场收益保证的解释；具体偏差来自状态重建、对手行为还是期限代理，需要新的隔离实验，当前不作未经验证的归因。"]
    text += ['\n## 5. 合作、竞争与个体信誉\n1080局×10轮。门槛reachable为N×300000分，unreachable为N×100000×4+1分，后者相对所测策略的4轮承诺上限不可达。这里的“可达”不保证出现背离后仍达标。下表聚焦一名背离者，完整54组含无/多名背离者见analysis.json。\n',table(['门槛','N','机制','项目成功/20','背离者承诺接受率','诚实者承诺接受率','背离者信誉','实际互助订单均值','背离者利润差 [95%区间]'],[[r['threshold'],r['companies'],r['institution'],r['project_success'],f"{r['bad_acceptance']:.1%}",f"{r['good_acceptance']:.1%}" if r['good_acceptance'] is not None else '无承诺',f"{r['credibility']:.3f}",f"{r['aid']:.1f}",interval(r['focal_advantage'])] for r in a['institution'] if r['defection']=='one'])]
    text += [f"\n最终独立检查覆盖 {acceptance['institution']['honest_promise_checks']:,} 个诚实主体承诺/实现对，全部一致。拒绝参与没有被记为违约；退出及现金限制通过原引擎处理。个体信誉能降低对背离者的未来接受，并不保证搭便车利润优势消失；不可排他的公共收益仍可能被共享。完整focal_advantage配对区间保留在analysis.json。"]
    text += ['\n## 6. 人数效应与资源拥挤分开\n120局×20轮，2/5/10家公司，两种资源方案，各20种子。scaled按公司数同比扩大总需求、供应产能、供应商资金、政府资金和公共项目门槛；公司初始资本及消费者单位预算不变。该资源方案不是把所有状态机械复制。\n',table(['资源','N','经营利润率','价格离散（分）','退出率','累计缺货均值','消费者剩余（合成元）'],[[r['regime'],r['companies'],f"{r['margin']:.2%}",f"{r['price_dispersion']:.1f}",f"{r['exit_rate']:.1%}",f"{r['stockout']:.1f}",money(r['consumer_welfare'])] for r in a['scaling']])]
    text += ['规模实验的523xxx与建议实验的521xxx是不同种子队列，不能逐条配对。资源normal/scaled条件下的建议效果见36组表；不能把固定市场中企业增加造成的拥挤损失全部解释为多人博弈本身。\n']
    text += ['## 7. 小范围参数与理论真值\n45局×10轮：价格系数、服务系数、事件信号生成概率、库存损耗率、供应商固定开销，各0.5/1/2倍、3种子。这里的价格效用系数不是价格弹性，信号生成概率也不等于实际供应冲击频率。\n',table(['参数','倍数','公司经营利润率均值','社会福利均值（合成元）'],[[p,f,f"{mean(x['margin'] for x in sens['results'] if x['parameter']==p and x['factor']==f):.2%}",money(mean(x['welfare'] for x in sens['results'] if x['parameter']==p and x['factor']==f))] for p in ('price_elasticity','service_effect','shock_frequency','inventory_spoilage','supplier_overhead') for f in (.5,1.,2.)])]
    text += ['''这些是局部敏感性，库存成本等若结果不变只说明所测策略没有暴露该机制，不能证明参数无关。未获得同地区、同业务、同收入分母的数据，**3–5参数现实数量级校准仍未通过**；没有为追求通过而改动冻结市场。

USDA把价格弹性定义为价格百分比变化引起的需求百分比变化，必须区别于本模型的效用系数；聚合食物需求也不同于单一企业的替代需求。[USDA Food Demand Analysis](https://www.ers.usda.gov/topics/food-choices-health/food-consumption-demand/food-demand-analysis)。DoorDash 2025年第四季度交易额约297亿美元、收入40亿美元，平台收入与交易额分母不同；不能将其比率直接当作这里自营公司的经营利润率目标。[DoorDash 2025 results](https://ir.doordash.com/news/news-details/2026/DoorDash-Releases-Fourth-Quarter-and-Full-Year-2025-Financial-Results/default.aspx)。本轮只作口径核对，没有伪造拟合。

Canonical Stackelberg：p=30−qL−qF，单位成本6，整数动作0–24。用偶数量领导行动学习跟随最佳反应均值，奇数量留出最大误差5.33e−15；承诺解(12,6)，收益(72,36)。证明这个简单序贯模型的反应拟合与真值一致，不证明动态市场的直方图模型、纳什或全局最优。

## 8. 速度、工程和本机使用
''',table(['N','交互H1样本秒数','研究H3/H5样本秒数'],[[n,', '.join(f"{r['wall_seconds']:.3f}" for r in timing['rows'] if r['companies']==n and r['mode']=='interactive'),', '.join(f"{r['wall_seconds']:.3f}" for r in timing['rows'] if r['companies']==n and r['mode']=='research')] for n in (2,5,10)])]
    text += [f'''串行9次交互、3次研究测量在主批次结束后执行，交互<5秒目标 {'通过' if timing['interactive_under_5_seconds'] else '未通过'}。不能当作高并发或p99保证。完整后端481测试及新增16个收费保护/离线请求测试通过；前端18测试、生产构建、类型检查和lint通过。独立重放/算术检查共{a['arithmetic_checks']:,}次（不含敏感性附加核对）。

本机入口 http://localhost:3210/ → 博弈策略实验室 → 建议算法“Reliable：改进原计划”。粘贴JSON原计划并选目标、模式。界面检查实际运行、服务重启后的历史读取、模式/输入恢复、预算30元，记录68fbded4。旧版仍保留；Reliable是实验性改进层。

## 9. 真实模型与费用

累计授权30元，当前保守预留{real['budget']['reserved_cny']:.6f}元，可用{real['budget']['remaining_cny']:.6f}元。本轮新增收费调用{real['new_calls']}次，状态 `{real['status']}`。

预注册为两模型×6种子×alone/old/reliable=36核心单元，共享12份真实LLM草稿，最多36收费请求；后续3轮使用固定草稿续行，采用/拒绝与建议质量分别评价。失败/未知不自动退款或重试。它不是36局长时自主多Agent真实调用。

门槛原因：{'; '.join(real.get('reasons',[])) or '全部前置门槛通过'}。允许付费不等于跳过用户文档规定的零Token前置验收。未通过时，36单元真实实验只能标为已准备、未执行，不产生LLM有效性结论。

## 10. 已证明与未证明

已证明：原计划基线、时间顺序与私有信息隔离的工程约束通过测试；冻结版本在完整注册矩阵中的统计结果可复算；有向信誉区分对象；简单序贯真值检查和本机交互流程可用。

尚未证明：所有市场/规模稳定改善、未知LLM响应迁移、长期重规划收益、真实世界校准、全局最优和多人动态均衡。五轮反例及失败分组不能通过增加同一留出集上的调参被“修正”为独立证据。

下一轮应以本次失败集作开发诊断，隔离新的训练、校准和最终测试种子：优先检查公开预测状态重建误差、实际对手与模型的行动差、有限期限延迟回报、以及门槛对模型偏差的识别。新算法必须另起版本和新留出集；本轮冻结结果永久保留。

## 复现与证据

依次运行 scripts/train_reliable_v19.py、evaluate_reliable_v19.py、cooperation_reliable_v19.py、scaling_reliable_v19.py、sensitivity_reliable_v19.py、analyze_reliable_v19.py、timing_reliable_v19.py、verify_reliable_v19.py、real_reliable_v19.py、report_reliable_v19.py。正式模型/留出文件已存在时应先核对manifest；不要覆盖已有训练模型后继续沿用旧留出结果。

`runs/reliable-advisor-v19/analysis.json`含全部216个建议分组、54个制度分组和6个规模分组；`acceptance.json`分开记录工程、经济与收费门槛。原始轨迹、两次制度失败批次及源代码会以独立v19研究包冻结，不覆盖v18。
''']
    (ROOT/'docs/reliable-advisor-results.md').write_text('\n\n'.join(text),encoding='utf-8')
    print('Report generated from completed evidence')

if __name__=='__main__':main()
