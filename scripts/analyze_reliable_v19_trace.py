"""Build a reader's guide and 28 case-by-case analyses from reproducible traces."""
from collections import Counter
import shutil,zipfile
from reliable_v19_common import *
from report_reliable_v19 import MARKETS,table,money

D=ROOT/'artifacts/reliable-v19-trace'
LABELS={'price_up':'提价','price_down':'降价','procurement_up':'增加采购','cooperation_adjust':'调整公共投入'}
def main():
    verify=read_json(D/'trace-verification.json');assert verify['passed']
    cases=read_json(D/'negative-cases-summary.json');analysis=read_json(OUT/'analysis.json');index=read_json(D/'all-720-cases-index.json')
    all_rows=read_json(OUT/'advice.json')['results'];late=sum(x['actual_3round_cents']>=0 for x in cases)
    parts=['''# v19 完整执行 Trace 与逐例分析

本文件提供实际输入、动作、候选评分、验证、市场结算与审计记录，不是模型内部思维链。本轮为规则Agent与模拟器实验，新增真实LLM调用0次；没有可提供的本轮真实模型回答或采用/拒绝过程。

## 一、先看结论

原计划接入、时序隔离、保存恢复和合作协议误记违约的问题已修复；“所有市场稳定改善”没有修复到验收通过。利润18/18分组通过，福利7/18；720源状态中三轮下降24例、五轮下降28例。28个五轮负例均通过过模型内H3/H5验证。因此，模型内正下界不能当成实际经营保证。

条件响应相对同池纯搜索的三轮额外收益为-245.49合成元，95%种子配对bootstrap区间[-675.27,117.29]；没有证明增加该模块提高经营收益。分类Brier按观测行汇总较低，但按8个独立种子比较的区间仍跨零；也不能据此声称响应已可靠。

## 二、完整记录在哪里

- [原始全量证据包（4158文件）](../../releases/reliable-advisor-v19.0.0/evidence.zip)：720正式源状态及3个开发样本、三种建议结果、所有候选实际动作/指标/逐轮状态hash、1080主制度案例与两批历史修补前案例、120规模、45敏感性、校准、预注册与验收。不要把历史修补批次计为新增独立主样本。
- [720例索引](all-720-cases-index.json)：每个状态的市场、人数、目标、seed、初始hash、原文件SHA256和三种建议结果。
- [28例负收益汇总](negative-cases-summary.json)：逐轮收益、福利分项、原计划/建议动作与模型内下界。
- 每个 `negative-cases/*.json.gz` 包含该案例的原始完整记录、全部候选的逐轮完整市场状态，以及条件建议发现/验证阶段的全部预测状态。
- [展开验证结果](trace-verification.json)：冻结核心hash未变；每个负例重算输出除运行秒数外，与已冻结结果逐字段相等；实际回放逐轮hash相等。

原始全量记录足以从初态与动作重放；本次额外展开28个反例的全量状态，便于直接查看。本次没有声称把其余692例的所有预测中间态另行导出。费用门槛尚未通过，36单元真实LLM实验依然是未执行。

## 三、一条建议实际如何产生

```mermaid
flowchart TD
    A[冻结配置与seed：真实市场运行4轮] --> B[得到第5轮Agent原计划]
    B --> C[合法化原计划：候选0]
    C --> D[相对原计划局部扰动：最多12候选]
    A --> E[公开信息＋本公司私有信息重建预测状态]
    D --> F[每候选3个发现情景：H3]
    E --> F
    F --> G[选择一个非原计划候选]
    G --> H[8个新情景：分别H3、H5与原计划配对]
    H --> I{下界正、样本不下降、现金与退出检查通过}
    I -->|是| J[给出建议]
    I -->|否| K[原计划不变：弃权]
    J --> L[评价器：所有候选在实际市场各跑5轮]
    K --> L
    L --> M[3/5轮收益差、Oracle后悔值、独立回放]
```

对手本轮不观察候选；后续轮才根据候选造成的已结算状态变化预测回应。实际对手为TFT/Grim/WSLS规则，预测对手由一般/条件价格模型加原生动作策略组成；这两者不是同一个决策器。消费者、供应商、政府仍用原生规则。

候选只改当前一步，未来本公司返回同一原计划并按状态重新合法化，其他企业会更新各自策略记忆。实验没有证明每轮重新规划或长期自主LLM交互的效果。

### 收益与下界的计算口径

本次两种目标均为单一权重：利润或社会福利。三/五轮实际差值为各轮“建议减原计划”乘0.95的轮次幂再求和。预测效用用目标尺度归一化；案例表将其乘回尺度并除100，统一展示为合成元。模型内单侧95%下界为8个配对差的均值减1.895×样本标准差/√8；它只反映该模拟分布内的采样不确定性。

实际市场还受到未准确建模的对手行动、隐藏状态估计和后续经营过程影响。预测方差小，不意味着这些误差小。最终20-seed bootstrap是另一层评价，不能与模型内部8情景下界混为一谈。

## 四、完整矩阵和消融摘要
''']
    rows=[]
    for mode in SPEC['modes']:
        for goal in SPEC['goals']:
            gg=[g for g in analysis['groups'] if g['mode']==mode and g['goal']==goal and g['horizon']==3]
            rr=[next(r for r in x['results'] if r['mode']==mode) for x in all_rows if x['goal']==goal]
            rows.append([mode,goal,sum(x['effectiveness_pass'] for x in gg),sum(x['delta']['3']['value']<0 for x in rr),sum(x['delta']['5']['value']<0 for x in rr),sum(x['disposition']=='abstain' for x in rr)])
    parts+=[table(['模型','目标','通过/18组','H3下降/360','H5下降/360','弃权/360'],rows)]
    histogram=Counter((x['goal'],x['selected']) for x in cases)
    parts+=['\n## 五、负收益模式\n',table(['目标','动作','五轮负例数'],[[g,LABELS.get(k,k),n] for (g,k),n in histogram.most_common()]),f'28例中有{late}例三轮差非负、五轮转负；其余22例在三轮已为负。这说明延长模型内验证并没有自动修复实际路径的延迟损失，也说明仅看总体平均正收益会掩盖个别反例。']
    parts+=['''### 当前证据支持什么，不能支持什么

1. **已经定位的观测事实**：福利负例集中于采购扩张；利润负例集中于提价；所有负例通过模型内门槛。逐轮账本可进一步判断损失落在消费者、本公司、其他企业或供应商。
2. **代码中已确认的模型限制**：公开预测状态用先验估计对手现金、成本、产能等；条件模型只改变对手价格，并未学习其完整投资/采购反应；候选空间有限，本公司未来计划固定。这些与实际评价策略有明确差异。
3. **尚未证明的因果归因**：不能仅因采购负例多，就删掉采购动作；不能仅因预测/实际不同，就断言某个账本公式算错。本次实际重放与算术检查通过，说明偏差可重复，并不等于经济模型现实有效。
4. **需要独立验证的修复**：把本次反例作为开发集，另设新seed最终留出；分别隔离公开状态重建误差、价格之外的采购/投资响应、福利成本分项预测误差，再验证分布外识别与弃权。不能在这720例上反复调参后仍称独立验证通过。

## 六、28个负例逐一展开

以下逐轮差值均为未折扣的“建议减原计划”，累计目标差按0.95折扣。福利分解采用：消费者剩余＋下游生产者剩余＋上游生产者剩余＋政府净预算－缺货及退出外部成本。分项含完整市场，不仅是company_A。
''']
    for i,row in enumerate(cases):
        key=row['case_id'];data=read(D/'negative-cases'/(key+'.json.gz'));src=data['source'];r=src['details']['conditional'];goal=row['goal'];scale=r['objective']['scales'][goal]
        assert r['objective']['weights'][goal]==1
        changes={k:[src['draft'].get(k),v] for k,v in r['action'].items() if k not in ('action_id','strategy_summary') and src['draft'].get(k)!=v}
        parts += [f"### {i+1}. {key}\n\n市场：{MARKETS[row['market']]}；{row['companies']}家；目标{goal}；seed {row['seed']}；建议{LABELS.get(row['selected'],row['selected'])}。\n\n实际动作变化：`{json.dumps(changes,ensure_ascii=False)}`。\n\n[完整展开JSON](negative-cases/{key}.json.gz)；源初态hash `{src['source_state']['state_hash']}`。",
                  table(['期限','预测平均增益（合成元）','模型内95%下界（合成元）','模型负样本','实际增益（合成元）'],[[v['horizon'],money(v['mean_gain']*scale),money(v['lower_bound']*scale),f"{v['negative_samples']}/8",money(row['actual_3round_cents'] if v['horizon']==3 else row['actual_5round_cents'])] for v in r['validation']])]
        cumulative=0;round_rows=[];components=Counter()
        for z in row['round_deltas']:
            target=z['profit_delta_cents'] if goal=='profit' else z['welfare_delta_cents'];cumulative+=target*z['discount'];w=z['welfare_components_delta']
            cs=w['round_consumer_surplus_cents'];down=w['round_downstream_producer_surplus_cents'];up=w['round_upstream_producer_surplus_cents'];gov=w['round_government_net_budget_cents'];external=-(w['round_stockout_externality_cents']+w['round_business_exit_externality_cents'])
            assert cs+down+up+gov+external==z['welfare_delta_cents']
            for label,value in [('消费者',cs),('下游企业',down),('上游供应商',up),('政府',gov),('外部成本项',external)]:components[label]+=value*z['discount']
            round_rows.append([z['round'],money(target),money(cumulative),money(z['profit_delta_cents']),money(cs),money(down),money(up),money(gov),money(external)])
        parts += [table(['实际轮','当轮目标差','累计折扣目标差','本公司利润差','消费者剩余差','下游剩余差','上游剩余差','政府净额差','外部成本项差'],round_rows)]
        ordered=sorted(components.items(),key=lambda x:x[1]);parts += [f"账本分析：五轮折扣福利分项中最负的是{ordered[0][0]}（{money(ordered[0][1])}元），其次是{ordered[1][0]}（{money(ordered[1][1])}元）。这是已验证的会计分解，不是机制因果识别。{'本例三轮时非负，后两轮损失令五轮转负。' if row['actual_3round_cents']>=0 else '本例三轮和五轮实际增益均为负。'} 预测门槛却均通过，需要改进误差识别，不能把小采样方差当成安全证明。"]
    parts+=['''## 七、Trace字段字典与复现

| 字段 | 内容与用途 |
| --- | --- |
| source.source_state / config | 第5轮实际源状态与冻结配置；评价器真值，不直接暴露给预测模型 |
| source.draft / history | 原始合法计划与决策前公开价格历史 |
| source.details.none/generic/conditional | 三种消融的候选、发现分数、目标权重、验证8情景、最终动作、弃权与边界 |
| source.actual | 所有候选及辅助规则对照在同源市场中的H1/H3/H5指标、联合动作与逐轮hash |
| actual_full_states | 本次负例额外展开所有候选每轮结算后的完整状态 |
| predicted_full_states | 冻结条件建议重演的每条路径：候选ID、发现/验证、情景号、期限、预测初态、每轮动作及完整预测状态 |
| round_deltas | 实际原计划/建议成对的逐轮目标和福利分项差 |
| advisor_reproduction_equal_except_runtime | 重演建议与已冻结结果除运行时间外是否逐字段相等 |

读取压缩JSON：

```python
import gzip, json
with gzip.open("negative-cases/price_sensitive-5-521017-welfare.json.gz", "rt", encoding="utf-8") as f:
    trace = json.load(f)
print(trace["round_deltas"])
print(trace["predicted_full_states"][0]["context"])
```

在项目根目录使用现有.venv运行 `scripts/export_reliable_v19_trace.py` 可重新展开同一冻结记录；`scripts/analyze_reliable_v19_trace.py` 可重建本报告。它们不调用真实模型，也不改动原建议算法。原始4158文件证据包和源码包保持原SHA256。

## 八、代码修复状态

已经修复并验证：原计划基线与元数据、缺少原计划的错误处理、同时行动的时间边界、预算不足弃权、UI保存/恢复/解释、预算授权保留旧账、诚实暂停合作误记违约，以及互助规范化缩减承诺的问题。

还没有完成的研究目标：福利跨市场稳定改善、未知策略/LLM泛化、对模型偏差可靠弃权、36单元真实模型实验、3–5参数现实数量级校准。原文要求前置验收通过后再收费，因此真实模型阶段仍未执行；这不是预算耗尽，30元上限尚有20.006435元保守额度。
''']
    (D/'TRACE_ANALYSIS.md').write_text('\n\n'.join(parts),encoding='utf-8')
    doc='\n\n'.join(parts).replace('](negative-cases/','](../artifacts/reliable-v19-trace/negative-cases/').replace('](all-720-cases-index.json)','](../artifacts/reliable-v19-trace/all-720-cases-index.json)').replace('](negative-cases-summary.json)','](../artifacts/reliable-v19-trace/negative-cases-summary.json)').replace('](trace-verification.json)','](../artifacts/reliable-v19-trace/trace-verification.json)').replace('](../../releases/','](../releases/')
    (ROOT/'docs/reliable-v19-trace-analysis.md').write_text(doc,encoding='utf-8')
    for name in ('export_reliable_v19_trace.py','analyze_reliable_v19_trace.py'):shutil.copyfile(ROOT/'scripts'/name,D/name)
    manifest={str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write_json(D/'manifest.json',dict(files=manifest,original_release_evidence_sha256=read_json(ROOT/'releases/reliable-advisor-v19.0.0/manifest.json')['evidence']['sha256'],scope='Additive trace expansion; original frozen evidence unchanged.'))
    archive=ROOT/'artifacts/reliable-v19-expanded-traces.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in D.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(D))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(report=str(ROOT/'docs/reliable-v19-trace-analysis.md'),archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size)))

if __name__=='__main__':main()
