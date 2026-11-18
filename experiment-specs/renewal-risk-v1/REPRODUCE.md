# 独立复核

从仓库根目录执行。市场沿用 [four-prices-v1](../four-prices-v1) 和 [price-causes-v1](../price-causes-v1/REPRODUCE.md) 的 Python 环境与 EconEvals 上游版本。若上游不在默认目录 `.tmp/econ-evals-upstream`，将 `ECONEVALS_ROOT` 配到该固定提交的检出位置。复算需要 NumPy、SciPy、scikit-learn 和项目模拟环境。读取单笔建议仅需项目的运行依赖和已提交的权重文件。

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
python scripts/four_prices/renewal_risk_study.py freeze
python scripts/four_prices/renewal_information.py freeze
python scripts/four_prices/export_renewal.py
python scripts/four_prices/renewal_risk_study.py predict
python scripts/four_prices/renewal_risk_study.py confirm
python scripts/four_prices/renewal_diagnostics.py
python scripts/four_prices/renewal_information.py confirm
python scripts/four_prices/renewal_verify.py
python scripts/four_prices/renewal_draws.py
python scripts/four_prices/renewal_label_stability.py
python scripts/four_prices/renewal_precision.py
python scripts/four_prices/renewal_timing.py primary
python scripts/four_prices/renewal_timing.py precision
python scripts/four_prices/renewal_late_veto.py freeze
python scripts/four_prices/renewal_late_veto.py observe
python scripts/four_prices/renewal_late_veto.py measure
python scripts/four_prices/review_price.py --own-renewal experiment-specs/price-causes-v1/panel-example.json
```

`freeze` 只读取上一阶段的 216 个拟合种子和 96 个校准种子（每市场）。`predict` 从新种子取结算前公共前缀，存下对两种改价的预测和决定。`confirm` 对齐已封存决定，打开三个动作的真实终局标签。新的主确认文件与随机续局文件在 [归档目录](../../experiment-results/renewal-risk-v1)。`renewal_verify.py` 用只含可见信息的最小输入重算全部 240 次建议。`renewal_draws.py` 默认严格回放，缺记录会报错。

看到单笔建议时，`candidate` 是冻结方法想做什么，`disposition` 是实际处理意见，研究影子模式保留店长原报价。`future_renewals` 是将预测的两次续约轮次。`predicted_log_quote_difference` 估计本店在改价和保持两条路径间的申请报价对数差，退出/非共同在营时目标按零处理；挂牌概率差另列。`loss_probability_if_executed` 的条件是本次干预确实改变执行价格。成交价受市场约束，候选跌 4% 不保证能跌满 4%。

完整归档齐备时以上操作均可断开模型后端执行。需要从零采集，先按上一阶段本地配置放好 CPU 推理程序和固定 SmolLM2 权重，再依序运行以下两条。新采集是新的随机实验，论文数字以已入库的抽样回包复算。

```powershell
python scripts/deployment/cpu_server.py --models smol scripts/four_prices/renewal_trials.py
python scripts/deployment/cpu_server.py --models smol scripts/four_prices/renewal_draws.py --collect
python scripts/deployment/cpu_server.py --models smol scripts/four_prices/renewal_precision.py --collect
```

原始补样同时在第二个终端运行 `renewal_late_veto.py live`，只监测预留的 288 份公共输入，发现已结算而未存预报即失败。`seal` 选项限首次事前封存，它要求预留种子的报价前缀尚不存在；发布归档请用 `freeze` 离线复算。在线落盘时序来自原机器，Git 检出会改变文件系统时间；预测、价格、利润及分组由归档内容复算。
