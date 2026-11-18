# 复算和单店报价复核

沿用 four-prices-v1 已固定的 EconEvals 上游提交和 Python 依赖。完整检出内，以下命令只读归档进行建模、反事实结算和统计，无需启动模型服务，不调用商业 API。

~~~powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python scripts/four_prices/cause_value.py calibrate
python scripts/four_prices/cause_value.py test
python scripts/four_prices/cause_ablation.py
python scripts/four_prices/cause_selective.py calibrate
python scripts/four_prices/cause_selective.py test
python scripts/four_prices/cause_medians.py
python scripts/four_prices/cause_hardware.py
python scripts/four_prices/cause_repro.py
python scripts/four_prices/cause_paths.py
python scripts/four_prices/cause_market.py
python scripts/four_prices/cause_models.py
python scripts/four_prices/cause_units.py
python scripts/four_prices/cause_credit.py
python scripts/four_prices/cause_verify.py
python scripts/four_prices/review_price.py experiment-specs/price-causes-v1/panel-example.json
~~~

fit / calibrate / test 各种子归档包含真实经营模型的请求摘要、回包与三条 60 轮轨迹；市场原始账号、真实商户交易均未使用。复核 price-causes-v1/cross 和 units 需要上阶段 price-guard-v1 的配对直接报价归档。cross、units、credit 在归档不全时会尝试生成缺失案例。需要从零生成时，按协议先拟合、再校准、最后测试，例如 `python scripts/deployment/cpu_server.py --models smol scripts/four_prices/cause_trials.py test`。双模型的自由数字实验将相应脚本交给 `--models both`。只在需要真实生成时启动本地服务；预存结果可直接调用脚本复算。数字基准旧入口支持 `serve_numeric.py --offline`；一般生成模式只在缺少回包时按需启动相应模型。16 个重复市场中，两套后端配置对 601 份相同请求有 211 份抽样不同，复现论文数字使用归档回包。后端划分在 hardware-split.json，原始市场种子不因硬件重复计数。

对一份新报价，输入 `panel`（每行四家真实公开结算价，未挂牌的企业填 null）、`seat`（0 到 3）、日期、报价草稿和市场标识。程序返回逐对手的本合同及下合同预测、简单估值、使用对手预测的候选以及独立认证后的执行意见。模型权重是只含数值数组的 NPZ，可用 export_panel.py 从拟合样本重建。两列的旧输入接口仍支持外部给定方向，改用放行亏损率检查及分方向独立校准。实验输出中以 known_next 和 best_terminal 标出的结果分别看到了对手当期真实报价及完整事后终局，运行时接口不会读取它们。

主表的一次机会按第23至43轮预抽，终局从该轮起按0.99折现；降/涨均相对于当期草稿的4%。实际落价相同或退出照常计入。效益区间使用种子配对自举的1.25%与98.75%分位，亏损率按放行机会单侧98.75%上界。小样本、已有价格轨迹上的控制实验，以及按已亏损案例分组的分解均单列。范围是三家同行和一个实际经营者的竞争性定价。
