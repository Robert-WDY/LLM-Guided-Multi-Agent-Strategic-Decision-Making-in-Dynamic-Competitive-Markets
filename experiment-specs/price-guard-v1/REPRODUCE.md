# 复算公开报价筛选和续约撤回

从仓库根目录运行，使用与上轮四企业实验相同的 Python 环境及固定的 EconEvals 检出。上游提交为 `e1f2a40fec96f0d27f5414873c4310f2b5c51935`，位置由 `ECONEVALS_ROOT` 指定，默认是 `.tmp/econ-evals-upstream`。模型是 SmolLM2-1.7B-Instruct，Q4_K_M 权重 SHA-256 为 `decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33`，温度 0.85。同一公司、日期、提示和随机种子的各条分支共用真实回包，不同公司独立抽样。模型名称和权重与上轮相同，独立性来自市场种子。公司可看自己的账，公开建议器不收销量、利润、人格或未来对手行动。

在已有归档的检出中，以下命令离线重算汇总、历史训练样本、逐个请求的身份与随机种子绑定、各终局和部分全路径。tail 和 quotes 命令在归档齐全时也直接复用原回包：

~~~powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python scripts/four_prices/guard_report.py
python scripts/four_prices/guard_verify.py
python scripts/four_prices/guard_decisions.py
python scripts/four_prices/guard_spillovers.py
python scripts/four_prices/guard_tail.py
python scripts/four_prices/guard_quotes.py
python scripts/four_prices/expiry_study.py
python scripts/four_prices/guard_transfer.py
python scripts/four_prices/plot_guard.py
python -m pytest
~~~

原始记录在 `experiment-results/price-guard-v1`。menu 是首批九档报价，confirm 是第二批原经营方式，renewal 复用 confirm 的同局真实回包并追加经营者撤回加价带来的新分支。legal 和 legacy 使用相同的独立新种子。legacy 只跑零安装，用来检验报价边界。输出的每个公司、每轮都保留实际执行价、成交量、经营利润、营收、期初期末是否营业。只计算仍在营企业的价格趋同，退出企业在利润计算中保留。一次只撤掉一个合同的事后诊断位于 tail 目录，不能把不同合同的撤回收益相加。

如从头生成，同一模型以 llama.cpp 服务形式提供本机 19943 端口的 completion 和 chat completions 接口。使用新的实验输出工作副本，现有完整归档会自动复用。依次运行 `fit_guard.py`、`guard_study.py`、`renewal_study.py` 后执行上面的复算命令。所有请求仅连接回环地址。按模型、硬件及服务端批调度重新采样可能改变数值，核对本文数据应使用随仓库提供的实际响应。

原推断和简单估值都从保留、降低4%、提高4%中选择；筛选器只决定是否否决。对手建模的净价值应比较 safe_inf 和 safe_dir，两者使用相同公开数据、历史结果、三个候选和计算额度。inf 与 safe_inf 的差异包含额外历史训练标签的价值。每家利润按从第19轮起的0.99折现计到第60轮，正数指同市场同店比自主报价多赚。原门槛采用每个预定机会为分母；一次性风险在预抽单个合同上计算。观测最差值不是可能损失的上限。


追试的 expiry-control / expiry 按同一新种子切换是否使用原草稿的续约参考。qwen 在原 legal 的相同场景上改用 Qwen2.5-0.5B，本地权重 SHA 与协议见 MODEL.md，若重采样则使用端口 19944、服务模型名 qwen。完整检出从已有归档计算这两项时均无需模型服务。
