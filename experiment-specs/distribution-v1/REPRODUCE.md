# 从归档复算

沿用仓库 Python 虚拟环境及固定的 EconEvals 上游版本 e1f2a40fec96f0d27f5414873c4310f2b5c51935。若检出路径与默认不同，在运行前设置 ECONEVALS_ROOT。模型采集端和离线分析都从仓库根目录运行。

~~~powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python scripts/four_prices/distribution_trials.py smol
python scripts/four_prices/distribution_trials.py qwen
python scripts/four_prices/distribution_study.py freeze
python scripts/four_prices/distribution_study.py predict
python scripts/four_prices/distribution_study.py measure
python scripts/four_prices/distribution_figure.py
~~~

归档完整时，上述命令逐个核对请求摘要、既有开局、后续随机序列、三者实际执行的唯一干预及所有轮数。若干每组首例用原回包从头复算。评分需要公开前缀及模型哈希，终局单独用于 measure。最后一行重建图，需要支持中文的本机字体。图和报告在 docs 中。含公开输入和预测的逐报价文件、阈值、原始结果和汇总在 experiment-results/distribution-v1，基线及复用续局在两个前序实验目录。

按 PLAN 重新采样需要项目外的固定模型权重及 CPU 运行时。以下两条仅在需要新生成时执行，依次启动和回收各自拥有的 CPU 服务，不使用商业 API。

~~~powershell
python scripts/deployment/cpu_server.py --models smol scripts/four_prices/distribution_trials.py smol --collect
python scripts/deployment/cpu_server.py --models qwen scripts/four_prices/distribution_trials.py qwen --collect
~~~

96 个 SmolLM2 测试市场来自两个市场各 48 个新种子。另在 48 个相同零售种子换成 Qwen，模型比较仍按市场种子配对。请求预算用已归档回包的摘要计算，由同一开局的多个续局共享首份历史和相同请求。同一个种子换模型及四次续局都不会增加独立市场种子数。
