# 复算与数据合同

从项目根目录运行。代码使用 Python 3.12、NumPy、SciPy、scikit-learn 1.8.0 和 joblib。全量模拟另需项目依赖及前阶段固定版本的 EconEvals 作者结算。真实模型来源为 [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/9217f5db79a29953eb74d5343926648285ec7e67) 和 [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/tree/2d4a76a30b4af41ecd395c35725ac11688d4cfe4)。GGUF 大文件未放入 Git；逐提示词实际输出、提示内容、模型哈希和结算结果已收录。

## 离线复算（无需模型服务）

```powershell
python scripts/deployment/results.py blind
python scripts/deployment/results.py independent
python scripts/deployment/results.py numeric
python scripts/deployment/precision.py
python scripts/deployment/figure.py
python scripts/deployment/verify.py
python scripts/deployment/verify.py --replay
python scripts/field_validation/check_design.py
```

最后两项分别额外需要完整市场环境和现场分析依赖。回放验证使用已存真实推理响应重新形成价格、执行市场转移，样本包括两市场和两模型；不会重新请求商业或本地模型。仓库中无企业现场交易台账。

## 实验文件

- 地点为 `experiment-results/independent-deployment-v1`。每个三分支机会的 `profit` 按保留、降 4%、涨 4% 排列，三条分支均到第 60 轮。生鲜原始利润是分，分析转换成模拟元；外部使用作者利润单位。
- `llm-*.json.gz` 含固定模型文件摘要和 162 个合法有限提示的原始响应；`independent-*.json.gz` 含对应路径索引、模型草稿、公开特征、决定及利润。同种子两个模型分别分析。
- `numeric-requests.json.gz` 用完整标准化 JSON 请求的 SHA-256 作键，保存模型、数字提示、原报价和 token 用量；逐机会 `prefix_calls`、`branch_calls` 的每行是轮次、请求哈希、原始数值、按当时边界裁剪后的数值、反馈参考价、下限、上限。指令拒绝的候选也保留相同来源的完整分支。
- `blind-*.json.gz` 的 `base` 是从第 1 轮开始保留原计划的整局利润，`paths` 是八个时点的两个干预后缀。`x` 是决策当时的合法观测，`onset` 和 `condition` 只能由评估者使用。`action` 的 0、1、2、3 为永久弃权、立即降价、立即涨价、等待。动态策略在第一次非等待时停止。
- `blind-transfer.json.gz` 对独立语言模型和新的隐藏切换种子记录三个政策的决定及利润；`adoption.json.gz` 在相同数值报价机会记录真实 LLM 是否自选采纳以及交换选项次序后的决定。两个回答均没有未来利润信息。
- 工作台的默认决策权未改变。字段 `bounds` 为种子级配对自举的 1.25%、98.75% 分位数，同时给出普通 `ci95`。前者构成每侧 98.75% 单侧界，也可读作 97.5% 双侧区间。预定门槛按机会而非按已采纳者计算。`worst5` 是所有机会中最差 5% 的实际平均差值，计入弃权的零收益。另报放行组最差 5% 均值及损失率的单侧 Clopper–Pearson 上界。

## 重新运行程序

未知时点规则实验依次执行 `blind.py train`、`blind.py fit`、`blind.py test`。已存在的冻结文件会校验权重；重新独立拟合需在单独检出中省略相应生成的拟合文件。随后执行 `blind_transfer.py`。独立模型的菜单推理服务在 127.0.0.1 的 19867、19868 端口，模型别名分别为 qwen、smol。使用 llama.cpp build 10976、贪心解码、固定 seed、原生聊天模板。将已校验权重放入 `.tmp/independent-models/{qwen,smol}.gguf`，启动两个模型后运行 `local_prices.py`，再运行 `independent.py`。数值组可通过 `serve_numeric.py` 托管两个本地服务并依次完成 `numeric.py`、`adoption.py`、`numeric_stopping.py`。菜单顺序诊断可在模型启动期间运行 `menu_order.py`。数值停止迁移的额外请求保存为 `numeric-stopping-requests.json.gz`，与主数值请求按哈希联合回放。运行时缓存均按精确请求恢复；研究可复算性以已发布的输出与权重为准，重新推理的结果还依赖推理后端、硬件与数值实现。
