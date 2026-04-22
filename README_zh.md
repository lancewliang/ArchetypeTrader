# ArchetypeTrader

这是一个基于 PyTorch 的三阶段分层强化学习交易框架实现，源自提交至 **AAAI 2026** 的论文。原论文研究的是使用可复用策略原型进行加密货币交易。当前代码库保留了这组三阶段研究结构，同时已经演进为以 AL 数据为主的实验实现，包含利润感知的 Phase I 训练、模块化的 PPO 风格 Phase II 训练器，以及更完整的评估和审计流程。

> **声明：** 本代码库仅用于研究和工程实验，不构成投资建议，也不应直接用于实盘交易。若干实现选择有意超出了论文范围，差异已在下文列出。

> **论文：** *ArchetypeTrader: Reinforcement Learning for Selecting and Refining Learnable Strategic Archetypes in Quantitative Trading* [[PDF]](AAAI26_ArchetypeTrader.pdf)
> Chuqiao Zong, Molei Qin, Haochong Xia, Bo An - Nanyang Technological University, Singapore
>
> 代码注释引用了论文中的具体章节、公式和算法，以便追溯。对于论文描述模糊或当前代码已经超出论文的部分，本 README 会明确列出差异。

## 概述

ArchetypeTrader 遵循原论文的三阶段流水线：

1. **Phase I - 原型发现**：动态规划规划器（Algorithm 1）在单次交易 / 单次切换约束下生成示范轨迹。VQ 编码器-码本-解码器模型将这些轨迹压缩为离散交易原型。

2. **Phase II - 原型选择**：horizon 级别的 RL 智能体在每个 72 步交易周期开始时选择一个原型。冻结的解码器再将选中的原型码转换为逐步基础动作。

3. **Phase III - 原型精炼**：step 级别的 RL 智能体可以用遗憾感知奖励信号精炼所选原型的基础动作，同时每个 horizon 最多允许一次有效调整。

当前仓库状态：
- 当前工作区中可用的数据集是 `AL`；ETH 仍保留在配置中，但本 checkout 不包含 ETH feather 文件。
- 特征管道现在使用 `固定特征 + 可选周期特征集`，不再是硬编码的 45 维状态。CLI 解析默认使用 `--cycle-feature-sets middle`，对应 `state_dim=57`。
- 默认流水线脚本当前训练 Phase I 和 Phase II，然后在 val/test 上评估 Phase II，并包含 DP baseline。Phase III 已实现，但在 `run_pipeline.sh` 中处于注释状态。

```
Feather 数据 -> 特征管道（固定特征 + 周期特征）
    -> DP 规划器 -> 缓存的 DP 轨迹
    -> VQ 编码器-码本-解码器（Phase I）
    -> Phase I 验证 + 利润门控检查点选择
    -> 选择智能体（Phase II, PPO 风格）-> 冻结解码器 -> 基础动作
    -> 可选精炼智能体（Phase III, AdaLN）-> 最终交易动作
    -> 评估引擎（TR, Sharpe, Calmar, Sortino, MDD, Volatility）
    -> PortfolioTracker + TradeAuditor + Backtrader 交叉验证
```

## 项目结构

```
ArchetypeTrader/
├── data/
│   ├── AL/                             # 本 checkout 中可用的 feather 数据
│   │   ├── df_train.feather
│   │   ├── df_val.feather
│   │   └── df_test.feather
│   └── AL_10s/
├── src/
│   ├── config.py                       # 全局超参数（dataclass + CLI）
│   ├── data/
│   │   ├── dataset.py                  # TrajectoryDataset + 归一化统计量
│   │   └── feature_pipeline.py         # Feather 加载和固定/周期特征
│   ├── env/
│   │   └── trading_env.py              # 带 5 档 LOB 滑点的 MDP 交易环境
│   ├── phase1/
│   │   ├── dp_planner.py               # Algorithm 1：单次交易 DP 规划器 + 采样
│   │   ├── vq_encoder.py               # 带时间注意力池化的 LSTM 编码器
│   │   ├── vq_decoder.py               # 带约束解码的 BiLSTM 解码器
│   │   ├── codebook.py                 # 带 k-means 初始化和 dead-code reset 的 VQ 码本
│   │   ├── checkpoint.py               # 利润门控检查点选择
│   │   ├── validation.py               # Phase I 产物验证
│   │   └── env_validation.py           # 环境级原型验证
│   ├── phase2/
│   │   ├── selection_agent.py          # Horizon 级 Actor-Critic 选择器
│   │   ├── rollout.py                  # 批量解码和向量化 horizon 执行
│   │   ├── evaluation.py               # 验证和选择器诊断
│   │   └── diagnostics.py              # 执行 / 原型直方图
│   ├── phase3/
│   │   ├── refinement_agent.py         # 带 AdaLN 的 step 级 Actor-Critic
│   │   ├── policy_adapter.py           # Eq. 6：最终动作计算
│   │   ├── adaln.py                    # 自适应层归一化
│   │   └── regret_reward.py            # Eq. 8 + top-5 后验最优适配
│   ├── evaluation/
│   │   ├── metrics.py                  # TR / AVOL / MDD / ASR / ACR / ASoR
│   │   ├── inference_runner.py         # Phase II / 可选 Phase III 推理
│   │   ├── model_loader.py             # 集中式模型加载
│   │   ├── portfolio_tracker.py        # 跨 horizon 投资组合和现金管理
│   │   ├── trade_auditor.py            # 评估后的交易统计和一致性检查
│   │   └── bt_verifier.py              # Backtrader 回放验证
│   └── utils/
│       ├── gpu_guard.py
│       ├── logger.py
│       ├── normalizer.py
│       └── progress.py
├── scripts/
│   ├── train_phase1.py                 # Phase I：DP 轨迹 + VQ 训练
│   ├── train_phase2.py                 # Phase II：PPO 风格选择器训练
│   ├── train_phase3.py                 # Phase III：遗憾感知精炼训练
│   ├── evaluate.py                     # val/test 评估，可选 DP baseline
│   ├── analyze_dataset.py              # 数据集漂移 / DP oracle 分析
│   ├── diagnose_archetype.py           # Phase II 原型诊断
│   └── diagnose_train_trajectories.py  # AL/batch_001 快速诊断脚本
├── tests/                              # 单元测试 + 属性测试
├── docs/                               # 开发和优化日志
├── run_pipeline.sh                     # 当前端到端 Phase I -> Phase II -> eval 脚本
├── requirements.txt
└── result/
    └── {PAIR}/{BATCH_ID}/
        ├── dp_trajectories/
        ├── phase1_archetype_discovery/
        ├── phase2_archetype_selection/
        ├── phase3_archetype_refinement/
        ├── phase2_eval_val/
        ├── phase2_eval_test/
        ├── dp_val/
        └── dp_test/
```

## 环境配置

```bash
conda create -n ArchetypeTrade python=3.12
conda activate ArchetypeTrade
pip install -r requirements.txt
pip install torch  # 按你的 CUDA 版本单独安装 PyTorch
```

`requirements.txt` 中的依赖：
- `pyarrow>=14.0.0`：feather 文件 I/O
- `numpy>=1.24.0`
- `polars>=0.20.0`：高性能 DataFrame 操作
- `pandas>=2.0.0`
- `tqdm>=4.64.0`：进度条

可选脚本/测试使用的额外工具：
- `pytest`, `hypothesis`：测试
- `backtrader`：独立回放验证
- `scipy`：数据集分析脚本

## 使用方法

### 完整流水线（推荐）

运行当前 Phase I -> Phase II -> Phase II 评估流水线：

```bash
bash run_pipeline.sh AL batch_001 --cycle-feature-sets middle
# 日志保存到 logs/AL/batch_001/AL_pipeline_YYYYMMDD_HHMMSS.log
```

说明：
- 第一个位置参数是 `PAIR`，默认值是 `AL`。
- 第二个位置参数是 `BATCH_ID`，默认值是 `batch_001`。
- 这两个参数之后的额外参数会转发给每个 Python 阶段。
- Phase III 已实现，但当前在 `run_pipeline.sh` 中被注释。

### 单独运行各阶段

训练需要顺序执行，每个阶段依赖前一阶段的产物：

```bash
# Phase I：生成 DP 轨迹 + 训练 VQ 编码器-码本-解码器
python scripts/train_phase1.py --pair AL --train-batch-id batch_001 --cycle-feature-sets middle

# Phase II：训练原型选择智能体
python scripts/train_phase2.py --pair AL --train-batch-id batch_001 --cycle-feature-sets middle

# 在 val/test 上评估 Phase II，并包含 DP baseline
python scripts/evaluate.py --pair AL --train-batch-id batch_001 \
  --split val test --stage-label phase2_eval --with-dp \
  --cycle-feature-sets middle

# 可选 Phase III：训练精炼智能体
python scripts/train_phase3.py --pair AL --train-batch-id batch_001 \
  --cycle-feature-sets middle --beta1 0.5

# 可选 Phase III 评估
python scripts/evaluate.py --pair AL --train-batch-id batch_001 \
  --split val test --stage-label phase3_eval \
  --cycle-feature-sets middle
```

诊断脚本：

```bash
python scripts/analyze_dataset.py --pair AL
python scripts/diagnose_archetype.py --pair AL --batch-id batch_001 --split val --cycle-feature-sets middle
python scripts/diagnose_train_trajectories.py
```

`diagnose_train_trajectories.py` 当前是一个面向 AL/batch_001 的快速诊断脚本，路径是硬编码的。

### 主要 CLI 参数

这些参数都是可选的；默认值定义在 `src/config.py` 和 `parse_args()` 中：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--pair` | 当前训练脚本中为 `AL` | 单个交易标的；覆盖 `Config.pairs` |
| `--train-batch-id` | `default`（`run_pipeline.sh` 中为 `batch_001`） | 用于隔离结果目录 |
| `--cycle-feature-sets` | CLI 中默认为 `middle` | 逗号分隔的特征组：`short,middle,long` |
| `--horizon` | 72 | 每个交易 horizon 的步数 |
| `--commission-rate` | 0.0002 | 评估佣金率 |
| `--num-trajectories` | 50000 | DP 示范轨迹数量 |
| `--phase1-epochs` | 400 | Phase I 训练轮数 |
| `--pretrain-epochs` | 10 | Phase A 连续潜变量预训练轮数 |
| `--latent-dim` | 32 | VQ 潜变量/码本维度 |
| `--lstm-hidden-dim` | 256 | Phase I 编码器/解码器隐藏维度 |
| `--phase1-start-sampling-mode` | `hybrid_stratified_importance` | DP 起始索引采样模式 |
| `--phase2-total-steps` | 1000000 | 选择智能体训练步数 |
| `--selection-alpha` | 0.5 | 初始 imitation/KL 系数 |
| `--phase2-alpha-schedule` | `linear` | `selection_alpha` 调度方式 |
| `--phase2-alpha-final-ratio` | 0.0 | 最终 alpha 与初始 alpha 的比例 |
| `--phase2-imitation-min-raw-return` | 0.0 | 只对 raw return 高于该阈值的 horizon 施加 imitation |
| `--phase3-total-steps` | 1000000 | 精炼智能体训练步数 |
| `--phase3-num-envs` | 16 | 每个 Phase III batch 收集的 horizon 数 |
| `--beta1` | 0.5 | 遗憾系数 beta1 |
| `--beta2` | 1.0 | Phase III 中 hindsight CE 的系数 |
| `--lr` | 3e-4 | 学习率 |
| `--batch-size` | 256 | Phase I batch size |

## 评估流水线

评估系统不只是计算简单指标：

1. **Phase II / 可选 Phase III 推理**（`inference_runner.py`）：运行冻结的 Phase I 和 Phase II 模型，并可选运行 Phase III 精炼模型。
2. **投资组合跟踪**（`portfolio_tracker.py`）：管理现金、多头/空头仓位、平均持仓价格、空头债务、horizon 结算和最终清算。
3. **交易审计**（`trade_auditor.py`）：从导出的操作记录中计算详细交易统计和一致性检查。
4. **Backtrader 交叉验证**（`bt_verifier.py`）：通过 Backtrader 回放相同的仓位序列，作为独立验证引擎。
5. **DP baseline**（`evaluate_pair_dp`）：在相同 split 上运行 DP 规划器，并报告 model-vs-DP 的 TR gap。
6. **CSV 导出**：逐步操作日志会分块导出，方便外部分析。

示例输出路径：

```
result/AL/batch_001/phase2_eval_val/AL_results.json
result/AL/batch_001/phase2_eval_test/AL_results.json
result/AL/batch_001/dp_val/AL_dp_results.json
result/AL/batch_001/dp_test/AL_dp_results.json
```

## 测试

```bash
python -m pytest tests/ -v
```

测试套件使用单元测试和基于 [Hypothesis](https://hypothesis.readthedocs.io/) 的属性测试覆盖核心组件。检查内容包括：
- 特征维度不变量和特征集解析
- 交易环境的仓位 / 奖励 / 成本不变量
- DP 单次交易约束和小规模最优性
- VQ 最近邻量化和受约束解码器行为
- 码本坍塌检测和 dead-code reset 行为
- 每个 horizon 最多一次精炼调整
- 遗憾奖励和后验最优适配逻辑
- 评估指标公式（TR、AVOL、MDD、ASR、ACR、ASoR）
- 投资组合跟踪、交易审计和 Backtrader 验证逻辑

## 核心超参数

| 参数 | 当前代码默认值 | 论文值 | 说明 |
|---|---:|---:|---|
| 状态维度 | CLI 默认 `middle` 时为 57 | 45 | **不同**：当前状态维度是 `24 fixed + selected cycle features` |
| 固定特征维度 | 24 | 45 总状态维度 | **不同**：特征定义已经变化 |
| 动作空间 | {short, flat, long} | {0, 1, 2} | 一致 |
| Horizon h | 72 | 72 | 一致 |
| 评估佣金 | 0.0002 | 0.0002 | 显式使用 config 时与论文费率一致 |
| DP / 训练佣金 | 0.0008 | 0.0002 | **不同**：训练/轨迹阶段使用更高费率作为安全边际 |
| `TradingEnv.COMMISSION_RATE` fallback | 0.0003 | 0.0002 | **实现注意**：主要入口会显式传入 config |
| 码本大小 K | 10 | 10 | 一致 |
| 潜变量维度 | 32 | 16 | **不同**：扩展了瓶颈维度 |
| LSTM 隐藏维度 | 256 | 128 | **不同**：模型更大 |
| VQ commitment beta0 | 0.25 | 0.25 | 一致 |
| Phase I 轨迹数 | 50000 | 30000 | **不同**：增加了数据覆盖 |
| Phase I 训练轮数 | 400 | 100 | **不同**：更长训练和检查点搜索 |
| 预训练轮数 | 10 | N/A | **新增**：连续潜变量预训练 |
| Selection alpha | 0.5 -> 0.0 线性调度 | 1.0 | **不同**：imitation/KL 退火 |
| Phase II 训练步数 | 1000000 | 3000000 | **不同**：当前文档记录的是较短的成功运行 |
| Phase III 训练步数 | 1000000 | 1000000 | 一致 |
| 年化因子 | 52560 | 52560 | 与 10 分钟 bar 一致 |

常见状态维度：

| CLI `--cycle-feature-sets` | 状态维度 |
|---|---:|
| 直接 `Config()`，无周期特征 | 24 |
| `short` | 54 |
| `middle` | 57 |
| `long` | 54 |
| `short,middle` | 73 |
| `middle,long` | 70 |
| `short,middle,long` | 84 |

## 支持的交易标的

| 标的 | 当前代码最大持仓 (m) | 本 checkout 是否有数据 | 论文最大持仓 (m) | 说明 |
|---|---:|---|---:|---|
| AL | 10 | 是 | N/A | 当前实验主线 |
| ETH/USDT | 100 | 否 | 100 | 配置仍支持 ETH，但需要补充数据 |
| BTC/USDT | 未配置 | 否 | 8 | 论文资产；不在当前 `max_positions` 默认值中 |
| DOT/USDT | 未配置 | 否 | 2500 | 论文资产；不在当前 `max_positions` 默认值中 |
| BNB/USDT | 未配置 | 否 | 200 | 论文资产；不在当前 `max_positions` 默认值中 |

## 与论文的差异

本节记录论文和当前代码库之间的已知差异，包括工程增强、有意设计变更，以及近期实验中发现的实现注意事项。

### Phase I - 原型发现

| 方面 | 论文 | 当前代码 | 类型 |
|---|---|---|---|
| 输入状态 | 固定 45 维市场状态 | 24 个固定特征 + 可选 `short/middle/long` 周期特征集 | 设计变更 |
| DP chunk | 采样固定长度 chunk | 使用 `uniform`、`stratified` 或 `hybrid_stratified_importance` 采样合法滑动窗口起点 | 增强 |
| DP 缓存 | 未指定 | `.npz` 缓存保存 pair、horizon、gamma、state_dim、commission、采样元数据；不兼容缓存会备份 | 增强 |
| DP 佣金 | 与论文佣金相同 | 单独的 `dp_commission_rate=0.0008`，用于筛选更高质量轨迹 | 设计变更 |
| 编码器架构 | LSTM-based encoder | LSTM + 对所有 hidden states 的时间注意力池化 | 增强 |
| 解码器架构 | 解码器未完全指定 | 以状态序列和 code vector 为条件的 BiLSTM 解码器 | 增强 |
| 解码器推理 | 未指定 | `decode_with_single_trade_constraint()` 在解码器 logits 上搜索最佳单次切换序列 | 增强 |
| 解码器约束语义 | Algorithm 1 从 flat 开始，并通过 DP 状态 `c` 约束动作变化 | DP planner 遵循 Algorithm 1；decoder 后处理只约束输出序列内部最多一次动作变化，因此执行诊断仍会跟踪相对 flat 初始仓位的持仓变化/direct flip | 实现注意 |
| 数据集归一化 | 未讨论 | Phase I 保存 z-score `norm_stats`，供下游 Phase II/III/evaluation 使用 | 增强 |
| 训练策略 | 使用 Eq. (4) loss 的端到端 VQ 训练 | 两阶段训练：连续潜变量预训练，然后完整 VQ 训练 | 增强 |
| 码本初始化 | 未讨论 | 带利润感知 init/reset 的方向感知 k-means | 增强 |
| 码本坍塌 | 未讨论 | 从近期/高收益潜变量样本中重置 dead code | 增强 |
| 利润语义 | 未显式约束 | Return bucket 辅助目标 + 轻量 usage-profit alignment + codebook separation | 增强 |
| 检查点选择 | 未指定 | 周期性检查点评估 + 严格利润门控，之后才物化 `{PAIR}_vq_model.pt` | 增强 |

`docs/2026-04-19_phase1_dp_structure_optimization_log.md` 中的关键实验发现是：只有重构质量还不够；只有当 Phase I 原型语义与收益结构对齐时，Phase II 才会改善。当前有效配方是 return bucket objective + light usage-profit alignment + strict profit gate。

### Phase II - 原型选择

| 方面 | 论文 | 当前代码 | 类型 |
|---|---|---|---|
| 目标函数 | Eq. (5)：horizon reward + alpha KL 到示范标签 | PPO clipped surrogate + value loss + entropy bonus + imitation/KL term | 增强 |
| Selection alpha | 实验设置中为常数 alpha=1 | 默认 `selection_alpha=0.5`，并线性调度到 0 | 设计变更 |
| 模仿目标 | VQ 分配的示范原型 | 与 one-hot NLL 等价的 KL，并按 raw horizon return 做 mask（默认 `phase2_imitation_min_raw_return=0.0`） | 增强 |
| Return scale | 原始 horizon return | Batch 标准化 return 和归一化 advantage | 工程变更 |
| Actor / critic 更新 | 未指定 | actor 和 critic 使用独立优化器；critic 使用更高学习率 | 工程变更 |
| Rollout | 未指定 | 批量解码器推理和向量化 horizon 执行 | 工程变更 |
| 推理动作 | 论文描述 policy distribution | 验证/评估使用对原型概率的 greedy argmax | 设计变更 |
| 诊断 | 未讨论 | learned/random/oracle/fixed baselines、原型直方图、执行成本/换手率/direct-flip 统计 | 增强 |
| 代码组织 | 单一 selector 概念 | 重构为 `config.py`、`data_loader.py`、`rollout.py`、`evaluation.py`、`diagnostics.py`、`checkpoint.py` | 工程变更 |

实现注意：Phase II 当前通过 `horizon_indices` 直接索引 DP 轨迹缓存来获得 imitation labels。当前 Phase I 轨迹生成使用随机采样的滑动窗口起点，因此这些缓存轨迹不保证与 Phase II rollout 使用的非重叠窗口完全一致。如果需要严格复现 Eq. (5) 的同窗口标签协议，需要重新检查这里的对齐方式。

### Phase III - 原型精炼

| 方面 | 论文 | 当前代码 | 类型 |
|---|---|---|---|
| 流水线状态 | 第三阶段是完整方法的一部分 | 已实现，但当前 `run_pipeline.sh` 中被注释 | 当前工作流选择 |
| 职责 | 在固定 Phase II 选择之后做局部动作精炼 | 一致；Phase III 不重新选择原型 | 一致 |
| `tau_remain` | 绝对剩余步数 | 训练和推理中使用归一化 `(h - step_idx) / h` | 设计变更 |
| `R_arche` | 原始累计奖励 | 使用 `notional = m * p_0` 归一化，以稳定上下文尺度 | 增强 |
| 架构 | 概念上使用 AdaLN | 3 层 MLP，带 residual + LayerNorm + AdaLN conditioning | 增强 |
| 目标函数 | Eq. (9)，带遗憾感知奖励和 CE | PPO clipped surrogate + value loss + hindsight CE + entropy | 增强 |
| Batch 收集 | 未指定 | 每次 update 通过 `phase3_num_envs` 收集多个 horizon，提高 GPU 利用率 | 工程变更 |
| Hindsight optimal | Top-5 DP adaptations | 使用 LOB-aware costs 的向量化候选评估 | 增强 |
| Beta 调参 / 检查点 | 通过验证集在 `{0.3, 0.5, 0.7}` 中调 `beta1`，`beta2=1` | CLI 暴露 `--beta1/--beta2`，默认 `beta1=0.5`；训练只保存该 beta 下的 final model，没有内置 beta sweep | 设计变更 |

`docs/2026-04-12_phase3_optimization_log.md` 记录了一次 AL 的 Phase III 训练已完成，但最终操作序列与 Phase II 评估相同。因此当前代码库的优先级是先改进 Phase I/II，再把 Phase III 扩展为更高层级的 selector-correction 模块。

### 全局 / 配置

| 方面 | 论文 | 当前代码 | 类型 |
|---|---|---|---|
| 主要数据资产 | BTC/ETH/DOT/BNB 加密货币交易对 | 当前存在 AL 数据；ETH 配置仍保留但数据缺失 | 设计变更 |
| 数据集切分日期 | 论文报告固定 train/val/test 日历区间 | 当前 `FeaturePipeline` 直接加载已经切好的 feather 文件；`Config.train_start/val_start/test_start` 是报告元数据，不参与行过滤 | 实现注意 |
| 结果目录 | 未指定 | `result/{PAIR}/{BATCH_ID}/{stage}`，便于并行实验隔离 | 增强 |
| 特征配置 | 固定状态定义 | CLI `--cycle-feature-sets` 控制 state dim；checkpoint loaders 会检查 state_dim 一致性 | 增强 |
| 佣金处理 | 单一佣金率 | 评估、训练和 DP 使用分离的佣金率 | 设计变更 |
| 评估基础设施 | 论文指标 | PortfolioTracker、TradeAuditor、BacktraderVerifier、DP baseline、分块 CSV 导出 | 增强 |
| 数据集分析 | 未讨论 | `scripts/analyze_dataset.py` 将 shift/oracle 报告写入 `docs/` | 增强 |
| 论文 baseline / ablation | 报告 DQN/PPO/CDQNRP/CLSTM-PPO/EarnHFT/MacroHFT/IV/MACD，以及 VQ/refinement/regret 消融 | 当前仓库聚焦 ArchetypeTrader 主流水线；这些外部 baseline 和完整论文消融 runner 没有作为一等脚本实现 | 范围差异 |
| 硬件设置 | 论文实验使用 4 张 RTX-4090 | 当前脚本在可用 CUDA/CPU 上单进程运行；GPU guard 工具用于降低小显存环境风险 | 工程变更 |
| 论文复现守卫 / 注释 | 论文值以 manuscript 为准 | 部分 inline comments 和 legacy `PAPER_PHASE1_SPEC`/"strict paper" 日志文案已经过时或被禁用；应以 `Config`、当前代码路径和本 README 为准 | 文档注意 |
| 开发日志 | 不属于论文 | `docs/` 记录 Phase I/II/III 的优化决策和实验结果 | 文档 |

### 当前注意事项汇总

1. **当前脚本中 Phase III 是可选的**：`run_pipeline.sh` 当前会在 Phase II 评估后停止，除非取消 Phase III 代码块的注释或手动运行。
2. **pair 默认值可能影响评估**：`Config.pairs` 包含 `["AL", "ETH"]`，当 ETH 数据不可用时请使用 `--pair AL`。
3. **特征集一致性很重要**：Phase I、Phase II、Phase III、评估和诊断应使用相同的 `--cycle-feature-sets`。
4. **若要严格复现论文，Phase II 标签对齐需要复查**：当前 DP 缓存采样和 Phase II 非重叠 horizon 索引不保证指向完全相同的窗口。
5. **Phase III hindsight 代码有一路使用类级佣金 fallback**：`src/phase3/regret_reward.py` 在 `compute_top5_hindsight_optimal` 中读取 `env.COMMISSION_RATE`；如果要精确比较 Phase III 训练成本，需要复查。
6. **评估默认可能加载 Phase III**：`scripts/evaluate.py` 默认会使用 Phase III，除非传入 `--stage-label phase2_eval`；这点很重要，因为当前默认流水线不会训练 Phase III。
7. **Config 日期字段只是预切分文件的元数据**：代码直接加载 `df_train/df_val/df_test.feather`，不会按 `Config` 中的日期做过滤。
8. **历史 docs 包含中间实验**：例如 `docs/2026-04-03_phase1_decoder_optimization_log.md` 包含较早的 MLP/LSTM/teacher-forcing 变体；当前代码使用 BiLSTM + constrained decoding。

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@inproceedings{zong2026archetypetrader,
  title     = {ArchetypeTrader: Reinforcement Learning for Selecting and Refining Learnable Strategic Archetypes in Quantitative Trading},
  author    = {Zong, Chuqiao and Qin, Molei and Xia, Haochong and An, Bo},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```

## 许可

本项目是用于学术研究目的的实现。
