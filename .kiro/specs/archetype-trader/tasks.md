# 实现计划：ArchetypeTrader

## 概述

基于 PyTorch 的三阶段强化学习量化交易框架实现。按照基础设施 → 数据层 → 环境层 → Phase I → Phase II → Phase III → 训练脚本 → 评估引擎 → 属性测试的顺序，逐步构建完整系统。每个任务增量构建，确保无孤立代码。

## 任务

- [x] 1. 基础设施：项目结构与全局配置
  - [x] 1.1 创建项目目录结构和包初始化文件
    - 创建 `src/__init__.py`, `src/data/__init__.py`, `src/env/__init__.py`, `src/phase1/__init__.py`, `src/phase2/__init__.py`, `src/phase3/__init__.py`, `src/evaluation/__init__.py`, `src/utils/__init__.py`, `tests/__init__.py`
    - 创建 `result/` 及其子目录 `dp_trajectories/`, `phase1_archetype_discovery/`, `phase2_archetype_selection/`, `phase3_archetype_refinement/`, `evaluation/`
    - 创建 `scripts/` 目录
    - _需求: 7.7, 9.4_

  - [x] 1.2 实现全局配置模块 `src/config.py`
    - 定义 `Config` dataclass，包含所有超参数：数据路径、特征维度（36/9/45）、MDP 配置（action_dim=3, horizon=72, commission_rate=0.0002）、持仓量映射、Phase I/II/III 训练参数、数据划分时间范围、评估年化因子
    - 支持通过命令行参数覆盖默认配置
    - _需求: 1.1, 1.2, 1.6, 2.3, 4.2, 4.3, 5.5, 6.7, 6.8, 7.5_

  - [x] 1.3 实现日志工具 `src/utils/logger.py`
    - 提供统一的日志接口，支持 INFO/WARNING/ERROR 级别
    - 训练过程中输出损失值、奖励等关键指标
    - _需求: 7.6, 9.2_

- [x] 2. 数据层：特征管道
  - [x] 2.1 实现 Feature Pipeline `src/data/feature_pipeline.py`
    - 实现 `FeaturePipeline` 类：`load_single_features()` 加载 36 维特征、`load_trend_features()` 加载 9 维特征、`get_state_vector()` 拼接为 45 维状态向量
    - 实现维度校验：加载时检查 single_features 最后一维为 36、trend_features 最后一维为 9，不匹配时抛出 `ValueError`
    - 实现 `split_by_date()` 按时间范围划分训练/验证/测试集
    - 实现 `split_into_horizons()` 按 h=72 切分数据
    - 支持按交易对（BTC/ETH/DOT/BNB）组织数据
    - _需求: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 2.2 实现 Dataset 封装 `src/data/dataset.py`
    - 创建 PyTorch Dataset 类，封装 DP 轨迹数据用于 VQ 训练
    - 支持 (s_demo, a_demo, r_demo) 三元组的批量加载
    - _需求: 4.7_

  - [x] 2.3 编写特征管道单元测试
    - 测试加载实际 .npy 文件
    - 测试维度不匹配时的错误处理
    - 测试按交易对组织数据
    - _需求: 1.1, 1.2, 1.4, 1.5_

- [x] 3. 环境层：MDP 交易环境
  - [x] 3.1 实现 MDP Trading Environment `src/env/trading_env.py`
    - 实现 `TradingEnv` 类：定义状态空间、动作空间 {0,1,2}、持仓映射 POSITION_MAP
    - 实现 `reset(horizon_idx)` 重置到指定 horizon 起始状态
    - 实现 `step(action)` 执行交易动作，返回 (next_state, reward, done, info)
    - 实现奖励计算：r_step_t = P_t × (p_{t+1} - p_t) - O_t（论文 Eq. 1）
    - 实现 `compute_fill_cost()` 基于 LOB 深度计算成交价差
    - 实现 `compute_execution_cost()` 计算总执行损失 = fill cost + 佣金（δ=0.02%）
    - 维护持仓状态 P_t ∈ {-m, 0, m}，m 按交易对映射
    - Episode 在 h=72 步后终止
    - 添加论文步骤注释（# Section 3.1: MDP 定义）
    - _需求: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 9.1_

  - [x] 3.2 编写 MDP 环境单元测试
    - 测试无效动作输入的错误处理
    - 测试具体交易场景的奖励计算
    - _需求: 2.2, 2.4_

- [x] 4. Checkpoint — 确保基础设施和数据层测试通过
  - 确保所有测试通过，如有问题请向用户确认。

- [x] 5. Phase I：DP Planner + VQ Encoder-Decoder
  - [x] 5.1 实现 DP Planner `src/phase1/dp_planner.py`
    - 实现 `DPPlanner` 类，接收 `TradingEnv` 实例
    - 实现 `plan()` 方法：Algorithm 1 单次交易约束 DP（反向填表 + 前向追踪）
    - 状态表 V[N+1, |A|, 2]，策略表 Π[N, |A|, 2]，约束 c ∈ {0, 1}
    - 实现 `generate_trajectories(num=30000)` 批量生成示范轨迹
    - 无有效交易路径时输出全 flat 轨迹并记录 WARNING
    - 保存轨迹到 `result/dp_trajectories/{pair}_trajectories.npz`
    - 添加论文步骤注释（# Algorithm 1, Step 1/2/3）
    - _需求: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 9.1_

  - [x] 5.2 编写 DP Planner 单元测试
    - 测试全 flat 轨迹的边界情况
    - 测试小规模轨迹生成
    - _需求: 3.7, 3.4_

  - [x] 5.3 实现 VQ Codebook `src/phase1/codebook.py`
    - 实现 `VQCodebook(nn.Module)`：K=10 个可学习向量，维度 16
    - 实现 `quantize()` 方法：最近邻查找 k = argmin_j ||z_e - e_j||²，返回 z_q、indices、commitment_loss
    - 使用 stop-gradient 操作
    - _需求: 4.3, 4.4, 4.6, 9.1_

  - [x] 5.4 实现 VQ Encoder `src/phase1/vq_encoder.py`
    - 实现 `VQEncoder(nn.Module)`：LSTM 架构，hidden_dim=128，输出 z_e 维度 16
    - 输入 (s_demo, a_demo, r_demo)，输出连续嵌入 z_e
    - 添加论文注释（# Section 4.1: LSTM-based encoder）
    - _需求: 4.1, 4.2, 9.1_

  - [x] 5.5 实现 VQ Decoder `src/phase1/vq_decoder.py`
    - 实现 `VQDecoder(nn.Module)`：根据状态和 z_q 生成动作 logits
    - 输入 states (batch, h, state_dim) 和 z_q (batch, code_dim)，输出 action_logits (batch, h, 3)
    - _需求: 4.5, 9.1_

  - [x] 5.6 编写 VQ 模块单元测试
    - 测试码本初始化
    - 测试梯度流（stop-gradient 正确性）
    - _需求: 4.3, 4.6_

- [x] 6. Checkpoint — 确保 Phase I 组件测试通过
  - 确保所有测试通过，如有问题请向用户确认。

- [x] 7. Phase II：Selection Agent
  - [x] 7.1 实现 Selection Agent `src/phase2/selection_agent.py`
    - 实现 `SelectionAgent(nn.Module)`：输入 state_dim，输出 K=10 个原型的概率分布和状态价值
    - 实现 `forward()` 返回 (action_probs, value)
    - 实现 `select_archetype()` 推理时选择原型索引
    - 添加论文注释（# Section 4.2: Horizon-level RL agent）
    - _需求: 5.1, 5.2, 9.1_

  - [x] 7.2 编写 Selection Agent 单元测试
    - 测试具体状态下的原型选择
    - 测试模型保存/加载一致性
    - _需求: 5.2_

- [x] 8. Phase III：Refinement Agent + Policy Adapter + AdaLN
  - [x] 8.1 实现 AdaLN 模块 `src/phase3/adaln.py`
    - 实现 `AdaptiveLayerNorm(nn.Module)`：AdaLN(x, c) = γ(c) × LayerNorm(x) + β(c)
    - 输入 feature_dim 和 condition_dim，条件化市场状态
    - 添加论文注释（# Section 4.3: Adaptive Layer Normalization）
    - _需求: 6.6, 9.1_

  - [x] 8.2 实现 Refinement Agent `src/phase3/refinement_agent.py`
    - 实现 `RefinementAgent(nn.Module)`：输入 s_ref1（市场观测）和 s_ref2（上下文 [e_a_sel, a_base, R_arche, τ_remain]）
    - 使用 AdaLN 条件化 s_ref1
    - 输出调整信号概率分布 {-1, 0, 1} 和状态价值
    - 添加论文注释（# Section 4.3: Step-level policy adapter）
    - _需求: 6.1, 6.3, 6.6, 9.1_

  - [x] 8.3 实现 Policy Adapter `src/phase3/policy_adapter.py`
    - 实现 `PolicyAdapter` 类：`compute_final_action()` 按论文 Eq. 6 计算最终动作
    - 维护 `adjusted_in_horizon` 标志，确保每 horizon 最多一次调整
    - 实现 `reset()` 在新 horizon 开始时重置
    - 添加论文注释（# Eq. 6: 最终动作计算）
    - _需求: 6.2, 6.3, 9.1_

  - [x] 8.4 实现 Regret-aware Reward 计算
    - 在 Refinement Agent 训练逻辑中实现 r_ref = (R - R_base) + β_1 × (R - R_1_opt)
    - 实现 top-5 hindsight-optimal adaptations 计算
    - 支持 β_1 ∈ {0.3, 0.5, 0.7} 调优
    - _需求: 6.4, 6.5, 6.8_

  - [x] 8.5 编写 Refinement Agent 单元测试
    - 测试 AdaLN 条件化输出
    - 测试具体 regret reward 计算示例
    - _需求: 6.3, 6.4, 6.6_

- [x] 9. Checkpoint — 确保 Phase II/III 组件测试通过
  - 确保所有测试通过，如有问题请向用户确认。

- [x] 10. 训练脚本
  - [x] 10.1 实现 Phase I 训练脚本 `scripts/train_phase1.py`
    - 加载特征数据，初始化 TradingEnv
    - 调用 DPPlanner 生成 30k 示范轨迹并保存
    - 初始化 VQ Encoder、Codebook、Decoder
    - 训练 100 epochs，损失函数 L = L_rec + ||sg[z_e] - z_q||² + 0.25 × ||z_e - sg[z_q]||²
    - 保存模型到 `result/phase1_archetype_discovery/`
    - 输出训练日志（loss 曲线）
    - 支持命令行参数指定交易对和超参数
    - _需求: 7.1, 4.6, 4.7, 4.8, 7.5, 7.6, 7.7_

  - [x] 10.2 实现 Phase II 训练脚本 `scripts/train_phase2.py`
    - 加载 Phase I 的码本和冻结 Decoder（检查文件存在性，不存在则报错）
    - 初始化 SelectionAgent
    - 训练 3M 步，目标函数含 KL 惩罚（α=1）
    - 定期在验证集上评估，保存最优检查点
    - 保存模型到 `result/phase2_archetype_selection/`
    - 支持命令行参数
    - _需求: 7.2, 5.3, 5.4, 5.5, 5.7, 7.4, 7.5, 7.6, 7.7_

  - [x] 10.3 实现 Phase III 训练脚本 `scripts/train_phase3.py`
    - 加载 Phase I 和 Phase II 模型（检查文件存在性）
    - 初始化 RefinementAgent、PolicyAdapter、AdaLN
    - 训练 1M 步，使用 regret-aware reward（β_1 可配置，β_2=1）
    - 保存模型到 `result/phase3_archetype_refinement/`
    - 支持命令行参数（含 β_1 选择）
    - _需求: 7.3, 6.4, 6.7, 6.8, 7.4, 7.5, 7.6, 7.7_

- [x] 11. 评估引擎
  - [x] 11.1 实现评估指标模块 `src/evaluation/metrics.py`
    - 实现 `EvaluationEngine` 类，年化因子 m=52560
    - 实现 `compute_total_return()`: TR = Π(1 + r_t) - 1
    - 实现 `compute_annual_volatility()`: AVOL = σ(r) × √m
    - 实现 `compute_max_drawdown()`: MDD = max(peak - trough) / peak
    - 实现 `compute_annual_sharpe_ratio()`: ASR = mean(r) / σ(r) × √m
    - 实现 `compute_annual_calmar_ratio()`: ACR = mean(r) / MDD × m
    - 实现 `compute_annual_sortino_ratio()`: ASoR = mean(r) / DD × √m
    - 实现 `evaluate()` 汇总所有指标
    - 处理除零边界情况（σ=0 或 MDD=0 时返回 0.0 并记录 WARNING）
    - _需求: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [x] 11.2 实现评估脚本 `scripts/evaluate.py`
    - 加载三阶段模型
    - 在测试集（2024-01-01 至 2024-09-01）上运行完整推理流程
    - 按交易对分别输出评估结果
    - 保存结果到 `result/evaluation/`
    - _需求: 8.7, 8.8, 8.9_

  - [x] 11.3 编写评估指标单元测试
    - 测试已知收益序列的指标计算
    - 测试除零边界情况
    - 测试空收益序列错误处理
    - _需求: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [x] 12. Checkpoint — 确保训练脚本和评估引擎测试通过
  - 确保所有测试通过，如有问题请向用户确认。

- [x] 13. 属性测试（Property-Based Tests）
  - [x] 13.1 编写特征管道属性测试 `tests/test_feature_pipeline.py`
    - **Property 1: 特征维度验证** — 验证 single_features 最后一维为 36，trend_features 最后一维为 9，拼接后为 45
    - **验证: 需求 1.1, 1.2, 1.3**

  - [x] 13.2 编写特征拼接属性测试 `tests/test_feature_pipeline.py`
    - **Property 2: 特征拼接保持内容不变** — 拼接后前 36 列等于 single_features，后 9 列等于 trend_features
    - **验证: 需求 1.3**

  - [x] 13.3 编写时间划分属性测试 `tests/test_feature_pipeline.py`
    - **Property 3: 时间划分不重叠且完整覆盖** — 三个集合时间范围互不重叠，合并后覆盖完整范围
    - **验证: 需求 1.6**

  - [x] 13.4 编写 Horizon 切分属性测试 `tests/test_feature_pipeline.py`
    - **Property 4: Horizon 切分长度一致** — 除最后一个片段外每个长度为 h，拼接还原原始序列
    - **验证: 需求 1.7**

  - [x] 13.5 编写持仓状态属性测试 `tests/test_trading_env.py`
    - **Property 5: 持仓状态不变量** — P_t 始终属于 {-m, 0, m}
    - **验证: 需求 2.3**

  - [x] 13.6 编写奖励计算属性测试 `tests/test_trading_env.py`
    - **Property 6: 奖励计算公式正确性** — r = P_t × (p_{t+1} - p_t) - O_t
    - **验证: 需求 2.4**

  - [x] 13.7 编写执行损失属性测试 `tests/test_trading_env.py`
    - **Property 7: 执行损失非负性** — O_t ≥ 0
    - **验证: 需求 2.5, 2.6**

  - [x] 13.8 编写 Episode 长度属性测试 `tests/test_trading_env.py`
    - **Property 8: Episode 长度不变量** — episode 在 h 步后终止
    - **验证: 需求 2.7**

  - [x] 13.9 编写 DP 单次交易约束属性测试 `tests/test_dp_planner.py`
    - **Property 9: DP 单次交易约束** — 持仓变化次数 ≤ 2
    - **验证: 需求 3.1**

  - [x] 13.10 编写 DP 最优性属性测试 `tests/test_dp_planner.py`
    - **Property 10: DP 最优性（小规模模型测试）** — 长度 ≤ 10 时 DP 收益等于暴力枚举最大收益
    - **验证: 需求 3.2**

  - [x] 13.11 编写 DP 轨迹结构属性测试 `tests/test_dp_planner.py`
    - **Property 11: DP 轨迹结构完整性** — s_demo shape (h, state_dim)，a_demo shape (h,) 值域 {0,1,2}，r_demo shape (h,)
    - **验证: 需求 3.5**

  - [x] 13.12 编写 VQ 维度属性测试 `tests/test_vq.py`
    - **Property 12: VQ 维度不变量** — z_e 维度 16，码本 K=10 × 16
    - **验证: 需求 4.2, 4.3**

  - [x] 13.13 编写最近邻量化属性测试 `tests/test_vq.py`
    - **Property 13: 最近邻量化正确性** — 选中索引 k 满足 ||z_e - e_k|| ≤ ||z_e - e_j|| ∀j
    - **验证: 需求 4.4**

  - [x] 13.14 编写解码器输出属性测试 `tests/test_vq.py`
    - **Property 14: 解码器输出有效动作** — argmax 后值域 {0, 1, 2}
    - **验证: 需求 4.5**

  - [x] 13.15 编写 VQ 损失函数属性测试 `tests/test_vq.py`
    - **Property 15: VQ 损失函数正确性** — L = L_rec + ||sg[z_e] - z_q||² + 0.25 × ||z_e - sg[z_q]||²
    - **验证: 需求 4.6**

  - [x] 13.16 编写 Selection Agent 输出属性测试 `tests/test_selection_agent.py`
    - **Property 16: Selection Agent 输出范围** — 索引 ∈ {0,...,K-1}，概率非负且和为 1
    - **验证: 需求 5.2**

  - [x] 13.17 编写冻结 Decoder 属性测试 `tests/test_selection_agent.py`
    - **Property 17: 冻结 Decoder 参数不变性** — Phase II/III 中参数与 Phase I 结束时相同
    - **验证: 需求 5.3**

  - [x] 13.18 编写每 Horizon 调整次数属性测试 `tests/test_refinement_agent.py`
    - **Property 18: 每 Horizon 最多一次调整** — 非零调整次数 ≤ 1
    - **验证: 需求 6.2**

  - [x] 13.19 编写 Refinement Agent 输出属性测试 `tests/test_refinement_agent.py`
    - **Property 19: Refinement Agent 输出范围** — a_ref ∈ {-1, 0, 1}
    - **验证: 需求 6.3**

  - [x] 13.20 编写 Regret-aware Reward 属性测试 `tests/test_refinement_agent.py`
    - **Property 20: Regret-aware Reward 计算正确性** — a_ref ≠ 0 时 r = (R - R_base) + β_1 × (R - R_1_opt)；a_ref = 0 时 r = 0
    - **验证: 需求 6.4**

  - [x] 13.21 编写 Top-5 Hindsight 排序属性测试 `tests/test_refinement_agent.py`
    - **Property 21: Top-5 Hindsight 排序** — 返回结果按收益降序排列
    - **验证: 需求 6.5**

  - [x] 13.22 编写最终动作计算属性测试 `tests/test_refinement_agent.py`
    - **Property 22: 最终动作计算正确性（Eq. 6）** — 验证 a_base ≠ a_base_prev 或 a_ref=0 → a_final=a_base；a_ref=-1 → a_final=0；a_ref=1 → a_final=2
    - **验证: 需求 6.2, 6.3**

  - [x] 13.23 编写评估指标属性测试 `tests/test_metrics.py`
    - **Property 23: 评估指标公式正确性** — 验证 TR/AVOL/ASR/ACR/ASoR/MDD 公式
    - **验证: 需求 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**

- [x] 14. 最终 Checkpoint — 确保所有测试通过
  - 确保所有测试通过，如有问题请向用户确认。

## 说明

- 标记 `*` 的子任务为可选测试任务，可跳过以加速 MVP 开发
- 每个任务引用具体需求编号以确保可追溯性
- Checkpoint 任务用于增量验证
- 属性测试使用 `hypothesis` 库，每个属性至少运行 100 次迭代
- 所有代码使用 Python + PyTorch 实现
