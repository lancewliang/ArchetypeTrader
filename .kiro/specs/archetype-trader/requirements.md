# 需求文档：ArchetypeTrader — 基于原型的强化学习加密货币交易框架

## 简介

ArchetypeTrader 是一个基于 PyTorch 的三阶段强化学习量化交易框架（AAAI26 论文）。该框架通过动态规划生成示范轨迹，使用向量量化（VQ）编码器-解码器发现可复用的交易原型，再由 RL agent 选择和精炼原型以执行交易。系统处理 10 分钟级别的加密货币数据（含 25 级限价订单簿），支持 BTC/ETH/DOT/BNB vs USDT 交易对。

## 术语表

- **DP_Planner**: 动态规划规划器，在单次交易约束下生成示范轨迹（Algorithm 1）
- **VQ_Encoder**: 基于 LSTM 的向量量化编码器，将示范轨迹 (s_demo, a_demo, r_demo) 编码为连续嵌入 z_e
- **VQ_Decoder**: 向量量化解码器，根据状态和量化嵌入重建动作序列
- **Codebook**: 可学习码本 ε = {e_0, ..., e_{K-1}}，存储 K 个离散交易原型向量
- **Selection_Agent**: 原型选择 RL agent，在 horizon 级别 MDP 中选择最合适的原型
- **Refinement_Agent**: 原型精炼 RL agent，在 step 级别 MDP 中对选定原型进行微调
- **Policy_Adapter**: 策略适配器，在每个 horizon 内最多进行一次动作调整
- **Horizon**: 一个交易周期，长度 h=72 步（即 72 个 10 分钟 bar）
- **Micro_Action**: 由冻结解码器根据原型码生成的逐步交易动作
- **LOB**: 限价订单簿（Limit Order Book），深度 M=25 级
- **State**: 市场观测向量，包含 LOB 数据、OHLCV bar、技术指标
- **Action**: 交易动作 a_t ∈ {0, 1, 2}，分别对应 short/flat/long
- **Position**: 持仓状态 P_t ∈ {-m, 0, m}，m 为最大持仓量
- **Reward**: 逐步奖励 r_step_t = V_t - V_{t-1} = P_t(p_mark_{t+1} - p_mark_t) - O_t
- **Execution_Cost**: 执行损失，包含 LOB fill cost 和佣金
- **Regret_Reward**: 遗憾感知奖励 r_ref = (R - R_base) + β_1(R - R_1_opt)
- **AdaLN**: Adaptive Layer Normalization，用于条件化市场状态
- **Feature_Pipeline**: 特征处理管道，加载并组织 single_features（36维）和 trend_features（9维）
- **Evaluation_Engine**: 评估引擎，计算 TR/AVOL/MDD/ASR/ACR/ASoR 等指标
- **Training_Pipeline**: 训练管道，协调三阶段的顺序训练流程
- **Result_Store**: 结果存储模块，将中间数据和模型检查点保存到 result 目录

## 需求

### 需求 1：数据加载与特征管道

**用户故事：** 作为量化研究员，我希望加载并组织多维市场特征数据，以便为后续的 DP 规划和 RL 训练提供标准化输入。

#### 验收标准

1. THE Feature_Pipeline SHALL 从 `data/feature_list/single_features.npy` 加载 36 维单步特征向量（包含 volume、bid/ask sizes、wap、spreads、vwap、log returns、K 线形态等）
2. THE Feature_Pipeline SHALL 从 `data/feature_list/trend_features.npy` 加载 9 维趋势特征向量（60 期趋势指标）
3. THE Feature_Pipeline SHALL 将单步特征和趋势特征拼接为完整的状态向量
4. WHEN 加载的特征文件维度与预期不符时, THE Feature_Pipeline SHALL 抛出包含实际维度和预期维度的错误信息
5. THE Feature_Pipeline SHALL 按交易对（BTC/ETH/DOT/BNB vs USDT）组织数据
6. THE Feature_Pipeline SHALL 按时间范围划分训练集（2021-06-01 至 2023-05-31）、验证集（2023-06-01 至 2023-12-31）和测试集（2024-01-01 至 2024-09-01）
7. THE Feature_Pipeline SHALL 提供按 horizon（h=72 步）切分数据的功能

### 需求 2：MDP 环境定义

**用户故事：** 作为量化研究员，我希望有一个符合论文定义的 MDP 交易环境，以便 DP 规划器和 RL agent 能在其中运行。

#### 验收标准

1. THE MDP_Environment SHALL 定义状态空间，包含 LOB 数据（M=25 级）、OHLCV bar 和技术指标
2. THE MDP_Environment SHALL 定义动作空间 a_t ∈ {0, 1, 2}，分别对应 short、flat、long
3. THE MDP_Environment SHALL 维护持仓状态 P_t ∈ {-m, 0, m}，其中 m 为交易对对应的最大持仓量（BTC=8, ETH=100, DOT=2500, BNB=200）
4. THE MDP_Environment SHALL 按公式 r_step_t = P_t × (p_mark_{t+1} - p_mark_t) - O_t 计算逐步奖励
5. THE MDP_Environment SHALL 计算执行损失 O_t，包含 LOB fill cost 和佣金（佣金率 δ=0.02%）
6. WHEN 动作导致持仓变化时, THE MDP_Environment SHALL 根据 LOB 深度计算实际成交价格（fill cost）
7. THE MDP_Environment SHALL 支持按 horizon 长度 h=72 步进行 episode 管理

### 需求 3：动态规划示范轨迹生成（Phase I 前置）

**用户故事：** 作为量化研究员，我希望使用动态规划在历史数据上生成最优示范轨迹，以便为原型发现提供训练数据。

#### 验收标准

1. THE DP_Planner SHALL 实现论文 Algorithm 1 中描述的单次交易约束动态规划算法
2. THE DP_Planner SHALL 在每个 horizon（h=72 步）内生成最优动作序列
3. THE DP_Planner SHALL 在规划过程中考虑执行损失（LOB fill cost + 佣金）
4. THE DP_Planner SHALL 生成 30,000 条示范轨迹用于原型发现训练
5. THE DP_Planner SHALL 为每条轨迹输出 (s_demo, a_demo, r_demo) 三元组
6. THE Result_Store SHALL 将生成的示范轨迹保存到 `result/dp_trajectories/` 目录
7. IF DP_Planner 在某个 horizon 内无法找到有效交易路径, THEN THE DP_Planner SHALL 输出全 flat（动作全为 1）的轨迹并记录警告日志

### 需求 4：VQ 编码器-解码器与原型发现（Phase I）

**用户故事：** 作为量化研究员，我希望通过向量量化编码器-解码器将示范轨迹压缩为离散的交易原型，以便后续 RL agent 复用这些原型。

#### 验收标准

1. THE VQ_Encoder SHALL 使用 LSTM 架构，隐藏层维度为 128，将示范轨迹 (s_demo, a_demo, r_demo) 编码为连续嵌入 z_e
2. THE VQ_Encoder SHALL 输出维度为 16 的连续嵌入向量 z_e
3. THE Codebook SHALL 维护 K=10 个可学习的原型向量，每个向量维度为 16
4. WHEN VQ_Encoder 输出 z_e 时, THE Codebook SHALL 通过最近邻查找选择最近的码本条目 z_q = e_k, 其中 k = argmin_j ||z_e - e_j||
5. THE VQ_Decoder SHALL 根据状态 s 和量化嵌入 z_q 重建动作序列
6. THE VQ_Encoder_Decoder SHALL 使用损失函数 L = L_rec + ||sg[z_e] - z_q||^2 + β_0 × ||z_e - sg[z_q]||^2 进行训练，其中 β_0=0.25，sg 表示 stop-gradient 操作
7. THE Training_Pipeline SHALL 使用 30,000 条 DP 示范轨迹训练 VQ 编码器-解码器，训练 100 个 epoch
8. THE Result_Store SHALL 将训练好的 VQ 编码器-解码器模型和码本保存到 `result/phase1_archetype_discovery/` 目录

### 需求 5：原型选择 RL Agent（Phase II）

**用户故事：** 作为量化研究员，我希望训练一个 RL agent 在每个 horizon 开始时选择最合适的交易原型，以便根据当前市场状态执行最优交易策略。

#### 验收标准

1. THE Selection_Agent SHALL 在 horizon 级别 MDP M_sel = ⟨S_sel, A_sel, R_sel, γ⟩ 中运行
2. THE Selection_Agent SHALL 在每个 horizon 开始时，根据当前市场状态 s_sel 选择一个原型索引 a_sel ∈ {0, 1, ..., K-1}
3. WHEN Selection_Agent 选择原型索引后, THE VQ_Decoder（冻结参数）SHALL 根据选定的原型码和当前状态生成 horizon 内的 micro actions 序列
4. THE Selection_Agent SHALL 使用目标函数 J = E[Σ(γ^t × r_sel_t - α × KL(â_sel_t || π_sel(a_sel_t | s_sel_t)))] 进行训练，其中 α=1
5. THE Training_Pipeline SHALL 训练 Selection_Agent 共 3,000,000 步
6. THE Result_Store SHALL 将训练好的 Selection_Agent 模型保存到 `result/phase2_archetype_selection/` 目录
7. WHILE Selection_Agent 训练过程中, THE Training_Pipeline SHALL 定期在验证集上评估性能并保存最优模型检查点

### 需求 6：原型精炼 RL Agent（Phase III）

**用户故事：** 作为量化研究员，我希望训练一个精炼 agent 对选定原型的执行进行微调，以便在实际交易中获得更好的收益。

#### 验收标准

1. THE Refinement_Agent SHALL 在 step 级别 MDP M_ref = {S_ref, A_ref, R_ref} 中运行
2. THE Policy_Adapter SHALL 在每个 horizon 内最多进行一次动作调整
3. THE Refinement_Agent SHALL 输出调整信号 a_ref ∈ {-1, 0, 1}，分别对应减仓、不变、加仓
4. THE Refinement_Agent SHALL 使用 regret-aware reward：r_ref = (R - R_base) + β_1 × (R - R_1_opt) 进行训练
5. THE DP_Planner SHALL 计算 top-5 hindsight-optimal adaptations 用于计算 R_1_opt
6. THE Refinement_Agent SHALL 使用 adaptive layer normalization（AdaLN）条件化市场状态
7. THE Training_Pipeline SHALL 训练 Refinement_Agent 共 1,000,000 步，β_2=1
8. THE Training_Pipeline SHALL 支持 β_1 在 {0.3, 0.5, 0.7} 范围内调优
9. THE Result_Store SHALL 将训练好的 Refinement_Agent 模型保存到 `result/phase3_archetype_refinement/` 目录

### 需求 7：训练脚本与流程编排

**用户故事：** 作为量化研究员，我希望有独立的训练脚本分别运行三个阶段的训练，以便灵活控制训练流程和调试各阶段。

#### 验收标准

1. THE Training_Pipeline SHALL 提供独立的 Phase I 训练脚本，执行 DP 轨迹生成和 VQ 编码器-解码器训练
2. THE Training_Pipeline SHALL 提供独立的 Phase II 训练脚本，加载 Phase I 的码本和冻结解码器，训练 Selection_Agent
3. THE Training_Pipeline SHALL 提供独立的 Phase III 训练脚本，加载 Phase I 和 Phase II 的模型，训练 Refinement_Agent
4. WHEN 前置阶段的模型文件不存在时, THE Training_Pipeline SHALL 抛出明确的错误信息，指示需要先完成哪个阶段的训练
5. THE Training_Pipeline SHALL 在每个训练脚本中支持通过命令行参数或配置文件指定超参数
6. THE Training_Pipeline SHALL 在训练过程中输出损失值、奖励等关键指标的日志
7. THE Result_Store SHALL 将所有中间数据（DP 轨迹、模型检查点、训练日志）保存到 `result/` 目录下的对应子目录

### 需求 8：评估与回测

**用户故事：** 作为量化研究员，我希望在测试集上评估完整的三阶段交易系统，以便量化策略的实际表现。

#### 验收标准

1. THE Evaluation_Engine SHALL 计算 Total Return（TR）指标
2. THE Evaluation_Engine SHALL 计算 Annual Volatility（AVOL）指标
3. THE Evaluation_Engine SHALL 计算 Maximum Drawdown（MDD）指标
4. THE Evaluation_Engine SHALL 计算 Annual Sharpe Ratio（ASR）指标
5. THE Evaluation_Engine SHALL 计算 Annual Calmar Ratio（ACR）指标
6. THE Evaluation_Engine SHALL 计算 Annual Sortino Ratio（ASoR）指标
7. THE Evaluation_Engine SHALL 在测试集（2024-01-01 至 2024-09-01）上运行完整的三阶段推理流程
8. THE Evaluation_Engine SHALL 支持按交易对分别输出评估结果
9. THE Result_Store SHALL 将评估结果保存到 `result/evaluation/` 目录

### 需求 9：代码质量与可维护性

**用户故事：** 作为量化研究员，我希望代码中包含论文步骤的对应注释和不确定部分的说明，以便理解实现与论文的对应关系。

#### 验收标准

1. THE Training_Pipeline SHALL 在代码中以注释形式标注论文中对应的步骤描述（如 "# Phase I, Step 3: 向量量化最近邻查找"）
2. WHEN 论文描述模糊或缺少实现细节时, THE Training_Pipeline SHALL 在对应代码位置添加 `# [NOTE: 论文未明确描述] ...` 注释说明
3. WHEN 实现细节需要合理假设时, THE Training_Pipeline SHALL 生成空方法（stub）并添加 `# [TODO: 论文未提供具体实现]` 注释
4. THE Training_Pipeline SHALL 使用 PyTorch 框架实现所有深度学习模型
5. THE Training_Pipeline SHALL 确保功能正确性为第一优先级，避免猜测生成论文中未描述的代码
