# Bugfix Requirements Document

## Introduction

VQ-VAE 训练中存在潜在的 codebook collapse 鲁棒性风险：`VQCodebook` 使用 `nn.Embedding` 默认随机初始化，初始码本向量可能不在 `z_e` 的实际分布空间中。虽然之前的修复（temporal attention pooling + Phase A 连续预训练）在当前实验中已经解决了 collapse 问题（ETH: `used_code_count=10`, `codebook_perplexity=9.93`, `dominant_code_ratio=0.123`），但随机初始化在不同数据集/随机种子组合下仍可能导致部分码本条目成为 dead codes，降低 codebook 利用率。

本修复引入 k-means 初始化策略作为额外的鲁棒性保障：在 Phase A（连续预训练）结束后、Phase B（VQ 训练）开始前，收集 Phase A 最后一个 epoch 的所有 `z_e` 向量，对其执行 k-means 聚类（K=`num_archetypes`=10），用聚类中心替换码本的随机初始值。这确保码本向量从一开始就分布在 `z_e` 的实际空间中，消除因初始化不良导致的 dead codes 风险。

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN Phase B VQ 训练开始且码本仅通过 `nn.Embedding` 随机初始化（无 k-means） THEN 系统的初始码本向量可能远离 `z_e` 的实际分布空间，在某些数据/种子组合下导致部分条目永远不被选为最近邻，成为 dead codes。

1.2 WHEN 码本随机初始化的向量恰好聚集在 `z_e` 分布的某个局部区域 THEN 系统中多个码本条目竞争同一区域的 `z_e` 样本，而 `z_e` 分布的其他区域无码本覆盖，导致 codebook 利用率下降。

1.3 WHEN 不同交易对（BTC/ETH/DOT/BNB）或不同随机种子下训练 THEN 系统的 codebook 利用率不稳定，`used_code_count` 可能因初始化运气而在不同运行间产生较大波动。

### Expected Behavior (Correct)

2.1 WHEN Phase A 连续预训练的最后一个 epoch 执行时 THEN 系统 SHALL 收集该 epoch 所有 batch 的 `z_e` 向量（detached，不影响梯度计算）。

2.2 WHEN Phase A 结束且 Phase B 开始之前 THEN 系统 SHALL 对收集的 `z_e` 向量执行 k-means 聚类（K=`num_archetypes`，使用 k-means++ 初始化选择分散的初始中心），并用聚类中心覆盖 `codebook.embeddings.weight`。

2.3 WHEN k-means 初始化完成后 Phase B VQ 训练开始 THEN 系统 SHALL 使码本向量分布在 `z_e` 的实际空间中，使所有 K 个条目都有机会被选为最近邻，提高 codebook 利用率的鲁棒性。

2.4 WHEN `z_e` 样本数少于码本大小 K THEN 系统 SHALL 跳过 k-means 初始化并记录警告日志，回退到默认随机初始化。

2.5 WHEN k-means 聚类过程中出现空簇（某个聚类中心没有分配到任何样本） THEN 系统 SHALL 从最大簇中随机采样一个点作为该空簇的新中心，确保所有 K 个中心都有效。

### Unchanged Behavior (Regression Prevention)

3.1 WHEN 编码器接收输入轨迹 `(batch, h, state_dim)` THEN 系统 SHALL CONTINUE TO 产生 `z_e` 形状为 `(batch, 16)`，梯度正常流过编码器所有参数。

3.2 WHEN VQ 量化执行时（`codebook.quantize()` 方法） THEN 系统 SHALL CONTINUE TO 使用最近邻查找 `k = argmin ||z_e - e_i||²` 和 straight-through 梯度估计器，`quantize()` 方法的接口和行为不变。

3.3 WHEN 解码器接收状态和 `z_q` THEN 系统 SHALL CONTINUE TO 产生动作 logits 形状为 `(batch, h, 3)`，有效动作值域为 `{0, 1, 2}`。

3.4 WHEN 完整 VQ 损失计算时 THEN 系统 SHALL CONTINUE TO 使用公式 `L = L_rec + ||sg[z_e] - z_q||² + β₀ × ||z_e - sg[z_q]||²`（Eq. 4），损失公式不变。

3.5 WHEN Phase A 连续预训练阶段（epoch 1-10） THEN 系统 SHALL CONTINUE TO 跳过 VQ 量化，仅使用 `L_rec` 损失训练编码器和解码器。

3.6 WHEN 论文核心超参数检查时 THEN 系统 SHALL CONTINUE TO 保持 K=10、latent_dim=16、hidden_dim=128、β₀=0.25、phase1_epochs=100、pretrain_epochs=10 不变。

3.7 WHEN 模型保存和验证时 THEN 系统 SHALL CONTINUE TO 保存 encoder、codebook、decoder 状态字典以及训练历史，并通过 Phase I 验证。
