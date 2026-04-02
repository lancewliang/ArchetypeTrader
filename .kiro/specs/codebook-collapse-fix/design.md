# Codebook K-Means 初始化 Bugfix Design

## Overview

VQ codebook 使用 `nn.Embedding` 默认随机初始化，初始码本向量可能不在 `z_e` 的实际分布空间中，在不同数据集/随机种子组合下可能导致部分码本条目成为 dead codes。本修复在 Phase A（连续预训练，epoch 1-10）结束后、Phase B（VQ 训练，epoch 11-100）开始前，收集 Phase A 最后一个 epoch 的所有 `z_e` 向量，执行 k-means 聚类（K=10，k-means++ 初始化），用聚类中心覆盖 `codebook.embeddings.weight`。

修复范围严格限定在 Phase A → Phase B 过渡点的码本初始化逻辑，不改变论文公式 Eq(2)、Eq(4)、`quantize()` 方法、VQDecoder 或任何超参数。

## Glossary

- **Bug_Condition (C)**: Phase B VQ 训练开始时码本仅通过 `nn.Embedding` 随机初始化，向量远离 `z_e` 实际分布空间
- **Property (P)**: Phase B 开始时码本向量应分布在 `z_e` 的实际空间中，所有 K 个条目都有机会被选为最近邻
- **Preservation**: `quantize()` 方法、VQDecoder、损失公式 Eq(4)、Phase A/B 训练流程、超参数均不变
- **`VQCodebook.init_from_data()`**: `src/phase1/codebook.py` 中的方法，用 k-means 聚类中心初始化码本权重
- **`train_one_epoch()`**: `scripts/train_phase1.py` 中的单 epoch 训练函数，新增 `collect_z_e` 参数用于收集 z_e
- **`run_training_loop()`**: `scripts/train_phase1.py` 中的完整训练循环，在 Phase A 最后一个 epoch 触发 k-means 初始化
- **Phase A**: epoch 1-10，连续潜在预训练，跳过 VQ 量化，仅使用 L_rec 损失
- **Phase B**: epoch 11-100，完整 VQ 训练，L = L_rec + commitment + β₀ × encoder_commitment

## Bug Details

### Bug Condition

当 Phase B VQ 训练开始时，码本仅通过 `nn.Embedding` 随机初始化（默认 N(0,1)），初始码本向量可能远离 `z_e` 的实际分布空间。在某些数据集/随机种子组合下，部分码本条目永远不被选为最近邻，成为 dead codes，降低 codebook 利用率。

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type {codebook_weights: Tensor(K, D), z_e_distribution: Tensor(N, D)}
  OUTPUT: boolean

  LET centroids = kmeans(z_e_distribution, K)
  LET avg_dist_random = mean(min_distance(codebook_weights, z_e_distribution))
  LET avg_dist_kmeans = mean(min_distance(centroids, z_e_distribution))

  RETURN codebook_weights 是随机初始化（未经 k-means）
         AND avg_dist_random >> avg_dist_kmeans
         AND EXISTS k IN [0, K-1]: code_count[k] == 0 (dead code)
END FUNCTION
```

### Examples

- **正常情况**: Phase A 训练 10 个 epoch 后，z_e 分布在 16 维空间的某个子区域。随机初始化的码本向量分散在整个空间，部分向量远离 z_e 分布 → 这些向量成为 dead codes
- **k-means 初始化后**: 收集 ~30000 个 z_e 样本，k-means 聚类得到 10 个中心，每个中心都在 z_e 密集区域 → 所有码本条目都有样本分配
- **边界情况**: z_e 样本数 < K=10（极端小数据集）→ 跳过 k-means，回退到随机初始化并记录警告
- **空簇情况**: k-means 迭代中某个聚类中心没有分配到样本 → 从最大簇中随机采样一个点作为新中心

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- `VQCodebook.quantize()` 方法的接口和行为完全不变：最近邻查找 `k = argmin ||z_e - e_i||²` + straight-through 梯度估计器
- VQDecoder 接收状态和 `z_q` 产生动作 logits `(batch, h, 3)`，值域 `{0, 1, 2}`
- VQ 损失公式 `L = L_rec + ||sg[z_e] - z_q||² + β₀ × ||z_e - sg[z_q]||²`（Eq. 4）不变
- Phase A（epoch 1-10）跳过 VQ 量化，仅使用 `L_rec` 损失
- 编码器输出 `z_e` 形状 `(batch, 16)`，梯度正常流过所有参数
- 超参数 K=10、latent_dim=16、hidden_dim=128、β₀=0.25、phase1_epochs=100、pretrain_epochs=10 不变
- 模型保存和验证流程不变

**Scope:**
所有不涉及 Phase A → Phase B 过渡点码本初始化的行为完全不受影响。具体包括：
- Phase A 的所有训练逻辑（仅在最后一个 epoch 额外收集 z_e，不影响梯度和损失计算）
- Phase B 的所有训练逻辑（quantize、损失计算、死码重置）
- 编码器、解码器的前向/反向传播
- 数据加载、轨迹生成、模型保存

## Hypothesized Root Cause

基于 bug 分析，根本原因是：

1. **码本初始化与 z_e 分布不匹配**: `nn.Embedding` 默认使用 N(0,1) 初始化，而 Phase A 训练后的 z_e 分布可能集中在 16 维空间的某个特定子区域。随机初始化的码本向量可能远离这个子区域，导致部分条目永远不被选为最近邻。

2. **缺少数据驱动的初始化步骤**: 当前代码在 `VQCodebook.__init__()` 中直接使用 `nn.Embedding` 的默认初始化，没有利用 Phase A 已经训练好的编码器产生的 z_e 分布信息来初始化码本。

3. **已有的缓解措施不够鲁棒**: 虽然 `reset_dead_codes()` 在每个 Phase B epoch 后重置死码，但如果初始化严重偏离 z_e 分布，前几个 epoch 的训练可能已经将编码器推向不良方向，死码重置只能部分修复。

**解决方案**: 在 Phase A 最后一个 epoch 收集所有 z_e，用 k-means 聚类中心初始化码本，确保码本向量从一开始就在 z_e 的实际空间中。

## Correctness Properties

Property 1: Bug Condition — K-Means 初始化使码本覆盖 z_e 分布

_For any_ Phase A 训练后收集的 z_e 样本集（N ≥ K），经过 k-means 初始化后的码本权重 SHALL 满足：每个聚类中心都在 z_e 分布的密集区域内，且对 z_e 样本执行量化时 `used_code_count == K`（所有码本条目都被使用）。

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation — Quantize 方法和损失公式不变

_For any_ 输入 z_e 向量，无论码本是否经过 k-means 初始化，`quantize()` 方法 SHALL 产生相同结构的输出（z_q_st, indices, commitment_loss），且 VQ 损失公式 `L = L_rec + ||sg[z_e] - z_q||² + β₀ × ||z_e - sg[z_q]||²` 的计算逻辑不变，编码器输出形状 `(batch, 16)` 不变，解码器输出形状 `(batch, h, 3)` 不变。

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**


## Fix Implementation

### Changes Required

已有代码已经实现了大部分修复逻辑。以下是现有实现的确认和可能需要的调整：

**File**: `src/phase1/codebook.py`

**Method**: `VQCodebook.init_from_data()`

**现有实现确认**:
1. **k-means++ 初始化**: 已实现 — 第一个中心随机选取，后续中心按距离概率采样，确保初始中心分散
2. **k-means 迭代**: 已实现 — 默认 10 次迭代，每次重新分配样本到最近中心并更新中心
3. **空簇处理**: 已实现 — 从最大簇中随机采样一个点作为空簇的新中心
4. **样本数不足保护**: 已实现 — N < K 时跳过初始化并记录警告
5. **权重覆盖**: 已实现 — `self.embeddings.weight.copy_(centroids)`

**File**: `scripts/train_phase1.py`

**Function**: `train_one_epoch()`

**现有实现确认**:
1. **`collect_z_e` 参数**: 已添加 — 控制是否收集 z_e 向量
2. **z_e 收集逻辑**: 已实现 — `z_e_list.append(z_e.detach())`，不影响梯度计算
3. **返回值**: 已更新 — 返回 `(metrics, z_e_all)`，z_e_all 为拼接后的所有 z_e 样本

**Function**: `run_training_loop()`

**现有实现确认**:
1. **Phase A 最后一个 epoch 标记**: 已实现 — `collect_z_e = (epoch == config.pretrain_epochs)`
2. **k-means 初始化调用**: 已实现 — `codebook.init_from_data(z_e_all)` 在 Phase A 最后一个 epoch 后调用
3. **时机正确**: 在 `train_one_epoch()` 返回后、下一个 epoch 开始前执行

**无需新增外部依赖**: k-means 完全用 PyTorch 实现（`torch.cdist`、`torch.multinomial`），不依赖 sklearn。

### Specific Changes

当前代码已经包含了完整的 k-means 初始化实现。需要验证的关键点：

1. **`init_from_data()` 的 k-means++ 初始化是否正确**: 验证概率采样是否按距离平方加权
2. **空簇处理是否鲁棒**: 验证 `torch.bincount` 在所有边界情况下正确工作
3. **z_e 收集是否不影响训练**: 验证 `detach()` 确保不影响梯度图
4. **内存管理**: 30000 × 16 的 z_e 张量约 1.9MB，内存开销可忽略

## Testing Strategy

### Validation Approach

测试策略分两阶段：首先在未修复代码上展示 bug（随机初始化导致的 dead codes），然后验证 k-means 初始化修复了问题且不引入回归。

### Exploratory Bug Condition Checking

**Goal**: 在未应用 k-means 初始化的代码上展示 bug — 随机初始化的码本在量化 z_e 时产生 dead codes。

**Test Plan**: 创建随机初始化的码本，用模拟的 z_e 分布（集中在某个子区域）执行量化，观察 dead codes 的出现。

**Test Cases**:
1. **随机初始化 vs z_e 分布**: 生成集中在某个区域的 z_e 样本，用随机初始化码本量化，观察 used_code_count < K（将在未修复代码上失败）
2. **多种子测试**: 用不同随机种子重复测试，观察 dead codes 出现的频率（将在未修复代码上不稳定）
3. **极端分布测试**: z_e 分布极度集中时，随机初始化码本的大部分条目成为 dead codes（将在未修复代码上失败）

**Expected Counterexamples**:
- 随机初始化码本量化 z_e 时 `used_code_count < K`
- 不同随机种子下 `used_code_count` 波动较大

### Fix Checking

**Goal**: 验证对所有满足 bug condition 的输入（N ≥ K 的 z_e 样本集），k-means 初始化后码本覆盖 z_e 分布。

**Pseudocode:**
```
FOR ALL z_e_samples WHERE len(z_e_samples) >= K DO
  codebook = VQCodebook(K, D)
  codebook.init_from_data(z_e_samples)
  z_q_st, indices, _ = codebook.quantize(z_e_samples)
  ASSERT len(unique(indices)) == K  // 所有码本条目都被使用
  ASSERT mean(min_dist(codebook.weight, z_e_samples)) < threshold  // 码本在 z_e 分布内
END FOR
```

### Preservation Checking

**Goal**: 验证对所有不涉及 k-means 初始化的输入，修复后的代码行为与原代码完全一致。

**Pseudocode:**
```
FOR ALL z_e WHERE z_e.shape == (batch, 16) DO
  // quantize() 行为不变
  ASSERT quantize_original(z_e) == quantize_fixed(z_e)
  // 编码器输出形状不变
  ASSERT encoder(s, a, r).shape == (batch, 16)
  // 解码器输出形状不变
  ASSERT decoder(s, z_q).shape == (batch, h, 3)
  // 损失公式不变
  ASSERT vq_loss_original(z_e) == vq_loss_fixed(z_e)
END FOR
```

**Testing Approach**: 使用 property-based testing（Hypothesis）生成随机输入，验证 quantize()、编码器、解码器的行为在修复前后一致。

**Test Cases**:
1. **Quantize 接口保持**: 验证 quantize() 返回 (z_q_st, indices, commitment_loss) 的形状和类型不变
2. **Straight-through 梯度保持**: 验证 z_q_st 的梯度仍然流过 z_e
3. **损失公式保持**: 验证 commitment_loss = ||sg[z_e] - z_q||² 不变
4. **编码器输出保持**: 验证 z_e 形状 (batch, 16) 不变
5. **解码器输出保持**: 验证 logits 形状 (batch, h, 3) 不变

### Unit Tests

- `init_from_data()` 基本功能：K=10, N=1000 的 z_e 样本，验证码本权重被更新
- `init_from_data()` 边界情况：N < K 时跳过初始化，码本权重不变
- `init_from_data()` 空簇处理：构造会产生空簇的数据，验证所有中心都有效
- `init_from_data()` k-means++ 初始化：验证初始中心足够分散
- `train_one_epoch()` z_e 收集：验证 collect_z_e=True 时返回正确形状的 z_e_all
- `train_one_epoch()` z_e 不影响训练：验证 collect_z_e 不改变损失值和梯度

### Property-Based Tests

- 生成随机 z_e 分布（不同均值、方差、维度），验证 k-means 初始化后所有码本条目都被使用
- 生成随机 z_e 输入，验证 quantize() 的最近邻正确性在 k-means 初始化前后一致
- 生成随机输入轨迹，验证编码器→码本→解码器的完整流水线输出形状不变
- 生成随机 z_e，验证 VQ 损失分解公式在 k-means 初始化前后一致

### Integration Tests

- 完整 Phase A → k-means 初始化 → Phase B 流程：验证 `run_training_loop()` 在 Phase A 最后一个 epoch 正确触发 k-means 初始化
- 验证 k-means 初始化后 Phase B 第一个 epoch 的 `used_code_count == K`
- 验证模型保存和加载后码本权重与 k-means 初始化结果一致
