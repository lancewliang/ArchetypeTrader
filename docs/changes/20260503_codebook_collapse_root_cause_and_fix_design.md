# Phase I Codebook 坍塌根因分析与修复设计

**日期**: 2026-05-03
**影响阶段**: Phase I Archetype Discovery
**目标**: 诊断当前 ArchetypeTrader codebook 坍塌的根本原因，对比 ArchetypeTrader1 的成功经验，设计修复方案

---

## 1. 问题陈述

当前 ArchetypeTrader 的 Phase I 训练频繁出现 codebook 坍塌现象：

- **硬坍塌**：大部分 code 不被使用（`code_usage_ratio < 0.5`），1~2 个 code 占据 90%+ 样本
- **软坍塌**：表面 `code_usage_ratio` 达标，但多个 code 解码出的 action 序列几乎相同（`inter_code_action_diversity < 0.15`），codebook 向量余弦相似度 > 0.9
- **方向性坍塌**：所有 code 都预测 flat（action=1），long/short 方向完全丢失

现有的防护机制（KL-uniform 正则、dead code restart、selection guardrail）只能**检测和事后修复**，无法从根源**预防**坍塌发生。

---

## 2. 对比分析：ArchetypeTrader1 为什么不坍塌

### 2.1 架构差异总览

| 机制 | ArchetypeTrader1 | ArchetypeTrader | 影响程度 |
|------|------------------|-----------------|----------|
| 两阶段训练（Phase A + Phase B） | ✅ 10 epoch 预训练 | ❌ 直接 VQ 训练 | **致命** |
| Usage-Profit Alignment Loss | ✅ 鼓励高收益 code 被使用 | ❌ 只有 KL-uniform | **致命** |
| Codebook Separation Loss | ✅ 惩罚 code 向量相似 | ❌ 无 | **严重** |
| Return Bucket Auxiliary Loss | ✅ 给 code 有意义的身份 | ❌ 无 | **严重** |
| 方向感知初始化 | ✅ 按 long/short/flat 分组 k-means | ❌ 普通 k-means | **中等** |
| Dead Code Restart 频率 | 每 epoch | 每 5 epoch | **中等** |
| Dead Code Restart 策略 | 高收益样本优先 + 低使用率码重置 | 仅高重构误差样本 | **轻微** |

### 2.2 差异 1：两阶段训练（最根本差异）

#### ArchetypeTrader1 的做法

```
Phase A (epoch 1~10):
  encoder → z_e → decoder (跳过 VQ)
  loss = L_rec only
  → encoder+decoder 学会有意义的表示

Phase A 结束:
  收集所有 z_e → 方向感知 k-means → 初始化 codebook

Phase B (epoch 11~300):
  encoder → z_e → VQ quantize → z_q → decoder
  loss = L_rec + commitment + alignment + separation + return_aux
```

关键：Phase A 结束时，encoder 已经训练了 10 个 epoch，输出的 `z_e` 包含有意义的交易模式信息。对这些有意义的向量做 k-means + 方向感知分组，codebook 从一开始就覆盖了 long/short/flat 的真实分布。

代码位置：`ArchetypeTrader1/scripts/train_phase1.py:741-751`（Phase A 跳过 VQ）、`train_phase1.py:928-939`（Phase A 结束时初始化 codebook）

#### ArchetypeTrader 的做法

```
Warmup (训练前):
  随机 encoder → 随机 z_e → k-means → ≈随机 codebook

训练 (epoch 1~100):
  encoder → z_e → VQ quantize → z_q → decoder
  loss = L_rec + codebook + commitment + usage_KL
```

关键：warmup 时 encoder 还没训练过，输出的是随机噪声。对随机向量做 k-means，得到的聚类中心本质上还是随机的——和 `random_normal` 初始化没有本质区别。

代码位置：`ArchetypeTrader/src/trainers/phase1_trainer.py:586-622`（warmup）、`phase1_trainer.py:624-823`（训练循环）

#### 为什么这个区别致命

VQ 训练存在鸡生蛋问题：

1. **好的 codebook** 需要 encoder 输出有意义的 `z_e` 来初始化
2. **好的 encoder** 需要好的 codebook 来学习有意义的表示（commitment loss 把 `z_e` 拉向 code）

ArchetypeTrader 的 warmup 没有打破这个循环——encoder 还是随机的，k-means 初始化的 codebook 也近似随机，训练一开始就在"随机 codebook + 随机 encoder"的状态下，很容易坍塌到单一 code。

ArchetypeTrader1 的 Phase A **直接绕过了 VQ**，先让 encoder+decoder 学会重构，打破了鸡生蛋问题。等 encoder 能输出有意义的 `z_e` 后再初始化 codebook，codebook 从一开始就处于好的位置，Phase B 只需微调。

### 2.3 差异 2：Usage-Profit Alignment Loss

#### ArchetypeTrader1 的做法

```python
def compute_usage_profit_alignment_loss(soft_assignments, trajectory_returns, target_corr):
    """鼓励高收益 archetype 具有更高使用率，避免收益-使用率负相关。"""
    # 计算 soft assignment 加权的 code 使用率和 code 平均收益
    # 计算 Pearson 相关系数
    # 相关系数低于 target_corr 时惩罚
    loss = torch.relu(target_corr - corr)
```

代码位置：`ArchetypeTrader1/scripts/train_phase1.py:237-257`

这个损失直接鼓励高收益的 archetype 被更多使用，创造了 codebook 使用率与收益之间的**正反馈**。即使某个 code 暂时使用率低，只要它对应高收益，这个损失就会推动模型使用它。

#### ArchetypeTrader 的做法

只有 `KL(U(K) || p_code)` 均匀分布正则（`vq_losses.py:146-153`），这是一个**弱约束**——它只鼓励均匀分布，不关心每个 code 的质量。当某个 code 的重构误差很大时，均匀正则会强制 encoder 使用它，但 commitment loss 又会把 encoder 拉向更好的 code，两者矛盾导致训练不稳定。

### 2.4 差异 3：Codebook Separation Loss

#### ArchetypeTrader1 的做法

```python
def compute_codebook_separation_loss(embeddings, margin):
    """惩罚过高的 codebook cosine 相似度，降低 archetype 同质化。"""
    normalized = F.normalize(embeddings, dim=1)
    cosine = normalized @ normalized.t()
    penalties = torch.relu(cosine[off_diag_mask] - margin)
    return torch.mean(penalties ** 2)
```

代码位置：`ArchetypeTrader1/scripts/train_phase1.py:260-274`

直接惩罚码本向量过于相似，防止多个 code 收敛到同一位置（软坍塌）。

#### ArchetypeTrader 的做法

完全没有这个损失。虽然 `evaluation/metrics/behavior.py` 有 `inter_code_action_diversity` 和 `decoder_sensitivity_to_code` 指标来**检测**软坍塌，但**没有训练时的损失来预防它**。这就像只有报警器没有灭火器。

### 2.5 差异 4：Return Bucket Auxiliary Loss

#### ArchetypeTrader1 的做法

```python
return_bucket_head = nn.Sequential(
    nn.LayerNorm(latent_dim),
    nn.Linear(latent_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, num_buckets),
)
```

代码位置：`ArchetypeTrader1/scripts/train_phase1.py:277-286`

这个辅助头强制每个 codebook 向量预测其对应 archetype 的收益分布。给每个 code 一个有意义的"身份"——如果两个 code 预测的收益分布相同，它们就没有存在的必要。这从根本上防止了 code 退化为无意义的重复。

#### ArchetypeTrader 的做法

没有任何类似的收益预测机制。

### 2.6 差异 5：方向感知初始化

#### ArchetypeTrader1 的做法

```python
def init_from_data_direction_aware(self, z_e_samples, actions, trajectory_returns, ...):
    """方向感知的码本初始化：先按交易方向分组，再在组内 k-means。
    将 K 个码本条目按方向分配（如 4 long + 4 short + 2 flat），
    在每个方向子集内做 k-means，确保码本覆盖所有方向。"""
```

代码位置：`ArchetypeTrader1/src/phase1/codebook.py:112-250`

确保 codebook 从一开始就覆盖 long/short/flat 三个方向。LSTM decoder 有很强的方向性偏置（倾向于预测 flat），如果初始化时没有覆盖所有方向，decoder 很快就会让所有 code 都预测 flat，导致方向性坍塌。

#### ArchetypeTrader 的做法

只有普通 k-means 初始化（`vector_quantizer.py:133-154`），不保证方向覆盖。

### 2.7 差异 6：Dead Code Restart 策略

| 特性 | ArchetypeTrader1 | ArchetypeTrader |
|------|------------------|-----------------|
| 频率 | 每 epoch | 每 5 epoch（评估时） |
| 触发条件 | `code_counts == 0` 或低使用率 | EMA count < 0.1 |
| 重置来源 | 最后 batch 的 z_e + 高收益样本优先 | 高重构误差样本 |
| 低使用码重置 | ✅ 支持 | ❌ 不支持 |

ArchetypeTrader1 的 dead code restart 更激进（每 epoch），且优先用高收益样本重置——这确保被重启的 code 会被分配到有价值的区域。ArchetypeTrader 的 restart 间隔太长（5 epoch），且只从高重构误差样本重置，可能把 code 放到"困难但无价值"的区域。

---

## 3. 根因总结

ArchetypeTrader 的防坍塌体系是**检测型**的：

```
训练 → 坍塌发生 → 检测（KL-uniform / behavior metrics / selection guardrail）→ 修复（dead code restart）
```

ArchetypeTrader1 的防坍塌体系是**预防型**的：

```
Phase A 预训练 → 有意义的初始化 → 训练中持续约束（alignment + separation + return bucket）→ 坍塌不会发生
```

**根本问题**：ArchetypeTrader 把所有防坍塌机制都放在了"训练后"（评估、选择、重启），而 ArchetypeTrader1 把它们放在了"训练中"（损失函数）。前者只能检测和修复，后者能从根源预防。

---

## 4. 为什么现有机制修不好

### 4.1 KL-uniform 正则不够

- 只鼓励均匀分布，不关心 code 质量
- 与 commitment loss 矛盾：均匀正则强制使用差 code，commitment 拉向好 code
- 权重太小（0.01）无法对抗重构损失的强大梯度

### 4.2 Dead code restart 治标不治本

- 重启后的 code 很快再次变死——因为训练动态（重构损失 + commitment）仍然偏向已有 code
- 5 epoch 的间隔太长，坍塌在这期间已经固化
- 高重构误差样本不等于高价值样本

### 4.3 Selection guardrail 只能拒绝坏结果

- 只阻止坍塌的 checkpoint 被选为 best
- 不改善训练过程本身
- 如果所有 epoch 都坍塌，训练直接 fatal

### 4.4 Warmup 不等于 Phase A

- warmup 时的 encoder 是随机的，k-means 中心近似随机
- Phase A 先训练 encoder 10 epoch，k-means 中心基于有意义的 z_e
- 这是"写 10 个随机数字再分组"vs"学 10 天数学再分组"的区别

---

## 5. 技术设计变更

### 5.1 变更总览

本次变更将 ArchetypeTrader 的防坍塌体系从"检测型"升级为"预防型"，核心改动 4 项，辅助改动 3 项：

| 编号 | 变更项 | 优先级 | 涉及文件 | 与论文公式关系 |
|------|--------|--------|----------|---------------|
| C1 | Phase A 预训练 | P0 | `phase1_trainer.py`, `phase1_config.py`, `vq_archetype.py` | 不改变公式 (4)，Phase A 只用 L_rec |
| C2 | Usage-Profit Alignment Loss | P0 | `vq_losses.py`, `phase1_config.py`, `phase1_trainer.py` | 扩展训练 loss，需记录配置 |
| C3 | Codebook Separation Loss | P1 | `vq_losses.py`, `phase1_config.py` | 扩展训练 loss，需记录配置 |
| C4 | Return Bucket Auxiliary Loss | P1 | `vq_archetype.py`, `vq_losses.py`, `phase1_config.py`, `phase1_trainer.py` | 扩展训练 loss，需记录配置 |
| C5 | 方向感知初始化 | P2 | `vector_quantizer.py`, `phase1_trainer.py` | 不改变公式，只改变初始化 |
| C6 | Dead Code Restart 增强 | P2 | `vector_quantizer.py`, `phase1_trainer.py` | 不改变公式，只改变重启策略 |
| C7 | Soft Assignment 计算 | P0（C2 前置） | `vq_losses.py` | 不改变公式，为 C2 提供输入 |

### 5.2 C1: Phase A 预训练

#### 5.2.1 设计目标

在 VQ 训练之前，先训练 encoder+decoder 联合学习重构（跳过 VQ 量化），使 encoder 输出有意义的 `z_e`，为后续 codebook 初始化提供高质量输入。

#### 5.2.2 配置变更

在 `TrainingConfig` 中新增：

```python
@dataclass(frozen=True)
class TrainingConfig:
    # ... 现有字段 ...
    pretrain_epochs: int = 10          # Phase A 预训练 epoch 数
    pretrain_lr: Optional[float] = None  # Phase A 学习率，None 表示复用 training.lr
```

#### 5.2.3 模型变更

在 `VQArchetypeModel` 中新增 `forward_pretrain` 方法：

```python
class VQArchetypeModel(nn.Module):
    def forward_pretrain(self, states, actions, rewards):
        """Phase A 前向：跳过 VQ，z_e 直接传给 decoder。"""
        fused = self.input_adapter(states, actions, rewards)
        z_e = self.encoder(fused)
        logits = self.decoder(states, z_e)  # z_e 不走 quantizer
        return ModelOutputs(
            action_logits=logits,
            z_e=z_e,
            z_q=z_e,           # Phase A 中 z_q = z_e
            z_q_no_grad=z_e.detach(),
            code_id=None,       # Phase A 无 code_id
        )
```

#### 5.2.4 损失函数变更

在 `Phase1Loss` 中新增 `forward_pretrain` 方法：

```python
class Phase1Loss(nn.Module):
    def forward_pretrain(self, *, action_logits, target_actions):
        """Phase A 损失：仅重构 CE。"""
        b, h, c = action_logits.shape
        rec = F.cross_entropy(action_logits.reshape(b * h, c), target_actions.reshape(b * h))
        return LossOutputs(
            total=rec,
            reconstruction=rec,
            codebook=torch.tensor(0.0),
            commitment=torch.tensor(0.0),
            usage=None,
            contrastive=None,
        )
```

#### 5.2.5 训练循环变更

在 `Phase1Trainer._train_loop` 中：

```
for epoch in range(epochs):
    is_phase_a = epoch < pretrain_epochs

    if is_phase_a:
        # Phase A: forward_pretrain + forward_pretrain loss
        outputs = model.forward_pretrain(states, actions, rewards)
        loss = loss_fn.forward_pretrain(
            action_logits=outputs.action_logits,
            target_actions=actions,
        )
    else:
        # Phase B: 正常 VQ 训练
        outputs = model(states, actions, rewards)
        loss = loss_fn(...)

    # Phase A → Phase B 过渡
    if epoch == pretrain_epochs - 1:
        # 收集 z_e，重新初始化 codebook
        self._warmup_codebook(model, train_horizons, normalizer)
```

#### 5.2.6 Warmup 时机变更

当前 warmup 在训练前执行（`_warmup_codebook` 在 `_train_loop` 之前调用）。变更后：

- Phase A 期间：codebook 不参与训练，不需要初始化
- Phase A 结束时：重新调用 `_warmup_codebook`，此时 encoder 已收敛
- Phase B 期间：codebook 基于有意义的 z_e 初始化

#### 5.2.7 `paper_strict_reproduction` 兼容

当 `paper_strict_reproduction=True` 时：
- `pretrain_epochs` 强制设为 0（跳过 Phase A）
- 不执行方向感知初始化
- 与论文公式 (4) 严格对齐

### 5.3 C7: Soft Assignment 计算（C2 前置）

#### 5.3.1 设计目标

为 Usage-Profit Alignment Loss 和 Return Bucket Auxiliary Loss 提供 soft code assignment。基于 encoder latent 与 codebook 距离的 softmax 概率。

#### 5.3.2 实现

在 `vq_losses.py` 中新增：

```python
def compute_soft_code_assignments(z_e, codebook, temperature):
    """基于 encoder latent 与 codebook 距离计算 soft assignment。

    Args:
        z_e: encoder 输出 [B, code_dim]
        codebook: codebook 向量 [K, code_dim]
        temperature: softmax 温度（越高越均匀）

    Returns:
        soft_assignments: [B, K]，每行是对各 code 的软分配概率
    """
    temp = max(float(temperature), 1e-6)
    distances = (
        torch.sum(z_e ** 2, dim=1, keepdim=True)
        - 2 * z_e @ codebook.t()
        + torch.sum(codebook ** 2, dim=1, keepdim=False)
    )
    return torch.softmax(-distances / temp, dim=1)
```

### 5.4 C2: Usage-Profit Alignment Loss

#### 5.4.1 设计目标

鼓励高收益 archetype 具有更高使用率，避免收益-使用率负相关。这是从 ArchetypeTrader1 移植的核心防坍塌机制。

#### 5.4.2 配置变更

在 `CodebookHealthConfig` 中新增：

```python
@dataclass(frozen=True)
class CodebookHealthConfig:
    # ... 现有字段 ...
    usage_profit_alignment_weight: float = 0.1
    usage_profit_alignment_target_corr: float = 0.3
    usage_profit_alignment_temperature: float = 2.0
```

#### 5.4.3 损失函数变更

在 `Phase1Loss` 中新增：

```python
def _usage_profit_alignment(
    self,
    soft_assignments,     # [B, K]
    trajectory_returns,   # [B]
    target_corr: float,   # 目标相关系数
    eps: float = 1e-6,
):
    """鼓励高收益 archetype 具有更高使用率。"""
    if soft_assignments.shape[0] < 2 or soft_assignments.shape[1] < 2:
        return soft_assignments.new_zeros(())

    returns = trajectory_returns.reshape(-1, 1)
    code_mass = soft_assignments.sum(dim=0).clamp_min(eps)
    usage = code_mass / soft_assignments.shape[0]
    code_returns = (soft_assignments * returns).sum(dim=0) / code_mass

    usage_centered = usage - usage.mean()
    return_centered = code_returns - code_returns.mean()
    covariance = torch.mean(usage_centered * return_centered)
    denom = torch.sqrt(
        torch.mean(usage_centered ** 2) * torch.mean(return_centered ** 2) + eps
    )
    corr = covariance / denom
    target = soft_assignments.new_tensor(float(np.clip(target_corr, -1.0, 1.0)))
    return torch.relu(target - corr)
```

#### 5.4.4 训练循环变更

在 `Phase1Loss.forward` 中：

```python
# 新增参数
def forward(self, *, action_logits, target_actions, z_e, z_q_no_grad, code_id,
            contrastive_pair_ids=None,
            trajectory_returns=None,       # 新增
            codebook=None,                # 新增
            soft_assignment_temperature=2.0,  # 新增
):
    # ... 现有计算 ...

    # Usage-profit alignment
    alignment = None
    if self.usage_profit_alignment_weight > 0 and trajectory_returns is not None and codebook is not None:
        soft_assignments = compute_soft_code_assignments(
            z_e, codebook, soft_assignment_temperature
        )
        alignment = self._usage_profit_alignment(
            soft_assignments, trajectory_returns,
            self.usage_profit_alignment_target_corr,
        )
        total = total + self.usage_profit_alignment_weight * alignment
```

在 `LossOutputs` 中新增 `alignment` 字段。

#### 5.4.5 训练循环传入 trajectory_returns

在 `Phase1Trainer._train_loop` 中，需要将 `trajectory_returns`（每个样本的 DP 累积收益）传入 loss 函数。这需要：

1. `Phase1DemoDataset` 的 collate 函数返回 `trajectory_returns` 字段
2. 训练循环中从 batch 取出 `trajectory_returns`
3. 传入 `loss_fn(trajectory_returns=..., codebook=model.quantizer.codebook, ...)`

#### 5.4.6 `paper_strict_reproduction` 兼容

当 `paper_strict_reproduction=True` 时，`usage_profit_alignment_weight` 强制设为 0。

### 5.5 C3: Codebook Separation Loss

#### 5.5.1 设计目标

惩罚过高的 codebook cosine 相似度，防止多个 code 收敛到同一位置（软坍塌）。

#### 5.5.2 配置变更

在 `CodebookHealthConfig` 中新增：

```python
@dataclass(frozen=True)
class CodebookHealthConfig:
    # ... 现有字段 ...
    codebook_separation_weight: float = 0.01
    codebook_separation_margin: float = 0.5
```

#### 5.5.3 损失函数变更

在 `Phase1Loss` 中新增：

```python
def _codebook_separation(self, codebook, margin: float):
    """惩罚过高的 codebook cosine 相似度。"""
    if codebook.shape[0] < 2:
        return codebook.new_zeros(())

    normalized = F.normalize(codebook, dim=1)
    cosine = normalized @ normalized.t()
    off_diag_mask = ~torch.eye(cosine.shape[0], dtype=bool, device=cosine.device)
    penalties = torch.relu(cosine[off_diag_mask] - float(np.clip(margin, -1.0, 1.0)))
    if penalties.numel() == 0:
        return codebook.new_zeros(())
    return torch.mean(penalties ** 2)
```

在 `Phase1Loss.forward` 中：

```python
separation = None
if self.codebook_separation_weight > 0 and codebook is not None:
    separation = self._codebook_separation(codebook, self.codebook_separation_margin)
    total = total + self.codebook_separation_weight * separation
```

#### 5.5.4 `paper_strict_reproduction` 兼容

当 `paper_strict_reproduction=True` 时，`codebook_separation_weight` 强制设为 0。

### 5.6 C4: Return Bucket Auxiliary Loss

#### 5.6.1 设计目标

给每个 codebook 向量一个有意义的"身份"——强制预测其对应 archetype 的收益分布。如果两个 code 预测的收益分布相同，它们就没有存在的必要。

#### 5.6.2 配置变更

在 `CodebookHealthConfig` 中新增：

```python
@dataclass(frozen=True)
class CodebookHealthConfig:
    # ... 现有字段 ...
    return_aux_weight: float = 0.1
    return_aux_hidden_dim: int = 32
    return_num_buckets: int = 5
    return_soft_assignment_weight: float = 0.5
```

#### 5.6.3 模型变更

在 `VQArchetypeModel` 中新增收益预测辅助头：

```python
class VQArchetypeModel(nn.Module):
    def __init__(self, feature_dim, config):
        # ... 现有初始化 ...
        self.return_bucket_head = nn.Sequential(
            nn.LayerNorm(config.code_dim),
            nn.Linear(config.code_dim, config.codebook.health.return_aux_hidden_dim),
            nn.GELU(),
            nn.Linear(config.codebook.health.return_aux_hidden_dim,
                      config.codebook.health.return_num_buckets),
        )
```

#### 5.6.4 损失函数变更

在 `Phase1Loss` 中新增：

```python
def _return_bucket_aux(
    self,
    codebook,                    # [K, code_dim]
    soft_assignments,            # [B, K]
    return_bucket_head,          # nn.Module
    return_bucket_targets,       # [B] long
    soft_weight: float,          # soft vs hard assignment 权重
):
    """收益分桶辅助损失：强制每个 code 预测其 archetype 的收益分布。"""
    code_bucket_logits = return_bucket_head(codebook)  # [K, num_buckets]
    soft_bucket_logits = soft_assignments @ code_bucket_logits  # [B, num_buckets]

    # hard assignment（取 argmin distance 的 code）
    hard_indices = soft_assignments.argmax(dim=1)
    hard_bucket_logits = code_bucket_logits[hard_indices]  # [B, num_buckets]

    soft_loss = F.cross_entropy(soft_bucket_logits, return_bucket_targets)
    hard_loss = F.cross_entropy(hard_bucket_logits, return_bucket_targets)
    return soft_weight * soft_loss + (1.0 - soft_weight) * hard_loss
```

#### 5.6.5 收益分桶边界构建

需要在训练开始前，基于训练集 trajectory returns 构建分桶边界：

```python
def build_return_bucket_edges(trajectory_returns, num_buckets):
    """根据全局轨迹收益分位数构建收益分桶边界。"""
    quantiles = np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]
    edges = np.quantile(trajectory_returns, quantiles)
    return np.unique(edges)
```

#### 5.6.6 `paper_strict_reproduction` 兼容

当 `paper_strict_reproduction=True` 时，`return_aux_weight` 强制设为 0。

### 5.7 C5: 方向感知初始化

#### 5.7.1 设计目标

在 Phase A 结束后、codebook 初始化时，按交易方向（long/short/flat）分组做 k-means，确保 codebook 覆盖所有方向。

#### 5.7.2 实现变更

在 `VectorQuantizer` 中新增 `warmup_initialize_direction_aware` 方法：

```python
@_no_grad()
def warmup_initialize_direction_aware(
    self,
    encoder_outputs,    # [N, code_dim]
    actions,            # [N, h] 值域 {0, 1, 2}
    trajectory_returns=None,
    profit_top_ratio=0.0,
    profit_code_ratio=0.0,
):
    """方向感知的码本初始化：先按交易方向分组，再在组内 k-means。"""
    # 判断每条轨迹的主要方向
    # 按 long/short/flat 分配码本条目（如 4+4+2）
    # 在每个方向子集内做 k-means
    # 同步 EMA buffer
```

逻辑移植自 `ArchetypeTrader1/src/phase1/codebook.py:112-250`。

#### 5.7.3 配置变更

在 `CodebookConfig` 中新增：

```python
@dataclass(frozen=True)
class CodebookConfig:
    # ... 现有字段 ...
    direction_aware_init: bool = True
    direction_aware_long_codes: int = 4
    direction_aware_short_codes: int = 4
    direction_aware_flat_codes: Optional[int] = None  # None = K - long - short
    profit_init_top_ratio: float = 0.0
    profit_init_code_ratio: float = 0.0
```

#### 5.7.4 训练循环变更

在 `Phase1Trainer._warmup_codebook` 中，当 `direction_aware_init=True` 且 Phase A 结束时，调用 `warmup_initialize_direction_aware` 而非 `warmup_initialize`。需要额外传入 `actions` 和 `trajectory_returns`。

### 5.8 C6: Dead Code Restart 增强

#### 5.8.1 设计目标

提高 dead code restart 的频率和策略有效性，解决当前 restart 机制的 5 个核心问题。

#### 5.8.2 当前 restart 机制的 5 个问题

**问题 1：频率过低（每 5 epoch vs 每 epoch）**

当前 restart 只在评估轮次（每 5 epoch）执行。4 个 epoch 的空窗期内，死码无法被恢复，坍塌会在这段时间内固化。一旦某个 code 变死，encoder 的 commitment loss 会把所有 z_e 拉向剩余的活跃 code，4 个 epoch 后死码周围已经没有 z_e 了，重启也难以挽回。

**问题 2：判定标准依赖 EMA count，反应迟钝**

当前用 `_ema_count < 0.1` 判定死码。EMA count 是衰减累积值，一个 code 可能在当前 epoch 完全没被使用，但因为历史 EMA count 还没衰减到 0.1 以下，所以不会被重启。等到 EMA count 衰减到 0.1 以下时，这个 code 已经死了好几个 epoch 了。

**问题 3：不支持低使用率码重置**

当前只重置完全死码（EMA count < 0.1），不支持低使用率码重置。这导致"僵尸码"问题——一个 code 每个 epoch 只被 1~2 个样本选中，使用率极低，但不算死码，不会被重启，却也不贡献有意义的 archetype。

**问题 4：重置来源用高重构误差样本，不可靠**

当前从重构误差最高的样本中取 z_e 重置。但高重构误差 ≠ 高价值。一个样本重构误差高可能是因为：
- 它是一个噪声样本，DP 标签本身就不稳定 → 重启到这是浪费
- 它是一个极端行情，大部分 code 都无法重构 → 重启后很快又会变死

高收益样本更可靠——如果一个样本的 DP 累积收益高，说明 DP teacher 认为它包含有价值的交易模式。

**问题 5：重启后 EMA count 初始化为 1.0，步长过大**

重启后 `_ema_count[code_id].fill_(1.0)`，而其他活跃 code 的 EMA count 可能是几百甚至几千。在下一个 batch 的 EMA 更新中，重启 code 的 N_i = 1.0，Laplace smoothing 中 `smoothed = (N_i + eps) / (n_total + K * eps) * n_total`，N_i 小时 smoothed 小，`new_codebook = m_i / smoothed` 步长大。大步长可能导致重启 code 在前几个 batch 内剧烈震荡，然后又变死。

#### 5.8.3 变更内容

**变更 1：频率从每 5 epoch 改为每 epoch**

在 `_train_loop` 中，将 `_maybe_restart_dead_codes` 的调用从 `if should_evaluate:` 块中移出，改为每 epoch 执行。

**变更 2：判定标准改为实际使用计数**

不再依赖 EMA count，改为在训练循环中统计每个 epoch 的 `code_counts`（`torch.bincount`），基于实际使用次数判定死码和低使用率码：

```python
# 在 _train_loop 的每个 epoch 末尾统计
epoch_code_counts = torch.bincount(
    torch.cat(epoch_code_ids), minlength=num_codes
).numpy()
```

传入 `restart_dead_codes` 作为判定依据。

**变更 3：支持低使用率码重置**

```python
def restart_dead_codes(self, encoder_outputs, reconstruction_errors, current_epoch,
                       trajectory_returns=None, code_usage_counts=None):
    # 判定死码：实际使用次数 == 0
    dead_mask = code_usage_counts == 0
    # 判定低使用率码：使用率 < low_usage_reset_threshold
    low_usage_mask = np.zeros_like(dead_mask, dtype=bool)
    if low_usage_reset_threshold > 0 and code_usage_counts.sum() > 0:
        usage_ratio = code_usage_counts / code_usage_counts.sum()
        low_usage_mask = (code_usage_counts > 0) & (usage_ratio < low_usage_reset_threshold)
    reset_mask = dead_mask | low_usage_mask
```

**变更 4：重置来源支持高收益样本优先**

新增 `restart_source` 选项：

```python
if restart_source == "high_return_samples" and trajectory_returns is not None:
    # 从高收益样本中取 z_e
    top_n = max(len(reset_indices), int(encoder_outputs.shape[0] * top_ratio))
    top_idx = torch.topk(trajectory_returns, k=top_n, largest=True).indices
    pool = encoder_outputs[top_idx]
elif restart_source == "high_reconstruction_error_samples":
    # 从高重构误差样本中取 z_e（现有行为）
    topk = torch.topk(reconstruction_errors, k=len(reset_indices), largest=True).indices
    pool = encoder_outputs[topk]
```

**变更 5：重启后 EMA count 初始化为活跃码中位数**

```python
# 重启后 EMA count 初始化策略
active_counts = self._ema_count[self._ema_count >= threshold]
if active_counts.numel() > 0:
    init_count = active_counts.median().item()
else:
    init_count = 1.0
self._ema_count[code_id].fill_(init_count)
```

这确保重启 code 的 EMA 更新步长与活跃 code 相近，避免剧烈震荡。

**变更 6：噪声注入**

重置时加 1e-3 高斯噪声，防止多个死码被重置到相同位置：

```python
for code_id, sample_idx in zip(reset_indices, sample_indices):
    sampled = pool[sample_idx % pool.shape[0]]
    noise = torch.randn_like(sampled) * 1e-3
    self.codebook.data[code_id].copy_(sampled + noise)
    self._ema_weight[code_id].copy_(sampled + noise)
    self._ema_count[code_id].fill_(init_count)
```

**变更 7：开销优化——采样子集代替全训练集遍历**

当前 `_maybe_restart_dead_codes` 每次遍历全训练集，开销大。改为采样 10% 训练集：

```python
def _maybe_restart_dead_codes(self, *, model, train_dataset, epoch):
    # 采样子集（10% 或至少 1024 个样本）
    subset_size = max(1024, len(train_dataset) // 10)
    indices = torch.randperm(len(train_dataset))[:subset_size]
    subset = torch.utils.data.Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=self.config.training.batch_size, ...)
    # ... 收集 z_e 和重构误差 ...
```

这使得每 epoch 执行 restart 的开销可接受。

#### 5.8.4 配置变更

在 `CodebookHealthConfig` 中新增/修改：

```python
@dataclass(frozen=True)
class CodebookHealthConfig:
    # ... 现有字段 ...
    dead_code_restart_every_epoch: bool = True     # 新增：每 epoch 执行
    low_usage_reset_threshold: float = 0.05        # 新增：低使用率重置阈值
    restart_source: str = "high_return_samples"    # 修改：默认改为高收益样本优先
    restart_high_return_top_ratio: float = 0.3     # 新增：高收益样本 top 比例
    restart_noise_std: float = 1e-3                # 新增：重置噪声标准差
    restart_ema_count_init: str = "active_median"  # 新增：EMA count 初始化策略
    restart_subset_ratio: float = 0.1              # 新增：采样子集比例
    restart_cooldown_epochs: int = 1               # 修改：从 3 缩短为 1
```

#### 5.8.5 VectorQuantizer 变更

`restart_dead_codes` 方法签名变更：

```python
def restart_dead_codes(
    self,
    encoder_outputs: "torch.Tensor",
    reconstruction_errors: "torch.Tensor",
    current_epoch: int,
    trajectory_returns: Optional["torch.Tensor"] = None,       # 新增
    code_usage_counts: Optional["np.ndarray"] = None,          # 新增
) -> List[int]:
```

#### 5.8.6 训练循环变更

1. 在每个 epoch 的训练循环中收集 `epoch_code_ids` 和 `epoch_trajectory_returns`
2. epoch 结束时计算 `code_usage_counts = torch.bincount(epoch_code_ids, minlength=K).numpy()`
3. 调用 `restart_dead_codes(..., trajectory_returns=..., code_usage_counts=...)`
4. 不再限制只在评估轮次执行

---

## 6. 变更依赖关系

```
C7 (Soft Assignment) ──→ C2 (Usage-Profit Alignment) ──→ C4 (Return Bucket)
                                                              │
C1 (Phase A 预训练) ──→ C5 (方向感知初始化)                    │
         │                                                     │
         └─────────────→ C3 (Separation Loss) ←────────────────┘
                                │
                                └──→ C6 (Dead Code Restart 增强)
```

建议实施顺序：C1 → C7 → C2 → C3 → C4 → C5 → C6

其中 C1 和 C2 是 P0 优先级，C3 和 C4 是 P1 优先级，C5 和 C6 是 P2 优先级。

---

## 7. `paper_strict_reproduction` 兼容性

所有新增损失和训练策略在 `paper_strict_reproduction=True` 时全部关闭，确保严格对齐论文公式 (4)：

| 变更项 | paper_strict 时的行为 |
|--------|----------------------|
| C1 Phase A | `pretrain_epochs=0`，跳过 Phase A |
| C2 Usage-Profit Alignment | `usage_profit_alignment_weight=0` |
| C3 Separation Loss | `codebook_separation_weight=0` |
| C4 Return Bucket | `return_aux_weight=0` |
| C5 方向感知初始化 | `direction_aware_init=False`，使用普通 k-means |
| C6 Dead Code Restart 增强 | `dead_code_restart=False`（沿用现有 paper_strict 行为） |

---

## 8. 测试策略

### 8.1 单元测试

| 测试项 | 文件 | 验证内容 |
|--------|------|----------|
| `test_forward_pretrain` | `test_vq_archetype.py` | Phase A 前向不走 VQ，z_q = z_e |
| `test_phase_a_loss_only_rec` | `test_vq_losses.py` | Phase A loss 只含 reconstruction |
| `test_soft_assignment_shape` | `test_vq_losses.py` | soft assignment 输出 [B, K]，行和为 1 |
| `test_usage_profit_alignment` | `test_vq_losses.py` | 高收益 code 使用率高时 loss 小 |
| `test_codebook_separation` | `test_vq_losses.py` | codebook 向量越相似 loss 越大 |
| `test_return_bucket_aux` | `test_vq_losses.py` | 辅助头输出 shape 正确 |
| `test_direction_aware_init` | `test_vector_quantizer.py` | 初始化后各方向 code 数量符合配置 |
| `test_low_usage_reset` | `test_vector_quantizer.py` | 低使用率码被正确识别和重置 |

### 8.2 集成测试

| 测试项 | 文件 | 验证内容 |
|--------|------|----------|
| `test_phase_a_to_b_transition` | `test_phase1_trainer.py` | Phase A 结束后 codebook 被正确初始化 |
| `test_no_collapse_with_new_mechanisms` | `test_phase1_collapse_handling.py` | 新机制下 code_usage_ratio ≥ 0.7 |
| `test_paper_strict_disables_all` | `test_phase1_config_docs.py` | paper_strict 关闭所有新增机制 |

### 8.3 回归测试

使用现有 smoke test fixture 验证：
- Phase I pipeline 端到端运行不报错
- 最终 code_usage_ratio ≥ 0.7
- composite_score 与改动前在同一量级

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Phase A 预训练时间增加 | 训练时间增加约 10% | Phase A 只用 L_rec，计算量小；可配置 `pretrain_epochs` |
| 新增损失超参多 | 调参复杂度增加 | 所有新超参有合理默认值；`paper_strict` 一键关闭 |
| Return Bucket 辅助头增加模型参数 | 显存增加 | 参数量极小（LayerNorm + 2 层 MLP，约 1K 参数） |
| 方向感知初始化需要 actions | warmup 时需额外数据 | Phase A 已收集 actions，无额外 IO |
| EMA 模式下 separation loss 梯度 | codebook 不走梯度，separation loss 无法直接优化 codebook | 改为在 `update_codebook` 中对 codebook 做 in-place 分离正则化，或切换到 gradient 模式 |

### 9.1 EMA 模式下 Separation Loss 的特殊处理

当前默认 `update_method="ema"`，codebook `requires_grad=False`，separation loss 的梯度无法传到 codebook。两种解决方案：

**方案 A**：在 `update_codebook` 的 EMA 更新后，对 codebook 做一次 in-place 分离正则化：

```python
def update_codebook(self, z_e, code_id):
    # ... EMA 更新 ...
    # Separation regularization (in-place)
    if self.config.health.codebook_separation_weight > 0:
        self._apply_separation_regularization()
```

**方案 B**：Phase B 前几个 epoch 使用 gradient 模式，后续切换到 EMA。

推荐方案 A，实现简单且不改变训练动态。

---

## 10. 实施计划

### Phase 1（P0，解决核心坍塌问题）

1. C1: Phase A 预训练
2. C7: Soft Assignment 计算
3. C2: Usage-Profit Alignment Loss

### Phase 2（P1，防止软坍塌）

4. C3: Codebook Separation Loss
5. C4: Return Bucket Auxiliary Loss

### Phase 3（P2，锦上添花）

6. C5: 方向感知初始化
7. C6: Dead Code Restart 增强

每个 Phase 完成后运行完整测试套件，确认无回归后再进入下一 Phase。

---

## 11. 采纳/不采纳决策记录（2026-05-03）

本节维护从设计方案到执行计划的取舍结论。执行计划见：

`docs/changes/20260503_codebook_collapse_fix_change_plan.md`

### 11.1 本轮采纳

| 编号 | 结论 | 原因 | 执行约束 |
|------|------|------|----------|
| C1 Phase A 预训练 | 采纳 | 直接解决随机 encoder 输出初始化 codebook 的根因；不新增模型参数，不改变 Phase II/III artifact 契约。 | 通过 `pretrain_epochs` 控制；`paper_strict_reproduction=True` 时强制关闭；Phase A 不进入 checkpoint selection。 |
| C7 Soft Assignment 计算 | 采纳 | C2 的低风险前置能力；纯函数实现，可充分单测。 | 只服务训练 loss，不改变 quantizer 的 hard assignment 推理路径。 |
| C2 Usage-Profit Alignment Loss | 采纳 | 补足 KL-uniform 不区分 code 质量的缺口；不改变模型结构。 | 使用原始 trajectory return，不使用 normalized reward；默认权重保守；strict 模式强制关闭。 |

### 11.2 本轮不采纳或暂缓

| 编号 | 结论 | 不采纳原因 | 后续重新评审条件 |
|------|------|------------|------------------|
| C3 Codebook Separation Loss | 暂缓，不纳入本轮 | 当前默认 `update_method="ema"` 下 codebook `requires_grad=False`，普通 loss 对 codebook 无直接优化效果；原设计中的 in-place separation 会重写 EMA 更新语义，风险较高。 | C1+C2 后仍出现软坍塌；单独设计 EMA 分离机制，并证明不会破坏 reconstruction 和 code usage。 |
| C4 Return Bucket Auxiliary Loss | 不采纳 | 需要新增辅助头，改变模型 state_dict 与 checkpoint 兼容面；还会牵涉 return bucket 边界、重尾收益分布和 artifact 切分策略。 | 仅当 C1+C2 后仍出现 code 身份不足，且 behavior metrics 指向收益身份缺失时再评审。 |
| C5 方向感知初始化 | 不采纳 | 会按 action 方向固定分配 code 容量，属于 codebook 语义和初始化策略改写；不同品种、`max_position` 与 no-trade 比例下固定配额不稳。 | C1 后仍出现方向性坍塌，并有诊断证明普通 k-means 未覆盖 long/short/flat。 |
| C6 Dead Code Restart 增强 | 不采纳 | 每 epoch restart、低使用率重置、高收益样本优先、EMA count 初始化等会同时改变训练轨迹，容易掩盖 C1/C2 是否修复根因；也会与现有 selection cooldown/fatal 逻辑交织。 | C1+C2 完成后仍有可复现 dead code，再逐项小步实验 restart 子策略。 |

### 11.3 测试要求纳入执行计划

本轮采纳变更必须包含对应单元测试代码和测试执行记录：

- `tests/unit/models/test_vq_archetype.py`: 覆盖 `forward_pretrain` 跳过 VQ。
- `tests/unit/models/test_vq_losses.py`: 覆盖 Phase A loss、soft assignment、usage-profit alignment。
- `tests/unit/data/test_dataset.py`: 覆盖 `trajectory_returns` 使用原始 reward。
- `tests/unit/trainers/test_phase1_trainer.py`: 覆盖 strict 关闭、Phase A 到 Phase B warmup 时机。
- `tests/unit/config/test_phase1_config_docs.py`: 覆盖新增配置文档字段。

最小执行命令记录在执行计划 §6。若集成 smoke 因 fixture/GPU 条件无法执行，必须记录阻塞原因。

---

## 12. 执行结果记录（2026-05-03）

执行计划文件：

`docs/changes/20260503_codebook_collapse_fix_change_plan.md`

### 12.1 执行情况看板

| 项目 | 完成标记 | 结果 |
|------|----------|------|
| C1 Phase A 预训练 | 【✅】 | 已实施，Phase A 跳过 VQ，结束后再执行 codebook warmup。 |
| C7 Soft Assignment | 【✅】 | 已实施，作为 usage-profit alignment 的 soft code assignment 输入。 |
| C2 Usage-Profit Alignment | 【✅】 | 已实施，使用原始 trajectory return，不使用 normalized reward。 |
| 单元测试 | 【✅】 | `34 passed in 2.68s`。 |
| 集成/回归测试 | 【✅】 | `6 passed in 3.69s`。 |
| C3-C6 高风险项 | 【✅】 | 按本设计 §11 的决策未实施。 |

### 12.2 代码执行摘要

- 新增 `TrainingConfig.pretrain_epochs/pretrain_lr`，并在 strict 模式下强制 `pretrain_epochs=0`。
- 新增 `CodebookHealthConfig.usage_profit_alignment_weight/target_corr/temperature`，并在 strict 模式下强制 alignment 权重为 0。
- 新增 `VQArchetypeModel.forward_pretrain` 与 `Phase1Loss.forward_pretrain`。
- 新增 `compute_soft_code_assignments` 与 usage-profit alignment loss。
- `Phase1DemoDataset` / `collate_phase1` 增加 `trajectory_return(s)`，明确使用原始 reward 求和。
- `Phase1Trainer` 接入 Phase A/Phase B 切换，Phase A 不更新 EMA codebook，Phase B 传入 alignment 所需字段。
- CPU 下 trainer/evaluator DataLoader 使用 0 worker，避免本地沙箱 multiprocessing socket 权限问题；CUDA 下保留 2 workers。
- 当 `pretrain_epochs >= epochs` 时，trainer 会自动收敛到 `epochs - 1`，保证至少执行一个 Phase B epoch。

### 12.3 测试命令

```bash
conda activate ArchetypeTrade
pytest tests/unit/models/test_vq_archetype.py \
  tests/unit/models/test_vq_losses.py \
  tests/unit/data/test_dataset.py \
  tests/unit/trainers/test_phase1_trainer.py \
  tests/unit/config/test_phase1_config_docs.py
```

结果：`34 passed in 2.68s`。

```bash
conda activate ArchetypeTrade
pytest tests/integration/test_phase1_collapse_handling.py \
  tests/integration/test_phase1_pipeline_smoke.py
```

结果：`6 passed in 3.69s`。
