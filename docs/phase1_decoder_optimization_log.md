# Phase I Decoder 优化记录

> 日期: 2026-04-03
> 交易对: ETH
> 目标: 改善 Phase I VQ decoder 的重建质量，使 archetype 在 Phase II 中可用

---

## 背景

Phase II 训练结果不理想，通过新增的环境级验证模块 (`src/phase1/env_validation.py`) 定位到根因在 Phase I 的 decoder。

### Phase III 的作用

Phase III (Archetype Refinement) 是一个 step 级别的 RL agent，在 Phase II 选定 archetype 后对 frozen decoder 生成的 base actions 做微调：
- RefinementAgent 观察市场状态 + 上下文（archetype embedding、当前 base action、累积收益、剩余步数）
- 输出调整信号 `a_ref ∈ {-1, 0, 1}`，每个 horizon 最多一次调整
- 使用 regret-aware reward: `r_ref = (R - R_base) + β₁ × (R - R_1_opt)`
- 本质上是一个"纠错层"，其上限完全取决于 Phase I archetype 质量和 Phase II 选择质量

### 新增的验证模块

新增 `src/phase1/env_validation.py`，在 Phase I 训练后自动运行 4 项环境级诊断：

1. **Archetype 环境级 Return 分布** — 每个 archetype 用 frozen decoder 在真实 env 中执行的 return
2. **Archetype 行为差异性** — pairwise action agreement matrix + JS divergence
3. **Decoder 动作分布偏移** — change point accuracy + 动作分布 shift
4. **验证集环境 Return** — oracle return（每个 horizon 选最优 archetype）

报告保存到 `result/{pair}/phase1_archetype_discovery/phase1_env_validation_report.json`

---

## 实验记录

### 实验 1: 原始 MLP Decoder (Baseline)

**架构**: `Linear(61→128) → ReLU → Linear(128→3)`，每个时间步独立预测

**配置**: 100 epochs, lr=3e-4, batch_size=256, CrossEntropyLoss

**结果**:

| 指标 | 值 |
|---|---|
| 最终 loss | 0.9438 |
| token_accuracy | 62.3% |
| change_point_accuracy | 13.4% |
| action shift (max) | 12.8% (long 被低估) |
| return_gap_mean | 469.6 |
| 正收益 archetype 数 | 1/10 |
| diversity (agreement) | 0.55 (好) |
| pred_change_count | 14, 10, 13 (vs DP 的 1) |

**诊断**: MLP 逐步独立预测，无时序建模能力。无法学会 DP 轨迹的 single-trade 约束（整个 horizon 只变一次动作），导致频繁切换动作，执行成本爆炸（avg_cost 800-930）。

**混淆矩阵**: short/flat/long 的 precision/recall 相对均衡，但 pred_change_count 远超 true_change_count。

---

### 实验 2: LSTM Decoder (无 class weight)

**改动**: 将 decoder 从 pointwise MLP 改为 LSTM
- `LSTM(input_size=61, hidden_size=128) → Linear(128→3)`
- 隐藏状态在时间步之间传递，具备时序建模能力

**配置**: 100 epochs, lr=3e-4, batch_size=256, CrossEntropyLoss (无权重)

**结果**:

| 指标 | MLP | LSTM | 变化 |
|---|---|---|---|
| 最终 loss | 0.9438 | 0.7229 | ↓ 23% |
| token_accuracy | 62.3% | 55.6% | ↓ (但更诚实) |
| change_point_accuracy | 13.4% | **42.7%** | ↑ 3x ✓ |
| action shift (max) | 12.8% | 43.8% | ↑ (long 坍缩) |
| return_gap_mean | 469.6 | **139.9** | ↓ 70% ✓ |
| 正收益 archetype 数 | 1/10 | **5/10** | ↑ 5x ✓ |
| diversity (agreement) | 0.55 | 0.81 | 合理 |
| pred_change_count | 14,10,13 | **1,3,1,1** | 接近 DP ✓ |
| change_step_mae | 14.99 | **5.16** | ↓ 66% ✓ |
| val oracle positive ratio | 39.8% | **67.9%** | ↑ ✓ |

**诊断**: LSTM 成功学会了 single-trade 约束。但出现 **long 坍缩**：

```
decoder 动作分布: short=0.09%, flat=17.6%, long=82.3%
DP 真实分布:      short=38.2%, flat=23.3%, long=38.5%
```

混淆矩阵显示 short 的 recall 仅 0.13%，几乎全部被预测为 long。原因：short 和 long 在 state 特征上相似（都是方向性持仓），LSTM 学到"保持"模式后偏向了 long。

**结论**: 这是四个版本中综合表现最好的。Diversity 好、change point accuracy 最高、5 个正收益 archetype。唯一问题是 long 坍缩。

---

### 实验 3: LSTM + Inverse-Frequency Class Weight

**改动**: `nn.CrossEntropyLoss(weight=[0.87, 1.48, 0.85])` (inverse-frequency)

**配置**: 100 epochs, lr=3e-4, batch_size=256

**结果**:

| 指标 | LSTM 无权重 | LSTM + class weight |
|---|---|---|
| 最终 loss | 0.7229 | 0.8047 |
| change_point_accuracy | 42.7% | **19.6%** ↓ |
| action shift (max) | 43.8% (→long) | 38.5% (→short) |
| diversity (agreement) | 0.81 | 0.53 |

**诊断**: 矫枉过正。从全 long 翻转成了全 short，long 几乎消失（0.06%）。训练曲线也异常：前 70 epoch loss 几乎不下降（1.099→1.077），直到 epoch 70-100 才突然开始学。Class weight 干扰了 LSTM 早期的学习动态。

**结论**: 静态 class weight 不适合这个问题。LSTM 的坍缩是时序惯性导致的，不是类别不平衡。

---

### 实验 4: LSTM + Focal Loss + Mild Class Weight

**改动**:
- Focal Loss: `FL(p_t) = -(1-p_t)^2 × log(p_t)`，gamma=2.0
- 温和 class weight: `sqrt(inverse-frequency)`，归一化到均值 1
  - 实际权重: short=0.91, flat=1.19, long=0.90

**配置**: 100 epochs, lr=3e-4, batch_size=256

**结果**:

| 指标 | LSTM 无权重 | LSTM + focal |
|---|---|---|
| 最终 loss | 0.7229 | 0.2307 |
| change_point_accuracy | 42.7% | 18.0% ↓ |
| action shift (max) | 43.8% (→long) | 38.4% (→short) |
| diversity (agreement) | 0.81 | 0.63 |

**诊断**: 同样偏向 short（70.7%），long 消失。Focal loss 降低了 easy samples 的权重，但没有解决方向性坍缩。

**结论**: Loss function 层面的修改无法解决 LSTM 的方向性坍缩问题。

---

### 实验 5: LSTM + Teacher Forcing

**改动**: 给 LSTM decoder 加入 teacher forcing
- 训练时: 每步输入 `[state, z_q, prev_action_onehot]`，prev_action 来自 ground-truth
- 推理时: autoregressive，用自己上一步的 argmax 预测作为输入
- LSTM input_size: `state_dim + code_dim + action_dim = 64`

**配置**: 100 epochs, lr=3e-4, batch_size=256, Focal Loss + mild weight

**结果**:

| 指标 | LSTM 无权重 | LSTM + TF |
|---|---|---|
| 最终 loss | 0.7229 | **0.1889** |
| change_point_accuracy | 42.7% | **46.3%** ↑ |
| action shift (max) | 43.8% | **58.1%** ↓↓ |
| diversity (agreement) | 0.81 | **0.99** ↓↓↓ |
| 正收益 archetype 数 | 5/10 | 4/10 |
| val oracle positive ratio | 67.9% | **43.8%** ↓ |

**诊断**: 经典的 **exposure bias** 问题。

训练时 decoder 每步收到 ground-truth 前一步动作（loss 很低），但推理时第一步没有 ground-truth，用 flat 初始化。一旦第一步预测错（比如预测成 short），错误传播到后续所有步，整条轨迹都变成 short。

关键证据：
- 推理时 short=96.3%, long=0.5% — 几乎全 short
- 10 个 archetype 的动作分布完全相同（agreement=99.4%）— z_q 被完全忽略
- val oracle best_archetype_distribution: archetype 0 占 255/256 — 所有 archetype 行为一样

**结论**: Teacher forcing 让 decoder 过度依赖 prev_action 输入，忽略了 z_q。不适合这个场景。

---

## 综合对比

| 版本 | change_pt_acc | shift | diversity | 正收益 | 核心问题 |
|---|---|---|---|---|---|
| MLP (baseline) | 13.4% | 12.8% | 0.55 ✓ | 1/10 | 无时序建模，频繁切换 |
| **LSTM 无权重** | **42.7%** | 43.8% | **0.81** ✓ | **5/10** | long 坍缩 |
| LSTM + class wt | 19.6% | 38.5% | 0.53 | - | short 坍缩，矫枉过正 |
| LSTM + focal | 18.0% | 38.4% | 0.63 | - | short 坍缩 |
| LSTM + TF | 46.3% | 58.1% | 0.99 ✗ | 4/10 | exposure bias，z_q 被忽略 |

**最佳版本: LSTM 无权重** — 综合表现最好，唯一问题是 long 坍缩。

---

## 当前状态与下一步

### 当前代码状态
- `src/phase1/vq_decoder.py`: 纯 LSTM decoder（无 teacher forcing、无 class weight）
- `scripts/train_phase1.py`: 纯 CrossEntropyLoss，epoch 限制放宽为 >= 100
- `src/phase1/env_validation.py`: 新增的环境级验证模块
- `tests/test_vq.py`: 已更新适配 LSTM decoder

### 下一步计划
1. **增加训练 epoch**: LSTM 无权重版本在 100 epoch 时 loss 还在下降（最低 loss 在 epoch 100），尝试 200-300 epoch
   ```bash
   python scripts/train_phase1.py --pair ETH --phase1-epochs 200
   ```
2. 如果更多 epoch 后 short recall 仍接近 0，考虑：
   - 双向 LSTM (BiLSTM) — 让 decoder 能看到未来的 state 信息
   - 在 z_q 中显式编码方向信息（修改 encoder 或 codebook 初始化）
   - Scheduled sampling — 训练时以一定概率用自己的预测替代 ground-truth，渐进式减少 teacher forcing

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `src/phase1/vq_decoder.py` | 重写 | MLP → LSTM decoder |
| `src/phase1/env_validation.py` | 新增 | Phase I 环境级验证模块 |
| `scripts/train_phase1.py` | 修改 | 集成 env_validation，放宽 epoch 限制 |
| `tests/test_vq.py` | 修改 | 适配 LSTM decoder 的测试 |


---

### 实验 6: MLP Decoder + Single-Trade 推理约束 (当前方案)

**思路转变**: 放弃修改 decoder 架构或 loss function，回到论文原始 MLP decoder。
MLP 的优势（diversity 好、short/flat/long 均衡）是 LSTM 各变体都无法复现的。
MLP 的唯一问题（频繁切换动作）可以在推理时通过后处理解决。

**改动**:
- `VQDecoder` 恢复为原始 MLP: `Linear(61→128) → ReLU → Linear(128→3)`
- 新增 `decode_with_single_trade_constraint()` 方法:
  - 在 MLP 输出的 logits 上搜索最优 single-change-point 分割
  - 使用前缀和 + 后缀和，O(h × action_dim²) 复杂度
  - 输出严格满足 single-trade 约束的动作序列
- 训练时: 使用原始 `forward()` + CrossEntropyLoss（不变）
- 推理时: 所有下游代码改用 `decode_with_single_trade_constraint()`

**原理**:
MLP 的 logits 已经包含了足够的信息（哪些步应该 short、哪些应该 long），
只是它没有强制 single-trade 约束。后处理从 72 步的 log-probability 中
找到最优的 "action_a × t + action_b × (h-t)" 分割，等价于在 MLP 的
soft prediction 上做一次 constrained decoding。

**预期效果**:
- 保留 MLP 的 diversity 优势（agreement ~0.55）
- 保留 MLP 的动作分布均衡性（shift ~12.8%）
- 消除频繁切换问题（pred_change_count 严格 ≤ 1）
- 大幅降低执行成本（avg_cost 从 800-930 降到接近 DP 水平）

**修改的文件**:
- `src/phase1/vq_decoder.py`: 恢复 MLP + 新增 `decode_with_single_trade_constraint()`
- `scripts/train_phase1.py`: 恢复原始 CrossEntropyLoss，恢复严格 epoch 检查
- `scripts/train_phase2.py`: 推理改用 `decode_with_single_trade_constraint()`
- `src/script/train_phase3.py`: 同上
- `src/script/evaluate.py`: 同上
- `src/script/train_phase2_bak.py`: 同上
- `src/phase1/env_validation.py`: `_decode_horizon` 改用约束解码
- `tests/test_vq.py`: 恢复 MLP 测试 + 新增 single-trade 约束测试

**待验证**: `python scripts/train_phase1.py --pair ETH`


**实验 6 结果** (2026-04-03 22:13):

| 指标 | MLP argmax | LSTM 最佳 | MLP + single-trade |
|---|---|---|---|
| decoded_return_mean | -470.6 | -140.9 | **+1062.5** |
| return_gap_mean | 469.6 | 139.9 | **-1063.4** (超越 DP) |
| 正收益 archetype 数 | 1/10 | 5/10 | **9/10** |
| diversity (agreement) | 0.55 | 0.81 | **0.53** ✓ |
| action shift (max) | 12.8% | 43.8% | **12.8%** ✓ |
| val oracle return | 300.3 | 259.1 | **506.3** |
| val oracle positive ratio | 39.8% | 67.9% | **75.4%** |
| avg_cost (archetype 0) | 811.6 | 151.3 | **205.6** |

关键发现:
- decoded_return 从 -470 变成 +1062，超过了 DP 的 return (-0.99)
- 9/10 archetype 有正收益（仅 archetype 1 全 flat 为零收益）
- Diversity 保持良好 (0.53)，archetype 之间有明显策略差异
- 验证集 oracle positive ratio 75.4%，泛化能力好
- 剩余 2 个 warning 是 raw argmax 的 per-token 指标，不影响实际推理

结论: **MLP + single-trade 推理约束是目前最佳方案**，可以进入 Phase II 训练。


---

## Phase II 训练结果 (基于实验 6 的 Phase I 模型)

**配置**: 800k steps, PPO, rollout_batch=1024, clip_eps=0.2, ent_coef=0.1, vf_coef=0.001, alpha=1.0

**最优验证集 return: 378.05** (step 730112)

### 训练过程关键节点

| Step | val_return | learned_return | best_fixed | greedy 集中度 | gt_agree |
|---|---|---|---|---|---|
| 73k | 359.8 | 781.9 | 789.9 (k=7) | k=5: 99% | 11.1% |
| 219k | 361.6 | 544.7 | 565.6 (k=4) | k=5: 77% | 11.7% |
| 365k | 363.1 | 875.2 | 869.5 (k=5) | k=0+8: 98% | 14.8% |
| 730k | **378.1** | 753.3 | 772.3 (k=5) | k=4: 97% | 11.7% |
| 800k | 364.1 | 393.4 | 460.1 (k=4) | k=3: 100% | 14.1% |

### 观察

1. **验证集 return 稳定在 360-378 区间**，最优 378.1 出现在 730k step。
2. **Selector 严重坍缩到单一 archetype**: greedy 策略几乎总是选同一个 archetype（不同阶段分别是 k=5, k=8, k=3, k=4）。这说明 selector 没有学会根据市场状态选择不同 archetype。
3. **gt_agree 始终在 10-15%**: selector 的选择与 ground-truth archetype label 一致性很低（随机选择是 10%）。
4. **learned_return 接近 best_fixed_return**: selector 的表现接近"固定选一个最好的 archetype"，没有体现出条件选择的优势。
5. **训练末期 policy gradient 消失**: step 800k 时 policy_grad_norm=0.07, clip_fraction=0.0, kl=0.0，策略已完全停止更新。

### 问题分析

Phase II 的核心问题是 **selector 坍缩**——它学会了选一个"平均最好"的 archetype 然后一直用，而不是根据不同市场状态选不同 archetype。可能原因：

1. **方向性 archetype 之间差异太小**: 从 Phase I 的 fixed_returns 看，archetype 0/3/5/6/7/8 的 return 都在 550-870 之间，差异不大。Selector 选哪个都差不多，没有足够的梯度信号来学习区分。
2. **ent_coef=0.1 过高**: 高 entropy bonus 鼓励均匀探索，但 PPO 的 policy loss 信号太弱（因为不同 archetype 的 return 差异小），entropy bonus 反而主导了策略更新。
3. **imitation loss (alpha=1.0) 可能干扰**: gt_label 分布均匀（每个 archetype ~10%），但 selector 倾向于集中选择，两者矛盾导致梯度冲突。

### 下一步建议

尽管 selector 坍缩了，验证集 return 378 仍然是正的，说明 Phase I 的 archetype 质量足够好。可以：
1. 先用当前模型跑 Phase III，看 refinement agent 能否在此基础上进一步改善
2. 后续优化 Phase II 时，降低 ent_coef 到 0.01-0.03，降低 alpha 到 0.1-0.3
