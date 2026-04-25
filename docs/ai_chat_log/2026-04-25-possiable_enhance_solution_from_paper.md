# 2026-04-25 possible enhance solution from paper

我重新对照了论文 `AAAI26_ArchetypeTrader_中英对照版.md` 后半部分、当前代码实现，以及这几天的实验日志。结论先写在前面：论文里后半部分提到的几项“架构增强”，在当前代码里其实已经实现了大半，尤其是 `Phase III refinement` 这条线；现在更值得做的，不是机械照着论文再补模块，而是先确认这些增强是否真正带来了收益，以及它们是否存在训练和推理不一致的问题。

## 先说结论

从论文后半部分抽出来、并且和当前项目最相关的增强点，大致是这些：

1. 在 horizon-level archetype selection 之后，增加 step-level refinement adapter。
2. 用受限干预机制，保证每个 horizon 最多只做一次 refinement。
3. refinement agent 同时接收市场状态和 archetype context。
4. 用 AdaLN 等条件化机制融合 market state 与 archetype context。
5. 用 hindsight top-k DP adaptation 作为训练引导。
6. 用 regret-aware reward 训练 refinement agent。
7. 在 RL 目标外增加 CE supervision，让 agent 学习 hindsight optimal action。

当前代码中，`2/4/5/6/7` 都已经做了，`3` 也已经以 `[e_a_sel, a_base, R_arche, tau_remain]` 这组 context 的形式接进去了。所以这份文档不再把它们当成“待实现清单”，而是把它们分成三类：

- 已实现且逻辑完整
- 已实现，但有训练/推理一致性风险
- 已实现，但仍值得做 ablation 或简化对照

## 已实现的论文增强

### 1. 单次局部 refinement 约束已经落地

论文建议每个 horizon 内最多只允许一次非零适配动作，避免 adapter 完全破坏 archetype 原意。当前代码已经实现这一点：

- `PolicyAdapter.compute_final_action(...)` 明确限制每个 horizon 最多一次调整。
- base action 在 change point 处不会被 refinement 覆盖。

对应代码：

- `src/phase3/policy_adapter.py`

这部分我暂时不建议改。你现在一行是 `20s`，`72` 步大约 `24` 分钟，已经比论文设定短很多；在这个时间粒度下，保留“单次局部修正”是合理的。

### 2. AdaLN 条件融合已经实现

论文里建议 refinement agent 不只看市场状态，还要把 archetype context 条件化进去。当前实现不是简单 concat 后 MLP，而是已经用了 AdaLN：

- `RefinementAgent` 用 `AdaptiveLayerNorm` 对 market feature 做条件化。
- `tests/test_refinement_agent.py` 里还有专门验证 “不同 context 会改变输出” 的测试。

对应代码：

- `src/phase3/refinement_agent.py`
- `src/phase3/adaln.py`
- `tests/test_refinement_agent.py`

所以“试试 5”这个动作，严格说已经做过了。后续如果要试，不是“加不加 AdaLN”，而是做 ablation：

- AdaLN vs 纯 concat
- AdaLN vs 更轻量 gating
- 有 context vs 去掉部分 context

### 3. hindsight top-5 DP guidance 已经实现

论文里提出用 DP 计算 hindsight-optimal top-5 adaptations，作为 refinement 训练时的辅助监督。当前代码已有完整实现：

- `compute_top5_hindsight_optimal(...)`
- 训练时对每个 sampled horizon 真实计算 top-5

对应代码：

- `src/phase3/regret_reward.py`
- `scripts/train_phase3.py`

这说明“试试 6”也不是新功能，而是已经在主训练路径里了。

### 4. regret-aware reward 已经实现

论文里的 reward：

- `r_ref = (R - R_base) + beta1 * (R - R_1_opt)`

当前代码已实现并接入训练：

- `compute_regret_reward(...)`
- `compute_step_rewards(...)`

对应代码：

- `src/phase3/regret_reward.py`
- `scripts/train_phase3.py`

所以“试试 7”同样已经做了。

### 5. hindsight CE supervision 已经实现

论文里还加了一个监督项，让 policy 对齐 hindsight optimal action。当前代码也已经实现：

- 先根据 top-5 adaptation 构造 `optimal_actions`
- 再在 PPO update 里加 `beta2 * ce_loss`

对应代码：

- `scripts/train_phase3.py`

所以“试试 8”也已经不是空白项。

## 已实现，但需要重点警惕的地方

### 1. refinement context 已经用了执行状态，但存在一致性风险

这是当前最值得认真看的一点。

论文建议的 context 是：

- `e_a_sel`
- `a_base`
- `R_arche`
- `tau_remain`

当前代码里已经照这个思路做了。训练和评估时，refinement agent 的输入都包含：

- selected archetype embedding
- 当前 base action
- 累积 reward
- 剩余步数

对应代码：

- 训练：`scripts/train_phase3.py`
- 推理：`src/evaluation/inference_runner.py`

但这里有一个你已经敏锐指出的问题：当前训练里用的 `R_arche`，并不是 refinement rollout 的在线真实累计，而是先跑一遍 `base rollout` 后构造出来的累计收益。代码自己也写明了，这是一个“合理近似”。

这意味着：

- 推理时，`R_arche` 来自真实在线执行过程。
- 训练时，`R_arche` 来自 base path 的预先累计结果。

两者并不完全一致。

这不是未来信息泄漏，但确实是一种 train/inference mismatch。它未必致命，但很可能会削弱 refinement 的稳定收益。

我的判断是，这个点比“要不要再加新模块”更值得优先处理。因为它直接影响你之前担心的那个问题：模型可能学到的是一种依赖离线构造执行状态的策略，而不是严格在线可泛化的修正逻辑。

### 2. Phase III 的训练采样是随机 horizon，不是连续时间展开

当前 `train_phase3.py` 每轮训练会随机采样 horizon：

- 随机选一个 `h_idx`
- 在该窗口内选 archetype
- 生成 base actions
- 计算 hindsight labels 和 reward

这本身不一定错，但如果想让 `R_arche` 真正表达“执行状态上下文”，那随机窗口采样会让这个上下文更多表现为“局部窗口内的执行摘要”，而不是跨时间连续策略状态。

所以如果后续还要继续强化 `execution-state context` 这条线，更合理的方向不是再塞更多上下文字段，而是先考虑：

- 训练是否要更多基于连续时间展开
- 至少是否要做一组“随机 horizon vs 连续 horizon”对照

## 哪些点不建议现在动

### 1. 不建议改 horizon=72

你已经有很明确的经验判断：

- 论文原始设定是一行 `1min`
- 你现在是一行 `20s`
- `72` 在你这里等价于更高频、更短粒度的决策窗口
- 实验上也观察到“间隔越短收益越高，决策频率越高”

所以现在如果再机械地调 `72`，意义不大。你实际上已经通过更细时间分辨率，变相完成了“缩短决策响应周期”。

### 2. 不建议现在取消“每 horizon 最多一次 refinement”

虽然这会限制 Phase III 的上限，但在当前项目里，这条约束反而有助于把 refinement 控制在“局部修补”而不是“重新规划”。如果贸然放开，Phase III 很容易变成另一个 step-level trader，训练难度和解释性都会明显变差。

## 我建议的后续优先级

### P1. 先确认 Phase III 是否真的提供了稳定增益

因为论文增强项大多已经实现了，所以接下来第一步不是再写代码，而是做一次明确对照：

- Phase II only
- Phase III full
- Phase III 去掉 regret
- Phase III 去掉 CE
- Phase III 去掉 context 中的 `R_arche`

如果 full model 并没有稳定优于简化版，那说明当前瓶颈不在“缺模块”，而在“训练协议或上下文定义不够干净”。

### P1. 优先检查 `R_arche` 的 train/inference 一致性

这是我目前最想优先做的小范围结构修正。可以考虑两条路：

1. 保守做法：先做 ablation，直接移除 `R_arche`，只保留
   - `e_a_sel`
   - `a_base`
   - `tau_remain`
2. 更严格做法：训练时也只使用真正在线可得的 refinement-path 累积信息，而不是 base rollout 近似。

第一条便宜，第二条更干净。

### P2. 如果 Phase III 本身有效，再继续做更细的 context 设计

只有在确认 Phase III 真的有贡献之后，才值得继续讨论：

- context 的数值归一化方式
- `R_arche` 是否换成 position / pnl ratio / recent return bucket
- context 是否增加当前持仓状态

否则就会陷入“往一个还没被证明有效的模块里继续堆信息”。

## 最后总结

这次从论文反推代码后的最大发现，不是“还有好多论文增强没做”，而是恰恰相反：`Phase III` 这条论文主线已经在代码里实现得相当完整了。真正的问题变成了：

1. 这些增强是否真的带来了稳定收益；
2. 其中最关键的执行状态上下文，是否存在轻度但足以伤害泛化的一致性问题。

所以我目前的建议是：

- 不再把 `5/6/7/8` 当成待实现功能
- 把它们当成“已实现，待验证收益与一致性”的模块
- 后续优先做 Phase III 的 ablation 和 context 清洗

如果要继续写下一版实验计划，我会把主题改成：

`Phase III refinement 的收益归因与 train/inference consistency 诊断`

而不是再写一份泛泛的“possible enhance solution”。
