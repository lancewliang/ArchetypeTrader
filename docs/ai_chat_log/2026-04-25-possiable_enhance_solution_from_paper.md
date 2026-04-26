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

## 2026-04-25 清理后补充

在把最近这轮没有带来收益的机制试验代码清掉之后，当前 `Phase III` 已经回到一个更干净的状态：

- 保留原始 `argmax` 推理路径
- 保留原始 hindsight label 与 CE 训练方式
- 保留更清晰的日志诊断
  - 训练侧：`sampled a_ref dist / optimal label dist / effective_adjust_horizons`
  - 评估侧：`proposal/effective` 计数、blocked diagnostics、prob diagnostics

这一步的意义是：后面如果再改 `Phase III`，我们面对的是“干净基线 + 可观测日志”，而不是一串已经证伪的阈值、加权、筛选分支。

### 现在不建议再做的事

- 不再继续调 `threshold`
- 不再继续调 `top-k hindsight label`
- 不再继续调 `min_improvement`
- 不再继续调 CE nonzero class weight

这些方向都已经试过，而且没有带来稳定收益；继续在这几条线上拧参数，信息增量很低。

### 现在最值得做的两件事

#### 1. 先做一个最干净的 context ablation：移除 `R_arche`

当前 `context` 是：

- `e_a_sel`
- `a_base`
- `R_arche`
- `tau_remain`

其中最可疑的不是 `e_a_sel` 或 `tau_remain`，而是 `R_arche`：

- 推理时它来自真实在线执行累计
- 训练时它来自 base rollout 的近似累计

这会引入轻度但真实存在的 train/inference mismatch。

所以第一优先级不是“往 context 里继续加东西”，而是先做一组更简单的对照：

- `full context = [e_a_sel, a_base, R_arche, tau_remain]`
- `no R_arche = [e_a_sel, a_base, tau_remain]`

如果去掉 `R_arche` 反而更稳，说明当前 Phase III 的问题不是信息不够，而是这个执行状态特征定义得不够干净。

#### 2. 如果去掉 `R_arche` 后反而更好，再考虑补 `current_position`

当前 Phase III 的 context 里并没有显式持仓状态，但现在评估路径已经支持跨 horizon 延续持仓，所以 `position` 反而比以前更值得显式建模。

这里我更倾向于先加最简单、最稳的版本：

- `current_position_norm = position / m`

而不是一上来就加：

- bucket 化的 recent return
- 更复杂的 pnl state
- 多个 execution summary 特征

原因很简单：`position` 是真正在线可得、定义稳定、和 Phase III 的“局部修补”职责高度相关的状态量。

### 更新后的 Phase III 建议顺序

1. 先在当前清理后的基线上重新确认一次 `Phase II only` 和 `Phase III baseline`
2. 做 `remove R_arche` ablation
3. 如果 `remove R_arche` 有改善，再试 `+ current_position_norm`
4. 只有在这两步都证明 Phase III 还有可挖空间时，才继续讨论更细的 context engineering

换句话说，下一轮不是“加更多机制”，而是：

`先减法，再只加一个最有信息密度的状态量。`

## 2026-04-26 Phase III 实验总归档

这一节记录 2026-04-25 到 2026-04-26 针对 `FU` 的 `Phase III` 全部主要实验、相应代码改动方向，以及最终结论。到当前为止，所有机制性改动都没有带来稳定收益改良；最好的可用结果仍然是“`Phase III` 在 eval 中基本不出手”的基线表现。

### 一、统一基线

统一对照基线来自：

- `logs/FU/batch_04-short/FU_pipeline_20260425_201447.log`

基线结果：

- val: `profit=142836.94`, `TR=4.346277`
- test: `profit=150571.46`, `TR=5.017258`
- `Phase III diagnostics`: `proposal_steps=0`, `effective_steps=0`

这意味着：当前最稳定、收益最高的版本，本质上仍然等价于 `Phase II` 决策，`Phase III` 没有在 eval 中产生实际动作。

### 二、实验时间线与结果

#### 1. threshold 推理实验

相关日志与结果：

- `batch_04-short` 上做 `threshold=0.04`
- 边界修复后结果存于：
  - `result/batch_04-short/evaluation/all_results_phase3_eval_t004_boundaryfix_val.json`
  - `result/batch_04-short/evaluation/all_results_phase3_eval_t004_boundaryfix_test.json`

结果：

- val: `profit=130935.05`, `TR=3.956813`
- test: `profit=141030.22`, `TR=4.689009`

结论：

- 放宽阈值后，`Phase III` 确实开始出手，但利润明显下降。
- 同时还暴露出跨 horizon 边界结算口径问题，后续虽然修正了边界逻辑，但收益退化仍然存在。

#### 2. context ablation 与 eligible-step masking

对应 batch：

- `logs/FU/batch_05-p3-baseline/FU_pipeline_20260425_214147.log`
- `logs/FU/batch_05-p3-no-r/FU_pipeline_20260425_214220.log`
- `logs/FU/batch_05-p3-no-r-pos/FU_pipeline_20260425_214243.log`
- `logs/FU/batch_05-p3-mask/FU_pipeline_20260425_214337.log`
- `logs/FU/batch_05-p3-no-r-pos-mask/FU_pipeline_20260425_214606.log`

改动点：

- baseline
- `remove R_arche`
- `remove R_arche + current_position_norm`
- `eligible-step masking`
- 三者组合

结果：

- 五组 val/test 利润都和基线完全一致
- 五组 eval 都是 `proposal_steps=0`, `effective_steps=0`

结论：

- `remove R_arche`、`position_norm`、`eligible-step masking` 只能改变概率分布，不能让 `Phase III` 在 `argmax` 下真正出手。
- 其中 `no-r-pos-mask` 的概率最“活”，但仍未转化为收益。

#### 3. policy head 改造：categorical vs factorized

对应 batch：

- `logs/FU/batch_06-p3-factorized/FU_pipeline_20260425_221224.log`
- `logs/FU/batch_06-p3-categorical/FU_pipeline_20260425_221412.log`

改动点：

- 保留旧 `3-way categorical`
- 新增 `factorized gate + direction`

结果：

- 两组 val/test 利润仍与基线完全一致
- `factorized`: `proposal_steps=0`
- `categorical`: 只出现极少量 proposal，但 `effective_steps=0`

进一步诊断发现：

- `categorical` 的少量 proposal 全部是冗余同向动作，例如 `base=short ref->short`
- 说明早期标签语义存在问题，模型学会了“非零但不改变最终动作”的假调整

结论：

- 头结构本身不是主要瓶颈。
- `factorized` 只是把问题暴露得更清楚，`categorical` 则在旧标签语义下学出了冗余 nonzero。

#### 4. cleanlabels：去除冗余 same-direction 标签

对应 batch：

- `logs/FU/batch_07-p3-factorized-cleanlabels/FU_pipeline_20260425_225156.log`
- `logs/FU/batch_07-p3-categorical-cleanlabels/FU_pipeline_20260425_225203.log`

改动点：

- hindsight label 只保留真正会改变最终动作的 refinement
- 去掉 `short->short`、`long->long` 这类冗余标签

结果：

- 默认 eval 下仍是基线利润，不出手
- 但 `redundant_same_direction=0`
- 概率分布更健康，说明标签修正方向是对的

结论：

- cleanlabels 修正了训练语义，但不足以带来收益改良。
- 问题从“学错动作”转成了“gate 太保守，eval 不触发”。

#### 5. factorized gate threshold 小范围 eval-only 试探

对应结果：

- `result/batch_07-p3-factorized-cleanlabels/evaluation/all_results_phase3_gate_t012_val.json`
- `result/batch_07-p3-factorized-cleanlabels/evaluation/all_results_phase3_gate_t012_test.json`
- `result/batch_07-p3-factorized-cleanlabels/evaluation/all_results_phase3_gate_t010_val.json`
- `result/batch_07-p3-factorized-cleanlabels/evaluation/all_results_phase3_gate_t010_test.json`

结果：

- `gate=0.12`
  - val: `profit=137134.34`, `TR=4.176042`
  - test: `profit=148169.24`, `TR=4.930766`
  - 已经会出手，但仍弱于基线
- `gate=0.10`
  - val: `profit=117731.97`, `TR=3.602190`
  - test: `profit=143399.72`, `TR=4.773916`
  - 明显更差

结论：

- 放宽 factorized gate 能让 `Phase III` 动起来，但净效果仍是负收益。
- `0.10` 比 `0.12` 更差，说明这条线很快进入过度交易。

#### 6. invalid refinement masking

对应 batch：

- `logs/FU/batch_08-p3-factorized-masked/FU_pipeline_20260425_234739.log`

改动点：

- 训练与推理统一折叠同向无效 refinement 概率
- `factorized` gate 改为看“折叠后的有效非零概率”

默认结果：

- val/test 仍为基线利润
- `proposal_steps=0`, `effective_steps=0`

进一步做 `gate=0.12` eval-only：

- `result/batch_08-p3-factorized-masked/evaluation/all_results_phase3_gate_t012_masked_val.json`
- `result/batch_08-p3-factorized-masked/evaluation/all_results_phase3_gate_t012_masked_test.json`

结果：

- val: `profit=122225.76`, `TR=3.731802`
- test: `profit=130811.36`, `TR=4.355283`
- `proposal/effective` 大量增加，而且几乎都来自 `base=flat`

结论：

- invalid masking 本身是语义清理，不是收益增强器。
- 一旦放开 gate，`Phase III` 会在 `flat` 上大量开仓，交易成本爆炸，利润显著退化。

#### 7. delta 语义

对应 batch：

- `logs/FU/batch_09-p3-factorized-delta/FU_pipeline_20260426_074935.log`

改动点：

- `a_ref` 从“绝对覆盖 short/long”改成相对 `delta`
- 最终动作按 `clip(a_base + a_ref, 0, 2)` 生成

结果：

- val: `profit=142836.94`, `TR=4.346277`
- test: `profit=150571.46`, `TR=5.017258`
- `proposal_steps=0`, `effective_steps=0`
- 但相较 `batch_08`，`adjust mean` 明显变小，更保守

结论：

- delta 语义修正了动作定义，使 `Phase III` 更像“局部加减档”而不是“flat 上额外开仓器”
- 但当前训练目标下，它又变得过于保守，依然没有收益改良

#### 8. factorized gate 正类加权

对应 batch：

- `logs/FU/batch_10-p3-factorized-delta-gatew/FU_pipeline_20260426_084630.log`

改动点：

- 在 factorized gate 的 BCE 中增加正类权重
- 实验使用较高权重，试图解决“gate 过于保守”

结果：

- val: `profit=122573.38`, `TR=3.740310`
- test: `profit=125891.85`, `TR=4.193052`
- `proposal/effective` 大量增加
- 绝大部分有效动作仍来自 `base=flat`

训练速度变化：

- `batch_09` 从日志时间看大约 `14m40s`
- `batch_10` 从日志时间看大约 `48m29s`

结论：

- 正类加权确实让 gate 不再保守，但幅度明显过头。
- 同时它让训练动态变差：大量 horizon 很快触发 adjustment，单位 horizon 累计 step 变少，为了凑满总 step 需要跑更多 horizon，整体训练反而更慢。
- 收益和速度两边都变差。

### 三、实验中做过的主要代码改动

下面这些都是本轮 Phase III 线上做过、但最终没有带来改良收益的改动方向：

1. `threshold` 推理与非零阈值 gate
2. `R_arche` 移除 / `position_norm` 增加 / `eligible-step masking`
3. `categorical` 与 `factorized` 双头对照
4. 更细的 proposal/effective/block/prob 日志诊断
5. 冗余 same-direction hindsight label 清理
6. `factorized gate threshold` 可配置
7. invalid refinement masking
8. `a_ref` 的 `delta` 语义
9. factorized gate 正类加权
10. 为加速 Phase III 训练而做的 LOB 预提取、批量 archetype/base-action 推理、hindsight 数组路径优化

这些改动中，只有“日志诊断增强”明确带来了正价值：它帮助确认问题根因，并避免继续在无效方向上浪费时间。其余机制性改动都没有带来正向收益。

### 四、最终结论

到 2026-04-26 为止，`Phase III` 的所有主要改造都没有实现稳定收益改良。

最核心的结论有三条：

1. 当前最优收益仍然来自“`Phase III` 在 eval 中基本不出手”的版本。
2. 一旦把 `Phase III` 触发得更积极，最常见的结果是：
   - 在 `flat` 上额外开仓
   - 交易成本快速上升
   - `TR` 和最终利润下降
3. 这些实验说明：在当前数据、目标和执行约束下，`Phase III` 还没有被证明具备稳定可部署价值。

因此，本轮结论不是“继续叠更多机制”，而是：

- 保留实验日志与诊断结论
- 回滚所有针对 `Phase III` 的代码改动
- 让代码库回到更早、更干净的稳定版本

### 五、当前归档决议

本节写入 `ai_chat_log` 后，按当前决议执行：

- 保留 `logs/` 下所有实验日志
- 保留 `docs/ai_chat_log/` 下本归档文档
- 回滚所有本轮 `Phase III` 机制改动、推理改动、训练改动、速度优化改动与相关测试改动

执行原因：

- 到当前为止，没有任何一条 `Phase III` 机制改动带来了收益改良
- 继续在现有分支上累积实验代码，只会增加维护成本并污染后续判断
