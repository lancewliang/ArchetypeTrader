# Phase III 优化讨论记录

> 日期: 2026-04-12
> 交易对: AL
> 目标: 梳理 Phase III 是否有效、是否应负责修复 selector 错误，以及后续优化方向

---

## 背景

在 `AL` 的三阶段流水线中，Phase II 在 `val/test` 上与 DP 基准存在明显差距。最初怀疑 Phase III 没有生效，但复核正确日志后确认：

- 正确日志文件是 `logs/AL/batch_001/AL_pipeline_20260412_012052.log`
- Phase III 实际完成了完整训练和后续评估
- 但 `phase3_eval` 与 `phase2_eval` 的最终指标完全一致，说明 Phase III 没有带来可见收益

本记录总结这轮讨论中的关键证据、结论和后续策略。

---

## 关键事实

### 1. Phase III 确实完整跑过

根据 `logs/AL/batch_001/AL_pipeline_20260412_012052.log`：

- Phase III 于 `2026-04-12 01:50:10` 开始训练
- 共训练约 `1,000,042` steps，处理 `32,336` 个 horizons
- 模型保存为 `result/AL/batch_001/phase3_archetype_refinement/AL_refinement_agent_beta0.5.pt`

说明问题不是“Phase III 没跑”，而是“跑了但没有改善最终行为”。

### 2. Phase III 训练目标有改善，但最终推理无改善

训练日志显示：

- 前期 `avg_reward` 明显为负
- 后期逐步转正，最近若干段有正值
- 最终最近 `1000 horizons` 平均奖励约为 `35.6889`

但评估结果显示：

- `phase2_eval/val` 与 `phase3_eval/val` 的 `TR` 均为 `0.446607`
- `phase2_eval/test` 与 `phase3_eval/test` 的 `TR` 均为 `0.819981`

说明：

- Phase III 在自己的训练目标上似乎学到了一些东西
- 但这些学习没有转化为最终评估收益

### 3. Phase III 最终操作序列与 Phase II 完全一致

进一步比对 `phase2_eval_*` 与 `phase3_eval_*` 的输出：

- `val` 的 operations CSV checksum 完全一致
- `test` 的 operations CSV checksum 完全一致

这意味着：

- Phase III 在最终推理阶段没有改变任何一步执行动作
- 问题不是“改善幅度太小”，而是“最终行为完全没变”

---

## 架构层面的结论

### 1. 按原论文，Phase III 不负责重新选择 archetype

查阅 `AAAI26_ArchetypeTrader_core.md` 后确认：

- Phase II 是 horizon-level archetype selector
- Phase III 会 **冻结** Phase II selector
- Phase III 只对已选 archetype 生成的 `base actions` 做 step-level refinement
- refinement action 仅为 `a_ref in {-1, 0, 1}`
- 每个 horizon 最多允许一次调整

因此，原论文架构中：

- Phase III **不会**重新选 archetype
- Phase III **不会**在 horizon 内切换 archetype
- Phase III 的定位是“局部动作修补器”，不是“高层决策纠错器”

### 2. 当前实现与论文原设定一致

当前代码实现与论文描述一致：

- `SelectionAgent` 在 Phase III 中被冻结
- `PolicyAdapter` 按 Eq. 6 将 refinement 限制为单次局部调整
- 推理阶段依旧先由 Phase II 选定 archetype，再由 Phase III尝试微调

所以：

- “让 Phase III 修复 selector 选错 archetype”是一个合理的扩展方向
- 但它属于 **超出原论文设计的架构扩展**

---

## 对问题本质的判断

当前更可能的根因不是 Phase III 本身，而是 Phase II 的泛化失败：

- selector 在 train 上学到了一套 “状态特征 -> archetype” 映射
- 这套映射在 val/test 上出现分布偏移
- 某些本应选择趋势型 archetype 的 horizons，被错误分到 flat/不盈利 archetype

在这种情况下：

- Phase II 一开始就把高层 archetype 选错了
- Phase III 只能在错误 archetype 生成的 `base actions` 上做一次有限修补
- 这很难从根本上恢复正确的趋势交易路径

结论：

- selector 错误属于 **Phase II 职责范围**
- 这个问题应优先在 Phase II 解决，而不是依赖 Phase III 弥补

---

## 数据划分与评估口径讨论

### 1. 当前严格意义上已经没有“完全未触碰的 test”

讨论中确认：

- `train` 用于参数训练
- `val` 用于 checkpoint 选择、超参数选择、方案比较
- 如果研发过程中持续观察 `test` 结果并据此做决策，那么 `test` 也会逐渐被“开发化”

因此当前更准确的口径是：

- `train`: 拟合参数
- `val`: 模型选择 / 超参选择
- `test`: 实际上已经更接近 `dev-test`
- 真正严格意义上的 `final-test`: 当前并不存在

### 2. 保守版数据协议被确定为后续方向

最终决定采用更保守的数据协议：

- `train`
- `val`
- `dev-test`
- `final-test`

含义：

- `train`: 训练参数
- `val`: 选 checkpoint / 选超参
- `dev-test`: 研发期间方案对比
- `final-test`: 最后一次性评估，不参与任何开发决策

### 3. “用验证集修模型”需要区分两种情况

讨论中明确区分了两类做法：

1. 用 `val` 做 early stop / checkpoint 选择 / 超参选择  
   这是标准做法，合理。

2. 用 `val reward` 继续更新模型权重  
   这会把 `val` 变成训练数据的一部分。此时它不再应被称为 validation set。

因此：

- 若未来要让某一阶段直接利用 `val reward` 做训练，必须同步调整数据协议
- 不能一边用 `val` 更新参数，一边还用同一段 `val` 报告泛化性能

---

## 关于“是否让 Phase III 修复 selector 错误”的结论

讨论结论分两层：

### 架构层面

- 从机制上说，让 Phase III 具备“修 selector 错误”的能力是可行的
- 但这会把 Phase III 从“局部 refinement”升级成“高层纠错或重规划模块”
- 这已经不是原论文的第三阶段定义

### 当前研发优先级

- 虽然这个方向以后可以探索
- 但当前 **暂不考虑** 改动 Phase III 的职责边界
- 现阶段应优先增强 Phase II 的能力，让 selector 本身学得更稳、更能适应分布偏移

---

## 当前决定的研发方向

本轮讨论后的明确决定：

1. 采用保守版数据协议：`train / val / dev-test / final-test`
2. 暂不扩展 Phase III 去修复 selector 错误
3. 优先在 Phase II 解决高层 archetype 选择错误问题

---

## Phase II 后续优化方向

当前达成共识的优化重点如下。

### 优先级 1: 改善 Phase II 的训练信号

核心思路：

- 减少 selector 对 imitation / archetype label 的依赖
- 提高其对真实环境收益的敏感性

具体方向：

- 下调 `selection_alpha`
- 或设计 `selection_alpha` schedule
- 让模型逐渐从“模仿 train 上的 archetype 标签”过渡到“更依赖真实 reward”

### 优先级 2: 增强 Phase II 的输入上下文

当前 selector 只看 horizon 第一个 bar 的状态向量，这可能不足以判断整个 horizon 的 archetype。

可考虑增强的信息：

- 前若干个 bar 的价格变化摘要
- 波动率 / ATR 类特征
- 趋势强度特征
- 更长的上下文状态编码

### 优先级 3: 强化诊断

在真正修改训练逻辑前，应继续增强诊断能力，明确 selector 失败的具体模式。

重点观察：

- 在 `val/dev-test` 上的 archetype 选择分布
- 是否坍缩到某个 archetype（如 flat archetype）
- 每个 archetype 在不同 split 上的条件收益
- selector 的置信度与误判模式

### 优先级 4: 中期再考虑扩大 codebook 表达能力

如果确认 Phase II 不仅是训练信号问题，还存在 archetype 表达不足，则可进一步考虑：

- 增大 `K`（如 10 -> 16/20）
- 提升 archetype 对不同市场形态的区分度

但这一项不应与前几项同时混改，避免变量耦合。

---

## 当前结论摘要

- Phase III 在 `AL` 上确实完整训练并评估过，但没有改变最终行为
- 原论文中的 Phase III 本来就不是用来重新选择 archetype 的
- selector 错误更应视为 Phase II 的核心问题
- 当前流程中严格意义上的 final test 已缺失，需要升级为 `train / val / dev-test / final-test`
- 现阶段不优先扩展 Phase III，而是优先增强 Phase II 的训练信号、输入上下文和诊断能力

---

## 后续行动建议

下一阶段建议按如下顺序推进：

1. 固化新的数据协议定义，避免 `test` 与 `final-test` 混用
2. 为 Phase II 增加更强的诊断输出
3. 开始测试降低 `selection_alpha` 或引入 schedule
4. 评估 selector 输入上下文增强方案
5. 在 Phase II 方案稳定后，再决定是否需要扩展 Phase III 架构
