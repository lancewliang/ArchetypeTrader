# Phase I 基于新 DP 结构的优化记录

> 日期: 2026-04-19  
> 交易对: AL  
> 主题: 基于新的 DP trajectory chunk / sampling 结构，对 Phase I archetype discovery 做持续优化，并记录完整测试过程  
> 当前结论: `return bucket + 轻量 usage-profit alignment + strict profit gate` 是目前最有效的 Phase I 方案，`batch_17-short` 已经整体超过 `batch_09-short`

---

## 1. 背景与目标

这轮优化的核心目标不是单纯提高 Phase I 重建精度，而是让 **Phase I 学出来的 archetype 在 Phase II 中真正可交易、可选择、可泛化**。

在 AL 数据上，早期现象很明确：

- DP 本身是对的，但 Phase I 学到的 archetype 经常不赚钱。
- 即便 decoder 能很好重放 DP 轨迹，也不代表固定 archetype 在验证环境中有高收益。
- Phase II 的收益高度依赖 Phase I 的 `best_fixed_archetype_return_mean`、`realizable_proxy` 和 `return_usage_correlation`。

因此，这一轮优化采用了“先修 Phase I，再看 Phase II”的路线。

---

## 2. 新 DP 结构与采样结构

### 2.1 DP trajectory chunk 的基础结构

当前 Phase I 使用的 DP 轨迹结构是：

1. 对时间序列做 **滑窗**，枚举所有合法起点 `start in [0, T-h]`。
2. 每个窗口长度为 `horizon = 72`。
3. 每个窗口内部运行 `paper_single_change` 变体：
   - 最多一次动作变化
   - 本质上是单次交易约束的 DP 规划
4. 生成 `(states, actions, rewards)` 三元组作为 Phase I 示范轨迹。

对应代码：
- `src/phase1/dp_planner.py`
- `scripts/train_phase1.py`

### 2.2 对 DPPlanner 的结构增强

在这轮优化中，`DPPlanner` 做了以下增强：

- 新增起点采样模式：
  - `uniform`
  - `stratified`
  - `hybrid_stratified_importance`
- 新增采样元数据写入 trajectory cache：
  - `sampling_mode`
  - `sampling_stratified_ratio`
  - `sampling_importance_ratio`
  - `sampling_num_strata`
  - `sampling_importance_vol_weight`
  - `sampling_importance_net_weight`
  - `sampling_stratified_count`
  - `sampling_importance_count`
- 增加 trajectory cache 兼容检查，避免旧 cache 被误复用。

相关代码文件：
- `src/phase1/dp_planner.py`
- `scripts/train_phase1.py`
- `src/config.py`

### 2.3 采样结论

这轮实验后，采样层面的结论已经比较明确：

- `90% stratified + 10% importance` 会把训练分布推向高收益/高波动窗口，导致 Phase I archetype 训练分布偏移。
- `pure stratified` 能修复一部分问题，但 **仅靠采样仍然不够**。
- 真正决定结果的关键，不是继续折腾采样，而是让 **code semantics** 更贴近收益结构。

最终成功方案 `batch_17-short` 实际运行使用的是：

- `sampling_mode = stratified`
- 启动日志显示 `sampling_mode=stratified(1.00/0.00)`

对应日志：
- `logs/AL/batch_17-short/AL_pipeline_20260419_215837.log`

---

## 3. Phase I 代码改动总览

### 3.1 Profit Gate 与 checkpoint 选择

为避免“训练了很久，但选中的 Phase I checkpoint 根本不该进 Phase II”，新增了 profit gate：

硬门槛：
- `realizable_proxy / oracle >= 0.40`
- `best_fixed / oracle >= 0.50`
- `return_usage_correlation >= 0`

作用：
- 候选 checkpoint 先经过收益准入过滤
- 只有满足收益约束的 checkpoint 才能进入选择池
- 若一个都不过，默认直接报错

相关代码：
- `scripts/train_phase1.py`
- `src/config.py`

### 3.2 Codebook 初始化与死码重置

新增 profit-aware 版本：

- `direction + return aware` 初始化
- 死码重置优先从高收益样本中抽取 latent

目的：
- 避免 codebook 从一开始就被低质量样本占据
- 提高高收益 archetype 的成型概率

相关代码：
- `src/phase1/codebook.py`

### 3.3 Phase I 主体训练机制的几轮演化

#### A. usage-profit alignment + separation + profit-aware init/reset

做法：
- 收益-使用率对齐正则
- codebook cosine separation penalty
- profit-aware 初始化 / 死码重置

结果：
- 修复了负相关的一部分问题
- 但会把 archetype 收益谱系压平
- 高收益 archetype 被削弱，`best_fixed` 反而下降

结论：
- 不能把 `alignment` 作为主驱动力

#### B. return regression auxiliary head

做法：
- 用 `z_e / z_q` 去回归轨迹收益

结果：
- 训练里 `return_reg` 太小
- decoder 重放能力强，但 code semantics 没真正改善
- `best_fixed` 仍然上不来

结论：
- 连续收益回归目标太弱，不足以约束 code semantics

#### C. return bucket objective

最终有效方案：
- 把轨迹收益按分位数切成 bucket
- 用 `code -> bucket` 的分类目标代替弱回归目标
- 同时使用：
  - `soft assignment -> code logits`
  - `hard assigned code -> code logits`
- 以交叉熵形式训练收益语义

附加日志指标：
- `return_reg`
- `bucket_acc`
- `usage_corr`

结果：
- 终于把“赚钱 archetype”本身学出来了
- `best_fixed`、`realizable_proxy` 显著抬升

#### D. 轻量 alignment（最终有效增强）

在 return bucket 已经生效后，只加一层很轻的 assignment 修正：

- `phase1_usage_profit_alignment_weight = 0.01`
- `phase1_usage_profit_alignment_target_corr = 0.02`

作用：
- 不是主导训练
- 只把 `return_usage_correlation` 从负值往正值轻推
- 避免再次把收益谱系压平

---

## 4. 关键实验时间线

### 4.1 基线: `batch_09-short`

角色：旧基线，后续所有优化都以它为参考。

Phase I:
- 选中 `epoch=300`
- `realizable_proxy = 160.42`
- `best_fixed = 172.10`
- `return_usage_correlation = 0.2358`

Phase II:
- BEST `avg_return = 96.10`
- FINAL `avg_return = 86.34`

Phase II 评估：
- val: `TR=1.1472`, `ASR=9.2787`, `MDD=0.0201`
- test: `TR=2.3042`, `ASR=9.8067`, `MDD=0.0190`

结论：
- 是一个强基线，但 Phase I 仍然比较“集中使用少数 archetype”。

### 4.2 `batch_11-short`: 90/10 混合采样

改动：
- `90% stratified + 10% importance`

结果：
- `realizable_proxy = 45.37`
- `best_fixed = 44.05`
- Phase II BEST `avg_return = -1.17`
- val `TR=0.3908`
- test `TR=1.0567`

结论：
- importance sampling 明显推坏了训练分布。

### 4.3 `batch_12-short`: pure stratified

改动：
- 回退到纯分层采样

结果：
- `realizable_proxy = 62.50`
- `best_fixed = 63.79`
- Phase II BEST `avg_return = 22.99`
- val `TR=0.5413`
- test `TR=1.1011`

结论：
- 纯采样修正只能恢复一部分，不够解决根因。

### 4.4 `batch_13-short`: strict profit gate

改动：
- 引入严格 profit gate

结果：
- 直接报错，不进入 Phase II
- 最优 fallback 候选：`epoch=240`
- 失败原因：
  - `best_fixed_archetype_return_mean=63.7909 < required=119.9311`
  - `return_usage_correlation=-0.8432 < 0`

结论：
- 选错 checkpoint 不是主因，整轮 Phase I 都不够好。

### 4.5 `batch_14-short`: 强 alignment / separation / profit-aware init-reset

改动：
- 强化收益-使用率对齐
- separation 正则
- profit-aware init/reset

结果：
- gate 仍未通过
- `best_fixed ≈ 44.19`
- `realizable ≈ 43.99`

结论：
- 正相关问题被修了一部分
- 但收益谱系被压平，真正赚钱的 archetype 消失了

### 4.6 `batch_15-short`: return regression auxiliary head

改动：
- 关闭强 alignment
- 加收益回归辅助头

结果：
- `return_reg` 太弱
- `best_fixed` 最高只有 `38.10`
- gate 失败

结论：
- 连续回归头不足以塑造 code-level profit semantics

### 4.7 `batch_16-short`: return bucket objective

改动：
- 上收益分桶目标
- 用 bucket classification 替代弱回归头

结果：
- `bucket_acc` 从 `0.42` 升到 `0.79+`
- `best_fixed` 恢复到 `178.41`
- gate 只剩一个失败项：
  - `return_usage_correlation = -0.1758 < 0`

结论：
- 赚钱 archetype 已经学出来了
- 但 assignment 还没完全跟上 semantics

### 4.8 `batch_17-short`: return bucket + 轻量 alignment

改动：
- 保留 return bucket 主目标
- `alignment_weight = 0.01`
- `alignment_target_corr = 0.02`

结果：
- `9 / 16` 个 checkpoint 通过 profit gate
- 最终选中 `epoch=420`
- `realizable_proxy = 201.37`
- `best_fixed = 191.04`
- `return_usage_correlation = 0.2332`

Phase II:
- BEST `avg_return = 116.61`
- FINAL `avg_return = 107.97`

Phase II 评估：
- val: `TR=1.3378`, `ASR=11.0851`, `MDD=0.0189`
- test: `TR=2.3079`, `ASR=11.0322`, `MDD=0.0114`

结论：
- 这是当前最优组合
- 已经实质超过 `batch_09-short`

---

## 5. 核心指标对比

### 5.1 Phase I 核心指标对比

| 批次 | 主要改动 | Phase I 结果 | 结论 |
|---|---|---:|---|
| `batch_09-short` | 旧基线 | `realizable=160.42`, `best_fixed=172.10`, `corr=0.2358` | 强基线 |
| `batch_11-short` | 90/10 混合采样 | `realizable=45.37`, `best_fixed=44.05` | 训练分布被 importance 拉坏 |
| `batch_12-short` | pure stratified | `realizable=62.50`, `best_fixed=63.79` | 仅修采样不够 |
| `batch_13-short` | strict profit gate | gate fail | 暴露了 Phase I 整体质量不足 |
| `batch_14-short` | 强 alignment + separation | `best_fixed=44.19` | 正相关改善，但收益被压平 |
| `batch_15-short` | return regression head | `best_fixed=38.10` | 弱回归头无效 |
| `batch_16-short` | return bucket | `best_fixed=178.41` | archetype 赚钱能力恢复 |
| `batch_17-short` | return bucket + 轻量 alignment | `realizable=201.37`, `best_fixed=191.04`, `corr=0.2332` | 当前最优 |

### 5.2 Phase II / 评估口径对比

| 批次 | Phase II BEST | Phase II FINAL | val TR / ASR / MDD | test TR / ASR / MDD |
|---|---:|---:|---:|---:|
| `batch_09-short` | `96.10` | `86.34` | `1.1472 / 9.2787 / 0.0201` | `2.3042 / 9.8067 / 0.0190` |
| `batch_11-short` | `-1.17` | `-10.73` | `0.3908 / 3.5425 / 0.0159` | `1.0567 / 4.8421 / 0.0520` |
| `batch_12-short` | `22.99` | `22.99` | `0.5413 / 4.8049 / 0.0144` | `1.1011 / 4.6680 / 0.0709` |
| `batch_17-short` | `116.61` | `107.97` | `1.3378 / 11.0851 / 0.0189` | `2.3079 / 11.0322 / 0.0114` |

解释：
- `batch_17-short` 的 test `TR` 只比 `batch_09-short` 略高一点，但 **ASR 明显更高，MDD 明显更低**。
- 这说明它不是“靠更大风险换收益”，而是 **风险调整后的整体质量更优**。

---

## 6. 当前生效的代码变更

### 6.1 主要改动文件

- `src/phase1/dp_planner.py`
  - 新增分层 / 混合采样
  - 写入 trajectory cache 元数据
  - 保持 `paper_single_change` DP 规划结构

- `scripts/train_phase1.py`
  - 新增 trajectory cache 兼容检查
  - 新增 profit gate checkpoint 选择
  - 新增 return bucket head / bucket loss
  - 新增 `bucket_acc`, `usage_corr` 训练日志
  - 集成轻量 alignment

- `src/phase1/codebook.py`
  - profit-aware 初始化
  - profit-aware 死码重置

- `src/config.py`
  - 新增采样、Phase I 正则、return bucket、profit gate 等超参数

### 6.2 当前最有效的 Phase I 配置（来自 `batch_17-short` 启动日志）

- `sampling_mode = stratified(1.00/0.00)`
- `align_w = 0.010`
- `align_target = 0.02`
- `return_w = 0.100`
- `return_hidden = 32`
- `return_bins = 5`
- `return_soft = 0.50`
- `sep_w = 0.020`
- `sep_margin = 0.35`
- `init_top = 0.25`
- `init_code = 0.50`
- `reset_top = 0.25`

对应日志：
- `logs/AL/batch_17-short/AL_pipeline_20260419_215837.log`

---

## 7. 结论

### 7.1 已经确认的结论

1. **采样不是主矛盾**。
   - importance 采样会明显伤害 Phase I。
   - pure stratified 只能部分修复。

2. **Phase I 的关键，不是重建得多像，而是 code semantics 是否对齐收益结构**。
   - decoder 能重放 DP，不代表 archetype 可交易。

3. **return bucket objective 是这轮最关键的突破点**。
   - 它第一次真正把“赚钱 archetype”学出来了。

4. **light alignment 是必要但必须克制的补充**。
   - 强 alignment 会把收益谱系压平。
   - 轻量 alignment 可以在不伤害 `best_fixed` 的前提下，把 `corr` 拉回正区间。

5. **`batch_17-short` 已经超过 `batch_09-short`**。
   - Phase I 更强
   - Phase II 验证更强
   - val/test 评估口径整体更优

### 7.2 当前仍保留的问题

`batch_17-short` 虽然已经超过旧基线，但 `Phase II` 健康度仍为 `weak_edge`：

- 优势已存在
- 但验证收益与执行成本仍处于同量级

这意味着：
- Phase I 主方向已经找对
- 下一阶段的主要优化重心应该转向 Phase II
- 但当前阶段先冻结 Phase II 调参是合理的，因为 Phase I 的新结构已经稳定可用

---

## 8. 推荐后续动作

### 8.1 当前建议

先冻结当前 Phase I 方案，不再继续大改。原因：

- 已经跑出超过 `batch_09-short` 的结果
- 目前 Phase I 的收益结构、gate 通过率、Phase II 表现都已经进入正循环

### 8.2 如果未来继续优化

推荐顺序：

1. 以 `batch_17-short` 为 Phase I 基线固定下来
2. 下一轮重点改 Phase II，而不是继续重写 Phase I
3. 如果要继续动 Phase I，只做微调：
   - 轻微调整 `alignment_weight`
   - 不再回到强 alignment / 强 importance / 弱回归头路线

---

## 9. 相关日志与结果目录

关键日志：
- `logs/AL/batch_09-short/AL_pipeline_20260419_144814.log`
- `logs/AL/batch_11-short/AL_pipeline_20260419_162002.log`
- `logs/AL/batch_12-short/AL_pipeline_20260419_171126.log`
- `logs/AL/batch_13-short/AL_pipeline_20260419_181131.log`
- `logs/AL/batch_14-short/AL_pipeline_20260419_185249.log`
- `logs/AL/batch_15-short/AL_pipeline_20260419_203433.log`
- `logs/AL/batch_16-short/AL_pipeline_20260419_212553.log`
- `logs/AL/batch_17-short/AL_pipeline_20260419_215837.log`

关键结果目录：
- `result/AL/batch_17-short/phase1_archetype_discovery/`
- `result/AL/batch_17-short/phase2_archetype_selection/`
- `result/AL/batch_17-short/phase2_eval_val/`
- `result/AL/batch_17-short/phase2_eval_test/`

