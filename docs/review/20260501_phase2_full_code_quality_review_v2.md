# Phase II Archetype Selection 完整代码 Review 报告（第二轮）

> 生成日期: 2026-05-01
> Review 范围: Phase II 全部实现代码（27 个源文件 + 2 个入口脚本 + 24 个测试文件）
> Review 依据: `docs/design/phase2_archetype_selection_design.md` + `docs/plan/phase2_archetype_selection_execution_plan.md`
> 前次 Review: `docs/review/20260501_phase2_full_code_quality_review.md`

---

## 1. Review 总结

| 维度 | 评级 | 说明 |
| --- | --- | --- |
| 架构对齐 | ✅ 基本对齐 | 主流程完整，模块边界清晰 |
| 实现完整性 | ⚠️ 大部分完成 | 核心链路完整，4 个关键功能未落地 |
| Bug | 🟡 存在中等 Bug | 0 个 P0 逻辑缺陷，3 个 P1 Bug |
| 工程缺陷 | 🟡 多处 | 7 个工程实现不够严谨的点 |
| 风险 | 🟡 中等风险 | 3 个可能影响训练/部署的风险 |

### 与前次 Review 对比

前次 Review 标记了 **2 个 P0 Bug + 5 个高风险点 + 10+ 配置缺失**。本轮 Review 确认以下问题已修复:

| 前次问题 | 当前状态 |
| --- | --- |
| Rolling Validation 实现错误（跑完整 val 再切片） | ✅ 已修复 — `entry_indices` 参数传入 fold 子集 |
| HorizonEnv 没有 truncated 语义 | ✅ 已修复 — PPOTrainer 在 rollout 末尾设 `truncated=True` |
| HorizonEnv.reset() 不接受 prev_terminal_position | ✅ 已修复 — 新增参数 + `restore_state()` 方法 |
| PPOTrainer 直接访问私有属性 | ✅ 已修复 — 改用 `env.restore_state()` |
| Composite Score 缺失 | ✅ 已修复 — `compute_phase2_composite_score` + sensitivity 分析 |
| Config 层大量字段缺失 | ✅ 已修复 — 补齐 18 个配置子组 |
| reward_scaling.clip_range 未实现 | ✅ 已修复 — `_scale_reward()` 实现 clip |
| explained_variance 未计算 | ✅ 已修复 — `PPOTrainer.update()` 调用 |
| dead code mask 阈值硬编码为 0 | ✅ 已修复 — `build_dead_code_mask()` 使用可配置阈值 |
| ScheduleManager KL 退火硬编码 | ✅ 已修复 — 读取 `kl_demo_anneal_to` / `kl_demo_anneal_fraction` |
| phase2_report.json 大量审计字段缺失 | ✅ 已修复 — 补齐 horizon_schedule / data_gap_filter / input_norm / env_shards / reward_scaling / cost_config_inherited / baselines / rolling_validation / resume_ready / guardrails |
| Phase2Dataset 无防御性校验 | ✅ 已修复 — `_validate_inputs()` 校验 feature / timestamp / split / horizon 边界 |
| horizon_index 产物字段缺失 | ✅ 已修复 — 补齐 last_execution_row / last_markout_row / phase1_sample_id / prev_terminal_position / timestamp_start |

---

## 2. 当前 Bug 清单

### 2.1 🟡 P1: Rolling Validation fold 间仓位继承缺失

**位置:** [phase2_evaluator.py:L182-L188](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/evaluation/phase2_evaluator.py#L182-L188)

**问题描述:** `evaluate_rolling_validation()` 调用 `run_walk_forward()` 时 `initial_position=0` 硬编码。每个 fold 都从 flat 开始，而非从上一个 fold 的终止仓位继承。

**影响:**
- fold 间的边界换仓成本被系统性低估
- rolling validation 无法检测仓位连续继承场景下的泛化性能
- 设计 §8.9 要求 "固定 fold 切法与种子"，隐含 fold 间仓位连续

**修复建议:** 在 fold 循环中传递上一个 fold 的 `final_position` 作为下一个 fold 的 `initial_position`。

### 2.2 🟡 P1: Phase2SelectionPolicy.evaluate() 中 rolling_result 参数完全未使用

**位置:** [phase2_selection_policy.py:L60](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase2_selection_policy.py#L60)

**问题描述:** 方法签名接受 `rolling_result: Optional[Dict[str, float]] = None`，但方法体中完全没有使用它。设计 §8.9 明确要求:

> rolling validation sign-off 附加硬约束。

当前实现完全忽略了 rolling validation 结果对 best 选择的影响。

**影响:** 即使 rolling validation 发现某个 checkpoint 在最差 fold 上表现极差，仍可能被选为 best。

**修复建议:** 在 `evaluate()` 中检查 `rolling_result` 的 `worst_fold_quantile` 和 `fold_volatility`，超阈值时拒绝 promote。

### 2.3 🟡 P1: Phase2Trainer.run() 完全没有实现 resume 逻辑

**位置:** [phase2_trainer.py:L102-L510](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase2_trainer.py#L102-L510)

**问题描述:** `config.resume_from` 字段存在，CLI `--resume-from` 参数被正确解析，但 `Phase2Trainer.run()` 从未读取这个字段来恢复训练状态。

**影响:**
- 训练中断后无法从 checkpoint 恢复
- 长时间训练（`total_timesteps=1M+`）中断后必须从头开始

**修复建议:** 在 `run()` 开头检查 `config.resume_from`，若非 None 则加载 `last_selector.pt` 中的 optimizer / schedule / env 状态，跳过已完成的 update 步数。

---

## 3. 实现不完整项

### 3.1 配置存在但功能未落地的模块

| 模块 | 配置类 | 实现状态 | 严重程度 |
| --- | --- | --- | --- |
| Early Stopping | `EarlyStoppingConfig` | ❌ 训练循环中无 early stopping 逻辑 | 🟡 P2 |
| Cost Alignment Check | `CostAlignmentCheckConfig` | ❌ 未校验 Phase I/II cost_config 一致性 | 🟡 P2 |
| Distribution Shift 监控 | `DistributionShiftConfig` + `Phase2DistributionShiftMonitor` | ❌ 模块存在但训练流程未调用 | 🟡 P2 |
| Execution Stress 测试 | `ExecutionStressConfig` + `Phase2ExecutionStressRunner` | ❌ 模块存在但训练流程未调用 | 🟡 P2 |
| Reward Normalization | `RewardNormalizationConfig` + `PPOConfig.reward_normalization` | ❌ 两处配置存在但均未实现 running_mean_std | 🟡 P2 |
| Resume | `ResumeConfig` + `config.resume_from` | ❌ 配置存在但 `run()` 未实现恢复逻辑 | 🟡 P1 |

### 3.2 Baseline 缺失

| Baseline | 设计依据 | 状态 |
| --- | --- | --- |
| `random_selector` | 设计 §8.2 | ✅ 已实现 |
| `single_archetype_k` | 设计 §8.2 | ✅ 已实现 |
| `buy_and_hold` | 设计 §8.2 | ⚠️ 实现为 `archetype_code=1`（flat），非真正 buy-and-hold |
| `phase1_demo_label` | 设计 §8.2 | ❌ 仅 docstring 提及，未实现 |
| `dp_teacher` | 设计 §8.2 | ❌ 未实现 |

**buy_and_hold 语义问题:** 当前 `buy_and_hold` baseline 选择 archetype code 1，让 decoder 自行决定每步动作。但 TradingEnv 的 action 映射是 `{0: -max_pos, 1: 0, 2: +max_pos}`，action=1 是 flat。这意味着 "buy_and_hold" 实际上是 "always_flat"，与金融语义中的 buy-and-hold（持续持有多头）完全不同。

### 3.3 其他缺失

| 缺失项 | 严重程度 | 说明 |
| --- | --- | --- |
| Periodic checkpoint (`checkpoints/step_*.pt`) | 🟢 P3 | 只保存 last/best，无法回溯中间状态 |
| `kl_demo_dominance_ratio` 监控与告警 | 🟡 P2 | `compute_kl_demo_dominance_ratio()` 存在但训练循环未监控 |
| Unmasked Diagnostic Rollout | 🟡 P2 | 训练初期探测 dead code 的无 mask rollout 未实现 |
| Train metrics 输出 | 🟡 P2 | 训练集的 per-horizon records 用 best selector 生成，不反映训练过程 |
| `fold_seed` 未使用 | 🟢 P3 | `RollingValidationConfig.fold_seed` 存在但 fold 切法未使用种子 |

---

## 4. 工程实现缺陷

### 4.1 Phase2Dataset.get_horizon_inputs() 中 row 越界 silent fallback

**位置:** [phase2_dataset.py:L194-L195](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/phase2_dataset.py#L194-L195)

```python
if row >= self._num_rows:
    row = self._num_rows - 1
```

当 `execution_row` 超出数据范围时，静默使用最后一行数据。这可能导致不易察觉的数据错误——不同 timestep 的 execution_book 可能被映射到同一行盘口数据。

**修复建议:** 至少记录 warning；或在 `Phase2HorizonIndexer.build_index()` 中确保 `last_execution_row < num_rows`。

### 4.2 Phase2BacktestRunner._run_fixed_strategy() 中 baseline record 缺少 boundary_cost

**位置:** [phase2_replay.py:L333-L343](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/evaluation/phase2_replay.py#L333-L343)

Baseline 的 `Phase2HorizonReplayRecord` 构造中没有 `boundary_cost` 字段（使用默认值 0.0），但主路径的 record 包含。这导致 baseline 的 `total_boundary_cost` 指标始终为 0，与主路径不可比。

### 4.3 Phase2HorizonIndexer.build_index() 中 gap_bars 计算语义混乱

**位置:** [phase2_horizon_index.py:L267](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/phase2_horizon_index.py#L267)

```python
gap_bars = max(gap_bars, int(gap_minutes))
```

`gap_bars` 字段名暗示 bar 数量，但实际记录的是分钟数（`int(gap_minutes)`）。如果 bar 间隔不是 1 分钟，这个值就不正确。同时，`gap_threshold` 的选择逻辑也有问题（line 241-244）:

```python
gap_threshold = (
    self.config.horizon_schedule.max_allowed_gap_minutes
    if self.config.horizon_schedule.data_gap_check_enabled
    else self.config.horizon_schedule.gap_threshold_bars
)
```

当 `data_gap_check_enabled=True` 时使用分钟阈值，否则使用 bar 阈值。但两者单位不同，比较逻辑（`gap_minutes > gap_threshold`）在 `data_gap_check_enabled=False` 时会把 bar 阈值当作分钟阈值来用。

### 4.4 PPOTrainer.collect_rollout() 直接访问 env 内部属性

**位置:** [ppo_trainer.py:L152-L153](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/rl/ppo_trainer.py#L152-L153)

```python
current_horizon_idx = env.horizon_indices[env.cursor] if env.cursor < len(env.horizon_indices) else None
entry = env.dataset.horizon_entries[current_horizon_idx] if current_horizon_idx is not None else None
```

直接访问 `env.horizon_indices`、`env.cursor`、`env.dataset.horizon_entries` 来获取 kl_label 和 is_labeled。这违反了封装原则，且如果 HorizonEnv 内部数据结构变化会导致隐蔽的 bug。

**修复建议:** 在 `HorizonEnv` 上提供 `current_label_info()` 方法返回 `(kl_label, is_labeled)`。

### 4.5 Phase2BacktestRunner 中 risk flatten 逻辑与 HorizonEnv 重复

**位置:** [phase2_replay.py:L155-L171](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/evaluation/phase2_replay.py#L155-L171) vs [horizon_env.py:L266-L298](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trading/horizon_env.py#L266-L298)

`run_walk_forward()` 中实现了独立的 risk flatten 逻辑（使用局部变量 `cumulative_loss` / `consecutive_losses`），与 `HorizonEnv._handle_mid_horizon_flatten()` 逻辑重复但实现不同。两处逻辑需要保持同步，增加了维护成本。

**修复建议:** 提取为共享的 `RiskStateManager` 类，HorizonEnv 和 BacktestRunner 共用。

### 4.6 Phase2Trainer.run() 中 all_entries 变量定义但未使用

**位置:** [phase2_trainer.py:L169](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase2_trainer.py#L169)

```python
all_entries = train_entries + val_entries + test_entries
```

此变量后续未使用，应删除。

### 4.7 Phase2Dataset._resolve_reward_alignment() 重复读 Phase I config

**位置:** [phase2_dataset.py:L144-L155](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/phase2_dataset.py#L144-L155)

`Phase2HorizonIndexer` 和 `Phase2Dataset` 都各自读取 `phase1_config.yaml` 来获取 `reward_alignment`，没有共享。每次创建 `Phase2Dataset` 实例都会重复读文件和解析 YAML。

**修复建议:** 在 `Phase2Trainer.run()` 中统一读取一次，通过参数传递。

---

## 5. 风险点

### 5.1 🟡 frozen_policy 的 fallback 路径丢失时序上下文

**位置:** [phase1_frozen_policy.py:L184-L199](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/models/phase1_frozen_policy.py#L184-L199)

当 decoder 缺少 `state_proj/lstm/head` 子模块时，fallback 路径每步独立推理，LSTM 时序上下文完全丢失。虽然第一次调用会发 warning，但:

1. 如果 decoder 结构名称不匹配但功能正确（如自定义子类），warning 可能被忽略
2. 丢失时序上下文意味着每步的 action 完全基于当前输入，无法利用历史信息
3. 这会直接影响 HorizonEnv 的 reward 计算，但不会报错

**风险等级:** 中等。标准 `ArchetypeDecoder` 有这些子模块，fallback 只在非标准 decoder 时触发。

### 5.2 🟡 entropy coef 退火到 0 可能导致策略塌缩

**位置:** [scheduling.py:L84](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/rl/scheduling.py#L84)

```python
self._current_entropy_coef = self._initial_entropy_coef * (1.0 - progress)
```

Entropy coef 线性退火到 0。接近训练结束时 entropy bonus 完全消失，加上 KL/demo regularization，selector 可能塌缩到单一 action。

**缓解措施:** `Phase2SelectionPolicy` 有 `max_action_dominance_ratio` guardrail，但只在 checkpoint 选择时生效，不阻止训练后期的策略退化。

**修复建议:** 设置 entropy coef 下界（如 `max(initial * 0.1, 1e-4)`），或使用 cosine 退火代替线性退火。

### 5.3 🟡 val 快速评估实际跑完整 val set

**位置:** [phase2_evaluator.py:L71-L72](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/evaluation/phase2_evaluator.py#L71-L72)

`evaluate_val_fast()` 方法名暗示 "快速评估（子集）"，但实际跑完整 val walk-forward。对于大 val set，每次评估可能耗时数分钟，在训练循环中频繁评估会显著拖慢训练。

**影响:** 假设 `eval_every=97`，总共 20 次评估，每次 2 分钟，额外开销约 40 分钟。

**修复建议:** 实现真正的子集评估（如每隔 N 个 horizon 取一个），或缓存 selector state 避免重复计算。

---

## 6. 优先级矩阵

| 优先级 | 问题 | 类别 | 影响 |
| --- | --- | --- | --- |
| P1 | Resume 逻辑未实现 | 缺失 §3.1 | 长训练中断无法恢复 |
| P1 | Rolling Validation fold 仓位继承缺失 | Bug §2.1 | fold 间边界成本低估 |
| P1 | rolling_result 未用于 best 选择 | Bug §2.2 | rolling validation 形同虚设 |
| P2 | Early stopping 未实现 | 缺失 §3.1 | 过拟合无法自动停止 |
| P2 | Cost alignment check 未实现 | 缺失 §3.1 | Phase I/II 成本不一致无法检测 |
| P2 | Distribution shift 监控未调用 | 缺失 §3.1 | OOD 场景无法检测 |
| P2 | Execution stress 未调用 | 缺失 §3.1 | 成本压力无法评估 |
| P2 | Reward normalization 未实现 | 缺失 §3.1 | 配置存在但无效 |
| P2 | buy_and_hold baseline 语义错误 | 缺失 §3.2 | baseline 不可比 |
| P2 | phase1_demo_label / dp_teacher baseline 缺失 | 缺失 §3.2 | 缺少关键对照 |
| P2 | gap_bars 计算语义混乱 | 缺陷 §4.3 | gap 检测可能不准 |
| P2 | row 越界 silent fallback | 缺陷 §4.1 | 隐蔽数据错误 |
| P2 | baseline record 缺 boundary_cost（复核后不采纳，见 §10.2） | 缺陷 §4.2 | 当前代码已写入 boundary_cost |
| P3 | env 内部属性直接访问 | 缺陷 §4.4 | 封装破坏 |
| P3 | risk flatten 逻辑重复 | 缺陷 §4.5 | 维护成本 |
| P3 | all_entries 未使用 | 缺陷 §4.6 | 死代码 |
| P3 | reward_alignment 重复读取 | 缺陷 §4.7 | 性能 |
| P3 | entropy coef 退火到 0 | 风险 §5.2 | 策略塌缩 |
| P3 | val 评估跑完整 set | 风险 §5.3 | 训练效率 |

---

## 7. 代码质量亮点

本轮 Review 也注意到以下显著的代码质量改进:

1. **配置层完整度大幅提升** — 从前次的 ~60% 提升到 ~95%，18 个配置子组全部补齐
2. **防御性校验** — `Phase2Dataset._validate_inputs()` 覆盖 feature / timestamp / split / horizon 边界
3. **封装改进** — `HorizonEnv.restore_state()` 替代直接访问私有属性
4. **Composite Score 完整实现** — 加权计算 + sensitivity 分析 + debug 信息
5. **Report 审计字段补齐** — horizon_schedule / data_gap_filter / input_norm / env_shards / reward_scaling / cost_config_inherited / baselines / rolling_validation / resume_ready / guardrails
6. **Dead code mask 可配置化** — `build_dead_code_mask()` 使用 Phase I report 的 usage ratio + 可配置阈值
7. **Truncated 语义正确实现** — PPOTrainer 在 rollout 末尾正确设置 `truncated=True`
8. **KL demo 退火可配置** — ScheduleManager 读取 `kl_demo_anneal_to` / `kl_demo_anneal_fraction`
9. **Test label 泄漏防护** — 双重检查（guard + per-entry check）
10. **Online Action Throttle** — 完整实现 confidence / switch-rate / cooldown / position-change 限制

---

## 8. 建议修复优先级

### 第一优先级（阻塞正式训练）:
1. 实现 Resume 逻辑（`Phase2Trainer.run()` 读取 `config.resume_from`）
2. 修复 Rolling Validation fold 仓位继承
3. 将 `rolling_result` 纳入 `Phase2SelectionPolicy.evaluate()` 的 guardrail

### 第二优先级（阻塞部署 sign-off）:
1. 实现 Early Stopping 逻辑
2. 实现 Cost Alignment Check
3. 调用 Distribution Shift 监控和 Execution Stress 测试
4. 修复 buy_and_hold baseline 语义
5. 实现 phase1_demo_label baseline
6. 修复 gap_bars 计算语义

### 第三优先级（工程优化）:
1. 提取共享 RiskStateManager
2. 为 HorizonEnv 添加 `current_label_info()` 方法
3. 实现 val 快速评估子集
4. 设置 entropy coef 下界
5. 删除 `all_entries` 死代码
6. 共享 reward_alignment 读取

---

## 9. 总结

Phase II 实现代码相比前次 Review 有**显著改进**。前次标记的 13 个关键问题已全部修复，核心训练链路（加载 Phase I 产物 → horizon index → HorizonEnv → PPO 训练 → 评估 → 报告）完整且正确。

当前剩余问题集中在**配置存在但功能未落地**的 6 个模块（resume / early stopping / cost alignment / distribution shift / execution stress / reward normalization），以及 3 个 P1 级 Bug（rolling validation 仓位继承 / rolling_result 未使用 / resume 逻辑缺失）。

建议在启动正式训练前修复 P1 级别的 3 个问题，在部署 sign-off 前修复 P2 级别中被采纳的问题；不采纳和暂缓项见 §10.2。

---

## 10. 变更采纳决策补充

> 生成变更计划时对本 Review 条目做了二次核对。执行计划见:
> `docs/changes/20260501_phase2_review_v2_change_execution_plan.md`

### 10.1 采纳的变更

| Review 条目 | 采纳状态 | 说明 |
| --- | --- | --- |
| `Phase2Trainer.run()` 未读取 `config.resume_from` | ✅ 采纳 | `PPOTrainer.load_state()` 已具备底层恢复能力，但 trainer 主流程没有接入，是正式长训练阻塞项 |
| Rolling Validation fold 间仓位继承缺失 | ✅ 采纳 | 当前每个 fold `initial_position=0`，应按时间顺序继承上一 fold 的 `final_position` |
| `rolling_result` 未用于 best selection guardrail | ✅ 采纳 | rolling validation 是 sign-off 附加硬约束；candidate best 应在 rolling 不达标时 reject |
| Early Stopping 未落地 | ✅ 采纳 | 训练循环已有周期 eval，补充 patience/min_delta 逻辑收益高、风险低 |
| Cost Alignment Check 未落地 | ✅ 采纳 | Phase I/II 成本、仓位、reward alignment 不一致会直接破坏回测可比性 |
| Distribution Shift 监控未调用 | ✅ 采纳 | 模块已存在，应接入 train fit、val/test score 和 report summary |
| Execution Stress 未调用 | ✅ 采纳 | 模块已存在，但需要接入真实 cost/lag scenario，作为部署 sign-off 报告项 |
| `buy_and_hold` baseline 语义错误 | ✅ 采纳 | 应改为 action-level `buy_and_hold_long` / `buy_and_hold_short`，不能用 archetype id 间接表达 |
| `phase1_demo_label` baseline 缺失 | ✅ 采纳 | 作为 posthoc hindsight baseline，可用于审计但不得进入 best/guardrail/sign-off 主决策 |
| `gap_bars` 单位混乱 | ✅ 采纳 | 需要区分 bar 数与分钟数，避免 gap 剔除和仓位处理使用错误单位 |
| `Phase2Dataset.get_horizon_inputs()` row 越界 silent fallback | ✅ 采纳 | 应 fail-fast 或由 indexer 保证不越界，不能静默复用最后一行盘口 |
| `kl_demo_dominance_ratio` 未监控 | ✅ 采纳 | 已有 metric 函数，训练健康与 report 应暴露该风险 |
| env 内部属性直接访问 | ✅ 采纳 | 增加 `HorizonEnv.current_label_info()` 后由 PPOTrainer 调用，降低封装风险 |
| val 快速评估跑完整 val set | ✅ 采纳 | 大验证集下会拖慢训练，应实现 deterministic subset fast eval |
| entropy coef 退火到 0 | ✅ 采纳 | 增加下界可以降低训练后期策略塌缩风险 |
| `all_entries` 未使用 | ✅ 采纳 | 死代码，直接删除 |
| reward_alignment 重复读取 Phase I config | ✅ 采纳 | 与 cost alignment 一起收口，统一由 validator 解析并传递 |

### 10.2 部分采纳、不采纳或暂缓的变更

| Review 条目 | 决策 | 为什么不完全采纳 |
| --- | --- | --- |
| baseline record 缺少 `boundary_cost` | ❌ 不采纳 | 二次核对当前代码后，该条已不成立: `Phase2BacktestRunner._run_fixed_strategy()` 已在 record 中写入 `boundary_cost`。执行计划只补回归测试，防止未来回退 |
| Reward Normalization 未实现 running_mean_std | ⚠️ 部分采纳 | 本批次不实现 running mean/std；为避免配置 no-op 和未来统计泄漏，改为启用时 fail-fast，并在 report 中标记 unsupported |
| `dp_teacher` / `dp_teacher_offline` baseline 缺失 | ❌ 不采纳 Phase II 主流程实现 | Phase II train/val/test walk-forward 禁止在线调用 DP。DP teacher 只能作为离线 hindsight 审计，且依赖 Phase I 是否已经导出 teacher 产物 |
| `fold_seed` 未使用 | ❌ 不采纳随机 fold | rolling validation 应保持时间顺序；随机打乱会破坏 regime/time-order 审计语义。该字段仅保留给未来 block-random fold 模式 |
| Train metrics 输出不反映训练过程 | ⏸ 暂缓 | 当前 `rollout_stats` 反映训练过程，per-horizon train records 应表示最终 best selector 的可复现表现。后续只补充 report 命名，避免误读 |
| 提取共享 `RiskStateManager` | ⏸ 暂缓 | 这是较大结构重构，不阻塞 P1/P2 修复；等 rolling/resume/stress 路径稳定后单独处理 |
| frozen_policy fallback 丢失 LSTM 时序上下文 | ❌ 不采纳硬失败 | 标准 `ArchetypeDecoder` 已走 streaming path；fallback 是兼容路径且已有 warning，硬失败可能破坏非标准 decoder 调试。后续可在 report 中计数 warning |
| Periodic checkpoint (`step_*.pt`) | ✅ 采纳但降级 | 作为 C 批次工程优化，不阻塞正式训练前 P1 修复 |

---

## 11. 执行结果回写

> 执行计划: `docs/changes/20260501_phase2_review_v2_change_execution_plan.md`
> 执行日期: 2026-05-01

### 11.1 已修复问题

| 原 Review 问题 | 当前状态 | 落地说明 |
| --- | --- | --- |
| P1 Resume 逻辑未实现 | ✅ 已修复 | `Phase2Trainer.run()` 已接入 `config.resume_from`，恢复 model / optimizer / scheduler / env state / RNG，并写入 report audit |
| P1 Rolling Validation fold 仓位继承缺失 | ✅ 已修复 | rolling folds 现在按时间顺序继承上一 fold 的 `final_position`，并记录 `fold_initial_position` |
| P1 `rolling_result` 未用于 best 选择 | ✅ 已修复 | candidate best 会触发 rolling validation，并用 fold volatility / worst fold guardrail 决定是否 reject |
| P2 Early stopping 未实现 | ✅ 已修复 | eval 点支持 patience/min_delta 早停，report 写入 `early_stopping` |
| P2 Cost alignment check 未实现 | ✅ 已修复 | `Phase1ArtifactValidator` 增加 Phase I/II max_position 与 cost alignment 校验 |
| P2 Distribution shift 未调用 | ✅ 已修复 | train fit、val/test score 已接入，report 写入 distribution shift summary |
| P2 Execution stress 未调用 | ⚠️ 部分修复 | commission/slippage stress 已接入；`execution_lag_offset` 仍仅记录 scenario，尚未改变 replay 行号 |
| P2 Reward normalization 未实现 | ✅ 已处理 | 不实现 no-op；启用时 fail-fast，并在 report 标记 unsupported |
| P2 buy-and-hold baseline 语义错误 | ✅ 已修复 | 新增 action-level `buy_and_hold_long` / `buy_and_hold_short` / `always_flat` |
| P2 `phase1_demo_label` baseline 缺失 | ⚠️ 部分修复 | train/val posthoc baseline 已支持；test label 文件显式 posthoc 读取尚未实现 |
| P2 gap_bars 计算语义混乱 | ✅ 已修复 | `max_timestamp_gap_minutes` 与 `gap_bars` 分离，分钟阈值与 bar 阈值不再混用 |
| P2 row 越界 silent fallback | ✅ 已修复 | `Phase2Dataset.get_horizon_inputs()` 遇到 execution row 越界会抛错 |
| P3 env 内部属性直接访问 | ✅ 已修复 | `HorizonEnv.current_label_info()` 替代 PPOTrainer 直接读 env 内部结构 |
| P3 all_entries 未使用 | ✅ 已修复 | 死代码已删除 |
| P3 reward_alignment 重复读取 | ✅ 已修复 | validator 解析后传给 indexer/dataset/trainer |
| P3 entropy coef 退火到 0 | ✅ 已修复 | 新增 `PPOConfig.entropy_min_coef` 下界 |
| P3 val fast eval 全量跑 val | ✅ 已修复 | `evaluate_val_fast()` 支持 deterministic subset |
| P3 periodic checkpoint 缺失 | ✅ 已修复 | 新增 `checkpoint_every_updates` 与 manifest `periodic` entry |

### 11.2 保持不采纳或仍需后续处理

| 条目 | 状态 | 原因 |
| --- | --- | --- |
| baseline record 缺 `boundary_cost` | ❌ 不采纳 | 复核时已确认当前代码已有 `boundary_cost`，本轮只补了回归覆盖 |
| `dp_teacher` / `dp_teacher_offline` 主流程 baseline | ❌ 不采纳 | Phase II 主流程仍禁止在线调用 DP；只能未来做离线 hindsight 报告 |
| `fold_seed` 随机打乱 folds | ❌ 不采纳 | rolling validation 保持时间顺序，避免破坏 regime 审计 |
| 共享 `RiskStateManager` | ⏸ 暂缓 | 不阻塞本轮正确性修复，仍建议后续单独重构 |

### 11.3 测试结果

使用项目 conda 环境执行:

```bash
/home/lanceliang/miniconda3/envs/ArchetypeTrade/bin/python -m pytest tests/unit -q
```

结果: `348 passed in 3.47s`

```bash
/home/lanceliang/miniconda3/envs/ArchetypeTrade/bin/python -m pytest tests/integration/test_phase2_pipeline_smoke.py tests/integration/test_phase2_resume_checkpoint.py tests/integration/test_phase2_no_test_label_leakage.py -q
```

结果: `10 passed, 10 warnings in 0.16s`

warnings 为既有 `pytest.mark.integration` 未注册警告。
