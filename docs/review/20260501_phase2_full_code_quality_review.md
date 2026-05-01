# Phase II Archetype Selection 完整代码质量 Review 报告

> 生成日期: 2026-05-01
> Review 范围: Phase II 全部实现代码（22 个源文件，~3500 行）
> Review 依据: `docs/design/phase2_archetype_selection_design.md` + `docs/plan/phase2_archetype_selection_execution_plan.md`

---

## 1. Review 总结

| 维度 | 评级 | 说明 |
| --- | --- | --- |
| 架构对齐 | ⚠️ 部分对齐 | 主流程骨架正确，但多模块精简过度 |
| 实现完整性 | ❌ 严重不足 | 10 个必配配置项缺失，6 个关键功能未实现 |
| Bug | 🔴 存在关键 Bug | 2 个逻辑缺陷（GAE 计数、Rolling Validation） |
| 工程缺陷 | 🟡 多处 | 8 个工程实现不够严谨的点 |
| 风险 | 🔴 高风险 | 5 个可能导致训练崩溃或线上失效的风险 |

---

## 2. 实现文件清单

以下为已实现的 Phase II 代码文件:

| 模块 | 文件 | 行数 | 设计覆盖度 |
| --- | --- | --- | --- |
| 配置 | `src/config/phase2_config.py` | 366 | ~60% |
| 数据索引 | `src/data/phase2_horizon_index.py` | 296 | ~65% |
| 数据集 | `src/data/phase2_dataset.py` | 197 | ~75% |
| Label 加载 | `src/data/phase2_label_loader.py` | 163 | ~85% |
| Selector 网络 | `src/models/archetype_selector.py` | 139 | ~70% |
| 冻结策略 | `src/models/phase1_frozen_policy.py` | 267 | ~90% |
| Rollout Buffer | `src/rl/rollout_buffer.py` | 265 | ~80% |
| Actor-Critic | `src/rl/actor_critic.py` | 132 | ~85% |
| PPO Loss | `src/rl/ppo_loss.py` | 160 | ~85% |
| PPO Trainer | `src/rl/ppo_trainer.py` | 391 | ~70% |
| 调度 | `src/rl/scheduling.py` | 113 | ~60% |
| HorizoEnv | `src/trading/horizon_env.py` | 274 | ~60% |
| HorizoFactory | `src/trading/horizon_factory.py` | 128 | ~70% |
| 训练器 | `src/trainers/phase2_trainer.py` | 417 | ~60% |
| 检查点 | `src/trainers/phase2_checkpoint.py` | 163 | ~65% |
| 选择策略 | `src/trainers/phase2_selection_policy.py` | 168 | ~60% |
| 评估器 | `src/evaluation/phase2_evaluator.py` | 238 | ~55% |
| 回放 | `src/evaluation/phase2_replay.py` | 268 | ~70% |
| 指标门面 | `src/evaluation/phase2_metrics.py` | 106 | ~75% |
| 报告 | `src/evaluation/phase2_report.py` | 135 | ~70% |
| 训练入口 | `scripts/train_phase2.py` | 114 | ~60% |
| 回测入口 | `scripts/backtest_phase2.py` | 154 | ~65% |
| Selection 指标 | `src/evaluation/metrics/selection.py` | 66 | ~80% |
| Portfolio 指标 | `src/evaluation/metrics/portfolio.py` | 145 | ~80% |
| Policy Health | `src/evaluation/metrics/policy_health.py` | 87 | ~75% |
| 可视化 | `src/evaluation/diagnostics/selector_visualization.py` | 146 | ~60% |
| 失败案例 | `src/evaluation/diagnostics/phase2_failure_case_report.py` | 147 | ~75% |

---

## 3. 实现不完整项

### 3.1 配置层缺失（关键）

**Phase2Config 顶层字段缺失:**

| 缺失字段 | 设计位置 | 影响 |
| --- | --- | --- |
| `horizon` 硬编码默认 72，CLI 无法覆盖 | 设计 §4.3 | 不同分钟级数据无法使用 |
| `max_position` 未读取 Phase I config 做一致性校验 | 设计 §4.3 | 可能 Phase I/II 仓位不一致 |
| `paper_strict_reproduction` 存在字段但未调用 `apply_paper_strict_overrides()` | 设计 §4.3 | 论文严格复现开关形同虚设 |
| **`cost_alignment_check`** 完全缺失 | 设计 §4.3 | 无法校验 Phase II cost_config 与 Phase I 一致 |
| `early_stopping` 配置组整个缺失 | 设计 §4.3 | 无法启用早停 |
| `resume` 配置组整个缺失 | 设计 §4.3 | resume 能力无配置声明 |
| `deployment_ladder` 配置组整个缺失 | 设计 §4.3 | 上线阶梯审计缺失 |
| `env_shards` mode 配置缺失（contiguous/round_robin/rollover） | 设计 §4.3 | 仅支持默认 contiguous |
| `state_dim_breakdown` 计算不在配置中 | 设计 §4.3 | 维度分解无法审计 |

**HorizonScheduleConfig 缺失字段:**

| 缺失字段 | 影响 |
| --- | --- |
| `walk_forward_enabled` / `walk_forward_seed` | 无法锁定 walk-forward 模式 |
| `chunk_reset_position` (`inherit` / `flat`) | 缺少默认值，PPOTrainer 未实现区分 |
| `data_gap_check_enabled` / `max_allowed_gap_minutes` / `drop_gap_horizons` | gap 检测逻辑仅部分实现 |
| `gap_position_carry_threshold_minutes` / `gap_large_reset_mode` | gap 仓位处理缺少阈值配置 |
| `reward_alignment_lookahead_check` | 缺失 lookahead 一致性检查 |

**SelectorNetworkConfig 缺失字段:**

| 缺失字段 | 影响 |
| --- | --- |
| `action_mask_dead_codes` 开关 | dead code mask 始终启用，无法关闭 |
| `dead_code_usage_threshold` | 无法配置 dead code 阈值 |
| `input_norm` (`layer_norm` / `running_mean_std`) | 仅支持 LayerNorm |
| `position_encoding` (`one_hot_3` / `scaled_integer` / `bucketed_position`) | 仅支持 scaled 编码 |

**PPOConfig 缺失字段:**

| 缺失字段 | 影响 |
| --- | --- |
| `value_clip_range` | 无 value clipping |
| `entropy_warmup_coef` / `entropy_warmup_fraction` | 无法做 entropy warmup |
| `kl_demo_label_smoothing` | 无法做 label smoothing |
| `kl_demo_anneal_to` / `kl_demo_anneal_fraction` | ScheduleManager 中的退火是硬编码的 |
| `batch_size` | 缺少总 batch 配置 |
| `reward_normalization` | 未实现 |
| `lr_schedule` (`constant` / `linear`) | ScheduleManager 始终用 linear |

**Phase2SelectionPolicyConfig 缺失字段:**

| 缺失字段 | 影响 |
| --- | --- |
| `selection_metric` ("phase2_composite_score") 与 `metric_weights` | 当前只用单一 `primary_metric` |
| `composite_score_sensitivity_perturbations` | 无法做权重敏感性分析 |
| `risk` (Phase2RiskGuardrailConfig) 独立子配置 | 散落在顶层字段中 |
| `behavior` (Phase2BehaviorGuardrailConfig) 独立子配置 | 散落在顶层字段中 |

### 3.2 功能缺失

| 缺失功能 | 设计依据 | 严重程度 |
| --- | --- | --- |
| **Composite Score 加权计算** | 设计 §4.3 / §8.5 | 🔴 高 - 当前只用单一 `val_net_return` 选 best |
| **Composite Score Sensitivity 分析** | 设计 §8.5 | 🔴 高 - 无法检测权重敏感 |
| **Rolling Validation 真实落地** | 设计 §8.9 | 🔴 高 - 实现有严重 Bug (见 §4.1) |
| **KL/demo 消融矩阵** (`α ∈ {0, 0.1, 0.5, 1.0}`) | 设计 §5.7.1 | 🟡 中 - CLI 支持但 trainer 未编排 |
| **Execution Stress 测试** | 设计 §8.7 | 🟡 中 - 配置存在但 runner 未实现 |
| **OOD/Distribution Shift 监控** | 设计 §8.8 | 🟡 中 - 配置存在但未在评估中调用 |
| **Online Action Throttle** | 设计 §8.3 备注 | 🟡 中 - 配置存在但未在推理中应用 |
| **Live Risk Control 完整路径** (mid-horizon flatten) | 设计 §8.10 后 | 🟡 中 - HorizonEnv 有基础实现但 backtest 未启用 |
| **Periodic Checkpoint** (`checkpoints/step_*.pt`) | 设计 §10 | 🟢 低 |
| **Train Metrics 输出** (用于 train/val gap 诊断) | 设计 §4.13 | 🟡 中 |
| **Unmasked Diagnostic Rollout** (训练初期探测 dead code) | 设计 §6.5 | 🟡 中 |
| **kl_demo_dominance_ratio 监控与告警** | 设计 §5.7.1 | 🟡 中 |
| **Deployment Readiness** (shadow/paper/canary) | 设计 §8.10 | 🟢 低 |

### 3.3 horizon_index 产物字段缺失

设计 §3.3 要求 `phase2_horizon_index_*.feather` 包含 13 个字段，实际实现仅包含:

| 设计字段 | 是否实现 |
| --- | --- |
| `sample_id` | ✅ |
| `start_index` (叫 `horizon_start`) | ✅ |
| `end_index` (叫 `horizon_end`) | ✅ |
| `last_execution_row` | ❌ |
| `last_markout_row` | ❌ |
| `has_data_gap` (叫 `is_gap`) | ✅ |
| `max_timestamp_gap_minutes` (叫 `gap_bars`) | ⚠️ 语义不同 |
| `phase1_sample_id` | ❌ |
| `code_label` | ✅ (train/val only) |
| `is_labeled` | ✅ |
| `prev_terminal_position` | ❌ |
| `split` | ✅ |
| `timestamp_start` | ❌ |

### 3.4 phase2_report.json 字段缺失

设计 §10 要求 `phase2_report.json` 包含 ~60 个字段，当前实现的 report summary 仅有约 20 个字段。缺失的关键字段:

| 缺失字段组 | 具体字段 |
| --- | --- |
| Horizon 调度审计 | `horizon_schedule.mode/stride/position_continuity/dense_overlap_ablation/chunk_reset_position` |
| 数据间隙审计 | `data_gap_filter` (gap 数量、比例、最大 gap) |
| 输入规范 | `input_norm` (模式、编码方式、state_dim_breakdown、stats_hash) |
| 多 Env 审计 | `env_shards` (分片数、horizon 分布、reward 分布警告) |
| Reward Scaling | `reward_scaling` (方法、clip_range、raw/scaled reward 分布) |
| Reward Normalization | `reward_normalization` (enabled、stats 冻结状态) |
| 成本一致性 | `cost_config_inherited` |
| Baseline 审计 | `baselines_val/baselines_test` (random/single_archetype/buy_and_hold) |
| 训练协议 | `kl_demo_signal_type`、`kl_demo_dominance_ratio`、`input_norm_stats_merge_protocol` |
| 部署审计 | `rolling_validation_summary`、`execution_stress_summary`、`distribution_shift_warning_count` |
| 恢复审计 | `resume_ready`、`hypothetical_early_stop_timestep` |
| Guardrail | `guardrails_pass` / `val_guardrails_pass` / `test_guardrails_pass_report_only` |
| 边界诊断 | `chunk_reset_distribution_shift` |

---

## 4. Bug 清单

### 4.1 🔴 Rolling Validation 实现错误（严重）

**位置:** [phase2_evaluator.py:L164-L179](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/evaluation/phase2_evaluator.py#L164-L179)

**问题描述:** 每个 fold 调用 `self.backtest_runner.run_walk_forward(split="val", deterministic=True)` 运行**完整 val set** 的 walk-forward，然后切片 `records[start:end]` 取对应 fold。

**为什么是 Bug:**
- 每个 fold 都重复跑了一遍完整的 val walk-forward，徒增计算量
- 更关键的是: 每个 fold 的 `prev_terminal_position` 都是从 0 开始的完整继承链，而不是从该 fold 的起始仓位开始，导致仓位继承序列与实际 fold 边界不一致
- 所有 fold 的 walk-forward 结果完全一样（除了切片区间），rolling validation 退化为"把同一个结果拆成 N 份"，无法检测不同市场阶段的泛化性能差异

**预期行为:** 每个 fold 应当只在 fold 对应的 horizon 子集上独立执行 walk-forward，仓位从 fold 起点继承。

**修复建议:** 为 `run_walk_forward` 增加 `entry_subset` 参数，或在 `Phase2BacktestRunner` 上创建 fold 专用的 runner 实例。

### 4.2 🔴 HorizonEnv 没有实现 truncated 语义（严重）

**位置:** [horizon_env.py:L105-L219](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trading/horizon_env.py#L105-L219)

**问题描述:** `step()` 方法的 `truncated` 返回值始终为 `False`（第 191 行: `truncated = False`），HorizonEnv 无法表达 "rollout buffer 已满但 episode 未结束" 的状态。

**影响:**
- `PPOTrainer.collect_rollout()` 中当 `done=False` 时（即单纯因 `rollout_length` 到达而截断），没有将 `truncated=True` 写入 RolloutSample
- `RolloutBuffer.compute_gae()` 中 `truncated` 始终为 `False`，导致所有非 done 的 buffer 截断被当作正常的 episode 中间步处理
- 这可能导致 GAE 在 buffer 截断处的 bootstrap 行为异常（虽然代码中有 `last_values` bootstrap 机制，但 truncated 语义的缺失使得区分"buffer 截断"和"episode 中途"无法实现）

**修复建议:**
1. `HorizonEnv` 需要接收一个 `max_steps` 参数或由外部调用方显式通知 truncated 状态
2. 或者在 `PPOTrainer.collect_rollout()` 中，当 `rollout_length` 达到但 `done=False` 时，显式设置 `RolloutSample.truncated=True`

### 4.3 🟡 PPOTrainer 访问已更新状态的潜在错误

**位置:** [ppo_trainer.py:L141-L150](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/rl/ppo_trainer.py#L141-L150)

**问题描述:** 在 `collect_rollout()` 循环内，先调用 `env.step(action)` 改变了 `env.cursor`，但 `step` 之前已经读出 `current_horizon_idx`。这本身是正确的。但 `env.step()` 返回的 `next_obs` 可能来自下一个 horizon，而 `self._current_obs[env_idx] = next_obs` 的赋值发生在后面的 `if done` 分支。由于 `done` 的判断依赖于 cursor 是否到达末尾，这条取 obs 的路径在逻辑上自洽，但不够直观，容易在后续改动中引入 bug。

**风险:** 低。当前逻辑正确，但代码可读性差。

### 4.4 🟢 buy_and_hold baseline 语义错误（轻微）

**位置:** [phase2_replay.py:L214-L216](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/evaluation/phase2_replay.py#L214-L216)

**问题描述:** `buy_and_hold` baseline 的实现是固定选择 archetype code 0（一直选 flat 的 archetype）加上 `strategy_fn=lambda: 1`。但实际上 `strategy_fn` 返回的是 **archetype id**（传给 `frozen_policy.reset(code_id)`），不是 TradingEnv 的 action。注释写 `# flat` 是误导的。

buy_and_hold 的预期行为应该是一直选择"能产生持续 long"的 archetype，而不是一直选 code_id=1 的 archetype 然后让 decoder 自行决定每一步动作。当前实现实际上是"single_archetype_0" baseline，不是真正的 buy_and_hold。

**修复建议:** 需要基于每个 archetype 在全部 horizon 上的平均 position 行为来判断哪个 archetype 最接近 buy_and_hold，或者独立实现一个不使用 decoder 的纯 buy_and_hold baseline。

---

## 5. 工程实现缺陷

### 5.1 `HorizonEnv._cursor` 和 `_prev_terminal_position` 无属性封装

**位置:** [horizon_env.py:L78-L83](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trading/horizon_env.py#L78-L83)

`reset()` 将 `_prev_terminal_position` 强制重置为 0（第 92 行），违背设计 §3.3 中 "reset() 注入 prev_terminal_position" 的要求。设计明确要求:

> `reset(prev_terminal_position)` 必须注入上一 horizon 末仓位。

当前实现 `reset()` 不接受参数，总是从 0 开始。这导致:
- 多 env 训练时每个 env 从 flat 开始，跨 env 分片边界的仓位继承丢失
- `PPOTrainer.get_state()` 保存 `env._prev_terminal_position`，但 `load_state()` 恢复后，下一次 `reset()` 会直接覆盖为 0

**修复建议:** `reset()` 改为 `reset(prev_terminal_position: int = 0)`，允许外部注入仓位。

### 5.2 `PPOTrainer` 直接访问私有属性

**位置:** [ppo_trainer.py:L390](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/rl/ppo_trainer.py#L390)

```python
env._cursor = es.get("cursor", 0)
env._prev_terminal_position = es.get("prev_terminal_position", 0)
```

直接修改私有属性，绕过所有 setter 逻辑，极易因后续重构引入 bug。

**修复建议:** 提供公开的 `restore_state(cursor, position)` 方法。

### 5.3 `Phase2Dataset.__init__` 无防御性输入校验

**位置:** [phase2_dataset.py:L55-L109](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/data/phase2_dataset.py#L55-L109)

- 不校验 `frame.feature_columns` 与 `input_schema.json` 是否一致（设计明确要求）
- 不校验 `frame.timestamp` 是否单调递增
- 不校验 `horizon_entries` 的 split 是否与 frame 匹配

### 5.4 `Phase2Trainer.run()` 行数过多，编排逻辑耦合

**位置:** [phase2_trainer.py:L96-L404](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase2_trainer.py#L96-L404)

单个方法 ~310 行，包含数据加载、索引生成、模型构建、训练循环、评估、报告等全部逻辑。违反单一职责原则，难以单独测试各个环节。

### 5.5 `ScheduleManager` 的 KL demo coef 退火逻辑硬编码

**位置:** [scheduling.py:L79](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/rl/scheduling.py#L79)

```python
self._current_kl_demo_coef = self._initial_kl_demo_coef * (1.0 - progress * 0.5)
```

退火终点被硬编码为 `initial * 0.5`（而不是 `initial * (1 - 0.5) = initial * 0.5`）。这实际上退火到 0.5x 初始值。但设计允许配置 `kl_demo_anneal_to` 指定具体终值。当前 ScheduleManager 完全忽略 PPOConfig 中应存在的 `kl_demo_anneal_to`。

### 5.6 `reward_scaling.clip_range` 完全未使用

**位置:** [ppo_trainer.py:L182-L187](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/rl/ppo_trainer.py#L182-L187)

`_scale_reward()` 仅支持 `divide_by_horizon` 方法，完全不读取 `clip_range` 配置。设计要求 "若启用 clip，必须同时记录 clipped/unclipped reward 统计"——此功能完全缺失。

### 5.7 `PPOTrainer.update()` 缺少 `explained_variance` 计算

**位置:** [ppo_trainer.py:L189-L322](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/rl/ppo_trainer.py#L189-L322)

设计 §8.4.4 要求 PPO 训练健康指标必须包含 `explained_variance`。虽然 `policy_health.py` 中有 `compute_explained_variance` 函数，但 `PPOTrainer.update()` 未调用它，`PPOUpdateStats.explained_variance` 始终为 0。

### 5.8 死代码 Mask 阈值硬编码为 0

**位置:** [phase2_trainer.py:L208-L210](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trainers/phase2_trainer.py#L208-L210)

```python
dead_code_mask = torch.tensor([c == 0 for c in counts], dtype=torch.bool)
```

将 usage count === 0 的 code 标记为 dead。设计要求 "阈值基于 Phase I global usage ratio < `dead_code_usage_threshold` (默认 0.01)"，即 usage ratio 低于 1% 才标记为 dead，而非完全未使用。当前实现过于宽松——只有从未使用过的 code 才被屏蔽。

---

## 6. 风险点

### 6.1 🔴 多 Env 训练中的仓位继承断裂

**风险描述:** `HorizonEnv.reset()` 总是从 `position=0` 开始（[horizon_env.py:L92](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/trading/horizon_env.py#L92)）。当训练数据按时间分片后，每个 env 从各自分片起点开始，仓位继承链在分片边界断裂。

**触发条件:** `position_continuity=true` 且 `num_envs > 1`。

**后果:**
- 每个 env 的训练 MDP 从 flat 开始，与实际 walk-forward（仓位跨分片连续继承）不一致
- selector 学到的策略可能在 walk-forward backtest 时表现差于训练期

### 6.2 🔴 PPO rollout 中 truncated 语义缺失导致跨 Rollout GAE 错误

**风险描述:** 当 `rollout buffer` 截断但 episode 未真正结束时（`truncated=True, done=False`），当前的 `RolloutBuffer.compute_gae()` 能正确处理（通过 `last_values` bootstrap），但因为 `truncated` 始终为 `False`，`done` 的分片末端（cursor 到达末尾）被当作 episode done 处理。如果 `done` 设置时机不当（分片末端总是 done=True），则跨分片的 GAE bootstrap 被切断，可能导致 advantage 估计偏差。

### 6.3 🟡 冻结 Decoder 的 streaming decode fallback 路径丢失时序上下文

**位置:** [phase1_frozen_policy.py:L184-L199](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/models/phase1_frozen_policy.py#L184-L199)

**风险描述:** 当 decoder 缺少 `state_proj/lstm/head` 子模块时，fallback 路径调 `self.decoder(state_t_seq, self._z_q)` 做单步推理，但 `self._recurrent_state` 被设为 `new_state = self._recurrent_state`（即不更新，保持上一次的值或 None）。这意味着 fallback 路径下每一步都是独立推理，没有 LSTM 时序上下文。

虽然第一次调用会发 warning，但如果 decoder 恰好只有那些子模块名称不匹配但实际结构正确的场景（例如自定义 decoder），warning 可能被忽略。

### 6.4 🟡 `PPOConfig.target_kl` 默认值 0.03 过于激进

**位置:** [phase2_config.py:L96](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/config/phase2_config.py#L96)

设计文档指定 `target_kl: Optional[float] = 0.05`，但实现默认 `0.03`。更低的 KL 阈值会导致 KL early stop 更频繁触发，PPO update 的实际 epoch 数可能远小于配置的 `update_epochs=4`，训练效率下降。

### 6.5 🟡 `schedule.py` 让 entropy 退火到 0，可能导致后期策略塌缩

**位置:** [scheduling.py:L76](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/src/rl/scheduling.py#L76)

```python
self._current_entropy_coef = self._initial_entropy_coef * (1.0 - progress)
```

Entropy coef 线性退火到 0。当接近训练结束时，entropy bonus 完全消失，加上 KL/demo regularization，selector 可能迅速塌缩到单一 action。虽然 `Phase2SelectionPolicy` 有 `max_action_dominance_ratio` guardrail，但 guardrail 只在 checkpoint 选择时生效，不阻止训练后期的策略退化。

### 6.6 🟡 test backtest 脚本中 dead_code_mask 被写死为全 False

**位置:** [backtest_phase2.py:L138](file:///home/lanceliang/opt/aiwork/lance/ArchetypeTrader/scripts/backtest_phase2.py#L138)

```python
metrics = phase2_composite_metrics(rec_dicts, {}, ... , [False] * frozen_policy.num_codes)
```

Backtest 脚本不加载 Phase I 的 `code_usage` 信息来构造 dead code mask，因此不会做 dead code selection 检查。这可能会导致报告遗漏 dead code usage 信息。

---

## 7. 高风险项优先级矩阵

| 优先级 | 问题 | 类别 | 影响 |
| --- | --- | --- | --- |
| P0 | Rolling Validation 实现错误 | Bug §4.1 | sign-off 的 rolling validation 完全无效 |
| P0 | HorizonEnv 没有 truncated 语义 | Bug §4.2 | GAE 在 buffer 边界异常 |
| P0 | 多 Env 仓位继承断裂 | 风险 §6.1 | 训练/回测不一致 |
| P0 | Composite Score 缺失（只用单一指标） | 缺失 §3.2 | checkpoint 选择脆弱 |
| P1 | `HorizonEnv.reset()` 不接受 prev_terminal_position | 缺陷 §5.1 | 仓位继承链断裂 |
| P1 | dead code mask 阈值硬编码为 0 | 缺陷 §5.8 | 可能漏掉低使用率但有风险的 code |
| P1 | phase2_report.json 缺失大量审计字段 | 缺失 §3.4 | sign-off 审计不完整 |
| P2 | Config 层缺失 10+ 必配字段 | 缺失 §3.1 | 实验可复现性降低 |
| P2 | reward_scaling.clip_range 未实现 | 缺陷 §5.6 | 设计功能缺失 |
| P2 | explained_variance 未计算 | 缺陷 §5.7 | PPO 健康监控缺失 |
| P3 | ScheduleManager KL 退火硬编码 | 缺陷 §5.5 | 可配置性降低 |
| P3 | Train metrics 未输出 | 缺失 §3.2 | train/val gap 无法诊断 |

---

## 8. 架构设计问题

### 8.1 Phase2Dataset 承担了过多职责

当前 `Phase2Dataset.__init__` 不仅存储数据，还自行解析 reward_alignment、预提取 numpy 数组、构建 mark price。这使得 `Phase2Dataset` 与 Phase I 的 `RewardAlignment` 紧耦合。设计 §4.4 明确要求 "phase2_dataset.py 不调用 decoder/selector，只负责把 raw feather 切成张量"。

### 8.2 Phase2HorizonIndexer 和 Phase1ArtifactValidator 不应在同一文件中

当前 `phase2_horizon_index.py` 同时包含 `Phase1ArtifactValidator`（Phase I 产物校验）和 `Phase2HorizonIndexer`（horizon index 生成）。这两个职责差异大，应拆分到独立模块。

### 8.3 PPOTrainer.state_dict 保存的是所有 env 的状态，但不包含 frozen_policy 的 recurrent state

设计 §4.11 要求 checkpoint 保存 "decoder_recurrent_state"，但 `PPOTrainer.get_state()` 只保存 `env._cursor` 和 `env._prev_terminal_position`，不保存 `frozen_policy._recurrent_state`。

---

## 9. 建议修复优先级

### 第一优先级（阻塞 sign-off）:
1. 修复 Rolling Validation Bug（evaluator §4.1）
2. 实现 truncated 语义（HorizonEnv §4.2 + PPOTrainer）
3. 实现 Composite Score 加权计算
4. 修复 `HorizonEnv.reset()` 接受 prev_terminal_position

### 第二优先级（阻塞正式部署）:
1. 补齐 `phase2_report.json` 审计字段
2. 实现 Execution Stress 测试
3. 实现 KL/demo 消融矩阵自动编排
4. 校准 dead code mask 阈值逻辑

### 第三优先级（工程优化）:
1. 补全配置层缺失字段
2. 添加 explained_variance 计算
3. 优化 Phase2Trainer.run() 的代码结构
4. 增加 `ScheduleManager` 的可配置退火终点

---

## 10. 总结

Phase II 实现代码的主流程骨架（加载 Phase I 产物 → 生成 horizon index → 构造 HorizonEnv → PPO 训练 → 评估 → 报告）是**正确和完整的**。核心的 PPO 训练、GAE 计算、streaming decode 等算法实现也基本正确。

但存在以下**关键问题**:

1. **Rolling Validation 实现有严重 Bug**，导致 rolling validation 形同虚设
2. **HorizonEnv 的 truncated 语义完全缺失**，影响 GAE bootstrap 准确性
3. **checkpoint 选择仅依赖单一指标**，缺少设计中的 Composite Score 加权和 sensitivity 分析
4. **多 Env 仓位继承链断裂**，训练和回测的行为不一致
5. **配置层和报告层大量字段缺失**，影响实验可复现性和审计完整性

建议在继续 Phase III 开发之前，至少修复 P0 和 P1 级别的 7 个问题。
