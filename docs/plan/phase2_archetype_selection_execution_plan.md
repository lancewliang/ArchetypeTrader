# Phase II Archetype Selection 可执行代码生成计划

本文档根据 `docs/design/phase2_archetype_selection_design_v3.md` 生成可执行实施计划，并参考 `docs/plan/phase1_archetype_discovery_execution_plan.md` 的格式与颗粒度。目标不是在本计划中粘贴生产代码，而是明确后续代码生成、审查、测试数据、单元测试、集成测试和验收命令的执行顺序。

## 1. 实施目标

Phase II 需要交付一条可运行的 horizon-level RL 训练与回测链路:

```text
加载 Phase I 冻结产物(decoder/codebook/labels/schema/report)
  -> 校验 Phase I sign-off 与 Phase II/Phase I 数据契约
  -> 生成 phase2_horizon_index_{train,val,test}
  -> join train/val KL/demo labels
  -> 构造 HorizonEnv 与 multi-env 时间分片
  -> 初始化 selector + PPO trainer + evaluator + checkpoint manager
  -> rollout / GAE / PPO update
  -> val 快速评估 + val walk-forward + checkpoint selection
  -> composite score sensitivity
  -> train/val/test walk-forward backtest
  -> report / manifest / diagnostics / best selector 导出
```

实施边界:

- 数据处理统一使用 `polars`，不使用 `pandas`。
- 输入与输出表格统一使用 Feather/Arrow IPC 格式，生产代码用 `polars.read_ipc` / `DataFrame.write_ipc` 读写 `.feather` 文件。
- Phase II 不重新训练 Phase I 的 encoder/decoder/codebook，不在线调用 DP。
- Phase II 的 horizon-level reward、成本、换仓和盘口成交语义必须完全复用 `src/trading/`，禁止另写一套 reward/cost 逻辑。
- selector 训练使用 `torch`；PPO 为自研实现，不依赖 Stable-Baselines3。
- 测试框架使用 `pytest`。
- test split 的 `code_label` 只能用于 posthoc baseline，对 Phase II 的训练、checkpoint 选择、早停和主回测决策路径一律不可见。
- HorizonEnv 正式 replay 只能使用 `Phase1FrozenPolicy.decode_step()` 的 streaming 因果接口，不允许用批量 `decode()` 替代。

## 2. 依赖安装计划

建议在沿用 Phase I 依赖的基础上，补齐 Phase II 训练、回测、可视化与数值诊断依赖。

必需依赖:

```text
polars>=0.20.0
pyarrow>=14.0.0
numpy>=1.24.0
torch>=2.2.0
tqdm>=4.64.0
PyYAML>=6.0.0
pydantic>=2.0.0
pytest>=8.0.0
pytest-cov>=5.0.0
```

建议依赖:

```text
matplotlib>=3.8.0      # selector_visualization / equity curve / diagnostics
scikit-learn>=1.4.0    # PSI / simple OOD diagnostics / optional calibration tools
tensorboard>=2.15.0    # PPO health / loss / entropy / KL monitoring
jinja2>=3.1.0          # failure case HTML 报告，第一版可暂缓
```

安装与验证命令:

```bash
python3 -m pip install -r requirements.txt
python3 - <<'PY'
import polars, pyarrow, numpy, torch, pytest, yaml, pydantic, matplotlib
print("phase2 dependencies ok")
PY
```

## 3. 代码生成顺序

按以下顺序生成代码，每一步完成后运行对应测试，避免一次性生成大面积不可定位的问题。

Step 依赖关系建议按下图执行，避免在上游契约未稳定时过早实现下游训练器:

```text
Step 1 ──► Step 2 ──► Step 3 ──► Step 4 ──► Step 5 ──► Step 6 ──► Step 7 ──► Step 9 ──► Step 10
                              │                         ▲
                              └────────► Step 8 ────────┘
```

其中 Step 8 的 live safety / OOD / stress 能力可以在 Step 7 的 report / diagnostics 框架稳定后并行补齐，但最终仍需在 Step 9 与 Step 10 中完成集成验收。

### 3.0 执行状态看板

后续每完成一个 Step，就在本表中把对应单元从 `[ ]` 改为 `[x]`，并在“备注”中记录关键产物、测试命令或阻塞原因。建议只在该 Step 的代码、测试、审查都完成后，才把“状态”改为 `DONE`。

状态约定:

| 状态 | 含义 |
| --- | --- |
| `TODO` | 尚未开始 |
| `IN_PROGRESS` | 正在实现或测试 |
| `BLOCKED` | 被依赖、数据、接口或决策阻塞 |
| `DONE` | 代码、测试、审查和产物验收均完成 |

| Step | 范围 | 代码 | 单元测试 | 集成/验收 | 代码审查 | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Step 1 | 配置、CLI 与基础产物 IO | [x] | [x] | [x] | [x] | DONE | `src/config/phase2_config.py`、`scripts/train_phase2.py`、`scripts/backtest_phase2.py` 已落地；两个 CLI 的 `--help` 验收通过。 |
| Step 2 | Phase I 产物校验、数据契约与 horizon index | [x] | [x] | [x] | [x] | DONE | `Phase1ArtifactValidator`、`Phase2HorizonIndexer`、`Phase2Dataset`、`Phase2LabelLoader` 已落地；数据层单测为真实断言并通过。 |
| Step 3 | Frozen decoder、HorizonEnv、multi-env shard 与 replay 语义 | [x] | [ ] | [ ] | [x] | IN_PROGRESS | 代码主干已落地；`test_phase1_frozen_policy.py`、`test_horizon_env.py`、`test_horizon_factory.py` 与 streaming decode 集成测试仍为 `pass` 占位，尚不能视为单元/集成验收完成。 |
| Step 4 | Selector 网络、actor-critic、action mask 与输入规范 | [x] | [ ] | [ ] | [x] | IN_PROGRESS | selector / actor-critic 主路径有真实单测；unmasked diagnostic rollout 与 `probe_pick_rate` 未见实现和断言，输入规范验收未收口。 |
| Step 5 | Rollout buffer、GAE、PPO loss 与 schedule | [x] | [ ] | [ ] | [x] | IN_PROGRESS | rollout buffer、PPO loss、schedule 有真实单测；`test_ppo_trainer.py` 关键用例和 GAE 集成测试仍为占位，GAE/PPO 端到端验收未完成。 |
| Step 6 | Evaluator、baseline、checkpoint、selection policy | [x] | [ ] | [ ] | [x] | IN_PROGRESS | checkpoint / selection policy 部分有真实单测；`phase2_replay`、`phase2_evaluator` 测试仍为占位，`phase2_ablation_summary.csv` 未见产出，test split `phase1_demo_label` posthoc baseline 仍是残余项。 |
| Step 7 | Report、diagnostics、visualization 与 risk/health audit | [x] | [ ] | [ ] | [x] | IN_PROGRESS | report writer 有真实单测；selector visualization 与 Phase II failure case report 单测仍为 `pass` 占位，诊断产物验收未完成。 |
| Step 8 | Live safety / OOD / latency / stress protocol | [x] | [ ] | [ ] | [x] | IN_PROGRESS | OOD / stress / throttle / numerical safety 模块已落地；execution lag scenario 已写入 report 但尚未改变 replay 行号，live risk / throttle / numerical safety 多个测试仍为占位。 |
| Step 9 | Phase2Trainer 与 train/backtest 集成入口 | [x] | [ ] | [ ] | [x] | IN_PROGRESS | trainer、resume、train/backtest 入口主干已落地；`test_phase2_trainer.py` 和真实 smoke pipeline 仍为占位，尚未完成可执行集成验收。 |
| Step 10 | 完整 smoke pipeline、resume、sign-off 验收 | [x] | [ ] | [ ] | [x] | IN_PROGRESS | `run_pipeline.sh` 语法检查通过；Phase II pipeline / resume / backtest / no-test-label-leakage / streaming decode 集成测试均仍为 `pass` 占位，sign-off 验收未完成。 |

Step 完成记录:

| 日期 | Step | 完成内容 | 验证命令 | 结果 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 2026-05-01 | Step 1-10 | 重新核对 Phase II 源码、单测、集成测试、两轮代码审查与 review 变更执行计划；按真实验收状态更新看板。 | `/home/lanceliang/miniconda3/envs/ArchetypeTrade/bin/python scripts/train_phase2.py --help`；`/home/lanceliang/miniconda3/envs/ArchetypeTrade/bin/python scripts/backtest_phase2.py --help`；`/home/lanceliang/miniconda3/envs/ArchetypeTrade/bin/python -m pytest -q`；`bash -n run_pipeline.sh` | CLI 和脚本语法检查通过；全量测试 `375 passed, 17 warnings`。 | Phase II 专项测试为 `183 passed, 17 warnings`，但多项 Phase II 单元/集成测试仍是 `pass` 占位，不能作为最终验收依据；Step 3-10 继续保持 `IN_PROGRESS`。 |

### Step 1: 配置、CLI 与基础产物 IO

生成文件:

```text
src/config/phase2_config.py
scripts/train_phase2.py
scripts/backtest_phase2.py
```

核心内容:

- `Phase2Config` 使用 frozen dataclass + 显式嵌套类型映射，镜像 Phase I 配置风格。
- `rolling_validation` 不是仅保留配置字段；必须在执行层落地为可调用 runner、固定 fold 切法、固定产物与固定验收逻辑。
- 配置组至少包含:
  - `phase1_artifacts`
  - `horizon_schedule`
  - `selector_network`
  - `ppo`
  - `selection_policy`
  - `reward_scaling`
  - `live_risk_controls`
  - `distribution_shift`
  - `execution_stress`
  - `rolling_validation`
  - `online_action_throttle`
  - `numerical_safety`
- CLI 参数至少包含:
  - `--pair`
  - `--phase1-batch-id`
  - `--phase2-batch-id`
  - `--train-file`
  - `--val-file`
  - `--test-file`
  - `--total-timesteps`
  - `--num-envs`
  - `--rollout-length`
  - `--seed`
  - `--allow-phase1-hindsight-warning`
  - `--paper-strict-reproduction`
  - `--resume-from`
- 输出目录固定为 `artifacts/{PAIR}/{PHASE2_BATCH_ID}/phase2/`。
- 入口写出 `phase2_config.yaml`。
- `scripts/backtest_phase2.py` 明确只加载 best selector + frozen Phase I 产物，不加载 test label 参与决策路径。

验收:

```bash
python scripts/train_phase2.py --help
python scripts/backtest_phase2.py --help
```

### Step 2: Phase I 产物校验、数据契约与 horizon index

生成文件:

```text
src/data/phase2_horizon_index.py
src/data/phase2_dataset.py
src/data/phase2_label_loader.py
```

核心内容:

- `Phase1ArtifactValidator`（可放在 `phase2_horizon_index.py` 或独立私有模块）负责:
  - 校验 `decoder.pt`、`codebook.pt`、`input_schema.json`、`phase1_report.json`、`phase1_config.yaml`、`feature_provenance.json`、`checkpoint_manifest.json` 是否齐全。
  - 校验 `fatal_collapse=false`、`code_assignment_drift_warning=false`。
  - 若 `hindsight_bias_warning=exceeded`，默认拒绝启动，除非显式传入 `--allow-phase1-hindsight-warning`。
  - 校验 `cost_config`、`reward_alignment`、`max_position` 与 Phase I 一致。
- `Phase2HorizonIndexer` 生成 `phase2_horizon_index_{train,val,test}.feather`:
  - 支持 `non_overlap`、`stride`、`phase1_index`。
  - 末尾 markout 越界 horizon 必须裁掉。
  - gap horizon 必须标注，并按配置裁掉。
  - test 索引默认不含 `code_label`。
- `Phase2LabelLoader` 仅 join train/val 的 `code_label` 与 `is_labeled`，并输出 `kl_label_temporal_coverage` 的原始统计。
- `phase2_dataset.py` 只负责将 horizon index + 原始 market frame 适配为 state 读取接口，不重写 `HorizonBuilder` 逻辑；若需要 `HorizonInputs`，必须函数级复用 Phase I 的切片协议。

验收:

```bash
pytest tests/unit/data/test_phase2_horizon_index.py
pytest tests/unit/data/test_phase2_dataset.py
pytest tests/unit/data/test_phase2_label_loader.py
```

### Step 3: Frozen decoder、HorizonEnv、multi-env shard 与 replay 语义

生成文件:

```text
src/models/phase1_frozen_policy.py
src/trading/horizon_env.py
src/trading/horizon_factory.py
```

核心内容:

- `Phase1FrozenPolicy`:
  - 只加载 `decoder.pt` + `codebook.pt`。
  - 参数全部 `requires_grad=False`。
  - 正式 replay 主接口为 `decode_step()`。
  - 批量 `decode()` 仅允许诊断或离线对比，不允许 HorizonEnv 主路径调用。
  - 自检 decoder 为单向因果结构，禁止双向 LSTM、双向 attention 或全 horizon pooling。
- `HorizonEnv`:
  - `reset()` 返回当前 horizon 的 `s^sel`。
  - `step(action)` 执行一个完整 horizon:
    - 维护 `cursor / prev_terminal_position / recurrent_state`。
    - 使用 `decode_step()` 循环 `h` 次生成 `base_actions`。
    - 调用 `TradingEnv.reset(initial_position=prev_terminal_position)`。
    - 执行 step-wise replay，累加 `r^sel`。
    - 返回 `(next_obs, reward, done, truncated, info)`。
  - `done=True` 仅在时间分片末端、split 末端，或显式 ablation 的独立 horizon episode 终止时出现。
  - `truncated=True` 仅在 `rollout_length` 到达、PPO buffer 需要截断并 bootstrap 时出现；此时必须 `done=False`。
  - gap 裁切后的仓位处理必须显式支持 `carry / force_flatten / warmup_only` 三种模式，禁止静默 flat reset。
  - 支持 `mid-horizon emergency flatten`，以服务 live risk control。
- `HorizonFactory`:
  - 负责按 `num_envs` 生成时间分片、实例化多 env。
  - 输出 `phase2_env_shards.feather`，包含边界位置、horizon 数、时间区间及 regime 摘要。
- `Phase2BacktestRunner`（可放在 `src/evaluation/phase2_replay.py`）必须实现 `_guard_no_test_label_in_decision_path()`：
  - 在每次 selector 决策前后检查 test split 上是否有 `code_label` 被加载、透传或消费。
  - 一旦发现 `code_label` 进入决策路径，立即抛出 `Phase2TestLabelLeakageError`。

验收:

```bash
pytest tests/unit/models/test_phase1_frozen_policy.py
pytest tests/unit/trading/test_horizon_env.py
pytest tests/unit/trading/test_horizon_factory.py
```

### Step 4: Selector 网络、actor-critic、action mask 与输入规范

生成文件:

```text
src/models/archetype_selector.py
src/rl/actor_critic.py
```

核心内容:

- `ArchetypeSelector`:
  - 输入 `s^sel`，输出 `K` 类离散 logits 与 critic value。
  - 支持 `deterministic=False/True` 两种推理模式。
  - 默认 `LayerNorm`，`RunningMeanStd` 只作为 ablation。
  - `position_continuity=true` 时强制要求状态包含 `prev_terminal_position` 编码。
- `ActorCritic`:
  - `act(obs, deterministic=False)` 返回 `action / log_prob / value`。
  - `evaluate_actions(obs, action)` 返回 `log_prob / entropy / value`。
- action mask:
  - dead code mask 基于 Phase I global usage，而不是 Phase II subset。
  - mask 作用于 logits = `-inf`。
  - 若 `kl_label` 指向 masked code，KL term 该样本置零。
  - 训练初期必须支持一次 `unmasked diagnostic rollout`，暂时关闭 dead code mask，记录 selector 是否会自发选择被 Phase I 标记为 dead 的 code，以及 `probe_pick_rate`。
- 状态维度校验:
  - `state_dim_breakdown` 必须与配置与 report 一致。
  - 多档仓位时禁止 `one_hot_3`。

验收:

```bash
pytest tests/unit/models/test_archetype_selector.py
pytest tests/unit/rl/test_actor_critic.py
```

### Step 5: Rollout buffer、GAE、PPO loss 与 schedule

生成文件:

```text
src/rl/rollout_buffer.py
src/rl/ppo_loss.py
src/rl/ppo_trainer.py
src/rl/scheduling.py
```

核心内容:

- `RolloutBuffer` 保存:
  - `obs / env_id / action / log_prob / value / reward / done / truncated`
  - `kl_label / is_labeled / dead_code_mask`
  - `info_cost_paid / info_boundary_cost / info_chosen_code`
- GAE:
  - 必须按 `env_id` 分组计算，不跨 env 混算。
  - `done=True` 才切断 bootstrap；`truncated=True` 只表示 buffer 截断。
- PPO loss:
  - `policy_clip`
  - `value_loss`
  - `entropy_bonus`
  - `kl_demo_loss`
  - `approx_kl` early stop
- reward scaling:
  - 默认 `divide_by_horizon`。
  - 默认 `clip_range=null`。
  - 若启用 clip，必须同时记录 clipped/unclipped reward 统计。
- `RunningMeanStd` 若作为 ablation 启用，必须满足:
  - 明确 `running_mean_std_mode`（`per_env_only` 或 `delayed_merge_next_rollout`）。
  - 当前 rollout 期间禁止消费同 rollout 内新合并得到的统计量。
  - `input_norm_stats_merge_protocol`、`running_mean_std_signoff_allowed=false` 必须写入 report。
- schedule:
  - 支持 learning rate、entropy coef、kl_demo coef 的退火。
  - 初期可选 higher entropy warmup。

验收:

```bash
pytest tests/unit/rl/test_rollout_buffer.py
pytest tests/unit/rl/test_ppo_loss.py
pytest tests/unit/rl/test_ppo_trainer.py
pytest tests/unit/rl/test_scheduling.py
```

### Step 6: Evaluator、baseline、checkpoint、selection policy

生成文件:

```text
src/trainers/phase2_checkpoint.py
src/trainers/phase2_selection_policy.py
src/evaluation/phase2_metrics.py
src/evaluation/phase2_replay.py
src/evaluation/phase2_evaluator.py
src/evaluation/metrics/selection.py
src/evaluation/metrics/portfolio.py
src/evaluation/metrics/policy_health.py
```

核心内容:

- `Phase2Evaluator`:
  - 支持 val 快速评估、完整 walk-forward 评估与 rolling validation 评估。
  - train/val/test 的 walk-forward 均按时间正序、仓位连续执行。
  - rolling validation 必须固定 fold 切法与种子，产出 `phase2_rolling_validation.json` 与 `phase2_rolling_validation_records.feather`。
- baseline:
  - `random_selector`
  - `single_archetype_k`
  - `buy_and_hold`
  - `phase1_demo_label` 仅 posthoc baseline，不能进入主 checkpoint 选择。
- `Phase2SelectionPolicy`:
  - 基于 val 主指标做 best verdict。
  - 若开启 rolling validation，sign-off 时必须同时检查 fold 均值、最差 fold 分位和 fold 间波动；rolling 结果可作为 sign-off 附加硬约束，即使主 best 仍由单一 val 主指标选出。
  - guardrails 包含:
    - `max_drawdown`
    - `min_sharpe`
    - `max_turnover_ratio`
    - `max_action_dominance_ratio`
    - `min_active_archetype_ratio`
  - `val_kl_to_demo`、`phase1_demo_label_selector_val_net_return` 仅作 diagnostic，不作为 hard gate。
- `Phase2CheckpointManager`:
  - 保存 `last_selector.pt`。
  - 根据 verdict promote `best_selector.pt`。
  - 写 `phase2_checkpoint_manifest.json`。
- KL/demo 必跑消融矩阵:
  - 最少固化 `kl_demo_coef ∈ {0, 0.1, 0.5, 1.0}`。
  - 另补 `kl_demo_coef=1.0` 且训练后期 `anneal_to=0` 的 baseline。
  - 产出 `phase2_ablation_kl_demo.json` 与 `phase2_ablation_summary.csv`，主实验 report 必须引用其摘要。

验收:

```bash
pytest tests/unit/trainers/test_phase2_checkpoint.py
pytest tests/unit/trainers/test_phase2_selection_policy.py
pytest tests/unit/evaluation/metrics/test_selection.py
pytest tests/unit/evaluation/metrics/test_portfolio.py
pytest tests/unit/evaluation/metrics/test_policy_health.py
pytest tests/unit/evaluation/test_phase2_metrics.py
pytest tests/unit/evaluation/test_phase2_replay.py
pytest tests/unit/evaluation/test_phase2_evaluator.py
pytest tests/unit/evaluation/test_phase2_rolling_validation.py
```

### Step 7: Report、diagnostics、visualization 与 risk/health audit

生成文件:

```text
src/evaluation/phase2_report.py
src/evaluation/diagnostics/selector_visualization.py
src/evaluation/diagnostics/phase2_failure_case_report.py
```

核心内容:

- `Phase2ReportWriter` 写出:
  - `phase2_report.json`
  - `phase2_baselines_{val,test}.json`
  - `composite_score_sensitivity_phase2.json`
  - `phase2_rolling_validation.json`
- `phase2_report.json` 至少包含:
  - 配置 hash、Phase I hash、schema hash
  - horizon 覆盖、label 覆盖、temporal coverage
  - PPO 健康统计
  - train/val/test scalar 指标
  - `equity_curve_summary`
  - `behavior_health_warnings`
  - `risk_health_warnings`
  - `ood_warning_count`
  - `max_risk_control_response_lag`
  - `input_norm_stats_merge_protocol`
  - `running_mean_std_mode`
  - `running_mean_std_signoff_allowed`
  - rolling validation summary（fold 均值、最差分位、波动）
- `equity_curve_summary` 最小结构建议固定为:

```json
{
  "start_value": 1.0,
  "end_value": 0.0,
  "max_value": 0.0,
  "min_value": 0.0,
  "max_drawdown_start_step": 0,
  "max_drawdown_end_step": 0,
  "peak_step": 0,
  "valley_step": 0,
  "per_horizon_cumulative_pnl": []
}
```
- `selector_visualization.py` 输出:
  - 时间 vs 累计收益 vs archetype 选择
  - action distribution
  - entropy / KL 曲线
  - label temporal coverage 可视化
- `phase2_failure_case_report.py` 输出:
  - worst return
  - largest regret
  - largest cost
  - unstable switching
  - risk trigger cases

验收:

```bash
pytest tests/unit/evaluation/test_phase2_report.py
pytest tests/unit/evaluation/test_phase2_rolling_validation.py
pytest tests/unit/evaluation/diagnostics/test_selector_visualization.py
pytest tests/unit/evaluation/diagnostics/test_phase2_failure_case_report.py
```

### Step 8: Live safety / OOD / latency / stress protocol

完善或新增逻辑文件（默认并入既有模块；只有当实现复杂度明显上升时才拆独立文件）:

```text
src/evaluation/phase2_evaluator.py
src/evaluation/phase2_report.py
src/trainers/phase2_selection_policy.py
src/trading/horizon_env.py                 # mid-horizon flatten / live risk trigger
src/rl/ppo_trainer.py                      # numerical safety fail-fast
# 可选：仅当 OOD 逻辑独立时再新增
# src/evaluation/distribution_shift.py
```

核心内容:

- `live_risk_controls`:
  - `daily_loss_limit`
  - `rolling_drawdown_limit`
  - `consecutive_loss_limit`
  - `flatten_on_trigger`
  - `mid_horizon_emergency_flatten`
  - 必须唯一确定收益结算语义：当 `mid_horizon_emergency_flatten=true` 时，触发点立即结算一次 liquidation action 及其 cost；之后当前 horizon 默认继续以 flat 状态推进到末尾并累加剩余 flat reward，除非显式配置 `terminate_episode_on_risk_trigger=true`。
  - `done/truncated/risk_triggered` 组合语义必须固定写入代码与 report，禁止实现分叉。
- `distribution_shift`:
  - `zscore / PSI / mahalanobis` 至少一种
  - 明确 OOD 使用哪些状态维度；默认只用 market features，不混入账户状态
  - 触发后 fallback 到 `flat_only` 或保守模式
- `execution_stress`:
  - commission × 1.5
  - slippage × 1.5 / 2.0
  - `execution_lag + 1 / +2 bars`
  - selector latency `p50 / p95 / p99`
- `online_action_throttle`:
  - `min_confidence_for_non_flat_action`
  - `max_archetype_switches_per_N_horizons`
  - `cooldown_after_large_turnover`
  - `max_position_change_per_horizon`
- `numerical_safety`:
  - tensor 非 finite fail-fast
  - gradient 爆炸 fail-fast
  - debug snapshot 导出

验收（优先并入已有模块测试，避免测试文件与实现边界错位）:

```bash
pytest tests/unit/trading/test_horizon_env.py
pytest tests/unit/evaluation/test_phase2_evaluator.py
pytest tests/unit/trainers/test_phase2_selection_policy.py
pytest tests/unit/rl/test_ppo_trainer.py
# 若 OOD 逻辑独立成模块，再追加
# pytest tests/unit/evaluation/test_distribution_shift.py
```

### Step 9: Phase2Trainer 与 train/backtest 集成入口

生成文件:

```text
src/trainers/phase2_trainer.py
scripts/train_phase2.py
scripts/backtest_phase2.py
```

核心内容:

- `Phase2Trainer` 编排完整流程:
  - 校验上游产物
  - 读取数据与 schema
  - 生成 horizon index
  - join labels
  - 构造 envs
  - rollout / update / evaluate / select / checkpoint
  - rolling validation / KL-demo ablation / sensitivity 分析
  - best checkpoint 冻结后输出 train/val/test per-horizon records
  - 在每个完整 checkpoint 边界刷出 `replay_log_last_complete_checkpoint.feather` 供 resume 一致性验证
- `scripts/train_phase2.py`:
  - 支持小数据 smoke run
  - 支持 resume
  - 训练入口默认不得加载 test labels
- `scripts/backtest_phase2.py`:
  - 强制主结果为 deterministic argmax
  - 可选 stochastic seed pack 诊断
  - 若检测到 test label 进入决策路径直接抛错

验收:

```bash
pytest tests/unit/trainers/test_phase2_trainer.py
```

### Step 10: 完整 smoke pipeline、resume、sign-off 验收

完善文件:

```text
scripts/train_phase2.py
scripts/backtest_phase2.py
run_pipeline.sh
```

核心内容:

- 增加基于 fixture 的 smoke 训练命令。
- 增加 Phase I smoke 产物的前置生成步骤，避免 Phase II smoke pipeline 依赖隐含前提。
- 增加 `--resume-from` 的 checkpoint 恢复路径。
- 增加 `shadow -> paper -> canary -> full deployment` 的上线前检查清单入口说明。
- `run_pipeline.sh` 中预留 Phase II 调用与 backtest 调用。

建议在 smoke run 前显式生成或校验 Phase I smoke 产物。若不复用 Phase I smoke 训练产物，则新增脚本:

```text
tests/fixtures/phase2/generate_phase1_smoke_artifacts.py
```

最小输出目录建议为:

```text
artifacts/TEST/smoke_phase1/phase1/
  decoder.pt
  codebook.pt
  encoder.pt                    # 可选
  horizon_labels_train.feather
  horizon_labels_val.feather
  horizon_labels_test.feather
  input_schema.json
  reward_normalizer.json
  feature_provenance.json
  phase1_config.yaml
  phase1_report.json
  checkpoint_manifest.json
```


Phase II checkpoint 恢复额外固定产物建议为:

```text
artifacts/{PAIR}/{PHASE2_BATCH_ID}/phase2/replay_log_last_complete_checkpoint.feather
```

最小字段 schema:

```text
update_idx
env_id
sample_id
timestamp_start
chosen_code
final_position
reward_raw
boundary_cost
risk_triggered
```

建议 smoke run:

```bash
# 确保 Phase I smoke 产物存在；二选一：
# 1) 直接复用已完成的 Phase I smoke 训练产物
# 2) 用专用脚本生成最小可用冻结产物
python tests/fixtures/phase2/generate_phase1_smoke_artifacts.py

python scripts/train_phase2.py \
  --pair TEST \
  --phase1-batch-id smoke_phase1 \
  --phase2-batch-id smoke_phase2 \
  --train-file tests/fixtures/phase2/market_train.feather \
  --val-file tests/fixtures/phase2/market_val.feather \
  --test-file tests/fixtures/phase2/market_test.feather \
  --total-timesteps 1024 \
  --num-envs 2 \
  --rollout-length 8 \
  --update-epochs 2 \
  --minibatch-size 8 \
  --seed 42
```

验收:

```bash
pytest tests/integration/test_phase2_pipeline_smoke.py
pytest tests/integration/test_phase2_resume_checkpoint.py
pytest tests/integration/test_phase2_backtest_walk_forward.py
pytest tests/integration/test_phase2_gae_no_cross_env_leakage.py
```

## 4. 单元测试用例计划

### 数据层

`tests/unit/data/test_phase2_horizon_index.py`

- `non_overlap` 生成的 horizons 不重叠。
- `stride` 模式按给定 stride 生效。
- `phase1_index` 模式下 `phase1_sample_id` 与 `sample_id` 对齐。
- markout 越界 horizon 被裁掉。
- gap horizon 被标记并按配置裁掉。
- test index 默认不包含 `code_label`。

`tests/unit/data/test_phase2_dataset.py`

- `state` 维度与 `feature_columns + position_encoding + optional extensions` 一致。
- `position_continuity=true` 时 `prev_terminal_position` 必须进入状态。
- 数据集不重写 Phase I horizon slicing 语义。
- `phase2_dataset` 不调用 DP。

`tests/unit/data/test_phase2_label_loader.py`

- 只 join train/val 的 `code_label`。
- 未标注 horizon `is_labeled=false`。
- `kl_label_temporal_coverage` 正确聚合，并能按时间顺序输出原始覆盖序列。
- label 时间分布熵过低时写入 warning。
- test labels 被请求时抛错。

### Frozen policy / 交易层

`tests/unit/models/test_phase1_frozen_policy.py`

- `decoder` 参数被冻结。
- `decode_step()` 可逐步输出 action logits。
- 修改未来 state 不改变过去 timestep 输出。
- `decode()` 仅诊断路径可用，主 replay 测试不调用。

`tests/unit/trading/test_horizon_env.py`

- `reset()` 返回第一个 horizon 的 `s^sel`。
- `step(action)` 会执行完整 horizon 并返回 `r^sel`。
- `prev_terminal_position` 正确继承到下一个 horizon。
- `decode_step()` 被调用 `h` 次，而不是批量 `decode()`。
- `position_continuity=false` 时每个 horizon 从 flat reset。
- `gap <= threshold` 且 mode=`carry` 时继承仓位。
- `gap > threshold` 且 mode=`force_flatten` 时仓位归零。
- `gap > threshold` 且 mode=`warmup_only` 时 warm-up 行为正确。
- 风险触发时支持 mid-horizon flatten。

`tests/unit/trading/test_horizon_factory.py`

- `num_envs` 连续时间分片正确。
- 每个 env 独立维护 `cursor / prev_terminal_position`。
- `phase2_env_shards.feather` 记录正确。
- `rollover` 模式仅在诊断配置下可启用。

### 模型与 RL 层

`tests/unit/models/test_archetype_selector.py`

- actor 输出 logits `[batch, K]`。
- critic 输出 value `[batch]`。
- dead code mask 正确把 logit 设为 `-inf`。
- deterministic 模式返回 argmax。
- stochastic 模式可采样不同 action。
- unmasked diagnostic rollout 中，关闭 dead code mask 后可记录 dead code `probe_pick_rate`。

`tests/unit/rl/test_actor_critic.py`

- `act()` 返回 `action/log_prob/value`。
- `evaluate_actions()` 返回 `log_prob/entropy/value`。
- 多档仓位编码与 `state_dim_breakdown` 一致。

`tests/unit/rl/test_rollout_buffer.py`

- buffer 保存字段完整。
- `done` 与 `truncated` 区分正确。
- flatten minibatch 前按 env 分组。
- raw/scaled reward 同步记录。

`tests/unit/rl/test_ppo_loss.py`

- clip surrogate 正确。
- value loss 正确。
- entropy bonus 正确。
- `kl_demo_loss` 只在 `is_labeled=true` 上生效。
- masked KL label 样本 loss=0。

`tests/unit/rl/test_ppo_trainer.py`

- rollout -> GAE -> update 主链路可跑通。
- `approx_kl > target_kl` 时 early stop。
- `advantage_normalization` 开关生效。
- reward clip 开启时同时记录 clipped/unclipped 统计。
- reward clip 开启时 report 同时保留 unclipped 对照统计。

`tests/unit/rl/test_scheduling.py`

- lr schedule 生效。
- entropy coef anneal 生效。
- kl_demo coef anneal 生效。

`tests/unit/rl/test_running_mean_std_ablation.py`

- `per_env_only` 模式下各 env 统计互不污染。
- `delayed_merge_next_rollout` 模式下合并结果只在下一个 rollout 生效。
- 当前 rollout 不会消费本 rollout 内刚更新/合并出的统计量。

### 评估与风控层

`tests/unit/evaluation/metrics/test_selection.py`

- action dominance / active archetype ratio 计算正确。
- dead code usage 检查正确。

`tests/unit/evaluation/metrics/test_portfolio.py`

- net_return / sharpe / sortino / MDD / calmar 正确。
- turnover / boundary_cost / cost_paid 正确。
- `equity_curve_summary` 结构正确。

`tests/unit/evaluation/metrics/test_policy_health.py`

- `approx_kl / clip_fraction / explained_variance` 诊断正确。
- `kl_demo_dominance_ratio` 计算正确。
- `per_archetype_reward_mean_and_std` 聚合正确。

`tests/unit/evaluation/test_phase2_live_risk_controls.py`

- 达到 `daily_loss_limit` 触发 flatten。
- `mid_horizon_emergency_flatten=true` 时能立即截断当前 horizon。
- 记录 `max_risk_control_response_lag`。

`tests/unit/evaluation/test_phase2_distribution_shift.py`

- OOD score 基于指定 state 维度计算。
- 超阈值触发 fallback。
- OOD 维度与 `state_dim_breakdown` 对齐。

`tests/unit/evaluation/test_phase2_execution_stress.py`

- `commission/slippage/execution_lag` stress 场景可运行。
- `execution_lag +2 bars` 结果写入 stress summary。
- selector latency `p50/p95/p99` 被记录。

`tests/unit/evaluation/test_phase2_online_action_throttle.py`

- 低置信度强制 `flat_only`。
- archetype 切换频率超阈时触发 cooldown。
- `max_position_change_per_horizon` 生效。

`tests/unit/evaluation/test_phase2_numerical_safety.py`

- 非 finite tensor 触发 fail-fast。
- gradient norm 爆炸触发 fail-fast。
- debug snapshot 路径被写出。

`tests/unit/evaluation/test_phase2_report.py`

- `phase2_report.json` 含配置、hash、coverage、scalar metrics、health warnings、stress summary。
- `equity_curve_summary` 字段存在。
- `input_norm_stats_merge_protocol` / `running_mean_std_mode` 字段存在。
- rolling validation summary 字段存在。
- `test_used_for_selection=false` 严格写入。

`tests/unit/evaluation/diagnostics/test_selector_visualization.py`

- 生成“时间 vs 累计收益 vs archetype 选择”图。
- 生成 label temporal coverage 图。

`tests/unit/evaluation/diagnostics/test_phase2_failure_case_report.py`

- 生成 worst_return / largest_cost / unstable_switching 案例。

`tests/unit/evaluation/test_phase2_rolling_validation.py`

- 固定 fold 切法在同 seed 下结果一致。
- fold 均值、最差分位、波动聚合正确。

### 训练与产物层

`tests/unit/trainers/test_phase2_checkpoint.py`

- 保存 `last_selector.pt`。
- verdict 允许时 promote `best_selector.pt`。
- manifest 写入 metrics、reasons、hash。

`tests/unit/trainers/test_phase2_selection_policy.py`

- `max_drawdown` 超阈拒绝 best。
- `action_dominance` 过高拒绝 best。
- `phase1_demo_label_selector_val_net_return` 仅写 warning，不拒绝 best。
- 当 selector argmax 收益持续低于 demo label selector 收益时，`behavior_health_warnings` 写入相应记录。
- `val_kl_to_demo` 不进入 composite score。

`tests/unit/trainers/test_phase2_trainer.py`

- trainer 可以跑通完整 orchestrator。
- 训练结束后导出 per-horizon records。
- sensitivity 结果写入 JSON。
- KL/demo ablation matrix 能生成 `phase2_ablation_kl_demo.json` 与 summary CSV。

## 5. 集成测试用例计划

`tests/integration/test_phase2_pipeline_smoke.py`

目标: 使用小型 fixture 数据与一个 smoke Phase I 产物跑通完整 Phase II。

输入:

```text
tests/fixtures/phase2/market_train.feather
tests/fixtures/phase2/market_val.feather
tests/fixtures/phase2/market_test.feather
artifacts/TEST/smoke_phase1/phase1/*
```

命令:

```bash
python scripts/train_phase2.py \
  --pair TEST \
  --phase1-batch-id smoke_phase1 \
  --phase2-batch-id integration_smoke \
  --train-file tests/fixtures/phase2/market_train.feather \
  --val-file tests/fixtures/phase2/market_val.feather \
  --test-file tests/fixtures/phase2/market_test.feather \
  --total-timesteps 1024 \
  --num-envs 2 \
  --rollout-length 8 \
  --update-epochs 2 \
  --minibatch-size 8 \
  --seed 7
```

断言:

- 进程退出码为 0。
- 以下文件存在:

```text
artifacts/TEST/integration_smoke/phase2/phase2_config.yaml
artifacts/TEST/integration_smoke/phase2/phase2_horizon_index_train.feather
artifacts/TEST/integration_smoke/phase2/phase2_horizon_index_val.feather
artifacts/TEST/integration_smoke/phase2/phase2_horizon_index_test.feather
artifacts/TEST/integration_smoke/phase2/phase2_env_shards.feather
artifacts/TEST/integration_smoke/phase2/best_selector.pt
artifacts/TEST/integration_smoke/phase2/last_selector.pt
artifacts/TEST/integration_smoke/phase2/phase2_checkpoint_manifest.json
artifacts/TEST/integration_smoke/phase2/phase2_rollout_stats.feather
artifacts/TEST/integration_smoke/phase2/phase2_per_horizon_records_train.feather
artifacts/TEST/integration_smoke/phase2/phase2_per_horizon_records_val.feather
artifacts/TEST/integration_smoke/phase2/phase2_per_horizon_records_test.feather
artifacts/TEST/integration_smoke/phase2/phase2_report.json
```

- `phase2_report.json` 中:
  - `test_used_for_selection == false`
  - `phase1_batch_id == "smoke_phase1"`
  - `kl_label_coverage_train` 字段存在
  - `equity_curve_summary` 字段存在
  - `behavior_health_warnings` 字段存在
  - `risk_health_warnings` 字段存在
  - `ood_warning_count` 字段存在
- `phase2_horizon_index_test.feather` 中默认无 `code_label`。

`tests/integration/test_phase2_backtest_walk_forward.py`

目标: 验证 best selector 在 test 上 walk-forward 回测时的仓位连续性和 deterministic argmax 主路径。

断言:

- `prev_terminal_position` 在相邻 horizons 间正确传递。
- 主结果使用 argmax。
- stochastic seed pack 只写诊断，不覆盖主结果。

`tests/integration/test_phase2_resume_checkpoint.py`

目标: 验证 checkpoint 恢复训练。

步骤:

1. 先跑 `total_timesteps=256`。
2. 再用 `--resume-from last_selector.pt --total-timesteps 512`。

断言:

- 第二次训练从上次 update 继续。
- optimizer / scheduler / RNG / env cursor / `prev_terminal_position` 被恢复。
- 恢复后第一个 horizon 的 `prev_terminal_position` 通过一致性校验。
- 一致性校验通过重放 checkpoint 前的 replay_log 比对 `(sample_id, chosen_code, final_position)` 完成；若不一致，写 warning 并用重放结果覆盖 checkpoint 中的仓位状态。

`tests/integration/test_phase2_gae_no_cross_env_leakage.py`

目标: 确保完整 rollout + GAE 过程中不会跨 env 混算。

断言:

- 构造 2 个 reward 方向相反的 env。
- 跑一次完整 rollout + GAE。
- 每个 env 的 advantage 只依赖自己 env 的 reward/value 序列。

`tests/integration/test_phase2_no_test_label_leakage.py`

目标: 确保 test label 无法进入决策路径。

断言:

- backtest 若检测到 `code_label` 进入 selector 决策路径直接抛错。
- `phase2_backtest_runner` 可以记录 posthoc baseline，但不能将其用于 action 选择。
- `_guard_no_test_label_in_decision_path()` 在 selector 调用前后强制执行。

`tests/integration/test_phase2_streaming_decode_only.py`

目标: 确保 HorizonEnv.step() 主路径只用 streaming decode。

断言:

- mock `Phase1FrozenPolicy.decode()`，若被调用则测试失败。
- `decode_step()` 被调用 `h` 次。

## 6. 单元测试数据计划

生成 fixture 脚本:

```text
tests/fixtures/phase2/generate_phase2_fixtures.py
tests/fixtures/phase2/generate_phase1_smoke_artifacts.py
```

生成文件:

```text
tests/fixtures/phase2/market_train.feather
tests/fixtures/phase2/market_val.feather
tests/fixtures/phase2/market_test.feather
tests/fixtures/phase2/market_with_gap.feather
tests/fixtures/phase2/market_bad_schema.feather
tests/fixtures/phase2/market_ood_shift.feather
```

正常 fixture 字段:

```text
timestamp
close
ask1_price ... ask5_price
ask1_size ... ask5_size
bid1_price ... bid5_price
bid1_size ... bid5_size
total_trade_volume
turnover
open_interest
feature_return_1
feature_vol_4
feature_momentum_8
```

fixture 约束:

- 需要与 smoke Phase I 的 `input_schema.json` 保持一致。
- `close` 仅用于 replay / reward / markout，不进入 selector state。
- selector state 默认由 `feature_* + prev_terminal_position` 组成。
- `close` 序列长度的 fixture 设计必须能覆盖 `h+2` 行访问，以同时支持 `paper_formula` 与 `next_row_execution`。
- `market_with_gap.feather` 人为制造 timestamp gap，用于测试 gap 过滤与仓位处理。
- `market_ood_shift.feather` 在后半段制造 feature 分布漂移，用于 OOD fallback 测试。

数据规模:

- train: 96 行。
- val: 48 行。
- test: 48 行。
- smoke horizon: 8。

价格场景:

- 前段缓慢上涨。
- 中段震荡横盘。
- 后段下跌并夹杂高波动段。
- 至少包含一个跨 horizon 的持续趋势段，用于测试 multi-env shard 边界。

fixture 生成命令:

```bash
python tests/fixtures/phase2/generate_phase2_fixtures.py
```

## 7. 代码审查计划

每个 Step 完成后按以下清单审查。

数据与泄漏审查:

- train/val/test 的 schema 与 `input_schema.json` 完全一致。
- `feature_provenance.json` 中所有 feature 的可用时间不晚于决策时点。
- selector 状态不读取 horizon 内未来行。
- HorizonEnv 正式 replay 路径不调用 `decode()`。
- test label 不进入训练、回测主路径、best 选择、早停和 report 主指标决策。
- `phase1_index` 不能作为正式 sign-off 默认模式。
- `RunningMeanStd` 如启用，只能作为 ablation，并按延迟合并或 per-env 独立协议执行。
- rolling validation 必须真实落地运行，不能只保留配置字段。
- KL/demo 消融矩阵必须产生产物并被主 report 引用。

交易语义审查:

- `TradingEnv`、`CostModel`、`RewardAlignment` 与 Phase I 完全一致。
- `initial_position != 0` 的边界换仓成本正确。
- gap horizon 裁切后，不能静默 flat reset；必须按配置决定继承、强平或 warm-up。
- live risk trigger 发生时，mid-horizon flatten 行为与 report 延迟统计一致。
- mid-horizon flatten 的 liquidation cost、剩余 horizon 的 flat 推进语义、`done/truncated` 组合必须唯一确定。

PPO / RL 审查:

- GAE 严格按 `env_id` 分组。
- `done` 与 `truncated` 语义未混淆。
- 需有集成测试证明多 env rollout + GAE 不会跨 env 混算。
- `kl_demo_loss` 只在 `is_labeled=true` 生效。
- masked dead code 不会导致 KL loss 或 evaluation 崩溃。
- reward scaling clip 默认关闭；若开启，必须审计 clipped/unclipped 差异。
- best checkpoint 主路径只用 deterministic argmax 主结果。

工程与恢复审查:

- resume 必须恢复 optimizer、scheduler、RNG、env cursor、`prev_terminal_position`、decoder recurrent state。
- 恢复后首个 horizon 的仓位一致性必须被验证。
- `replay_log_last_complete_checkpoint.feather` 的 schema 与刷盘时机必须固定，不能留给实现时自由发挥。
- numerical fail-fast 有明确退出码与 debug snapshot。
- latency / stress / OOD / throttle 结果必须写入 report。

产物审查:

- 所有输出写入 `artifacts/{PAIR}/{PHASE2_BATCH_ID}/phase2/`。
- `phase2_checkpoint_manifest.json` 记录 verdict、理由、hash、best timestep。
- `phase2_report.json` 包含 coverage、scalar、health、stress、equity curve summary。
- `phase2_per_horizon_records_{train,val,test}.feather` 可直接供 Phase III 消费或复核。

## 8. 执行命令总览

安装依赖:

```bash
python3 -m pip install -r requirements.txt
```

生成测试数据:

```bash
python tests/fixtures/phase2/generate_phase2_fixtures.py
python tests/fixtures/phase2/generate_phase1_smoke_artifacts.py
```

运行单元测试:

```bash
pytest tests/unit -q
```

运行集成测试:

```bash
pytest tests/integration -q
```

运行全部测试并统计覆盖率:

```bash
pytest tests --cov=src --cov=scripts --cov-report=term-missing
```

运行 Phase II smoke 训练:

```bash
python tests/fixtures/phase2/generate_phase1_smoke_artifacts.py
python scripts/train_phase2.py \
  --pair TEST \
  --phase1-batch-id smoke_phase1 \
  --phase2-batch-id smoke_phase2 \
  --train-file tests/fixtures/phase2/market_train.feather \
  --val-file tests/fixtures/phase2/market_val.feather \
  --test-file tests/fixtures/phase2/market_test.feather \
  --total-timesteps 1024 \
  --num-envs 2 \
  --rollout-length 8 \
  --update-epochs 2 \
  --minibatch-size 8 \
  --seed 42
```

运行 Phase II test 回测:

```bash
python scripts/backtest_phase2.py \
  --pair TEST \
  --phase1-batch-id smoke_phase1 \
  --phase2-batch-id smoke_phase2 \
  --test-file tests/fixtures/phase2/market_test.feather \
  --checkpoint artifacts/TEST/smoke_phase2/phase2/best_selector.pt
```

运行真实数据 Phase II:

```bash
python scripts/train_phase2.py \
  --pair AL \
  --phase1-batch-id batch_001 \
  --phase2-batch-id batch_001 \
  --train-file data/AL/train.feather \
  --val-file data/AL/val.feather \
  --test-file data/AL/test.feather \
  --total-timesteps 3000000 \
  --num-envs 8 \
  --rollout-length 256 \
  --update-epochs 4 \
  --minibatch-size 256 \
  --seed 42
```


运行 KL/demo 必跑消融矩阵（建议最少 5 组）:

```bash
for alpha in 0 0.1 0.5 1.0; do
  python scripts/train_phase2.py \
    --pair AL \
    --phase1-batch-id batch_001 \
    --phase2-batch-id batch_kl_${alpha//./_} \
    --train-file data/AL/train.feather \
    --val-file data/AL/val.feather \
    --test-file data/AL/test.feather \
    --total-timesteps 3000000 \
    --num-envs 8 \
    --rollout-length 256 \
    --kl-demo-coef $alpha \
    --seed 42
done

python scripts/train_phase2.py \
  --pair AL \
  --phase1-batch-id batch_001 \
  --phase2-batch-id batch_kl_1_anneal0 \
  --train-file data/AL/train.feather \
  --val-file data/AL/val.feather \
  --test-file data/AL/test.feather \
  --total-timesteps 3000000 \
  --num-envs 8 \
  --rollout-length 256 \
  --kl-demo-coef 1.0 \
  --kl-demo-anneal-to 0.0 \
  --seed 42
```

## 9. 完成定义

Phase II 代码生成完成需要同时满足:

- `pytest tests/unit -q` 通过。
- `pytest tests/integration -q` 通过。
- smoke 训练与 smoke backtest 能生成完整 artifacts。
- `phase2_report.json` 包含 horizon 覆盖、PPO 健康、train/val/test 指标、risk/behavior warnings、stress summary、rolling validation summary 与 equity curve summary。
- `best_selector.pt`、`last_selector.pt`、`phase2_horizon_index_*.feather`、`phase2_per_horizon_records_*.feather`、`phase2_checkpoint_manifest.json`、`replay_log_last_complete_checkpoint.feather` 可被后续阶段读取。
- 代码审查清单全部通过，尤其是:
  - test label 不泄漏
  - streaming decode 主路径锁定
  - GAE 不跨 env 混算
  - `prev_terminal_position` 连续性与 resume 一致性正确
  - live risk / OOD / stress / throttle / numerical fail-fast 有明确定义与产物输出
- backtest 主路径固定为 deterministic argmax，stochastic 只作为诊断输出。
- KL/demo 必跑消融矩阵已完成并产出 `phase2_ablation_kl_demo.json` / summary CSV。
- 设计中的正式 sign-off 限制被落实为代码级硬约束，而不是仅写在文档里。
