# Phase II Archetype Selection 技术设计

## 1. 目标与范围

本文档根据 `docs/paper/AAAI26_ArchetypeTrader_core.md` 中第二阶段 Archetype Selection 的要求，并以 `docs/design/phase1_archetype_discovery_design.md` 中已固化的 Phase I 产物和工程约束为输入，给出第二阶段可落地的工程设计。

第二阶段的目标是: 在 Phase I 训练得到的 `K=10` 个离散 archetype 上，训练一个 horizon 级 RL selector `π_φ^sel(a^sel | s^sel)`，使其在每个 horizon 起点根据当前市场观测 `s_t` 选出最合适的 archetype；该 archetype 的 codebook embedding 输入冻结的 Phase I decoder，由 decoder 因果地生成 horizon 内每一步的 base action `a^base_{t:t+h-1}`，再交由 `TradingEnv` 按统一 reward / cost 语义结算每步收益。

第二阶段只做四件事:

1. 加载 Phase I 的 `decoder.pt`、`codebook.pt`、`horizon_labels_train.feather`、`horizon_labels_val.feather` 和 `input_schema.json`；`horizon_labels_test.feather` 只允许在最终冻结后的 posthoc baseline 中读取。
2. 在 train/val 上按固定、非重叠、时间正序调度生成 horizon-level rollout，构造 horizon-level MDP；test 只在 checkpoint 冻结后做一次最终 walk-forward backtest。
3. 用 PPO 风格 discrete Actor-Critic 训练 selector，目标包含 horizon 累计 reward、KL/demo regularization 与 action entropy。
4. 在 validation 上选择最优 selector checkpoint，在 test 上做 walk-forward backtest，导出 `phase2_report.json` 与 selector 产物。

第二阶段不做以下事情:

- 不重新训练 Phase I 的 encoder/decoder/codebook。Phase I 产物在 Phase II 中全程冻结。
- 不在线调用 DP planner。DP 仅允许在 Phase I 离线生成 demonstration 与 horizon label 时使用，Phase II 训练、验证、测试与未来线上推理一律不能动态调用 DP，避免未来信息泄漏。
- 不重写 reward / cost 语义。所有 step reward 必须经由 Phase I 已有的 `src/trading/env.py`、`src/trading/cost_model.py`、`src/trading/reward_alignment.py` 计算，禁止 Phase II 内部再实现一套手续费、滑点、行号映射或持仓推进逻辑。
- 不引入 step-level refinement。Phase III 的 `{-1, 0, 1}` 单次 override 不属于 Phase II；Phase II selector 只决定 horizon 起点的 `code_id`，horizon 内的动作由冻结 decoder 因果生成。

## 2. 论文要求映射

| 论文要求 | 工程设计 |
| --- | --- |
| 把基础 MDP 抬升为 horizon-level MDP `M_sel = ⟨S_sel, A_sel, R_sel, γ⟩` | `src/trading/horizon_env.py` 的 `HorizonEnv`，外层每 step 为一个 horizon，内部以 Phase I `TradingEnv` 推进 `h` 步 |
| state `s^sel = s_t`，捕捉 horizon 第一根 bar | 严格论文复现只取 horizon 起点 market state；实盘连续仓位模式额外拼接当前已知的 `prev_terminal_position` |
| action `a^sel ∈ {0,...,K-1}` | `ArchetypeSelector` 输出 `K` 类离散 logits |
| 通过冻结 decoder 解码 `a^base_{t:t+h-1}` | `Phase1FrozenPolicy` 包装 `decoder.pt` + `codebook.pt`，按 timestep 因果推理 |
| `r_t^sel = Σ_{τ=t}^{t+h-1} r_τ^step` | `HorizonEnv.step()` 内部调用 `TradingEnv.replay()` 累加 `step_reward` 后返回 |
| 目标 `J = E[Σ γ^t (r_t^sel - α KL(â_t^sel ‖ π_φ^sel(a_t^sel ‖ s_t^sel)))]` | `Phase2Loss` 包含 PPO clipped surrogate + value loss + KL/demo loss + entropy bonus |
| `α = 1` | `selection_kl_coefficient = 1.0` 默认值；可通过 schedule 退火 |
| ground-truth archetype label `â_t^sel` 来自 Phase I VQ encoder | train/val 使用既有 `horizon_labels_*.feather.code_label`；Phase II 不重新跑 encoder，test label 仅 posthoc |
| selector 训练 3M steps | `total_timesteps = 3_000_000` 默认，按 horizon-step 计 |
| 保留 best validation checkpoint | `Phase2SelectionPolicy` 与 `Phase2CheckpointManager`，沿用 Phase I 选择规范 |
| 推理时禁用所有 DP 模块 | `Phase2Inferencer` 与 `Phase2BacktestRunner` 不引入 `src/planners/`，只引用 selector + decoder + codebook + env |

### 2.1 与论文第二阶段的一致性边界

设计文档保留论文第二阶段的核心架构和算法公式:

```text
horizon-level MDP
  -> π_φ^sel(a^sel | s^sel) selects archetype k
  -> codebook[k] -> frozen causal decoder -> base actions a^base_{t:t+h-1}
  -> TradingEnv replay -> r^sel = Σ r^step
  -> PPO update with KL/demo regularization toward â_t^sel
```

以下内容属于工程落地增强，不改变论文第二阶段主干:

| 设计项 | 与论文关系 | 是否改变核心公式 |
| --- | --- | --- |
| Walk-forward horizon 调度（非重叠或固定 stride） | 论文未指定 horizon 串接方式；工程必须给出可复现的调度 | 否，但需固定 stride 与 seed |
| 跨 horizon 仓位继承（`initial_position != flat`） | 论文公式聚焦单 horizon，未明确跨 horizon 边界；Phase I §11 已要求 selector 必须支持继承上一段末仓位 | 不改变公式，但补充边界换仓成本 |
| KL/demo 正则形式 | 论文 `â_t^sel` 是单一 archetype id，KL 等价于 cross-entropy；工程上以 cross-entropy 实现，并保留 label-smoothing 选项 | 等价代换，不改变目标函数 |
| Action mask（dead code 屏蔽） | Phase I 若出现长期低使用率 code，selector 不应再选；论文未涉及 | 否，仅约束策略输出 |
| Entropy bonus | PPO 标准实现细节，论文未指定但通用 | 否 |
| GAE 和 advantage normalization | PPO 标准实现细节 | 否 |
| 状态扩展（`s^sel` 候选包含 horizon 起点窗口或 archetype 历史使用） | 论文严格只用 `s_t`；任何扩展必须作为单独 ablation 标注 | 严格论文复现需关闭 |
| 真实交易场景 walk-forward 评估指标（Sharpe / Sortino / MDD / Calmar / Turnover） | 论文给出净收益曲线；工程上必须报告完整风险调整收益与成本 | 否，但要求成本与收益双面报告 |
| 多 asset 训练 | 论文 BTC/ETH/DOT/BNB 分别训练；工程允许 per-asset 配置或共享 selector + asset embedding 两种模式 | 否；共享模式必须作为单独 BATCH_ID |

## 3. 上游产物与数据契约

### 3.1 Phase I 产物消费清单

Phase II 启动前必须确认以下 Phase I 产物存在且通过 sign-off:

```text
artifacts/{PAIR}/{PHASE1_BATCH_ID}/phase1/
  decoder.pt
  codebook.pt
  encoder.pt                         # Phase II 训练/验证/测试默认不加载；仅 Phase III 或独立 posthoc 分析使用
  horizon_labels_train.feather
  horizon_labels_val.feather
  horizon_labels_test.feather          # 可选；仅 posthoc test baseline 读取，train_phase2 默认不加载
  input_schema.json
  reward_normalizer.json
  feature_provenance.json              # 正式 no-leakage sign-off 必需；记录每个 feature 的可用时间与计算窗口
  phase1_config.yaml
  phase1_report.json
  checkpoint_manifest.json
```

加载约束:

- `phase1_report.json.fatal_collapse=false` 且 `code_assignment_drift_warning=false`，否则 Phase II 拒绝启动并提示重新跑 Phase I。
- `phase1_report.json.hindsight_bias_warning != "exceeded"`，否则 Phase II 不可作为正式版本，必须显式 CLI flag `--allow-phase1-hindsight-warning` 才能进入实验流程，并在 `phase2_report.json` 写入风险确认。
- `phase1_config.yaml.dp.cost_config` 必须与 Phase II 计划使用的 `cost_config` 完全一致（`reward_alignment / commission_rate / slippage_model / book_levels / mark_price / execution_lag / insufficient_depth_policy`）。Phase II 不得"为了在线保守而切换 reward_alignment"，否则 student/teacher 收益、demonstration label 与 selector reward 不可比。
- `decoder.pt` 加载后必须冻结所有参数，且自检 `bidirectional=False`、`hidden_dim/code_dim` 与 `phase1_config.yaml.model` 一致。
- `codebook.pt` 加载后必须冻结，并校验 `num_codes` 与 `code_dim`。
- `input_schema.json` 必须保留 `feature_columns / price_column / excluded_columns`；Phase II 的状态特征列必须严格等于 Phase I `feature_columns`，不得新增或删除任何列。
- `feature_provenance.json` 必须声明每个 `feature_column` 的 `available_at_lag <= 0`、滚动窗口只使用 `t` 及历史行、且不存在 `future_return / centered_* / label / target` 等目标泄漏字段。缺失该文件时只允许实验运行，`phase2_report.json.no_leakage_signoff=false`。

`feature_provenance.json` 最小 schema:

```json
{
  "feature_columns": {
    "example_feature": {
      "source_columns": ["close"],
      "lookback_start_bars": -60,
      "lookback_end_bars": 0,
      "publish_delay_bars": 0,
      "fit_scope": "train_only",
      "uses_future_rows": false,
      "normalization_scope": "train_only"
    }
  },
  "lag_convention": "negative_or_zero_means_known_at_decision_time"
}
```

验收语义:

- `lookback_end_bars <= 0` 且 `uses_future_rows=false`。
- `publish_delay_bars <= 0` 表示该 feature 在 `s_t` 决策时已经可见；若实际数据源有发布延迟，必须把 feature 对齐到可见时间后再进入 `feature_columns`。
- 任意 scaler、rank、winsorize、PCA、行业/全市场截面标准化若需要拟合统计量，`fit_scope` / `normalization_scope` 必须为 `train_only`，val/test 只 transform。
- 缺少 provenance 的 feature、字段名命中 `future|target|label|centered|lead` 黑名单、或 `fit_scope=all_splits` 时，正式 sign-off 必须失败。

### 3.2 输入数据

Phase II 默认仍读取 Phase I 用过的三份 feather:

```text
data/{PAIR}/train.feather
data/{PAIR}/val.feather
data/{PAIR}/test.feather
```

也可通过 CLI 显式覆盖:

```text
--train-file data/{PAIR}/train.feather
--val-file data/{PAIR}/val.feather
--test-file data/{PAIR}/test.feather
```

读取实现复用 `src/data/market_reader.py` 与 `src/data/schema.py`，禁止在 Phase II 内重新写 feather IO。Phase II 读取后只做以下两件事:

1. 校验 `feature_columns` 与 Phase I `input_schema.json` 一致；
2. 计算每个文件的行数、`timestamp` 边界，用于 horizon 调度与 walk-forward 时间窗口。

### 3.3 Horizon 调度

Phase II 的"一步 RL"对应 Phase I 的"一个 horizon"。横向调度规则必须固定且可复现。默认训练/验证/测试都使用非重叠 horizon，因为跨 horizon 仓位继承会让 `stride < h` 的重叠窗口变成同一真实时间段被重复结算:

```yaml
horizon_schedule:
  mode: non_overlap               # non_overlap | stride | phase1_index
  stride: 72                      # mode=stride 时使用；正式训练要求 stride >= h
  non_overlap_stride_minutes: 72  # mode=non_overlap 时使用
  use_phase1_window_index: false  # 若 true，复用 Phase I window_index_*.feather 的 sampled 行
  position_continuity: true       # 默认继承上一 horizon 末仓位
  dense_overlap_ablation: false   # true 时才允许 stride < h；必须 reset position 且不参与正式 sign-off
  data_gap_check:
    enabled: true
    max_allowed_gap_minutes: 5
    drop_gap_horizons: true
  walk_forward:
    enabled: true                 # 仅 test backtest 强制启用
    chunk_minutes: null           # null = 一次性走完 test
    seed: 1234
  reward_alignment_lookahead_check: true
```

调度模式说明:

| `mode` | 用途 | 边界处理 |
| --- | --- | --- |
| `non_overlap` | 正式训练、验证和 test walk-forward 默认；horizon 起点按 `h` 等距 | 末尾 `last_markout_row > num_rows - 1` 的窗口必须裁掉 |
| `stride` | 只用于 dense ablation 或不继承仓位的独立 horizon 实验 | 若 `stride < h`，必须 `dense_overlap_ablation=true`，强制 `position_continuity=false`、`done=True`、每个 horizon 从 flat reset，不可 sign-off |
| `phase1_index` | 与 Phase I sampled horizons 完全对齐，便于 KL/demo label 100% 对应；**仅用于诊断或消融** | 若样本重叠，只能按独立 horizon 训练；不允许继承仓位或做跨重叠窗口 GAE；正式 sign-off 默认禁止把它作为主训练模式，并且 report 必须标注 `kl_label_sampling_bias="hindsight_stratified"` |

无论哪种模式，horizon 调度生成产物 `phase2_horizon_index_{split}.feather` 必须包含:

| 字段 | 说明 |
| --- | --- |
| `sample_id` | Phase II horizon ID（与 Phase I `sample_id` 在 `phase1_index` 模式下完全一致） |
| `start_index` | horizon 起点行号 |
| `end_index` | horizon 终点行号（`start + h - 1`） |
| `last_execution_row` | 由 `RewardAlignment` 决定 |
| `last_markout_row` | 由 `RewardAlignment` 决定 |
| `has_data_gap` | horizon 覆盖区间内是否存在 timestamp 间隔超阈值 |
| `max_timestamp_gap_minutes` | horizon 内最大相邻 timestamp 间隔 |
| `phase1_sample_id` | 若该 horizon 与 Phase I sample 对应，则填入 Phase I 的 `sample_id`；否则 `null` |
| `code_label` | train/val 来自对应 `horizon_labels_*.feather`；test 默认必须为 `null`，除非 posthoc baseline 明确读取 |
| `is_labeled` | 是否拥有 KL/demo 监督信号 |
| `prev_terminal_position` | 训练 rollout 中由前一 horizon 末仓位填充；初始 horizon 为 0 |
| `split` | train / val / test |

边界约束:

- `last_markout_row` 不得越过对应 split 文件的实际最大行号。markout 越界的 horizon 必须裁掉，禁止 NaN/前向填充（与 Phase I §9.2 一致）。
- horizon 覆盖范围内任意相邻 timestamp 间隔若大于 `data_gap_check.max_allowed_gap_minutes`，必须标记 `has_data_gap=true`；正式训练/验证/test walk-forward 默认裁掉该 horizon，禁止用前向填充或插值补成连续数据。
- `phase2_report.json.data_gap_filter` 必须记录每个 split 的候选 horizon 数、gap horizon 数、裁掉比例、最大 gap，以及 `gap_position_carry_applied_count / force_flatten_after_gap_count / warmup_after_gap_count`。
- 正式 Phase II 禁止 `stride < h` 与 `position_continuity=true` 同时出现；启动时必须直接失败。否则同一真实时间段会被多个 selector 决策重复执行，训练 MDP 与真实 walk-forward 不一致。
- val/test 的 horizon 不参与训练采样，必须在训练前固定 seed 生成并写入产物。
- `walk_forward.enabled=true` 时，test horizons 必须按时间严格非重叠串行执行，且每段 horizon 的 `prev_terminal_position` 必须接收前一段的真实末仓位。
- 若中间有 gap horizon 被裁掉，**不能** 因为“没有训练样本”就默认 flat reset。应按 gap 长度执行显式策略: gap 时长 `<= gap_position_carry_threshold_minutes` 时继续继承 `prev_terminal_position` 并写入 `gap_skip_carry=true`；gap 更大时只能按配置执行 `force_flatten` 或 `warmup_only`，不得静默继承或静默清仓。
- `phase1_index` 模式若被用于训练，`phase2_report.json` 必须额外写入 `kl_label_sampling_bias="hindsight_stratified"`、`signoff_eligible=false` 与 `phase1_index_overlap_ratio`；该模式只允许用于对齐 Phase I label 的机制诊断，不作为默认正式方案。

### 3.4 KL/demo label 来源

KL/demo regularization 的 `â_t^sel` 来自 Phase I `horizon_labels_*.feather.code_label`:

- 训练 horizon: 必须使用 Phase I 已经分配好的 `code_label`，不得在 Phase II 内重新跑 encoder。
- val horizon: 同上，仅用于 KL/demo regularization 的验证诊断；不得作为 checkpoint hard guardrail。
- test horizon: `train_phase2.py` 默认不得读取 `horizon_labels_test.feather`，也不得在 test 上计算 KL/demo 后再影响 checkpoint、阈值或配置。若需要 test "模仿基线"对照，只能在 best checkpoint 冻结后由独立 posthoc evaluator 显式读取既有 `horizon_labels_test.feather`；禁止在 Phase II 内对 test 重跑 encoder。

未标注的 horizon（`is_labeled=false`）在训练 batch 中应通过 mask 把 KL term 置零，仍可参与 PPO 的 reward 与 advantage 学习；不允许把没有 label 的 horizon 直接丢弃，否则 walk-forward 训练会出现时间空洞。

KL/demo 标签覆盖偏置审计要求:

- `phase2_report.json` 必须输出 `kl_label_temporal_coverage`，至少包含 train/val 两个 split 中 `is_labeled=true` 的 horizon 按月或按周的覆盖率、时间分布熵、最大/最小 bucket 覆盖差，以及按训练时间顺序排列的 coverage 序列或可视化数据引用，不能只给聚合统计。
- 若 `phase1_index` 训练模式或其他配置导致 `is_labeled` 只集中在少量时间区间，必须写入 `behavior_health_warnings`，并在 sign-off 结论中明确这是 **Phase I hindsight sampling 继承偏置**，不是 selector 自身表现。
- 当 `kl_label_temporal_coverage.time_entropy` 低于预注册阈值时，`behavior_health_warnings` 必须额外写入“可能存在 KL label 时间偏置，selector 的验证表现可能包含 regime-specific 过拟合信号”。
- 可选增强项 `kl_label_only_on_prospective_horizons` 只允许在存在 Phase I prospective strata 对照批次时启用；启用后仅对这些前瞻性 strata 对应的 horizon 计算 KL/demo，其余样本仍保留 PPO reward 学习但 `is_labeled=false`。

### 3.5 状态字段

严格论文复现时，`s^sel` 等于 horizon 第一根 bar 的 state vector。工程实盘默认还必须把上一 horizon 的真实末仓位纳入状态，因为 Phase II reward 会扣除继承仓位导致的边界换仓成本；若 reward 依赖 `prev_terminal_position` 而 selector/critic 看不到该值，训练会退化成 POMDP。

- market feature 部分严格等于 Phase I `feature_columns`，不重做特征工程，不在 Phase II 拟合任何 scaler。
- 状态经由 `Phase1RewardNormalizer` 之外的通道直接喂入 selector；reward normalizer 仅用于 Phase I encoder/decoder 输入，不能拿来变换 Phase II 状态。
- 严格论文复现禁止扩展 `s^sel`，并必须同时关闭跨 horizon 仓位继承或让每个 horizon 从 flat reset，避免隐藏账户状态。若启用工程增强（如把 horizon 起点之前 `L` 根 bar 的特征 pooling 进 selector），必须:
  - `L` 严格只用历史 bar，禁止读 `t+` 之后的任何行；
  - 在配置和 `phase2_report.json` 中标注 `state_extension=past_lookback(L)`；
  - 严格复现实验中关闭。

可选状态扩展（默认全部关闭）:

```yaml
state_extension:
  past_lookback_minutes: 0             # >0 时把过去 L 根 bar 的 mean/std 拼接到 s^sel
  archetype_usage_history_window: 0    # 把过去 W 个 horizon 的 selector action 编码进 s^sel
  account_state:
    include_position: true             # 工程默认开启；把 prev_terminal_position 编码进 s^sel
    include_recent_pnl: false          # 把过去 W 个 horizon 的 r^sel 编码进 s^sel
```

`account_state.include_position=false` 只允许在 `paper_strict_reproduction=true` 或 `position_continuity=false` 的独立 horizon 实验中使用，并必须在 `phase2_report.json` 标注该结果不是实盘连续仓位评估。

## 4. 目录与模块设计

Phase II 复用 Phase I 已有基础设施（`src/trading/`、`src/data/market_reader.py`、`src/data/schema.py`、`src/utils/feather_io.py`），新增 RL 训练、selector 模型、horizon-level env 与 Phase II 专属评估模块。原则:

- Phase II 不污染 Phase I 模块。Phase I 现有文件除 `src/trading/horizon_env.py` 之外不修改；后者作为新增文件，但必须放在 `src/trading/` 内以便复用 cost / reward_alignment。
- RL 算法（rollout buffer、actor-critic、PPO update）独立成 `src/rl/` 子包，保证未来 Phase III 能复用。
- evaluator/replay 与 Phase I 同名但前缀替换为 `phase2_`，避免在 import 时混淆。

```text
scripts/train_phase2.py
scripts/backtest_phase2.py

src/config/phase2_config.py

src/data/phase2_horizon_index.py
src/data/phase2_dataset.py
src/data/phase2_label_loader.py

src/models/archetype_selector.py
src/models/phase1_frozen_policy.py

src/rl/__init__.py
src/rl/rollout_buffer.py
src/rl/actor_critic.py
src/rl/ppo_loss.py
src/rl/ppo_trainer.py
src/rl/scheduling.py

src/trading/horizon_env.py
src/trading/horizon_factory.py

src/trainers/phase2_trainer.py
src/trainers/phase2_checkpoint.py
src/trainers/phase2_selection_policy.py

src/evaluation/phase2_evaluator.py
src/evaluation/phase2_replay.py
src/evaluation/phase2_metrics.py
src/evaluation/phase2_report.py
src/evaluation/metrics/selection.py
src/evaluation/metrics/portfolio.py
src/evaluation/metrics/policy_health.py
src/evaluation/diagnostics/selector_visualization.py
src/evaluation/diagnostics/phase2_failure_case_report.py
```

### 4.1 `scripts/train_phase2.py`

CLI 入口，负责解析参数、加载配置、检查 Phase I 产物完整性、启动训练。

建议 CLI:

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
  --kl-coef 1.0 \
  --entropy-coef 0.01 \
  --seed 42
```

入口职责:

- 校验 Phase I sign-off 状态（`phase1_report.json` 的 collapse / drift / hindsight warning）。
- 解析配置并写入 `phase2_config.yaml`。
- 校验 pipeline 集成参数: 不接受旧的 `--train-batch-id` 作为 Phase II batch 入口；必须显式区分 `--phase1-batch-id` 与 `--phase2-batch-id`。
- 训练流程不得透传 `--with-dp` 或任何会在 Phase II 评估中动态调用 DP 的参数。
- 调用 `Phase2Trainer.run()`。

### 4.2 `scripts/backtest_phase2.py`

回测入口；用 best selector + 冻结 decoder/codebook + `HorizonEnv` 在 test 上做 walk-forward backtest，并输出真实交易场景下的指标。回测不再调用任何 DP，禁止读取 `horizon_labels_test.feather` 用于决策（仅用于事后比较 KL baseline）。

### 4.3 `src/config/phase2_config.py`

镜像 Phase I `phase1_config.py` 风格，使用 frozen dataclass + 显式 `_NESTED_TYPE_MAP`。关键配置组:

```python
@dataclass(frozen=True)
class HorizonScheduleConfig:
    mode: Literal["non_overlap", "stride", "phase1_index"] = "non_overlap"
    stride: int = 72
    non_overlap_stride_minutes: int = 72
    use_phase1_window_index: bool = False
    position_continuity: bool = True
    dense_overlap_ablation: bool = False
    chunk_reset_position: Literal["inherit", "flat"] = "inherit"
    data_gap_check_enabled: bool = True
    max_allowed_gap_minutes: int = 5
    drop_gap_horizons: bool = True
    gap_position_carry_threshold_minutes: int = 120
    gap_large_reset_mode: Literal["force_flatten", "warmup_only"] = "force_flatten"
    env_shard_mode: Literal["contiguous", "round_robin", "rollover"] = "contiguous"
    env_shard_rollover_horizons: int = 0
    walk_forward_enabled: bool = True
    walk_forward_seed: int = 1234


@dataclass(frozen=True)
class StateExtensionConfig:
    past_lookback_minutes: int = 0
    archetype_usage_history_window: int = 0
    include_position: bool = True
    include_recent_pnl: bool = False


@dataclass(frozen=True)
class SelectorNetworkConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    use_layer_norm: bool = True
    actor_head_hidden: int = 128
    critic_head_hidden: int = 128
    action_mask_dead_codes: bool = True
    dead_code_usage_threshold: float = 0.01
    archetype_embedding_dim: int = 16
    input_norm: Literal["layer_norm", "running_mean_std"] = "layer_norm"
    position_encoding: Literal["one_hot_3", "scaled_integer", "bucketed_position"] = "one_hot_3"


@dataclass(frozen=True)
class RewardScalingConfig:
    mode: Literal["none", "divide_by_horizon", "constant"] = "divide_by_horizon"
    constant_scale: float = 1.0
    clip_range: Optional[float] = None
    report_percentiles: tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 3_000_000
    num_envs: int = 8
    rollout_length: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_warmup_coef: float = 0.05
    entropy_warmup_fraction: float = 0.15
    kl_demo_coef: float = 1.0           # 论文 α=1
    kl_demo_label_smoothing: float = 0.0
    kl_demo_anneal_to: Optional[float] = None
    kl_demo_anneal_fraction: float = 0.5
    target_kl: Optional[float] = 0.05
    learning_rate: float = 3.0e-4
    lr_schedule: Literal["constant", "linear"] = "linear"
    batch_size: int = 4096
    minibatch_size: int = 1024
    update_epochs: int = 4
    advantage_normalization: bool = True
    reward_normalization: bool = False
    grad_clip_norm: float = 0.5


@dataclass(frozen=True)
class Phase2RiskGuardrailConfig:
    max_drawdown: float = 0.25
    min_sharpe_ratio: float = 0.0
    max_turnover_ratio: float = 5.0
    max_reject_transition_rate: float = 0.10
    max_cost_paid_ratio: float = 0.50


@dataclass(frozen=True)
class LiveRiskControlConfig:
    enabled: bool = True
    daily_loss_limit: float = 0.03
    rolling_drawdown_limit: float = 0.05
    consecutive_loss_horizons_limit: int = 8
    consecutive_reject_limit: int = 5
    turnover_burst_limit: float = 3.0
    flatten_on_trigger: bool = True
    flatten_trigger_mode: Literal["immediate_mid_horizon", "end_of_horizon"] = "immediate_mid_horizon"
    halt_trading_on_trigger: bool = True


@dataclass(frozen=True)
class DataIntegrityConfig:
    schema_hash_match_required: bool = True
    timestamp_monotonic_check: bool = True
    stale_data_timeout_seconds: int = 120
    crossed_book_reject: bool = True
    nan_inf_feature_reject: bool = True
    feature_range_sanity_check: bool = True
    max_missing_depth_levels: int = 0


@dataclass(frozen=True)
class DistributionShiftConfig:
    enabled: bool = True
    method: Literal["zscore", "psi", "mahalanobis"] = "zscore"
    zscore_threshold: float = 6.0
    psi_threshold: float = 0.25
    mahalanobis_threshold: float = 8.0
    trigger_consecutive_horizons: int = 3
    fallback_mode: Literal["flat_only", "risk_reduced", "warn_only"] = "flat_only"


@dataclass(frozen=True)
class ExecutionStressConfig:
    required_for_signoff: bool = True
    commission_multipliers: tuple[float, ...] = (1.0, 1.5)
    slippage_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)
    execution_lag_stress_bars: tuple[int, ...] = (0, 1)
    reject_rate_injection: tuple[float, ...] = (0.0, 0.05)


@dataclass(frozen=True)
class RollingValidationConfig:
    enabled: bool = True
    mode: Literal["anchored_walk_forward", "multi_era_holdout"] = "anchored_walk_forward"
    num_folds: int = 3
    selection_metric: Literal["mean", "worst_fold", "mean_minus_std"] = "mean_minus_std"


@dataclass(frozen=True)
class OnlineActionThrottleConfig:
    enabled: bool = True
    max_archetype_switches_per_20_horizons: int = 12
    min_confidence_for_non_flat_action: float = 0.40
    cooldown_horizons_after_turnover_burst: int = 3
    unstable_selector_entropy_threshold: float = 1.80
    fallback_mode: Literal["flat_only", "hold_previous"] = "flat_only"


@dataclass(frozen=True)
class NumericalSafetyConfig:
    fail_on_non_finite_tensor: bool = True
    fail_on_non_finite_grad: bool = True
    fail_on_logit_overflow: bool = True
    dump_debug_snapshot_on_failure: bool = True


@dataclass(frozen=True)
class DeploymentLadderConfig:
    require_shadow_replay: bool = True
    require_paper_trading: bool = True
    require_canary: bool = True
    canary_position_scale: float = 0.10


@dataclass(frozen=True)
class Phase2BehaviorGuardrailConfig:
    max_action_dominance_ratio: float = 0.6
    min_active_archetype_ratio: float = 0.5
    warn_kl_to_demo: float = 1.0        # diagnostic only; never rejects best


@dataclass(frozen=True)
class Phase2SelectionPolicyConfig:
    selection_metric: str = "phase2_composite_score"
    selection_mode: Literal["max", "min"] = "max"
    metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "val_net_return": 0.35,
            "val_sharpe_ratio": 0.25,
            "val_calmar_ratio": 0.20,
            "val_active_archetype_ratio": 0.10,
            "val_action_entropy": 0.10,
        }
    )
    risk: Phase2RiskGuardrailConfig = field(default_factory=Phase2RiskGuardrailConfig)
    behavior: Phase2BehaviorGuardrailConfig = field(
        default_factory=Phase2BehaviorGuardrailConfig
    )
    composite_score_sensitivity_perturbations: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {"val_net_return": +0.10},
            {"val_net_return": -0.10},
            {"val_sharpe_ratio": +0.05},
            {"val_calmar_ratio": +0.05},
        ]
    )


@dataclass(frozen=True)
class EnvShardConfig:
    mode: Literal["contiguous", "round_robin"] = "contiguous"
    reward_mean_drift_warn: float = 2.0
    reward_std_drift_warn: float = 2.0


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = False
    metric: str = "val_composite_score"
    mode: Literal["max", "min"] = "max"
    patience_validations: int = 5
    min_delta: float = 0.0


@dataclass(frozen=True)
class ResumeConfig:
    enabled: bool = True
    save_optimizer_state: bool = True
    save_input_norm_stats: bool = True
    save_reward_norm_stats: bool = True
    save_env_cursors: bool = True


@dataclass(frozen=True)
class Phase2Config:
    pair: str
    phase1_batch_id: str
    phase2_batch_id: str
    train_file: str
    val_file: str
    test_file: str
    artifact_root: str = "artifacts"
    horizon: int = 72
    max_position: int = 1
    horizon_schedule: HorizonScheduleConfig = field(default_factory=HorizonScheduleConfig)
    env_shards: EnvShardConfig = field(default_factory=EnvShardConfig)
    state_extension: StateExtensionConfig = field(default_factory=StateExtensionConfig)
    selector_network: SelectorNetworkConfig = field(default_factory=SelectorNetworkConfig)
    reward_scaling: RewardScalingConfig = field(default_factory=RewardScalingConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    selection_policy: Phase2SelectionPolicyConfig = field(
        default_factory=Phase2SelectionPolicyConfig
    )
    cost_alignment_check: bool = True
    paper_strict_reproduction: bool = False
    allow_phase1_hindsight_warning: bool = False
```

`paper_strict_reproduction=True` 时调用 `apply_paper_strict_overrides`:

- 关闭所有 `state_extension`；
- 若关闭 `include_position`，必须同时设置 `horizon_schedule.position_continuity=false` 或强制每个 horizon 从 flat reset；
- 关闭 `action_mask_dead_codes`；
- `entropy_coef=0`；
- `kl_demo_label_smoothing=0`；
- `advantage_normalization=False`；
- `reward_normalization=False`；
- `selector_network.input_norm="layer_norm"`；
- 强制 `paper_strict_reproduction=True` 写入 `phase2_report.json`。
- `paper_strict_reproduction=True` 的结果不得与实盘连续仓位 walk-forward 指标混在同一张主表中比较。

`config_hash()` 必须包含 Phase I `phase1_config.config_hash()` 与 `phase1_checkpoint_hash`，以保证 Phase II 产物可追溯到具体的 Phase I best checkpoint。

启动校验补充:

- `max_position` 必须从 Phase I `phase1_config.yaml.dp.cost_config.max_position` 继承并做一致性校验；不允许 Phase II 单独改写。论文实验里不同 asset 的 `max_position` 不同，这个字段不能隐含掉。
- 当 `max_position > 1` 时，禁止 `selector_network.position_encoding="one_hot_3"`；必须改用 `scaled_integer` 或 `bucketed_position`，并把 `position_encoding_dim` 反映到 `state_dim_breakdown`。
- `state_dim_breakdown` 必须写入 report，明确 market feature 维度、position encoding 维度以及每个可选扩展块的维度。
- `early_stopping.enabled=false` 仍是默认值以对齐论文，但配置和 report 必须保留完整字段，便于审计“若启用早停会停在哪里”。

### 4.4 `src/data/*` 新增模块

| 模块 | 职责 |
| --- | --- |
| `phase2_horizon_index.py` | 按 `HorizonScheduleConfig` 与 `RewardAlignment` 枚举/对齐 horizon 起点，写入 `phase2_horizon_index_{split}.feather`；不读取 Phase I window_index 之外的字段。 |
| `phase2_label_loader.py` | 训练阶段只加载 `horizon_labels_train.feather` / `horizon_labels_val.feather`，按 `sample_id` join 进 `phase2_horizon_index`；为未标注 horizon 填 `code_label=null, is_labeled=False`。test label 只能由 posthoc evaluator 读取。 |
| `phase2_dataset.py` | PyTorch Dataset/IterableDataset；按 `start_index/end_index` 切出 `[h, feature_dim]` 状态张量与 `[h, levels, 4]` execution book，供 `HorizonEnv` 与 `Phase1FrozenPolicy` 推理。复用 Phase I `HorizonBuilder` 的切片逻辑（通过函数级 reuse 而非复制粘贴）。 |

边界约束:

- `phase2_horizon_index.py` 不再做分层采样；分层采样只属于 Phase I demonstration curation。Phase II 训练样本由 walk-forward 调度直接给出。
- `phase2_label_loader.py` 不写文件，只产出 in-memory join 结果。
- `phase2_label_loader.py` 在 `split="test"` 且非 posthoc baseline 模式时必须抛错，防止 test oracle label 进入训练或 checkpoint 选择路径。
- `phase2_dataset.py` 不调用 decoder/selector，只负责把 raw feather 切成张量；模型推理在 `HorizonEnv` 与 `PPOTrainer` 中完成。
- `phase2_dataset.py` 对 `states / prices / execution_books` 的切片必须函数级复用 Phase I `HorizonBuilder` 或其底层共享 helper；禁止在 Phase II 重新实现一套 off-by-one 风险更高的 slicing 逻辑。
- 必须提供集成测试: 对同一 `sample_id`，Phase II 构造出的 `HorizonInputs` 与 Phase I demo/horizon store 中对应样本逐字段完全一致（允许 dtype 一致性范围内的无损转换，但不允许行号偏移）。

### 4.5 `src/models/archetype_selector.py`

```python
class ArchetypeSelector(nn.Module):
    """Horizon-level Actor-Critic.

    Inputs
    ------
    s_sel : Tensor[batch, state_dim]
        horizon 起点的 state vector (含可选 state_extension)。
    action_mask : Optional[Tensor[batch, K]]
        bool mask；True 表示该 archetype 可选；dead code 屏蔽时关键。

    Outputs
    -------
    logits : Tensor[batch, K]
    value  : Tensor[batch]
    """
```

设计要点:

- 共享主干: `MLP(state_dim -> hidden_dim, num_layers, LayerNorm + GELU)`。
- 可选 archetype embedding fusion: 把 `codebook` 的 `K x code_dim` 作为可读不可写的 embedding，主干输出与每个 archetype embedding 做 dot-product/MLP score（仅作为消融，论文严格复现关闭）。
- Actor head: `Linear(hidden_dim, K)` 输出 logits；mask 后做 `log_softmax`。
- Critic head: `Linear(hidden_dim, 1)`。
- 不复用 decoder 内部状态，避免与 Phase I 模型耦合；selector 只看 `s^sel`。

### 4.6 `src/models/phase1_frozen_policy.py`

```python
class Phase1FrozenPolicy:
    """冻结的 decoder + codebook 包装。

    用法
    ----
    >>> action_t, logits_t = policy.decode_step(state_t, code_id, recurrent_state)
    >>> base_actions, decode_logits = policy.decode(states_seq, code_id)  # diagnostics only

    实现注意
    --------
    - decoder.pt 与 codebook.pt 必须从 Phase I best checkpoint 导出。
    - decoder LSTM 必须保持 ``bidirectional=False``；初始化时校验。
    - 推理时使用 ``with torch.no_grad():`` 并永不更新参数。
    - 若 codebook 中部分 code 在 Phase I 是 dead code，本类不阻止解码，但
      ``ArchetypeSelector`` 必须通过 action_mask 屏蔽，否则会输出 garbage 动作。
    """
```

边界:

- 不实现 selector，不实现 RL 算法。
- 不读 horizon_labels；KL/demo 不在本类计算。
- 接口幂等: 同一 `(states_seq, code_id)` 永远返回相同的动作序列。
- `decode_step()` 的返回值必须锁定为 `(base_action_t, action_logits_t, next_recurrent_state)`，其中 `base_action_t` 是已经过 `argmax` 或指定采样规则后的 **TradingEnv action id**，取值严格属于 `{0,1,2}`；`action_logits_t` 仅用于诊断和可解释性，不得把 logits 直接传给 `TradingEnv.step()`。
- `decode()` 的返回值必须锁定为 `(base_actions, decode_logits)`，其中 `base_actions.shape=[h]`、元素取值严格属于 `{0,1,2}`，并与 `TradingEnv` 的动作语义 `0=short, 1=flat, 2=long` 完全一致。
- 正式 walk-forward replay 必须优先使用 `decode_step()` streaming 接口: 每步只把当前可见 `state_t` 送入 decoder，并显式传递 decoder recurrent state。`decode(states_seq, code_id)` 仅用于批量诊断和因果性单测。
- `HorizonEnv.step()` 禁止为了方便而调用 `decode(states_seq, code_id)` 生成完整 horizon 动作后再 replay；正式实现必须循环调用 `decode_step()` 共 `h` 次，并对每一步传入的 `state_t` 做可见性校验。
- 必须提供 action mapping 单元测试，证明 decoder 输出空间与 `TradingEnv` 动作空间完全一致，且不会发生 logits/action id 混用。

### 4.7 `src/rl/*` 子包

```text
src/rl/rollout_buffer.py
src/rl/actor_critic.py
src/rl/ppo_loss.py
src/rl/ppo_trainer.py
src/rl/scheduling.py
```

| 模块 | 职责 |
| --- | --- |
| `rollout_buffer.py` | 存储 `obs / action / log_prob / value / reward / done / truncated / kl_label / is_labeled`；提供 GAE advantage、return、minibatch 采样。 |
| `actor_critic.py` | 把 `ArchetypeSelector` 封装为 PPO 标准接口（`act / evaluate_actions / get_value`），与 `RolloutBuffer` 解耦。 |
| `ppo_loss.py` | 计算 PPO clipped surrogate / value loss / entropy bonus / KL-demo loss；提供 KL early-stop。 |
| `ppo_trainer.py` | 编排 rollout 采集与 update；不负责 horizon-level env 创建（由 `phase2_trainer` 注入 `env_factory`）。 |
| `scheduling.py` | learning rate、entropy coef、kl_demo coef 的 linear/cosine schedule；唯一信源，trainer 不自实现。 |

边界约束:

- `src/rl/*` 不引用 `src/trading/`、`src/data/`、`src/models/` 之外的内容；只依赖 PyTorch 与自身模块。
- 任何 trader 业务字段（`code_label / is_labeled / dead_code_mask`）通过 `RolloutBuffer` 的可选字段透传，PPO 算法层不知道字段含义。
- 不写 IO；checkpoint 由 `phase2_checkpoint.py` 处理。

### 4.8 `src/trading/horizon_env.py`

`HorizonEnv` 是把 Phase I `TradingEnv` 抬升到 horizon 级 Gym-like env:

```python
class HorizonEnv:
    """Horizon-level RL env.

    每次 horizon step:
      action a^sel ∈ {0,...,K-1}
      -> codebook[a^sel] 输入 frozen decoder
      -> decoder streaming 因果生成 a^base_{0:h-1}
      -> Phase I TradingEnv.step/replay 累加 step rewards
      -> 返回 (next_obs, r^sel, done, info)

    obs:
      horizon 起点的 s^sel；done 后切到下一个 horizon。

    cross-horizon position:
      prev_terminal_position 在 reset 时注入；保证 selector 不假设 flat 起点。
    """

    def reset(self, *, prev_terminal_position: int = 0) -> dict: ...
    def step(self, action: int) -> tuple[dict, float, bool, dict]: ...
```

Gym-like 语义必须明确锁定:

- `reset()` 返回当前游标指向的 **第一个可执行 horizon** 的 `s^sel`，并把该 horizon 对应的 `HorizonInputs` 注入内部 `TradingEnv.reset(horizon=..., initial_position=prev_terminal_position)`。
- `step(action)` 必须执行一个完整 horizon：selector 传入一个 `code_id`，`Phase1FrozenPolicy` 在该 horizon 内 streaming 生成 `h` 步 `base_action_t`，内部 `TradingEnv` 回放完整 horizon 后返回 `(next_horizon_s_sel, r^sel, done, info)`。
- `next_obs` 不是当前 horizon 的中间状态，而是 **下一个 horizon 起点** 的 `s^sel`；若当前 horizon 已经是该 env 分片末端，则 `done=True` 且 `next_obs` 按 Gym 约定返回 reset 后首个观测或显式 `terminal_observation`。
- `HorizonEnv` 必须维护自己的 `horizon_cursor`，每次 `step()` 后前移 1 个 horizon；多 env 并行时每个 env 的 cursor 与 `rollout_buffer.env_id` 一一对应，禁止共享。

实现要点:

- `HorizonEnv` 不直接从原始 market frame 临时拼装对象；必须在构造时注入 `phase2_horizon_index` 与 `Phase2Dataset/HorizonBuilder` 提供的 `HorizonInputsProvider`，由 provider 按 `sample_id/start_index/end_index` 返回与 Phase I 同构的 `HorizonInputs`。
- 内部维护一个 `TradingEnv` 实例和 `Phase1FrozenPolicy`，不在 `HorizonEnv` 里重写成本/行号映射。
- `info` 至少返回 `step_rewards / cost_paid / num_switches / boundary_turnover_cost / chosen_code / horizon_index / sample_id / final_position`。
- 多并行 env: `num_envs` 个 `HorizonEnv` 实例对应 `num_envs` 个互不重叠的时间分片和独立时间游标，按训练数据顺序前进，不能 reshuffle；分片边界必须 `done=True` 并按配置 reset 仓位。

### 4.9 `src/trading/horizon_factory.py`

```python
def make_horizon_env(
    pair: str,
    split: str,
    horizon_index: pl.DataFrame,
    horizon_inputs_provider: HorizonInputsProvider,
    phase1_policy: Phase1FrozenPolicy,
    cost_config: CostConfig,
    max_position: int,
    seed: int,
    walk_forward: bool,
) -> HorizonEnv: ...
```

负责把 `cost_config` 转成 `LobDepthCostModel` + `RewardAlignment` + `TradingEnv`，再注入 `HorizonEnv`。`horizon_inputs_provider` 是唯一的数据注入路径，用来把 `phase2_dataset.py` 输出的切片恢复/包装成 `TradingEnv.reset()` 需要的 `HorizonInputs`。trainer 与 backtest 都通过 factory 创建 env，避免环境配置在两处分叉。

### 4.10 `src/trainers/phase2_trainer.py`

职责:

- 加载 Phase I 产物并校验 sign-off。
- 构造 `phase2_horizon_index_*.feather` 与 `Phase2Dataset`。
- 创建 `num_envs` 个 `HorizonEnv` 与 `Phase1FrozenPolicy`。
- 实例化 `ArchetypeSelector` 与 `PPOTrainer`。
- rollout / update 循环；按 PPO update 步触发 validation evaluator。
- 调用 `Phase2SelectionPolicy.evaluate(metrics)` 决定 best checkpoint。
- best checkpoint 冻结后调用 `Phase2BacktestRunner` 在 train/val/test 上导出 per-horizon records，并在 test 上做一次最终 walk-forward backtest。

边界:

- trainer 不直接计算指标，只消费 `Phase2Evaluator` 输出。
- trainer 不直接写 checkpoint 文件，统一交给 `Phase2CheckpointManager`。
- trainer 不直接实现 PPO update，统一交给 `PPOTrainer`。
- 若 `early_stopping.enabled=true`，trainer 必须基于 `selection_policy.metric` 在 validation 维度执行 patience 计数；即使默认关闭，也必须在训练结束后输出 `hypothetical_early_stop_timestep`（按同一 patience 规则离线回放 best 验证轨迹得到），供审计训练是否明显过长。
- trainer 必须输出 `convergence_diagnostics`：至少包含 `approx_kl`、`clip_fraction`、`explained_variance` 与 `kl_early_stop_count` 的时间序列；当 `approx_kl < 0.001` 连续 N 个 update 时写入 `risk_health_warnings`。

### 4.11 `src/trainers/phase2_checkpoint.py`

仿 Phase I `phase1_checkpoint.py`:

- 原子写 `best_selector.pt / last_selector.pt / checkpoints/step_*.pt`。
- 维护 `phase2_checkpoint_manifest.json`，记录每个 checkpoint 的 timestep、metrics、verdict（`best/rejected/periodic`）、Phase I batch id 与 hash。
- 不嵌入 best 选择规则。
- checkpoint 必须支持 resume：除 selector 参数外，还要保存 optimizer/scheduler state、PPO update counter、`selector_old` 参数、input/reward normalizer stats、每个 env 的 `horizon_cursor / prev_terminal_position / decoder_recurrent_state`，以及随机数种子状态。
- `last_selector.pt` 对应的 manifest 必须写入 `resume_ready=true/false` 与缺失项列表；若缺少任何恢复关键字段，不得宣称可断点续训。

### 4.12 `src/trainers/phase2_selection_policy.py`

仿 Phase I `phase1_selection_policy.py`，集中:

- `phase2_composite_score` 加权计算；
- 风险 guardrail（`max_drawdown / min_sharpe / max_turnover_ratio`）；
- 行为 guardrail（`max_action_dominance_ratio / min_active_archetype_ratio`）；
- KL/demo diagnostic（`val_kl_to_demo / warn_kl_to_demo`），只写 warning，不参与 best/reject；
- KL early-stop 与 dead-code mask 兼容性检查；
- 拒绝原因写入 manifest。
- `selection_policy.metric_weights` 必须支持 per-asset override；若沿用全局固定权重，report 至少要输出每个 asset 的 `composite_weight_sensitivity`，说明 checkpoint 排名是否对权重扰动敏感。

### 4.13 `src/evaluation/phase2_evaluator.py`

职责:

- 在 train/val horizon 上 freeze selector；best/sign-off 使用 argmax rollout，stochastic rollout 只用于 seed pack 诊断。
- 调度 `phase2_replay.py` 做 walk-forward backtest。
- 计算 selection-specific 指标（selector entropy、archetype 使用分布、KL to demo、turnover）。KL/demo 仅在 train/val 标注 horizon 上计算，不能读取 test label。
- 计算真实交易场景指标（净收益、Sharpe、Sortino、MDD、Calmar、turnover ratio、cost paid）。
- 必须同时输出 `train_metrics`（至少 `train_net_return / train_sharpe_ratio / train_action_entropy / train_action_distribution`），用于诊断 train/val gap；这些指标只用于过拟合分析，不参与 best 选择。
- 必须输出 `kl_label_temporal_coverage` 与 `per_env_reward_stats`，供 label coverage 偏置和多 env 非平稳性诊断。
- 提供 fixed probe rollout 给 diagnostics 使用。

子模块拆分（与 Phase I 类似）:

| 模块 | 职责 |
| --- | --- |
| `phase2_evaluator.py` | 调度 + 聚合 |
| `phase2_replay.py` | walk-forward replay、单 horizon replay、edge-case replay |
| `phase2_metrics.py` | 门面 + 稳定 API |
| `metrics/selection.py` | selector 行为指标（entropy / KL / dominance） |
| `metrics/portfolio.py` | 净收益 / Sharpe / Sortino / MDD / Calmar / turnover ratio / cost paid |
| `metrics/policy_health.py` | dead-code usage / mask 命中率 / clip fraction / approx_kl |
| `diagnostics/selector_visualization.py` | TensorBoard scalars + selector 决策可视化 |
| `diagnostics/phase2_failure_case_report.py` | walk-forward 中亏损最严重 / regret 最大 / cost 最高的 horizon HTML 错题本 |
| `phase2_report.py` | `phase2_report.json` schema 与原子写入 |

### 4.14 `src/evaluation/phase2_replay.py`

```python
class Phase2BacktestRunner:
    """Walk-forward backtest.

    流程
    ----
    1. 从 test 文件按 non_overlap stride 枚举 horizon。
    2. 串行执行: 每个 horizon
       reset(prev_terminal_position) -> selector(s^sel) -> code_id
       -> Phase1FrozenPolicy.decode_step(...) streaming 生成 base_actions
       -> TradingEnv.replay(base_actions) -> step_rewards / cost / final_position
    3. 累加 equity curve / position curve / cost curve。
    4. 输出真实交易指标与 per-horizon record。

    禁止
    ----
    - 调用 DP；
    - 调用 encoder.pt 决定 code_id；
    - 在 test 上读取 `horizon_labels_test.feather` 后再改变 checkpoint、配置、阈值或 guardrail；
    - 在 horizon 内修改 base_actions（Phase III 的工作）。
    """
```

### 4.15 `src/evaluation/phase2_report.py`

`phase2_report.json` 字段大类:

- 训练统计: total_timesteps / wall-clock / approx_kl / clip_fraction / explained_variance / lr 曲线。
- selector 行为: action 分布、entropy、KL to demo、dead-code mask 命中、code switch frequency between horizons。
- horizon-level 收益: train/val/test 的 per-horizon r^sel 分布、累计净收益。
- 真实交易指标: net_return / annualized_return / sharpe / sortino / max_drawdown / calmar / turnover_ratio / cost_paid / num_horizons / num_trades。
- 与 baseline 对照: vs random selector / vs single-archetype baselines（每个 code 单独 lock 后 walk-forward） / vs buy-and-hold；DP teacher 与 demo-label baseline 只能作为 posthoc hindsight 审计字段。
- 跨 horizon 边界: boundary_turnover_cost / boundary_position_consistency。
- Phase I 链路: phase1_batch_id / phase1_checkpoint_hash / phase1_config_hash / hindsight_warning_inherited。
- guardrail / sign-off: pass/fail 与原因。
- 训练协议审计: `kl_demo_signal_type`、`kl_demo_dominance_ratio`、`kl_label_temporal_coverage`、`input_norm_stats_merge_protocol`、`resume_ready`、`hypothetical_early_stop_timestep`、`rolling_validation_summary`、`execution_stress_summary`、`distribution_shift_warning_count`、`live_risk_trigger_count`、`deployment_readiness`。
- 多 env 分片审计: `env_shard_mode`、每个 env 的 reward/action 分布、是否触发 regime imbalance warning。

## 5. Selection MDP 详细设计

### 5.1 状态空间

论文严格形式:

$$
s^{sel}_t = s_t
$$

工程实盘默认形式:

```text
s^sel_live = [s_t, prev_terminal_position]
```

其中 `s_t` 是 horizon 第一根 bar 的 market state vector，与 Phase I encoder 输入第 0 步同源同 schema；`prev_terminal_position` 是上一 horizon 真实执行后的账户状态。可选增强形式:

```text
s^sel_extended = [s_t, prev_terminal_position, past_lookback_pool(s_{t-L:t-1}), one_hot(prev_a^sel), recent_pnl_window]
```

除 `prev_terminal_position` 之外，任何扩展都必须在 `phase2_config.yaml.state_extension` 中显式配置，并在 `phase2_report.json` 中标注。严格论文复现禁止扩展，并必须使用独立 flat horizon 或在报告中标注 POMDP 风险。

### 5.2 动作空间

$$
a^{sel} \in \{0, 1, \dots, K-1\}
$$

`K` 与 Phase I `model.num_codes` 严格一致。

Action mask（工程增强）:

- 默认 `selector_network.action_mask_dead_codes=True`。
- mask 来源: Phase I `phase1_report.json.code_usage` 中 usage ratio < `dead_code_usage_threshold` 的 code。
- mask 生效方式: 把对应 logit 设为 `-inf`，再 `log_softmax`；KL/demo label 若指向 mask 的 code（极端情况），KL term 该样本置 0 而不是把 mask 取消。
- 严格论文复现关闭 mask。

### 5.3 转移与回报

```text
t = horizon_index k 的起点行号
a^sel = π_φ^sel(s^sel)
codebook[a^sel] -> Phase1FrozenPolicy
  -> streaming base_actions a^base_{t:t+h-1}
TradingEnv.reset(initial_position=prev_terminal_position)
TradingEnv.replay(a^base)
  -> step_rewards[t:t+h-1]
  -> final_position = position[t+h-1]
r^sel_k = sum(step_rewards)
prev_terminal_position_{k+1} = final_position
```

回合（episode）切分:

- 正式训练默认使用非重叠 horizon 序列；同一 env 内 `done=True` 当且仅当当前 horizon 是该 env 时间分片末段或 walk-forward 末端。
- 跨 horizon 的 advantage / return 只允许在非重叠、时间正序、仓位连续的序列上由 GAE 计算（不强制 done=True）。
- 若使用 `stride < h` 或 `phase1_index` 中存在重叠窗口，必须把每个 horizon 视为独立 episode: `done=True`、`prev_terminal_position=0`、不继承仓位、不做跨 horizon GAE，并在 report 标注 `dense_overlap_ablation=true`。
- `prev_terminal_position` 在 reset 之间持续传递；若启用 multi-env 训练，每个 env 各自维护独立的 `prev_terminal_position`，不能跨 env 共享或乱序。
- `rollout_length` 只是 PPO buffer 截断长度，不是 episode 边界；rollout 收满时必须用最后一个 obs 的 critic value bootstrap，不能因为 buffer 满而 reset `prev_terminal_position`。

### 5.4 奖励对齐

`r^sel_k = Σ_τ r^step_τ`，其中 `r^step_τ` 必须由 Phase I `TradingEnv` 与 `LobDepthCostModel` 计算，禁止 Phase II 重写。

成本必须包含手续费与盘口逐档滑点。Phase II 不得为了"更平稳"切换 reward_alignment、降低 commission 或改用 `fixed_bps`，否则 selector 学到的策略与真实部署不一致。

### 5.5 跨 horizon 仓位继承

Phase I §11 已明确 selector 必须支持 `initial_position != flat`。Phase II 实现要求:

- `HorizonEnv.reset(prev_terminal_position)` 必须把上一段 horizon 的末仓位接入 `TradingEnv.reset(initial_position=...)`。
- 第一步 target_position（来自 decoder）与 inherited 不一致时，`LobDepthCostModel` 自动扣除换仓成本，selector 该 horizon 的 reward 必然反映边界成本。
- 当 `position_continuity=true` 时，`prev_terminal_position` 必须进入 selector/critic state；否则启动失败，除非显式 `paper_strict_reproduction=true` 且每个 horizon 从 flat reset。
- 训练数据按时间正序串行；不允许 reshuffle horizon 顺序，否则 `prev_terminal_position` 会被错误填充。
- multi-env 之间互不共享 `prev_terminal_position`；每个 env 独立维护时间游标。
- walk-forward backtest 必须 single-env 串行执行，确保仓位继承严格按真实时间。

### 5.6 折扣因子与 horizon-level γ

论文公式 (5) 用 `γ^t r_t^sel`。工程实现:

- `γ` 的语义是 horizon-level 折扣（不是 step-level）。
- 默认 `γ=0.99`；可调，但必须写入配置与 report。
- step-level reward 已经在 horizon 内累加完成；`γ` 只影响 horizon 之间的 advantage 计算。

### 5.7 KL/demo 项

论文目标:

$$
J = \mathbb{E} \sum_t \gamma^t \left[ r_t^{sel} - \alpha \mathrm{KL}\!\left(\hat{a}_t^{sel} \,\|\, \pi_\phi^{sel}(\cdot | s_t^{sel})\right) \right]
$$

其中 `â_t^sel` 是 Phase I VQ encoder 给该 horizon 分配的 archetype id。该值是单一类别标签，不是分布。

#### 5.7.1 KL/demo 信号性质声明

`â_t^sel` 是 **hindsight regularization label**，不是线上可观测状态:

- `code_label` 由 Phase I VQ encoder 生成，encoder 输入包含完整 horizon 的 `(s_demo, a_demo, r_demo)`。
- `a_demo / r_demo` 来自 Phase I DP demonstration；DP planner 通过 backward iteration 使用 horizon 内未来价格来构造 teacher trajectory。
- 因此 `â_t^sel` 编码了 horizon 内部的未来信息。
- 这不属于 Phase II 推理泄漏: selector 在 rollout、validation、test 和线上推理时只能看到 `s^sel_t` 与当前账户状态，不能读取 `code_label`、DP 或 encoder 输出。
- 但它属于 supervised learning 里的 hindsight label。Phase II 的泛化能力必须主要由在线 reward / cost 信号和 validation walk-forward 证明，不能把 KL/demo 贴合程度解释为预测能力。

设计约束:

- KL/demo 项只允许作为训练期 regularizer 和诊断指标。
- `val_kl_to_demo` 不得进入 `phase2_composite_score`、hard guardrail、best checkpoint 选择或 sign-off。
- report 必须写入 `kl_demo_signal_type="hindsight_regularization"`、`kl_label_coverage_train`、`kl_label_coverage_val` 与 `val_kl_to_demo`。
- 必须把 `kl_demo_coef` 视为高风险超参数；正式实验至少补跑 `α ∈ {0, 0.1, 0.5, 1.0}` 的消融，并在 report 的 validation 对照表中输出在线指标差异，而不只输出 `val_kl_to_demo`。
- report 必须输出 `kl_demo_dominance_ratio = kl_demo_loss / (policy_loss + 1e-8)` 的时间序列与分位数；当该比例连续超过阈值时写入 `behavior_health_warnings`，提示 selector 正在退化为 encoder label imitator。
- `kl_demo_anneal_to=0` 的训练后期去模仿 baseline 必须作为必跑对照之一；默认值仍可保持 `None` 以对齐论文，但正式 sign-off 报告不能缺少这个 baseline。

工程等价:

$$
\mathrm{KL}(\hat{a}_t^{sel} \| \pi_\phi) = -\log \pi_\phi(\hat{a}_t^{sel} | s_t^{sel}) + \text{const}
$$

实现:

- `kl_demo_loss = -log_pi[label]`，等价于 cross-entropy。
- 可选 label smoothing `kl_demo_label_smoothing > 0` 时把硬 label 平滑为 `(1-ε) δ_label + ε U(K)`，KL 化为分布形式。严格论文复现关闭 smoothing。
- 未标注 horizon 的 sample 必须 mask 掉 KL term（`is_labeled=False`），否则会引入零样本噪声。
- KL coef `α` 默认 `1.0`，可线性退火到 `kl_demo_anneal_to`，避免训练后期模仿信号继续主导自由探索。退火前后必须写入 `phase2_report.json`。

### 5.8 Episode 终止

- 默认按真实时间分片终止: 只有到达该 env 的时间分片末端、split 末端或显式独立 horizon ablation 时 `done=True`。
- 每 `rollout_length` 个 horizon 只是 PPO rollout truncation: `truncated=True`、`done=False`，计算 GAE 时用 `last_value` bootstrap，且下一轮 rollout 继续沿用同一 env 的 `prev_terminal_position`。
- walk-forward backtest 中: episode 不强制终止，只在 test 末段 `done=True`，确保跨 horizon 仓位继承贯穿整段 test。
- 默认 `chunk_reset_position="inherit"`，即 rollout chunk / buffer 截断后继续继承上一 chunk 末仓位，与 walk-forward 一致。
- 若某个实验显式启用 `chunk_reset_position="flat"`，必须标注 `paper_strict_reproduction=true` 或 `position_continuity=false`，并不得作为实盘连续仓位 sign-off；report 必须输出 `chunk_reset_distribution_shift`，量化 reset 为 flat 的 horizon 占比、reset 前真实仓位分布和因此少扣/多扣的边界成本估计。

### 5.9 多 Env 时间分片

`num_envs > 1` 时，每个 `HorizonEnv` 必须只处理一个互不重叠的连续时间分片:

1. 先在 train horizon index 上按 timestamp 升序生成非重叠 horizon 序列。
2. 设有效 horizon 数为 `M`，按连续区间均分为 `num_envs` 段；每段长度差最多 1 个 horizon。
3. 每段边界必须落在 horizon 边界上，不允许把同一个 horizon 拆给两个 env。
4. 每个 env 独立维护 `cursor / prev_terminal_position / recurrent_state / local_running_stats`。
5. env 分片末端返回 `done=True`；PPO GAE 不得跨 env 或跨分片 bootstrap。

分片健康检查:

- `phase2_env_shards.feather` 必须记录 `env_id / start_sample_id / end_sample_id / start_timestamp / end_timestamp / num_horizons / start_position_policy`。
- report 必须输出每个 env 的 horizon 数、时间跨度、reward mean/std、action distribution，用于检查分布是否严重不均衡。
- 若任一 env 的 `num_horizons < rollout_length` 或 reward 分布相对全局偏离超过配置阈值，写入 `risk_health_warnings`。
- `env_shards.mode="round_robin"` 必须作为可选 ablation：把按时间排序后的 horizon 轮转分配给不同 env，用于检验连续分片是否造成单 env 只看到单一 regime 的偏置；该模式不得替代默认连续分片，但报告中应提供对照。

## 6. Selector 网络与策略设计

### 6.1 主干

```text
state_dim
  -> input_norm (LayerNorm 或 RunningMeanStd 二选一)
  -> MLP(hidden_dim, num_layers, residual=False, activation=GELU, layer_norm=True)
  -> trunk_output (hidden_dim)
```

`input_norm` 的选择必须固定: 默认 `LayerNorm`，避免 PPO rollout 和 evaluation 中的 running stats 漂移；启用 `running_mean_std` 时必须按下述协议同步 freeze stats 后再做 backtest。

#### 6.1.1 输入预处理

Phase II 禁止在 train/val/test 上额外拟合任何 scaler、standardizer 或全局 normalizer。selector 输入预处理只允许使用当前样本内的确定性编码和网络内部 normalization:

| 输入块 | 处理方式 | 说明 |
| --- | --- | --- |
| `market_features = s_t` | 原值输入 selector trunk；第一层使用 `LayerNorm(elementwise_affine=True)` | `feature_columns` 与 Phase I 完全一致。LayerNorm 是样本内归一化，不拟合跨时间统计量。 |
| `prev_terminal_position` | 默认 `one_hot_3=[short, flat, long]` | 编码映射为 `short=[1,0,0]`、`flat=[0,1,0]`、`long=[0,0,1]`；若 asset 有多档仓位，必须改为 `scaled_integer=position/max_position` 并单独标注。 |
| `past_lookback_pool.mean/std` | 只用 `t-L:t-1` 当前样本内窗口计算；拼接后进入同一个 LayerNorm trunk | 不允许对 lookback 特征再做 train split scaler。 |
| `archetype_usage_history` | 过去 W 个已执行 selector action 的 one-hot count 或 exponentially decayed frequency | 仅使用过去已完成 horizon；归一化为 `[0,1]` 频率。 |
| `recent_pnl_window` | 过去 W 个已完成 horizon 的 raw/scaled PnL summary；默认关闭 | 若启用，scale 只能用固定常数或当前窗口内 robust statistic，不得拟合全局 scaler。 |

`state_dim` 计算必须写入 `phase2_config.yaml` 和 report:

```text
state_dim = len(feature_columns)
          + position_encoding_dim
          + optional_past_lookback_dim
          + optional_archetype_usage_dim
          + optional_recent_pnl_dim
```

验收:

- `position_continuity=true` 时，`position_encoding_dim > 0`。
- `position_encoding=one_hot_3` 只允许三状态仓位；多档仓位必须显式切换为 `scaled_integer` 或更完整的 bucket encoding。
- 任何输入预处理若需要跨样本统计量，必须移入 `feature_provenance.json` 并满足 train-only fit；Phase II selector 层不得自行 fit。

`RunningMeanStd` 协议（仅 ablation，正式 sign-off 默认禁止）:

- `RunningMeanStd` 只能在 `paper_strict_reproduction=false` 的独立 BATCH_ID 中启用。
- 训练 rollout 中必须按 env 内时间正序在线更新: 对 horizon `t` 做归一化时，统计量只能来自同一 env 已完成的历史 observation，不得包含当前 horizon 之后的数据。
- `input_norm_stats_merge_protocol` 必须写入 report。正式 sign-off 允许的默认协议只有两种: `independent_per_env` 或 `delayed_merge_exchangeable_stats`。
- `delayed_merge_exchangeable_stats` 的精确定义必须锁定为: 只允许合并 `count / mean / m2(sum_of_squares)` 等可交换统计量，且合并结果 **只能在下一个 rollout** 生效；当前 rollout 内任何 env 都不能看到其他 env 更晚时间段带来的统计量。
- 更保守的正式方案是 `independent_per_env`：多 env 完全独立维护 RunningMeanStd，不做跨 env 合并；若实现复杂度或审计成本过高，正式 sign-off 应优先回退到该模式。
- validation/test/backtest 前必须 freeze stats，并在 `phase2_report.json.input_norm_stats` 记录 freeze timestep、count、mean/std hash。
- 若实现无法保证以上时序协议，启动时必须拒绝 `input_norm="running_mean_std"`。

### 6.2 Actor head

```text
trunk_output -> Linear(actor_head_hidden) -> GELU -> Linear(K)
  -> mask -> log_softmax
```

输出 `log_pi`，`pi = exp(log_pi)`。采样:

- 训练 rollout: stochastic（`Categorical(probs).sample()`），保证 PPO importance sampling 有效。
- 评估 rollout: best/sign-off 主路径必须使用 `argmax`，避免 checkpoint 选择被采样噪声主导；stochastic 只作为诊断，必须用固定 seed pack 输出均值、标准差和最差分位。
- `HorizonEnv` 与 `Phase2Inferencer` 必须显式接收 `deterministic: bool` 或等价 flag；训练路径默认 `False`，validation/test walk-forward 主路径强制 `True`。

### 6.3 Critic head

```text
trunk_output -> Linear(critic_head_hidden) -> GELU -> Linear(1)
```

输出 `V(s^sel)`，参与 GAE。

### 6.4 Archetype embedding fusion（可选）

严格论文复现关闭。启用时:

```text
codebook ∈ R^{K x code_dim}
score_k = MLP([trunk_output, codebook[k]]) -> logit_k
```

理论收益: 让 selector 显式 condition on archetype 内容，提升 K=10 之外的扩展性。风险: 与论文公式不一致，必须独立 BATCH_ID。

### 6.5 Action mask 行为细则

- mask 输入: `dead_code_mask = Phase1.global_code_usage_ratio < dead_code_usage_threshold`；阈值必须基于 Phase I 全局 train demonstration usage，而不是 Phase II 训练子集的使用频率。
- 训练采样: mask 后 `log_softmax`，sample 的 action 永远不会是 dead code。
- KL/demo: 若 `code_label` 落在 dead code（说明 Phase I checkpoint 与 mask 阈值不一致），将该样本 KL term 置 0 并在 `phase2_report.json.behavior_health_warnings` 中记录。
- 评估: 评估 reward / Sharpe 时 mask 一致；不允许评估期临时关闭 mask "刷分"。
- 训练初期必须允许一次 diagnostic rollout 在不启用 mask 的条件下运行，用于观察 selector 是否自发选择被标记为 dead 的 code；若发生，报告必须在 `dead_code_mask_summary` 中记录 `code_id / phase1_usage / selector_probe_pick_rate`，并提示该 dead code 在当前 Phase II 时间段内可能仍有价值。

### 6.6 时间因果与 selector 输入

selector 在 horizon 起点决策，输入只包含截至 `t` 行可见的特征。任何 `state_extension` 必须满足:

- 不读 `t+1` 及之后的任何行；
- past_lookback 严格只用 `t-L` 至 `t-1`；
- archetype usage history 只用过去已经被 selector 决定并执行完的 horizon。

实现时禁止以下"快捷":

- 把 `phase1_horizon_labels` 直接拼进 `s^sel`（这会让 selector 直接看到 oracle）；
- 把 horizon 内未来行的 mid_price / volatility 作为 selector 输入；
- 把 walk-forward 后续 horizon 的 `prev_terminal_position` 提前注入。

## 7. PPO 训练算法与损失

### 7.1 总损失

$$
L_{PPO} = L_{policy}^{clip} + c_v L_{value} - c_e L_{entropy} + \alpha L_{kl\text{-}demo}
$$

其中:

- `L_policy^clip = -E[min(ρ A, clip(ρ, 1-ε, 1+ε) A)]`，`ρ = π_φ(a|s) / π_φ_old(a|s)`。
- `L_value = E[(V_φ(s) - V_target)^2]`，启用 value clip 时再做一层 clip。
- `L_entropy = E[H(π_φ(·|s))]`。
- `L_kl-demo = E_{is_labeled}[ -log π_φ(â_t^sel | s_t^sel) ]`。

KL early-stop: 每个 update epoch 末计算 `approx_kl`，超过 `target_kl` 时提前停止该 minibatch 循环；同时把 `kl_early_stop_count` 写入 report。

### 7.2 GAE

$$
\delta_t = r_t + \gamma V(s_{t+1}) (1 - done_t) - V(s_t)
$$

$$
A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

`λ = gae_lambda = 0.95` 默认。

advantage normalization 在每个 minibatch 内执行: `A = (A - mean) / (std + 1e-8)`，严格论文复现关闭。

### 7.3 Rollout buffer

字段:

| 字段 | dtype | 说明 |
| --- | --- | --- |
| `obs` | float32[N, T, state_dim] | horizon 起点状态 |
| `env_id` | int64[N, T] | 并行环境 ID；GAE 必须按 env 分组 |
| `action` | int64[N, T] | selector 选择的 archetype id |
| `log_prob` | float32[N, T] | 旧策略 log_prob |
| `value` | float32[N, T] | critic 估值 |
| `reward` | float32[N, T] | r^sel |
| `done` | bool[N, T] | 真实 episode 终止（时间分片末端、split 末端或独立 horizon ablation） |
| `truncated` | bool[N, T] | rollout buffer 截断；用于 bootstrap，不表示真实 episode 结束 |
| `kl_label` | int64[N, T] | â_t^sel；未标注样本任意值 |
| `is_labeled` | bool[N, T] | KL/demo mask |
| `dead_code_mask` | bool[N, T, K] | 训练期 mask 快照 |
| `info_cost_paid` | float32[N, T] | 用于评估 |
| `info_boundary_cost` | float32[N, T] | 用于评估 |
| `info_chosen_code` | int64[N, T] | 与 action 等价；冗余便于审计 |

`N=num_envs, T=rollout_length`。

`done=False, truncated=True` 时，GAE 必须使用下一 obs 的 value bootstrap；只有 `done=True` 时才切断 bootstrap。

多 env GAE 协议:

- GAE 必须按 `env_id` 分组独立计算，不允许把不同 env 的 transition 拼成一条时间序列。
- 每个 env 的 rollout buffer 维度保持 `[T]` 时间顺序；计算完该 env 的 advantages / returns 后才允许 flatten 合并成 minibatch。
- env 时间分片边界必须 `done=True`；跨 env 边界天然视为 episode boundary。
- 单元测试必须构造两个 env reward/value 方向相反的 fixture，证明跨 env 混算会失败、按 env 分组才通过。

### 7.4 Update 步骤

```text
for update in range(num_updates):
    rollout: 用 selector_old 采集 N*T 个 transitions
    compute advantages, returns
    for epoch in range(update_epochs):
        for minibatch in shuffle(rollout):
            log_pi, value, entropy = selector.evaluate(minibatch.obs, minibatch.action)
            ratio = exp(log_pi - minibatch.log_prob)
            surr1 = ratio * advantage
            surr2 = clip(ratio, 1-clip_range, 1+clip_range) * advantage
            policy_loss = -mean(min(surr1, surr2))
            value_loss = 0.5 * mean((value - returns)^2)
            entropy_loss = -mean(entropy)
            kl_demo_loss = sum((-log_pi_label) * is_labeled) / max(sum(is_labeled), 1)
            loss = policy_loss + value_loss * c_v + entropy_loss * c_e + kl_demo_loss * alpha
            backprop, clip grad, step
            if approx_kl > target_kl: break
    update selector_old <- selector
```

### 7.5 Schedule

- `learning_rate`: `linear` 从 `lr_init` 退火到 `lr_init * lr_floor_ratio`，默认 `lr_floor_ratio=0.0`。
- `clip_range`: 可线性退火（默认关闭）。
- `entropy_coef`: 可线性退火（默认关闭）。
- `kl_demo_coef`: 默认 `1.0` 恒定；启用退火时按 `kl_demo_anneal_fraction` 完成 `1.0 -> kl_demo_anneal_to`。

退火参数必须写入 `phase2_report.json.schedule`，便于复现。

训练健康监控补充:

- 每次 PPO update 必须记录 `policy_loss / value_loss / entropy / kl_demo_loss / kl_demo_dominance_ratio`。
- 当 `kl_demo_dominance_ratio` 超过阈值（默认例如 `>1.0`，具体阈值配置化）连续出现时，写入 `behavior_health_warnings`，并建议回看 `kl_demo_coef`、退火速度与 label coverage 偏置。

`ppo.reward_normalization` 协议:

- 默认 `False`，正式 sign-off 禁止启用。
- 若作为 ablation 启用，reward running mean/std 必须按 env 内时间正序在线更新；计算 horizon `t` 的 normalized reward 时，只能使用该 env 已完成的历史 reward 统计。
- validation/test 不允许更新 reward normalizer，只能使用训练结束 freeze 的 stats。
- report 必须写入 `reward_normalization_enabled`、`reward_norm_freeze_timestep`、`reward_norm_count` 和 stats hash。

### 7.6 Horizon Reward Scaling / Clipping

`r^sel = Σ_{τ=t}^{t+h-1} r^step_τ` 是 horizon 级累计 reward，量级通常显著大于单步 reward。为降低 PPO 数值不稳定风险，Phase II 在送入 rollout buffer 前应用固定、可复现的 reward scaling:

```text
raw_horizon_reward = sum(step_rewards)
scaled_reward = scale(raw_horizon_reward)
```

默认配置:

```yaml
reward_scaling:
  mode: divide_by_horizon       # none | divide_by_horizon | constant
  constant_scale: 1.0
  clip_range: null              # 例如 10.0；正式主实验默认不 clip
```

语义:

- `divide_by_horizon`: `scaled_reward = raw_horizon_reward / h`，默认启用，使 reward 量级接近 step reward 平均值。
- `constant`: `scaled_reward = raw_horizon_reward * constant_scale`，只作为 ablation。
- `clip_range != null` 时，`scaled_reward` 被裁到 `[-clip_range, clip_range]`；正式 sign-off 若启用 clip，必须在 report 中标注并同时报告 unclipped reward 指标。
- portfolio metrics、equity curve、net_return、Sharpe、MDD 永远使用 raw step reward / raw PnL 计算；reward scaling 只影响 PPO training signal。

report 必须记录:

- `reward_scaling.mode / constant_scale / clip_range`。
- train rollout 的 `raw_horizon_reward` 与 `scaled_reward` 分布: mean / std / min / max / p01 / p05 / p50 / p95 / p99。
- 若启用 clipping，记录 `reward_clip_ratio` 与被 clip 的 top-K horizon sample ids。

### 7.7 与 Phase III 的预留

PPO trainer 必须保留 `step_action_callback` 钩子，未来 Phase III refinement 可以在 selector 选完 archetype 之后、decoder 执行前插入 step-level adapter。Phase II 当前实现里该 callback 默认不挂载，但接口必须存在。

## 8. 真实交易场景评估

### 8.1 评估时机

| 时机 | 数据 | 用途 |
| --- | --- | --- |
| 每 `validate_every_updates` 个 PPO update | val horizons | 用 argmax 计算快速指标，参与 best 选择 |
| best checkpoint 候选触发时 | val horizons | 用 argmax 跑完整 walk-forward replay，验证候选是否通过 guardrail |
| best checkpoint 冻结后 | test horizons | 用 argmax 跑一次主 backtest，并额外输出 stochastic seed pack 诊断；不得再改变 checkpoint、配置或阈值 |

### 8.2 Walk-forward 协议

- **顺序**: 严格按 `timestamp` 升序枚举 horizon。
- **stride**: 默认 `non_overlap`（stride = h），保证每个 minute 只参与一次 r^step；walk-forward 起点偏移默认 0。任何偏移/seed 对比只能在 val 上做，test 不允许调参。
- **仓位继承**: 串行，每个 horizon `reset(prev_terminal_position)`。
- **selector 推理**: best/sign-off 与最终 test 主结果使用 argmax；stochastic 诊断使用预注册 seed pack（默认 10 个 seed），写入 `stochastic_mean/std/p05/p95`，不参与 best。
- **DP 禁用**: 整个 train/val/test walk-forward 过程中不允许调用 DP；`code_label` 在 test 上仅可由 posthoc baseline 读取，且不能影响 sign-off。

### 8.3 Baseline 对照

| Baseline | 描述 | 用途 |
| --- | --- | --- |
| `random_selector` | 在 mask 后 K 个 code 中均匀采样，固定 seed pack 评估均值/置信区间 | 检验 selector 是否优于乱选 |
| `single_archetype_k` | 每个 code 单独锁定后 walk-forward | 暴露每个 archetype 的"裸"能力 |
| `phase1_demo_label` | 用 `code_label` 当 selector，看模仿基线 | posthoc hindsight 对照；不参与 best / guardrail / sign-off |
| `dp_teacher_offline` | DP 离线最优；仅用于 hindsight 对照 | 上限参考；不参与 best / guardrail / sign-off |
| `buy_and_hold_long / short` | 全仓 long/short 不交易 | 市场基准 |

`baseline_metrics_val` 可参与 validation 审计与 sign-off；`baseline_metrics_test` 只进入最终报告。任何含 DP 或 demo-label 的 baseline 都必须标注 `hindsight=true`。与 random selector 比较时，selector argmax 必须优于 random seed pack 的均值，并报告相对 p95 的差距作为稳健性诊断。

### 8.4 关键指标

#### 8.4.1 收益与风险

| 指标 | 说明 |
| --- | --- |
| `val_net_return / test_net_return` | walk-forward 累计净收益 |
| `val_annualized_return / test_annualized_return` | 按分钟级 step return 年化 |
| `val_sharpe_ratio / test_sharpe_ratio` | 年化 Sharpe (annualization_factor=525600) |
| `val_sortino_ratio / test_sortino_ratio` | 年化 Sortino |
| `val_max_drawdown / test_max_drawdown` | equity curve 最大回撤 |
| `val_calmar_ratio / test_calmar_ratio` | 年化收益 / 最大回撤 |
| `val_return_capture_ratio_vs_dp` | `selector_net_return / dp_teacher_net_return`，仅用于 posthoc 审计，不进入 composite score |
| `val_regret_to_dp` | DP teacher 与 selector 的差距，仅用于 posthoc 审计 |

#### 8.4.2 成本与换手

| 指标 | 说明 |
| --- | --- |
| `cost_paid_total` | walk-forward 中所有 step 的 `fee + slippage` 总和 |
| `cost_paid_per_horizon` | 平均每 horizon 成本 |
| `turnover_ratio` | `Σ |Δposition| / num_horizons` |
| `boundary_turnover_cost` | horizon 边界换仓平均成本 |
| `boundary_position_consistency` | 边界仓位一致比例 |
| `reject_transition_rate` | walk-forward 中 `LobDepthCostModel` 拒绝率，监控真实盘口可执行性 |

#### 8.4.3 Selector 行为

| 指标 | 说明 |
| --- | --- |
| `val_action_entropy / test_action_entropy` | selector 输出分布的平均 entropy |
| `val_action_distribution / test_action_distribution` | 每个 archetype 被选频率 |
| `val_action_dominance_ratio` | 最热 archetype 的占比 |
| `val_active_archetype_ratio` | 被选频率 ≥ `min_use_threshold` 的 archetype 占比 |
| `val_kl_to_demo` | 在标注 horizon 上的 KL/demo cross-entropy |
| `kl_label_coverage_train / kl_label_coverage_val` | 有效 KL/demo label 覆盖率；用于解释 KL loss 量级 |
| `val_archetype_switch_rate` | 相邻 horizon `a^sel` 不同的比例 |
| `dead_code_mask_hit_rate` | mask 过滤掉的 logit 比例 |

#### 8.4.4 PPO 训练健康

| 指标 | 说明 |
| --- | --- |
| `approx_kl` | 每 update 的近似 KL |
| `clip_fraction` | PPO ratio 落在 clip 区间外的比例 |
| `explained_variance` | critic 解释的回报方差比例 |
| `policy_grad_norm / value_grad_norm` | 梯度范数 |
| `kl_early_stop_count` | KL early-stop 触发次数 |

### 8.5 Composite score 与 sensitivity

`phase2_composite_score` 默认权重见 §4.3，只允许使用 validation 在线可执行指标，禁止使用 DP teacher、demo-label baseline、test metrics 或任何 hindsight oracle 字段。主实验完成后必须做权重 sensitivity 检验:

- 以 §4.3 默认 perturbations 列表跑权重 ±10%；
- 写入 `composite_score_sensitivity_phase2.json`；
- 若不同权重下 best checkpoint 显著漂移，`phase2_report.json.composite_weight_sensitivity_warning=true`。
- sensitivity 只基于 validation checkpoint manifest 重新计算，不重新训练、不读取 test。

### 8.6 Per-asset 与 cross-asset 实验

论文按 BTC / ETH / DOT / BNB 各自训练 selector。工程实现要求:

- 每对 `{PAIR, PHASE2_BATCH_ID}` 独立产物目录，独立 selector 权重。
- 共享 selector 跨 asset 的实验属于 ablation: 必须使用独立 BATCH_ID，并以 `phase2_config.cross_asset.enabled=true` 标注；selector 输入必须包含 `asset_embedding` 才允许混合训练。
- 性能对比报告必须按 asset 拆分。

### 8.7 成本压力测试与可执行性压力测试

正式 sign-off 不能只看 nominal cost。best checkpoint 冻结后必须在 `execution_stress` 预注册场景下重跑 val/test backtest，至少包括:

- `commission x 1.5`
- `slippage x 1.5 / 2.0`
- `execution_lag +1 bar`
- `reject_rate_injection`
- 可选 partial-fill / depth truncation

要求:

- `execution_stress_summary` 必须写入 `phase2_report.json`。
- 若 stress 场景下 `max_drawdown`、`turnover_ratio`、`reject_transition_rate` 或 `net_return` 出现灾难性崩坏，则该 run 不得部署。
- stress 结果用于部署准入，不回流 best checkpoint 选择；best 仍只由主 validation 协议决定。

### 8.8 OOD / regime-shift 监控与降级

除了常规 val/test 指标，系统必须监控 selector 输入是否偏离训练分布。最小要求:

- 使用 `distribution_shift.method` 对 horizon 起点状态做 OOD 检测；
- 输出 `distribution_shift_warning_count`、`ood_max_score` 与按时间分布的告警记录；
- 连续超过阈值达到 `trigger_consecutive_horizons` 时，触发 `fallback_mode`。

`fallback_mode` 默认 `flat_only`，表示新 horizon 只允许选择 flat-safe 行为或直接 no-trade；`risk_reduced` 模式下至少要降低可用仓位和 archetype 切换频率。

### 8.9 Rolling validation 协议

单一 val 区间不足以支撑正式 sign-off。除主 val walk-forward 外，正式版本必须额外执行 `rolling_validation`:

- `anchored_walk_forward`: 逐步扩大的训练前缀 + 多个时间后移 validation fold；或
- `multi_era_holdout`: 多个互不重叠 era 的固定 holdout。

`rolling_validation_summary` 至少包含:

- `fold_net_return`
- `fold_sharpe_ratio`
- `fold_max_drawdown`
- `fold_turnover_ratio`
- `fold_selection_score`
- `worst_fold_id`

正式 sign-off 默认使用 `mean_minus_std` 或 `worst_fold` 作为部署审计准则，避免 checkpoint 只适配单一市场阶段。

### 8.10 上线前部署阶梯

训练成功、val/test 通过，不等于可以直接实盘。正式部署前必须按顺序通过:

1. shadow replay：仅消费实时行情，不下单；
2. paper trading：完整走下单链路但不触发真实成交；
3. canary：按 `canary_position_scale` 小仓位上线；
4. full deployment：仅在 canary 无异常后开启。

`deployment_readiness` 必须记录每一层是否完成、开始/结束时间、触发的 warning 数和最终 verdict。

## 9. 训练流程

```text
phase1 artifacts
  -> Phase1ProductValidator 校验 sign-off
  -> phase2_horizon_index_{split}.feather 生成
  -> Phase2LabelLoader join train/val code_label
  -> HorizonEnv * num_envs 创建 (frozen Phase1FrozenPolicy)
  -> ArchetypeSelector + PPOTrainer 初始化
  -> rollout / GAE / PPO update 循环
       每 K updates: Phase2Evaluator 在 val 上跑快速 rollout
       best 候选: Phase2BacktestRunner 在 val 上 walk-forward
       SelectionPolicy.evaluate -> verdict
  -> composite score sensitivity 基于 val manifest 复算
  -> best checkpoint 冻结: Phase2BacktestRunner 在 train/val/test 上各输出 per-horizon records
  -> Phase2BacktestRunner 在 test 上 walk-forward 一次
  -> Phase2ReportWriter 写 phase2_report.json + 产物
```

详细步骤:

1. 校验 Phase I 产物完整性与 sign-off 状态。
2. 加载 train/val/test 数据并校验 schema 与 Phase I `input_schema.json` 一致。
3. 生成 `phase2_horizon_index_{split}.feather`，按 `HorizonScheduleConfig` 与 `RewardAlignment` 严格枚举 horizon；越界 horizon 必须裁掉。
4. join Phase I `horizon_labels_train.feather` 与 `horizon_labels_val.feather`，标注 `code_label / is_labeled`；训练入口不得读取 `horizon_labels_test.feather`。
5. 加载 `decoder.pt / codebook.pt` 至 `Phase1FrozenPolicy`，自检结构与 hash。
6. 构造 `num_envs` 个 `HorizonEnv` 实例，每个 env 拥有独立时间游标与 `prev_terminal_position`。
7. 实例化 `ArchetypeSelector` 与 `PPOTrainer`；实例化 `Phase2Evaluator`、`Phase2CheckpointManager`、`Phase2SelectionPolicy`。
8. 进入 PPO 训练循环:
   - 采集 rollout `num_envs * rollout_length` 个 horizon transitions。
   - 计算 GAE / returns / advantage normalize。
   - update selector `update_epochs * minibatches`，统计 PPO 健康指标与 `kl_demo_dominance_ratio`。
   - 每 `validate_every_updates` 在 val 上跑快速评估指标。
   - best 候选触发时跑完整 walk-forward，并交给 `Phase2SelectionPolicy.evaluate`。
   - 若 `early_stopping.enabled=true` 且验证指标在 patience 窗口内无改进，则提前停止并写入 manifest；若关闭，也必须在训练结束后回溯报告 `hypothetical_early_stop_timestep`。
9. 训练结束后，先基于 validation manifest 跑 `composite_score_sensitivity` 权重扰动实验并冻结 best checkpoint。
10. 冻结后最终在 test 上跑一次 walk-forward backtest（argmax 主结果 + stochastic seed pack 诊断）；test 结果只能写报告，不能回流到模型选择。
11. 输出 `phase2_report.json`、`phase2_checkpoint_manifest.json`、selector 产物与 diagnostics。

## 10. 输出产物

```text
artifacts/{PAIR}/{PHASE2_BATCH_ID}/phase2/
```

| 文件 | 作用 | 主要内容 |
| --- | --- | --- |
| `phase2_config.yaml` | 固化本次 Phase II 实验配置 | horizon schedule、PPO、selection policy、Phase I batch id 与 hash |
| `phase2_horizon_index_train.feather` | 训练 horizon 索引 | sample_id / start_index / end_index / phase1_sample_id / code_label / is_labeled / split |
| `phase2_horizon_index_val.feather` | 验证 horizon 索引 | 同上 |
| `phase2_horizon_index_test.feather` | 测试 horizon 索引 | sample_id / start_index / end_index / split；默认不含 `code_label` |
| `phase2_env_shards.feather` | 多 env 时间分片记录 | env_id / 时间边界 / horizon 数 / 起止 sample_id |
| `best_selector.pt` | best checkpoint（selector 权重 + critic 权重） | actor + critic state_dict |
| `last_selector.pt` | 最后 update 的 selector state | 用于断点恢复 |
| `checkpoints/step_*.pt` | 周期 checkpoint | 调试用 |
| `phase2_checkpoint_manifest.json` | checkpoint 验证与选择记录 | timestep、metrics、verdict、Phase I batch id |
| `phase2_rollout_stats.feather` | 每 update 的 PPO 健康统计 | approx_kl / clip_fraction / explained_variance / lr |
| `phase2_per_horizon_records_train.feather` | train 上 best selector 的 walk-forward 明细 | sample_id / chosen_code / r^sel / cost_paid / boundary_cost / final_position |
| `phase2_per_horizon_records_val.feather` | val walk-forward 单 horizon 明细 | sample_id / chosen_code / r^sel / cost_paid / boundary_cost / final_position |
| `phase2_per_horizon_records_test.feather` | test walk-forward 单 horizon 明细 | 同上 |
| `phase2_baselines_val.json` | val 上各 baseline 的关键指标 | random_selector / single_archetype_k / buy_and_hold；DP/demo-label 仅 posthoc |
| `phase2_baselines_test.json` | test 上各 baseline 的关键指标 | random_selector / single_archetype_k / buy_and_hold；DP/demo-label 仅 posthoc |
| `phase2_failure_cases_val.html` | val 错题本 | worst_return / largest_regret / largest_cost / unstable_archetype_switch |
| `phase2_failure_cases_test.html` | test 错题本 | 同上 |
| `selector_visualization/` | selector 决策可视化 | per-horizon action distribution、archetype 时间序列、KL 曲线 |
| `composite_score_sensitivity_phase2.json` | 权重 sensitivity 检验结果 | best timestep 漂移、核心指标差异 |
| `phase2_report.json` | 训练 + 评估汇总报告 | 见下文字段 |

`phase2_report.json` 关键字段:

```json
{
  "phase2_batch_id": "batch_001",
  "phase1_batch_id": "batch_001",
  "phase1_checkpoint_hash": "",
  "phase1_config_hash": "",
  "feature_provenance_hash": "",
  "config_hash": "",
  "paper_strict_reproduction": false,
  "no_leakage_signoff": true,
  "test_used_for_selection": false,
  "allow_phase1_hindsight_warning": false,
  "hindsight_warning_inherited": "",
  "horizon_schedule": {
    "mode": "non_overlap",
    "stride": 72,
    "position_continuity": true,
    "dense_overlap_ablation": false,
    "chunk_reset_position": "inherit",
    "data_gap_check_enabled": true,
    "max_allowed_gap_minutes": 5,
    "gap_horizons_dropped": true,
    "walk_forward_enabled": true
  },
  "data_gap_filter": {
    "train_gap_horizon_count": 0,
    "val_gap_horizon_count": 0,
    "test_gap_horizon_count": 0,
    "max_timestamp_gap_minutes": 0
  },
  "input_norm": {
    "mode": "layer_norm",
    "position_encoding": "one_hot_3",
    "state_dim_breakdown": {},
    "running_mean_std_enabled": false,
    "stats_frozen": true,
    "stats_hash": ""
  },
  "env_shards": {
    "num_envs": 8,
    "min_horizons_per_env": 0,
    "max_horizons_per_env": 0,
    "reward_distribution_warning": false
  },
  "reward_scaling": {
    "mode": "divide_by_horizon",
    "constant_scale": 1.0,
    "clip_range": null,
    "raw_horizon_reward_stats": {},
    "scaled_reward_stats": {},
    "reward_clip_ratio": 0.0
  },
  "reward_normalization": {
    "enabled": false,
    "stats_frozen": true,
    "stats_hash": ""
  },
  "cost_config_inherited": {},
  "ppo_total_timesteps": 3000000,
  "ppo_num_updates": 0,
  "ppo_health": {
    "approx_kl": [],
    "clip_fraction": [],
    "explained_variance": [],
    "kl_early_stop_count": 0
  },
  "selector_behavior": {
    "val_action_distribution": {},
    "test_action_distribution": {},
    "val_action_entropy": 0.0,
    "test_action_entropy": 0.0,
    "val_action_dominance_ratio": 0.0,
    "test_action_dominance_ratio": 0.0,
    "val_active_archetype_ratio": 0.0,
    "test_active_archetype_ratio": 0.0,
    "val_archetype_switch_rate": 0.0,
    "test_archetype_switch_rate": 0.0,
    "kl_demo_signal_type": "hindsight_regularization",
    "val_kl_to_demo": 0.0,
    "kl_label_coverage_train": 0.0,
    "kl_label_coverage_val": 0.0,
    "dead_code_mask_hit_rate": 0.0
  },
  "val_metrics": {
    "evaluation_action_mode": "argmax",
    "net_return": 0.0,
    "annualized_return": 0.0,
    "sharpe_ratio": 0.0,
    "sortino_ratio": 0.0,
    "max_drawdown": 0.0,
    "calmar_ratio": 0.0,
    "turnover_ratio": 0.0,
    "cost_paid_total": 0.0,
    "boundary_turnover_cost": 0.0,
    "boundary_position_consistency": 0.0,
    "reject_transition_rate": 0.0
  },
  "test_metrics": {
    "argmax": {},
    "stochastic_seed_pack": {
      "mean": {},
      "std": {},
      "p05": {},
      "p95": {}
    }
  },
  "baselines_val": {
    "random_selector": {},
    "single_archetype": {},
    "buy_and_hold_long": {},
    "buy_and_hold_short": {}
  },
  "baselines_test": {
    "random_selector": {},
    "single_archetype": {},
    "phase1_demo_label": {},
    "dp_teacher_offline": {},
    "buy_and_hold_long": {},
    "buy_and_hold_short": {}
  },
	  "phase2_composite_score": 0.0,
	  "best_checkpoint_timestep": 0,
	  "best_checkpoint_path": "best_selector.pt",
	  "selection_metric": "phase2_composite_score",
  "guardrails_pass": false,
  "val_guardrails_pass": false,
  "test_guardrails_pass_report_only": false,
  "chunk_reset_distribution_shift": {},
  "guardrails_reasons": [],
  "composite_weight_sensitivity_warning": false,
  "behavior_health_warnings": [],
  "risk_health_warnings": [],
	  "boundary_health_warnings": []
	}
	```

`guardrails_pass` 语义固定为:

```text
guardrails_pass = val_guardrails_pass && no_leakage_signoff && !composite_weight_sensitivity_warning
```

`test_guardrails_pass_report_only` 只说明冻结 checkpoint 在 test 上是否也满足同一风险阈值，不得反向改变 `guardrails_pass`、`best_checkpoint_timestep` 或任一训练配置。

## 11. 单元测试与集成测试

### 11.1 单元测试目录

```text
tests/unit/data/
  test_phase2_horizon_index.py
  test_phase2_label_loader.py
  test_phase2_dataset.py

tests/unit/models/
  test_archetype_selector.py
  test_phase1_frozen_policy.py

tests/unit/rl/
  test_rollout_buffer.py
  test_actor_critic.py
  test_ppo_loss.py
  test_ppo_trainer.py
  test_scheduling.py

tests/unit/trading/
  test_horizon_env.py
  test_horizon_factory.py

tests/unit/trainers/
  test_phase2_selection_policy.py
  test_phase2_checkpoint.py

tests/unit/evaluation/
  test_phase2_metrics_selection.py
  test_phase2_metrics_portfolio.py
  test_phase2_metrics_policy_health.py
  test_phase2_replay.py
  test_phase2_evaluator.py
  test_phase2_report.py
```

### 11.2 关键单元测试用例

| 测试 | 文件 | 关键不变量 |
| --- | --- | --- |
| `test_horizon_index_should_drop_markout_overflow` | `test_phase2_horizon_index.py` | 末尾 `last_markout_row > num_rows - 1` 的 horizon 必须被裁掉 |
| `test_horizon_index_should_drop_data_gap_horizon` | 同上 | horizon 内 timestamp gap 超阈值时必须标记并裁掉 |
| `test_horizon_index_aligns_with_phase1_when_mode_phase1_index` | 同上 | `phase1_index` 模式下的 sample 集合等于 Phase I sampled 集合 |
| `test_horizon_index_rejects_overlap_with_position_continuity` | 同上 | `stride < h` 且 `position_continuity=true` 时启动失败 |
| `test_label_loader_marks_unlabeled_horizons` | `test_phase2_label_loader.py` | 没有 `code_label` 的 horizon `is_labeled=False`，KL term 应被 mask |
| `test_label_loader_rejects_test_labels_outside_posthoc` | 同上 | 训练/验证流程请求 `split=test` label 时必须抛错 |
| `test_phase1_frozen_policy_outputs_are_causal` | `test_phase1_frozen_policy.py` | 修改 `s_{τ+1:}` 不改变 `base_actions[:τ+1]` |
| `test_phase1_frozen_policy_decode_step_matches_prefix_decode` | 同上 | streaming `decode_step()` 与批量 causal prefix 输出一致 |
| `test_phase1_frozen_policy_parameters_never_update` | 同上 | 任意 forward + backward 后 decoder/codebook 参数不变 |
| `test_archetype_selector_action_mask_blocks_dead_codes` | `test_archetype_selector.py` | dead-code mask 对应 logit 等于 `-inf`，sample 永不返回 dead code |
| `test_archetype_selector_log_softmax_consistency` | 同上 | `log_pi.exp().sum(dim=-1) == 1.0` |
| `test_archetype_selector_input_preprocess_position_one_hot` | 同上 | `prev_terminal_position` 默认编码为 short/flat/long one-hot |
| `test_archetype_selector_rejects_phase2_scaler_fit` | 同上 | selector 输入预处理不得拟合 train/val/test scaler |
| `test_rollout_buffer_gae_matches_reference` | `test_rollout_buffer.py` | GAE 输出与手算（小 fixture）一致 |
| `test_rollout_buffer_bootstraps_when_truncated_not_done` | 同上 | `truncated=True, done=False` 时必须用 `last_value` bootstrap 且不 reset position |
| `test_rollout_buffer_gae_is_grouped_by_env_id` | 同上 | 不同 env 的 transition 不得互相 bootstrap 或串接 advantage |
| `test_multi_env_shards_are_contiguous_and_disjoint` | `test_scheduling.py` | env 分片必须连续、互不重叠且落在 horizon 边界 |
| `test_reward_scaling_divide_by_horizon` | `test_ppo_trainer.py` | 默认 scaled reward 等于 raw horizon reward / h |
| `test_reward_scaling_reports_raw_and_scaled_stats` | 同上 | report 同时包含 raw/scaled reward 分布统计 |
| `test_ppo_loss_clip_outside_window` | `test_ppo_loss.py` | `ratio > 1+ε` 与 `< 1-ε` 都触发 clip |
| `test_ppo_loss_kl_demo_masked_for_unlabeled` | 同上 | `is_labeled=False` 的 KL term=0 |
| `test_ppo_loss_kl_demo_normalizes_by_labeled_count` | 同上 | KL loss 分母为 labeled count，而非整个 minibatch |
| `test_ppo_loss_kl_early_stop_triggers` | 同上 | `approx_kl > target_kl` 时返回 early-stop signal |
| `test_running_mean_std_updates_time_ordered` | `test_actor_critic.py` | RunningMeanStd 归一化 horizon `t` 时不能使用 `t+1` 后 observation |
| `test_reward_normalization_rejected_for_signoff` | `test_ppo_trainer.py` | 正式 sign-off 配置启用 reward normalization 时启动失败 |
| `test_horizon_env_reward_equals_trading_env_replay` | `test_horizon_env.py` | `r^sel` 等于内部 `TradingEnv.replay` 累加值，无重写 |
| `test_horizon_env_inherits_prev_terminal_position` | 同上 | reset 注入非零 `prev_terminal_position` 后第一步 cost 必须出现在 reward |
| `test_horizon_env_requires_position_in_state_when_continuous` | 同上 | `position_continuity=true` 但 `include_position=false` 时非 strict 配置必须失败 |
| `test_horizon_env_walk_forward_serial` | 同上 | walk-forward 模式下 horizon 顺序严格按 timestamp，禁止 reshuffle |
| `test_phase2_selection_policy_blocks_high_drawdown` | `test_phase2_selection_policy.py` | `val_max_drawdown > risk.max_drawdown` 的 checkpoint 必被拒绝 |
| `test_phase2_selection_policy_blocks_action_dominance` | 同上 | `action_dominance_ratio > behavior.max_action_dominance_ratio` 必拒绝 |
| `test_phase2_selection_policy_ignores_dp_and_test_metrics` | 同上 | DP/demo-label/test 指标变化不得改变 best verdict |
| `test_phase2_selection_policy_treats_kl_to_demo_as_warning` | 同上 | `val_kl_to_demo` 超阈只写 warning，不改变 best/reject |
| `test_phase2_replay_walk_forward_uses_no_dp` | `test_phase2_replay.py` | walk-forward 全程不调用 `SingleTradeDPPlanner` |
| `test_phase2_replay_uses_streaming_decode` | 同上 | walk-forward replay 每步只传入当前可见 state |
| `test_phase2_replay_position_consistency_metric` | 同上 | 边界仓位一致比例计算正确 |
| `test_phase2_metrics_sharpe_annualization_factor` | `test_phase2_metrics_portfolio.py` | annualization_factor 与 Phase I 一致 (525600) |
| `test_phase2_report_writer_atomic` | `test_phase2_report.py` | 原子写: 中途异常不留下半成品 json |

所有测试遵循 Phase I TDD 风格: `test_should_X_when_Y` 命名、Given/When/Then 注释、固定 seed、fixture 体积尽可能小。

### 11.3 集成测试目录

```text
tests/integration/
  test_phase2_pipeline_smoke.py
  test_phase2_walk_forward_position_continuity.py
  test_phase2_kl_demo_anchors_to_phase1_label.py
  test_phase2_no_future_information_in_state.py
  test_phase2_reproducibility.py
  test_phase2_phase1_artifact_validation.py
  test_phase2_dead_code_mask_end_to_end.py
  test_phase2_action_collapse_guardrail.py
  test_phase2_no_test_feedback_loop.py
  test_phase2_feature_provenance_guardrail.py
  test_phase2_argmax_is_primary_selection_mode.py
  test_phase2_hindsight_kl_demo_is_diagnostic_only.py
  test_phase2_no_running_stats_future_leakage.py
  test_phase2_data_gap_filter_end_to_end.py
  test_phase2_reward_scaling_does_not_affect_portfolio_metrics.py
```

### 11.4 关键集成测试场景

| 测试 | 关键不变量 |
| --- | --- |
| `test_phase2_pipeline_smoke` | 在 small fixture 上跑一轮 PPO update + 一次 walk-forward；要求产生 `phase2_report.json` 与 selector 产物，且 `val_guardrails_pass` / `test_used_for_selection` 字段存在 |
| `test_phase2_walk_forward_position_continuity` | 构造两段 horizon，第一段末仓位为 long；第二段第一步 target 为 short；reward 必须包含从 long 到 short 的盘口逐档换仓成本 |
| `test_phase2_kl_demo_anchors_to_phase1_label` | 用 fixture `code_label` 全部为 3，PPO 收敛后 selector 的 action distribution 在 KL term 主导（`α=10`）下应集中到 code 3 |
| `test_phase2_no_future_information_in_state` | 修改 horizon 内未来行的特征，selector 第 0 步 logits 必须不变；修改 `prev_terminal_position` 时 logits 允许变化，因为账户状态是当前已知信息 |
| `test_phase2_reproducibility` | 固定 seed + 固定 Phase I batch id 时，重复运行得到相同的 `best_checkpoint_path` 与 `phase2_composite_score` |
| `test_phase2_phase1_artifact_validation` | 篡改 `phase1_report.json.fatal_collapse=true` 时 trainer 必须以非零退出码失败 |
| `test_phase2_dead_code_mask_end_to_end` | fixture 中 code 7 在 Phase I usage=0；Phase II rollout/test 中 selector 永不输出 7；但 `code_label=7` 的样本 KL term 被 mask 而非崩溃 |
| `test_phase2_action_collapse_guardrail` | 构造 selector 始终选 code 0 的 fixture；selection policy 必须以 `action_dominance` 拒绝 best 选举 |
| `test_phase2_no_test_feedback_loop` | 构造 test 指标极好/极差两种 fixture；best checkpoint、composite score 和 guardrail verdict 必须完全由 val 决定 |
| `test_phase2_feature_provenance_guardrail` | feature provenance 中出现 future/centered/target 字段时，正式 sign-off 必须失败 |
| `test_phase2_argmax_is_primary_selection_mode` | stochastic seed pack 大幅波动时，best checkpoint 仍由 argmax validation metrics 决定 |
| `test_phase2_hindsight_kl_demo_is_diagnostic_only` | 构造两个 val_kl_to_demo 相反但在线指标相同的 checkpoint；best verdict 必须不变，只产生 warning |
| `test_phase2_no_running_stats_future_leakage` | 修改未来 observation/reward 后，当前时刻 RunningMeanStd / reward normalization 输出必须不变 |
| `test_phase2_data_gap_filter_end_to_end` | 构造含数据间隙的 market fixture；跨 gap horizon 被裁掉，report 记录 gap 数量 |
| `test_phase2_reward_scaling_does_not_affect_portfolio_metrics` | 改变 reward scaling 后 PPO training reward 改变，但 raw backtest net_return / Sharpe 计算口径不变 |
| `test_phase2_argmax_is_primary_selection_mode` | stochastic seed pack 大幅波动时，best checkpoint 仍由 argmax validation metrics 决定 |

### 11.5 fixture 设计

```text
tests/fixtures/phase2/
  small_market.feather             # ~5 day 1m bars，含完整盘口
  phase1_artifacts/
    decoder.pt                     # 小尺寸 decoder（hidden_dim=32）
    codebook.pt                    # K=4，code_dim=8
    horizon_labels_train.feather
    horizon_labels_val.feather
    horizon_labels_test.feather
    input_schema.json
    phase1_report.json             # fatal_collapse=false 健康样本
  configs/
    phase2_smoke.yaml              # total_timesteps=2048，num_envs=2
    phase2_strict.yaml             # paper_strict_reproduction=true
    phase2_dead_code_mask.yaml     # code 1 标注为 dead
```

fixture 必须 deterministic、轻量（< 5 MB），并保留 schema/config hash 以触发 cache 失效路径。

## 12. 验收标准

### 12.1 数据验收

- `phase2_horizon_index_*.feather` 必须能被加载，且字段完备。
- 所有 horizon 的 `last_markout_row <= num_rows - 1`。
- horizon 内存在 timestamp gap 超阈值时必须裁掉，`data_gap_filter` 统计必须写入 report。
- 正式训练/验证/test walk-forward 的 horizon 必须非重叠；`stride < h` 只能在 `dense_overlap_ablation=true` 且 `position_continuity=false` 时运行。
- `is_labeled=False` 的 horizon 不进入 KL/demo 训练梯度。
- 训练入口不得读取 test label；`phase2_horizon_index_test.feather` 默认不得包含 `code_label`。
- `feature_columns` 与 Phase I `input_schema.json` 完全一致。
- `feature_provenance.json` 必须存在且通过可用时间检查；缺失或存在 future/centered/target 字段时 `no_leakage_signoff=false`。
- `feature_provenance.json` 中所有拟合类处理必须为 `fit_scope=train_only` / `normalization_scope=train_only`；val/test 不得参与 scaler 或截面统计拟合。
- `cost_config` 与 Phase I `cost_config` 完全一致；不一致时启动失败。
- `input_norm=running_mean_std` 或 `reward_normalization=true` 时，正式 sign-off 必须失败，除非本次 run 明确标注为 ablation。
- Phase II 不得拟合任何 selector 输入 scaler；`state_dim_breakdown`、`position_encoding` 与 `feature_columns` 数量必须一致。

### 12.2 Phase I 产物链路验收

- `phase1_report.json.fatal_collapse=false` 与 `code_assignment_drift_warning=false`。
- `phase1_report.json.hindsight_bias_warning != "exceeded"` 或 `--allow-phase1-hindsight-warning` 显式开启。
- `decoder.pt`、`codebook.pt` 加载后 hash 与 `phase1_checkpoint_manifest.json.is_best=true` 行一致。
- `phase2_report.json` 必须记录 `phase1_batch_id / phase1_checkpoint_hash / phase1_config_hash / feature_provenance_hash`。
- `schema_hash_match_required=true` 时，训练与回测加载的数据 schema hash 必须与 Phase I/预注册配置一致；不一致直接失败。

### 12.3 Selector 行为验收

- `val_action_dominance_ratio < behavior.max_action_dominance_ratio`，否则不可成为 best。
- `val_active_archetype_ratio >= behavior.min_active_archetype_ratio`，否则触发 warning。
- `val_kl_to_demo` 只能作为 diagnostic；超过 `behavior.warn_kl_to_demo` 时写入 warning，不得拒绝 checkpoint 或参与 composite score。
- `kl_label_coverage_train / kl_label_coverage_val` 必须进入 report，用于解释 KL loss 的有效样本覆盖。
- 当 `position_continuity=true` 时，`prev_terminal_position` 必须进入 selector/critic state。
- dead-code mask 启用时，selector test action 中 dead code 数量必须为 0。
- `distribution_shift_warning_count`、`ood_max_score` 与 `online_action_throttle` 触发次数必须进入 report；正式 sign-off 不接受缺失。
- live 模式下若 `min_confidence_for_non_flat_action` 未满足，必须降级为 `flat_only` 或配置指定的保守模式。

### 12.4 PPO 训练健康验收

- `approx_kl` 不持续超过 `target_kl`，否则记录 `kl_early_stop_count` 并视情况降低 lr。
- `explained_variance > 0` 在训练后期；若持续 ≤ 0，写入 `risk_health_warnings`。
- `clip_fraction` 落在 `[0.05, 0.4]`；超出时提示 lr / advantage scale 调整。
- `policy_grad_norm` 与 `value_grad_norm` 不爆炸，`grad_clip_norm=0.5` 默认开启。
- rollout buffer 截断不得被写成真实 episode done；`truncated=True` 时必须 bootstrap value 且继续继承 `prev_terminal_position`。
- GAE 必须按 `env_id` 分组计算；跨 env 边界不得 bootstrap。
- `chunk_reset_position` 默认必须为 `inherit`；若为 `flat`，不得作为连续仓位 sign-off，且必须输出 `chunk_reset_distribution_shift`。
- `phase2_env_shards.feather` 必须存在；env 分片必须连续、互不重叠、覆盖全部训练 horizon，且边界落在 horizon 边界；文件必须包含边界 regime 摘要。
- `reward_scaling.mode` 默认必须为 `divide_by_horizon`；report 必须包含 raw/scaled horizon reward 分布，portfolio metrics 必须使用 raw PnL。
- 若 `reward_scaling.clip_range != null`，必须同时提供 unclipped 对照 run 或等价诊断；否则该 run 不得 sign-off。
- 任一 rollout/update 中出现 non-finite logits、advantages、loss、gradients 或 optimizer state 时，run 必须 fail-fast，并导出 debug snapshot。
- resume 之后的 RNG、env cursor、optimizer/scheduler 状态与 `prev_terminal_position` 必须可重建到最近一个已提交 checkpoint。

### 12.5 真实交易场景验收

- Val walk-forward 是 best / sign-off 的唯一评估来源；test walk-forward 是冻结后一次性最终报告，`test_used_for_selection=false` 必须写入 report。
- Test walk-forward 必须串行执行，单 env、不 reshuffle。
- `val_metrics` 固定为 argmax 指标，是 best/sign-off 主指标；`test_metrics.argmax` 是最终主报告；stochastic 只以 seed pack 诊断形式输出。
- `baselines_val` 必须包含 `random_selector / single_archetype / buy_and_hold_long / buy_and_hold_short`，用于 validation 审计；`baselines_test` 同字段只作为最终报告。
- selector `val_metrics.net_return`（argmax）必须严格大于 `random_selector.val_net_return.mean`；否则 `phase2_report.json.val_guardrails_pass=false`。
- `val_max_drawdown <= risk.max_drawdown`、`val_sharpe_ratio >= risk.min_sharpe_ratio`、`val_turnover_ratio <= risk.max_turnover_ratio`，否则 best checkpoint 不可 sign-off。
- test 指标若不达阈值，只能记录 `test_guardrails_pass_report_only=false` 或阻止部署该冻结 run，不得触发重新选 checkpoint 或调参；若需要继续迭代，必须重新定义 validation/test 协议。
- `boundary_turnover_cost / boundary_position_consistency` 必须出现在 report；与 Phase I 边界诊断对齐。
- `reject_transition_rate` 必须低于 Phase I `cost_config.reject_transition_health.max_dataset_reject_rate`，否则提示数据/盘口异常。
- `execution_stress_summary` 必须存在，且 stress 场景下不得出现灾难性崩坏：默认要求 `stress_max_drawdown <= nominal_max_drawdown + 0.10` 且 `stress_net_return` 不得跌入预注册拒绝阈值。
- `deployment_readiness` 必须显示 shadow replay、paper trading、canary 三层均完成，正式部署 verdict 才能为 true。
- `selector_latency_benchmark.p99` 必须低于对应 bar 间隔的预注册阈值；否则只能停留在 shadow/paper 阶段。
- 若 `flatten_trigger_mode="end_of_horizon"`，必须证明 `max_risk_control_response_lag` 与最坏 stress loss 上界仍在接受范围内，否则不得部署。

### 12.6 Composite score 与 sensitivity 验收

- `phase2_composite_score` 必须由 §4.3 默认权重组合计算，且只使用 validation 在线可执行指标；DP teacher、demo-label baseline 和 test metrics 不得参与。
- `phase2_composite_score` 必须基于 argmax validation metrics；stochastic seed pack 不参与 best，只作为稳定性诊断。
- `composite_score_sensitivity_phase2.json` 必须存在并覆盖默认 perturbations。
- 不同权重下 best timestep 漂移 ≥ 1 个 update 时打 `composite_weight_sensitivity_warning=true`。

### 12.7 产物验收

- Phase III 可以仅依赖 `best_selector.pt`、Phase I `decoder.pt / codebook.pt`、Phase II `phase2_horizon_index_*.feather` 与 `phase2_per_horizon_records_train.feather` 启动训练。
- 全部产物位于 `artifacts/{PAIR}/{PHASE2_BATCH_ID}/phase2/` 目录。
- 固定 seed + 固定 Phase I batch id 时，复跑得到一致的 `best_checkpoint_path`、`phase2_composite_score`、`test_metrics.argmax.net_return` 与 `test_metrics.stochastic_seed_pack.mean.net_return`（在数值容差内），且 test 结果不得改变 best checkpoint。

## 13. 风险与处理

| 风险 | 表现 | 处理 |
| --- | --- | --- |
| Phase I codebook 塌缩或 decoder 忽略 code 进入 Phase II | 不同 archetype 解码出几乎相同动作，selector 学不到收益差 | 启动前校验 `phase1_report.json` 的 collapse / behavior diversity warning；不通过时拒绝启动 |
| Phase I hindsight 分层带来虚高 selector 表现 | 训练 horizon 都来自 horizon-internal strata，selector 在偏置数据上学得"轻松" | 默认拒绝 `hindsight_bias_warning=exceeded` 的 Phase I batch；显式 `--allow-phase1-hindsight-warning` 才可继续，并写入风险确认 |
| Selector 输入量纲混杂 | market feature、仓位、lookback/pnl 扩展量级不同，MLP 早期训练不稳定 | Phase II 禁止拟合 scaler；market feature 进 LayerNorm trunk，仓位 one-hot，扩展特征只用样本内或固定常数归一化 |
| 重叠 horizon 与仓位继承同时启用 | 同一真实时间段被重复结算，前一 archetype 还未结束时又选下一 archetype | 正式默认 `non_overlap`；`stride < h` 时必须关闭 `position_continuity`、每 horizon flat reset，并标注 dense ablation |
| 继承仓位未进入 selector state | 同一 `s_t` 和 action 因隐藏仓位不同得到不同 reward，critic 学成 POMDP | 工程默认 `include_position=true`；`position_continuity=true && include_position=false` 时启动失败 |
| KL/demo 单 label 形式让 KL term 早期主导，扼杀探索 | Selector 紧紧贴着 demonstration，无法探索更优策略 | 支持 `kl_demo_anneal`；KL coef 默认从 1.0 退火到目标值；`val_kl_to_demo` 与 `entropy` 双面监控 |
| KL/demo diagnostic 被误用为 hard guardrail | validation best 被 hindsight label 间接筛选，削弱在线评估独立性 | `val_kl_to_demo` 只写 warning，不进入 composite score、不拒绝 checkpoint；单测锁定该语义 |
| rollout 截断被当作 episode 结束 | 每 `rollout_length` 重置仓位，训练成本低估且与 walk-forward 不一致 | 区分 `truncated` 与 `done`；truncated 时 bootstrap value 并继承 `prev_terminal_position` |
| 跨 horizon 仓位继承被遗忘 | selector 假设每段 horizon 都从 flat 起点；上线时由于继承 long/short，第一步反复换仓导致成本爆炸 | `HorizonEnv.reset` 强制接受 `prev_terminal_position`；集成测试 `test_phase2_walk_forward_position_continuity` 覆盖 |
| PPO 高方差 horizon-level reward 导致训练崩溃 | r^sel 量级波动大，advantage 噪声主导，clip_fraction 飙升 | 默认 `reward_scaling.mode=divide_by_horizon`、`advantage_normalization=True`、`grad_clip_norm=0.5`、`target_kl=0.05` early stop；raw/scaled reward 分布进入 report |
| Selector action collapse | 训练后期所有 horizon 都选同一 archetype | `entropy_coef >= 0.01` 默认；`max_action_dominance_ratio=0.6` guardrail 拒绝 best；report 写入 `behavior_health_warnings` |
| stochastic evaluation 噪声影响 best | 单个随机 seed 下收益好坏改变 checkpoint verdict | best/sign-off 主路径固定为 argmax；stochastic 只用预注册 seed pack 报均值/方差 |
| Dead code mask 与 demo label 冲突 | Phase I 有 dead code，但部分 horizon `code_label` 指向该 code | mask 的 KL term 置 0 而非崩溃；report 记录冲突数量 |
| 状态扩展引入未来信息 | 工程师把 `phase1_horizon_labels` 误拼进 `s^sel` 或读了 horizon 内未来行 | 集成测试 `test_phase2_no_future_information_in_state` 覆盖；任何扩展必须在 `phase2_config.yaml.state_extension` 中显式启用 |
| 原始 feature 已含未来信息 | 外部数据中存在 centered rolling、future return、target-like 字段，Phase II 只按 schema 读取时无法识别 | 正式 sign-off 要求 `feature_provenance.json`；字段名黑名单和可用时间检查不通过时 `no_leakage_signoff=false` |
| 数据间隙跨 horizon | 维护期或缺失数据导致 horizon 内时间不连续，reward/状态含异常跳变 | horizon indexer 检测 timestamp gap，默认裁掉跨 gap horizon，并在 report 输出 gap 统计 |
| Walk-forward 顺序错误 | reshuffle 后 prev_terminal_position 错位，边界换仓成本计算错误 | walk-forward 强制 single-env 串行；多 env 训练时每个 env 独立时间游标 |
| 多 env 分片不均或串接错误 | 某些 env 只看到单一市场区间，或 GAE 跨 env bootstrap | 训练 horizon 按时间连续均分，`phase2_env_shards.feather` 记录分片与边界 regime 摘要；GAE 按 `env_id` 分组，`rollover` 仅做诊断 |
| 批量 decoder API 被误用为未来函数 | 实现把完整 horizon 状态一次性送入非因果模型，未来 state 影响早期 action | 正式 replay 使用 `decode_step()` streaming；批量 `decode()` 只允许诊断，并用因果单测锁住 |
| Reward / cost 配置不一致 | Phase II 切换 `reward_alignment` 或降低 commission 提升表面收益 | 启动时 `cost_alignment_check=True` 校验；不一致直接报错 |
| 验证指标过拟合 best 选择 | 单一 composite score 选出在 val 上脆弱的 checkpoint | composite weight sensitivity 强制开启；`composite_weight_sensitivity_warning` 触发时不可 sign-off |
| test 反馈回路 | 看过 test 后调 checkpoint、阈值、walk-forward offset 或超参 | test 只在 best 冻结后运行一次；report 写 `test_used_for_selection=false`；继续迭代必须重新注册评估协议 |
| DP/demo oracle 进入 checkpoint 选择 | `val_return_capture_ratio_vs_dp` 或 demo-label baseline 让 hindsight 信息影响 best | composite score 和 guardrail 禁止使用 DP/demo-label/test 字段；这些字段仅 posthoc 审计 |
| 真实账户的盘口拒绝率高 | walk-forward 中 `reject_transition_rate` 偏高，selector 难以按预期成交 | 把 reject 率纳入 risk guardrail；超阈值时回到数据采样与 Phase I 数据质量排查 |
| 多 asset 共享 selector 提升表面指标但伤害单 asset 收益 | 共享 selector 用更多数据"刷分"，但每个 asset 单独 walk-forward 表现更差 | 共享模式必须用独立 BATCH_ID，并在 report 中按 asset 拆分指标，每个 asset 单独通过 guardrail |
| 训练步数不足 | 3M 在某些 asset 上不够；selector 还在动 | `total_timesteps` 配置化；`approx_kl` / `explained_variance` 曲线作为收敛参考；可启用 patience-based early stop（默认关闭以对齐论文） |
| Schedule 不稳定 | linear lr / kl coef 退火过快 | schedule 写入 report；提供 `lr_schedule=constant` 与 `kl_demo_coef` 不退火两种 baseline |
| 事后 KL baseline 与训练 KL 不一致 | posthoc evaluator 读取的 label 与训练 label 来源不一致 | 评估用 label 只能来自既有 `horizon_labels_*.feather`；Phase II 禁止对 test 重跑 encoder |
| Phase III 接口被破坏 | Phase II 改动 `Phase1FrozenPolicy` 的接口，Phase III 无法直接复用 | `Phase1FrozenPolicy.decode_step` / `decode` 接口纳入设计文档锁定签名；变更必须经 Phase III 设计审阅 |

| live 风控只存在于离线 report，没有实时硬闸 | 线上连续亏损、拒单、异常换手时策略仍继续交易 | 增加 `live_risk_controls`；触发 `daily_loss_limit / rolling_drawdown_limit / consecutive_reject_limit / turnover_burst_limit` 时强制 flatten 或 halt，默认支持 mid-horizon emergency flatten |
| 线上数据损坏但未被检测 | 时间戳乱序、stale 行情、crossed book、特征列错位仍进入 selector | 增加 `data_integrity_guardrails`，任何 schema/hash、NaN/Inf、crossed book、stale data 检查失败时直接 no-trade |
| OOD / regime shift 下仍强行交易 | live state 明显偏离训练分布，selector logits 失真 | 增加 `distribution_shift_monitoring`；连续告警达到阈值时切到 `flat_only` 或 `risk_reduced` |
| nominal cost 下盈利，stress 成本下崩溃 | 真实手续费/滑点/延迟更高时收益快速转负 | `execution_stress` 作为部署前强制基线；stress 崩溃则禁止部署 |
| 单一 validation 窗口选出的 checkpoint 偏 regime-specific | 主 val 指标好，但换个市场阶段显著失效 | 增加 `rolling_validation`，以 `mean_minus_std` 或 `worst_fold` 做部署审计 |
| selector 在 live 中频繁抖动 | archetype 高频切换、低置信度仍交易、局部 turnover 爆炸 | 增加 `online_action_throttle` 与 `min_confidence_for_non_flat_action`；必要时 fallback 为 flat |
| 数值异常静默扩散 | logits/loss/gradients 出现 NaN/Inf，但训练继续写出坏 checkpoint | `numerical_safety` fail-fast，写入 `terminated_due_to_numerical_instability=true` 并导出 debug snapshot |
| 训练通过后直接上实盘 | 没有 shadow/paper/canary，执行链路问题在真金白银阶段暴露 | `deployment_ladder` 强制 shadow replay → paper trading → canary → full deployment |
| KL label 时间覆盖高度集中 | selector 早期通过 hindsight label 学到 regime-specific 时间偏置，validation 表现被夸大 | `kl_label_temporal_coverage` 输出顺序覆盖曲线和时间熵；低于阈值时写入 `behavior_health_warnings` |
| dead code mask 错杀在当前 Phase II 时间段仍有价值的 code | selector 被硬掩码束缚，收益上界被不必要压低 | dead code 阈值基于 Phase I global usage，训练初期跑一次 unmasked diagnostic rollout；若 probe pick rate 明显非零则告警 |
| reward scaling clip 改变策略方向 | 训练日志看似更稳，但其实靠裁掉极端 reward 学到不同策略 | 默认 `clip_range=null`；若启用必须报告 clipped/unclipped 差异 |
| gap horizon 裁掉后被错误 flat reset | 真实持仓在数据缺口期间仍存在，但训练/回测假装清仓 | 按 gap 长度执行显式 `carry / force_flatten / warmup_only` 策略，禁止静默 reset |
| test label 被 diagnostic 路径误用进决策 | backtest 表面上是 posthoc，实际上把 oracle label 混入 selector 决策 | backtest runner 硬检查 decision path；发现 test `code_label` 被消费时直接抛错 |
| selector 推理延迟超过 bar 间隔 | 使用陈旧状态选 archetype，live 表现大幅劣化 | 强制记录 latency p50/p95/p99，并纳入 `execution_lag +2 bars` stress |

## 14. 与 Phase III 的接口

Phase III 读取:

- Phase I `decoder.pt`、`codebook.pt`、`encoder.pt`、`input_schema.json`、`reward_normalizer.json`。
- Phase II `best_selector.pt`、`phase2_horizon_index_*.feather`、`phase2_per_horizon_records_*.feather`、`phase2_report.json`、`phase2_config.yaml`。

Phase III 训练时必须满足:

- 冻结 Phase I encoder/decoder/codebook 与 Phase II selector。
- 使用 Phase II 选出的 archetype 推断 base actions，再在 step 级用 `{-1, 0, 1}` adapter 进行至多一次 override。
- 复用 `src/trading/horizon_env.py` 的内部 `TradingEnv`，但需要扩展为 step-level MDP `M_ref`，引入新的 step-level state（`s^ref1 = s_τ` 与 `s^ref2 = [e_{a^sel}, a^base_τ, R^arche_τ, τ_remain]`）。
- 复用 Phase II `phase2_horizon_index_*.feather` 与 `phase2_per_horizon_records_*.feather` 提供的 `prev_terminal_position` 与 `chosen_code` 字段，避免在 Phase III 重新跑 selector 决策。
- 推理阶段保持因果性：`a^ref_τ` 只能依赖 `s^ref_{0:τ}`，禁止读未来状态。

接口锁定:

```python
class Phase1FrozenPolicy:
    def decode_step(
        self,
        state_t: Tensor,
        code_id: int,
        recurrent_state: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """returns (base_action_t, decode_logits_t [3], next_recurrent_state)"""

    def decode(self, states_seq: Tensor, code_id: int) -> tuple[Tensor, Tensor]:
        """diagnostics only; returns (base_actions [h], decode_logits [h, 3])"""

class HorizonEnv:
    def reset(self, *, prev_terminal_position: int = 0) -> dict: ...
    def step(self, action: int) -> tuple[dict, float, bool, dict]: ...

class Phase2Inferencer:
    def select(self, s_sel: Tensor, dead_code_mask: Optional[Tensor]) -> int: ...
```

Phase III 必须只通过这些接口与 Phase I/II 交互，不直接读取内部权重。任何破坏接口的改动必须更新本设计文档与 Phase III 设计文档双向评审。

因此，Phase II 的最终验收不只是 `phase2_composite_score` 的高低，而是能否在严格的"无未来信息 + 真实成本 + 跨 horizon 仓位继承"条件下，给出一个可被 Phase III 安全接管、可在真实交易场景持续 walk-forward 的 archetype 选择器。