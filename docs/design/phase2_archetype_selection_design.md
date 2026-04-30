# Phase II Archetype Selection 技术设计

## 1. 目标与范围

本文档根据 `docs/paper/AAAI26_ArchetypeTrader_core.md` 中第二阶段 Archetype Selection 的要求，并以 `docs/design/phase1_archetype_discovery_design.md` 中已固化的 Phase I 产物和工程约束为输入，给出第二阶段可落地的工程设计。

第二阶段的目标是: 在 Phase I 训练得到的 `K=10` 个离散 archetype 上，训练一个 horizon 级 RL selector `π_φ^sel(a^sel | s^sel)`，使其在每个 horizon 起点根据当前市场观测 `s_t` 选出最合适的 archetype；该 archetype 的 codebook embedding 输入冻结的 Phase I decoder，由 decoder 因果地生成 horizon 内每一步的 base action `a^base_{t:t+h-1}`，再交由 `TradingEnv` 按统一 reward / cost 语义结算每步收益。

第二阶段只做四件事:

1. 加载 Phase I 的 `decoder.pt`、`codebook.pt`、`horizon_labels_*.feather` 和 `input_schema.json`。
2. 在 train/val/test 上按 walk-forward 调度生成 horizon-level rollout，构造 horizon-level MDP。
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
| state `s^sel = s_t`，捕捉 horizon 第一根 bar | `Phase2HorizonAdapter` 从 train/val/test feather 切出 horizon 起点的 state vector |
| action `a^sel ∈ {0,...,K-1}` | `ArchetypeSelector` 输出 `K` 类离散 logits |
| 通过冻结 decoder 解码 `a^base_{t:t+h-1}` | `Phase1FrozenPolicy` 包装 `decoder.pt` + `codebook.pt`，按 timestep 因果推理 |
| `r_t^sel = Σ_{τ=t}^{t+h-1} r_τ^step` | `HorizonEnv.step()` 内部调用 `TradingEnv.replay()` 累加 `step_reward` 后返回 |
| 目标 `J = E[Σ γ^t (r_t^sel - α KL(â_t^sel ‖ π_φ^sel(a_t^sel ‖ s_t^sel)))]` | `Phase2Loss` 包含 PPO clipped surrogate + value loss + KL/demo loss + entropy bonus |
| `α = 1` | `selection_kl_coefficient = 1.0` 默认值；可通过 schedule 退火 |
| ground-truth archetype label `â_t^sel` 来自 Phase I VQ encoder | `horizon_labels_*.feather.code_label`；Phase II 不重新跑 encoder |
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
  encoder.pt                         # 仅用于 Phase II 内部对未标注 horizon 重算 code_label，不参与 selector 推理
  horizon_labels_train.feather
  horizon_labels_val.feather
  horizon_labels_test.feather
  input_schema.json
  reward_normalizer.json
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

Phase II 的"一步 RL"对应 Phase I 的"一个 horizon"。横向调度规则必须固定且可复现:

```yaml
horizon_schedule:
  mode: stride                    # stride | non_overlap | phase1_index
  stride: 36                      # 默认 h/2，与 Phase I min_gap_between_samples 一致
  non_overlap_stride_minutes: 72  # mode=non_overlap 时使用
  use_phase1_window_index: false  # 若 true，复用 Phase I window_index_*.feather 的 sampled 行
  walk_forward:
    enabled: true                 # 仅 test backtest 强制启用
    chunk_minutes: null           # null = 一次性走完 test
    seed: 1234
  reward_alignment_lookahead_check: true
```

调度模式说明:

| `mode` | 用途 | 边界处理 |
| --- | --- | --- |
| `stride` | 训练/验证默认；horizon 起点按固定 `stride` 等距枚举 | 末尾 `last_markout_row > num_rows - 1` 的窗口必须裁掉 |
| `non_overlap` | walk-forward 最严格回测；horizon 起点按 `h` 等距 | 同上 |
| `phase1_index` | 与 Phase I sampled horizons 完全对齐，便于 KL/demo label 100% 对应 | 不允许新增 horizon |

无论哪种模式，horizon 调度生成产物 `phase2_horizon_index_{split}.feather` 必须包含:

| 字段 | 说明 |
| --- | --- |
| `sample_id` | Phase II horizon ID（与 Phase I `sample_id` 在 `phase1_index` 模式下完全一致） |
| `start_index` | horizon 起点行号 |
| `end_index` | horizon 终点行号（`start + h - 1`） |
| `last_execution_row` | 由 `RewardAlignment` 决定 |
| `last_markout_row` | 由 `RewardAlignment` 决定 |
| `phase1_sample_id` | 若该 horizon 与 Phase I sample 对应，则填入 Phase I 的 `sample_id`；否则 `null` |
| `code_label` | 来自 `horizon_labels_*.feather`，若该 horizon 未在 Phase I 标注则为 `null` |
| `is_labeled` | 是否拥有 KL/demo 监督信号 |
| `prev_terminal_position` | 训练 rollout 中由前一 horizon 末仓位填充；初始 horizon 为 0 |
| `split` | train / val / test |

边界约束:

- `last_markout_row` 不得越过对应 split 文件的实际最大行号。markout 越界的 horizon 必须裁掉，禁止 NaN/前向填充（与 Phase I §9.2 一致）。
- 若 `mode=stride` 且 `stride < h`，相邻 horizon 部分重叠，但 RL training step 之间的 advantage 计算必须按时间正序、不能 reshuffle，否则跨 horizon 仓位继承会被错位。
- val/test 的 horizon 不参与训练采样，必须在训练前固定 seed 生成并写入产物。
- `walk_forward.enabled=true` 时，test horizons 必须按时间严格非重叠串行执行，且每段 horizon 的 `prev_terminal_position` 必须接收前一段的真实末仓位。

### 3.4 KL/demo label 来源

KL/demo regularization 的 `â_t^sel` 来自 Phase I `horizon_labels_*.feather.code_label`:

- 训练 horizon: 必须使用 Phase I 已经分配好的 `code_label`，不得在 Phase II 内重新跑 encoder。
- val horizon: 同上。
- test horizon: 默认不参与 KL/demo 计算（test 只评估 selector 在线表现）；若需要在 test 上做"模仿基线"对照，可显式启用 `evaluate_kl_baseline_on_test=true` 并在 evaluator 内调用 `encoder.pt`，但不进入 selector 训练梯度。

未标注的 horizon（`is_labeled=false`）在训练 batch 中应通过 mask 把 KL term 置零，仍可参与 PPO 的 reward 与 advantage 学习；不允许把没有 label 的 horizon 直接丢弃，否则 walk-forward 训练会出现时间空洞。

### 3.5 状态字段

`s^sel` 严格遵循论文: 等于 horizon 第一根 bar 的 state vector。工程实现:

- `state_dim` 等于 Phase I `feature_columns` 长度，不重做特征工程，不在 Phase II 拟合任何 scaler。
- 状态经由 `Phase1RewardNormalizer` 之外的通道直接喂入 selector；reward normalizer 仅用于 Phase I encoder/decoder 输入，不能拿来变换 Phase II 状态。
- 严格论文复现禁止扩展 `s^sel`。若启用工程增强（如把 horizon 起点之前 `L` 根 bar 的特征 pooling 进 selector），必须:
  - `L` 严格只用历史 bar，禁止读 `t+` 之后的任何行；
  - 在配置和 `phase2_report.json` 中标注 `state_extension=past_lookback(L)`；
  - 严格复现实验中关闭。

可选状态扩展（默认全部关闭）:

```yaml
state_extension:
  past_lookback_minutes: 0             # >0 时把过去 L 根 bar 的 mean/std 拼接到 s^sel
  archetype_usage_history_window: 0    # 把过去 W 个 horizon 的 selector action 编码进 s^sel
  account_state:
    include_position: false            # 把 prev_terminal_position 编码进 s^sel
    include_recent_pnl: false          # 把过去 W 个 horizon 的 r^sel 编码进 s^sel
```

`account_state.include_position` 默认关闭以严格对齐论文；启用时必须在 `phase2_config.yaml` 标注 `paper_strict_reproduction=false`。

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
- 调用 `Phase2Trainer.run()`。

### 4.2 `scripts/backtest_phase2.py`

回测入口；用 best selector + 冻结 decoder/codebook + `HorizonEnv` 在 test 上做 walk-forward backtest，并输出真实交易场景下的指标。回测不再调用任何 DP，禁止读取 `horizon_labels_test.feather` 用于决策（仅用于事后比较 KL baseline）。

### 4.3 `src/config/phase2_config.py`

镜像 Phase I `phase1_config.py` 风格，使用 frozen dataclass + 显式 `_NESTED_TYPE_MAP`。关键配置组:

```python
@dataclass(frozen=True)
class HorizonScheduleConfig:
    mode: Literal["stride", "non_overlap", "phase1_index"] = "stride"
    stride: int = 36
    non_overlap_stride_minutes: int = 72
    use_phase1_window_index: bool = False
    walk_forward_enabled: bool = True
    walk_forward_seed: int = 1234


@dataclass(frozen=True)
class StateExtensionConfig:
    past_lookback_minutes: int = 0
    archetype_usage_history_window: int = 0
    include_position: bool = False
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


@dataclass(frozen=True)
class Phase2BehaviorGuardrailConfig:
    max_action_dominance_ratio: float = 0.6
    min_active_archetype_ratio: float = 0.5
    max_kl_to_demo: float = 1.0


@dataclass(frozen=True)
class Phase2SelectionPolicyConfig:
    selection_metric: str = "phase2_composite_score"
    selection_mode: Literal["max", "min"] = "max"
    metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "val_net_return": 0.30,
            "val_sharpe_ratio": 0.20,
            "val_return_capture_ratio_vs_dp": 0.15,
            "val_calmar_ratio": 0.15,
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
class Phase2Config:
    pair: str
    phase1_batch_id: str
    phase2_batch_id: str
    train_file: str
    val_file: str
    test_file: str
    artifact_root: str = "artifacts"
    horizon: int = 72
    horizon_schedule: HorizonScheduleConfig = field(default_factory=HorizonScheduleConfig)
    state_extension: StateExtensionConfig = field(default_factory=StateExtensionConfig)
    selector_network: SelectorNetworkConfig = field(default_factory=SelectorNetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    selection_policy: Phase2SelectionPolicyConfig = field(
        default_factory=Phase2SelectionPolicyConfig
    )
    cost_alignment_check: bool = True
    paper_strict_reproduction: bool = False
    allow_phase1_hindsight_warning: bool = False
```

`paper_strict_reproduction=True` 时调用 `apply_paper_strict_overrides`:

- 关闭所有 `state_extension`；
- 关闭 `action_mask_dead_codes`；
- `entropy_coef=0`；
- `kl_demo_label_smoothing=0`；
- `advantage_normalization=False`；
- `reward_normalization=False`；
- 强制 `paper_strict_reproduction=True` 写入 `phase2_report.json`。

`config_hash()` 必须包含 Phase I `phase1_config.config_hash()` 与 `phase1_checkpoint_hash`，以保证 Phase II 产物可追溯到具体的 Phase I best checkpoint。

### 4.4 `src/data/*` 新增模块

| 模块 | 职责 |
| --- | --- |
| `phase2_horizon_index.py` | 按 `HorizonScheduleConfig` 与 `RewardAlignment` 枚举/对齐 horizon 起点，写入 `phase2_horizon_index_{split}.feather`；不读取 Phase I window_index 之外的字段。 |
| `phase2_label_loader.py` | 加载 `horizon_labels_{split}.feather`，按 `sample_id` join 进 `phase2_horizon_index`；为未标注 horizon 填 `code_label=null, is_labeled=False`。 |
| `phase2_dataset.py` | PyTorch Dataset/IterableDataset；按 `start_index/end_index` 切出 `[h, feature_dim]` 状态张量与 `[h, levels, 4]` execution book，供 `HorizonEnv` 与 `Phase1FrozenPolicy` 推理。复用 Phase I `HorizonBuilder` 的切片逻辑（通过函数级 reuse 而非复制粘贴）。 |

边界约束:

- `phase2_horizon_index.py` 不再做分层采样；分层采样只属于 Phase I demonstration curation。Phase II 训练样本由 walk-forward 调度直接给出。
- `phase2_label_loader.py` 不写文件，只产出 in-memory join 结果。
- `phase2_dataset.py` 不调用 decoder/selector，只负责把 raw feather 切成张量；模型推理在 `HorizonEnv` 与 `PPOTrainer` 中完成。

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
    >>> base_actions, decode_logits = policy.decode(states_seq, code_id)

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
| `rollout_buffer.py` | 存储 `obs / action / log_prob / value / reward / done / kl_label / is_labeled`；提供 GAE advantage、return、minibatch 采样。 |
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

    每次 step:
      action a^sel ∈ {0,...,K-1}
      -> codebook[a^sel] 输入 frozen decoder
      -> decoder 因果生成 a^base_{0:h-1}
      -> Phase I TradingEnv.replay(a^base) 累加 step rewards
      -> 返回 (next_obs, r^sel, done, info)

    obs:
      horizon 起点的 s^sel；done 后切到下一个 horizon。

    cross-horizon position:
      prev_terminal_position 在 reset 时注入；保证 selector 不假设 flat 起点。
    """

    def reset(self, *, prev_terminal_position: int = 0) -> dict: ...
    def step(self, action: int) -> tuple[dict, float, bool, dict]: ...
```

实现要点:

- 内部维护一个 `TradingEnv` 实例和 `Phase1FrozenPolicy`，不在 `HorizonEnv` 里重写成本/行号映射。
- `info` 至少返回 `step_rewards / cost_paid / num_switches / boundary_turnover_cost / chosen_code / horizon_index`。
- 多并行 env: `num_envs` 个 `HorizonEnv` 实例对应 `num_envs` 个独立时间游标，按训练数据顺序滑窗，不能 reshuffle。

### 4.9 `src/trading/horizon_factory.py`

```python
def make_horizon_env(
    pair: str,
    split: str,
    horizon_index: pl.DataFrame,
    frame: pl.DataFrame,
    phase1_policy: Phase1FrozenPolicy,
    cost_config: CostConfig,
    seed: int,
    walk_forward: bool,
) -> HorizonEnv: ...
```

负责把 `cost_config` 转成 `LobDepthCostModel` + `RewardAlignment` + `TradingEnv`，再注入 `HorizonEnv`。trainer 与 backtest 都通过 factory 创建 env，避免环境配置在两处分叉。

### 4.10 `src/trainers/phase2_trainer.py`

职责:

- 加载 Phase I 产物并校验 sign-off。
- 构造 `phase2_horizon_index_*.feather` 与 `Phase2Dataset`。
- 创建 `num_envs` 个 `HorizonEnv` 与 `Phase1FrozenPolicy`。
- 实例化 `ArchetypeSelector` 与 `PPOTrainer`。
- rollout / update 循环；按 PPO update 步触发 validation evaluator。
- 调用 `Phase2SelectionPolicy.evaluate(metrics)` 决定 best checkpoint。
- 训练结束后调用 `Phase2BacktestRunner` 在 test 上做一次 walk-forward backtest，导出 `phase2_report.json` 与 selector 产物。

边界:

- trainer 不直接计算指标，只消费 `Phase2Evaluator` 输出。
- trainer 不直接写 checkpoint 文件，统一交给 `Phase2CheckpointManager`。
- trainer 不直接实现 PPO update，统一交给 `PPOTrainer`。

### 4.11 `src/trainers/phase2_checkpoint.py`

仿 Phase I `phase1_checkpoint.py`:

- 原子写 `best_selector.pt / last_selector.pt / checkpoints/step_*.pt`。
- 维护 `phase2_checkpoint_manifest.json`，记录每个 checkpoint 的 timestep、metrics、verdict（`best/rejected/periodic`）、Phase I batch id 与 hash。
- 不嵌入 best 选择规则。

### 4.12 `src/trainers/phase2_selection_policy.py`

仿 Phase I `selection_policy.py`，集中:

- `phase2_composite_score` 加权计算；
- 风险 guardrail（`max_drawdown / min_sharpe / max_turnover_ratio`）；
- 行为 guardrail（`max_action_dominance_ratio / min_active_archetype_ratio / max_kl_to_demo`）；
- KL early-stop 与 dead-code mask 兼容性检查；
- 拒绝原因写入 manifest。

### 4.13 `src/evaluation/phase2_evaluator.py`

职责:

- 在 train/val horizon 上 freeze selector，跑确定性 rollout（取 argmax 或随机 stochastic）评估指标。
- 调度 `phase2_replay.py` 做 walk-forward backtest。
- 计算 selection-specific 指标（selector entropy、archetype 使用分布、KL to demo、turnover）。
- 计算真实交易场景指标（净收益、Sharpe、Sortino、MDD、Calmar、turnover ratio、cost paid）。
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
       -> Phase1FrozenPolicy.decode(states_seq, code_id) -> base_actions
       -> TradingEnv.replay(base_actions) -> step_rewards / cost / final_position
    3. 累加 equity curve / position curve / cost curve。
    4. 输出真实交易指标与 per-horizon record。

    禁止
    ----
    - 调用 DP；
    - 调用 encoder.pt 决定 code_id（仅可在 KL baseline 对照模式下调用，且不进入 selector 决策）；
    - 在 horizon 内修改 base_actions（Phase III 的工作）。
    """
```

### 4.15 `src/evaluation/phase2_report.py`

`phase2_report.json` 字段大类:

- 训练统计: total_timesteps / wall-clock / approx_kl / clip_fraction / explained_variance / lr 曲线。
- selector 行为: action 分布、entropy、KL to demo、dead-code mask 命中、code switch frequency between horizons。
- horizon-level 收益: train/val/test 的 per-horizon r^sel 分布、累计净收益。
- 真实交易指标: net_return / annualized_return / sharpe / sortino / max_drawdown / calmar / turnover_ratio / cost_paid / num_horizons / num_trades。
- 与 baseline 对照: vs DP teacher / vs random selector / vs single-archetype baselines（每个 code 单独 lock 后 walk-forward） / vs buy-and-hold。
- 跨 horizon 边界: boundary_turnover_cost / boundary_position_consistency。
- Phase I 链路: phase1_batch_id / phase1_checkpoint_hash / phase1_config_hash / hindsight_warning_inherited。
- guardrail / sign-off: pass/fail 与原因。

## 5. Selection MDP 详细设计

### 5.1 状态空间

$$
s^{sel}_t = s_t
$$

其中 `s_t` 是 horizon 第一根 bar 的 state vector，与 Phase I encoder 输入第 0 步同源同 schema。

可选扩展（默认关闭）:

```text
s^sel_extended = [s_t, past_lookback_pool(s_{t-L:t-1}), one_hot(prev_a^sel), prev_terminal_position, recent_pnl_window]
```

任何扩展都必须在 `phase2_config.yaml.state_extension` 中显式配置，并在 `phase2_report.json` 中标注。严格论文复现禁止扩展。

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
  -> base_actions a^base_{t:t+h-1}
TradingEnv.reset(initial_position=prev_terminal_position)
TradingEnv.replay(a^base)
  -> step_rewards[t:t+h-1]
  -> final_position = position[t+h-1]
r^sel_k = sum(step_rewards)
prev_terminal_position_{k+1} = final_position
```

回合（episode）切分:

- 默认每个 horizon 是一个独立 transition；`done=True` 当且仅当当前 horizon 是当前训练 chunk 末段或 walk-forward 末端。
- 跨 horizon 的 advantage / return 由 GAE 在时间序列上计算（不强制 done=True）。
- `prev_terminal_position` 在 reset 之间持续传递；若启用 multi-env 训练，每个 env 各自维护独立的 `prev_terminal_position`，不能跨 env 共享或乱序。

### 5.4 奖励对齐

`r^sel_k = Σ_τ r^step_τ`，其中 `r^step_τ` 必须由 Phase I `TradingEnv` 与 `LobDepthCostModel` 计算，禁止 Phase II 重写。

成本必须包含手续费与盘口逐档滑点。Phase II 不得为了"更平稳"切换 reward_alignment、降低 commission 或改用 `fixed_bps`，否则 selector 学到的策略与真实部署不一致。

### 5.5 跨 horizon 仓位继承

Phase I §11 已明确 selector 必须支持 `initial_position != flat`。Phase II 实现要求:

- `HorizonEnv.reset(prev_terminal_position)` 必须把上一段 horizon 的末仓位接入 `TradingEnv.reset(initial_position=...)`。
- 第一步 target_position（来自 decoder）与 inherited 不一致时，`LobDepthCostModel` 自动扣除换仓成本，selector 该 horizon 的 reward 必然反映边界成本。
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

- 默认按 chunk 终止: 每 `rollout_length` 个 horizon 视为一个 mini-episode，切换 chunk 时 `done=True`。
- walk-forward backtest 中: episode 不强制终止，只在 test 末段 `done=True`，确保跨 horizon 仓位继承贯穿整段 test。
- 训练时若启用 chunk 终止，必须在 chunk 边界把 `prev_terminal_position` 重置为 0（可配置 `chunk_reset_position=0`），并在 `phase2_report.json` 中记录由此引入的边界成本上限估计。

## 6. Selector 网络与策略设计

### 6.1 主干

```text
state_dim
  -> input_norm (LayerNorm 或 RunningMeanStd 二选一)
  -> MLP(hidden_dim, num_layers, residual=False, activation=GELU, layer_norm=True)
  -> trunk_output (hidden_dim)
```

`input_norm` 的选择必须固定: 默认 `LayerNorm`，避免 PPO rollout 和 evaluation 中的 running stats 漂移；启用 `RunningMeanStd` 时必须同步 freeze stats 后再做 backtest。

### 6.2 Actor head

```text
trunk_output -> Linear(actor_head_hidden) -> GELU -> Linear(K)
  -> mask -> log_softmax
```

输出 `log_pi`，`pi = exp(log_pi)`。采样:

- 训练 rollout: stochastic（`Categorical(probs).sample()`），保证 PPO importance sampling 有效。
- 评估 rollout: 默认 stochastic，与训练一致；可选 `evaluation_action_mode=argmax` 用于诊断 deterministic 表现。

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

- mask 输入: `dead_code_mask = Phase1.code_usage_ratio < dead_code_usage_threshold`。
- 训练采样: mask 后 `log_softmax`，sample 的 action 永远不会是 dead code。
- KL/demo: 若 `code_label` 落在 dead code（说明 Phase I checkpoint 与 mask 阈值不一致），将该样本 KL term 置 0 并在 `phase2_report.json.behavior_health_warnings` 中记录。
- 评估: 评估 reward / Sharpe 时 mask 一致；不允许评估期临时关闭 mask "刷分"。

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
| `action` | int64[N, T] | selector 选择的 archetype id |
| `log_prob` | float32[N, T] | 旧策略 log_prob |
| `value` | float32[N, T] | critic 估值 |
| `reward` | float32[N, T] | r^sel |
| `done` | bool[N, T] | chunk 边界 |
| `kl_label` | int64[N, T] | â_t^sel；未标注样本任意值 |
| `is_labeled` | bool[N, T] | KL/demo mask |
| `dead_code_mask` | bool[N, T, K] | 训练期 mask 快照 |
| `info_cost_paid` | float32[N, T] | 用于评估 |
| `info_boundary_cost` | float32[N, T] | 用于评估 |
| `info_chosen_code` | int64[N, T] | 与 action 等价；冗余便于审计 |

`N=num_envs, T=rollout_length`。

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
            kl_demo_loss = -mean(log_pi_label * is_labeled)
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

### 7.6 与 Phase III 的预留

PPO trainer 必须保留 `step_action_callback` 钩子，未来 Phase III refinement 可以在 selector 选完 archetype 之后、decoder 执行前插入 step-level adapter。Phase II 当前实现里该 callback 默认不挂载，但接口必须存在。

## 8. 真实交易场景评估

### 8.1 评估时机

| 时机 | 数据 | 用途 |
| --- | --- | --- |
| 每 `validate_every_updates` 个 PPO update | val horizons | 计算快速指标，参与 best 选择 |
| best checkpoint 候选触发时 | val horizons | 跑完整 walk-forward replay，验证候选是否通过 guardrail |
| 训练结束后 | test horizons | 跑一次 walk-forward backtest 作为最终结果 |

### 8.2 Walk-forward 协议

- **顺序**: 严格按 `timestamp` 升序枚举 horizon。
- **stride**: 默认 `non_overlap`（stride = h），保证每个 minute 只参与一次 r^step；可选 `non_overlap_stride_minutes` 与 walk-forward 起点偏移 `walk_forward_seed`。
- **仓位继承**: 串行，每个 horizon `reset(prev_terminal_position)`。
- **selector 推理**: 训练期 stochastic；最终 backtest 默认 stochastic + 固定 seed，再额外跑一次 argmax 作为对照（写入 `evaluation_action_mode=stochastic / argmax`）。
- **DP 禁用**: 整个 walk-forward 过程中不允许调用 DP；`code_label` 仅作为事后 KL baseline 对照。

### 8.3 Baseline 对照

| Baseline | 描述 | 用途 |
| --- | --- | --- |
| `random_selector` | 在 mask 后 K 个 code 中均匀采样 | 检验 selector 是否优于乱选 |
| `single_archetype_k` | 每个 code 单独锁定后 walk-forward | 暴露每个 archetype 的"裸"能力 |
| `phase1_demo_label` | 用 `code_label` 当 selector，看模仿基线 | 对照 KL anchor 的上限 |
| `dp_teacher_offline` | DP 离线最优；仅用于 hindsight 对照 | 上限参考；不参与 sign-off |
| `buy_and_hold_long / short` | 全仓 long/short 不交易 | 市场基准 |

`baseline_metrics` 必须出现在 `phase2_report.json`，便于直接比较 net_return / Sharpe / MDD。

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
| `val_return_capture_ratio_vs_dp` | `selector_net_return / dp_teacher_net_return`，仅用于审计 |
| `val_regret_to_dp` | DP teacher 与 selector 的差距 |

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

`phase2_composite_score` 默认权重见 §4.3。主实验完成后必须做权重 sensitivity 检验:

- 以 §4.3 默认 perturbations 列表跑权重 ±10%；
- 写入 `composite_score_sensitivity_phase2.json`；
- 若不同权重下 best checkpoint 显著漂移，`phase2_report.json.composite_weight_sensitivity_warning=true`。

### 8.6 Per-asset 与 cross-asset 实验

论文按 BTC / ETH / DOT / BNB 各自训练 selector。工程实现要求:

- 每对 `{PAIR, PHASE2_BATCH_ID}` 独立产物目录，独立 selector 权重。
- 共享 selector 跨 asset 的实验属于 ablation: 必须使用独立 BATCH_ID，并以 `phase2_config.cross_asset.enabled=true` 标注；selector 输入必须包含 `asset_embedding` 才允许混合训练。
- 性能对比报告必须按 asset 拆分。

## 9. 训练流程

```text
phase1 artifacts
  -> Phase1ProductValidator 校验 sign-off
  -> phase2_horizon_index_{split}.feather 生成
  -> Phase2LabelLoader join code_label
  -> HorizonEnv * num_envs 创建 (frozen Phase1FrozenPolicy)
  -> ArchetypeSelector + PPOTrainer 初始化
  -> rollout / GAE / PPO update 循环
       每 K updates: Phase2Evaluator 在 val 上跑快速 rollout
       best 候选: Phase2BacktestRunner 在 val 上 walk-forward
       SelectionPolicy.evaluate -> verdict
  -> 训练结束: Phase2BacktestRunner 在 test 上 walk-forward
  -> Phase2ReportWriter 写 phase2_report.json + 产物
```

详细步骤:

1. 校验 Phase I 产物完整性与 sign-off 状态。
2. 加载 train/val/test 数据并校验 schema 与 Phase I `input_schema.json` 一致。
3. 生成 `phase2_horizon_index_{split}.feather`，按 `HorizonScheduleConfig` 与 `RewardAlignment` 严格枚举 horizon；越界 horizon 必须裁掉。
4. join Phase I `horizon_labels_*.feather`，标注 `code_label / is_labeled`。
5. 加载 `decoder.pt / codebook.pt` 至 `Phase1FrozenPolicy`，自检结构与 hash。
6. 构造 `num_envs` 个 `HorizonEnv` 实例，每个 env 拥有独立时间游标与 `prev_terminal_position`。
7. 实例化 `ArchetypeSelector` 与 `PPOTrainer`；实例化 `Phase2Evaluator`、`Phase2CheckpointManager`、`Phase2SelectionPolicy`。
8. 进入 PPO 训练循环:
   - 采集 rollout `num_envs * rollout_length` 个 horizon transitions。
   - 计算 GAE / returns / advantage normalize。
   - update selector `update_epochs * minibatches`，统计 PPO 健康指标。
   - 每 `validate_every_updates` 在 val 上跑快速评估指标。
   - best 候选触发时跑完整 walk-forward，并交给 `Phase2SelectionPolicy.evaluate`。
9. 训练结束，最终在 test 上跑 walk-forward backtest（stochastic + argmax 两种模式）。
10. 跑 `composite_score_sensitivity` 权重扰动实验。
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
| `phase2_horizon_index_test.feather` | 测试 horizon 索引 | 同上 |
| `best_selector.pt` | best checkpoint（selector 权重 + critic 权重） | actor + critic state_dict |
| `last_selector.pt` | 最后 update 的 selector state | 用于断点恢复 |
| `checkpoints/step_*.pt` | 周期 checkpoint | 调试用 |
| `phase2_checkpoint_manifest.json` | checkpoint 验证与选择记录 | timestep、metrics、verdict、Phase I batch id |
| `phase2_rollout_stats.feather` | 每 update 的 PPO 健康统计 | approx_kl / clip_fraction / explained_variance / lr |
| `phase2_per_horizon_records_val.feather` | val walk-forward 单 horizon 明细 | sample_id / chosen_code / r^sel / cost_paid / boundary_cost / final_position |
| `phase2_per_horizon_records_test.feather` | test walk-forward 单 horizon 明细 | 同上 |
| `phase2_baselines_test.json` | test 上各 baseline 的关键指标 | random_selector / single_archetype_k / dp_teacher / buy_and_hold |
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
  "config_hash": "",
  "paper_strict_reproduction": false,
  "allow_phase1_hindsight_warning": false,
  "hindsight_warning_inherited": "",
  "horizon_schedule": {
    "mode": "stride",
    "stride": 36,
    "walk_forward_enabled": true
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
    "val_kl_to_demo": 0.0,
    "dead_code_mask_hit_rate": 0.0
  },
  "val_metrics": {
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
    "stochastic": {},
    "argmax": {}
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
  "guardrails_reasons": [],
  "composite_weight_sensitivity_warning": false,
  "behavior_health_warnings": [],
  "risk_health_warnings": [],
  "boundary_health_warnings": []
}
```

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
| `test_horizon_index_aligns_with_phase1_when_mode_phase1_index` | 同上 | `phase1_index` 模式下的 sample 集合等于 Phase I sampled 集合 |
| `test_label_loader_marks_unlabeled_horizons` | `test_phase2_label_loader.py` | 没有 `code_label` 的 horizon `is_labeled=False`，KL term 应被 mask |
| `test_phase1_frozen_policy_outputs_are_causal` | `test_phase1_frozen_policy.py` | 修改 `s_{τ+1:}` 不改变 `base_actions[:τ+1]` |
| `test_phase1_frozen_policy_parameters_never_update` | 同上 | 任意 forward + backward 后 decoder/codebook 参数不变 |
| `test_archetype_selector_action_mask_blocks_dead_codes` | `test_archetype_selector.py` | dead-code mask 对应 logit 等于 `-inf`，sample 永不返回 dead code |
| `test_archetype_selector_log_softmax_consistency` | 同上 | `log_pi.exp().sum(dim=-1) == 1.0` |
| `test_rollout_buffer_gae_matches_reference` | `test_rollout_buffer.py` | GAE 输出与手算（小 fixture）一致 |
| `test_ppo_loss_clip_outside_window` | `test_ppo_loss.py` | `ratio > 1+ε` 与 `< 1-ε` 都触发 clip |
| `test_ppo_loss_kl_demo_masked_for_unlabeled` | 同上 | `is_labeled=False` 的 KL term=0 |
| `test_ppo_loss_kl_early_stop_triggers` | 同上 | `approx_kl > target_kl` 时返回 early-stop signal |
| `test_horizon_env_reward_equals_trading_env_replay` | `test_horizon_env.py` | `r^sel` 等于内部 `TradingEnv.replay` 累加值，无重写 |
| `test_horizon_env_inherits_prev_terminal_position` | 同上 | reset 注入非零 `prev_terminal_position` 后第一步 cost 必须出现在 reward |
| `test_horizon_env_walk_forward_serial` | 同上 | walk-forward 模式下 horizon 顺序严格按 timestamp，禁止 reshuffle |
| `test_phase2_selection_policy_blocks_high_drawdown` | `test_phase2_selection_policy.py` | `val_max_drawdown > risk.max_drawdown` 的 checkpoint 必被拒绝 |
| `test_phase2_selection_policy_blocks_action_dominance` | 同上 | `action_dominance_ratio > behavior.max_action_dominance_ratio` 必拒绝 |
| `test_phase2_replay_walk_forward_uses_no_dp` | `test_phase2_replay.py` | walk-forward 全程不调用 `SingleTradeDPPlanner` |
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
```

### 11.4 关键集成测试场景

| 测试 | 关键不变量 |
| --- | --- |
| `test_phase2_pipeline_smoke` | 在 small fixture 上跑一轮 PPO update + 一次 walk-forward；要求产生 `phase2_report.json` 与 selector 产物，且 `guardrails_pass` 字段存在 |
| `test_phase2_walk_forward_position_continuity` | 构造两段 horizon，第一段末仓位为 long；第二段第一步 target 为 short；reward 必须包含从 long 到 short 的盘口逐档换仓成本 |
| `test_phase2_kl_demo_anchors_to_phase1_label` | 用 fixture `code_label` 全部为 3，PPO 收敛后 selector 的 action distribution 在 KL term 主导（`α=10`）下应集中到 code 3 |
| `test_phase2_no_future_information_in_state` | 修改 horizon 内未来行的特征，selector 第 0 步 logits 必须不变；修改未来行盘口，r^sel 受影响但 selector 决策不变（因为 selector 只看 `s_t`） |
| `test_phase2_reproducibility` | 固定 seed + 固定 Phase I batch id 时，重复运行得到相同的 `best_checkpoint_path` 与 `phase2_composite_score` |
| `test_phase2_phase1_artifact_validation` | 篡改 `phase1_report.json.fatal_collapse=true` 时 trainer 必须以非零退出码失败 |
| `test_phase2_dead_code_mask_end_to_end` | fixture 中 code 7 在 Phase I usage=0；Phase II rollout/test 中 selector 永不输出 7；但 `code_label=7` 的样本 KL term 被 mask 而非崩溃 |
| `test_phase2_action_collapse_guardrail` | 构造 selector 始终选 code 0 的 fixture；selection policy 必须以 `action_dominance` 拒绝 best 选举 |

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
- `is_labeled=False` 的 horizon 不进入 KL/demo 训练梯度。
- `feature_columns` 与 Phase I `input_schema.json` 完全一致。
- `cost_config` 与 Phase I `cost_config` 完全一致；不一致时启动失败。

### 12.2 Phase I 产物链路验收

- `phase1_report.json.fatal_collapse=false` 与 `code_assignment_drift_warning=false`。
- `phase1_report.json.hindsight_bias_warning != "exceeded"` 或 `--allow-phase1-hindsight-warning` 显式开启。
- `decoder.pt`、`codebook.pt` 加载后 hash 与 `phase1_checkpoint_manifest.json.is_best=true` 行一致。
- `phase2_report.json` 必须记录 `phase1_batch_id / phase1_checkpoint_hash / phase1_config_hash`。

### 12.3 Selector 行为验收

- `val_action_dominance_ratio < behavior.max_action_dominance_ratio`，否则不可成为 best。
- `val_active_archetype_ratio >= behavior.min_active_archetype_ratio`，否则触发 warning。
- `val_kl_to_demo < behavior.max_kl_to_demo`，否则提示 KL 退火不足或 selector 与 demonstration 显著背离。
- dead-code mask 启用时，selector test action 中 dead code 数量必须为 0。

### 12.4 PPO 训练健康验收

- `approx_kl` 不持续超过 `target_kl`，否则记录 `kl_early_stop_count` 并视情况降低 lr。
- `explained_variance > 0` 在训练后期；若持续 ≤ 0，写入 `risk_health_warnings`。
- `clip_fraction` 落在 `[0.05, 0.4]`；超出时提示 lr / advantage scale 调整。
- `policy_grad_norm` 与 `value_grad_norm` 不爆炸，`grad_clip_norm=0.5` 默认开启。

### 12.5 真实交易场景验收

- Test walk-forward 必须串行执行，单 env、不 reshuffle。
- `test_metrics.stochastic` 与 `test_metrics.argmax` 必须同时输出。
- `baselines_test` 必须包含 `random_selector / single_archetype / phase1_demo_label / buy_and_hold_long / buy_and_hold_short`。
- selector net_return 必须严格大于 `random_selector.net_return`；否则 `phase2_report.json.guardrails_pass=false`。
- `test_max_drawdown <= risk.max_drawdown`、`test_sharpe_ratio >= risk.min_sharpe_ratio`、`test_turnover_ratio <= risk.max_turnover_ratio`，否则 best checkpoint 不可 sign-off。
- `boundary_turnover_cost / boundary_position_consistency` 必须出现在 report；与 Phase I 边界诊断对齐。
- `reject_transition_rate` 必须低于 Phase I `cost_config.reject_transition_health.max_dataset_reject_rate`，否则提示数据/盘口异常。

### 12.6 Composite score 与 sensitivity 验收

- `phase2_composite_score` 必须由 §4.3 默认权重组合计算，且通过所有 guardrails 才能成为 best。
- `composite_score_sensitivity_phase2.json` 必须存在并覆盖默认 perturbations。
- 不同权重下 best timestep 漂移 ≥ 1 个 update 时打 `composite_weight_sensitivity_warning=true`。

### 12.7 产物验收

- Phase III 可以仅依赖 `best_selector.pt`、Phase I `decoder.pt / codebook.pt` 与 Phase II `phase2_horizon_index_*.feather` 启动。
- 全部产物位于 `artifacts/{PAIR}/{PHASE2_BATCH_ID}/phase2/` 目录。
- 固定 seed + 固定 Phase I batch id 时，复跑得到一致的 `best_checkpoint_path`、`phase2_composite_score`、`test_metrics.stochastic.net_return`（在数值容差内）。

## 13. 风险与处理

| 风险 | 表现 | 处理 |
| --- | --- | --- |
| Phase I codebook 塌缩或 decoder 忽略 code 进入 Phase II | 不同 archetype 解码出几乎相同动作，selector 学不到收益差 | 启动前校验 `phase1_report.json` 的 collapse / behavior diversity warning；不通过时拒绝启动 |
| Phase I hindsight 分层带来虚高 selector 表现 | 训练 horizon 都来自 horizon-internal strata，selector 在偏置数据上学得"轻松" | 默认拒绝 `hindsight_bias_warning=exceeded` 的 Phase I batch；显式 `--allow-phase1-hindsight-warning` 才可继续，并写入风险确认 |
| KL/demo 单 label 形式让 KL term 早期主导，扼杀探索 | Selector 紧紧贴着 demonstration，无法探索更优策略 | 支持 `kl_demo_anneal`；KL coef 默认从 1.0 退火到目标值；`val_kl_to_demo` 与 `entropy` 双面监控 |
| 跨 horizon 仓位继承被遗忘 | selector 假设每段 horizon 都从 flat 起点；上线时由于继承 long/short，第一步反复换仓导致成本爆炸 | `HorizonEnv.reset` 强制接受 `prev_terminal_position`；集成测试 `test_phase2_walk_forward_position_continuity` 覆盖 |
| PPO 高方差 horizon-level reward 导致训练崩溃 | r^sel 量级波动大，advantage 噪声主导，clip_fraction 飙升 | 默认 `advantage_normalization=True`、`grad_clip_norm=0.5`、`target_kl=0.05` early stop；提供 reward normalization 选项作为 ablation |
| Selector action collapse | 训练后期所有 horizon 都选同一 archetype | `entropy_coef >= 0.01` 默认；`max_action_dominance_ratio=0.6` guardrail 拒绝 best；report 写入 `behavior_health_warnings` |
| Dead code mask 与 demo label 冲突 | Phase I 有 dead code，但部分 horizon `code_label` 指向该 code | mask 的 KL term 置 0 而非崩溃；report 记录冲突数量 |
| 状态扩展引入未来信息 | 工程师把 `phase1_horizon_labels` 误拼进 `s^sel` 或读了 horizon 内未来行 | 集成测试 `test_phase2_no_future_information_in_state` 覆盖；任何扩展必须在 `phase2_config.yaml.state_extension` 中显式启用 |
| Walk-forward 顺序错误 | reshuffle 后 prev_terminal_position 错位，边界换仓成本计算错误 | walk-forward 强制 single-env 串行；多 env 训练时每个 env 独立时间游标 |
| Reward / cost 配置不一致 | Phase II 切换 `reward_alignment` 或降低 commission 提升表面收益 | 启动时 `cost_alignment_check=True` 校验；不一致直接报错 |
| 验证指标过拟合 best 选择 | 单一 composite score 选出在 val 上脆弱的 checkpoint | composite weight sensitivity 强制开启；`composite_weight_sensitivity_warning` 触发时不可 sign-off |
| 真实账户的盘口拒绝率高 | walk-forward 中 `reject_transition_rate` 偏高，selector 难以按预期成交 | 把 reject 率纳入 risk guardrail；超阈值时回到数据采样与 Phase I 数据质量排查 |
| 多 asset 共享 selector 提升表面指标但伤害单 asset 收益 | 共享 selector 用更多数据"刷分"，但每个 asset 单独 walk-forward 表现更差 | 共享模式必须用独立 BATCH_ID，并在 report 中按 asset 拆分指标，每个 asset 单独通过 guardrail |
| 训练步数不足 | 3M 在某些 asset 上不够；selector 还在动 | `total_timesteps` 配置化；`approx_kl` / `explained_variance` 曲线作为收敛参考；可启用 patience-based early stop（默认关闭以对齐论文） |
| Schedule 不稳定 | linear lr / kl coef 退火过快 | schedule 写入 report；提供 `lr_schedule=constant` 与 `kl_demo_coef` 不退火两种 baseline |
| 事后 KL baseline 与训练 KL 不一致 | evaluator 用 encoder.pt 重算 label 与训练 label 不同 | 评估用 label 默认全部来自 `horizon_labels_*.feather`；如启用 encoder.pt 重算，必须用独立 BATCH_ID 并在 report 中区分 |
| Phase III 接口被破坏 | Phase II 改动 `Phase1FrozenPolicy` 的接口，Phase III 无法直接复用 | `Phase1FrozenPolicy.decode` 接口纳入设计文档锁定签名；变更必须经 Phase III 设计审阅 |

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
    def decode(self, states_seq: Tensor, code_id: int) -> tuple[Tensor, Tensor]:
        """returns (base_actions [h], decode_logits [h, 3])"""

class HorizonEnv:
    def reset(self, *, prev_terminal_position: int = 0) -> dict: ...
    def step(self, action: int) -> tuple[dict, float, bool, dict]: ...

class Phase2Inferencer:
    def select(self, s_sel: Tensor, dead_code_mask: Optional[Tensor]) -> int: ...
```

Phase III 必须只通过这些接口与 Phase I/II 交互，不直接读取内部权重。任何破坏接口的改动必须更新本设计文档与 Phase III 设计文档双向评审。

因此，Phase II 的最终验收不只是 `phase2_composite_score` 的高低，而是能否在严格的"无未来信息 + 真实成本 + 跨 horizon 仓位继承"条件下，给出一个可被 Phase III 安全接管、可在真实交易场景持续 walk-forward 的 archetype 选择器。
