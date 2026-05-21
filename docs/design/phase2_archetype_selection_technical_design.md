# Phase II Archetype Selection Technical Design

本文档根据论文 `Archetype Selection` 小节设计第二阶段工程实现。

Phase II 的目标是训练一个 horizon-level selector：

- 输入 selector 在线可见状态 `s_sel = (previous_t_states, current_t_states)`：
  上一分片完整状态序列，以及当前分片前 `TSize` 个状态；
- 输出一个 archetype id `a_sel in {0, ..., K-1}`;
- 使用 Phase I 冻结的 codebook embedding 和 decoder 生成该 horizon 的基础动作序列;
- 以 horizon 内逐步交易收益之和作为环境 reward;
- 额外加入 Phase I VQ assigned label 的 KL / imitation 约束，使 selector 不偏离可学习的 demonstration archetype 分布。

论文公式对应关系：

- `s_sel = (s_prev[0:H], s_cur[0:TSize])`: selector 的观察，只使用上一分片完整状态和当前分片前 `TSize` 个状态；
- `a_sel`: selector 选择的 archetype id；
- `p_theta_d(a_base | s, e_a_sel)`: 冻结 Phase I decoder；
- `r_t_sel = sum_tau r_tau_step`: horizon-level reward；
- `J = E[sum_t gamma^t r_t_sel - alpha KL(a_hat_sel || pi_sel)]`: RL reward 加 demonstration label 约束。

## 1. 设计目标

Phase II 工程实现需要满足：

- 复用 Phase I 产物：best checkpoint、codebook embedding、frozen decoder、state normalizer、horizon dataset；
- selector 在线推理只读取上一分片完整状态序列和当前分片前 `TSize` 个状态，不读取当前分片 `TSize` 之后的状态、未来价格、teacher action、teacher reward；
- frozen decoder 可在训练/评估中读取完整 horizon states，用于模拟选择某个 archetype 后的基础动作执行；
- reward 统一复用 `ActionExecutionCalculator`，避免收益口径和 Phase I validation 不一致；
- assigned label 由 Phase I 离线 label export 产物提供，只作为训练约束和诊断标签，不作为推理输入；
- checkpoint 中保存 q-network 权重、配置、训练指标、验证指标、Phase I checkpoint 引用，保证可审计。

## 2. 非目标

本阶段不负责：

- 重新训练 Phase I VQ encoder-decoder；
- 修改 Phase I codebook validation 规则；
- 实现 Phase III step-level refinement；
- 使用 DP teacher 在线推理；
- 在 selector 观察中引入未来 horizon 价格、future states、demo action 或 demo reward。

## 3. 命名原则与类型复用

Phase II 命名参考 Phase I 当前代码组织：

- 阶段内文件使用 `phase2_` 前缀，对齐 Phase I 的 `phase1_main.py`、`phase1_artifact_store.py`、`phase1_evaluator.py`；
- checkpoint 使用 `src/phase2/checkpoint/` 包，对齐 `src/phase1/checkpoint/`；
- evaluator 使用 `src/phase2/evaluators/` 包，对齐 `src/phase1/evaluators/`；
- report 使用 `src/phase2/report/` 包，对齐 `src/phase1/report/`；
- Phase II 自有神经网络定义放在 `src/phase2/model/`，对齐根目录 `src/model/` 的模型边界；
- selection 指标放在 `src/phase2/metrics/`，对齐 `src/phase1/metrics/` 的指标边界；
- RL 算法实现单独放在 `src/phase2/rl/`，只包含 reward、Double DQN loss、replay buffer 和 trainer，文件仍保留 `phase2_` 前缀；
- 数据 schema 文件命名为 `phase2_selection_data_schema.py`，对齐 Phase I metrics 中 `phase1_validation_data_schema.py` 的语义：只放数据契约，不放计算逻辑。

Phase I 或通用模块中已有的类型优先复用，不在 Phase II 重复定义：

| 已有类型/工具 | 来源 | Phase II 用途 |
|---|---|---|
| `HorizonDataset` | `src/model/data_types.py` | Phase II dataset builder 的 horizon 输入，包含 `states/prices/depthprices`。 |
| `TSize` | `src/model/data_types.py` | selector 可见的当前分片状态窗口长度。 |
| `VisibleStatesDataset` | `src/model/data_types.py` | selector observation，结构为 `(previous_t_states, current_t_states)`。 |
| `DemonstrationHorizonLabelDataset` | `src/model/data_types.py` | Phase I 离线 label，结构为 `(sample_ids, code_labels)`。 |
| `ArtifactPaths` | `src/model/data_types.py` | `Phase2ArtifactStore` 路径字典类型。 |
| `ArchetypeLabelTensor` | `src/model/tensor_data_types.py` | selector action、assigned label 和 code id 的 tensor 类型。 |
| `ActionLogitTensor` | `src/model/tensor_data_types.py` | frozen decoder 输出动作 logits 的类型。 |
| `LatentTensor` | `src/model/tensor_data_types.py` | codebook embedding / archetype latent 类型。 |
| `ActionExecutionResult` | `src/utils/trade_execution.py` | 环境执行结果和 reward 计算输入。 |
| `ActionExecutionCalculator` | `src/utils/trade_execution.py` | 统一交易收益、手续费、滑点和换手计算。 |
| `DataFileStore` | `src/store/artifact_store.py` | `Phase2ArtifactStore` 可继承该类复用数据集读写能力。 |

`phase2_selection_dataset.py` 不直接使用 `TrajectoryDataset`、
`TrajectoryTensorBatch` 或 Phase I encoder。Phase I 已通过
`src/phase1/horizon_train_label_builder.py` 离线导出 horizon-level `code_label`；
Phase II 只读取该 label 表，并和 `HorizonDataset` 按 `sample_id` 对齐。

Phase I 中带有明确阶段语义的 payload，例如 `Phase1Checkpoint`、
`Phase1ValidationResult`、`Phase1MetricResult`，不建议在 Phase II 直接复用。
如果后续发现 checkpoint/report result 可以跨阶段共享，应先抽到通用模块，再由
Phase I 和 Phase II 同时依赖。

## 4. 推荐文件结构

```text
scripts/
  train_phase2.py

src/phase2/
  __init__.py
  phase2_main.py
  phase2_config.py
  phase2_selection_data_schema.py
  phase2_selection_dataset.py
  phase2_env.py
  phase2_artifact_store.py
  model/
    __init__.py
    phase2_decoder_policy.py
    phase2_q_network.py
  metrics/
    __init__.py
    phase2_selection_metrics.py
  rl/
    __init__.py
    phase2_double_dqn_loss.py
    phase2_double_dqn_trainer.py
    phase2_replay_buffer.py
    phase2_selection_reward.py
  evaluators/
    __init__.py
    phase2_evaluator.py
    phase2_selection_evaluator.py
  checkpoint/
    __init__.py
    phase2_checkpoint.py
    phase2_checkpoint_selector.py
  report/
    __init__.py
    phase2_selection_report.py
    templates/
      phase2_selection_report.html

tests/
  test_phase2_selection_dataset_label_contract.py
  test_phase2_decoder_policy.py
  test_phase2_double_dqn_loss.py
  test_phase2_env_reward_contract.py
  test_phase2_replay_buffer.py
  test_phase2_selection_metrics_schema.py
  test_phase2_q_network_forward.py
```

## 5. 文件职责清单

| 文件 | 职责 |
|---|---|
| `scripts/train_phase2.py` | Phase II 命令入口，初始化 conda env 下的运行时、日志、随机种子和 `Phase2MainFlow`。 |
| `src/phase2/phase2_main.py` | 第二阶段主流程编排：加载 Phase I 产物、构建数据、训练 selector、评估、保存 checkpoint/report。 |
| `src/phase2/phase2_config.py` | 定义训练、环境、模型、reward、checkpoint/report 配置。 |
| `src/phase2/phase2_selection_data_schema.py` | 定义 Phase II 特有的数据流对象，例如 selection dataset、tensor batch、env step result 和 DQN transition batch；不放评估指标。 |
| `src/phase2/phase2_selection_dataset.py` | 从 horizon dataset 和 Phase I 导出的 label 表生成 selector dataset，包括 `VisibleStatesDataset`、当前 horizon dataset、以及 `DemonstrationHorizonLabelDataset`。 |
| `src/phase2/phase2_env.py` | horizon-level MDP 环境：接收 selector action，执行 decoder 动作，返回 horizon reward 和下一观察。 |
| `src/phase2/phase2_artifact_store.py` | 统一规划 Phase II 产物路径：dataset cache、checkpoint、metrics、report。 |
| `src/phase2/model/phase2_decoder_policy.py` | 封装冻结 Phase I decoder 的模型推理策略：根据 selected code id 生成 base actions。 |
| `src/phase2/model/phase2_q_network.py` | Double DQN 使用的 selector Q-network，输入 visible state，输出每个 archetype 的 Q value。 |
| `src/phase2/metrics/phase2_selection_metrics.py` | 定义 selection 评估指标、训练结果摘要和可序列化 metrics payload。 |
| `src/phase2/rl/phase2_selection_reward.py` | RL reward 和 imitation regularization 的基础计算，供 Double DQN loss/trainer 复用。 |
| `src/phase2/rl/phase2_replay_buffer.py` | Double DQN replay buffer，存储 horizon-level transition 并提供 batch sample。 |
| `src/phase2/rl/phase2_double_dqn_loss.py` | Double DQN TD target、TD loss 和 assigned-label imitation regularization。 |
| `src/phase2/rl/phase2_double_dqn_trainer.py` | Double DQN 训练循环：epsilon-greedy 采样、replay 更新、target network 同步和 checkpoint 触发。 |
| `src/phase2/evaluators/phase2_evaluator.py` | Phase II 基础 evaluator 入口，保持和 Phase I `phase1_evaluator.py` 命名一致。 |
| `src/phase2/evaluators/phase2_selection_evaluator.py` | selection 专项评估：收益、label consistency、code usage 和 decoder action quality。 |
| `src/phase2/checkpoint/phase2_checkpoint.py` | 保存/加载 Phase II checkpoint payload，记录 Phase I checkpoint lineage。 |
| `src/phase2/checkpoint/phase2_checkpoint_selector.py` | 从多个 Phase II checkpoint 中按 validation 指标选择 best checkpoint。 |
| `src/phase2/report/phase2_selection_report.py` | 输出 JSON/HTML selection report，不重新计算核心指标。 |

## 6. 核心数据流

```text
Phase I best checkpoint
  -> Phase I exports horizon code labels
  -> Phase II loads train/val/test horizon datasets
  -> Phase II loads exported horizon label tables
  -> Phase2SelectionDatasetBuilder.build_from_horizon_and_labels()
  -> Phase2SelectionDataset
  -> ArchetypeSelectionEnv + FrozenArchetypeDecoderPolicy
  -> Phase2DoubleDqnTrainer.train()
  -> Phase2SelectionEvaluator.evaluate()
  -> Phase2Checkpoint.save()
  -> Phase2CheckpointSelector.select_best_from_dir()
  -> Phase2SelectionReport.write_report()
```

训练时每个 horizon 样本包含两类信息：

- selector 可见信息：`visible_states = (previous_t_states, current_t_states)`；
- 环境模拟信息：当前分片的 `states[:, :, :]`、`prices[:, :, :]`、`depthprices[:, :, :] | None`、`code_label`。

`Phase2SelectionDatasetBuilder` 会用相邻 horizon 构造样本：

- `previous_t_states = horizon_states[:-1]`，形状 `[sample - 1, horizon, feature_dim]`；
- `current_t_states = horizon_states[1:, :TSize, :]`，形状 `[sample - 1, TSize, feature_dim]`；
- `horizon_dataset = (horizon_states[1:], prices[1:], depthprices[1:])`；
- `demonstration_horizon_label_dataset = (sample_ids[1:], code_labels[1:])`。

因此第 0 条 horizon 只作为第 1 条训练样本的上一分片上下文，不会形成独立 selector 训练样本。

`assigned_label` 来自 Phase I 已落盘的 horizon label 表。Phase II dataset builder
只读取和对齐该字段，不调用 Phase I encoder，不读取 demonstration trajectory。
该字段只能用于 KL/CE penalty 和评估，不允许拼进 selector observation。

## 7. 方法骨架与功能说明

### 7.1 `scripts/train_phase2.py`

```python
def initialize_runtime() -> None:
    """初始化日志、随机种子和 deterministic runtime。"""

def create_phase2_flow() -> Phase2MainFlow:
    """组装 Phase2MainConfig 并创建主流程。"""

def main() -> None:
    """Phase II 训练入口。"""
```

说明：

- 与 `scripts/train_phase1.py` 保持风格一致；
- 默认使用 `ArachetypeTrade` conda env 中的依赖运行；
- 第一版可硬编码 `pair`、`batch_id`、Phase I checkpoint 路径，后续再恢复 CLI。

### 7.2 `src/phase2/phase2_config.py`

```python
@dataclass(frozen=True)
class Phase2ModelConfig:
    state_dim: int
    num_archetypes: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

@dataclass(frozen=True)
class Phase2RewardConfig:
    gamma: float = 0.99
    fee_rate: float = 0.0002
    imitation_alpha: float = 1.0
    reward_clip: float | None = None
    normalize_rewards: bool = True

@dataclass(frozen=True)
class Phase2TrainConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3e-4
    replay_capacity: int = 200_000
    learning_start_epoch: int = 1
    updates_per_epoch: int = 1
    target_update_interval_epochs: int = 5
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_epochs: int = 20
    td_loss_beta: float = 1.0
    imitation_loss_beta: float = 1.0
    max_grad_norm: float = 0.5
    seed: int = 42

@dataclass(frozen=True)
class Phase2MainConfig:
    pair: str
    train_batch_id: str
    phase1_checkpoint_path: Path
    artifacts_root: Path = Path("artifacts")
    device: str = "cuda"
```

说明：

#### `Phase2ModelConfig`

使用场景：

- `Phase2MainFlow._create_q_network()` 根据它创建 online/target `Phase2QNetwork`；
- checkpoint 保存该配置，确保恢复 checkpoint 时能重建相同网络结构；
- evaluator/report 读取 `num_archetypes` 做 code usage、混淆矩阵和输出维度校验。

字段说明：

| 字段 | 作用 | 使用位置 |
|---|---|---|
| `state_dim` | 单个 state 的特征维度，等于 `current_t_states.shape[-1]`。Q-network 需要结合 `previous_t_states/current_t_states` 的形状决定实际输入编码方式。 | `Phase2QNetwork.__init__()` 输入层或 temporal encoder。 |
| `num_archetypes` | Phase I codebook 中可选 archetype 数量，也是 Q value 输出维度。 | Q-network 输出层、epsilon-greedy action 范围、评估 code usage。 |
| `hidden_dim` | Q-network MLP 隐层宽度。 | `model/phase2_q_network.py`。 |
| `num_layers` | Q-network MLP 隐层层数。 | `model/phase2_q_network.py`。 |
| `dropout` | 训练时的正则化 dropout。 | Q-network hidden block；eval 时自动关闭。 |

#### `Phase2RewardConfig`

使用场景：

- `phase2_env.py` 和 `phase2_selection_reward.py` 用它统一交易 reward 口径；
- `phase2_double_dqn_loss.py` 用 `gamma` 计算 TD target；
- checkpoint/report 保存它，保证收益、折扣和 imitation regularization 可审计。

字段说明：

| 字段 | 作用 | 使用位置 |
|---|---|---|
| `gamma` | Double DQN bootstrap 折扣因子。 | `compute_double_dqn_targets()`。 |
| `fee_rate` | 交易手续费率。 | `ActionExecutionCalculator` / `compute_selection_reward()`。 |
| `imitation_alpha` | assigned-label imitation regularization 的全局权重。 | `compute_double_dqn_loss()` 中组合 imitation loss。 |
| `reward_clip` | 可选 reward 裁剪阈值，降低极端 horizon return 对 TD target 的影响。 | `compute_selection_reward()` 或 replay 入库前。 |
| `normalize_rewards` | 是否对 replay batch reward 做标准化。 | `normalize_horizon_rewards()` / trainer update。 |

#### `Phase2TrainConfig`

使用场景：

- `Phase2DoubleDqnTrainer` 的训练循环、探索策略、replay buffer 和 target network 同步；
- `Phase2ReplayBuffer` 的容量和采样 batch size；
- optimizer 和梯度裁剪；
- checkpoint selection 记录训练轮数和训练超参。

字段说明：

| 字段 | 作用 | 使用位置 |
|---|---|---|
| `epochs` | Double DQN 总训练轮数；每轮遍历或采样一组 horizon-level transition。 | `Phase2DoubleDqnTrainer.train()` 主循环。 |
| `batch_size` | 每次从 replay buffer 采样的 transition 数。 | `Phase2ReplayBuffer.sample()`。 |
| `learning_rate` | online Q-network optimizer 学习率。 | trainer 初始化 optimizer。 |
| `replay_capacity` | replay buffer 最大 transition 数，满后按环形覆盖旧样本。 | `Phase2ReplayBuffer`。 |
| `learning_start_epoch` | 从第几轮开始更新 Q-network。 | trainer update gate。 |
| `updates_per_epoch` | 每轮执行多少次 Q-network update。 | trainer 主循环。 |
| `target_update_interval_epochs` | 每多少轮将 online network 硬同步到 target network。 | `sync_target_network()`。 |
| `epsilon_start` | epsilon-greedy 初始探索率。 | `build_epsilon_by_epoch()` / `collect_transition()`。 |
| `epsilon_end` | epsilon-greedy 最低探索率。 | `build_epsilon_by_epoch()`。 |
| `epsilon_decay_epochs` | epsilon 从 start 线性衰减到 end 的轮数。 | `build_epsilon_by_epoch()`。 |
| `td_loss_beta` | TD loss 权重。 | `compute_double_dqn_loss()`。 |
| `imitation_loss_beta` | assigned-label imitation loss 权重。 | `compute_double_dqn_loss()`。 |
| `max_grad_norm` | 梯度裁剪阈值，防止 TD target 波动导致梯度爆炸。 | `update_q_network()`。 |
| `seed` | replay 采样、epsilon action 和 torch/numpy 随机性的种子。 | runtime 初始化、replay buffer、trainer。 |

#### `Phase2MainConfig`

使用场景：

- `scripts/train_phase2.py` 创建 `Phase2MainFlow` 的入口配置；
- artifact store 用它定位 `artifacts/{pair}/{train_batch_id}/phase2/`；
- model/checkpoint loader 用它追踪 Phase I best checkpoint lineage；
- 所有子配置可以后续作为字段挂到 `Phase2MainConfig` 中，统一落盘。

字段说明：

| 字段 | 作用 | 使用位置 |
|---|---|---|
| `pair` | 交易标的或数据域名称。 | artifact 路径、日志、report。 |
| `train_batch_id` | 当前训练批次 ID。 | artifact 路径和 checkpoint metadata。 |
| `phase1_checkpoint_path` | Phase I best checkpoint 路径，用于加载 frozen decoder/codebook。 | `_load_phase1_model()`、checkpoint lineage。 |
| `artifacts_root` | 全阶段产物根目录。 | `Phase2ArtifactStore`。 |
| `device` | 训练和推理设备。 | model、decoder policy、trainer、evaluator。 |

### 7.3 `src/phase2/phase2_selection_data_schema.py`

```python
@dataclass(frozen=True)
class Phase2SelectionDataset:
    visible_states: VisibleStatesDataset
    horizon_dataset: HorizonDataset
    demonstration_horizon_label_dataset: DemonstrationHorizonLabelDataset

@dataclass(frozen=True)
class Phase2SelectionTensorBatch:
    visible_states: VisibleStatesDataset
    horizon_dataset: HorizonDataset
    demonstration_horizon_label_dataset: DemonstrationHorizonLabelDataset

@dataclass(frozen=True)
class Phase2SelectionStepResult:
    observation: VisibleStatesDataset
    reward: float
    done: bool
    info: dict[str, Any]

@dataclass(frozen=True)
class Phase2SelectionTransitionBatch:
    visible_states: VisibleStatesDataset
    actions: ArchetypeLabelTensor
    rewards: torch.Tensor
    next_visible_states: VisibleStatesDataset
    dones: torch.Tensor
    demonstration_horizon_label_dataset: DemonstrationHorizonLabelDataset
```

说明：

- `Phase2SelectionDataset`
  使用场景：dataset builder 的输出、dataset cache 的落盘结构、env/evaluator 的 numpy 输入。
  当前实现位于 `src/phase2/phase2_selection_dataset.py`，包含 selector 可见的
  `VisibleStatesDataset`、环境模拟需要的当前 `HorizonDataset`、以及 imitation/诊断使用的
  `DemonstrationHorizonLabelDataset`。
- `Phase2SelectionTensorBatch`
  使用场景：`DataLoader` 或 trainer 小批量训练输入。当前 schema 保持和
  `Phase2SelectionDataset` 一致的三段结构：selector 可见状态、当前 horizon
  数据、以及 Phase I assigned label 数据。后续 tensor 化实现可在这些 tuple
  内部承载 tensor，但字段边界不拆散。
- `Phase2SelectionStepResult`
  使用场景：`ArchetypeSelectionEnv.step()` / `run_horizon()` 的返回值。
  它描述一个 horizon-level action 执行后的 reward、done 和诊断信息。
- `Phase2SelectionTransitionBatch`
  使用场景：`Phase2ReplayBuffer.sample()` 的输出，以及 Double DQN TD 更新的输入。
  它保存 horizon-level `visible_states/action/reward/next_visible_states/done`，
  并额外保留 `demonstration_horizon_label_dataset` 用于 imitation regularization
  和样本追踪。

### 7.4 `src/phase2/metrics/phase2_selection_metrics.py`

```python
@dataclass(frozen=True)
class Phase2SelectionEvaluationMetrics:
    mean_return: float
    median_return: float
    sharpe_like: float
    win_rate: float
    mean_turnover: float
    label_top1_match: float
    code_usage_entropy: float
    oracle_label_return: float
    random_label_return: float

    def to_dict(self) -> dict[str, object]:
        """序列化为 checkpoint/report 可保存的 dict。"""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2SelectionEvaluationMetrics":
        """从 checkpoint/report payload 恢复 selection evaluation metrics。"""

@dataclass(frozen=True)
class Phase2TrainingResult:
    epoch: int
    best_checkpoint_path: Path
    train_metrics: Mapping[str, float]
    validation_metrics: Phase2SelectionEvaluationMetrics
    selected_checkpoint_summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, object]:
        """序列化为 report/checkpoint selector 可保存的训练结果摘要。"""
```

说明：

- `Phase2SelectionEvaluationMetrics` 同时比较 selector、oracle assigned label、
  random label 的收益差异；
- evaluator 只负责计算该对象，checkpoint selector 和 report 只消费该对象；
- `Phase2TrainingResult` 是训练主流程返回给 checkpoint selector 和 report 的
  交接结果，因此和 metrics payload 放在同一模块，不放在 data schema；
- 后续如果需要 hard gate、tie-breaker 或 risk finding，应继续在
  `src/phase2/metrics/` 下拆分，不放回 data schema。

### 7.5 `src/phase2/phase2_selection_dataset.py`

```python
@dataclass(frozen=True)
class Phase2SelectionDataset:
    visible_states: VisibleStatesDataset
    horizon_dataset: HorizonDataset
    demonstration_horizon_label_dataset: DemonstrationHorizonLabelDataset

class Phase2SelectionDatasetBuilder:
    def __init__(
        self,
        *,
        tsize: TSize = 1,
    ) -> None:
        """初始化 Phase II dataset builder；不持有 Phase I model。"""

    def build_from_horizon_and_labels(
        self,
        horizon_dataset: HorizonDataset,
        label_table: pl.DataFrame,
    ) -> Phase2SelectionDataset:
        """从 horizon 数据和 Phase I 导出的 label 表生成 Phase II selector dataset。"""

    def validate_horizon_dataset(
        self,
        horizon_dataset: HorizonDataset,
    ) -> None:
        """检查 states/prices/depthprices 的 3D shape 和 sample/horizon 维度一致性。"""

    def extract_demonstration_horizon_label_dataset(
        self,
        label_table: pl.DataFrame,
        sample_count: int,
    ) -> DemonstrationHorizonLabelDataset:
        """从 label 表读取 sample_id/code_label，并按 sample_id 排序。"""

    def validate_label_alignment(
        self,
        label_table: pl.DataFrame,
        sample_count: int,
    ) -> None:
        """检查 label 表 sample_id 是否完整、唯一，并与 horizon sample 数一致。"""

    def build_visible_states(self, horizon_states: np.ndarray) -> VisibleStatesDataset:
        """提取 selector 可见状态：上一分片完整状态 + 当前分片前 tsize 个状态。"""

    def validate_no_future_leakage(
        self,
        dataset: Phase2SelectionDataset,
    ) -> None:
        """检查 visible_states 形状和来源，防止未来 states/prices 混入 observation。"""

    def to_tensor_dataset(
        self,
        dataset: Phase2SelectionDataset,
    ) -> torch.utils.data.TensorDataset:
        """把 numpy dataset 转成 PyTorch TensorDataset。"""
```

说明：

- `horizon_dataset` 提供 `states/prices/depthprices`；
- `label_table` 是 Phase I 离线导出的 horizon label 表，至少包含 `sample_id` 和 `code_label`；
- `horizon_dataset` 和 `label_table` 必须按 `sample_id` 完整对齐，`sample_id` 必须唯一且为完整零基连续区间 `[0, sample_count)`；
- `build_from_horizon_and_labels()` 先生成全量 label dataset，再丢弃第 0 行，使返回 dataset 的 `sample_ids/code_labels` 对齐当前分片 `horizon_states[1:]`；
- `to_tensor_dataset()` 的返回列顺序固定为 `previous_t_states, current_t_states, horizon_states, prices, depthprices, assigned_labels, sample_ids`；
- 本 builder 不加载 Phase I checkpoint，不调用 Phase I encoder，不读取 `TrajectoryDataset`；
- 若未来 Phase II selector 改变可见窗口，只允许在 `build_visible_states()` 中调整，且必须同步更新文档和 leakage test。

### 7.6 `src/phase2/model/phase2_decoder_policy.py`

```python
class FrozenArchetypeDecoderPolicy:
    def __init__(
        self,
        phase1_model: ArchetypeVQModel,
        device: torch.device | str,
    ) -> None:
        """冻结 Phase I model，只暴露 code id -> base action sequence 的解码能力。"""

    def decode_actions(
        self,
        horizon_states: torch.Tensor,
        selected_code_ids: ArchetypeLabelTensor,
    ) -> torch.Tensor:
        """根据 horizon states 和 selected code ids 输出 base actions，形状 [batch, H]。"""

    def decode_all_codes(
        self,
        horizon_states: torch.Tensor,
    ) -> torch.Tensor:
        """为每个样本解码全部 K 个 archetype 的 base actions，形状 [batch, K, H]。"""

    def get_code_embeddings(
        self,
        selected_code_ids: ArchetypeLabelTensor,
    ) -> LatentTensor:
        """从 Phase I codebook 读取 selected archetype embeddings。"""
```

说明：

- 内部必须 `phase1_model.eval()` 且关闭梯度；
- 本文件属于 Phase II model 层：它不训练 Phase I decoder，只封装冻结模型的推理调用；
- `decode_actions()` 是训练环境执行 selector action 的主入口；
- `decode_all_codes()` 可用于 evaluator 计算 oracle best-code upper bound 或诊断不同 code 的收益分布。

### 7.7 `src/phase2/phase2_env.py`

```python
class ArchetypeSelectionEnv:
    def __init__(
        self,
        dataset: Phase2SelectionDataset,
        decoder_policy: FrozenArchetypeDecoderPolicy,
        reward_config: Phase2RewardConfig,
    ) -> None:
        """构建 horizon-level selection MDP。"""

    def reset(self, index: int | None = None) -> VisibleStatesDataset:
        """重置到一个 horizon 样本，返回该样本的 visible states。"""

    def step(self, selected_code_id: int) -> Phase2SelectionStepResult:
        """执行一个 horizon-level archetype action 并返回 reward/done/info。"""

    def run_horizon(
        self,
        sample_index: int,
        selected_code_id: int,
    ) -> Phase2SelectionStepResult:
        """对指定样本和 code id 做无状态执行，便于 evaluator 批量调用。"""

    def _decode_base_actions(
        self,
        horizon_states: np.ndarray,
        selected_code_id: int,
    ) -> np.ndarray:
        """调用 frozen decoder 得到单个 horizon 的基础动作序列。"""

    def _execute_actions(
        self,
        prices: np.ndarray,
        actions: np.ndarray,
        depthprices: np.ndarray | None,
    ) -> ActionExecutionResult:
        """复用 ActionExecutionCalculator 计算交易收益、费用、换手。"""

    def _build_info(
        self,
        sample_index: int,
        selected_code_id: int,
        assigned_label: int,
        execution: ActionExecutionResult,
    ) -> dict[str, Any]:
        """返回训练和诊断所需的 horizon 执行详情。"""
```

说明：

- `step()` 在 Phase II 中一步即完成一个 horizon，因此 `done=True`；
- dataset index 的推进策略可以是顺序、随机或 vectorized sampler；
- `info` 至少包含 `sample_id`、`selected_code_id`、`assigned_label`、`gross_return`、`fee`、`turnover`。

### 7.8 `src/phase2/rl/phase2_selection_reward.py`

```python
def compute_selection_reward(
    execution: ActionExecutionResult,
    reward_config: Phase2RewardConfig,
) -> np.ndarray:
    """把交易执行结果转成 horizon-level reward r_t_sel。"""

def compute_imitation_kl_loss(
    q_values: torch.Tensor,
    assigned_labels: torch.Tensor,
) -> torch.Tensor:
    """用 Q value logits 对 assigned label 计算 cross-entropy imitation loss。"""

def build_epsilon_by_epoch(
    epoch: int,
    train_config: Phase2TrainConfig,
) -> float:
    """按线性退火配置返回当前 epsilon-greedy 探索率。"""

def normalize_horizon_rewards(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """对 replay batch reward 做标准化，降低不同市场阶段收益尺度差异。"""
```

说明：

- `compute_selection_reward()` 只处理交易收益，不混入 KL；
- imitation regularization 作为 Double DQN loss 的辅助项，便于分别记录 TD loss 和 label consistency；
- 第一版 imitation target 使用 one-hot `assigned_labels`，后续可扩展为 Phase I label posterior 或 smoothed target。

### 7.9 `src/phase2/model/phase2_q_network.py`

```python
@dataclass(frozen=True)
class Phase2QNetworkOutput:
    q_values: torch.Tensor

class Phase2QNetwork(nn.Module):
    def __init__(self, config: Phase2ModelConfig) -> None:
        """构建 horizon-level archetype selector Q-network。"""

    def forward(
        self,
        previous_t_states: torch.Tensor,
        current_t_states: torch.Tensor,
    ) -> Phase2QNetworkOutput:
        """输入 selector 可见状态，输出每个 archetype 的 Q value，形状 [batch, K]。"""

    def select_action(
        self,
        previous_t_states: torch.Tensor,
        current_t_states: torch.Tensor,
        epsilon: float,
        deterministic: bool = False,
    ) -> ArchetypeLabelTensor:
        """按 epsilon-greedy 或 greedy 策略选择 archetype id。"""

    def greedy_action(
        self,
        previous_t_states: torch.Tensor,
        current_t_states: torch.Tensor,
    ) -> ArchetypeLabelTensor:
        """返回 Q value 最大的 archetype id。"""

    def predict_proba(
        self,
        previous_t_states: torch.Tensor,
        current_t_states: torch.Tensor,
    ) -> torch.Tensor:
        """对 Q value 做 softmax，返回仅供评估/report 使用的伪概率。"""
```

说明：

- 第一版 Q-network 输入与 `Phase2SelectionDatasetBuilder.to_tensor_dataset()` 对齐，显式接收
  `previous_t_states` 和 `current_t_states` 两列，输出 `num_archetypes` 个 Q value；
- 本文件只定义网络结构和 action selection，不包含 replay、target network 同步或 TD loss；
- 由于 `previous_t_states` 是完整上一分片序列、`current_t_states` 是当前分片前 `TSize` 个状态，模型实现可以先 flatten，也可以增加 temporal encoder，但不能读取当前分片 `TSize` 之后的状态；
- 评估时使用 greedy action；训练时由 trainer 传入 epsilon 做探索。

### 7.10 `src/phase2/rl/phase2_replay_buffer.py`

```python
@dataclass(frozen=True)
class Phase2ReplayTransition:
    visible_states: VisibleStatesDataset
    action: int
    reward: float
    next_visible_states: VisibleStatesDataset
    done: bool
    demonstration_horizon_label_dataset: DemonstrationHorizonLabelDataset

class Phase2ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        visible_state_shapes: tuple[tuple[int, ...], tuple[int, ...]],
        seed: int,
    ) -> None:
        """初始化固定容量 replay buffer。"""

    def add(self, transition: Phase2ReplayTransition) -> None:
        """写入一个 horizon-level transition。"""

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> Phase2SelectionTransitionBatch:
        """随机采样 Double DQN 训练 batch。"""

    def __len__(self) -> int:
        """返回当前 buffer 中可采样 transition 数量。"""
```

说明：

- Phase II 环境一步就是一个 horizon，`next_visible_states` 是下一个可训练
  horizon 样本的 selector 可见状态；
- `demonstration_horizon_label_dataset` 不参与环境状态，只作为 imitation
  regularization target 和样本追踪信息；
- replay buffer 不调用模型、不计算 reward，只管理 transition 存取。

### 7.11 `src/phase2/rl/phase2_double_dqn_loss.py`

```python
@dataclass(frozen=True)
class Phase2DoubleDqnLossOutput:
    total_loss: torch.Tensor
    td_loss: torch.Tensor
    imitation_loss: torch.Tensor
    mean_q_selected: torch.Tensor
    mean_td_target: torch.Tensor

def compute_double_dqn_targets(
    online_q_network: Phase2QNetwork,
    target_q_network: Phase2QNetwork,
    batch: Phase2SelectionTransitionBatch,
    gamma: float,
) -> torch.Tensor:
    """用 online network 选择 next action，用 target network 估计 next Q target。"""

def compute_td_loss(
    online_q_values: torch.Tensor,
    actions: ArchetypeLabelTensor,
    td_targets: torch.Tensor,
) -> torch.Tensor:
    """计算 selected action Q value 和 Double DQN target 的 Huber/MSE TD loss。"""

def compute_double_dqn_loss(
    online_q_network: Phase2QNetwork,
    target_q_network: Phase2QNetwork,
    batch: Phase2SelectionTransitionBatch,
    reward_config: Phase2RewardConfig,
    train_config: Phase2TrainConfig,
) -> Phase2DoubleDqnLossOutput:
    """组合 TD loss 和 assigned-label imitation loss。"""
```

说明：

- Double DQN target 使用 `online_q_network(*batch.next_visible_states).argmax()` 选动作；
- target value 使用 `target_q_network(*batch.next_visible_states).gather(next_action)`；
- `done=True` 时不加 bootstrap 项；
- `total_loss = td_loss_beta * td_loss + imitation_loss_beta * imitation_loss`。

### 7.12 `src/phase2/rl/phase2_double_dqn_trainer.py`

```python
class Phase2DoubleDqnTrainer:
    def __init__(
        self,
        online_q_network: Phase2QNetwork,
        target_q_network: Phase2QNetwork,
        env: ArchetypeSelectionEnv,
        replay_buffer: Phase2ReplayBuffer,
        train_config: Phase2TrainConfig,
        reward_config: Phase2RewardConfig,
        device: torch.device | str,
    ) -> None:
        """初始化 online/target network、optimizer、replay buffer 和训练配置。"""

    def train(self) -> Phase2TrainingResult:
        """运行完整 Double DQN 训练循环。"""

    def collect_transition(self, epoch: int) -> Phase2ReplayTransition:
        """按 epsilon-greedy 选择 archetype，执行 env.step() 并生成 replay transition。"""

    def update_q_network(
        self,
        batch: Phase2SelectionTransitionBatch,
    ) -> dict[str, float]:
        """计算 Double DQN loss，反向传播并更新 online q-network。"""

    def sync_target_network(self) -> None:
        """将 online q-network 参数同步到 target q-network。"""

    def should_save_checkpoint(
        self,
        epoch: int,
        validation_metrics: Phase2SelectionEvaluationMetrics,
    ) -> bool:
        """根据验证收益、label consistency 和风险指标判断是否保存 checkpoint。"""
```

说明：

- 工程第一版按 epoch 控制训练循环，每轮执行 horizon transition 采样和若干次 Q-network update；
- horizon 环境一步结束，但 replay 仍保留下一条可训练样本的
  `next_visible_states`，用于跨 horizon bootstrap；
- `update_q_network()` 必须分别记录 `td_loss`、`imitation_loss`、`total_loss`、
  `mean_q_selected`、`mean_td_target`、`epsilon`、`mean_horizon_reward`；
- target network 按 `target_update_interval_epochs` 周期性硬同步，第一版不做 soft update。

### 7.13 `src/phase2/evaluators/phase2_evaluator.py`

```python
class Phase2Evaluator:
    def __init__(
        self,
        selection_evaluator: Phase2SelectionEvaluator,
    ) -> None:
        """保存 Phase II 各专项 evaluator，作为主流程统一评估入口。"""

    def evaluate(
        self,
        dataset: Phase2SelectionDataset,
        deterministic: bool = True,
    ) -> Phase2SelectionEvaluationMetrics:
        """调用 selection evaluator，返回 Phase II 统一评估指标。"""
```

说明：

- 对齐 Phase I 的 `phase1_evaluator.py`：主流程只依赖基础 evaluator；
- 后续如果 Phase II 增加风险、漂移或归因评估，可以继续拆到独立专项 evaluator。

### 7.14 `src/phase2/evaluators/phase2_selection_evaluator.py`

```python
class Phase2SelectionEvaluator:
    def __init__(
        self,
        q_network: Phase2QNetwork,
        decoder_policy: FrozenArchetypeDecoderPolicy,
        reward_config: Phase2RewardConfig,
        device: torch.device | str,
    ) -> None:
        """初始化评估依赖。"""

    def evaluate(
        self,
        dataset: Phase2SelectionDataset,
        deterministic: bool = True,
    ) -> Phase2SelectionEvaluationMetrics:
        """用 greedy Q action 评估 selector 在一个 split 上的 horizon return 和 label consistency。"""

    def evaluate_oracle_assigned_labels(
        self,
        dataset: Phase2SelectionDataset,
    ) -> float:
        """执行 Phase I assigned labels 对应 decoder actions，作为 imitation baseline。"""

    def evaluate_random_labels(
        self,
        dataset: Phase2SelectionDataset,
        seed: int,
    ) -> float:
        """随机选择 archetype，作为 lower baseline。"""

    def compute_code_usage_entropy(
        self,
        selected_code_ids: np.ndarray,
        num_archetypes: int,
    ) -> float:
        """统计 selector 实际使用 code 的均匀度，识别 collapse。"""

    def build_confusion_matrix(
        self,
        selected_code_ids: np.ndarray,
        assigned_labels: np.ndarray,
    ) -> np.ndarray:
        """生成 selected code 与 assigned label 的混淆矩阵。"""
```

说明：

- validation 选择 checkpoint 时使用 greedy Q action；
- 报告中同时呈现 selector return、assigned-label baseline、random baseline；
- `label_top1_match` 不是唯一目标，selector 可以为了收益偏离 assigned label，但偏离需要被记录。

### 7.15 `src/phase2/checkpoint/phase2_checkpoint.py`

```python
@dataclass(frozen=True)
class Phase2SelectionCheckpointMetadata:
    pair: str
    train_batch_id: str
    phase1_checkpoint_path: str
    phase1_checkpoint_hash: str | None
    epoch: int
    validation_metrics: Phase2SelectionEvaluationMetrics

@dataclass(frozen=True)
class Phase2Checkpoint:
    epoch: int
    is_best: bool
    config: Mapping[str, Any]
    q_network_state_dict: Mapping[str, Any]
    optimizer_state_dict: Mapping[str, Any]
    metadata: Phase2SelectionCheckpointMetadata

    def to_dict(self) -> dict[str, object]:
        """转换为 torch.save 友好的 checkpoint payload。"""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2Checkpoint":
        """从 checkpoint payload 恢复强类型对象。"""
```

### 7.16 `src/phase2/checkpoint/phase2_checkpoint_selector.py`

```python
class Phase2CheckpointSelector:
    def select_best_from_dir(
        self,
        checkpoint_dir: Path,
    ) -> Path:
        """按 validation mean_return 和风险/一致性 tie-breaker 选择最佳 checkpoint。"""
```

说明：

- checkpoint 必须记录 Phase I checkpoint 路径或 hash；
- 不能只保存 q-network 权重，否则无法确认 decoder/codebook 来源；
- `select_best_from_dir()` 不重新跑评估，只读取 checkpoint metadata。

### 7.17 `src/phase2/phase2_artifact_store.py`

```python
class Phase2ArtifactStore(DataFileStore):
    def __init__(
        self,
        pair: str,
        train_batch_id: str,
        artifacts_root: Path,
    ) -> None:
        """初始化 Phase II 产物根目录。"""

    def initialize_phase2_artifact_dirs(self) -> None:
        """初始化 Phase II 训练产出物目录，并写入标准 artifact path keys。"""

    def save_dataset_cache(
        self,
        split_name: str,
        dataset: Phase2SelectionDataset,
    ) -> Path:
        """保存 Phase II dataset cache。"""

    def load_dataset_cache(
        self,
        split_name: str,
    ) -> Phase2SelectionDataset:
        """读取 Phase II dataset cache。"""

    def save_metrics(
        self,
        split_name: str,
        metrics: Phase2SelectionEvaluationMetrics,
    ) -> Path:
        """保存评估指标 JSON。"""

    def save_phase2_checkpoint(
        self,
        checkpoint: Phase2Checkpoint,
    ) -> Path:
        """保存 Phase II checkpoint payload，对齐 Phase1ArtifactStore.save_phase1_checkpoint()。"""

    def load_phase2_checkpoint(
        self,
        *,
        epoch: int | None = None,
        best: bool = False,
    ) -> Phase2Checkpoint:
        """读取 Phase II checkpoint payload，对齐 Phase1ArtifactStore.load_phase1_checkpoint()。"""
```

说明：

- 推荐路径：`artifacts/{pair}/{batch_id}/phase2/`;
- dataset cache 只缓存 Phase II 所需数组，不重复保存 Phase I model；
- metrics JSON 用于 report 和 checkpoint selector 复用。

### 7.18 `src/phase2/phase2_main.py`

```python
class Phase2MainFlow:
    def __init__(self, config: Phase2MainConfig) -> None:
        """保存主配置并创建 artifact store。"""

    def run(self) -> None:
        """执行 Phase II 完整训练流程。"""

    def _load_phase1_model(self) -> ArchetypeVQModel:
        """加载并冻结 Phase I best checkpoint。"""

    def _build_or_load_dataset(
        self,
        split_name: str,
    ) -> Phase2SelectionDataset:
        """优先读取 Phase II dataset cache；不存在时从 horizon 和 Phase I label 产物构建。"""

    def _create_q_network(
        self,
        dataset: Phase2SelectionDataset,
    ) -> Phase2QNetwork:
        """根据 visible state shape 和 num_archetypes 创建 selector。"""

    def _train_double_dqn(
        self,
        q_network: Phase2QNetwork,
        train_dataset: Phase2SelectionDataset,
        val_dataset: Phase2SelectionDataset,
        phase1_model: ArchetypeVQModel,
    ) -> Phase2TrainingResult:
        """创建 online/target q-network、env、replay buffer、trainer/evaluator 并运行训练。"""

    def _write_final_report(
        self,
        result: Phase2TrainingResult,
    ) -> None:
        """写出 Phase II selection report。"""
```

说明：

- `run()` 是唯一对外主入口；
- 私有方法只负责编排，不放置模型细节；
- 所有路径通过 `Phase2ArtifactStore` 获取。

### 7.19 `src/phase2/report/phase2_selection_report.py`

```python
class Phase2SelectionReport:
    def write_report(
        self,
        output_dir: Path,
        training_result: Phase2TrainingResult,
        validation_metrics: Phase2SelectionEvaluationMetrics,
        test_metrics: Phase2SelectionEvaluationMetrics | None = None,
    ) -> None:
        """写出 JSON 和 HTML 报告。"""

    def build_context(
        self,
        training_result: Phase2TrainingResult,
        validation_metrics: Phase2SelectionEvaluationMetrics,
        test_metrics: Phase2SelectionEvaluationMetrics | None,
    ) -> dict[str, Any]:
        """构建模板上下文。"""

    def render_html(
        self,
        context: dict[str, Any],
    ) -> str:
        """渲染 HTML 报告。"""
```

说明：

- report 只消费训练/评估结果，不重新解码、不重新执行交易；
- 第一版至少输出 `phase2_selection_report.json`；
- HTML 可复用 Phase I report 的 `_template.py` 工具风格。

## 8. 最小测试清单

| 测试文件 | 验证内容 |
|---|---|
| `tests/test_phase2_selection_dataset_label_contract.py` | `previous_t_states == original_horizon_states[:-1]`、`current_t_states == original_horizon_states[1:, :tsize, :]`、返回的 `sample_ids/code_labels == sorted_labels[1:]`，并验证缺列、重复 sample_id、非连续 sample_id、null code_label 和未来泄漏校验。 |
| `tests/test_phase2_decoder_policy.py` | frozen decoder 对同一 code id 输出稳定动作，动作形状 `[batch, H]` 且取值在 `{0,1,2}`。 |
| `tests/test_phase2_double_dqn_loss.py` | Double DQN target 使用 online argmax 和 target gather，done 样本不 bootstrap。 |
| `tests/test_phase2_env_reward_contract.py` | env `step()` 一步结束，reward 等于 `ActionExecutionCalculator` 的 horizon return。 |
| `tests/test_phase2_replay_buffer.py` | replay buffer add/sample 的 shape、dtype、容量覆盖和 seed 可复现性正确。 |
| `tests/test_phase2_selection_metrics_schema.py` | selection metrics 和 training result 支持稳定 `to_dict()` / `from_dict()` 或等价序列化。 |
| `tests/test_phase2_q_network_forward.py` | q-network forward/select_action 输出 shape 正确，Q value 最后一维等于 `num_archetypes`。 |

## 9. 实现顺序建议

1. 实现 `phase2_config.py`、`phase2_selection_data_schema.py` 和 `metrics/phase2_selection_metrics.py`，稳定数据与指标契约；
2. 实现 `phase2_selection_dataset.py`，读取/校验 Phase I exported labels 并做 no-leakage 测试；
3. 实现 `model/phase2_decoder_policy.py` 和 `phase2_env.py`，打通 code id 到 horizon reward；
4. 实现 `model/phase2_q_network.py`，完成 Q-network forward/select_action 单测；
5. 实现 `rl/phase2_replay_buffer.py`、`rl/phase2_double_dqn_loss.py` 和 `rl/phase2_double_dqn_trainer.py` 的最小 Double DQN 更新；
6. 实现 `evaluators/phase2_selection_evaluator.py`、`checkpoint/phase2_checkpoint.py`、`checkpoint/phase2_checkpoint_selector.py` 和 `report/phase2_selection_report.py`；
7. 增加 `scripts/train_phase2.py`，接入完整训练流程。

## 10. 关键约束

- Phase I model 在 Phase II 必须冻结；
- selector observation 只允许包含上一分片完整状态序列和当前分片前 `TSize` 个状态；
- DP teacher 和 VQ encoder 只允许在 Phase I 离线流程生成 label；Phase II dataset builder 不调用它们；
- reward 口径必须复用 `ActionExecutionCalculator`；
- checkpoint 必须记录 Phase I checkpoint lineage；
- validation/test 指标必须区分 trading return 和 label imitation quality。
