# Phase I Metrics Submodule Technical Design

本文档描述 `src/phase1/metrics` 子模块的技术设计，以及它和
`evaluators`、`checkpoint`、`report` 目录的协作边界。

本文档是 `docs/design/phase1_codebook_validation_criteria.md` 的工程实现设计。
前者定义“要验证什么、阈值是什么”，本文档定义“代码如何组织、数据如何流动、
结果如何落盘和被 report/selector 使用”。

## 1. 设计目标

Phase I codebook validation 需要回答五类问题：

0. DP teacher 数据是否值得学习；
1. VQ codebook 是否稳定、未塌缩、可用；
2. 每个 archetype 是否有清晰、可区分的行为含义；
3. oracle assigned-label decoder 是否保留了 DP 盈利能力；
4. assigned label 是否能被 Phase II selector 从可见状态中学习。

工程实现需要同时满足以下目标：

- `evaluator` 负责计算和收集中间数据；
- `metrics` 负责指标数据结构、阈值配置、判定规则和每个指标的判定结果；
- `report` 只负责最终报表呈现，不重新计算核心指标；
- `checkpoint selector` 只消费 metrics 结果，不理解每个指标的内部计算细节；
- 所有 checkpoint 中保存的 validation 结果可复现、可审计、可序列化。

## 2. 非目标

本设计不负责：

- 重写 Phase I VQ 模型结构；
- 替代 Phase II selector validation；
- 在 report 层重新实现交易收益、motif、morphology 等计算；
- 在 metrics 层直接调用 model 或 dataloader；
- 在 checkpoint selector 中散落五层过滤规则。

## 3. 当前代码上下文

当前相关目录如下：

```text
src/phase1/
  evaluators/
    phase1_evaluator.py
  metrics/
    __init__.py
    phase1_metrics.py
  checkpoint/
    phase1_checkpoint.py
    phase1_checkpoint_selector.py
  report/
    phase1_report.py
  phase1_main.py
```

现状：

- `Phase1MainFlow._run_epoch()` 和 `Phase1Evaluator.evaluate()` 已经引用
  `Phase1Metrics`；
- `metrics/phase1_metrics.py` 当前尚未实现；
- `Phase1CheckpointSelector` 当前仍是骨架，默认按单个 metric 选择；
- `Phase1Report` 当前仍是 report 写出骨架；
- `phase1_codebook_validation_criteria.md` 已经定义完整五层验证标准。

因此实现顺序应先让 `metrics` 的数据契约稳定，再扩展 evaluator 和 selector。

## 4. 模块职责边界

### 4.1 evaluators

`src/phase1/evaluators` 负责：

- 遍历 `DataLoader`；
- 调用 `ArchetypeVQModel.forward()` / `encode()` / `decode()`；
- 收集中间张量，包括 logits、decoded actions、code id、latent、distance 等；
- 基于收集的数据计算 raw metric values；
- 调用 `metrics` 中的判定规则，生成结构化 validation result；
- 不负责 HTML/JSON report 的排版；
- 不在内部硬编码 report 样式。

### 4.2 metrics

`src/phase1/metrics` 负责：

- 定义训练期基础指标 `Phase1Metrics`；
- 定义 validation 中间数据 schema；
- 定义 metric result / layer result / checkpoint validation result；
- 定义阈值、评分权重和 metric name 常量；
- 定义每层 hard gate 判定规则；
- 定义综合评分规则；
- 提供稳定的 `to_dict()` / `from_dict()` 序列化契约；
- 不直接依赖 `torch.nn.Module`、`DataLoader` 或文件系统。

### 4.3 report

`src/phase1/report` 负责：

- 消费 `Phase1ValidationResult`；
- 写出 JSON report；
- 后续可渲染 HTML report；
- 呈现每个 metric 的 value、threshold、status、message；
- 呈现 code-level diagnostics 和 drift diagnostics；
- 不重新计算 metric；
- 不改写 checkpoint selection 逻辑。

### 4.4 checkpoint

`src/phase1/checkpoint` 负责：

- 将 validation result 随 checkpoint 一起保存；
- 读取 checkpoint 中的 `validation.passed`、`validation.score` 和 tie-breaker 字段；
- 从通过 hard gates 的候选中选 score 最高者；
- 如果没有候选通过，返回失败摘要或抛出可解释错误；
- 不重新实现五层判定规则。

## 5. 推荐文件结构

```text
src/phase1/metrics/
  __init__.py
  phase1_metrics.py
  phase1_validation_config.py
  phase1_validation_data.py
  phase1_metric_results.py
  phase1_validation_rules.py
  phase1_validation_score.py

src/phase1/evaluators/
  __init__.py
  phase1_evaluator.py
  phase1_codebook_evaluator.py
  phase1_validation_layers/
    __init__.py
    layer0_teacher_quality.py
    layer1_vq_internal.py
    layer2_behavior_quality.py
    layer3_oracle_profitability.py
    layer4_label_predictability.py

src/phase1/report/
  __init__.py
  phase1_report.py
  phase1_codebook_report.py
```

说明：

- `phase1_metrics.py` 保留给训练期 loss/accuracy 指标，兼容现有调用；
- `phase1_validation_config.py` 放阈值、权重和开关；
- `phase1_validation_data.py` 放 evaluator 收集到的中间数据结构；
- `phase1_metric_results.py` 放判定结果结构；
- `phase1_validation_rules.py` 放 hard gate 规则；
- `phase1_validation_score.py` 放 normalized score 和 tie-breaker 所需字段；
- `phase1_codebook_evaluator.py` 是完整五层 validation 的统一编排入口；
- `phase1_validation_layers/layer*.py` 是五层 raw metric 的独立计算文件；
- `phase1_codebook_report.py` 是后续 HTML/JSON 呈现入口。

## 6. 核心数据流

训练期每个 epoch 的基础流程：

```text
Phase1MainFlow.train()
  -> _run_epoch(train_loader)
       -> Phase1Metrics
  -> Phase1Evaluator.evaluate(val_loader)
       -> Phase1Metrics
  -> Phase1CodebookEvaluator.evaluate_checkpoint(...)
       -> Phase1ValidationResult
  -> Phase1ArtifactStore.save_phase1_checkpoint(metrics={...})
```

checkpoint 选择流程：

```text
Phase1CheckpointSelector.select_best_from_dir()
  -> load checkpoint metrics
  -> require validation.passed == true
  -> sort by validation.score desc
  -> apply tie-breaker
  -> return Phase1CheckpointSelectionResult
```

report 流程：

```text
Phase1Report.write_report()
  -> load best checkpoint / selection result
  -> read Phase1ValidationResult dict
  -> Phase1CodebookReport render JSON/HTML
```

## 7. 训练期指标设计

`Phase1Metrics` 用于 `_run_epoch()` 和 `Phase1Evaluator.evaluate()`。

建议位置：

```text
src/phase1/metrics/phase1_metrics.py
```

建议职责：

- 聚合 batch-level loss；
- 计算 action reconstruction accuracy；
- 支持 `add_batch()`、`averaged()`、`to_dict()`；
- 字段保持简单，避免承载五层 validation 结果。

建议字段：

```python
@dataclass
class Phase1Metrics:
    stage: str | None = None
    split: str | None = None
    epoch: int | None = None
    num_samples: int = 0
    total_loss: float = 0.0
    reconstruction_loss: float = 0.0
    vq_loss: float = 0.0
    codebook_loss: float = 0.0
    commitment_loss: float = 0.0
    action_accuracy: float = 0.0
```

`add_batch()` 接收当前模型 `VqModelOutputs`：

```python
def add_batch(self, batch_size: int, outputs: VqModelOutputs) -> None:
    ...
```

注意：

- loss 需要按 sample 加权累加，再在 `averaged()` 中除以 `num_samples`；
- action accuracy 按 timestep 统计，不能简单按 batch 平均；
- `to_dict(include_context=True)` 需要包含 `stage/split/epoch/num_samples`；
- `to_dict(include_context=False)` 只输出指标字段，便于嵌入其他 payload。

## 8. Validation 配置设计

建议位置：

```text
src/phase1/metrics/phase1_validation_config.py
```

配置分为三类：

- 阈值；
- 综合评分权重；
- evaluator 运行参数。

建议结构：

```python
@dataclass(frozen=True)
class Phase1ValidationThresholds:
    dp_win_rate_min: float = 0.58
    near_zero_opportunity_ratio_max: float = 0.35
    fee_sensitivity_min: float = 0.60
    morphology_coverage_min: float = 0.60

    action_accuracy_min: float = 0.85
    reconstruction_loss_gap_max: float = 1.25
    active_code_ratio_min: float = 0.80
    max_code_occupancy_max: float = 0.40
    normalized_perplexity_min: float = 0.50
    normalized_perplexity_max: float = 0.90
    dead_code_ratio_max: float = 0.20
    churn_recent_mean_max: float = 0.15
    margin_median_min: float = 0.10
    direction_accuracy_min: float = 0.88
    entry_timing_error_ratio_max: float = 0.15

    min_code_support_abs: int = 100
    min_code_support_ratio: float = 0.02
    weak_code_ratio_max: float = 0.20
    dominant_morphology_ratio_min: float = 0.35
    dominant_motif_ratio_min: float = 0.40
    dominant_pair_ratio_min: float = 0.30
    morphology_lift_min: float = 1.25
    intra_code_similarity_min: float = 0.65
    inter_intra_separation_min: float = 1.30
    duplicate_code_similarity_max: float = 0.85

    decoded_win_rate_min: float = 0.55
    retention_ratio_min: float = 0.50
    random_label_relative_lift_min: float = 0.20
    bad_code_ratio_max: float = 0.30
    top_5_contribution_max: float = 0.60
    dominant_pair_positive_ratio_min: float = 0.60

    probe_top1_floor: float = 0.25
    probe_top1_k_factor: float = 1.5
    probe_top3_floor: float = 0.55
    probe_top3_k_factor: float = 3.0
    probe_balanced_accuracy_min: float = 0.25
    mutual_information_lift_min: float = 2.0
    probe_return_retention_min: float = 0.35
```

评分权重：

```python
@dataclass(frozen=True)
class Phase1ValidationScoreWeights:
    teacher_quality: float = 0.10
    reconstruction: float = 0.20
    codebook_health: float = 0.15
    behavior_structure: float = 0.20
    oracle_profitability: float = 0.25
    label_predictability: float = 0.10
```

总配置：

```python
@dataclass(frozen=True)
class Phase1ValidationConfig:
    thresholds: Phase1ValidationThresholds = Phase1ValidationThresholds()
    score_weights: Phase1ValidationScoreWeights = Phase1ValidationScoreWeights()
    fee_rate: float = 0.0002
    random_label_trials: int = 3
    churn_window_epochs: int = 5
    active_code_min_occupancy: float = 0.01
    dead_code_max_occupancy: float = 0.001
    top_contribution_ratio: float = 0.05
    probe_epochs: int = 20
    probe_learning_rate: float = 1e-3
    probe_batch_size: int = 256
```

## 9. Validation 中间数据设计

建议位置：

```text
src/phase1/metrics/phase1_validation_data.py
```

### 9.1 Split snapshot

`Phase1EvaluationSnapshot` 是 evaluator 的核心输出。它描述一个 split 在某个
checkpoint 下的完整可计算状态。

```python
@dataclass(frozen=True)
class Phase1EvaluationSnapshot:
    split: str
    epoch: int
    sample_ids: np.ndarray
    states: np.ndarray
    prices: np.ndarray | None
    demo_actions: np.ndarray
    demo_rewards: np.ndarray
    decoded_actions: np.ndarray
    decoded_logits: np.ndarray
    code_ids: np.ndarray
    z_e: np.ndarray
    z_q: np.ndarray
    distances: np.ndarray
    reconstruction_loss: float
    action_accuracy: float
```

说明：

- `sample_ids` 用于 assignment churn，必须在 train/val 内稳定；
- `prices` 来自 `HorizonDataset`，如果没有价格，则第三层盈利指标无法完整计算；
- `distances` 来自 quantizer 到 codebook 的距离，用于 margin 和 quantization diagnostics；
- `decoded_actions` 为 `argmax(decoded_logits)`，用于 action stats、motif、profitability。

### 9.2 Assignment history

```python
@dataclass(frozen=True)
class CodeAssignmentSnapshot:
    epoch: int
    split: str
    sample_ids: np.ndarray
    code_ids: np.ndarray
    active_codes: tuple[int, ...]
```

用途：

- 计算相邻 epoch assignment churn；
- 计算 active code lifetime；
- 诊断 codebook 是否仍在重排。

### 9.3 Code diagnostics

```python
@dataclass(frozen=True)
class Phase1CodeDiagnostic:
    code_id: int
    support: int
    occupancy: float
    dominant_morphology: str | None
    dominant_morphology_ratio: float | None
    morphology_lift: float | None
    dominant_motif: str | None
    dominant_motif_ratio: float | None
    dominant_pair: str | None
    dominant_pair_ratio: float | None
    decoded_mean_advantage: float | None
    decoded_win_rate: float | None
    retention_ratio: float | None
    fee_drag: float | None
    status: str
```

该结构直接供 report 呈现 code-level 表格。

## 10. Metric 判定结果设计

建议位置：

```text
src/phase1/metrics/phase1_metric_results.py
```

### 10.1 Metric status

```python
MetricSeverity = Literal["pass", "warn", "fail", "skip"]
```

- `pass`：指标通过；
- `warn`：触发警戒，但不直接淘汰；
- `fail`：触发 hard gate 失败；
- `skip`：缺少必要输入，无法计算。对于 hard gate 指标，`skip` 默认视为失败；
  对 drift diagnostics 可视为 warning。

### 10.2 Metric result

```python
@dataclass(frozen=True)
class Phase1MetricResult:
    name: str
    value: int | float | str | bool | None
    threshold: str
    severity: MetricSeverity
    passed: bool
    layer: str
    message: str = ""
```

约定：

- `name` 使用稳定 snake_case，例如 `validation_action_accuracy`；
- `threshold` 使用人类可读字符串，例如 `">= 0.85"`；
- `passed` 只表示该指标是否满足 hard gate；
- `severity == "warn"` 时 `passed` 可以为 `True`；
- `message` 用于 report 中解释失败原因。

### 10.3 Layer result

```python
@dataclass(frozen=True)
class Phase1LayerResult:
    layer_id: int
    name: str
    passed: bool
    metrics: tuple[Phase1MetricResult, ...]
```

### 10.4 Checkpoint validation result

```python
@dataclass(frozen=True)
class Phase1ValidationResult:
    checkpoint_id: str
    stage: str
    epoch: int
    passed: bool
    score: float | None
    failed_layers: tuple[str, ...]
    layers: tuple[Phase1LayerResult, ...]
    scalar_metrics: Mapping[str, float]
    code_diagnostics: tuple[Phase1CodeDiagnostic, ...]
    drift_diagnostics: Mapping[str, Phase1MetricResult]
    tie_breaker_metrics: Mapping[str, float]
```

必须提供：

```python
def to_dict(self) -> dict[str, object]:
    ...

@classmethod
def from_dict(cls, payload: Mapping[str, object]) -> Phase1ValidationResult:
    ...
```

## 11. Metric 命名规范

checkpoint 内建议使用扁平 key 便于 selector 和 report 读取：

```text
validation.passed
validation.score
validation.failed_layer_count
validation.layer0.teacher_quality.passed
validation.layer1.vq_internal.passed
validation.layer2.behavior_quality.passed
validation.layer3.oracle_profitability.passed
validation.layer4.label_predictability.passed
validation.tie_breaker.risk_adjusted_return
validation.tie_breaker.probe_top3_accuracy
validation.tie_breaker.retention_ratio
validation.tie_breaker.active_code_ratio
validation.tie_breaker.max_code_occupancy
validation.tie_breaker.reconstruction_loss
```

完整结构化结果建议保存到：

```text
checkpoint.metrics["validation"] = validation_result.to_dict()
```

同时为了 selector 快速读取，可额外冗余几个 top-level scalar：

```python
metrics = {
    "train": train_metrics.to_dict(include_context=True),
    "val": val_metrics.to_dict(include_context=True),
    "validation": validation_result.to_dict(),
}
```

## 12. Evaluator 设计

### 12.1 Phase1Evaluator

`Phase1Evaluator` 继续保留当前职责：计算 split-level 训练损失和重构准确率。

```python
class Phase1Evaluator:
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader[TrajectoryTensorBatch],
        *,
        stage: str | None = None,
        split: str | None = None,
        epoch: int | None = None,
    ) -> Phase1Metrics:
        ...
```

### 12.2 Phase1CodebookEvaluator

新增完整 codebook validation evaluator。该类只做统一编排，不把五层指标计算
全部写在一个大文件中。五层 raw metric 计算分别委托给
`phase1_validation_layers/layer0_teacher_quality.py` 到
`phase1_validation_layers/layer4_label_predictability.py`。

建议位置：

```text
src/phase1/evaluators/phase1_codebook_evaluator.py
```

建议接口：

```python
class Phase1CodebookEvaluator:
    def __init__(
        self,
        model: ArchetypeVQModel,
        config: Phase1ValidationConfig,
        device: torch.device | str,
    ) -> None:
        ...

    @torch.no_grad()
    def collect_snapshot(
        self,
        dataloader: DataLoader[TrajectoryTensorBatch],
        *,
        split: str,
        epoch: int,
        horizon_dataset: HorizonDataset | None = None,
    ) -> Phase1EvaluationSnapshot:
        ...

    def evaluate_checkpoint(
        self,
        *,
        train_loader: DataLoader[TrajectoryTensorBatch],
        val_loader: DataLoader[TrajectoryTensorBatch],
        epoch: int,
        checkpoint_id: str,
        train_horizon_dataset: HorizonDataset | None = None,
        val_horizon_dataset: HorizonDataset | None = None,
        assignment_history: Sequence[CodeAssignmentSnapshot] = (),
    ) -> Phase1ValidationResult:
        ...
```

`evaluate_checkpoint()` 的职责：

1. 调用 `collect_snapshot()` 收集 train/val snapshot；
2. 调用五个 layer calculator，分别得到 raw scalar metrics 和 diagnostics；
3. 将 raw metrics 交给 `metrics.phase1_validation_rules` 做 hard gate 判定；
4. 将通过 hard gates 的结果交给 `metrics.phase1_validation_score` 计算 score；
5. 组装 `Phase1ValidationResult`。

该类不直接实现具体指标公式，避免单文件过大，也方便后续单独测试每一层。

### 12.3 Snapshot 收集步骤

`collect_snapshot()` 执行：

1. 遍历 dataloader；
2. 调用 `model(batch)`；
3. 从 `outputs.action_logits` 得到 `decoded_actions`；
4. 收集 `outputs.code_id`、`outputs.z_e`、`outputs.z_q`；
5. 重新调用 quantizer 或扩展模型输出拿到 `distances`；
6. 拼接全部 batch；
7. 如果传入 `horizon_dataset`，按样本顺序关联 `prices`；
8. 返回 `Phase1EvaluationSnapshot`。

注意：

- dataloader 不能在 validation snapshot 中 shuffle，否则 `sample_ids` 和
  `horizon_dataset` 无法稳定对齐；
- train loader 用于 full validation 时也建议单独构造 `shuffle=False` 版本；
- 若训练主流程中的 train loader 是 shuffle=True，codebook evaluator 应接收
  独立的 eval loader。

## 13. 五层指标计算归属

本节定义“每层 raw metric 由哪个独立文件计算、metrics 层如何判定”。

五层计算代码不放在 `phase1_codebook_evaluator.py` 中，而是放在 5 个独立文件：

```text
src/phase1/evaluators/phase1_validation_layers/
  layer0_teacher_quality.py
  layer1_vq_internal.py
  layer2_behavior_quality.py
  layer3_oracle_profitability.py
  layer4_label_predictability.py
```

统一约定：

- 每个 layer 文件只负责本层 raw metric 计算；
- 每个 layer 文件可以包含本层私有 helper；
- 每个 layer 文件返回 `Phase1LayerComputation`；
- 每个 layer 文件不返回 PASS/FAIL；
- PASS/FAIL 统一由 `metrics.phase1_validation_rules.py` 判定；
- `Phase1CodebookEvaluator` 只负责调用五个 layer calculator 并组装结果。

计算顺序可以和报告展示顺序不同。报告和 hard gate 展示仍按 Layer 0 到 Layer 4；
实际计算时建议先算 Layer 3，再把 `per_code_profitability` 传给 Layer 2，用于
`profitable_code_coverage` 和 weak-lift-but-profitable code 的判定输入：

```text
collect_snapshot
  -> layer0_teacher_quality
  -> layer1_vq_internal
  -> layer3_oracle_profitability
  -> layer2_behavior_quality
  -> layer4_label_predictability
  -> validation_rules 按 0,1,2,3,4 输出 layer results
```

建议中间返回结构定义在 `metrics/phase1_validation_data.py`：

```python
@dataclass(frozen=True)
class Phase1LayerComputation:
    layer_id: int
    layer_name: str
    scalar_metrics: Mapping[str, float]
    code_diagnostics: tuple[Phase1CodeDiagnostic, ...] = ()
    extra_payload: Mapping[str, object] = field(default_factory=dict)
```

### 13.1 `layer0_teacher_quality.py`

文件职责：

```text
src/phase1/evaluators/phase1_validation_layers/layer0_teacher_quality.py
```

计算第零层 DP teacher 质量。该层只依赖 train/val snapshot、prices、demo actions
和 demo rewards，不依赖 decoded actions 的盈利结果。

建议入口：

```python
def compute_teacher_quality_metrics(
    *,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    config: Phase1ValidationConfig,
) -> Phase1LayerComputation:
    ...
```

输出 raw metrics：

- `dp_advantage_vs_flat`
- `dp_win_rate_vs_flat`
- `near_zero_opportunity_ratio`
- `fee_sensitivity`
- `morphology_coverage`
- `dp_return_concentration_after_top5_removed`

Metrics 判定仍在 `phase1_validation_rules.py`：

- advantage `> 0`
- win rate `>= 0.58`
- near-zero ratio `<= 0.35`
- fee sensitivity `>= 0.60`
- top 5% removed advantage `> 0`

### 13.2 `layer1_vq_internal.py`

文件职责：

```text
src/phase1/evaluators/phase1_validation_layers/layer1_vq_internal.py
```

计算 VQ 内部质量。该层依赖 train/val snapshot 和 assignment history，不负责
morphology、motif 或 profitability。

建议入口：

```python
def compute_vq_internal_metrics(
    *,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    assignment_history: Sequence[CodeAssignmentSnapshot],
    config: Phase1ValidationConfig,
) -> Phase1LayerComputation:
    ...
```

输出 raw metrics：

- `validation_action_accuracy`
- `reconstruction_loss_gap`
- `active_code_ratio`
- `max_code_occupancy`
- `normalized_code_perplexity`
- `dead_code_ratio`
- `assignment_churn_recent_mean`
- `code_lifetime_pass_ratio`
- `quantization_distance`
- `nearest_second_margin_median`
- `decoder_turnover_error`
- `entry_timing_error_median`
- `direction_accuracy`

Metrics 判定仍在 `phase1_validation_rules.py`，按
`phase1_codebook_validation_criteria.md` 的第一层 hard gates 执行。

### 13.3 `layer2_behavior_quality.py`

文件职责：

```text
src/phase1/evaluators/phase1_validation_layers/layer2_behavior_quality.py
```

计算原型行为质量。该层负责 morphology/motif 统计、code-level 行为诊断和
code 间分离度；它可以消费第三层传入的 per-code profitability 摘要，但不
直接计算交易收益。

建议入口：

```python
def compute_behavior_quality_metrics(
    *,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    config: Phase1ValidationConfig,
    per_code_profitability: Mapping[int, Mapping[str, float]] | None = None,
) -> Phase1LayerComputation:
    ...
```

输出 raw metrics：

- `weak_support_code_ratio`
- `weak_morphology_code_ratio`
- `weak_motif_code_ratio`
- `weak_pair_code_ratio`
- `weak_lift_nonprofitable_code_ratio`
- `intra_code_action_similarity`
- `inter_intra_separation`
- `latent_silhouette_score`
- `duplicate_code_pair_count`
- `profitable_code_coverage`

输出 code diagnostics：

- per-code support；
- dominant morphology；
- morphology ratio；
- morphology lift；
- dominant motif；
- motif ratio；
- dominant morphology-motif pair；
- pair ratio。

Metrics 判定仍在 `phase1_validation_rules.py`：

- weak support code ratio；
- weak morphology ratio code ratio；
- weak motif ratio code ratio；
- weak pair ratio code ratio；
- weak lift and non-profitable code ratio；
- intra-code similarity；
- inter/intra separation；
- duplicate code pair count。

### 13.4 `layer3_oracle_profitability.py`

文件职责：

```text
src/phase1/evaluators/phase1_validation_layers/layer3_oracle_profitability.py
```

计算 oracle assigned-label decoded profitability。该层统一执行收益计算、
random label baseline、retention、fee drag 和 per-code profitability。

建议入口：

```python
def compute_oracle_profitability_metrics(
    *,
    model: ArchetypeVQModel,
    val_snapshot: Phase1EvaluationSnapshot,
    config: Phase1ValidationConfig,
    device: torch.device | str,
) -> Phase1LayerComputation:
    ...
```

说明：

- 该层需要 `model` 是因为 random label baseline 需要用随机 code 重新 decode；
- 如果 random label actions 已经由上层 evaluator 预先收集，也可以把入口改成
  只接收 `random_label_decoded_actions`，以减少本层对 model 的依赖；
- 收益执行口径必须集中在本文件或同文件私有 helper 中，不能分散到 report。

输出 raw metrics：

- `mean_decoded_advantage_vs_flat`
- `decoded_win_rate_vs_flat`
- `mean_advantage_vs_random_label`
- `random_label_relative_lift`
- `retention_ratio`
- `downside_control`
- `risk_adjusted_return`
- `top_5_contribution`
- `trimmed_decoded_advantage`
- `fee_drag`
- `turnover_return_correlation`
- `bad_code_ratio`
- `dominant_pair_positive_ratio`

输出 `extra_payload`：

- `per_code_profitability`
- `decoded_returns`
- `dp_returns`
- `flat_returns`
- `random_label_returns`

Metrics 判定仍在 `phase1_validation_rules.py`：

- decoded advantage `> 0`
- win rate `>= 0.55`
- random label relative lift `>= 0.20`
- retention ratio `>= 0.50`
- bad-code ratio `<= 0.30`
- top 5% contribution `<= 0.60`
- trimmed decoded advantage `> 0`
- dominant pair positive ratio `>= 0.60`

### 13.5 `layer4_label_predictability.py`

文件职责：

```text
src/phase1/evaluators/phase1_validation_layers/layer4_label_predictability.py
```

计算 assigned label 是否能从 selector 可见状态中预测。该层负责 probe training、
probe evaluation、mutual information lift 和 probe decoded return retention。

建议入口：

```python
def compute_label_predictability_metrics(
    *,
    model: ArchetypeVQModel,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    config: Phase1ValidationConfig,
    device: torch.device | str,
) -> Phase1LayerComputation:
    ...
```

说明：

- 该层可以训练轻量 probe；
- probe 训练只使用 train snapshot；
- probe evaluation 只使用 val snapshot；
- probe return retention 需要用 probe top-1 label 经 decoder 执行，因此需要
  `model` 或预先 decode 好的 probe actions；
- probe 训练必须支持 seed，避免 checkpoint score 抖动。

输出 raw metrics：

- `probe_top1_accuracy`
- `probe_top3_accuracy`
- `probe_balanced_accuracy`
- `label_entropy_given_morphology`
- `mutual_information_lift`
- `probe_return_retention`

Metrics 判定仍在 `phase1_validation_rules.py`：

- top-1 `>= max(0.25, 1.5 / K)`
- top-3 `>= max(0.55, 3.0 / K)`
- balanced accuracy `>= 0.25`
- mutual information lift `>= 2.0`
- probe return retention `>= 0.35`

## 14. Rule 层设计

建议位置：

```text
src/phase1/metrics/phase1_validation_rules.py
```

设计原则：

- 输入 raw scalar metrics；
- 输出 `Phase1LayerResult`；
- 不访问 model、dataloader、文件系统；
- 不重复计算 evaluator 已经计算过的数值；
- 对缺失 hard gate 指标返回 `fail` 或明确 `skip-as-fail`。

建议函数：

```python
def evaluate_teacher_quality_rules(
    metrics: Mapping[str, float],
    thresholds: Phase1ValidationThresholds,
) -> Phase1LayerResult:
    ...

def evaluate_vq_internal_rules(...) -> Phase1LayerResult:
    ...

def evaluate_behavior_quality_rules(...) -> Phase1LayerResult:
    ...

def evaluate_oracle_profitability_rules(...) -> Phase1LayerResult:
    ...

def evaluate_label_predictability_rules(...) -> Phase1LayerResult:
    ...

def aggregate_validation_result(
    *,
    checkpoint_id: str,
    stage: str,
    epoch: int,
    layers: Sequence[Phase1LayerResult],
    scalar_metrics: Mapping[str, float],
    code_diagnostics: Sequence[Phase1CodeDiagnostic],
    drift_diagnostics: Mapping[str, Phase1MetricResult],
    score: float | None,
    tie_breaker_metrics: Mapping[str, float],
) -> Phase1ValidationResult:
    ...
```

## 15. Scoring 设计

建议位置：

```text
src/phase1/metrics/phase1_validation_score.py
```

只有所有 hard gates 通过时才计算最终 score。若未通过，`score=None` 或 `0.0`，
推荐使用 `None`，避免误把失败 checkpoint 排进候选。

建议函数：

```python
def compute_phase1_validation_score(
    metrics: Mapping[str, float],
    weights: Phase1ValidationScoreWeights,
) -> float:
    ...
```

子分数：

- `teacher_quality_score`
- `reconstruction_score`
- `codebook_health_score`
- `behavior_structure_score`
- `oracle_profitability_score`
- `label_predictability_score`

每个子分数归一化到 `[0, 1]`，最后按文档权重加权。

### 15.1 Tie-breaker 说明

`tie_breaker` 是 checkpoint 综合分接近时的决胜规则。它不替代 hard gate，
也不替代主排序分数 `validation.score`；只有多个 checkpoint 都通过五层 hard
gate，且综合分差距小于 `tie_score_tolerance` 时才启用。

需要 tie-breaker 的原因：

- normalized score 会压缩不同指标的尺度，两个 checkpoint 分数可能非常接近；
- probe training 和 random label baseline 存在少量随机性；
- 训练后期相邻 epoch 的 codebook 可能都合格，但侧重点不同；
- 当总分接近时，应优先选择更符合 Phase II 使用目标的 checkpoint。

默认触发条件：

```text
abs(best_score - candidate_score) < tie_score_tolerance
```

其中 `tie_score_tolerance` 默认 `0.03`，表示综合分差距小于 3% 时认为二者接近。

比较顺序：

1. `risk_adjusted_return` 更高者优先；
2. `probe_top3_accuracy` 更高者优先；
3. `retention_ratio` 更高者优先；
4. `active_code_ratio` 更高者优先；
5. `max_code_occupancy` 更低者优先；
6. `reconstruction_loss` 更低者优先。

这个顺序体现的业务优先级是：当综合分几乎一样时，先选 oracle decoded
收益质量更好的 checkpoint；如果收益质量仍接近，再选 label 更容易被 selector
学习的 checkpoint；随后比较 DP 盈利保留、codebook 使用健康度和基础重构质量。

示例：

```text
checkpoint A: validation.score = 0.842
checkpoint B: validation.score = 0.836
tie_score_tolerance = 0.03
```

两者差距为 `0.006`，小于 `0.03`，因此进入 tie-breaker。如果 B 的
`risk_adjusted_return` 高于 A，则选择 B，即使 A 的综合分略高。

Tie-breaker 字段：

```python
tie_breaker_metrics = {
    "risk_adjusted_return": ...,
    "probe_top3_accuracy": ...,
    "retention_ratio": ...,
    "active_code_ratio": ...,
    "max_code_occupancy": ...,
    "reconstruction_loss": ...,
}
```

## 16. Report 设计

`report` 不重新计算指标。它只消费：

```python
Phase1ValidationResult
Phase1CheckpointSelectionResult
config snapshot
artifact index
```

推荐新增：

```text
src/phase1/report/phase1_codebook_report.py
```

建议接口：

```python
class Phase1CodebookReport:
    def build_payload(
        self,
        *,
        validation_result: Phase1ValidationResult,
        config: Mapping[str, object],
        artifacts: Mapping[str, Path],
    ) -> dict[str, object]:
        ...

    def write_json(...):
        ...

    def write_html(...):
        ...
```

`Phase1Report.write_report()` 可以作为总入口，内部委托
`Phase1CodebookReport`。

## 17. Phase1MainFlow 集成设计

### 17.1 构建组件

在 `build_components()` 中新增：

```python
self.codebook_evaluator = Phase1CodebookEvaluator(
    model=model,
    config=Phase1ValidationConfig(),
    device=self.device,
)
```

### 17.2 eval dataloader

当前 train dataloader `shuffle=True`，不适合 snapshot。建议额外维护：

```python
self.eval_dataloaders: dict[str, DataLoader[TrajectoryTensorBatch]]
```

其中所有 split 都 `shuffle=False`。

### 17.3 train 阶段

每个 epoch：

```python
train_metrics = self._run_epoch(...)
val_metrics = self.evaluator.evaluate(...)
validation_result = self.codebook_evaluator.evaluate_checkpoint(
    train_loader=self.eval_dataloaders["train"],
    val_loader=self.eval_dataloaders["val"],
    train_horizon_dataset=self.horizon_datasets.get("train"),
    val_horizon_dataset=self.horizon_datasets.get("val"),
    assignment_history=self.assignment_history,
    checkpoint_id=f"vq_epoch_{epoch:04d}",
    epoch=epoch,
)
```

保存 checkpoint：

```python
metrics = {
    "train": train_metrics.to_dict(include_context=True),
    "val": val_metrics.to_dict(include_context=True),
    "validation": validation_result.to_dict(),
}
```

更新 history：

```python
self.assignment_history.append(
    validation_result.to_assignment_snapshot(...)
)
```

或者由 evaluator 单独返回 `CodeAssignmentSnapshot`。

## 18. Checkpoint selector 集成设计

`Phase1CheckpointSelectionConfig` 建议扩展：

```python
@dataclass(frozen=True)
class Phase1CheckpointSelectionConfig:
    stage: Phase1CheckpointStage = "vq"
    split: str = "validation"
    metric_name: str = "score"
    metric_mode: Phase1CheckpointMetricMode = "max"
    require_validation_passed: bool = True
    tie_score_tolerance: float = 0.03
```

选择规则：

1. 只扫描 `stage == "vq"` 的 checkpoint；
2. 读取 `checkpoint.metrics["validation"]`；
3. 若 `require_validation_passed=True`，过滤 `passed != True`；
4. 按 `score` 降序；
5. 若最高分差距 `< tie_score_tolerance`，按 tie-breaker：
   - `risk_adjusted_return` 更高；
   - `probe_top3_accuracy` 更高；
   - `retention_ratio` 更高；
   - `active_code_ratio` 更高；
   - `max_code_occupancy` 更低；
   - `reconstruction_loss` 更低。

Tie-breaker 只在候选 checkpoint 的综合分接近时启用。它的作用是让选择结果在
相邻 epoch 分数接近、probe/random baseline 有轻微噪声时保持稳定，并把业务
优先级固定为：收益质量优先，其次 label 可学习性，再次 codebook 健康和重构质量。
如果所有 tie-breaker 字段仍完全相同，则可选择 epoch 更早者，以减少训练后期
codebook 继续漂移带来的风险。

没有通过候选时：

- 不应静默回退到最低 loss checkpoint；
- 应返回明确错误或 `Phase1CheckpointSelectionResult` 中带失败摘要；
- Phase I 主流程应阻断 Phase II artifact export。

## 19. 交易收益计算设计

第三层 profitability 需要统一 execution 语义。建议 evaluator 内部实现纯函数，
后续可迁移到 shared utility。

输入：

```python
prices: np.ndarray       # [n, h] or [n, h, 1]
actions: np.ndarray      # [n, h], values in {0, 1, 2}
fee_rate: float
```

动作映射：

```python
0 -> -1  # short
1 -> 0   # flat
2 -> 1   # long
```

收益建议：

```text
position_t = action_to_position[action_t]
bar_return_t = price_{t+1} / price_t - 1
gross_return_t = position_t * bar_return_t
turnover_t = abs(position_t - position_{t-1})
fee_t = turnover_t * fee_rate
net_return_t = gross_return_t - fee_t
```

注意：

- horizon 最后一个 action 没有下一根 bar，可不计收益；
- 初始 position 默认为 flat；
- 若上游 DP planner 的 reward 定义不同，需要在 config 中记录并统一；
- report 中必须注明 execution 口径。

## 20. 测试策略

### 20.1 metrics 单元测试

覆盖：

- `Phase1Metrics.add_batch()` 加权平均；
- action accuracy 按 timestep 统计；
- `Phase1MetricResult.to_dict()/from_dict()`；
- 每层 rules 的 pass/fail 边界；
- score 权重总和和归一化截断。

### 20.2 evaluator 单元测试

使用小型 fake model / fixed tensors：

- snapshot shape 正确；
- code occupancy/perplexity 正确；
- nearest/second margin 正确；
- motif/morphology 分类稳定；
- profitability execution 口径正确；
- random label baseline 可设置 seed 保持确定性。

### 20.3 integration 测试

覆盖：

- train epoch 保存 checkpoint metrics 包含 `validation`；
- selector 过滤失败 checkpoint；
- selector 对通过 checkpoint 按 score 选择；
- report 可以从 `Phase1ValidationResult` 写 JSON。

## 21. 分阶段实现计划

### Phase A: 打通基础训练指标

实现：

- `phase1_metrics.py`
- `metrics/__init__.py`

目标：

- `_run_epoch()` 和 `Phase1Evaluator.evaluate()` 可正常返回基础 loss 指标；
- checkpoint 可以保存 train/val metrics。

### Phase B: 建立 validation 数据契约

实现：

- `phase1_validation_config.py`
- `phase1_validation_data.py`
- `phase1_metric_results.py`
- `phase1_validation_rules.py` 的空规则框架和少量核心规则。

目标：

- 可以构造 `Phase1ValidationResult`；
- 可以序列化进 checkpoint；
- report/selector 有稳定读取对象。

### Phase C: 实现 VQ 内部质量

实现第一层指标：

- action accuracy；
- loss gap；
- occupancy；
- perplexity；
- dead code；
- quantization distance；
- nearest/second margin；
- direction accuracy；
- entry timing error。

目标：

- selector 可以先基于第一层 hard gate 和基础 score 工作。

### Phase D: 实现 behavior 和 profitability

实现：

- morphology；
- motif；
- code-level diagnostics；
- decoded return；
- random label baseline；
- retention；
- per-code profitability。

目标：

- 覆盖第二层和第三层 hard gates；
- report 可以展示 code-level 表格。

### Phase E: 实现 label predictability 和 drift

实现：

- probe training；
- mutual information lift；
- probe return retention；
- train/val KL drift；
- predictability gap。

目标：

- 五层 validation 完整闭环；
- checkpoint selection 完全按设计文档执行。

### Phase F: HTML report

实现：

- `phase1_codebook_report.py`
- JSON payload；
- 可选 HTML 模板。

目标：

- 对齐 `phase1_codebook_validation_report_sample.html` 的呈现结构。

## 22. 风险与约束

### 22.1 样本顺序

assignment churn、prices 对齐、horizon diagnostics 都依赖稳定样本顺序。
validation evaluator 必须使用 `shuffle=False` dataloader。

### 22.2 缺少 prices

如果没有 `HorizonDataset.prices`，第三层 profitability 无法完整计算。
对于 hard gate 指标，缺失输入应视为失败，并在 report 中明确写出。

### 22.3 计算成本

完整五层 validation 比普通 loss evaluation 更重。可以通过 config 控制：

- 每几个 epoch 跑一次完整 validation；
- 其他 epoch 只跑基础 metrics；
- probe 和 random-label baseline 支持 sample 限制。

### 22.4 随机性

random label baseline 和 probe training 必须支持 seed，避免 checkpoint 选择不稳定。

### 22.5 checkpoint 兼容

旧 checkpoint 可能没有 `metrics["validation"]`。
selector 应明确报错或跳过，不能把旧 checkpoint 当成通过 validation。

## 23. 最终边界总结

推荐最终边界：

```text
evaluator:
  负责 model forward、数据收集、raw metric 计算。

metrics:
  负责 metric/result/config/schema/rules/score。

report:
  负责把 metrics result 呈现为 JSON/HTML。

checkpoint selector:
  负责消费 validation result 做选择。
```

这一边界可以避免三类常见问题：

- report 层偷偷重新计算指标，导致报表和 checkpoint 选择不一致；
- selector 层散落阈值，导致选择逻辑难以审计；
- evaluator 层混入展示逻辑，导致 validation 结果难以复用。
