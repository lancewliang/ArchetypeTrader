# Phase I Metrics Submodule Technical Design

本文档描述 `src/phase1/metrics` 子模块的总体技术边界，以及五层
codebook validation 和 tie-breaker 的拆分文件索引。

本文档是 `docs/design/phase1_codebook_validation_criteria.md` 的工程实现总览。
前者定义“要验证什么、阈值是什么”；本文件和分层文件定义“代码如何组织、数据如何
流动、结果如何落盘和被 report/checkpoint selector 使用”。

## 1. 分层技术设计文件

Phase I codebook validation 按指标层拆分为五个技术设计文件；tie-breaker 作为
第六个独立文件维护：

| 层级 | 主题 | 技术设计文件 |
|---:|---|---|
| Layer 0 | DP teacher quality | [phase1_metrics_layer0_teacher_quality_technical_design.md](phase1_metrics_layer0_teacher_quality_technical_design.md) |
| Layer 1 | VQ internal quality | [phase1_metrics_layer1_vq_internal_technical_design.md](phase1_metrics_layer1_vq_internal_technical_design.md) |
| Layer 2 | Behavior quality | [phase1_metrics_layer2_behavior_quality_technical_design.md](phase1_metrics_layer2_behavior_quality_technical_design.md) |
| Layer 3 | Oracle profitability | [phase1_metrics_layer3_oracle_profitability_technical_design.md](phase1_metrics_layer3_oracle_profitability_technical_design.md) |
| Layer 4 | Label predictability | [phase1_metrics_layer4_label_predictability_technical_design.md](phase1_metrics_layer4_label_predictability_technical_design.md) |
| Layer 5 | Tie-breaker | [phase1_metrics_layer5_tie_breaker_technical_design.md](phase1_metrics_layer5_tie_breaker_technical_design.md) |

命名中的 Layer 5 表示第六个独立文件，不改变验证标准中 Layer 0 到 Layer 4 的
hard gate 定义。tie-breaker 不属于 hard gate，只在多个 checkpoint 都通过五层
验证且综合分接近时使用。

## 2. 设计目标

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

## 3. 非目标

本设计不负责：

- 重写 Phase I VQ 模型结构；
- 替代 Phase II selector validation；
- 在 report 层重新实现交易收益、motif、morphology 等计算；
- 在 metrics 层直接调用 model 或 dataloader；
- 在 checkpoint selector 中散落五层过滤规则。

## 4. 当前代码上下文

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
    phase1_codebook_report.py
  phase1_main.py
```

现状：

- `Phase1MainFlow._run_epoch()` 和 `Phase1Evaluator.evaluate()` 已经引用
  `Phase1Metrics`；
- `metrics/phase1_metrics.py` 当前尚未实现；
- `Phase1CheckpointSelector` 当前仍是骨架，默认按单个 metric 选择；
- `Phase1CodebookReport` 负责 codebook validation report 的 JSON/HTML 写出；
- `phase1_codebook_validation_criteria.md` 已经定义完整五层验证标准。

因此实现顺序应先让 `metrics` 的数据契约稳定，再扩展 evaluator 和 selector。

## 5. 模块职责边界

### 5.1 evaluators

`src/phase1/evaluators` 负责：

- 遍历 `DataLoader`；
- 调用 `ArchetypeVQModel.forward()` / `encode()` / `decode()`；
- 收集中间张量，包括 logits、decoded actions、code id、latent、distance 等；
- 基于收集的数据计算 raw metric values；
- 调用 `metrics` 中的判定规则，生成结构化 validation result；
- 不负责 HTML/JSON report 的排版；
- 不在内部硬编码 report 样式。

### 5.2 metrics

`src/phase1/metrics` 负责：

- 定义训练期基础指标 `Phase1Metrics`；
- 定义 validation 中间数据 schema；
- 定义 metric result / layer result / checkpoint validation result；
- 定义阈值、评分权重和 metric name 常量；
- 定义每层 hard gate 判定规则；
- 定义综合评分规则；
- 提供稳定的 `to_dict()` / `from_dict()` 序列化契约；
- 不直接依赖 `torch.nn.Module`、`DataLoader` 或文件系统。

### 5.3 report

`src/phase1/report` 负责：

- 消费 `Phase1ValidationResult`；
- 写出 JSON report；
- 后续可渲染 HTML report；
- 呈现每个 metric 的 value、threshold、status、message；
- 呈现 code-level diagnostics 和 drift diagnostics；
- 不重新计算 metric；
- 不改写 checkpoint selection 逻辑。

### 5.4 checkpoint

`src/phase1/checkpoint` 负责：

- 将 validation result 随 checkpoint 一起保存；
- 读取 checkpoint 中的 `validation.passed`、`validation.score` 和 tie-breaker 字段；
- 从通过 hard gates 的候选中选 score 最高者；
- 如果没有候选通过，返回失败摘要或抛出可解释错误；
- 不重新实现五层判定规则。

## 6. 推荐文件结构

```text
src/phase1/metrics/
  __init__.py
  phase1_metrics.py
  phase1_validation_config.py
  phase1_validation_data_schema.py
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
  phase1_codebook_report.py
```

说明：

- `phase1_metrics.py` 保留给训练期 loss/accuracy 指标，兼容现有调用；
- `phase1_validation_config.py` 放五个分层阈值类、评分权重类和 evaluator 运行参数类；
- `phase1_validation_data_schema.py` 放 evaluator 收集到的中间数据结构；
- `phase1_metric_results.py` 放判定结果结构；
- `phase1_validation_rules.py` 放 hard gate 规则；
- `phase1_validation_score.py` 放 normalized score 和 tie-breaker 所需字段；
- `phase1_codebook_evaluator.py` 是完整五层 validation 的统一编排入口；
- `phase1_validation_layers/layer*.py` 是五层 raw metric 的独立计算文件；
- `phase1_codebook_report.py` 是后续 HTML/JSON 呈现入口。

## 7. 核心数据流

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
  -> apply tie-breaker when scores are close
  -> return Phase1CheckpointSelectionResult
```

report 流程：

```text
Phase1CodebookReport.write_report()
  -> load best checkpoint / selection result
  -> read Phase1ValidationResult dict
  -> render JSON/HTML
```

## 8. 训练期基础指标

`Phase1Metrics` 用于 `_run_epoch()` 和 `Phase1Evaluator.evaluate()`，不承载五层
validation 结果。

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

注意：

- loss 需要按 sample 加权累加，再在 `averaged()` 中除以 `num_samples`；
- action accuracy 按 timestep 统计，不能简单按 batch 平均；
- `to_dict(include_context=True)` 需要包含 `stage/split/epoch/num_samples`；
- `to_dict(include_context=False)` 只输出指标字段，便于嵌入其他 payload。

## 9. Validation 共享数据契约

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
- `prices` 来自 `HorizonDataset`，如果没有价格，则盈利指标无法完整计算；
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

### 9.4 Layer computation

五层 raw metric 计算统一返回 `Phase1LayerComputation`。该对象只承载计算结果，
不承载 PASS/FAIL。

```python
Phase1LayerMetrics: TypeAlias = (
    Phase1TeacherQualityMetrics
    | Phase1VQInternalMetrics
    | Phase1BehaviorQualityMetrics
    | Phase1OracleProfitabilityMetrics
    | Phase1LabelPredictabilityMetrics
)


@dataclass(frozen=True)
class Phase1LayerComputation:
    layer_id: int
    layer_name: str
    metrics: Phase1LayerMetrics
    code_diagnostics: tuple[Phase1CodeDiagnostic, ...] = ()
    extra_payload: Mapping[str, object] = field(default_factory=dict)
```

## 10. 强类型 Layer Metrics

所有五层指标都必须使用明确 dataclass 字段承载，不使用 `Mapping[str, float]`
作为主要数据结构。字符串 key 只允许出现在 `to_dict()` 序列化结果中，供
checkpoint/report 落盘。

设计原则：

- 每个指标是一个显式字段，便于 IDE 补全、类型检查和重构；
- rules/scoring/report 都消费强类型对象；
- `to_flat_dict()` 只作为 checkpoint selector 快速读取的派生视图；
- 缺失或不可计算指标用 `float("nan")` 或 `None`，不要省略字段。

各层强类型 metrics 详见分层文件。五层聚合对象如下：

```python
@dataclass(frozen=True)
class Phase1ValidationMetrics:
    teacher_quality: Phase1TeacherQualityMetrics
    vq_internal: Phase1VQInternalMetrics
    behavior_quality: Phase1BehaviorQualityMetrics
    oracle_profitability: Phase1OracleProfitabilityMetrics
    label_predictability: Phase1LabelPredictabilityMetrics

    def to_flat_dict(self) -> dict[str, float | int]:
        ...
```

序列化要求：

- 每个 metrics dataclass 都实现 `to_dict()` / `from_dict()`；
- `Phase1ValidationMetrics.to_flat_dict()` 负责生成 checkpoint selector 使用的扁平
  key，例如 `oracle_profitability.risk_adjusted_return`；
- 代码内部禁止通过字符串 key 读取指标值，必须访问字段，例如
  `metrics.vq_internal.active_code_ratio`。

## 11. Metric 判定结果

建议位置：

```text
src/phase1/metrics/phase1_metric_results.py
```

```python
MetricSeverity = Literal["pass", "warn", "fail", "skip"]


@dataclass(frozen=True)
class Phase1MetricResult:
    name: str
    value: int | float | str | bool | None
    threshold: str
    severity: MetricSeverity
    passed: bool
    layer: str
    message: str = ""


@dataclass(frozen=True)
class Phase1LayerResult:
    layer_id: int
    name: str
    passed: bool
    metrics: tuple[Phase1MetricResult, ...]
```

约定：

- `pass` 表示指标通过；
- `warn` 表示触发警戒，但不直接淘汰；
- `fail` 表示触发 hard gate 失败；
- `skip` 表示缺少必要输入，无法计算；对于 hard gate 指标，`skip` 默认视为失败。

Checkpoint 级结果：

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
    metrics: Phase1ValidationMetrics
    code_diagnostics: tuple[Phase1CodeDiagnostic, ...]
    drift_diagnostics: Mapping[str, Phase1MetricResult]
    tie_breaker_metrics: Phase1TieBreakerMetrics
```

必须提供 `to_dict()` / `from_dict()`。

## 12. Metric 命名规范

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

## 13. Evaluator 编排

`Phase1CodebookEvaluator` 是完整 codebook validation 的统一编排入口。它不把五层
指标计算全部写在一个大文件中，而是委托给 `phase1_validation_layers/layer*.py`。

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
        teacher_thresholds: Phase1TeacherQualityThresholds,
        vq_internal_thresholds: Phase1VQInternalThresholds,
        behavior_thresholds: Phase1BehaviorQualityThresholds,
        oracle_profitability_thresholds: Phase1OracleProfitabilityThresholds,
        label_predictability_thresholds: Phase1LabelPredictabilityThresholds,
        score_weights: Phase1ValidationScoreWeights,
        runtime_config: Phase1ValidationRuntimeConfig,
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
2. 调用五个 layer calculator，分别得到强类型 layer metrics 和 diagnostics；
3. 将强类型 metrics 交给 `metrics.phase1_validation_rules` 做 hard gate 判定；
4. 将通过 hard gates 的结果交给 `metrics.phase1_validation_score` 计算 score；
5. 组装 `Phase1ValidationResult`。

建议计算顺序：

```text
collect_snapshot
  -> layer0_teacher_quality
  -> layer1_vq_internal
  -> layer3_oracle_profitability
  -> layer2_behavior_quality
  -> layer4_label_predictability
  -> validation_rules 按 0,1,2,3,4 输出 layer results
```

Layer 3 先于 Layer 2，是因为 Layer 2 可以消费 per-code profitability，用于
`profitable_code_coverage` 和 weak-lift-but-nonprofitable code 判定。

## 14. 配置归属

阈值配置按 validation layer 拆成五个 dataclass；不定义一个巨大总阈值类。

```python
@dataclass(frozen=True)
class Phase1TeacherQualityThresholds: ...

@dataclass(frozen=True)
class Phase1VQInternalThresholds: ...

@dataclass(frozen=True)
class Phase1BehaviorQualityThresholds: ...

@dataclass(frozen=True)
class Phase1OracleProfitabilityThresholds: ...

@dataclass(frozen=True)
class Phase1LabelPredictabilityThresholds: ...
```

评分权重和运行参数独立：

```python
@dataclass(frozen=True)
class Phase1ValidationScoreWeights:
    teacher_quality: float = 0.10
    reconstruction: float = 0.20
    codebook_health: float = 0.15
    behavior_structure: float = 0.20
    oracle_profitability: float = 0.25
    label_predictability: float = 0.10


@dataclass(frozen=True)
class Phase1ValidationRuntimeConfig:
    fee_rate: float = 0.0002
    random_label_trials: int = 3
    churn_window_epochs: int = 5
    active_code_min_occupancy: float = 0.01
    dead_code_max_occupancy: float = 0.001
    top_contribution_ratio: float = 0.05
    probe_epochs: int = 20
    probe_learning_rate: float = 1e-3
    probe_batch_size: int = 256
    random_seed: int = 42
```

职责边界：

- 五个分层 thresholds 只给 `phase1_validation_rules.py` 的对应 layer rule 使用；
- `Phase1ValidationScoreWeights` 只给 `phase1_validation_score.py` 使用；
- `Phase1ValidationRuntimeConfig` 给 evaluator 和五个 layer calculator 使用。

## 15. Rule 层设计

建议位置：

```text
src/phase1/metrics/phase1_validation_rules.py
```

设计原则：

- 输入强类型 layer metrics；
- 输出 `Phase1LayerResult`；
- 不访问 model、dataloader、文件系统；
- 不重复计算 evaluator 已经计算过的数值；
- 对缺失 hard gate 指标返回 `fail` 或明确 `skip-as-fail`。

建议函数：

```python
def evaluate_teacher_quality_rules(
    metrics: Phase1TeacherQualityMetrics,
    thresholds: Phase1TeacherQualityThresholds,
) -> Phase1LayerResult:
    ...

def evaluate_vq_internal_rules(
    metrics: Phase1VQInternalMetrics,
    thresholds: Phase1VQInternalThresholds,
) -> Phase1LayerResult:
    ...

def evaluate_behavior_quality_rules(
    metrics: Phase1BehaviorQualityMetrics,
    thresholds: Phase1BehaviorQualityThresholds,
) -> Phase1LayerResult:
    ...

def evaluate_oracle_profitability_rules(
    metrics: Phase1OracleProfitabilityMetrics,
    thresholds: Phase1OracleProfitabilityThresholds,
) -> Phase1LayerResult:
    ...

def evaluate_label_predictability_rules(
    metrics: Phase1LabelPredictabilityMetrics,
    thresholds: Phase1LabelPredictabilityThresholds,
) -> Phase1LayerResult:
    ...
```

## 16. Scoring 与 Tie-breaker

只有所有 hard gates 通过时才计算最终 score。若未通过，`score=None`，避免误把
失败 checkpoint 排进候选。

建议位置：

```text
src/phase1/metrics/phase1_validation_score.py
```

```python
def compute_phase1_validation_score(
    metrics: Phase1ValidationMetrics,
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

Tie-breaker 规则详见
[phase1_metrics_layer5_tie_breaker_technical_design.md](phase1_metrics_layer5_tie_breaker_technical_design.md)。

## 17. Checkpoint selector 集成

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
5. 若最高分差距 `< tie_score_tolerance`，按 tie-breaker 选择。

没有通过候选时：

- 不应静默回退到最低 loss checkpoint；
- 应返回明确错误或 `Phase1CheckpointSelectionResult` 中带失败摘要；
- Phase I 主流程应阻断 Phase II artifact export。

## 18. Report 设计

`report` 不重新计算指标。它只消费：

```python
Phase1ValidationResult
Phase1CheckpointSelectionResult
config snapshot
artifact index
```

推荐入口：

```text
src/phase1/report/phase1_codebook_report.py
```

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

## 19. 测试策略

### 19.1 metrics 单元测试

- `Phase1Metrics.add_batch()` 加权平均；
- action accuracy 按 timestep 统计；
- `Phase1MetricResult.to_dict()/from_dict()`；
- 每层 rules 的 pass/fail 边界；
- score 权重总和和归一化截断。

### 19.2 evaluator 单元测试

使用小型 fake model / fixed tensors：

- snapshot shape 正确；
- code occupancy/perplexity 正确；
- nearest/second margin 正确；
- motif/morphology 分类稳定；
- profitability execution 口径正确；
- random label baseline 可设置 seed 保持确定性。

### 19.3 integration 测试

- train epoch 保存 checkpoint metrics 包含 `validation`；
- selector 过滤失败 checkpoint；
- selector 对通过 checkpoint 按 score 选择；
- report 可以从 `Phase1ValidationResult` 写 JSON。

## 20. 风险与约束

### 20.1 样本顺序

assignment churn、prices 对齐、horizon diagnostics 都依赖稳定样本顺序。
validation evaluator 必须使用 `shuffle=False` dataloader。

### 20.2 缺少 prices

如果没有 `HorizonDataset.prices`，Layer 0 的 fee/morphology 指标、Layer 3 的
profitability 指标、Layer 4 的 probe return retention 都无法完整计算。对于
hard gate 指标，缺失输入应视为失败，并在 report 中明确写出。

### 20.3 计算成本

完整五层 validation 比普通 loss evaluation 更重。可以通过 config 控制：

- 每几个 epoch 跑一次完整 validation；
- 其他 epoch 只跑基础 metrics；
- probe 和 random-label baseline 支持 sample 限制。

### 20.4 随机性

random label baseline 和 probe training 必须支持 seed，避免 checkpoint 选择不稳定。

### 20.5 checkpoint 兼容

旧 checkpoint 可能没有 `metrics["validation"]`。selector 应明确报错或跳过，
不能把旧 checkpoint 当成通过 validation。

## 21. 最终边界总结

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
