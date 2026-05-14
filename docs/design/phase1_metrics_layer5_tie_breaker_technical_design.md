# Phase I Metrics Layer 5 Technical Design: Tie-breaker

Tie-breaker 是第六个独立技术设计文件，但它不是第六个 hard gate。它不替代五层
validation，也不替代主排序分数 `validation.score`；只有多个 checkpoint 都通过
Layer 0 到 Layer 4，且综合分接近时才启用。

## 1. 工程位置

Tie-breaker 字段定义建议放在：

```text
src/phase1/metrics/phase1_validation_data_schema.py
```

Tie-breaker selector 编排逻辑建议放在：

```text
src/phase1/checkpoint/phase1_checkpoint_selector.py
```

Tie-breaker 指标抽取、score 接近判定和字段级比较 helper 可放在：

```text
src/phase1/metrics/phase1_validation_score.py
```

这样划分后，selector 只负责 checkpoint 级过滤、排序和选择原因记录；
`phase1_validation_score.py` 负责可复用的数值判断原语，避免 selector 直接理解
五层 metrics 的内部字段路径。

## 2. 启用条件

默认触发条件：

```text
abs(best_score - candidate_score) < tie_score_tolerance
```

其中：

```python
tie_score_tolerance: float = 0.03
```

含义是综合分差距小于 3% 时认为两个 checkpoint 接近。

Tie-breaker 只比较已经通过五层 hard gate 的 checkpoint。失败 checkpoint 不允许
通过 tie-breaker 进入候选。

## 3. 输入字段

强类型字段：

```python
@dataclass(frozen=True)
class Phase1TieBreakerMetrics:
    risk_adjusted_return: float
    probe_top3_accuracy: float
    retention_ratio: float
    active_code_ratio: float
    max_code_occupancy: float
    reconstruction_loss: float
```

字段来源：

| 字段 | 来源层 | 方向 |
|---|---|---|
| `risk_adjusted_return` | Layer 3 oracle profitability | 越高越好 |
| `probe_top3_accuracy` | Layer 4 label predictability | 越高越好 |
| `retention_ratio` | Layer 3 oracle profitability | 越高越好 |
| `active_code_ratio` | Layer 1 VQ internal quality | 越高越好 |
| `max_code_occupancy` | Layer 1 VQ internal quality | 越低越好 |
| `reconstruction_loss` | Layer 1 / val metrics | 越低越好 |

checkpoint 中建议保存扁平 key：

```text
validation.tie_breaker.risk_adjusted_return
validation.tie_breaker.probe_top3_accuracy
validation.tie_breaker.retention_ratio
validation.tie_breaker.active_code_ratio
validation.tie_breaker.max_code_occupancy
validation.tie_breaker.reconstruction_loss
```

完整结构化结果仍保存在：

```python
checkpoint.metrics["validation"]["tie_breaker_metrics"]
```

## 4. 比较顺序

当两个 checkpoint 的 `validation.score` 差距小于 `tie_score_tolerance`，按以下顺序
比较：

1. `risk_adjusted_return` 更高者优先；
2. `probe_top3_accuracy` 更高者优先；
3. `retention_ratio` 更高者优先；
4. `active_code_ratio` 更高者优先；
5. `max_code_occupancy` 更低者优先；
6. `reconstruction_loss` 更低者优先；
7. 若仍完全相同，选择 epoch 更早者。

这个顺序体现的业务优先级是：当综合分几乎一样时，先选 oracle decoded 收益质量
更好的 checkpoint；如果收益质量仍接近，再选 label 更容易被 selector 学习的
checkpoint；随后比较 DP 盈利保留、codebook 使用健康度和基础重构质量。epoch 更早
作为最后兜底，是为了减少训练后期 codebook 继续漂移带来的风险。

## 5. 公共 Helper 设计

Tie-breaker 相关公共 helper 需要保持无状态、无 I/O、只依赖强类型 validation
metrics。它们不读取 checkpoint 文件，不重新运行 evaluator，也不判断 hard gate
是否通过。

### 5.1 `DEFAULT_TIE_SCORE_TOLERANCE`

```python
DEFAULT_TIE_SCORE_TOLERANCE: float = 0.03
```

这是 score 接近判定的默认阈值。它只影响 checkpoint selector 的排序决策，不参与
`validation.score` 计算，也不影响 Layer 0 到 Layer 4 的 pass/fail。

### 5.2 `build_tie_breaker_metrics()`

```python
def build_tie_breaker_metrics(
    metrics: Phase1ValidationMetrics,
    *,
    reconstruction_loss: float,
) -> Phase1TieBreakerMetrics:
    ...
```

该 helper 从五层强类型 metrics 中抽取 tie-breaker 所需字段，生成稳定的
`Phase1TieBreakerMetrics`：

| 输出字段 | 读取路径 |
|---|---|
| `risk_adjusted_return` | `metrics.oracle_profitability.risk_adjusted_return` |
| `probe_top3_accuracy` | `metrics.label_predictability.probe_top3_accuracy` |
| `retention_ratio` | `metrics.oracle_profitability.retention_ratio` |
| `active_code_ratio` | `metrics.vq_internal.active_code_ratio` |
| `max_code_occupancy` | `metrics.vq_internal.max_code_occupancy` |
| `reconstruction_loss` | evaluator 传入的 validation reconstruction loss |

这个 helper 的核心目的不是做复杂计算，而是隔离字段路径。selector、report 和
checkpoint payload 只依赖 `Phase1TieBreakerMetrics`，不直接访问五层 metrics 的
内部结构。

### 5.3 `scores_are_tied()`

```python
def scores_are_tied(
    best_score: float,
    candidate_score: float,
    *,
    tolerance: float = DEFAULT_TIE_SCORE_TOLERANCE,
) -> bool:
    ...
```

该 helper 只回答两个 score 是否接近：

```python
abs(best_score - candidate_score) < tolerance
```

返回 `False` 时，selector 应直接选择 score 更高的 checkpoint；返回 `True` 时，
selector 才进入 tie-breaker 字段比较。边界值使用严格小于，因此分差正好等于
`tolerance` 时不触发 tie-breaker。

### 5.4 `compare_phase1_tie_breaker()`

```python
def compare_phase1_tie_breaker(
    left: Phase1TieBreakerMetrics,
    right: Phase1TieBreakerMetrics,
) -> int:
    ...
```

该 helper 对两个 `Phase1TieBreakerMetrics` 做字典序比较：

- 返回 `1` 表示 `left` 更优；
- 返回 `-1` 表示 `right` 更优；
- 返回 `0` 表示 tie-breaker 也无法区分。

比较顺序必须与第 4 节一致。前一个字段已经分出胜负时，不继续比较后续字段。
这样能保持业务优先级可审计，避免把不同量纲的 tie-breaker 字段再次混合成
二级加权分数。

字段级比较应统一处理缺失值：

- `nan` 视为缺失；
- 有限数值优于缺失值；
- 双方同一字段都缺失时继续比较下一个字段；
- `inf` 和 `-inf` 不视为缺失，应按字段方向参与比较；
- 对 `reconstruction_loss` 这类越低越好的字段，`inf` 自然是极差值。

建议在该 helper 内部使用私有函数集中处理字段状态，例如：

```python
def _is_missing_tie_breaker_value(value: float) -> bool:
    return math.isnan(value)

def _compare_tie_breaker_value(
    left_value: float,
    right_value: float,
    *,
    higher_is_better: bool,
) -> int:
    ...
```

这样可以保证 selector、report 解释和单元测试使用同一套缺失值语义。

## 6. Selector 算法

建议 selector 流程：

```python
def select_best_checkpoint(candidates: Sequence[CheckpointRecord]) -> CheckpointRecord:
    passed = [c for c in candidates if c.validation.passed]
    if not passed:
        raise NoPassingPhase1CheckpointError(...)

    passed = sorted(passed, key=lambda c: c.validation.score, reverse=True)
    best = passed[0]

    for candidate in passed[1:]:
        if abs(best.validation.score - candidate.validation.score) >= tie_score_tolerance:
            break
        best = choose_by_tie_breaker(best, candidate)

    return best
```

Tie-breaker 比较：

```python
def choose_by_tie_breaker(a: CheckpointRecord, b: CheckpointRecord) -> CheckpointRecord:
    keys = (
        ("risk_adjusted_return", "max"),
        ("probe_top3_accuracy", "max"),
        ("retention_ratio", "max"),
        ("active_code_ratio", "max"),
        ("max_code_occupancy", "min"),
        ("reconstruction_loss", "min"),
    )
    ...
```

## 7. 缺失字段策略

Tie-breaker 缺失字段不应导致失败 checkpoint 被选中。

推荐策略：

- 如果 checkpoint 缺少 `validation`，跳过或标记为 incompatible；
- 如果 `validation.passed != True`，跳过；
- 如果 `validation.score` 缺失，跳过；
- 如果 score 接近但某个 tie-breaker 字段缺失，缺失方在该字段上视为劣于有值方；
- 如果双方同一字段都缺失，继续比较下一个字段；
- 如果所有 tie-breaker 字段都缺失，选择 score 更高者；score 完全相同则选择
  epoch 更早者。

`nan` 视为缺失。`inf` 只允许用于明确方向的字段；例如 `reconstruction_loss=inf`
应视为极差。

## 8. 与 Score 的关系

`validation.score` 是主排序分数，由五层 metrics 归一化加权得到。Tie-breaker 不参与
score 计算，原因是：

- score 需要稳定反映整体质量；
- tie-breaker 是接近分数下的业务偏好；
- 把 tie-breaker 混入 score 会让权重解释变复杂；
- selector 审计时需要清楚知道是“总分胜出”还是“接近总分下按业务优先级胜出”。

report 应在 selection summary 中显示：

- 原始最高 score checkpoint；
- 最终被选 checkpoint；
- 是否触发 tie-breaker；
- 触发时每个 tie-breaker 字段的比较结果。

## 9. 示例

```text
checkpoint A: validation.score = 0.842
checkpoint B: validation.score = 0.836
tie_score_tolerance = 0.03
```

两者差距为 `0.006`，小于 `0.03`，因此进入 tie-breaker。如果 B 的
`risk_adjusted_return` 高于 A，则选择 B，即使 A 的综合分略高。

## 10. 测试要点

- 分数差距大于等于 tolerance 时不触发 tie-breaker；
- 分数差距小于 tolerance 时按固定字段顺序比较；
- `max_code_occupancy` 和 `reconstruction_loss` 使用越低越好；
- `nan` 在 tie-breaker 字段比较中视为缺失；
- 单侧缺失时，有值方胜出；双方都缺失时继续比较下一个字段；
- `inf` 按字段方向参与比较，`reconstruction_loss=inf` 应视为极差；
- 缺失字段不会抛出不可解释异常；
- 全部字段相同或缺失时选择 epoch 更早者；
- 失败 checkpoint 即使 tie-breaker 字段很好，也不能被选中。
