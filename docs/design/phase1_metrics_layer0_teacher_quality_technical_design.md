# Phase I Metrics Layer 0 Technical Design: DP Teacher Quality

Layer 0 判断 DP teacher 数据本身是否值得学习。VQ codebook 只能压缩 teacher
信号；如果 teacher 的扣费后优势微弱、集中在少数样本或对手续费极度敏感，后续
Layer 1 到 Layer 4 即使通过重构指标，也很难产生可交易的 selector artifact。

## 1. 工程位置

计算代码建议放在：

```text
src/phase1/evaluators/phase1_validation_layers/layer0_teacher_quality.py
```

阈值、结果和规则分别放在：

```text
src/phase1/metrics/phase1_validation_config.py
src/phase1/metrics/phase1_validation_data_schema.py
src/phase1/metrics/phase1_validation_rules.py
```

Layer 0 calculator 只计算 raw metrics，不返回 PASS/FAIL。PASS/FAIL 统一由
`phase1_validation_rules.py` 判定。

## 2. 输入依赖

Layer 0 只依赖 teacher 相关数据，不依赖 decoded actions 的盈利结果。

必需输入：

- `train_snapshot: Phase1EvaluationSnapshot`
- `val_snapshot: Phase1EvaluationSnapshot`
- `runtime_config: Phase1ValidationRuntimeConfig`

实际使用字段：

- `val_snapshot.demo_actions`
- `val_snapshot.demo_rewards`
- `val_snapshot.prices`
- `runtime_config.fee_rate`
- `runtime_config.top_contribution_ratio`

其中 `prices` 用于 fee sensitivity、morphology coverage 和统一 execution 口径。
若没有 `prices`，hard gate 相关指标写入 `nan`，由 rules 层判定为失败。

## 3. 输出 Metrics

强类型 metrics 定义：

```python
@dataclass(frozen=True)
class Phase1TeacherQualityMetrics:
    dp_advantage_vs_flat: float
    dp_win_rate_vs_flat: float
    near_zero_opportunity_ratio: float
    fee_sensitivity: float
    morphology_coverage: float
    dp_return_concentration_after_top5_removed: float
```

Layer computation 返回：

```python
Phase1LayerComputation(
    layer_id=0,
    layer_name="teacher_quality",
    metrics=Phase1TeacherQualityMetrics(...),
    code_diagnostics=(),
    extra_payload={
        "dp_returns": ...,
        "flat_returns": ...,
        "advantages": ...,
        "missing_reason": ...,
    },
)
```

`extra_payload` 只给 report/debug 使用，不作为 rules 的主要输入。

## 4. 推荐入口

```python
def compute_teacher_quality_metrics(
    *,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
) -> Phase1LayerComputation:
    ...
```

## 5. Helper 设计

```python
def compute_flat_returns(prices: np.ndarray) -> np.ndarray:
    ...

def compute_demo_returns(snapshot: Phase1EvaluationSnapshot) -> np.ndarray:
    ...

def compute_fee_sensitivity(
    prices: np.ndarray,
    actions: np.ndarray,
    fee_rate: float,
) -> float:
    ...

def compute_top_removed_total_advantage(
    advantages: np.ndarray,
    top_ratio: float,
) -> float:
    ...

def classify_market_morphology(prices: np.ndarray) -> np.ndarray:
    ...
```

各 helper 的功能定义如下：

- `compute_flat_returns(prices)`：计算 flat baseline 的每条 horizon 收益。
  flat baseline 表示全程不交易、保持空仓；第一版默认无资金利息、无持仓成本，
  因此返回与样本数一致的全 0 数组，作为 `R_DP` 的对照基准 `R_flat`。

- `compute_demo_returns(snapshot)`：计算 DP teacher 在每条 horizon 上的总收益
  `R_DP`。第一版优先使用 `snapshot.demo_rewards.sum(axis=1)`；当 reward
  缺失、不可信或需要统一交易执行口径时，应使用 execution helper 根据
  `snapshot.prices + snapshot.demo_actions + fee_rate` 重新执行得到收益。

- `compute_fee_sensitivity(prices, actions, fee_rate)`：计算 teacher 策略对手续费
  的敏感性。函数使用同一批 `actions` 在 `fee_rate * 2` 下重新执行，并返回翻倍
  手续费后的总优势保留比例：
  `sum(adv_double_fee) / (sum(original_advantage) + eps)`。该值越低，说明 teacher
  的优势越容易被手续费侵蚀。

- `compute_top_removed_total_advantage(advantages, top_ratio)`：衡量 teacher 收益
  是否过度集中在少数样本上。函数按 `advantages` 从高到低剔除最高
  `top_ratio` 比例样本后，返回剩余样本的优势总和。若剔除头部样本后仍为正，
  说明 teacher 优势不是完全依赖少数极端行情。

- `classify_market_morphology(prices)`：根据每条价格路径分类市场形态，返回
  `[N]` 形状的标签数组。标签包括 `uptrend`、`downtrend`、`reversal-up`、
  `reversal-down`、`range-high-vol`、`range-low-vol`、`volatile-mixed` 和
  `neutral`。Layer 0 使用该结果计算
  `morphology_coverage = mean(morphology != "neutral")`，用于判断 validation
  数据是否覆盖足够多非中性市场机会。

第一版可以优先使用 `demo_rewards.sum(axis=1)` 作为 `R_DP`，但 fee sensitivity
和 morphology coverage 仍需要 `prices`。如果 DP reward 口径和统一 execution
口径不同，report 中必须记录两者口径，正式 checkpoint selection 应使用统一
execution helper 的结果。

## 6. 计算流程

1. 计算 `R_DP`。优先使用 `demo_rewards.sum(axis=1)`；如果 reward 口径不可信或
   需要手续费敏感性，则用统一 execution helper 根据 `prices + demo_actions`
   重新计算。
2. 计算 `R_flat`。默认全 0；如果后续环境有资金利息或持仓成本，应通过 runtime
   config 接入 execution helper。
3. 计算 `advantage = R_DP - R_flat`。
4. 计算 `dp_advantage_vs_flat = mean(advantage)`。
5. 计算 `dp_win_rate_vs_flat = mean(R_DP > R_flat)`。
6. 计算 `near_zero_opportunity_ratio = mean(abs(advantage) < fee_threshold)`。
   第一版 `fee_threshold = runtime_config.fee_rate`。
7. 用 `fee_rate * 2` 重新执行 demo actions，计算翻倍手续费后的总优势保留比例：
   `fee_sensitivity = sum(adv_double_fee) / (sum(advantage) + eps)`。
8. 通过 morphology helper 计算 validation horizon 的形态标签，得到
   `morphology_coverage = mean(morphology != "neutral")`。
9. 去掉 advantage 最高的 `top_contribution_ratio` 样本，计算剩余总优势：
   `dp_return_concentration_after_top5_removed`。

## 7. Rule 判定

Layer 0 thresholds：

```python
@dataclass(frozen=True)
class Phase1TeacherQualityThresholds:
    dp_win_rate_min: float = 0.58
    near_zero_opportunity_ratio_max: float = 0.35
    fee_sensitivity_min: float = 0.60
    morphology_coverage_min: float = 0.60
```

Hard gates：

- `dp_advantage_vs_flat > 0`
- `dp_win_rate_vs_flat >= 0.58`
- `near_zero_opportunity_ratio <= 0.35`
- `fee_sensitivity >= 0.60`
- `morphology_coverage >= 0.60`
- `dp_return_concentration_after_top5_removed > 0`

Rule 函数：

```python
def evaluate_teacher_quality_rules(
    metrics: Phase1TeacherQualityMetrics,
    thresholds: Phase1TeacherQualityThresholds,
) -> Phase1LayerResult:
    ...
```

规则层必须把 `nan`、`None`、`inf` 的 hard gate 指标标记为 `fail` 或
`skip-as-fail`，并在 `Phase1MetricResult.message` 中写明原因。

## 8. 缺失数据策略

- 缺少 `prices` 时，`fee_sensitivity` 和 `morphology_coverage` 不能可靠计算；
- 对 hard gate 指标，缺失值写入 `nan`，由 rules 层判定为 fail；
- report 中需要明确标记 failure reason 为 `missing_prices`；
- `sum(advantage) <= 0` 时，`fee_sensitivity` 不应被当作通过信号，Layer 0 已经
  由 `dp_advantage_vs_flat` 和 concentration gate 淘汰。

## 9. 与其他层的关系

- Layer 0 失败时，后续 checkpoint 不应进入 selector 候选；
- Layer 3 的 `retention_ratio` 依赖 `R_DP` 的正优势，Layer 0 失败通常会导致
  Layer 3 retention 不可靠；
- Layer 2/4 可复用 Layer 0 的 morphology helper，但不能从 Layer 0 payload 中
  读取字符串 key 作为主流程依赖，应该调用共享 helper 或显式传递 typed 结果。

## 10. 测试要点

- `demo_rewards.sum(axis=1)` 与 execution helper 的 shape 对齐；
- `near_zero_opportunity_ratio` 在阈值边界上稳定；
- top 5% removed advantage 在样本数很小时不会产生空数组；
- fee 翻倍后优势保留比例可复现；
- 缺少 `prices` 时 hard gate 失败且 report message 可解释。
