# Phase I Metrics Layer 1 Technical Design: VQ Internal Quality

Layer 1 判断 VQ 内部表示是否稳定、可用、未塌缩，并且 decoder 是否保留了 DP
示范动作的主要交易语义。该层不判断盈利，不做 morphology/motif 解释，也不训练
selector probe。

## 1. 工程位置

计算代码建议放在：

```text
src/phase1/evaluators/phase1_validation_layers/layer1_vq_internal.py
```

Layer 1 calculator 返回 `Phase1LayerComputation`，PASS/FAIL 由
`metrics.phase1_validation_rules.evaluate_vq_internal_rules()` 统一判定。

## 2. 输入依赖

必需输入：

- `train_snapshot: Phase1EvaluationSnapshot`
- `val_snapshot: Phase1EvaluationSnapshot`
- `assignment_history: Sequence[CodeAssignmentSnapshot]`
- `runtime_config: Phase1ValidationRuntimeConfig`

实际使用字段：

- `train_snapshot.reconstruction_loss`
- `val_snapshot.reconstruction_loss`
- `val_snapshot.demo_actions`
- `val_snapshot.decoded_actions`
- `val_snapshot.code_ids`
- `val_snapshot.z_e`
- `val_snapshot.z_q`
- `val_snapshot.distances`
- `val_snapshot.sample_ids`

`assignment_history` 必须使用稳定 `sample_ids` 对齐。用于 validation 的 dataloader
必须 `shuffle=False`。

## 3. 输出 Metrics

```python
@dataclass(frozen=True)
class Phase1VQInternalMetrics:
    validation_action_accuracy: float
    reconstruction_loss_gap: float
    active_code_ratio: float
    max_code_occupancy: float
    normalized_code_perplexity: float
    dead_code_ratio: float
    assignment_churn_recent_mean: float
    code_lifetime_pass_ratio: float
    quantization_distance: float
    nearest_second_margin_median: float
    decoder_turnover_error: float
    entry_timing_error_median: float
    direction_accuracy: float
```

返回结构：

```python
Phase1LayerComputation(
    layer_id=1,
    layer_name="vq_internal",
    metrics=Phase1VQInternalMetrics(...),
    extra_payload={
        "code_distribution": ...,
        "active_codes": ...,
        "assignment_churn_by_epoch": ...,
    },
)
```

## 4. 推荐入口

```python
def compute_vq_internal_metrics(
    *,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    assignment_history: Sequence[CodeAssignmentSnapshot],
    runtime_config: Phase1ValidationRuntimeConfig,
) -> Phase1LayerComputation:
    ...
```

## 5. Helper 设计

```python
def compute_action_accuracy(demo: np.ndarray, decoded: np.ndarray) -> float:
    ...

def compute_code_distribution(code_ids: np.ndarray, k: int) -> np.ndarray:
    ...

def compute_normalized_perplexity(p: np.ndarray) -> float:
    ...

def compute_assignment_churn(
    current: CodeAssignmentSnapshot,
    history: Sequence[CodeAssignmentSnapshot],
    window: int,
) -> float:
    ...

def compute_code_lifetime_pass_ratio(
    current_active_codes: Sequence[int],
    history: Sequence[CodeAssignmentSnapshot],
    min_lifetime_epochs: int,
) -> float:
    ...

def compute_nearest_second_margin(distances: np.ndarray) -> np.ndarray:
    ...

def classify_main_direction(actions: np.ndarray) -> np.ndarray:
    ...

def compute_first_trade_t(actions: np.ndarray) -> np.ndarray:
    ...
```

## 6. 计算流程

1. `validation_action_accuracy = mean(decoded_actions == demo_actions)`，按所有
   horizon 和 timestep 展开统计。
2. `reconstruction_loss_gap = val_rec_loss / (train_rec_loss + eps)`。
3. `p_k = bincount(code_ids, minlength=K) / N`。
4. `active_code_ratio = mean(p_k >= active_code_min_occupancy)`。
5. `max_code_occupancy = max(p_k)`。
6. `normalized_code_perplexity = exp(-sum(p_k * log(p_k + eps))) / K`。
7. `dead_code_ratio = mean(p_k < dead_code_max_occupancy)`。
8. `assignment_churn_recent_mean`：取最近 `churn_window_epochs` 个历史 snapshot，
   按稳定 `sample_ids` 对齐，计算同一 sample 的 label 改变比例，再求均值。
9. `code_lifetime_pass_ratio`：对当前 active code，统计其连续 active epoch 数，
   计算 lifetime 达到 10 个 epoch 的 active code 比例。
10. `quantization_distance = mean(norm(z_e - z_q, axis=-1))`。
11. `nearest_second_margin_median`：对每个样本取最近距离 `d1` 和第二近距离 `d2`，
    计算 `(d2 - d1) / (d1 + eps)` 的中位数。
12. `decoder_turnover_error`：分别计算 demo/decoded action 的 position change
    次数，取 `mean(abs(turnover_dec - turnover_demo))`。
13. `entry_timing_error_median`：只对 demo 和 decoded 都存在交易的样本统计
    `abs(first_trade_dec - first_trade_demo)` 的中位数。
14. `direction_accuracy`：把每条 action sequence 归为 `long/short/flat/mixed`，
    统计 demo 和 decoded 主方向一致比例。

## 7. Rule 判定

Layer 1 thresholds：

```python
@dataclass(frozen=True)
class Phase1VQInternalThresholds:
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
```

Hard gates：

- `validation_action_accuracy >= 0.85`
- `reconstruction_loss_gap <= 1.25`
- `active_code_ratio >= 0.80`
- `max_code_occupancy <= 0.40`
- `0.50 <= normalized_code_perplexity <= 0.90`
- `dead_code_ratio <= 0.20`
- `assignment_churn_recent_mean <= 0.15`
- `nearest_second_margin_median >= 0.10`
- `direction_accuracy >= 0.88`
- `entry_timing_error_median <= 0.15 * horizon_length`

`quantization_distance` 和 `code_lifetime_pass_ratio` 第一版可作为 warn/scoring
信号；如果训练历史足够完整，也可以升级为 hard gate。

## 8. 缺失数据策略

- 训练初期 history 不足时，`assignment_churn_recent_mean` 可标记为 `nan`；
  rules 层可在前 `churn_window_epochs` 内降级为 warn，正式 checkpoint selection
  阶段必须有足够 history；
- 如果 `distances` 未收集，margin 和 quantization distance 视为不可计算；
- 如果没有任何样本同时存在 demo/decoded entry，`entry_timing_error_median`
  写入 `nan`，由 rules 层结合 direction/flat ratio 决定 fail 或 warn；
- 如果 `K` 无法从 model/config 获取，应从 `distances.shape[-1]` 推断；仍失败则
  不能计算 occupancy/perplexity，Layer 1 fail。

## 9. 与其他层的关系

- Layer 1 通过是后续所有层的前提；code collapse 会让 Layer 2/3/4 的分层统计失真；
- Layer 2 使用 `code_ids`、`decoded_actions`、`z_e` 做行为结构分析；
- Layer 3 使用 `decoded_actions` 和 `code_ids` 做 oracle profitability；
- tie-breaker 使用 `active_code_ratio`、`max_code_occupancy` 和 `reconstruction_loss`。

## 10. 测试要点

- action accuracy 必须按 timestep 统计；
- occupancy/perplexity 在 dead code、单 code collapse、均匀分布三种场景下正确；
- churn 计算必须按 `sample_ids` 对齐，而不是按数组位置盲目比较；
- nearest/second margin 在距离并列时稳定；
- entry timing 只统计 demo/decoded 都有交易的样本；
- `shuffle=True` 的 dataloader 不应被用于 validation snapshot。
