# Phase I Metrics Layer 4 Technical Design: Label Predictability

Layer 4 判断 Phase I assigned label 是否能从 Phase II selector 可见状态中学习。
Layer 3 证明 oracle label 有交易价值；Layer 4 证明这些 label 对未来 selector
不是不可预测的未来信息标签。

## 1. 工程位置

计算代码建议放在：

```text
src/phase1/evaluators/phase1_validation_layers/layer4_label_predictability.py
```

Layer 4 负责 probe training、probe evaluation、mutual information lift 和 probe
decoded return retention。PASS/FAIL 由 `evaluate_label_predictability_rules()`
判定。

## 2. 输入依赖

必需输入：

- `model: ArchetypeVQModel`
- `train_snapshot: Phase1EvaluationSnapshot`
- `val_snapshot: Phase1EvaluationSnapshot`
- `runtime_config: Phase1ValidationRuntimeConfig`
- `device: torch.device | str`

实际使用字段：

- `train_snapshot.states`
- `train_snapshot.code_ids`
- `val_snapshot.states`
- `val_snapshot.code_ids`
- `val_snapshot.prices`
- `val_snapshot.decoded_actions`
- `runtime_config.probe_epochs`
- `runtime_config.probe_learning_rate`
- `runtime_config.probe_batch_size`
- `runtime_config.random_seed`

probe training 只使用 train snapshot；probe evaluation 只使用 val snapshot。

## 3. 输出 Metrics

```python
@dataclass(frozen=True)
class Phase1LabelPredictabilityMetrics:
    probe_top1_accuracy: float
    probe_top3_accuracy: float
    probe_balanced_accuracy: float
    label_entropy_given_morphology: float
    mutual_information_lift: float
    probe_return_retention: float
```

返回 payload：

```python
Phase1LayerComputation(
    layer_id=4,
    layer_name="label_predictability",
    metrics=Phase1LabelPredictabilityMetrics(...),
    extra_payload={
        "probe_train_accuracy": ...,
        "probe_validation_accuracy": ...,
        "probe_predictability_gap": ...,
        "probe_confusion_matrix": ...,
        "probe_seed": runtime_config.random_seed,
    },
)
```

## 4. 推荐入口

```python
def compute_label_predictability_metrics(
    *,
    model: ArchetypeVQModel,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
    device: torch.device | str,
) -> Phase1LayerComputation:
    ...
```

## 5. Probe Feature 约束

第一版使用 horizon 起点可见状态：

```python
train_x = train_snapshot.states[:, 0, :]
val_x = val_snapshot.states[:, 0, :]
```

如果后续 selector 可见历史窗口，应改为：

```python
states[:, :visible_window, :]
```

并做 flatten 或小型 temporal encoder。

禁止输入：

- 完整未来 horizon 的价格路径；
- demo action；
- demo reward；
- decoded action；
- oracle return；
- 任何 Phase II selector 推理时不可见的未来信息。

Layer 4 的目标是验证 label 可从 selector 可见状态学习，而不是训练一个使用未来信息的
高精度诊断模型。

## 6. Helper 设计

```python
def build_probe_features(states: np.ndarray) -> np.ndarray:
    ...

def train_probe_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    runtime_config: Phase1ValidationRuntimeConfig,
) -> ProbeModel:
    ...

def evaluate_probe(
    probe: ProbeModel,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> ProbeMetrics:
    ...

def compute_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    active_codes: np.ndarray,
) -> float:
    ...

def compute_mutual_information_lift(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> float:
    ...

def decode_probe_top1_actions(
    model: ArchetypeVQModel,
    states: np.ndarray,
    predicted_code_ids: np.ndarray,
    device: torch.device | str,
) -> np.ndarray:
    ...
```

第一版 probe 建议使用 shallow MLP；若需要更快基线，可实现 multinomial logistic
regression 风格的单层线性分类器。

## 7. 计算流程

1. 构造 `train_x = build_probe_features(train_snapshot.states)`。
2. 构造 `val_x = build_probe_features(val_snapshot.states)`。
3. `train_y = train_snapshot.code_ids`，`val_y = val_snapshot.code_ids`。
4. 固定 seed，训练轻量 probe。
5. 在 validation 上输出 `probe_probs`。
6. `probe_top1_accuracy = mean(argmax(probe_probs) == val_y)`。
7. `probe_top3_accuracy = mean(val_y in top3(probe_probs))`。
8. `probe_balanced_accuracy`：对每个 active code 分别计算 recall 后取均值。
9. `label_entropy_given_morphology`：用 Layer 2 的 morphology label 或本层重新
   计算 morphology，统计 `H(label | morphology)`。
10. `mutual_information_lift`：计算 label 与可见 feature/morphology 的 MI，再与
    随机置换 label 后的 MI 均值比较。
11. 用 probe top-1 label 通过 decoder 得到 probe actions，并用 Layer 3 同一
    execution helper 计算 probe decoded return。
12. `probe_return_retention = sum(R_probe - R_flat) / (sum(R_oracle - R_flat) + eps)`。

## 8. Rule 判定

Layer 4 thresholds：

```python
@dataclass(frozen=True)
class Phase1LabelPredictabilityThresholds:
    probe_top1_floor: float = 0.25
    probe_top1_k_factor: float = 1.5
    probe_top3_floor: float = 0.55
    probe_top3_k_factor: float = 3.0
    probe_balanced_accuracy_min: float = 0.25
    mutual_information_lift_min: float = 2.0
    probe_return_retention_min: float = 0.35
```

Hard gates：

- `probe_top1_accuracy >= max(0.25, 1.5 / K)`
- `probe_top3_accuracy >= max(0.55, 3.0 / K)`
- `probe_balanced_accuracy >= 0.25`
- `mutual_information_lift >= 2.0`
- `probe_return_retention >= 0.35`

`label_entropy_given_morphology` 第一版作为 diagnostic/scoring 信号；若熵过高且
top-k accuracy 低，report 应提示 label 可能不可由可见状态学习。

## 9. 缺失数据策略

- 缺少 validation prices 时，`probe_return_retention` 不可计算，应 fail；
- active code 数量小于 2 时，probe accuracy 没有意义，应 fail；
- probe 训练必须 deterministic：固定 seed，并避免 dataloader shuffle 的非确定性；
- 若某些 active code 在 train 中没有样本但在 val 中出现，balanced accuracy 应按
  0 recall 计入；
- 如果 model decode probe label 失败，不能只报告 accuracy，应将
  `probe_return_retention` 标记为 `nan` 并 fail。

## 10. 与其他层的关系

- Layer 1 的 code distribution 决定 top-k floor 中的 `K` 和 active code 集合；
- Layer 2 的 morphology label 可用于 `label_entropy_given_morphology`；
- Layer 3 的 execution helper 和 oracle decoded returns 用于 `probe_return_retention`；
- tie-breaker 的第二优先级是 `probe_top3_accuracy`。

## 11. 防泄漏要求

Layer 4 是最容易引入未来信息泄漏的层，必须满足：

- probe 输入只包含 Phase II selector 推理时可见状态；
- 不使用 demo action、demo reward、decoded action 或未来 price path 作为 feature；
- 所有 feature transform 必须能在 inference 时复现；
- report 中记录 `visible_window`、feature shape、seed 和 probe 类型；
- 如果后续 selector 真实输入发生变化，Layer 4 的 `build_probe_features()` 必须同步更新。

## 12. 测试要点

- probe 在固定 seed 下输出稳定；
- top-3 在 `K < 3` 时有定义；
- balanced accuracy 对 train 缺失但 val 出现的 code 计 0 recall；
- MI lift 的 permutation baseline 固定 seed；
- probe return retention 与 Layer 3 execution helper 口径完全一致；
- feature builder 不读取未来 timestep。
