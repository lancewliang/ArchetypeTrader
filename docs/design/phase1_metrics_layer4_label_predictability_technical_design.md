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

## 6. 公共 Helper 设计

Layer 4 的公共 helper 用于把 label predictability 计算拆成可独立测试、可复用的
步骤。公共 helper 必须保持两个约束：

- 只消费 Phase II selector 推理时可见的信息；
- 与其他 layer 共享的指标口径必须复用既有 helper，尤其是 Layer 3 的收益执行
  helper。

推荐公共接口如下：

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

### 6.1 `build_probe_features()`

功能：把 snapshot 中的 `states` 转成 probe 可使用的二维特征矩阵。

输入：

- `states`: 通常形状为 `[N, H, state_dim]`；其中 `N` 是 horizon 样本数，`H`
  是 horizon 长度。

输出：

- `features`: 形状为 `[N, feature_dim]` 的二维数组。

第一版实现必须只取 horizon 起点状态：

```python
features = states[:, 0, :]
```

如果输入已经是二维 `[N, state_dim]`，可以直接返回或做必要的 dtype 转换。如果输入
维度更高，应只在不引入未来信息的前提下 reshape。

禁止在本 helper 中读取或派生以下信息：

- 完整未来 horizon 的 price path；
- demo action、demo reward；
- decoded action；
- oracle return；
- 任何 Phase II selector 在线推理时不可见的字段。

后续如果 Phase II selector 的真实输入从单点状态扩展为可见历史窗口，本 helper
必须同步调整为：

```python
features = states[:, :visible_window, :]
```

然后再 flatten 或送入轻量 temporal encoder。report 中应记录 `visible_window` 和
最终 feature shape。

### 6.2 `train_probe_classifier()`

功能：用 train snapshot 训练一个轻量 probe，从可见状态预测 assigned label。

输入：

- `train_x`: `build_probe_features(train_snapshot.states)` 的输出，形状 `[N_train, D]`；
- `train_y`: `train_snapshot.code_ids`，形状 `[N_train]`；
- `runtime_config`: 提供 `probe_epochs`、`probe_learning_rate`、
  `probe_batch_size` 和 `random_seed`。

输出：

- `ProbeModel`: 已训练好的 probe，至少应能对 validation features 输出每个 label
  的 logits 或 probability。

实现要求：

- 训练只允许使用 train split，不允许读取 validation label；
- 必须固定 `runtime_config.random_seed`，保证同一输入下结果稳定；
- 第一版推荐 shallow MLP；若追求更快、更容易复现的 baseline，可使用单层线性
  softmax classifier；
- label 空间以 train split 中出现过的 code 为主，但 evaluation 时必须能处理
  validation 中出现、train 中未出现的 code，并把对应 recall 计为 0；
- probe 是诊断模型，不应反向更新 Phase I 主模型。

建议 `ProbeModel` 至少保存以下内容：

- label id 与 classifier 输出列的映射；
- feature 标准化参数；
- 模型权重；
- probe 类型和 seed，便于写入 report。

### 6.3 `evaluate_probe()`

功能：在 validation split 上评估 probe 的 label 预测能力。

输入：

- `probe`: `train_probe_classifier()` 返回的模型；
- `val_x`: `build_probe_features(val_snapshot.states)` 的输出；
- `val_y`: `val_snapshot.code_ids`。

输出：

- `ProbeMetrics`: 建议包含 `probe_probs`、`top1_predictions`、
  `probe_top1_accuracy`、`probe_top3_accuracy`、`probe_validation_accuracy`。

核心计算：

```python
top1 = np.argmax(probe_probs, axis=1)
probe_top1_accuracy = np.mean(top1 == val_y)

top_k = min(3, probe_probs.shape[1])
probe_top3_accuracy = np.mean(val_y in top_k_predictions)
```

注意事项：

- `K < 3` 时 top-3 必须退化为 top-`K`，不能越界；
- `probe_probs` 的列顺序必须能映射回真实 code id，不能默认列号等于 code id；
- 如果 validation 中存在 train 未见过的 code，top-k 命中只能为 false；
- train accuracy 与 validation accuracy 应同时记录，用于计算
  `probe_predictability_gap`。

### 6.4 `compute_balanced_accuracy()`

功能：按 active code 逐个计算 recall 后取平均，避免高频 code 主导 top-1 accuracy。

输入：

- `y_true`: validation assigned label；
- `y_pred`: probe top-1 预测 label；
- `active_codes`: 需要纳入统计的 code 集合，通常来自 validation active code。

输出：

- `probe_balanced_accuracy`: 每个 active code recall 的均值。

计算口径：

```python
recall_k = np.mean(y_pred[y_true == k] == k)
probe_balanced_accuracy = np.mean([recall_k for k in active_codes])
```

边界策略：

- 如果某个 `active_code` 在 validation 中没有样本，recall 计为 0；
- 如果某个 code 在 train 中缺失但在 validation 中出现，也必须作为 active code
  纳入统计，recall 计为 0；
- active code 数量小于 2 时，balanced accuracy 没有诊断意义，应返回 `nan`
  并由 rules 层 fail。

### 6.5 `compute_mutual_information_lift()`

功能：衡量 label 与 selector 可见 feature 之间的统计依赖强度，作为 probe accuracy
之外的补充证据。

输入：

- `features`: `build_probe_features()` 的输出；
- `labels`: assigned code ids；
- `seed`: permutation baseline 使用的随机种子。

输出：

- `mutual_information_lift`: 真实 MI 相对随机置换 label baseline 的提升倍数。

计算流程：

1. 将连续 feature 离散化，或使用可复现的离散/连续 MI estimator。
2. 计算真实标签下的 `observed_mi = I(features; labels)`。
3. 固定 `seed`，多次随机置换 `labels`，计算 `shuffled_mi_mean`。
4. 返回：

```python
mutual_information_lift = observed_mi / (shuffled_mi_mean + eps)
```

解释：

- `lift` 接近 1 表示 label 与可见 feature 的关系接近随机；
- `lift` 明显大于 1 表示 assigned label 至少包含可由可见状态解释的信息；
- Layer 4 hard gate 第一版要求 `mutual_information_lift >= 2.0`。

实现注意：

- permutation baseline 必须 deterministic；
- feature 离散化不能使用未来信息；
- 如果 `observed_mi <= 0`，建议返回 0；
- 如果样本数过小导致 MI 不稳定，应在 report 中标记 diagnostic 风险。

### 6.6 `decode_probe_top1_actions()`

功能：把 probe top-1 预测 label 通过 frozen decoder 解码成 action 序列，用于计算
probe decoded return。

输入：

- `model`: `ArchetypeVQModel` 或兼容对象；
- `states`: validation states，形状通常为 `[N, H, state_dim]`；
- `predicted_code_ids`: probe top-1 预测 code id，形状 `[N]`；
- `device`: decoder 推理设备。

输出：

- `probe_decoded_actions`: 形状 `[N, H]` 的 action id 数组。

推荐实现流程：

```python
z_q = model.quantizer.embedding_from_code(predicted_code_ids)
decoded_logits = model.decoder(states, z_q)
probe_decoded_actions = decoded_logits.argmax(axis=-1)
```

本 helper 不直接计算收益。收益必须交给 Layer 3 同一套 execution helper：

```python
R_probe = execution_helper(prices, probe_decoded_actions)
R_oracle = execution_helper(prices, val_snapshot.decoded_actions)
R_flat = flat_baseline(prices)

probe_return_retention = (
    sum(R_probe - R_flat) / (sum(R_oracle - R_flat) + eps)
)
```

边界策略：

- 缺少 validation prices 时，`probe_return_retention` 不可计算，应返回 `nan`
  并由 rules 层 fail；
- model decode 失败时，不能只报告 accuracy，必须将 `probe_return_retention`
  标记为 `nan` 并 fail；
- decoder 必须使用 eval/no-grad 模式，不更新模型参数；
- 收益执行口径必须与 Layer 3 完全一致，包括手续费和 action execution 规则。

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
