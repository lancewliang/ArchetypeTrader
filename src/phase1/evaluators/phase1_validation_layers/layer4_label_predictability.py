"""Phase I validation Layer 4: assigned-label predictability raw metrics。

文件功能说明:
    本文件负责计算 assigned label 是否能从 Phase II selector 可见状态中学习出来，
    包括 probe top-1/top-3 accuracy、balanced accuracy、label entropy given
    morphology、mutual information lift 和 probe decoded return retention。

设计边界:
    - probe 只使用 horizon 起点可见状态，不读取未来价格路径、demo action 或 reward
      作为分类输入；
    - 本文件只计算 raw metrics，不判断 pass/fail；
    - probe 当前采用 deterministic linear classifier，不训练主模型；
    - probe return retention 直接复用统一交易执行工具；
    - 缺失 prices 时收益保留指标返回 NaN，由 rules 层失败。

使用场景:
    ``Phase1CodebookEvaluator`` 在 train/validation snapshot 收集完成后调用
    ``compute_label_predictability_metrics()``，验证 Phase I assigned label 是否具备
    后续 Phase II selector 的可学习性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.utils import ActionExecutionCalculator, nan_value as _nan

from ...metrics import (
    Phase1EvaluationSnapshot,
    Phase1LabelPredictabilityMetrics,
    Phase1LabelPredictabilityPayload,
    Phase1LayerComputation,
    Phase1ValidationRuntimeConfig,
)
from .layer2_behavior_quality import classify_market_morphology
from .layer3_oracle_profitability import decode_labels


_EPS = 1e-12


@dataclass(frozen=True)
class LinearProbe:
    """确定性线性 probe 模型。

    字段说明:
        labels: probe 训练集中出现过的 code label。
        weight: 线性分类器权重，shape=[num_labels, feature_dim]。
        bias: 线性分类器 bias，shape=[num_labels]。
        feature_mean: 训练集 feature 均值。
        feature_std: 训练集 feature 标准差。

    使用场景:
        Layer 4 用轻量、可复现的可训练 probe 估计 assigned label 可预测性。
    """

    labels: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray


ProbeModel = LinearProbe


@dataclass(frozen=True)
class ProbeMetrics:
    """probe validation 评估结果。

    字段说明:
        probe_probs: 按 ``probe.labels`` 列顺序排列的预测概率，shape=[N, K_seen]。
        ranked_labels: 每个样本按 logit 从高到低排序的 label id，shape=[N, K_seen]。
        top1_predictions: 每个样本的 top-1 label id，shape=[N]。
        probe_top1_accuracy: validation top-1 label accuracy。
        probe_top3_accuracy: validation top-3 label accuracy；K<3 时使用 top-K。
        probe_validation_accuracy: 与 ``probe_top1_accuracy`` 相同，保留明确语义供
            payload/report 使用。
    """

    probe_probs: np.ndarray
    ranked_labels: np.ndarray
    top1_predictions: np.ndarray
    probe_top1_accuracy: float
    probe_top3_accuracy: float
    probe_validation_accuracy: float


def build_probe_features(states: np.ndarray) -> np.ndarray:
    """构造 probe 分类特征。

    输入参数:
        states: 状态数组，通常形状为 ``[N, H, state_dim]``。

    输出:
        ``[N, feature_dim]`` 形状的 probe feature。三维输入只取 horizon 起点
        ``states[:, 0, :]``，避免泄露未来信息。

    使用场景:
        Layer 4 probe 训练和 validation evaluation 的统一 feature 构造入口。
    """

    values = np.asarray(states, dtype=np.float64)
    if values.ndim == 3:
        return values[:, 0, :].reshape(values.shape[0], -1)
    if values.ndim == 2:
        return values
    return values.reshape(values.shape[0], -1)


def _balanced_class_weights(
    target_indices: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """根据训练集标签频次生成轻量 class-balanced probe 权重。"""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    labels = np.asarray(target_indices, dtype=np.int64)
    if labels.size == 0:
        return np.ones(num_classes, dtype=np.float64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    positive = counts[counts > 0]
    mean_count = float(np.mean(positive)) if positive.size else 1.0
    weights = np.ones(num_classes, dtype=np.float64)
    seen = counts > 0
    weights[seen] = np.sqrt(mean_count / np.maximum(counts[seen], 1.0))
    return weights


def train_probe_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    runtime_config: Phase1ValidationRuntimeConfig,
    class_weights: np.ndarray | None = None,
) -> ProbeModel:
    """训练 deterministic linear probe。

    输入参数:
        train_x: probe train features，形状为 ``[N, D]``。
        train_y: assigned code labels，形状为 ``[N]``。
        runtime_config: 提供 probe 训练 epoch、learning rate、batch size 和 seed。
        class_weights: 可选 class-balanced 权重，按 ``unique_labels`` 顺序排列。

    输出:
        ``LinearProbe``，包含标准化参数和训练后的线性分类器参数。

    使用场景:
        使用 runtime config 中的 probe_epochs、probe_learning_rate 和
        probe_batch_size，以低成本、可复现方式估计 label 是否和可见状态存在
        可学习关系。训练过程默认使用 class-balanced cross entropy，避免 probe
        只学会预测高频 code。
    """

    features = np.asarray(train_x, dtype=np.float64)
    labels = np.asarray(train_y, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("train_x must be a 2D feature matrix")
    if labels.ndim != 1:
        raise ValueError("train_y must be a 1D label array")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("train_x and train_y must have the same sample count")
    if features.shape[0] == 0:
        raise ValueError("train_x must contain at least one sample")

    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    feature_std[feature_std <= _EPS] = 1.0
    normalized = ((features - feature_mean) / feature_std).astype(np.float32)

    unique_labels = np.unique(labels.astype(np.int64))
    label_to_index = {int(label): index for index, label in enumerate(unique_labels)}
    target_indices = np.asarray(
        [label_to_index[int(label)] for label in labels],
        dtype=np.int64,
    )
    if class_weights is None:
        class_weights = _balanced_class_weights(target_indices, unique_labels.size)
    class_weights_array = np.asarray(class_weights, dtype=np.float32)
    if class_weights_array.ndim != 1 or class_weights_array.shape[0] != unique_labels.size:
        raise ValueError(
            "class_weights must be a 1D array aligned with unique_labels"
        )

    torch.manual_seed(int(runtime_config.random_seed))
    x_tensor = torch.as_tensor(normalized, dtype=torch.float32)
    y_tensor = torch.as_tensor(target_indices, dtype=torch.long)
    weight_tensor = torch.as_tensor(class_weights_array, dtype=torch.float32)
    model = torch.nn.Linear(x_tensor.shape[1], unique_labels.size)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(runtime_config.probe_learning_rate),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(runtime_config.random_seed))
    batch_size = max(1, int(runtime_config.probe_batch_size))
    epochs = max(1, int(runtime_config.probe_epochs))

    model.train()
    for _ in range(epochs):
        order = torch.randperm(x_tensor.shape[0], generator=generator)
        for start in range(0, x_tensor.shape[0], batch_size):
            batch_index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(x_tensor[batch_index]),
                y_tensor[batch_index],
                weight=weight_tensor,
            )
            loss.backward()
            optimizer.step()

    weight = model.weight.detach().cpu().numpy().astype(np.float64)
    bias = model.bias.detach().cpu().numpy().astype(np.float64)
    return LinearProbe(
        labels=unique_labels,
        weight=weight,
        bias=bias,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def _probe_logits(probe: ProbeModel, features: np.ndarray) -> np.ndarray:
    """计算 probe logits，列顺序与 ``probe.labels`` 一致。"""

    values = np.asarray(features, dtype=np.float64)
    normalized = (values - probe.feature_mean) / probe.feature_std
    return normalized @ probe.weight.T + probe.bias


def _softmax(logits: np.ndarray) -> np.ndarray:
    """稳定 softmax。"""

    if logits.shape[0] == 0:
        return np.zeros_like(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)



def evaluate_probe(
    probe: ProbeModel,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> ProbeMetrics:
    """在 validation split 上评估 probe label 预测能力。

    输入参数:
        probe: ``train_probe_classifier()`` 返回的 probe。
        val_x: validation features，形状为 ``[N, D]``。
        val_y: validation assigned labels，形状为 ``[N]``。

    输出:
        ``ProbeMetrics``，包含 probability、top-1/top-3 结果和 validation accuracy。
    """

    labels = np.asarray(val_y, dtype=np.int64)
    logits = _probe_logits(probe, val_x)
    if logits.shape[0] != labels.size:
        raise ValueError("val_x and val_y must have the same sample count")
    probe_probs = _softmax(logits)
    ranked_labels = probe.labels[np.argsort(-logits, axis=1)]
    if labels.size == 0:
        top1_predictions = np.asarray([], dtype=np.int64)
        top1_accuracy = _nan()
        top3_accuracy = _nan()
    else:
        top1_predictions = ranked_labels[:, 0]
        top_k = min(3, ranked_labels.shape[1])
        top1_accuracy = float(np.mean(top1_predictions == labels))
        top3_accuracy = float(
            np.mean(
                [
                    label in row[:top_k]
                    for label, row in zip(labels, ranked_labels, strict=False)
                ]
            )
        )
    return ProbeMetrics(
        probe_probs=probe_probs,
        ranked_labels=ranked_labels,
        top1_predictions=top1_predictions,
        probe_top1_accuracy=top1_accuracy,
        probe_top3_accuracy=top3_accuracy,
        probe_validation_accuracy=top1_accuracy,
    )


def compute_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    active_codes: np.ndarray,
) -> float:
    """计算 active code recall 的平均值。

    输入参数:
        y_true: validation assigned labels。
        y_pred: probe top-1 predicted labels。
        active_codes: validation split 中需要纳入统计的 active code。

    输出:
        每个 active code recall 的平均值；无 active code 时返回 NaN。

    使用场景:
        防止 probe 只预测高频 code 而获得虚高 top-1 accuracy。
    """

    active = np.asarray(active_codes, dtype=np.int64)
    if active.size < 2:
        return _nan()
    true_values = np.asarray(y_true, dtype=np.int64)
    pred_values = np.asarray(y_pred, dtype=np.int64)
    recalls: list[float] = []
    for code_id in active:
        mask = true_values == code_id
        if not np.any(mask):
            recalls.append(0.0)
            continue
        recalls.append(float(np.mean(pred_values[mask] == code_id)))
    return float(np.mean(recalls)) if recalls else _nan()


def _discretize_features(features: np.ndarray, *, max_features: int = 4) -> np.ndarray:
    """把连续 feature 粗粒度离散化。

    输入参数:
        features: 连续 feature 数组，形状为 ``[N, D]``。
        max_features: 最多使用前多少个 feature 做离散化。

    输出:
        ``[N]`` 形状的离散 feature bin id。

    使用场景:
        mutual information 估计需要离散变量，本 helper 提供轻量 deterministic
        分箱。
    """

    selected = np.asarray(features[:, :max_features], dtype=np.float64)
    bins: list[np.ndarray] = []
    for column in selected.T:
        quantiles = np.unique(np.quantile(column, [0.25, 0.5, 0.75]))
        bins.append(np.digitize(column, quantiles, right=False))
    if not bins:
        return np.zeros(features.shape[0], dtype=np.int64)
    stacked = np.stack(bins, axis=1)
    return np.ravel_multi_index(stacked.T, dims=(4,) * stacked.shape[1])


def _mutual_information(labels: np.ndarray, feature_bins: np.ndarray) -> float:
    """计算 label 与离散 feature bin 的 mutual information。

    输入参数:
        labels: assigned code labels。
        feature_bins: 离散化后的 feature bin id。

    输出:
        mutual information 标量。

    使用场景:
        计算 observed MI 以及随机置换 baseline MI。
    """

    label_values, label_inverse = np.unique(labels, return_inverse=True)
    bin_values, bin_inverse = np.unique(feature_bins, return_inverse=True)
    joint = np.zeros((label_values.size, bin_values.size), dtype=np.float64)
    for label_index, bin_index in zip(label_inverse, bin_inverse, strict=False):
        joint[label_index, bin_index] += 1.0
    joint /= max(1.0, np.sum(joint))
    label_prob = np.sum(joint, axis=1, keepdims=True)
    bin_prob = np.sum(joint, axis=0, keepdims=True)
    expected = label_prob @ bin_prob
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / (expected[mask] + _EPS))))


def compute_mutual_information_lift(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> float:
    """计算 mutual information 相对随机置换 baseline 的提升倍数。

    输入参数:
        features: probe features。
        labels: assigned code labels。
        seed: 随机置换 label 的 seed。

    输出:
        ``observed_mi / shuffled_mi_mean``；observed MI 非正时返回 0。

    使用场景:
        判断 label 与 horizon 起点可见状态之间是否存在显著统计关系。
    """

    label_values = np.asarray(labels, dtype=np.int64)
    if label_values.size == 0:
        return _nan()
    feature_bins = _discretize_features(np.asarray(features, dtype=np.float64))
    observed = _mutual_information(label_values, feature_bins)
    rng = np.random.default_rng(seed)
    shuffled_values = []
    for _ in range(5):
        shuffled = np.array(label_values, copy=True)
        rng.shuffle(shuffled)
        shuffled_values.append(_mutual_information(shuffled, feature_bins))
    baseline = float(np.mean(shuffled_values))
    if observed <= 0.0:
        return 0.0
    return float(observed / (baseline + _EPS))


def _label_entropy_given_morphology(
    labels: np.ndarray,
    morphologies: np.ndarray,
) -> float:
    """计算给定 morphology 后的 label 条件熵。

    输入参数:
        labels: assigned code labels。
        morphologies: 每条 horizon 的 morphology 标签。

    输出:
        ``H(label | morphology)``；morphology 缺失或长度不匹配时返回 NaN。

    使用场景:
        诊断 label 是否仍有大量无法由市场形态解释的不确定性。
    """

    if morphologies.size != labels.size:
        return _nan()
    total_entropy = 0.0
    for morphology in np.unique(morphologies):
        mask = morphologies == morphology
        counts = np.bincount(labels[mask].astype(np.int64))
        probabilities = counts[counts > 0] / max(1, np.sum(counts))
        entropy = -float(np.sum(probabilities * np.log(probabilities + _EPS)))
        total_entropy += entropy * (np.sum(mask) / labels.size)
    return float(total_entropy)


def _label_entropy(labels: np.ndarray) -> float:
    """计算 assigned label 的全局熵 H(label)。"""

    if labels.size == 0:
        return _nan()
    counts = np.bincount(labels.astype(np.int64))
    probabilities = counts[counts > 0] / max(1, np.sum(counts))
    return float(-np.sum(probabilities * np.log(probabilities + _EPS)))


def _confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_codes: int,
) -> list[list[int]]:
    """构造 validation probe confusion matrix。"""

    matrix = np.zeros((num_codes, num_codes), dtype=np.int64)
    for target, predicted in zip(y_true, y_pred, strict=False):
        if 0 <= int(target) < num_codes and 0 <= int(predicted) < num_codes:
            matrix[int(target), int(predicted)] += 1
    return matrix.tolist()


def decode_probe_top1_actions(
    model: Any,
    states: np.ndarray,
    predicted_code_ids: np.ndarray,
    device: torch.device | str,
) -> np.ndarray:
    """用指定 label 调用 decoder 生成 probe action。

    输入参数:
        model: ``ArchetypeVQModel`` 或兼容对象，需要提供 ``quantizer`` 和 ``decoder``。
        states: 状态序列数组，形状为 ``[N, H, state_dim]``。
        code_ids: probe 预测或指定的 code id，形状为 ``[N]``。
        device: decoder 推理设备。

    输出:
        ``[N, H]`` 形状的 argmax decoded action 数组。

    使用场景:
        计算 probe top-1 label 执行后的 return retention。
    """

    return decode_labels(
        model=model,
        states=states,
        code_ids=predicted_code_ids,
        device=device,
    )


def _probe_return_retention(
    *,
    model: Any,
    val_snapshot: Phase1EvaluationSnapshot,
    predicted_labels: np.ndarray,
    runtime_config: Phase1ValidationRuntimeConfig,
    device: torch.device | str,
) -> float:
    """计算 probe label 执行收益相对 oracle label 的保留比例。

    输入参数:
        model: VQ 模型或兼容对象，用于解码 probe labels。
        val_snapshot: validation snapshot，提供 states、prices 和 oracle decoded actions。
        predicted_labels: probe top-1 预测 code id。
        runtime_config: validation 运行参数，提供手续费率。
        device: decoder 推理设备。

    输出:
        ``sum(R_probe - R_flat) / sum(R_oracle - R_flat)``；缺少 prices 时返回 NaN。

    使用场景:
        判断可预测 label 经过 decoder 执行后能保留多少 oracle assigned-label 收益。
    """

    if val_snapshot.prices is None:
        return _nan()
    probe_actions = decode_probe_top1_actions(
        model=model,
        states=val_snapshot.states,
        predicted_code_ids=predicted_labels,
        device=device,
    )
    probe_returns = ActionExecutionCalculator.execute_actions(
        val_snapshot.prices,
        probe_actions,
        runtime_config.fee_rate,
        val_snapshot.depthprices,
    ).returns
    oracle_returns = ActionExecutionCalculator.execute_actions(
        val_snapshot.prices,
        val_snapshot.decoded_actions,
        runtime_config.fee_rate,
        val_snapshot.depthprices,
    ).returns
    flat_returns = np.zeros_like(oracle_returns)
    return float(
        np.nansum(probe_returns - flat_returns)
        / (np.nansum(oracle_returns - flat_returns) + _EPS)
    )


def compute_label_predictability_metrics(
    *,
    model: Any,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
    device: torch.device | str,
) -> Phase1LayerComputation:
    """计算 Layer 4 assigned-label 可预测性 raw metrics。

    功能说明:
        用 train snapshot 训练 deterministic linear probe，并在 validation snapshot
        上评估 top-k accuracy、balanced accuracy、mutual information lift 和 probe
        return retention。

    输入参数:
        model: VQ 模型或兼容对象，用于 probe top-1 label 解码。
        train_snapshot: 训练集 snapshot，提供 probe feature 和 assigned labels。
        val_snapshot: 验证集 snapshot，提供 probe feature、assigned labels、prices
            和 oracle decoded actions。
        runtime_config: validation 运行参数，提供随机 seed 和手续费率。
        device: decoder 推理设备。

    输出:
        ``Phase1LayerComputation``，其中 ``metrics`` 为
        ``Phase1LabelPredictabilityMetrics``，``extra_payload`` 包含 probe train
        accuracy、validation accuracy、gap 和 seed。

    使用场景:
        full checkpoint validation 的第四层 selector 可学习性 raw metric 计算；
        结果交给 ``evaluate_label_predictability_rules()`` 判定 hard gate。
    """

    train_x = build_probe_features(train_snapshot.states)
    val_x = build_probe_features(val_snapshot.states)
    train_y = np.asarray(train_snapshot.code_ids, dtype=np.int64)
    val_y = np.asarray(val_snapshot.code_ids, dtype=np.int64)

    if np.unique(train_y).size < 2 or np.unique(val_y).size < 2:
        metrics = Phase1LabelPredictabilityMetrics(
            probe_top1_accuracy=_nan(),
            probe_top3_accuracy=_nan(),
            probe_balanced_accuracy=_nan(),
            label_entropy_given_morphology=_nan(),
            mutual_information_lift=_nan(),
            probe_return_retention=_nan(),
            label_entropy=_label_entropy(val_y),
            num_codes=int(max(np.unique(train_y).size, np.unique(val_y).size)),
        )
        return Phase1LayerComputation(
            layer_id=4,
            layer_name="label_predictability",
            metrics=metrics,
            extra_payload=Phase1LabelPredictabilityPayload(
                probe_train_accuracy=_nan(),
                probe_validation_accuracy=_nan(),
                probe_predictability_gap=_nan(),
                probe_confusion_matrix=tuple(),
                probe_seed=runtime_config.random_seed,
            ),
        )

    probe = train_probe_classifier(train_x, train_y, runtime_config)
    probe_metrics = evaluate_probe(probe, val_x, val_y)
    top1 = probe_metrics.top1_predictions
    active_codes = np.unique(val_y)
    num_codes = (
        int(val_snapshot.distances.shape[1])
        if np.asarray(val_snapshot.distances).ndim == 2
        and val_snapshot.distances.shape[1] > 0
        else int(max(np.max(train_y), np.max(val_y)) + 1)
    )
    morphologies = classify_market_morphology(
        val_snapshot.prices,
        fee_rate=runtime_config.fee_rate,
    )

    metrics = Phase1LabelPredictabilityMetrics(
        probe_top1_accuracy=probe_metrics.probe_top1_accuracy,
        probe_top3_accuracy=probe_metrics.probe_top3_accuracy,
        probe_balanced_accuracy=compute_balanced_accuracy(val_y, top1, active_codes),
        label_entropy_given_morphology=_label_entropy_given_morphology(
            val_y,
            morphologies,
        ),
        mutual_information_lift=compute_mutual_information_lift(
            val_x,
            val_y,
            seed=runtime_config.random_seed,
        ),
        probe_return_retention=_probe_return_retention(
            model=model,
            val_snapshot=val_snapshot,
            predicted_labels=top1,
            runtime_config=runtime_config,
            device=device,
        ),
        label_entropy=_label_entropy(val_y),
        num_codes=num_codes,
    )
    train_probe_metrics = evaluate_probe(probe, train_x, train_y)
    train_accuracy = train_probe_metrics.probe_top1_accuracy
    return Phase1LayerComputation(
        layer_id=4,
        layer_name="label_predictability",
        metrics=metrics,
        extra_payload=Phase1LabelPredictabilityPayload(
            probe_train_accuracy=train_accuracy,
            probe_validation_accuracy=metrics.probe_top1_accuracy,
            probe_predictability_gap=float(
                train_accuracy - metrics.probe_top1_accuracy
            ),
            probe_confusion_matrix=_confusion_matrix(
                val_y,
                top1,
                num_codes=num_codes,
            ),
            probe_seed=runtime_config.random_seed,
        ),
    )


__all__ = [
    "LinearProbe",
    "ProbeMetrics",
    "ProbeModel",
    "build_probe_features",
    "compute_balanced_accuracy",
    "compute_label_predictability_metrics",
    "compute_mutual_information_lift",
    "decode_probe_top1_actions",
    "evaluate_probe",
    "train_probe_classifier",
]
