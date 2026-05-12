"""Phase I validation Layer 4: assigned-label predictability raw metrics。

文件功能说明:
    本文件负责计算 assigned label 是否能从 Phase II selector 可见状态中学习出来，
    包括 probe top-1/top-3 accuracy、balanced accuracy、label entropy given
    morphology、mutual information lift 和 probe decoded return retention。

设计边界:
    - probe 只使用 horizon 起点可见状态，不读取未来价格路径、demo action 或 reward
      作为分类输入；
    - 本文件只计算 raw metrics，不判断 pass/fail；
    - probe 当前采用 deterministic centroid baseline，不训练主模型；
    - probe return retention 复用 Layer 3 的统一 execution helper；
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
    Phase1LayerComputation,
    Phase1ValidationRuntimeConfig,
)
from .layer2_behavior_quality import classify_market_morphology


_EPS = 1e-12


@dataclass(frozen=True)
class CentroidProbe:
    """确定性 centroid probe 模型。

    字段说明:
        labels: probe 训练集中出现过的 code label。
        centroids: 每个 label 在标准化 feature 空间中的中心。
        feature_mean: 训练集 feature 均值。
        feature_std: 训练集 feature 标准差。

    使用场景:
        Layer 4 用轻量、可复现的 baseline 估计 assigned label 可预测性。
    """

    labels: np.ndarray
    centroids: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray


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


def _fit_centroid_probe(features: np.ndarray, labels: np.ndarray) -> CentroidProbe:
    """训练 deterministic centroid probe。

    输入参数:
        features: probe train features，形状为 ``[N, D]``。
        labels: assigned code labels，形状为 ``[N]``。

    输出:
        ``CentroidProbe``，包含每个 label 的标准化 feature centroid。

    使用场景:
        以低成本、可复现方式估计 label 是否和可见状态存在可学习关系。
    """

    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    feature_std[feature_std <= _EPS] = 1.0
    normalized = (features - feature_mean) / feature_std

    unique_labels = np.unique(labels.astype(np.int64))
    centroids = np.stack(
        [np.mean(normalized[labels == label], axis=0) for label in unique_labels],
        axis=0,
    )
    return CentroidProbe(
        labels=unique_labels,
        centroids=centroids,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def _predict_probe(probe: CentroidProbe, features: np.ndarray) -> np.ndarray:
    """使用 centroid probe 输出按距离排序的候选 label。

    输入参数:
        probe: 已拟合的 ``CentroidProbe``。
        features: validation features，形状为 ``[N, D]``。

    输出:
        ``[N, K_seen]`` 形状的 label 排名数组，每行按距离从近到远排序。

    使用场景:
        计算 probe top-1/top-3 accuracy 和 train/validation accuracy gap。
    """

    normalized = (features - probe.feature_mean) / probe.feature_std
    distances = np.linalg.norm(
        normalized[:, None, :] - probe.centroids[None, :, :],
        axis=-1,
    )
    order = np.argsort(distances, axis=1)
    return probe.labels[order]


def _balanced_accuracy(
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

    recalls: list[float] = []
    for code_id in active_codes:
        mask = y_true == code_id
        if not np.any(mask):
            recalls.append(0.0)
            continue
        recalls.append(float(np.mean(y_pred[mask] == code_id)))
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


def _mutual_information_lift(
    features: np.ndarray,
    labels: np.ndarray,
    *,
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

    feature_bins = _discretize_features(features)
    observed = _mutual_information(labels, feature_bins)
    rng = np.random.default_rng(seed)
    shuffled_values = []
    for _ in range(5):
        shuffled = np.array(labels, copy=True)
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


def _decode_labels(
    *,
    model: Any,
    states: np.ndarray,
    code_ids: np.ndarray,
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

    if model is None:
        raise ValueError("model is required to decode probe labels")
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    with torch.no_grad():
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=torch_device)
        label_tensor = torch.as_tensor(code_ids, dtype=torch.long, device=torch_device)
        z_q = model.quantizer.embedding_from_code(label_tensor)
        logits = model.decoder(state_tensor, z_q)
        return logits.argmax(dim=-1).cpu().numpy()


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
    probe_actions = _decode_labels(
        model=model,
        states=val_snapshot.states,
        code_ids=predicted_labels,
        device=device,
    )
    probe_returns = ActionExecutionCalculator.execute_actions(
        val_snapshot.prices,
        probe_actions,
        runtime_config.fee_rate,
    ).returns
    oracle_returns = ActionExecutionCalculator.execute_actions(
        val_snapshot.prices,
        val_snapshot.decoded_actions,
        runtime_config.fee_rate,
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
        用 train snapshot 训练 deterministic centroid probe，并在 validation snapshot
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
            extra_payload={"probe_seed": runtime_config.random_seed},
        )

    probe = _fit_centroid_probe(train_x, train_y)
    ranked_labels = _predict_probe(probe, val_x)
    top1 = ranked_labels[:, 0]
    top_k = min(3, ranked_labels.shape[1])
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
        probe_top1_accuracy=float(np.mean(top1 == val_y)),
        probe_top3_accuracy=float(
            np.mean([label in row[:top_k] for label, row in zip(val_y, ranked_labels, strict=False)])
        ),
        probe_balanced_accuracy=_balanced_accuracy(val_y, top1, active_codes),
        label_entropy_given_morphology=_label_entropy_given_morphology(
            val_y,
            morphologies,
        ),
        mutual_information_lift=_mutual_information_lift(
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
    train_top1 = _predict_probe(probe, train_x)[:, 0]
    return Phase1LayerComputation(
        layer_id=4,
        layer_name="label_predictability",
        metrics=metrics,
        extra_payload={
            "probe_train_accuracy": float(np.mean(train_top1 == train_y)),
            "probe_validation_accuracy": metrics.probe_top1_accuracy,
            "probe_predictability_gap": float(
                np.mean(train_top1 == train_y) - metrics.probe_top1_accuracy
            ),
            "probe_seed": runtime_config.random_seed,
        },
    )


__all__ = [
    "build_probe_features",
    "compute_label_predictability_metrics",
]
