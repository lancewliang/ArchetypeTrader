"""Phase I validation Layer 1: VQ internal health raw metrics。

文件功能说明:
    本文件负责计算 VQ codebook 内部健康度和稳定性 raw metrics，包括动作重构
    准确率、train/validation loss gap、code 使用分布、perplexity、dead code、
    assignment churn、lifetime、量化距离、nearest-second margin 和交易方向保真度。

设计边界:
    - 只消费 evaluator 已收集好的 snapshot 和 assignment history；
    - 不访问模型、DataLoader 或文件系统；
    - 不计算 morphology、motif 或 profitability；
    - 不做 pass/fail 判定，所有 hard gate 由 rules 层统一执行。

使用场景:
    ``Phase1CodebookEvaluator`` 在收集 train/validation snapshot 后调用
    ``compute_vq_internal_metrics()``，输出 ``Phase1VQInternalMetrics`` 给
    rules、score、report 和 checkpoint selector 使用。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.utils import nan_value as _nan

from ...metrics import (
    CodeAssignmentSnapshot,
    Phase1EvaluationSnapshot,
    Phase1LayerComputation,
    Phase1VQInternalPayload,
    Phase1VQInternalMetrics,
    Phase1ValidationRuntimeConfig,
)
from .hungarian_matching_helper import (
    active_codes_aligned_to_current,
    align_previous_codes_to_current,
)


_EPS = 1e-12


def _num_codes(
    snapshot: Phase1EvaluationSnapshot,
    runtime_config: Phase1ValidationRuntimeConfig,
) -> int:
    """推断当前 snapshot 的 codebook size。

    输入参数:
        snapshot: validation snapshot，当运行配置未显式提供 codebook size 时，
            从 ``distances.shape[1]`` 推断。不能从已使用的 ``code_ids`` 反推，
            否则会漏掉尾部 dead codes。
        runtime_config: validation 运行参数；``codebook_size`` 显式设置时优先使用。

    输出:
        code 数量；无法推断时返回 0。

    使用场景:
        计算 code occupancy、active ratio、dead ratio 和 normalized perplexity。
    """

    if runtime_config.codebook_size is not None:
        if runtime_config.codebook_size <= 0:
            raise ValueError("runtime_config.codebook_size must be positive when set")
        return int(runtime_config.codebook_size)

    distances = np.asarray(snapshot.distances)
    if distances.ndim == 2 and distances.shape[1] > 0:
        return int(distances.shape[1])
    return 0


def compute_action_accuracy(demo: np.ndarray, decoded: np.ndarray) -> float:
    """计算 decoder 重建 action 的逐 timestep 准确率。

    输入参数:
        demo: DP teacher action 数组，通常为 ``[N, H]``。
        decoded: decoder 输出离散 action 数组，shape 必须与 ``demo`` 一致。

    输出:
        按全部非 batch 维度展开后的 ``mean(demo == decoded)``；空输入返回 NaN。

    使用场景:
        计算 ``validation_action_accuracy``。
    """

    demo_values = np.asarray(demo)
    decoded_values = np.asarray(decoded)
    if demo_values.shape != decoded_values.shape:
        raise ValueError(
            "demo and decoded actions must have the same shape, "
            f"got {demo_values.shape} and {decoded_values.shape}"
        )
    if demo_values.size == 0:
        return _nan()
    return float(np.mean(decoded_values == demo_values))


def compute_code_distribution(code_ids: np.ndarray, k: int) -> np.ndarray:
    """计算 code occupancy 分布。

    输入参数:
        code_ids: 每个样本的 assigned code id。
        k: codebook size。

    输出:
        ``[K]`` 形状的 occupancy 概率分布；无样本时返回全 0。

    使用场景:
        active code ratio、max occupancy、dead code ratio 和 perplexity 计算。
    """

    if k <= 0:
        return np.asarray([], dtype=np.float64)
    values = np.asarray(code_ids, dtype=np.int64).reshape(-1)
    if np.any((values < 0) | (values >= k)):
        raise ValueError("code_ids must be in [0, k)")
    counts = np.bincount(values, minlength=k)
    total = np.sum(counts)
    if total <= 0:
        return np.zeros(k, dtype=np.float64)
    return counts.astype(np.float64) / float(total)


def compute_normalized_perplexity(p: np.ndarray) -> float:
    """计算归一化 code perplexity。

    输入参数:
        p: code occupancy 概率分布。

    输出:
        ``exp(entropy) / K``；分布为空或无有效 code 时返回 NaN。

    使用场景:
        诊断 code 分布是否塌缩或过于接近随机均匀。
    """

    probabilities = np.asarray(p, dtype=np.float64)
    if probabilities.size == 0:
        return _nan()
    if float(np.sum(probabilities)) <= 0.0:
        return _nan()
    entropy = -float(np.sum(probabilities * np.log(probabilities + _EPS)))
    return float(np.exp(entropy) / probabilities.size)


def _compute_code_prototypes(
    embeddings: np.ndarray,
    code_ids: np.ndarray,
    k: int,
) -> np.ndarray | None:
    """按 code 聚合 latent/code embedding prototype。"""

    values = np.asarray(embeddings, dtype=np.float64)
    assignments = np.asarray(code_ids, dtype=np.int64).reshape(-1)
    if k <= 0 or values.ndim != 2 or values.shape[0] != assignments.shape[0]:
        return None
    prototypes = np.full((k, values.shape[1]), np.nan, dtype=np.float64)
    for code_id in range(k):
        mask = assignments == code_id
        if np.any(mask):
            prototypes[code_id] = np.mean(values[mask], axis=0)
    return prototypes


def _compute_action_prototypes(
    decoded_actions: np.ndarray,
    code_ids: np.ndarray,
    k: int,
) -> np.ndarray | None:
    """按 code 聚合 decoded position path prototype。"""

    actions = np.asarray(decoded_actions, dtype=np.float64)
    assignments = np.asarray(code_ids, dtype=np.int64).reshape(-1)
    if k <= 0 or actions.ndim != 2 or actions.shape[0] != assignments.shape[0]:
        return None
    positions = actions - 1.0
    prototypes = np.full((k, positions.shape[1]), np.nan, dtype=np.float64)
    for code_id in range(k):
        mask = assignments == code_id
        if np.any(mask):
            prototypes[code_id] = np.mean(positions[mask], axis=0)
    return prototypes


def _current_assignment_snapshot(
    snapshot: Phase1EvaluationSnapshot,
    probabilities: np.ndarray,
    runtime_config: Phase1ValidationRuntimeConfig,
) -> CodeAssignmentSnapshot:
    """构造当前 epoch 的 assignment snapshot。

    输入参数:
        snapshot: 当前 validation snapshot。
        probabilities: 当前 code occupancy 分布。
        runtime_config: 提供 active code 最小占用比例。

    输出:
        ``CodeAssignmentSnapshot``，包含当前 sample-code 对齐关系和 active codes。

    使用场景:
        assignment churn 和 code lifetime 计算，也可写入 extra payload 给后续保存。
    """

    active_codes = tuple(
        int(code_id)
        for code_id, occupancy in enumerate(probabilities)
        if occupancy >= runtime_config.active_code_min_occupancy
    )
    return CodeAssignmentSnapshot(
        epoch=snapshot.epoch,
        split=snapshot.split,
        sample_ids=np.asarray(snapshot.sample_ids),
        code_ids=np.asarray(snapshot.code_ids),
        active_codes=active_codes,
        code_prototypes=_compute_code_prototypes(
            snapshot.z_q,
            snapshot.code_ids,
            probabilities.size,
        ),
        action_prototypes=_compute_action_prototypes(
            snapshot.decoded_actions,
            snapshot.code_ids,
            probabilities.size,
        ),
    )


def _sample_code_map(snapshot: CodeAssignmentSnapshot) -> dict[object, int]:
    """按 sample id 构建 code assignment 映射。"""

    return {
        sample_id.item() if hasattr(sample_id, "item") else sample_id: int(code_id)
        for sample_id, code_id in zip(snapshot.sample_ids, snapshot.code_ids, strict=False)
    }


def _churn_between(
    left: CodeAssignmentSnapshot,
    right: CodeAssignmentSnapshot,
    *,
    prototype_kind: str = "auto",
) -> float:
    """计算两个 snapshot 之间同一样本的 assignment 改变比例。"""

    right_map = _sample_code_map(right)
    alignment = align_previous_codes_to_current(
        left,
        right,
        prototype_kind=prototype_kind,
    )
    changed = 0
    total = 0
    for sample_id, left_code in zip(left.sample_ids, left.code_ids, strict=False):
        key = sample_id.item() if hasattr(sample_id, "item") else sample_id
        if key not in right_map:
            continue
        total += 1
        aligned_left_code = alignment.get(int(left_code), int(left_code))
        changed += int(aligned_left_code != right_map[key])
    return changed / total if total > 0 else _nan()


def _assignment_churn_by_epoch(
    current: CodeAssignmentSnapshot,
    history: Sequence[CodeAssignmentSnapshot],
    window: int,
    *,
    prototype_kind: str = "auto",
) -> dict[int, float]:
    """计算最近窗口内每个相邻 epoch pair 的 churn rate。"""

    if window <= 0:
        return {}
    recent = sorted(
        [
            item
            for item in history
            if item.split == current.split and item.epoch < current.epoch
        ],
        key=lambda item: item.epoch,
    )[-window:]
    if len(recent) < window:
        return {}
    snapshots = [*recent, current]
    churn_by_epoch = {}
    for left, right in zip(snapshots[:-1], snapshots[1:], strict=False):
        value = _churn_between(left, right, prototype_kind=prototype_kind)
        if not np.isfinite(value):
            return {}
        churn_by_epoch[int(right.epoch)] = value
    return churn_by_epoch


def compute_assignment_churn(
    current: CodeAssignmentSnapshot,
    history: Sequence[CodeAssignmentSnapshot],
    window: int,
    *,
    prototype_kind: str = "auto",
) -> float:
    """计算近期 assignment churn 均值。

    输入参数:
        current: 当前 epoch 的 assignment snapshot。
        history: 历史 assignment snapshots。
        window: 使用最近多少个历史 epoch。

    输出:
        同一样本 code label 改变比例的近期均值；history 不足时返回 NaN。

    使用场景:
        检查 code 语义是否仍在频繁重排。
    """

    churn_values = list(
        _assignment_churn_by_epoch(
            current,
            history,
            window,
            prototype_kind=prototype_kind,
        ).values()
    )
    return float(np.mean(churn_values)) if churn_values else _nan()


def compute_code_lifetime_pass_ratio(
    current_active_codes: Sequence[int],
    history: Sequence[CodeAssignmentSnapshot],
    min_lifetime_epochs: int,
    *,
    current_epoch: int | None = None,
    split: str | None = None,
    current_assignment: CodeAssignmentSnapshot | None = None,
    prototype_kind: str = "auto",
) -> float:
    """计算当前 active code 的 lifetime 达标比例。

    输入参数:
        current_active_codes: 当前 epoch 的 active code 列表。
        history: 已按同一 split 过滤好的历史 assignment snapshots，不包含当前 epoch。
        min_lifetime_epochs: 连续 active 的最小 epoch 数。

    输出:
        lifetime 达标的 active code 比例；没有 active code 时返回 NaN。

    使用场景:
        诊断 active code 是否只是短暂出现，还是已经形成稳定 code。
    """

    if len(set(current_active_codes)) != len(current_active_codes):
        raise ValueError("current_active_codes must be unique")
    if not current_active_codes:
        return _nan()
    if min_lifetime_epochs <= 0:
        raise ValueError("min_lifetime_epochs must be positive")
    ordered_history = sorted(history, key=lambda item: item.epoch)
    history_splits = {item.split for item in ordered_history}
    if split is not None and any(item.split != split for item in ordered_history):
        raise ValueError("history split must match current assignment split")
    if split is None and len(history_splits) > 1:
        raise ValueError("history must contain snapshots from a single split")
    if len({item.epoch for item in ordered_history}) != len(ordered_history):
        raise ValueError("history must not contain duplicate epochs")
    if current_epoch is not None and any(
        item.epoch >= current_epoch for item in ordered_history
    ):
        raise ValueError("history must contain only epochs before current_epoch")

    pass_count = 0
    for code_id in current_active_codes:
        lifetime = 1
        for previous in reversed(ordered_history):
            previous_active_codes = (
                active_codes_aligned_to_current(
                    previous,
                    current_assignment,
                    prototype_kind=prototype_kind,
                )
                if current_assignment is not None
                else set(previous.active_codes)
            )
            if code_id not in previous_active_codes:
                break
            lifetime += 1
        pass_count += int(lifetime >= min_lifetime_epochs)
    return pass_count / len(current_active_codes)


def _previous_assignments(
    current: CodeAssignmentSnapshot,
    history: Sequence[CodeAssignmentSnapshot],
) -> list[CodeAssignmentSnapshot]:
    """返回与当前 snapshot 同 split 且早于当前 epoch 的历史 assignment。"""

    return sorted(
        [
            item
            for item in history
            if item.split == current.split and item.epoch < current.epoch
        ],
        key=lambda item: item.epoch,
    )


def compute_nearest_second_margin(distances: np.ndarray) -> np.ndarray:
    """计算每个样本最近 code 与第二近 code 距离 margin。

    输入参数:
        distances: 每个样本到全部 codebook vectors 的距离，形状为 ``[N, K]``。

    输出:
        ``[N]`` 形状的 ``(d2 - d1) / (d1 + eps)``；K < 2 时返回全 NaN。

    使用场景:
        判断 quantizer assignment 边界是否清晰。
    """

    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 2:
        return np.asarray([], dtype=np.float64)
    if values.shape[1] < 2:
        return np.full(values.shape[0], _nan(), dtype=np.float64)
    sorted_distances = np.sort(values, axis=1)
    return (sorted_distances[:, 1] - sorted_distances[:, 0]) / (
        sorted_distances[:, 0] + _EPS
    )


def _nearest_second_margin_median(distances: np.ndarray) -> float:
    """计算最近 code 与第二近 code 距离 margin 的中位数。"""

    margins = compute_nearest_second_margin(distances)
    if margins.size == 0 or not np.any(np.isfinite(margins)):
        return _nan()
    return float(np.nanmedian(margins))


def _positions(actions: np.ndarray) -> np.ndarray:
    """将动作 id 映射为持仓值。

    输入参数:
        actions: 动作数组，约定 ``0=short``、``1=flat``、``2=long``。

    输出:
        同形状持仓数组，取值为 ``-1/0/1``。

    使用场景:
        turnover、entry timing 和主方向计算。
    """

    return np.asarray(actions, dtype=np.float64) - 1.0


def _turnover(actions: np.ndarray) -> np.ndarray:
    """计算每条 action sequence 的换手次数。

    输入参数:
        actions: 动作数组，形状为 ``[N, H]``。

    输出:
        ``[N]`` 形状的 turnover 数组。

    使用场景:
        计算 decoded turnover 与 DP demo turnover 的平均差异。
    """

    positions = _positions(actions)
    path = np.concatenate(
        [np.zeros((positions.shape[0], 1), dtype=np.float64), positions],
        axis=1,
    )
    return np.sum(np.abs(np.diff(path, axis=1)), axis=1)


def compute_first_trade_t(actions: np.ndarray) -> np.ndarray:
    """计算每条 action sequence 的首次非 flat 入场位置。

    输入参数:
        actions: 动作数组，形状为 ``[N, H]``。

    输出:
        ``[N]`` 形状的首次入场 index；全程 flat 的样本返回 -1。

    使用场景:
        计算 decoded entry timing 相对 DP demo 的偏移比例。
    """

    positions = _positions(actions)
    traded = positions != 0
    first = np.argmax(traded, axis=1)
    first[~np.any(traded, axis=1)] = -1
    return first


def classify_main_direction(actions: np.ndarray) -> np.ndarray:
    """按整条 action sequence 归纳主交易方向。

    输入参数:
        actions: 动作数组，形状为 ``[N, H]``。

    输出:
        ``[N]`` 形状的方向标签，取值包括 ``long``、``short``、``flat``、
        ``mixed``。

    使用场景:
        计算 decoded 主方向与 DP demo 主方向的一致率。
    """

    positions = _positions(actions)
    labels = np.full(positions.shape[0], "flat", dtype=object)
    long_count = np.sum(positions > 0, axis=1)
    short_count = np.sum(positions < 0, axis=1)
    labels[long_count > short_count] = "long"
    labels[short_count > long_count] = "short"
    mixed = (long_count == short_count) & (long_count > 0)
    labels[mixed] = "mixed"
    return labels


def _quantization_distance(snapshot: Phase1EvaluationSnapshot) -> float:
    """计算 snapshot 的 mean(||z_e - z_q||_2)。"""

    z_e = np.asarray(snapshot.z_e, dtype=np.float64)
    z_q = np.asarray(snapshot.z_q, dtype=np.float64)
    return (
        float(np.mean(np.linalg.norm(z_e - z_q, axis=-1)))
        if z_e.shape == z_q.shape and z_e.size
        else _nan()
    )


def compute_vq_internal_metrics(
    *,
    train_snapshot: Phase1EvaluationSnapshot,
    val_snapshot: Phase1EvaluationSnapshot,
    assignment_history: Sequence[CodeAssignmentSnapshot],
    runtime_config: Phase1ValidationRuntimeConfig,
) -> Phase1LayerComputation:
    """计算 Layer 1 VQ 内部质量 raw metrics。

    功能说明:
        基于 train/validation snapshot 和 assignment history 计算 codebook 是否
        稳定、未塌缩、边界清晰且 decoder 保留 DP action 信息。

    输入参数:
        train_snapshot: 训练集 snapshot，用于 reconstruction loss gap。
        val_snapshot: 验证集 snapshot，用于重构、code 分布和 latent 诊断。
        assignment_history: 历史 code assignment，用于 churn 和 lifetime 诊断。
        runtime_config: validation 运行参数，提供 active/dead code 和 churn window 配置。

    输出:
        ``Phase1LayerComputation``，其中 ``metrics`` 为
        ``Phase1VQInternalMetrics``，``extra_payload`` 包含当前 assignment snapshot
        和辅助诊断元数据。

    使用场景:
        full checkpoint validation 的第一层 VQ health raw metric 计算；结果交给
        ``evaluate_vq_internal_rules()`` 判定 hard gate。
    """

    code_ids = np.asarray(val_snapshot.code_ids, dtype=np.int64)
    num_codes = _num_codes(val_snapshot, runtime_config)
    probabilities = compute_code_distribution(code_ids, num_codes)
    has_code_samples = probabilities.size > 0 and code_ids.size > 0
    current_assignment = _current_assignment_snapshot(
        val_snapshot,
        probabilities,
        runtime_config,
    )
    assignment_churn_by_epoch = _assignment_churn_by_epoch(
        current_assignment,
        assignment_history,
        runtime_config.churn_window_epochs,
        prototype_kind=runtime_config.code_alignment_prototype,
    )
    payload = Phase1VQInternalPayload(
        code_distribution=tuple(float(value) for value in probabilities),
        active_codes=tuple(
            int(code_id) for code_id in current_assignment.active_codes
        ),
        current_assignment=current_assignment,
        assignment_churn_by_epoch=assignment_churn_by_epoch,
        codebook_size=num_codes,
        codebook_size_available=num_codes > 0,
        code_distribution_sample_count=int(code_ids.size),
    )

    action_accuracy = compute_action_accuracy(
        val_snapshot.demo_actions,
        val_snapshot.decoded_actions,
    )
    train_loss = float(train_snapshot.reconstruction_loss)
    val_loss = float(val_snapshot.reconstruction_loss)

    train_quantization_distance = _quantization_distance(train_snapshot)
    quantization_distance = _quantization_distance(val_snapshot)

    demo_turnover = _turnover(val_snapshot.demo_actions)
    decoded_turnover = _turnover(val_snapshot.decoded_actions)
    turnover_error = (
        float(np.mean(np.abs(decoded_turnover - demo_turnover)))
        if demo_turnover.size
        else _nan()
    )

    demo_entry = compute_first_trade_t(val_snapshot.demo_actions)
    decoded_entry = compute_first_trade_t(val_snapshot.decoded_actions)
    both_entered = (demo_entry >= 0) & (decoded_entry >= 0)
    entry_error = (
        float(np.median(np.abs(decoded_entry[both_entered] - demo_entry[both_entered])))
        if np.any(both_entered)
        else _nan()
    )
    demo_direction = classify_main_direction(val_snapshot.demo_actions)
    decoded_direction = classify_main_direction(val_snapshot.decoded_actions)
    direction_accuracy = (
        float(np.mean(demo_direction == decoded_direction))
        if demo_direction.size
        else _nan()
    )

    metrics = Phase1VQInternalMetrics(
        validation_action_accuracy=action_accuracy,
        reconstruction_loss_gap=float(val_loss / (train_loss + _EPS)),
        active_code_ratio=float(
            np.mean(probabilities >= runtime_config.active_code_min_occupancy)
        )
        if has_code_samples
        else _nan(),
        max_code_occupancy=float(np.max(probabilities)) if has_code_samples else _nan(),
        normalized_code_perplexity=compute_normalized_perplexity(probabilities)
        if has_code_samples
        else _nan(),
        dead_code_ratio=float(
            np.mean(probabilities < runtime_config.dead_code_max_occupancy)
        )
        if has_code_samples
        else _nan(),
        assignment_churn_recent_mean=compute_assignment_churn(
            current_assignment,
            assignment_history,
            runtime_config.churn_window_epochs,
            prototype_kind=runtime_config.code_alignment_prototype,
        ),
        code_lifetime_pass_ratio=compute_code_lifetime_pass_ratio(
            current_assignment.active_codes,
            _previous_assignments(current_assignment, assignment_history),
            5,
            current_epoch=current_assignment.epoch,
            split=current_assignment.split,
            current_assignment=current_assignment,
            prototype_kind=runtime_config.code_alignment_prototype,
        ),
        quantization_distance=quantization_distance,
        nearest_second_margin_median=_nearest_second_margin_median(
            val_snapshot.distances,
        ),
        decoder_turnover_error=turnover_error,
        entry_timing_error_median=entry_error,
        direction_accuracy=direction_accuracy,
        quantization_distance_gap=float(
            quantization_distance / (train_quantization_distance + _EPS)
        )
        if np.isfinite(quantization_distance)
        and np.isfinite(train_quantization_distance)
        else _nan(),
    )
    return Phase1LayerComputation(
        layer_id=1,
        layer_name="vq_internal",
        metrics=metrics,
        vq_internal_payload=payload,
    )


__all__ = [
    "classify_main_direction",
    "compute_action_accuracy",
    "compute_assignment_churn",
    "compute_code_distribution",
    "compute_code_lifetime_pass_ratio",
    "compute_first_trade_t",
    "compute_nearest_second_margin",
    "compute_normalized_perplexity",
    "compute_vq_internal_metrics",
]
