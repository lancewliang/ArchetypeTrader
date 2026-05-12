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
    Phase1VQInternalMetrics,
    Phase1ValidationRuntimeConfig,
)


_EPS = 1e-12


def _num_codes(snapshot: Phase1EvaluationSnapshot) -> int:
    """推断当前 snapshot 的 codebook size。

    输入参数:
        snapshot: validation snapshot，优先读取 ``distances.shape[1]``，其次从
            ``code_ids`` 最大值推断。

    输出:
        code 数量；无法推断时返回 0。

    使用场景:
        计算 code occupancy、active ratio、dead ratio 和 normalized perplexity。
    """

    distances = np.asarray(snapshot.distances)
    if distances.ndim == 2 and distances.shape[1] > 0:
        return int(distances.shape[1])
    code_ids = np.asarray(snapshot.code_ids, dtype=np.int64)
    return int(np.max(code_ids) + 1) if code_ids.size else 0


def _code_distribution(code_ids: np.ndarray, num_codes: int) -> np.ndarray:
    """计算 code occupancy 分布。

    输入参数:
        code_ids: 每个样本的 assigned code id。
        num_codes: codebook size。

    输出:
        ``[K]`` 形状的 occupancy 概率分布；无样本时返回全 0。

    使用场景:
        active code ratio、max occupancy、dead code ratio 和 perplexity 计算。
    """

    if num_codes <= 0:
        return np.asarray([], dtype=np.float64)
    counts = np.bincount(np.asarray(code_ids, dtype=np.int64), minlength=num_codes)
    total = np.sum(counts)
    if total <= 0:
        return np.zeros(num_codes, dtype=np.float64)
    return counts.astype(np.float64) / float(total)


def _normalized_perplexity(probabilities: np.ndarray) -> float:
    """计算归一化 code perplexity。

    输入参数:
        probabilities: code occupancy 概率分布。

    输出:
        ``exp(entropy) / K``；分布为空或无有效 code 时返回 NaN。

    使用场景:
        诊断 code 分布是否塌缩或过于接近随机均匀。
    """

    if probabilities.size == 0:
        return _nan()
    positive = probabilities[probabilities > 0]
    if positive.size == 0:
        return _nan()
    entropy = -float(np.sum(positive * np.log(positive + _EPS)))
    return float(np.exp(entropy) / probabilities.size)


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
    )


def _assignment_churn(
    current: CodeAssignmentSnapshot,
    history: Sequence[CodeAssignmentSnapshot],
    *,
    window: int,
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

    recent = [
        item
        for item in history
        if item.split == current.split and item.epoch < current.epoch
    ][-max(1, window) :]
    if not recent:
        return _nan()

    current_map = {
        sample_id.item() if hasattr(sample_id, "item") else sample_id: code_id
        for sample_id, code_id in zip(current.sample_ids, current.code_ids, strict=False)
    }
    churn_values: list[float] = []
    for previous in recent:
        changed = 0
        total = 0
        for sample_id, previous_code in zip(
            previous.sample_ids,
            previous.code_ids,
            strict=False,
        ):
            key = sample_id.item() if hasattr(sample_id, "item") else sample_id
            if key not in current_map:
                continue
            total += 1
            changed += int(int(current_map[key]) != int(previous_code))
        if total > 0:
            churn_values.append(changed / total)
    return float(np.mean(churn_values)) if churn_values else _nan()


def _code_lifetime_pass_ratio(
    current: CodeAssignmentSnapshot,
    history: Sequence[CodeAssignmentSnapshot],
    *,
    required_epochs: int = 10,
) -> float:
    """计算当前 active code 的 lifetime 达标比例。

    输入参数:
        current: 当前 epoch 的 assignment snapshot。
        history: 历史 assignment snapshots。
        required_epochs: 连续 active 的最小 epoch 数。

    输出:
        lifetime 达标的 active code 比例；没有 active code 时返回 NaN。

    使用场景:
        诊断 active code 是否只是短暂出现，还是已经形成稳定 code。
    """

    if not current.active_codes:
        return _nan()
    same_split = [
        item
        for item in history
        if item.split == current.split and item.epoch < current.epoch
    ]
    pass_count = 0
    for code_id in current.active_codes:
        lifetime = 1
        for previous in reversed(same_split):
            if code_id not in previous.active_codes:
                break
            lifetime += 1
        pass_count += int(lifetime >= required_epochs)
    return pass_count / len(current.active_codes)


def _nearest_second_margin_median(distances: np.ndarray) -> float:
    """计算最近 code 与第二近 code 距离 margin 的中位数。

    输入参数:
        distances: 每个样本到全部 codebook vectors 的距离，形状为 ``[N, K]``。

    输出:
        ``median((d2 - d1) / abs(d1))``；距离矩阵缺失或 K < 2 时返回 NaN。

    使用场景:
        判断 quantizer assignment 边界是否清晰。
    """

    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        return _nan()
    sorted_distances = np.sort(values, axis=1)
    margins = (sorted_distances[:, 1] - sorted_distances[:, 0]) / (
        np.abs(sorted_distances[:, 0]) + _EPS
    )
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


def _first_trade_index(actions: np.ndarray) -> np.ndarray:
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


def _main_direction(actions: np.ndarray) -> np.ndarray:
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
    net = np.sum(positions, axis=1)
    labels = np.full(positions.shape[0], "flat", dtype=object)
    labels[net > 0] = "long"
    labels[net < 0] = "short"
    mixed = (np.any(positions > 0, axis=1)) & (np.any(positions < 0, axis=1))
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
        ``Phase1VQInternalMetrics``，``extra_payload`` 包含当前 assignment snapshot。

    使用场景:
        full checkpoint validation 的第一层 VQ health raw metric 计算；结果交给
        ``evaluate_vq_internal_rules()`` 判定 hard gate。
    """

    code_ids = np.asarray(val_snapshot.code_ids, dtype=np.int64)
    num_codes = _num_codes(val_snapshot)
    probabilities = _code_distribution(code_ids, num_codes)
    current_assignment = _current_assignment_snapshot(
        val_snapshot,
        probabilities,
        runtime_config,
    )

    action_accuracy = float(
        np.mean(
            np.asarray(val_snapshot.decoded_actions)
            == np.asarray(val_snapshot.demo_actions)
        )
    )
    train_loss = float(train_snapshot.reconstruction_loss)
    val_loss = float(val_snapshot.reconstruction_loss)

    train_quantization_distance = _quantization_distance(train_snapshot)
    quantization_distance = _quantization_distance(val_snapshot)

    demo_turnover = _turnover(val_snapshot.demo_actions)
    decoded_turnover = _turnover(val_snapshot.decoded_actions)
    horizon = max(1, np.asarray(val_snapshot.demo_actions).shape[1])

    demo_entry = _first_trade_index(val_snapshot.demo_actions)
    decoded_entry = _first_trade_index(val_snapshot.decoded_actions)
    both_entered = (demo_entry >= 0) & (decoded_entry >= 0)
    entry_error = (
        float(np.median(np.abs(decoded_entry[both_entered] - demo_entry[both_entered])) / horizon)
        if np.any(both_entered)
        else _nan()
    )

    metrics = Phase1VQInternalMetrics(
        validation_action_accuracy=action_accuracy,
        reconstruction_loss_gap=float(val_loss / (train_loss + _EPS)),
        active_code_ratio=float(
            np.mean(probabilities >= runtime_config.active_code_min_occupancy)
        )
        if probabilities.size
        else _nan(),
        max_code_occupancy=float(np.max(probabilities)) if probabilities.size else _nan(),
        normalized_code_perplexity=_normalized_perplexity(probabilities),
        dead_code_ratio=float(
            np.mean(probabilities < runtime_config.dead_code_max_occupancy)
        )
        if probabilities.size
        else _nan(),
        assignment_churn_recent_mean=_assignment_churn(
            current_assignment,
            assignment_history,
            window=runtime_config.churn_window_epochs,
        ),
        code_lifetime_pass_ratio=_code_lifetime_pass_ratio(
            current_assignment,
            assignment_history,
        ),
        quantization_distance=quantization_distance,
        nearest_second_margin_median=_nearest_second_margin_median(
            val_snapshot.distances,
        ),
        decoder_turnover_error=float(
            np.mean(np.abs(decoded_turnover - demo_turnover)) / horizon
        ),
        entry_timing_error_median=entry_error,
        direction_accuracy=float(
            np.mean(
                _main_direction(val_snapshot.demo_actions)
                == _main_direction(val_snapshot.decoded_actions)
            )
        ),
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
        extra_payload={"current_assignment": current_assignment},
    )


__all__ = ["compute_vq_internal_metrics"]
