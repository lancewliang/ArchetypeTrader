"""Phase II validation Layer 3: demonstration consistency raw metrics."""

from __future__ import annotations

import numpy as np

from ...metrics import (
    Phase2DemonstrationConsistencyMetrics,
    Phase2DemonstrationConsistencyPayload,
    Phase2LayerComputation,
)
from ._numeric import as_float_array, nan_value, safe_mean


def compute_demonstration_consistency_metrics(
    payload: Phase2DemonstrationConsistencyPayload,
    *,
    cross_entropy_to_assigned: float = float("nan"),
    kl_to_assigned_onehot: float = float("nan"),
) -> Phase2LayerComputation:
    """Compute selector consistency with Phase I assigned labels.

    算法:
        1. 将 selected code 与 Phase I assigned label 对齐到相同长度；
        2. 计算 label match rate，得到 selector 复用 demonstration label 的比例；
        3. 构造 deviation mask：``selected_code != assigned_label``；
        4. 仅在 deviation 样本上比较 selector return 与 assigned-label return；
        5. 用 selected Q 与 assigned-label Q 的差计算 Q margin；
        6. cross entropy/KL 由上游 evaluator 根据 Q softmax 计算后传入，本函数
           只负责装入统一 metrics payload。

    核心公式:
        - ``label_match_rate = mean(1[selected_i == assigned_i])``
        - ``deviation_mask_i = selected_i != assigned_i``
        - ``profitable_deviation_rate = mean(1[selector_return_i > assigned_return_i])``
          over deviated finite pairs
        - ``unprofitable_deviation_rate = mean(1[selector_return_i < assigned_return_i])``
          over deviated finite pairs
        - ``deviation_return_delta = mean(selector_return_i - assigned_return_i)``
          over deviated finite pairs
        - ``label_q_margin = mean(selected_q_i - assigned_q_i)``

    说明:
        Layer 3 不要求 selector 完全复制 Phase I label；它检查偏离是否有收益和
        Q-value 支持。match rate 是区间约束，偏离收益差和 Q margin 越大越好。
    """

    selected = np.asarray(payload.selected_code_ids, dtype=np.int64)
    assigned = np.asarray(payload.assigned_code_labels, dtype=np.int64)
    selector_returns = as_float_array(payload.selector_returns)
    assigned_returns = as_float_array(payload.assigned_label_returns)
    selected_q = as_float_array(payload.selected_q_values)
    assigned_q = as_float_array(payload.assigned_label_q_values)

    size = min(selected.size, assigned.size)
    if size <= 0:
        label_match_rate = nan_value()
        deviation_mask = np.asarray([], dtype=np.bool_)
    else:
        selected = selected[:size]
        assigned = assigned[:size]
        # label_match_rate 衡量 imitation 先验保留度；deviation_mask 是后续只在
        # “真正偏离 assigned label”的样本上计算收益差的筛选条件。
        label_match_rate = float(np.mean((selected == assigned).astype(np.float64)))
        deviation_mask = selected != assigned

    deviation_delta = _paired_delta(selector_returns, assigned_returns, deviation_mask)
    profitable_deviation_rate = _deviation_rate(
        selector_returns,
        assigned_returns,
        deviation_mask,
        greater=True,
    )
    unprofitable_deviation_rate = _deviation_rate(
        selector_returns,
        assigned_returns,
        deviation_mask,
        greater=False,
    )
    metrics = Phase2DemonstrationConsistencyMetrics(
        label_match_rate=label_match_rate,
        cross_entropy_to_assigned=float(cross_entropy_to_assigned),
        kl_to_assigned_onehot=float(kl_to_assigned_onehot),
        label_q_margin=_q_margin_mean(selected_q, assigned_q),
        profitable_deviation_rate=profitable_deviation_rate,
        unprofitable_deviation_rate=unprofitable_deviation_rate,
        deviation_return_delta=deviation_delta,
    )
    return Phase2LayerComputation(
        layer_id=3,
        layer_name="demonstration_consistency",
        metrics=metrics,
        extra_payload={"demonstration_consistency_payload": payload},
    )


def _paired_delta(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean left-right on masked finite pairs.

    公式:
        ``mean(left_i - right_i)``，其中 ``mask_i`` 为 True 且两侧都是有限值。

    用途:
        Layer 3 用它计算偏离 assigned label 时 selector 相对 assigned baseline
        的平均收益差。
    """

    size = min(left.shape[0], right.shape[0], mask.shape[0])
    if size <= 0:
        return nan_value()
    valid = mask[:size] & np.isfinite(left[:size]) & np.isfinite(right[:size])
    if not np.any(valid):
        return nan_value()
    return float(np.mean(left[:size][valid] - right[:size][valid]))


def _deviation_rate(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
    *,
    greater: bool,
) -> float:
    """Rate among deviated finite pairs.

    公式:
        - profitable: ``mean(1[left_i > right_i])``
        - unprofitable: ``mean(1[left_i < right_i])``
        只统计 ``mask_i`` 为 True 且两侧有限的 pair。

    用途:
        区分“有收益证明的偏离”和“危险偏离”。前者越高越好，后者越低越好。
    """

    size = min(left.shape[0], right.shape[0], mask.shape[0])
    if size <= 0:
        return nan_value()
    valid = mask[:size] & np.isfinite(left[:size]) & np.isfinite(right[:size])
    if not np.any(valid):
        return nan_value()
    comparison = left[:size][valid] > right[:size][valid]
    if not greater:
        comparison = left[:size][valid] < right[:size][valid]
    return float(np.mean(comparison.astype(np.float64)))


def _q_margin_mean(selected_q: np.ndarray, assigned_q: np.ndarray) -> float:
    """Mean selected-assigned Q margin over finite paired values.

    公式:
        ``label_q_margin = mean(selected_q_i - assigned_q_i)``。

    用途:
        当 selector 不选 assigned label 时，正 margin 表示 Q-network 认为 selected
        code 更有价值；margin 越大，偏离越有模型内证据。
    """

    size = min(selected_q.shape[0], assigned_q.shape[0])
    if size <= 0:
        return nan_value()
    return safe_mean(selected_q[:size] - assigned_q[:size])


__all__ = ["compute_demonstration_consistency_metrics"]
