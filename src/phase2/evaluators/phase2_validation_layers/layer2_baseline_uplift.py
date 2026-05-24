"""Phase II validation Layer 2: baseline uplift raw metrics."""

from __future__ import annotations

import numpy as np

from ...metrics import (
    Phase2BaselineUpliftMetrics,
    Phase2BaselineUpliftPayload,
    Phase2LayerComputation,
)
from ._numeric import as_float_array, finite_values, nan_value, safe_mean, safe_ratio


def compute_baseline_uplift_metrics(
    payload: Phase2BaselineUpliftPayload,
) -> Phase2LayerComputation:
    """Compute selector uplift versus assigned-label, random and oracle baselines.

    算法:
        1. 将 selector、assigned-label、random、oracle 四条逐样本 return 序列
           标准化为 float64 数组；
        2. 对每条序列只在有限值上计算平均 return；
        3. 用 selector 平均收益减去 baseline 平均收益得到 uplift；
        4. 用样本级配对比较计算 beat rate；
        5. 计算相对 assigned uplift、oracle capture 和 oracle regret。

    核心公式:
        - ``uplift_vs_assigned = mean(selector) - mean(assigned)``
        - ``uplift_vs_random = mean(selector) - mean(random)``
        - ``relative_uplift_vs_assigned = uplift_vs_assigned / abs(mean(assigned))``
        - ``oracle_capture_ratio = mean(selector) / mean(oracle)``
        - ``regret_to_oracle = mean(oracle) - mean(selector)``
        - ``beat_rate(a, b) = mean(1[a_i > b_i])`` over finite paired samples

    说明:
        assigned-label baseline 表示只复用 Phase I label；random baseline 表示随机
        code selection；oracle baseline 使用未来信息，只是上界参考，不能作为部署
        策略。
    """

    selector = as_float_array(payload.selector_returns)
    assigned = as_float_array(payload.assigned_label_returns)
    random = as_float_array(payload.random_returns)
    oracle = as_float_array(payload.oracle_returns)

    # 所有 uplift 都以有限样本均值为基础；若某条 baseline 完全缺失，safe_mean
    # 返回 NaN，并让后续 rule/report 暴露该缺口。
    selector_mean = safe_mean(selector)
    assigned_mean = safe_mean(assigned)
    random_mean = safe_mean(random)
    oracle_mean = safe_mean(oracle)
    uplift_vs_assigned = selector_mean - assigned_mean
    uplift_vs_random = selector_mean - random_mean

    metrics = Phase2BaselineUpliftMetrics(
        assigned_mean_return=assigned_mean,
        random_mean_return=random_mean,
        oracle_mean_return=oracle_mean,
        uplift_vs_assigned=uplift_vs_assigned,
        uplift_vs_random=uplift_vs_random,
        relative_uplift_vs_assigned=safe_ratio(
            uplift_vs_assigned,
            abs(assigned_mean),
        ),
        oracle_capture_ratio=safe_ratio(selector_mean, oracle_mean),
        regret_to_oracle=oracle_mean - selector_mean,
        beat_assigned_rate=_paired_beat_rate(selector, assigned),
        beat_random_rate=_paired_beat_rate(selector, random),
    )
    return Phase2LayerComputation(
        layer_id=2,
        layer_name="baseline_uplift",
        metrics=metrics,
        extra_payload={"baseline_uplift_payload": payload},
    )


def _paired_beat_rate(left: np.ndarray, right: np.ndarray) -> float:
    """Return finite paired rate of left > right.

    算法:
        1. 两条序列按相同 horizon order 配对，长度取较短者；
        2. 过滤任一侧非有限的 pair；
        3. 计算 ``mean(1[left_i > right_i])``。

    用途:
        beat rate 比均值 uplift 更能说明收益提升是否分布广泛；例如均值为正但
        beat rate 很低，通常表示 uplift 依赖少数大盈利样本。
    """

    size = min(left.shape[0], right.shape[0])
    if size <= 0:
        return nan_value()
    left_values = left[:size]
    right_values = right[:size]
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    if not np.any(valid):
        return nan_value()
    return float(np.mean((left_values[valid] > right_values[valid]).astype(np.float64)))


__all__ = ["compute_baseline_uplift_metrics"]
