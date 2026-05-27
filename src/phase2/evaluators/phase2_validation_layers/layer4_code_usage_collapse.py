"""Phase II validation Layer 4: code usage and collapse raw metrics."""

from __future__ import annotations

import math

import numpy as np

from ...metrics import (
    Phase2CodeUsageCollapseMetrics,
    Phase2CodeUsageCollapsePayload,
    Phase2Layer4CodeUsageCollapseComputation,
    Phase2LayerComputation,
    Phase2PerCodeUsageDiagnostic,
)
from src.utils._numeric import nan_value


def compute_code_usage_collapse_metrics(
    payload: Phase2CodeUsageCollapsePayload,
    *,
    num_archetypes: int,
    train_label_distribution: tuple[float, ...] | None = None,
) -> Phase2LayerComputation:
    """Compute code usage entropy, collapse and distribution drift metrics.

    算法:
        1. 将 selector selected code 和 Phase I assigned label 分别转换为
           ``[0, K)`` 上的归一化分布；
        2. 对 selector 分布计算 Shannon entropy 和 perplexity；
        3. 统计 active code 数、最大/最小使用比例；
        4. 计算 selector 分布相对 train label 分布和 validation label 分布的 KL；
        5. 从 per-code diagnostics 中汇总 dead profitable code 数和 active code
           最小 support。

    核心公式:
        - ``p_k = count(selected_code == k) / N``
        - ``entropy = -Σ_k p_k log(p_k)``，忽略 ``p_k = 0`` 项
        - ``perplexity = exp(entropy)``
        - ``active_code_count = Σ_k 1[p_k > 0]``
        - ``max_code_usage_ratio = max_k p_k``
        - ``KL(p || q) = Σ_k p_k log(p_k / q_k)``，实现中加 eps 平滑

    说明:
        本层关注 selector 是否真正利用 archetype set。entropy/perplexity 和
        active count 过低表示 code collapse；max usage 过高表示单 code 支配。
    """

    selected = np.asarray(payload.selected_code_ids, dtype=np.int64)
    assigned = np.asarray(payload.assigned_code_labels, dtype=np.int64)
    selector_distribution = _distribution(selected, num_archetypes)
    val_label_distribution = _distribution(assigned, num_archetypes)
    train_distribution = (
        np.asarray(train_label_distribution, dtype=np.float64)
        if train_label_distribution is not None
        else np.asarray([], dtype=np.float64)
    )

    entropy = _entropy(selector_distribution)
    active_count = int(np.sum(selector_distribution > 0.0))
    # min_code_usage_ratio 只在被使用过的 code 上取最小值；完全未使用 code 已通过
    # active_count/perplexity 体现，不把 0 混入该诊断值。
    positive_ratios = selector_distribution[selector_distribution > 0.0]
    per_code_diagnostics = tuple(payload.per_code_diagnostics)

    metrics = Phase2CodeUsageCollapseMetrics(
        selected_code_entropy=entropy,
        selected_code_perplexity=(
            float(math.exp(entropy)) if math.isfinite(entropy) else nan_value()
        ),
        active_code_count=active_count,
        max_code_usage_ratio=(
            float(np.max(selector_distribution))
            if selector_distribution.size > 0
            else nan_value()
        ),
        min_code_usage_ratio=(
            float(np.min(positive_ratios)) if positive_ratios.size > 0 else nan_value()
        ),
        usage_kl_to_train_label_distribution=_kl(
            selector_distribution,
            train_distribution,
        ),
        usage_kl_to_val_label_distribution=_kl(
            selector_distribution,
            val_label_distribution,
        ),
        dead_profitable_code_count=sum(
            1 for item in per_code_diagnostics if item.is_dead_profitable
        ),
        min_per_code_sample_count=_min_per_code_sample_count(per_code_diagnostics),
    )
    return Phase2Layer4CodeUsageCollapseComputation(
        metrics=metrics,
        code_usage_collapse_payload=payload,
        per_code_diagnostics=per_code_diagnostics,
    )


def build_per_code_usage_diagnostics(
    *,
    selected_code_ids: np.ndarray,
    assigned_code_labels: np.ndarray,
    selector_returns: np.ndarray,
    kl_returns: np.ndarray,
    num_archetypes: int,
    active_ratio_min: float = 0.01,
    profitable_return_min: float = 0.0,
) -> tuple[Phase2PerCodeUsageDiagnostic, ...]:
    """Build per-code usage rows used by Layer 4 and report cards.

    算法:
        1. 分别计算 selector selected code 分布和 assigned label 分布；
        2. 对每个 code k 构造两个 mask：
           ``selected_mask = selected_code == k``，
           ``kl_mask = assigned_label == k``；
        3. 在两个 mask 下分别计算 selector/assigned mean return 和 support；
        4. 用 selector usage ratio 判断该 code 是否 active；
        5. 如果 assigned baseline 中该 code 盈利但 selector 不 active，则标记为
           dead profitable code。

    核心公式:
        - ``selector_ratio_k = count(selected == k) / N_selected``
        - ``kl_ratio_k = count(assigned == k) / N_assigned``
        - ``selector_mean_return_k = mean(selector_return_i | selected_i = k)``
        - ``kl_mean_return_k = mean(kl_return_i | assigned_i = k)``
        - ``uplift_vs_kl_k = selector_mean_return_k - kl_mean_return_k``

    用途:
        per-code 诊断用于报告中定位哪些 archetype 被充分使用、贡献收益，哪些
        原本盈利但被 selector 忽略。
    """

    selected_distribution = _distribution(selected_code_ids, num_archetypes)
    kl_distribution = _distribution(assigned_code_labels, num_archetypes)
    rows: list[Phase2PerCodeUsageDiagnostic] = []
    for code_id in range(int(num_archetypes)):
        selected_mask = selected_code_ids == code_id
        kl_mask = assigned_code_labels == code_id
        selector_mean = _masked_mean(selector_returns, selected_mask)
        kl_mean = _masked_mean(kl_returns, kl_mask)
        selector_count = int(np.sum(selected_mask))
        kl_count = int(np.sum(kl_mask))
        is_active = bool(selected_distribution[code_id] >= active_ratio_min)
        is_dead_profitable = bool(
            not is_active and math.isfinite(kl_mean) and kl_mean > profitable_return_min
        )
        rows.append(
            Phase2PerCodeUsageDiagnostic(
                code_id=code_id,
                selector_count=selector_count,
                selector_ratio=float(selected_distribution[code_id]),
                kl_count=kl_count,
                kl_ratio=float(kl_distribution[code_id]),
                selector_mean_return=selector_mean,
                kl_mean_return=kl_mean,
                uplift_vs_kl=(
                    selector_mean - kl_mean
                    if math.isfinite(selector_mean) and math.isfinite(kl_mean)
                    else nan_value()
                ),
                is_active=is_active,
                is_dead_profitable=is_dead_profitable,
            )
        )
    return tuple(rows)


def _distribution(values: np.ndarray, num_archetypes: int) -> np.ndarray:
    """Return normalized code distribution over [0, K).

    公式:
        ``p_k = count(values == k and 0 <= k < K) / total_valid_count``。

    防御:
        非法 code 会被过滤；没有合法 code 时返回全 0 分布，让上层指标可继续
        生成并由 Layer 0/Layer 4 rules 暴露问题。
    """

    if num_archetypes <= 0:
        return np.asarray([], dtype=np.float64)
    valid = values[(values >= 0) & (values < num_archetypes)]
    counts = np.bincount(valid, minlength=num_archetypes).astype(np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros(num_archetypes, dtype=np.float64)
    return counts / total


def _entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy over non-zero probabilities.

    公式:
        ``H(p) = -Σ_k p_k log(p_k)``，忽略 ``p_k = 0``，因为极限项为 0。

    用途:
        分布越集中，entropy 越低；单一 code collapse 时 entropy 接近 0。
    """

    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return nan_value()
    return float(-np.sum(positive * np.log(positive)))


def _kl(left: np.ndarray, right: np.ndarray) -> float:
    """KL(left || right) with small smoothing.

    公式:
        ``KL(p || q) = Σ_k p_k log(p_k / q_k)``。

    实现细节:
        给 p、q 加 ``eps=1e-12`` 后重新归一化，避免 q_k=0 时除零或 log inf。

    用途:
        衡量 selector 使用分布相对 train/validation label 先验的漂移；越小表示
        分布越接近，偏离较大时需要由收益 uplift 解释。
    """

    if left.size == 0 or right.size != left.size:
        return nan_value()
    eps = 1e-12
    p = left + eps
    q = right + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean over masked finite values.

    公式:
        ``mean(values_i | mask_i and isfinite(values_i))``。

    用途:
        per-code return 只在该 code 对应样本上聚合，且跳过 NaN/inf。
    """

    size = min(values.shape[0], mask.shape[0])
    if size <= 0:
        return nan_value()
    valid = mask[:size] & np.isfinite(values[:size])
    if not np.any(valid):
        return nan_value()
    return float(np.mean(values[:size][valid]))


def _min_per_code_sample_count(
    rows: tuple[Phase2PerCodeUsageDiagnostic, ...],
) -> int:
    """Minimum selected support among active codes.

    公式:
        ``min(selector_count_k | code k is active)``。

    用途:
        判断 per-code return 是否有足够样本支撑；没有 active code 时返回 0。
    """

    active_counts = [item.selector_count for item in rows if item.is_active]
    if not active_counts:
        return 0
    return int(min(active_counts))


__all__ = [
    "build_per_code_usage_diagnostics",
    "compute_code_usage_collapse_metrics",
]
