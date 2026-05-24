"""Phase II validation Layer 4: code usage and collapse raw metrics."""

from __future__ import annotations

import math

import numpy as np

from ...metrics import (
    Phase2CodeUsageCollapseMetrics,
    Phase2CodeUsageCollapsePayload,
    Phase2LayerComputation,
    Phase2PerCodeUsageDiagnostic,
)
from ._numeric import nan_value


def compute_code_usage_collapse_metrics(
    payload: Phase2CodeUsageCollapsePayload,
    *,
    num_archetypes: int,
    train_label_distribution: tuple[float, ...] | None = None,
) -> Phase2LayerComputation:
    """Compute code usage entropy, collapse and distribution drift metrics."""

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
    return Phase2LayerComputation(
        layer_id=4,
        layer_name="code_usage_collapse",
        metrics=metrics,
        extra_payload={
            "code_usage_collapse_payload": payload,
            "per_code_diagnostics": per_code_diagnostics,
        },
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
    """Build per-code usage rows used by Layer 4 and report cards."""

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
    """Return normalized code distribution over [0, K)."""

    if num_archetypes <= 0:
        return np.asarray([], dtype=np.float64)
    valid = values[(values >= 0) & (values < num_archetypes)]
    counts = np.bincount(valid, minlength=num_archetypes).astype(np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros(num_archetypes, dtype=np.float64)
    return counts / total


def _entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy over non-zero probabilities."""

    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return nan_value()
    return float(-np.sum(positive * np.log(positive)))


def _kl(left: np.ndarray, right: np.ndarray) -> float:
    """KL(left || right) with small smoothing."""

    if left.size == 0 or right.size != left.size:
        return nan_value()
    eps = 1e-12
    p = left + eps
    q = right + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean over masked finite values."""

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
    """Minimum selected support among active codes."""

    active_counts = [item.selector_count for item in rows if item.is_active]
    if not active_counts:
        return 0
    return int(min(active_counts))


__all__ = [
    "build_per_code_usage_diagnostics",
    "compute_code_usage_collapse_metrics",
]
