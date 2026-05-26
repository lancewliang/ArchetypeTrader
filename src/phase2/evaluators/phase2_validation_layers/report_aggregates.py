"""Phase II report aggregate builders.

本文件只把 evaluator 已经拿到的逐样本数组聚合成 report payload，不应用
validation rule，也不重新选择 checkpoint。
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Sequence

import numpy as np

from src.phase1.evaluators.phase1_validation_layers.layer2_behavior_quality import (
    classify_action_motif,
    classify_market_morphology,
)
from src.utils import nan_value


_EPS = 1e-12


def build_selector_pair_profitability_matrix(
    *,
    selected_code_ids: np.ndarray,
    selector_returns: np.ndarray,
    kl_returns: np.ndarray,
    random_returns: np.ndarray | None = None,
    selector_fees: np.ndarray | None = None,
    selector_turnover: np.ndarray | None = None,
    prices: np.ndarray | None = None,
    selector_actions: np.ndarray | None = None,
    morphologies: Sequence[str] | np.ndarray | None = None,
    selector_motifs: Sequence[str] | np.ndarray | None = None,
    fee_rate: float = 0.0002,
) -> tuple[dict[str, Any], ...]:
    """Build ``selector_pair_profitability_matrix`` report rows.

    Phase II 口径：
        - morphology 来自价格路径，或调用方显式传入的 ``morphologies``；
        - motif 来自 selector 选择 code 后 decoder 生成的 action sequence，或
          调用方显式传入的 ``selector_motifs``；
        - advantage 使用 selector return 相对 KL/assigned-label baseline 和
          random baseline 的均值差。

    返回值是 JSON-friendly rows，可直接放入
    ``Phase2ValidationPayloads.report_payload``。
    """

    selector_values = _as_float_array(selector_returns)
    kl_values = _as_float_array(kl_returns)
    selected_codes = _as_int_array(selected_code_ids)
    sample_count = selector_values.shape[0]
    if sample_count == 0:
        return ()
    if kl_values.shape[0] != sample_count or selected_codes.shape[0] != sample_count:
        return ()

    morphology_values = _resolve_morphologies(
        morphologies=morphologies,
        prices=prices,
        sample_count=sample_count,
        fee_rate=fee_rate,
    )
    motif_values = _resolve_motifs(
        selector_motifs=selector_motifs,
        selector_actions=selector_actions,
        prices=prices,
        sample_count=sample_count,
    )
    if morphology_values.shape[0] != sample_count or motif_values.shape[0] != sample_count:
        return ()

    random_values = _optional_float_array(random_returns, sample_count)
    fee_values = _optional_float_array(selector_fees, sample_count)
    turnover_values = _optional_float_array(selector_turnover, sample_count)

    rows: list[dict[str, Any]] = []
    pairs = sorted(
        {
            (str(morphology), str(motif))
            for morphology, motif in zip(morphology_values, motif_values, strict=False)
        }
    )
    for morphology, motif in pairs:
        mask = (morphology_values == morphology) & (motif_values == motif)
        support = int(np.sum(mask))
        if support <= 0:
            continue
        selector_mean = _masked_mean(selector_values, mask)
        kl_mean = _masked_mean(kl_values, mask)
        random_mean = _masked_mean(random_values, mask)
        dominant_code, dominant_ratio = _dominant_value(selected_codes[mask])
        rows.append(
            {
                "morphology": morphology,
                "motif": motif,
                "support": support,
                "selector_mean_return": selector_mean,
                "kl_mean_return": kl_mean,
                "random_mean_return": random_mean,
                "mean_advantage_vs_kl": _safe_difference(selector_mean, kl_mean),
                "mean_advantage_vs_random": _safe_difference(
                    selector_mean,
                    random_mean,
                ),
                "win_rate": _masked_rate(selector_values > 0.0, mask),
                "fee_drag_ratio": _fee_drag_ratio(fee_values[mask], selector_values[mask]),
                "mean_turnover": _masked_mean(turnover_values, mask),
                "dominant_selected_code": dominant_code,
                "dominant_selected_code_ratio": dominant_ratio,
            }
        )
    return tuple(rows)


def build_phase2_code_diagnostics(
    *,
    selected_code_ids: np.ndarray,
    assigned_code_labels: np.ndarray,
    selector_returns: np.ndarray,
    kl_returns: np.ndarray,
    num_archetypes: int,
    selector_fees: np.ndarray | None = None,
    selector_turnover: np.ndarray | None = None,
    q_margins: np.ndarray | None = None,
    low_confidence_margin: float = 0.10,
    prices: np.ndarray | None = None,
    selector_actions: np.ndarray | None = None,
    morphologies: Sequence[str] | np.ndarray | None = None,
    selector_motifs: Sequence[str] | np.ndarray | None = None,
    active_ratio_min: float = 0.01,
    dominant_ratio_warn_min: float = 0.30,
    fee_rate: float = 0.0002,
) -> tuple[dict[str, Any], ...]:
    """Build complete ``code_diagnostics`` report rows.

    该函数补充 Layer 4 基础 usage row 没有覆盖的行为归因字段：
    dominant morphology、dominant motif、dominant pair、Q margin、低置信比例和
    坏偏离比例。
    """

    selected_codes = _as_int_array(selected_code_ids)
    assigned_labels = _as_int_array(assigned_code_labels)
    selector_values = _as_float_array(selector_returns)
    kl_values = _as_float_array(kl_returns)
    sample_count = selector_values.shape[0]
    if (
        sample_count == 0
        or selected_codes.shape[0] != sample_count
        or assigned_labels.shape[0] != sample_count
        or kl_values.shape[0] != sample_count
        or num_archetypes <= 0
    ):
        return ()

    morphology_values = _resolve_morphologies(
        morphologies=morphologies,
        prices=prices,
        sample_count=sample_count,
        fee_rate=fee_rate,
    )
    motif_values = _resolve_motifs(
        selector_motifs=selector_motifs,
        selector_actions=selector_actions,
        prices=prices,
        sample_count=sample_count,
    )
    if morphology_values.shape[0] != sample_count:
        morphology_values = np.full(sample_count, "", dtype=object)
    if motif_values.shape[0] != sample_count:
        motif_values = np.full(sample_count, "", dtype=object)

    selector_distribution = _distribution(selected_codes, num_archetypes)
    kl_distribution = _distribution(assigned_labels, num_archetypes)
    fee_values = _optional_float_array(selector_fees, sample_count)
    turnover_values = _optional_float_array(selector_turnover, sample_count)
    q_margin_values = _optional_float_array(q_margins, sample_count)

    rows: list[dict[str, Any]] = []
    for code_id in range(int(num_archetypes)):
        selected_mask = selected_codes == code_id
        assigned_mask = assigned_labels == code_id
        selector_support = int(np.sum(selected_mask))
        kl_support = int(np.sum(assigned_mask))
        selector_mean = _masked_mean(selector_values, selected_mask)
        kl_mean = _masked_mean(kl_values, assigned_mask)
        uplift = _safe_difference(selector_mean, kl_mean)
        dominant_morphology, dominant_morphology_ratio = _dominant_label(
            morphology_values[selected_mask]
        )
        dominant_motif, dominant_motif_ratio = _dominant_label(
            motif_values[selected_mask]
        )
        dominant_pair, dominant_pair_ratio = _dominant_pair(
            morphology_values[selected_mask],
            motif_values[selected_mask],
        )
        profitable_deviation_count, unprofitable_deviation_count = _deviation_counts(
            selected_mask=selected_mask,
            assigned_labels=assigned_labels,
            selected_codes=selected_codes,
            selector_returns=selector_values,
            kl_returns=kl_values,
        )
        unprofitable_deviation_rate = (
            unprofitable_deviation_count / selector_support
            if selector_support > 0
            else nan_value()
        )
        status, risk_reason = _code_status(
            selector_support=selector_support,
            selector_ratio=float(selector_distribution[code_id]),
            selector_mean_return=selector_mean,
            uplift_vs_kl=uplift,
            unprofitable_deviation_rate=unprofitable_deviation_rate,
            dominant_pair_ratio=dominant_pair_ratio,
            active_ratio_min=active_ratio_min,
            dominant_ratio_warn_min=dominant_ratio_warn_min,
        )
        rows.append(
            {
                "code_id": code_id,
                "selector_support": selector_support,
                "selector_usage_ratio": float(selector_distribution[code_id]),
                "kl_support": kl_support,
                "kl_usage_ratio": float(kl_distribution[code_id]),
                "usage_delta": float(selector_distribution[code_id] - kl_distribution[code_id]),
                "selector_mean_return": selector_mean,
                "kl_mean_return": kl_mean,
                "uplift_vs_kl": uplift,
                "selector_win_rate": _masked_rate(selector_values > 0.0, selected_mask),
                "selector_fee_drag_ratio": _fee_drag_ratio(
                    fee_values[selected_mask],
                    selector_values[selected_mask],
                ),
                "selector_turnover": _masked_mean(turnover_values, selected_mask),
                "dominant_morphology": dominant_morphology,
                "dominant_morphology_ratio": dominant_morphology_ratio,
                "dominant_motif": dominant_motif,
                "dominant_motif_ratio": dominant_motif_ratio,
                "dominant_pair": dominant_pair,
                "dominant_pair_ratio": dominant_pair_ratio,
                "mean_q_margin": _masked_mean(q_margin_values, selected_mask),
                "low_confidence_ratio": _masked_rate(
                    q_margin_values <= float(low_confidence_margin),
                    selected_mask & np.isfinite(q_margin_values),
                ),
                "profitable_deviation_count": profitable_deviation_count,
                "unprofitable_deviation_count": unprofitable_deviation_count,
                "unprofitable_deviation_rate": unprofitable_deviation_rate,
                "status": status,
                "risk_reason": risk_reason,
            }
        )
    return tuple(rows)


def _resolve_morphologies(
    *,
    morphologies: Sequence[str] | np.ndarray | None,
    prices: np.ndarray | None,
    sample_count: int,
    fee_rate: float,
) -> np.ndarray:
    if morphologies is not None:
        values = np.asarray(morphologies, dtype=object).reshape(-1)
        return values if values.shape[0] == sample_count else np.asarray([], dtype=object)
    if prices is None:
        return np.asarray([], dtype=object)
    values = classify_market_morphology(prices, fee_rate=fee_rate)
    return values.astype(object).reshape(-1)


def _resolve_motifs(
    *,
    selector_motifs: Sequence[str] | np.ndarray | None,
    selector_actions: np.ndarray | None,
    prices: np.ndarray | None,
    sample_count: int,
) -> np.ndarray:
    if selector_motifs is not None:
        values = np.asarray(selector_motifs, dtype=object).reshape(-1)
        return values if values.shape[0] == sample_count else np.asarray([], dtype=object)
    if selector_actions is None:
        return np.asarray([], dtype=object)
    values = classify_action_motif(selector_actions, prices)
    return values.astype(object).reshape(-1)


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _as_int_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _optional_float_array(values: Any, sample_count: int) -> np.ndarray:
    if values is None:
        return np.full(sample_count, nan_value(), dtype=np.float64)
    array = _as_float_array(values)
    if array.shape[0] != sample_count:
        return np.full(sample_count, nan_value(), dtype=np.float64)
    return array


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    size = min(values.shape[0], mask.shape[0])
    if size <= 0:
        return nan_value()
    selected = values[:size][mask[:size] & np.isfinite(values[:size])]
    if selected.size == 0:
        return nan_value()
    return float(np.mean(selected))


def _masked_rate(mask_values: np.ndarray, mask: np.ndarray) -> float:
    size = min(mask_values.shape[0], mask.shape[0])
    if size <= 0:
        return nan_value()
    selected = mask_values[:size][mask[:size]]
    if selected.size == 0:
        return nan_value()
    return float(np.mean(selected.astype(np.float64)))


def _safe_difference(left: float, right: float) -> float:
    if not math.isfinite(left) or not math.isfinite(right):
        return nan_value()
    return float(left - right)


def _fee_drag_ratio(fees: np.ndarray, returns: np.ndarray) -> float:
    fee_values = fees[np.isfinite(fees)]
    positive_returns = returns[np.isfinite(returns) & (returns > 0.0)]
    if fee_values.size == 0:
        return nan_value()
    denominator = float(np.sum(positive_returns))
    if denominator <= _EPS:
        return float("inf")
    return float(np.sum(fee_values) / denominator)


def _distribution(values: np.ndarray, num_archetypes: int) -> np.ndarray:
    valid = values[(values >= 0) & (values < num_archetypes)]
    counts = np.bincount(valid, minlength=num_archetypes).astype(np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros(num_archetypes, dtype=np.float64)
    return counts / total


def _dominant_value(values: np.ndarray) -> tuple[int | None, float]:
    if values.size == 0:
        return None, nan_value()
    counter = Counter(int(value) for value in values)
    value, count = counter.most_common(1)[0]
    return value, float(count / values.size)


def _dominant_label(values: np.ndarray) -> tuple[str | None, float]:
    filtered = [str(value) for value in values if str(value)]
    if not filtered:
        return None, nan_value()
    value, count = Counter(filtered).most_common(1)[0]
    return value, float(count / len(filtered))


def _dominant_pair(
    morphologies: np.ndarray,
    motifs: np.ndarray,
) -> tuple[str | None, float]:
    pairs = [
        (str(morphology), str(motif))
        for morphology, motif in zip(morphologies, motifs, strict=False)
        if str(morphology) and str(motif)
    ]
    if not pairs:
        return None, nan_value()
    (morphology, motif), count = Counter(pairs).most_common(1)[0]
    return f"{morphology} / {motif}", float(count / len(pairs))


def _deviation_counts(
    *,
    selected_mask: np.ndarray,
    assigned_labels: np.ndarray,
    selected_codes: np.ndarray,
    selector_returns: np.ndarray,
    kl_returns: np.ndarray,
) -> tuple[int, int]:
    deviation_mask = selected_mask & (selected_codes != assigned_labels)
    finite = deviation_mask & np.isfinite(selector_returns) & np.isfinite(kl_returns)
    differences = selector_returns[finite] - kl_returns[finite]
    return int(np.sum(differences > 0.0)), int(np.sum(differences < 0.0))


def _code_status(
    *,
    selector_support: int,
    selector_ratio: float,
    selector_mean_return: float,
    uplift_vs_kl: float,
    unprofitable_deviation_rate: float,
    dominant_pair_ratio: float,
    active_ratio_min: float,
    dominant_ratio_warn_min: float,
) -> tuple[str, str]:
    if selector_support <= 0 or selector_ratio < active_ratio_min:
        return "warn_low_support", "selector 使用样本太少，per-code 统计不稳定。"
    if math.isfinite(selector_mean_return) and selector_mean_return < 0.0:
        return "warn_unprofitable", "selector 选择该 code 后平均收益为负。"
    if math.isfinite(uplift_vs_kl) and uplift_vs_kl < 0.0:
        return "warn_bad_deviation", "selector 在该 code 上弱于 assigned-label baseline。"
    if (
        math.isfinite(unprofitable_deviation_rate)
        and unprofitable_deviation_rate > 0.25
    ):
        return "warn_bad_deviation", "偏离 assigned label 后亏损样本比例偏高。"
    if math.isfinite(dominant_pair_ratio) and dominant_pair_ratio < dominant_ratio_warn_min:
        return "warn_low_support", "dominant morphology-motif pair 不清晰。"
    return "pass", "使用充分且未发现明显 per-code 风险。"


__all__ = [
    "build_phase2_code_diagnostics",
    "build_selector_pair_profitability_matrix",
]
