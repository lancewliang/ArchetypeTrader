"""Phase I 指标门面（重新导出 metrics/ 子包稳定 API）.

设计文档锚点: §4.11。
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from src.evaluation.metrics.action import (  # noqa: F401
    ConfusionMatrix,
    SwitchMetrics,
    action_confusion_matrix,
    non_flat_accuracy,
    reconstruction_accuracy,
    single_trade_consistency_rate,
    switch_metrics,
    weighted_reconstruction_accuracy,
)
from src.evaluation.metrics.archetype import (  # noqa: F401
    ArchetypeDiagnostics,
    PerCodeStats,
    TeacherQuality,
    dp_teacher_quality,
    per_code_summary,
)
from src.evaluation.metrics.behavior import (  # noqa: F401
    decoder_sensitivity_to_code,
    inter_code_action_diversity,
    inter_code_distance,
    latent_silhouette_score,
    per_code_action_entropy,
)
from src.evaluation.metrics.risk import (  # noqa: F401
    DEFAULT_ANNUALIZATION_FACTOR,
    calmar_ratio,
    equity_curve_from_step_returns,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from src.evaluation.metrics.stability import (  # noqa: F401
    codebook_displacement,
    epoch_code_stability,
    horizon_boundary_metrics,
    matched_epoch_code_stability,
)


def codebook_perplexity(code_ids: Sequence[int], num_codes: int) -> float:
    """``exp(-Σ p_i log p_i)``，其中 ``p_i = count_i / N``。

    放在门面而不是子包，是因为 perplexity 跨 action / archetype 域使用。
    接近 ``1.0`` 表示崩塌到单 code；理想情况下接近 ``K``（均匀分布）。
    """
    counts = [0] * num_codes
    for cid in code_ids:
        if 0 <= int(cid) < num_codes:
            counts[int(cid)] += 1
    total = sum(counts) or 1
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return math.exp(h)


def code_usage_ratio(code_ids: Sequence[int], num_codes: int) -> float:
    """使用过的 code 数 / K。selection_policy guardrail 默认要求 ≥ 0.7。"""
    used = len({int(c) for c in code_ids})
    return used / max(num_codes, 1)


def return_capture_ratio(student_return: float, teacher_return: float, eps: float = 1e-8) -> float:
    """``student / max(abs(teacher), eps)``；teacher 接近 0 时不返回 inf。

    Notes
    -----
    与 ``regret_to_dp`` 配合使用：
    - capture 接近 1 但 regret 仍 > 0 → 仍不及老师；
    - capture 远低于 1 但 teacher 本身 profitable_ratio 低 → 不能解读为学生学得差。
    """
    return student_return / max(abs(teacher_return), eps)


def regret_to_dp(student_return: float, teacher_return: float) -> float:
    """``teacher - student``：值越大表示越远落后于 DP teacher。"""
    return teacher_return - student_return


def phase1_composite_score(metrics: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, dict]:
    """加权和；返回 ``(score, debug_info)``。

    实现注意
    --------
    - 缺失指标视为 0；同时把缺失 key 写入 debug 的 ``missing_metrics`` 字段，
      供 sensitivity 试验与 selection_policy 审计追溯。
    - 不在内部归一化指标量级；权重必须由配置层（``SelectionPolicyConfig.metric_weights``）
      预先合理选定。

    Returns
    -------
    (score, debug) :
        - ``score`` : float，加权和。
        - ``debug`` : ``{"weights": ..., "contrib": {key: value*weight}, "missing_metrics": [...]}``。
    """
    score = 0.0
    missing: List[str] = []
    contrib: Dict[str, float] = {}
    for key, w in weights.items():
        if key not in metrics:
            missing.append(key)
            continue
        v = float(metrics[key])
        contrib[key] = v * w
        score += contrib[key]
    debug = {"weights": dict(weights), "contrib": contrib, "missing_metrics": missing}
    return score, debug


def composite_score_sensitivity(
    metrics: Dict[str, float],
    base_weights: Dict[str, float],
    perturbations: List[Dict[str, float]],
) -> dict:
    """权重 sensitivity: 对每组扰动重算 ``composite_score``。

    Steps
    -----
    1. 用 ``base_weights`` 计算 ``base_score``。
    2. 对每组 ``perturbation``（``key -> delta`` 字典）拷贝并加和到 weights，
       重新算 score。
    3. 输出结构供 ``composite_score_sensitivity.json`` 直接序列化。

    Returns
    -------
    dict
        ``{"base_score": float, "base_debug": dict,
           "results": [{"perturbation": ..., "score": ..., "weights": ..., "debug": ...}, ...]}``
    """
    base_score, base_debug = phase1_composite_score(metrics, base_weights)
    results = []
    for delta in perturbations:
        weights = dict(base_weights)
        for k, dv in delta.items():
            weights[k] = weights.get(k, 0.0) + dv
        score, debug = phase1_composite_score(metrics, weights)
        results.append({"perturbation": delta, "score": score, "weights": weights, "debug": debug})
    return {"base_score": base_score, "base_debug": base_debug, "results": results}


def composite_score_sensitivity_across_epochs(
    epoch_metrics: Sequence[Dict[str, float]],
    base_weights: Dict[str, float],
    perturbations: List[Dict[str, float]],
) -> dict:
    """权重 sensitivity: 对全部 epoch metrics 重新选择 best epoch。

    设计 §9.5 要求观察不同权重下 best epoch 是否漂移。该函数不只对
    单个 best metrics 重算分数，而是对 manifest 中所有 epoch metrics
    逐个打分，再按每组权重选出新的 best。
    """
    metrics_list = [dict(m) for m in epoch_metrics]
    if not metrics_list:
        return {
            "base_best": None,
            "results": [],
            "best_epoch_drift": False,
            "best_epochs": [],
        }
    eligible_metrics = [
        m for m in metrics_list
        if m.get("_manifest_verdict") not in {"reject", "fatal"}
    ] or metrics_list

    def _best_for(weights: Dict[str, float]) -> dict:
        best_payload = None
        best_score = -float("inf")
        best_debug = {}
        for payload in eligible_metrics:
            score, debug = phase1_composite_score(payload, weights)
            if score > best_score:
                best_score = score
                best_debug = debug
                best_payload = payload
        assert best_payload is not None
        core = {
            key: best_payload.get(key)
            for key in (
                "teacher_val_code_usage_ratio",
                "teacher_val_return_capture_ratio",
                "teacher_val_max_drawdown",
                "teacher_val_sharpe_ratio",
                "phase1_composite_score",
            )
            if key in best_payload
        }
        return {
            "best_epoch": int(best_payload.get("epoch", -1)),
            "score": best_score,
            "weights": dict(weights),
            "debug": best_debug,
            "core_metrics": core,
        }

    base_best = _best_for(base_weights)
    results = []
    best_epochs = [base_best["best_epoch"]]
    for delta in perturbations:
        weights = dict(base_weights)
        for key, value in delta.items():
            weights[key] = weights.get(key, 0.0) + value
        best = _best_for(weights)
        best["perturbation"] = dict(delta)
        results.append(best)
        best_epochs.append(best["best_epoch"])
    return {
        "base_best": base_best,
        "results": results,
        "best_epoch_drift": len(set(best_epochs)) > 1,
        "best_epochs": best_epochs,
    }
