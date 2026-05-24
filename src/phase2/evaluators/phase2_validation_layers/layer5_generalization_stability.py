"""Phase II validation Layer 5: generalization, stability and predictability."""

from __future__ import annotations

import math

import numpy as np

from ...metrics import (
    Phase2GeneralizationStabilityMetrics,
    Phase2GeneralizationStabilityPayload,
    Phase2LayerComputation,
    Phase2PredictabilityMetrics,
    Phase2PredictabilityPayload,
)
from ._numeric import as_float_array, nan_value, safe_mean, safe_std


def compute_predictability_metrics(
    payload: Phase2PredictabilityPayload,
    *,
    selected_code_entropy: float,
    selected_code_entropy_given_morphology: float,
    mutual_information_lift: float,
) -> Phase2PredictabilityMetrics:
    """Build predictability metrics from probe payload and aggregate statistics.

    算法:
        1. 将 probe validation accuracy 作为当前 top-1 predictability 代理；
        2. top-3 和 balanced accuracy 当前没有完整 probe 输出，保留为 NaN；
        3. 将上游已计算的 selected code entropy、条件熵和 mutual information
           lift 合并成统一 predictability metrics。

    公式:
        - ``probe_top1_accuracy = probe_validation_accuracy``
        - ``selected_code_entropy_given_morphology = H(code | morphology)``
        - ``mutual_information_lift`` 由上游按信息增益口径传入

    说明:
        该函数不训练 probe，只把 probe 结果和分布统计组装成 Layer 5 可消费的
        metrics。top-3/balanced 缺失时返回 NaN，由 rules/report 标记为参考缺口。
    """

    return Phase2PredictabilityMetrics(
        probe_top1_accuracy=float(payload.probe_validation_accuracy),
        probe_top3_accuracy=nan_value(),
        probe_balanced_accuracy=nan_value(),
        selected_code_entropy_given_morphology=float(
            selected_code_entropy_given_morphology
        ),
        selected_code_entropy=float(selected_code_entropy),
        mutual_information_lift=float(mutual_information_lift),
    )


def compute_generalization_stability_metrics(
    payload: Phase2GeneralizationStabilityPayload,
    *,
    validation_mean_return: float,
    train_mean_return: float | None = None,
    test_mean_return: float | None = None,
    train_usage_distribution: tuple[float, ...] | None = None,
    validation_usage_distribution: tuple[float, ...] | None = None,
    q_margins: tuple[float, ...] = (),
    low_confidence_margin_threshold: float = 0.10,
    td_loss_history: tuple[float, ...] = (),
    imitation_loss_history: tuple[float, ...] = (),
    reward_mean_history: tuple[float, ...] = (),
    predictability_metrics: Phase2PredictabilityMetrics | None = None,
) -> Phase2LayerComputation:
    """Compute Layer 5 stability diagnostics from aggregate histories.

    算法:
        1. 将 validation score history、selected action churn history、Q value
           scale history 和 Q margin 序列转为 float64 数组；
        2. 用 train/validation/test mean return 计算泛化 gap；
        3. 用 train/validation selected-code 分布计算 usage KL；
        4. 用历史序列最后两个点的绝对差表示 validation score churn；
        5. 用 action churn history 均值表示跨 epoch 决策边界稳定性；
        6. 用 Q scale 均值/标准差和 Q margin 均值诊断估值尺度与置信度；
        7. 用 history 的 last - first 作为 loss/reward trend 代理；
        8. 可选合并 predictability metrics。

    核心公式:
        - ``train_val_return_gap = abs(train_mean_return - validation_mean_return)``
        - ``val_test_return_gap = abs(validation_mean_return - test_mean_return)``
        - ``train_val_usage_kl = KL(train_usage_distribution || validation_usage_distribution)``
        - ``validation_score_churn = abs(score_t - score_{t-1})``
        - ``selected_action_churn = mean(churn_history)``
        - ``q_value_scale_mean = mean(q_scale_history)``
        - ``q_value_scale_std = std(q_scale_history)``
        - ``q_margin_mean = mean(q_margin_i)``
        - ``low_confidence_selection_rate = mean(1[q_margin_i < threshold])``
        - ``trend = last_finite(history) - first_finite(history)``

    说明:
        Layer 5 当前主要是 warning/reference 层，不直接阻断 checkpoint selection。
        gap、KL、churn、Q scale、低置信比例越小越稳；Q margin 越大越稳。
    """

    validation_score_history = as_float_array(payload.validation_score_history)
    selected_action_churn_history = as_float_array(payload.selected_action_churn_history)
    q_value_scale_history = as_float_array(payload.q_value_scale_history)
    q_margin_values = as_float_array(q_margins)

    # 这里全部使用聚合历史，不回读模型、不重新 rollout。缺失历史会产生 NaN，
    # 由 Layer 5 rules/report 以 warn/reference 方式展示。
    metrics = Phase2GeneralizationStabilityMetrics(
        train_val_return_gap=_abs_gap(train_mean_return, validation_mean_return),
        val_test_return_gap=_abs_gap(validation_mean_return, test_mean_return),
        train_val_usage_kl=_kl_optional(
            train_usage_distribution,
            validation_usage_distribution,
        ),
        validation_score_churn=_last_abs_diff(validation_score_history),
        selected_action_churn=safe_mean(selected_action_churn_history),
        q_value_scale_mean=safe_mean(q_value_scale_history),
        q_value_scale_std=safe_std(q_value_scale_history),
        q_margin_mean=safe_mean(q_margin_values),
        low_confidence_selection_rate=_low_confidence_rate(
            q_margin_values,
            low_confidence_margin_threshold,
        ),
        td_loss_trend=_last_minus_first(td_loss_history),
        imitation_loss_trend=_last_minus_first(imitation_loss_history),
        reward_mean_trend=_last_minus_first(reward_mean_history),
        predictability=predictability_metrics,
    )
    return Phase2LayerComputation(
        layer_id=5,
        layer_name="generalization_stability",
        metrics=metrics,
        extra_payload={"generalization_stability_payload": payload},
    )


def _abs_gap(left: float | None, right: float | None) -> float:
    """Absolute finite gap or NaN.

    公式:
        ``gap = abs(left - right)``。

    用途:
        泛化 gap 只关心差距大小，不关心哪一侧更高；输入缺失或非有限时返回 NaN。
    """

    if left is None or right is None:
        return nan_value()
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        return nan_value()
    return abs(float(left) - float(right))


def _last_abs_diff(values: np.ndarray) -> float:
    """Absolute difference between last two finite history points.

    公式:
        ``abs(x_last - x_previous)``，其中 last/previous 都取有限历史点。

    用途:
        validation score churn 用最近两次 validation 的跳动幅度衡量 checkpoint
        高点是否稳定。
    """

    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return nan_value()
    return float(abs(finite[-1] - finite[-2]))


def _last_minus_first(values: tuple[float, ...]) -> float:
    """Trend proxy: last finite value minus first finite value.

    公式:
        ``trend = x_last_finite - x_first_finite``。

    用途:
        作为 loss/reward 是否整体上升或下降的轻量趋势代理；不拟合斜率，避免
        引入额外统计依赖。
    """

    array = as_float_array(values)
    finite = array[np.isfinite(array)]
    if finite.size < 2:
        return nan_value()
    return float(finite[-1] - finite[0])


def _low_confidence_rate(values: np.ndarray, threshold: float) -> float:
    """Rate of q margins below threshold.

    公式:
        ``mean(1[q_margin_i < threshold])`` over finite margins。

    用途:
        Q margin 低表示 top1/top2 code 区分度不足；比例越高说明 selector 决策
        越不稳定。
    """

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return nan_value()
    return float(np.mean((finite < threshold).astype(np.float64)))


def _kl_optional(
    left: tuple[float, ...] | None,
    right: tuple[float, ...] | None,
) -> float:
    """KL(left || right) for optional distributions.

    公式:
        ``KL(p || q) = Σ_k p_k log(p_k / q_k)``。

    实现细节:
        分布缺失、长度不一致或为空时返回 NaN；计算前对 p/q 加 ``1e-12`` 并重新
        归一化，避免零概率导致除零。

    用途:
        衡量 train 与 validation selected-code 使用分布漂移；越小表示泛化更稳。
    """

    if left is None or right is None:
        return nan_value()
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.size == 0 or left_values.shape != right_values.shape:
        return nan_value()
    eps = 1e-12
    p = left_values + eps
    q = right_values + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


__all__ = [
    "compute_generalization_stability_metrics",
    "compute_predictability_metrics",
]
