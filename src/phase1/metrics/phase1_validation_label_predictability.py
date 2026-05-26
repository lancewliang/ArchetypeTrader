"""Phase I layer 4 label predictability schema and reference checks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.utils import PydanticMappingModel

if TYPE_CHECKING:
    from .phase1_metric_results import Phase1LayerResult
    from .phase1_validation_data_schema import Phase1ValidationMetrics


class Phase1LabelPredictabilityPayload(PydanticMappingModel):
    """第四层 label predictability 计算的中间 payload。

    使用场景:
        保存 probe train/validation accuracy、generalization gap、confusion matrix
        和 probe seed。该对象实现 ``Mapping``，用于兼容现有
        ``extra_payload["..."]`` 调用。
    """

    # probe 在 train split 上的 top-1 accuracy。
    probe_train_accuracy: float

    # probe 在 validation split 上的 top-1 accuracy。
    probe_validation_accuracy: float

    # train accuracy - validation accuracy。
    probe_predictability_gap: float

    # validation probe confusion matrix，行是真实 code，列是预测 code。
    probe_confusion_matrix: tuple[tuple[int, ...], ...]

    # probe 训练和随机 baseline 使用的随机种子。
    probe_seed: int

class Phase1LabelPredictabilityMetrics(PydanticMappingModel):
    """第四层 label 可预测性 raw metrics。"""

    # probe 在 validation split 上的 top-1 label accuracy。
    probe_top1_accuracy: float

    # probe 在 validation split 上的 top-3 label accuracy。
    probe_top3_accuracy: float

    # 按 active code recall 取平均的 balanced accuracy。
    probe_balanced_accuracy: float

    # 给定 morphology 后的 label 条件熵。
    label_entropy_given_morphology: float

    # label 与可见状态或 morphology 的互信息相对随机置换 baseline 的提升倍数。
    mutual_information_lift: float

    # probe top-1 label decoded return 相对 oracle assigned-label decoded return 的保留比例。
    probe_return_retention: float

    # validation split assigned label 的全局熵 H(label)。
    label_entropy: float = float("nan")

    # 当前 validation split 中的 codebook size，用于 top-k accuracy 自适应阈值。
    num_codes: int = 0

class Phase1LabelPredictabilityThresholds(PydanticMappingModel):
    """第四层 label 可预测性阈值配置。"""

    # probe top-1 accuracy 的固定下限。实际阈值取 max(floor, k_factor / K)。
    probe_top1_floor: float = 0.25

    # probe top-1 accuracy 的 codebook size 自适应倍数。用于保证表现明显优于随机猜测。
    probe_top1_k_factor: float = 1.5

    # probe top-3 accuracy 的固定下限。实际阈值取 max(floor, k_factor / K)。
    probe_top3_floor: float = 0.55

    # probe top-3 accuracy 的 codebook size 自适应倍数。用于判断 selector 是否能缩小候选范围。
    probe_top3_k_factor: float = 3.0

    # probe balanced accuracy 下限。用于避免 probe 只学会预测高频 code。
    probe_balanced_accuracy_min: float = 0.25

    # label 和可见状态之间互信息相对随机置换 baseline 的提升下限。
    mutual_information_lift_min: float = 2.0

    # probe label decoded return 相对 oracle assigned-label return 的保留比例下限。
    probe_return_retention_min: float = 0.35

    # H(label | morphology) 相对 H(label) 的上限，用于判断 morphology 是否解释 label 结构。
    label_entropy_given_morphology_max_ratio: float = 0.80

def evaluate_label_predictability_rules(
    metrics: Phase1LabelPredictabilityMetrics,
    thresholds: Phase1LabelPredictabilityThresholds,
) -> Phase1LayerResult:
    """构造第四层 label 可预测性参考检查。

    Label predictability 是 Phase II selector 的训练目标，单靠轻量 probe
    validation 不应阻断 Phase I checkpoint。这里保留参考阈值、距离和 warning，
    供 report、risk finding 和 tie-breaker 使用，但所有结果都不作为 hard gate。
    """

    from .phase1_validation_rule_helpers import (
        _is_missing,
    )
    from .phase1_metric_results import Phase1LayerResult

    layer = "label_predictability"
    num_codes = max(1, int(metrics.num_codes))
    top1_threshold = max(
        thresholds.probe_top1_floor,
        thresholds.probe_top1_k_factor / num_codes,
    )
    top3_threshold = max(
        thresholds.probe_top3_floor,
        thresholds.probe_top3_k_factor / num_codes,
    )
    label_entropy_threshold = (
        metrics.label_entropy * thresholds.label_entropy_given_morphology_max_ratio
        if not _is_missing(metrics.label_entropy) and metrics.label_entropy > 0.0
        else float("nan")
    )
    results = (
        _reference_ge(
            name="probe_top1_accuracy",
            value=metrics.probe_top1_accuracy,
            threshold_value=top1_threshold,
            layer=layer,
            message="probe top-1 accuracy 用于参考是否明显高于随机水平",
        ),
        _reference_ge(
            name="probe_top3_accuracy",
            value=metrics.probe_top3_accuracy,
            threshold_value=top3_threshold,
            layer=layer,
            message="probe top-3 accuracy 用于 tie-breaker，参考 selector 缩小候选范围的能力",
        ),
        _reference_ge(
            name="probe_balanced_accuracy",
            value=metrics.probe_balanced_accuracy,
            threshold_value=thresholds.probe_balanced_accuracy_min,
            layer=layer,
            message="balanced accuracy 用于参考 probe 是否只预测高频 code",
        ),
        _reference_metric_result(
            name="label_entropy_given_morphology",
            value=(
                metrics.label_entropy_given_morphology
                if not _is_missing(label_entropy_threshold)
                else None
            ),
            threshold=(
                "<= "
                f"{thresholds.label_entropy_given_morphology_max_ratio:g} * H(label)"
            ),
            threshold_value=(
                float(label_entropy_threshold)
                if not _is_missing(label_entropy_threshold)
                else None
            ),
            direction="less_is_better",
            distance_to_threshold=(
                float(label_entropy_threshold - metrics.label_entropy_given_morphology)
                if not _is_missing(metrics.label_entropy_given_morphology)
                and not _is_missing(label_entropy_threshold)
                else None
            ),
            passed=(
                metrics.label_entropy_given_morphology <= label_entropy_threshold
                if not _is_missing(metrics.label_entropy_given_morphology)
                and not _is_missing(label_entropy_threshold)
                else False
            ),
            layer=layer,
            message="给定 morphology 后的 label 条件熵用于参考 label 结构可解释性",
            direction_message=(
                "指标方向：越小越好；变大表示 morphology 对 label 的解释力变弱，"
                "变小表示 label 结构更能被 morphology 解释"
            ),
        ),
        _reference_ge(
            name="mutual_information_lift",
            value=metrics.mutual_information_lift,
            threshold_value=thresholds.mutual_information_lift_min,
            layer=layer,
            message="mutual information lift 用于参考 label 与可见状态的统计关系",
        ),
        _reference_ge(
            name="probe_return_retention",
            value=metrics.probe_return_retention,
            threshold_value=thresholds.probe_return_retention_min,
            layer=layer,
            message="probe label decoded return retention 用于参考可预测 label 的收益损失",
        ),
    )
    return Phase1LayerResult(
        layer_id=4,
        name=layer,
        passed=True,
        metrics=results,
    )


def _reference_metric_result(
    *,
    name: str,
    value: int | float | str | bool | None,
    threshold: str,
    threshold_value: float | tuple[float, float] | None = None,
    direction: str | None = None,
    distance_to_threshold: float | None = None,
    passed: bool,
    layer: str,
    message: str,
    direction_message: str | None = None,
) -> "Phase1MetricResult":
    """创建不参与 hard gate 的参考 metric result。"""

    from .phase1_metric_results import Phase1MetricResult
    from .phase1_validation_rule_helpers import _is_missing

    full_message = message
    if direction_message is not None:
        full_message = f"{full_message}；{direction_message}"
    if _is_missing(value):
        return Phase1MetricResult(
            name=name,
            value=None,
            threshold=threshold,
            severity="warn",
            passed=True,
            layer=layer,
            message=f"{full_message}；参考指标缺失或不可计算，不作为 hard gate 失败处理",
            threshold_value=threshold_value,
            direction=direction,  # type: ignore[arg-type]
            distance_to_threshold=None,
        )
    return Phase1MetricResult(
        name=name,
        value=value,
        threshold=threshold,
        severity="pass" if passed else "warn",
        passed=True,
        layer=layer,
        message=(
            full_message
            if passed
            else f"{full_message}；低于参考阈值，仅作为风险参考和 tie-breaker 信号"
        ),
        threshold_value=threshold_value,
        direction=direction,  # type: ignore[arg-type]
        distance_to_threshold=distance_to_threshold,
    )


def _reference_ge(
    *,
    name: str,
    value: float,
    threshold_value: float,
    layer: str,
    message: str,
) -> Phase1MetricResult:
    """构造“越大越好”的非 gate 参考指标。"""

    from .phase1_validation_rule_helpers import _finite_distance, _is_missing

    return _reference_metric_result(
        name=name,
        value=value,
        threshold=f">= {threshold_value:g}",
        threshold_value=float(threshold_value),
        direction="greater_is_better",
        distance_to_threshold=(
            _finite_distance(value - threshold_value)
            if not _is_missing(value)
            else None
        ),
        passed=value >= threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
        direction_message=(
            "指标方向：越大越好；变大表示 selector 可学习性参考信号增强，"
            "变小表示 Phase II 训练难度可能上升"
        ),
    )


def compute_label_predictability_score(metrics: Phase1ValidationMetrics) -> float:
    """计算 label predictability 子分数。"""

    from .phase1_validation_score_helpers import _clip01, _threshold_progress

    label = metrics.label_predictability
    entropy_ratio_score = (
        1.0 - _clip01(label.label_entropy_given_morphology / label.label_entropy)
        if not math.isnan(label.label_entropy)
        and label.label_entropy > 0.0
        and not math.isnan(label.label_entropy_given_morphology)
        else 0.0
    )
    parts = (
        _threshold_progress(label.probe_top1_accuracy, 0.25),
        _threshold_progress(label.probe_top3_accuracy, 0.55),
        _threshold_progress(label.probe_balanced_accuracy, 0.25),
        entropy_ratio_score,
        _threshold_progress(label.mutual_information_lift, 2.0),
        _threshold_progress(label.probe_return_retention, 0.35),
    )
    return sum(parts) / len(parts)


__all__ = [
    "compute_label_predictability_score",
    "Phase1LabelPredictabilityMetrics",
    "Phase1LabelPredictabilityPayload",
    "Phase1LabelPredictabilityThresholds",
    "evaluate_label_predictability_rules",
]
