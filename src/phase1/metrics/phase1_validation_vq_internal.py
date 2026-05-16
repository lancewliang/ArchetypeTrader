"""Phase I layer 1 VQ internal schema, thresholds, and hard gate rules."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

from src.utils import _dataclass_from_mapping

if TYPE_CHECKING:
    from .phase1_metric_results import Phase1LayerResult
    from .phase1_validation_data_schema import Phase1ValidationMetrics


@dataclass(frozen=True)
class Phase1VQInternalMetrics:
    """第一层 VQ 内部质量 raw metrics。

    使用场景:
        输入 ``evaluate_vq_internal_rules()``，判断 codebook 是否稳定、可用、
        未塌缩，并为 scoring/report 提供基础诊断。
    """

    # validation 动作重构准确率。
    validation_action_accuracy: float

    # validation reconstruction loss / train reconstruction loss。
    reconstruction_loss_gap: float

    # active code 数量 / codebook size。
    active_code_ratio: float

    # 单个 code 的最大 occupancy。
    max_code_occupancy: float

    # exp(code entropy) / codebook size。
    normalized_code_perplexity: float

    # dead code 数量 / codebook size。
    dead_code_ratio: float

    # 最近若干 epoch assignment churn 均值。
    assignment_churn_recent_mean: float

    # active code 中 lifetime 达标的比例。
    code_lifetime_pass_ratio: float

    # mean(||z_e - z_q||_2) 或同类量化距离指标。
    quantization_distance: float

    # 最近 code 与第二近 code 距离 margin 的中位数。
    nearest_second_margin_median: float

    # decoded turnover 与 demo turnover 的平均误差。
    decoder_turnover_error: float

    # decoded 首次入场时点与 demo 首次入场时点的 median error。
    entry_timing_error_median: float

    # decoded 主方向与 demo 主方向一致率。
    direction_accuracy: float

    # validation quantization distance / train quantization distance。
    # 旧 checkpoint 可能没有该字段，因此给 NaN 默认值。
    quantization_distance_gap: float = float("nan")

    # validation split 的 code 使用分布，索引为 code id，值为 occupancy probability。
    code_distribution: tuple[float, ...] = ()

    # validation split 中达到 active occupancy 阈值的 code id。
    active_codes: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict，供 checkpoint/report 落盘。"""

        payload = asdict(self)
        payload["code_distribution"] = [
            float(value) for value in self.code_distribution
        ]
        payload["active_codes"] = [int(code_id) for code_id in self.active_codes]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1VQInternalMetrics":
        """从 dict 恢复第一层 metrics。"""

        field_names = {field.name for field in fields(cls)}
        values = {key: value for key, value in payload.items() if key in field_names}
        values["code_distribution"] = tuple(
            float(value) for value in payload.get("code_distribution", ())
        )
        values["active_codes"] = tuple(
            int(code_id) for code_id in payload.get("active_codes", ())
        )
        return cls(**values)


@dataclass(frozen=True)
class Phase1VQInternalThresholds:
    """第一层 VQ 内部质量阈值配置。

    功能说明:
        保存用于判断 VQ codebook 是否稳定、未塌缩、可用的 hard gate 阈值。

    使用场景:
        由 ``evaluate_vq_internal_rules()`` 消费。该层重点检查重构保真、
        code 使用健康度、assignment 稳定性和量化边界清晰度。
    """

    # validation 动作重构准确率下限。用于第一层判断 decoder 是否保留 DP 动作信息。
    action_accuracy_min: float = 0.85

    # validation/train 重构损失比值上限。用于检查重构能力是否明显过拟合训练集。
    reconstruction_loss_gap_max: float = 1.25

    # active code 占 codebook 的比例下限。用于检查 codebook 是否被充分使用。
    active_code_ratio_min: float = 0.80

    # 单个 code 最大占用比例上限。用于检测 label collapse 或单 code 吃掉过多样本。
    max_code_occupancy_max: float = 0.40

    # 归一化 perplexity 下限。用于防止 code 分布过度塌缩。
    normalized_perplexity_min: float = 0.50

    # 归一化 perplexity 上限。用于防止分配过于接近随机、缺少结构。
    normalized_perplexity_max: float = 0.90

    # dead code 比例上限。用于限制长期几乎没有样本分配的无效 code 数量。
    dead_code_ratio_max: float = 0.20

    # 最近若干 epoch assignment churn 均值上限。用于判断 label 语义是否稳定。
    churn_recent_mean_max: float = 0.15

    # 最近 code 距离和第二近 code 距离的 margin 中位数下限。用于判断分配边界是否清晰。
    margin_median_min: float = 0.10

    # decoded 主方向和 demo 主方向一致率下限。用于检查 long/short/flat 大方向是否保真。
    direction_accuracy_min: float = 0.88

    # 入场时点误差 timestep 上限。固定 horizon=72 时，10.8 对应 15% horizon。
    entry_timing_error_max: float = 10.8

    # active code 中 lifetime 达标的比例下限。用于判断 code 是否已经形成稳定语义。
    code_lifetime_pass_ratio_min: float = 0.80

    # decoded 与 demo 的换手误差上限，单位为原始换手次数。
    decoder_turnover_error_max: float = 18.0

    # validation/train 量化距离比值 warn/scoring 参考值，不参与当前 hard gate。
    quantization_distance_gap_max: float = 1.25

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供 checkpoint、report 或日志落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1VQInternalThresholds":
        """从 checkpoint/report 中的 dict 恢复第一层阈值配置。"""

        return _dataclass_from_mapping(cls, payload)


def _missing_warn_result(
    *,
    name: str,
    threshold: str,
    layer: str,
    message: str,
    direction_message: str,
) -> Phase1MetricResult:
    """构造缺失但可解释的 VQ internal warning。"""

    from .phase1_metric_results import Phase1MetricResult

    return Phase1MetricResult(
        name=name,
        value=None,
        threshold=threshold,
        severity="warn",
        passed=True,
        layer=layer,
        message=f"{message}；{direction_message}",
    )


def evaluate_vq_internal_rules(
    metrics: Phase1VQInternalMetrics,
    thresholds: Phase1VQInternalThresholds,
) -> Phase1LayerResult:
    """判定第一层 VQ 内部质量。

    审计问题:
        codebook 是否完成了“稳定且有保真的压缩”。这一层只看 VQ 自身质量，不判断
        每个 code 是否有交易语义，也不证明可预测性。
    """

    from .phase1_validation_rule_helpers import _between, _build_layer_result, _ge, _le

    layer = "vq_internal"
    churn_result = (
        _missing_warn_result(
            name="assignment_churn_recent_mean",
            threshold=f"<= {thresholds.churn_recent_mean_max:g}",
            layer=layer,
            message=(
                "近期 assignment churn 缺失，通常表示训练初期 history 不足；"
                "正式 checkpoint selection 前应补齐 history"
            ),
            direction_message=(
                "指标方向：越小越好；变大表示 label 语义更不稳定，"
                "变小表示 assignment 更稳定"
            ),
        )
        if math.isnan(metrics.assignment_churn_recent_mean)
        else _le(
            name="assignment_churn_recent_mean",
            value=metrics.assignment_churn_recent_mean,
            threshold_value=thresholds.churn_recent_mean_max,
            layer=layer,
            message="近期 assignment churn 需要足够低，保证 label 语义稳定",
        )
    )
    entry_timing_result = (
        _missing_warn_result(
            name="entry_timing_error_median",
            threshold=f"<= {thresholds.entry_timing_error_max:g}",
            layer=layer,
            message=(
                "入场时点误差缺失，通常表示没有 demo/decoded 同时入场样本；"
                "需要结合 direction accuracy 和 flat 样本占比解释"
            ),
            direction_message=(
                "指标方向：越小越好；变大表示 decoded 入场时点偏离 demo 更远，"
                "变小表示入场时点更一致"
            ),
        )
        if math.isnan(metrics.entry_timing_error_median)
        else _le(
            name="entry_timing_error_median",
            value=metrics.entry_timing_error_median,
            threshold_value=thresholds.entry_timing_error_max,
            layer=layer,
            message="入场时点误差的 timestep 偏移不能过大",
        )
    )
    results = (
        _ge(
            name="validation_action_accuracy",
            value=metrics.validation_action_accuracy,
            threshold_value=thresholds.action_accuracy_min,
            layer=layer,
            message="decoder 需要在 assigned code 条件下重构 DP 动作",
        ),
        _le(
            name="reconstruction_loss_gap",
            value=metrics.reconstruction_loss_gap,
            threshold_value=thresholds.reconstruction_loss_gap_max,
            layer=layer,
            message="validation/train 重构损失差距不能过大",
        ),
        _ge(
            name="active_code_ratio",
            value=metrics.active_code_ratio,
            threshold_value=thresholds.active_code_ratio_min,
            layer=layer,
            message="codebook 需要有足够比例的 code 被有效使用",
        ),
        _le(
            name="max_code_occupancy",
            value=metrics.max_code_occupancy,
            threshold_value=thresholds.max_code_occupancy_max,
            layer=layer,
            message="单个 code 不能占用过多样本，避免 label collapse",
        ),
        _between(
            name="normalized_code_perplexity",
            value=metrics.normalized_code_perplexity,
            lower=thresholds.normalized_perplexity_min,
            upper=thresholds.normalized_perplexity_max,
            layer=layer,
            message="code 分布既不能塌缩，也不能接近随机无结构",
        ),
        _le(
            name="dead_code_ratio",
            value=metrics.dead_code_ratio,
            threshold_value=thresholds.dead_code_ratio_max,
            layer=layer,
            message="dead code 比例不能过高",
        ),
        churn_result,
        _ge(
            name="code_lifetime_pass_ratio",
            value=metrics.code_lifetime_pass_ratio,
            threshold_value=thresholds.code_lifetime_pass_ratio_min,
            layer=layer,
            message="active code 需要有足够比例保持稳定生命周期",
        ),
        _ge(
            name="nearest_second_margin_median",
            value=metrics.nearest_second_margin_median,
            threshold_value=thresholds.margin_median_min,
            layer=layer,
            message="样本到最近和第二近 code 的距离 margin 需要清晰",
        ),
        _ge(
            name="direction_accuracy",
            value=metrics.direction_accuracy,
            threshold_value=thresholds.direction_accuracy_min,
            layer=layer,
            message="decoded 主方向需要和 DP demo 主方向一致",
        ),
        entry_timing_result,
        _le(
            name="decoder_turnover_error",
            value=metrics.decoder_turnover_error,
            threshold_value=thresholds.decoder_turnover_error_max,
            layer=layer,
            message="decoded 不能相对 DP demo 引入过多额外换手",
        ),
    )
    return _build_layer_result(layer_id=1, name=layer, metrics=results)


def compute_codebook_health_score(metrics: Phase1ValidationMetrics) -> float:
    """计算 codebook health 子分数。"""

    from .phase1_validation_score_helpers import (
        _clip01,
        _inverse_ratio_score,
        _threshold_progress,
    )

    vq = metrics.vq_internal
    perplexity_center = 0.70
    perplexity_score = 1.0 - min(
        1.0,
        abs(vq.normalized_code_perplexity - perplexity_center) / 0.40,
    )
    parts = (
        _threshold_progress(vq.active_code_ratio, 0.80),
        _clip01(perplexity_score),
        _inverse_ratio_score(vq.max_code_occupancy, 0.40),
        _inverse_ratio_score(vq.dead_code_ratio, 0.20),
        _inverse_ratio_score(vq.assignment_churn_recent_mean, 0.15),
        _threshold_progress(vq.nearest_second_margin_median, 0.10),
        _threshold_progress(vq.direction_accuracy, 0.88),
    )
    return sum(parts) / len(parts)


__all__ = [
    "compute_codebook_health_score",
    "Phase1VQInternalMetrics",
    "Phase1VQInternalThresholds",
    "evaluate_vq_internal_rules",
]
