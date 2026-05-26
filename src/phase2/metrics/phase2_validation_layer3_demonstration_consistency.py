"""Phase II layer 3 demonstration consistency metrics."""

from __future__ import annotations

from src.utils import PydanticBaseModel

from .phase2_metric_results import Phase2LayerResult
from .phase2_validation_rule_helpers import (
    _between,
    _build_layer_result,
    _ge,
    _le,
)


class Phase2DemonstrationConsistencyPayload(PydanticBaseModel):
    """Layer 3 raw metrics 计算的中间 payload。"""

    # selector 实际选择的 code id 序列。用途：和 assigned label 比较一致性/偏离；
    # 方向：过程数据，无直接好坏方向。
    selected_code_ids: tuple[int, ...] = ()

    # Phase I assigned code label 序列。用途：作为 demonstration/先验标签；
    # 方向：过程数据，无直接好坏方向。
    assigned_code_labels: tuple[int, ...] = ()

    # selector 选择 code 后的逐样本 return。用途：和 assigned-label return 比较；
    # 方向：越大越好。
    selector_returns: tuple[float, ...] = ()

    # assigned-label baseline 的逐样本 return。用途：判断偏离 assigned label 是否有
    # 收益证明；方向：作为 baseline，selector 应高于它。
    assigned_label_returns: tuple[float, ...] = ()

    # selector 选中 code 的 Q value。用途：和 assigned label Q value 比较 margin；
    # 方向：相对 assigned_label_q_values 越高越好。
    selected_q_values: tuple[float, ...] = ()

    # assigned label 对应 code 的 Q value。用途：计算 label_q_margin；方向：过程
    # baseline，无直接好坏方向。
    assigned_label_q_values: tuple[float, ...] = ()

class Phase2DemonstrationConsistencyMetrics(PydanticBaseModel):
    """Layer 3 demonstration consistency raw metrics。"""

    # selected code 等于 Phase I assigned label 的比例。用途：衡量 selector 对
    # demonstration 先验的保留程度；方向：区间约束，过低表示漂移，过高表示只复制。
    label_match_rate: float

    # selector softmax policy 对 assigned label 的 cross entropy。用途：衡量和
    # demonstration label 的距离；方向：越小越贴近 assigned label，但过小可能缺少选择价值。
    cross_entropy_to_assigned: float

    # 与 assigned one-hot label 的 KL/CE 等价指标。用途：和 imitation regularization
    # 口径对齐；方向：越小越贴近 assigned label，但不是单独越小越好。
    kl_to_assigned_onehot: float

    # selected code Q value - assigned label Q value 的均值。用途：判断偏离 label
    # 是否有 Q-value 支持；方向：越大越好。
    label_q_margin: float

    # 偏离 assigned label 且 selector_return > assigned_return 的样本比例。用途：
    # 衡量有收益证明的偏离；方向：越大越好。
    profitable_deviation_rate: float

    # 偏离 assigned label 且 selector_return < assigned_return 的样本比例。用途：
    # 衡量危险漂移；方向：越小越好。
    unprofitable_deviation_rate: float

    # 仅在偏离样本上的 selector_return - assigned_return 平均值。用途：衡量偏离
    # 的平均收益质量；方向：越大越好。
    deviation_return_delta: float

class Phase2DemonstrationConsistencyThresholds(PydanticBaseModel):
    """Layer 3 demonstration consistency 阈值配置。"""

    # label_match_rate 下限。方向：过低不好。
    label_match_rate_min: float = 0.20

    # label_match_rate 上限。方向：过高可能退化为复制 assigned label。
    label_match_rate_max: float = 0.90

    # unprofitable_deviation_rate 上限。方向：越小越好。
    unprofitable_deviation_rate_max: float = 0.25

    # profitable_deviation_rate warning 下限。方向：越大越好。
    profitable_deviation_rate_warn_min: float = 0.20

    # deviation_return_delta warning 下限。方向：越大越好。
    deviation_return_delta_warn_min: float = 0.0

    # label_q_margin warning 下限。方向：越大越好。
    label_q_margin_warn_min: float = 0.0

def evaluate_demonstration_consistency_rules(
    metrics: Phase2DemonstrationConsistencyMetrics,
    thresholds: Phase2DemonstrationConsistencyThresholds,
) -> Phase2LayerResult:
    """构造 Layer 3 hard gate/warn 结果。"""

    layer = "demonstration_consistency"
    results = (
        _between(
            name="label_match_rate",
            value=metrics.label_match_rate,
            lower=thresholds.label_match_rate_min,
            upper=thresholds.label_match_rate_max,
            layer=layer,
            message="selector 需要保留 Phase I label 先验，但不应完全退化为复现 assigned label",
        ),
        _le(
            name="unprofitable_deviation_rate",
            value=metrics.unprofitable_deviation_rate,
            threshold_value=thresholds.unprofitable_deviation_rate_max,
            layer=layer,
            message="偏离 assigned label 且亏于 KL baseline 的比例不能过高",
        ),
        _ge(
            name="profitable_deviation_rate",
            value=metrics.profitable_deviation_rate,
            threshold_value=thresholds.profitable_deviation_rate_warn_min,
            layer=layer,
            message="有收益证明的偏离比例过低时，Phase II 选择价值不足",
            severity_when_failed="warn",
        ),
        _ge(
            name="deviation_return_delta",
            value=metrics.deviation_return_delta,
            threshold_value=thresholds.deviation_return_delta_warn_min,
            layer=layer,
            message="偏离 assigned label 的样本平均收益差应非负",
            severity_when_failed="warn",
        ),
        _ge(
            name="label_q_margin",
            value=metrics.label_q_margin,
            threshold_value=thresholds.label_q_margin_warn_min,
            layer=layer,
            message="selected code 相对 assigned label 的 Q 值应有优势",
            severity_when_failed="warn",
        ),
    )
    return _build_layer_result(layer_id=3, name=layer, metrics=results)


__all__ = [
    "Phase2DemonstrationConsistencyMetrics",
    "Phase2DemonstrationConsistencyPayload",
    "Phase2DemonstrationConsistencyThresholds",
    "evaluate_demonstration_consistency_rules",
]
