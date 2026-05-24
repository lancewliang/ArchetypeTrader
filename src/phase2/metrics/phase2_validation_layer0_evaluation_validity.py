"""Phase II layer 0 evaluation validity metrics.

Layer 0 只回答 validation/test 结果是否可信，不评价 selector 好坏。任何
hard gate 失败都表示该 checkpoint 的评估结果不可用于 best checkpoint selection。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.utils import _dataclass_from_mapping

from .phase2_metric_results import Phase2LayerResult
from .phase2_validation_rule_helpers import (
    _build_layer_result,
    _eq_bool,
    _ge,
)


@dataclass(frozen=True)
class Phase2EvaluationValidityPayload:
    """Layer 0 raw metrics 计算的中间 payload。"""

    # 当前评估 split 名称，例如 validation/test。用途：落盘路径和 report 分组；
    # 方向：过程数据，无好坏方向。
    split_name: str

    # 当前评估对应 epoch。用途：关联 model checkpoint；方向：过程数据，无好坏方向。
    epoch: int | None

    # 当前 split 参与评估的样本数。用途：计算样本充分性；方向：越大越可靠，
    # 但只通过 num_samples metric 判断是否达到最低门槛。
    num_samples: int

    # rollout 失败样本数。用途：推导 valid_rollout_ratio；方向：越小越好，理想为 0。
    failed_rollout_count: int

    # reward/fee/turnover 等关键数值非有限样本数。用途：推导 finite_reward_ratio；
    # 方向：越小越好，理想为 0。
    non_finite_reward_count: int

    # selected code 不在 [0, K) 范围内的样本数。用途：推导 valid_selected_code_ratio；
    # 方向：越小越好，理想为 0。
    invalid_selected_code_count: int

    # Phase I codebook size K。用途：校验 selected code 合法范围；方向：过程数据，
    # 不直接作为好坏排序。
    num_archetypes: int

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2EvaluationValidityPayload":
        """从 dict 恢复 payload。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase2EvaluationValidityMetrics:
    """Layer 0 evaluation validity raw metrics。"""

    # 当前 split 参与评估的样本数。用途：确认收益、分位数和 per-code 统计有足够
    # 支撑；方向：越大越可靠，必须不低于 min_eval_samples。
    num_samples: int

    # 成功完成 selector、decoder 和 execution 的样本比例。用途：确认评估完整性；
    # 方向：越大越好，正式评估应为 1.0。
    valid_rollout_ratio: float

    # reward/gross return/fee/turnover 为有限数值的比例。用途：防止 NaN/inf 污染
    # 均值和排序；方向：越大越好，正式评估应为 1.0。
    finite_reward_ratio: float

    # selected code 落在合法 codebook 范围内的比例。用途：检查 selector 输出和
    # 后处理是否有效；方向：越大越好，正式评估应为 1.0。
    valid_selected_code_ratio: float

    # validation/test 是否使用 deterministic greedy action。用途：确保 checkpoint
    # selection 不受 exploration 影响；方向：必须等于 True。
    deterministic_eval: bool

    # sample_id/code_label 是否和 horizon dataset 对齐。用途：避免 baseline 和
    # consistency 指标失真；方向：必须等于 True。
    label_alignment_valid: bool

    # selector observation 是否只包含在线可见信息。用途：防止未来信息泄露；
    # 方向：必须等于 True。
    visible_state_contract_valid: bool

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2EvaluationValidityMetrics":
        """从 dict 恢复 metrics。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase2EvaluationValidityThresholds:
    """Layer 0 evaluation validity 阈值配置。"""

    # 最小评估样本数。用途：过滤统计不稳定的评估；方向：num_samples 越大越好。
    min_eval_samples: int = 500

    # rollout 成功比例下限。用途：完整性 hard gate；方向：越大越好。
    valid_rollout_ratio_min: float = 1.0

    # finite reward 比例下限。用途：数值有效性 hard gate；方向：越大越好。
    finite_reward_ratio_min: float = 1.0

    # legal selected code 比例下限。用途：selector 输出合法性 hard gate；方向：
    # 越大越好。
    valid_selected_code_ratio_min: float = 1.0

    # 是否要求 deterministic eval。用途：checkpoint selection 契约；方向：
    # deterministic_eval 必须等于该值。
    deterministic_eval_required: bool = True

    # 是否要求 label 对齐。用途：baseline/consistency 契约；方向：
    # label_alignment_valid 必须等于该值。
    label_alignment_required: bool = True

    # 是否要求 visible state 契约有效。用途：泄露防护；方向：
    # visible_state_contract_valid 必须等于该值。
    visible_state_contract_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase2EvaluationValidityThresholds":
        """从 dict 恢复 thresholds。"""

        return _dataclass_from_mapping(cls, payload)


def evaluate_evaluation_validity_rules(
    metrics: Phase2EvaluationValidityMetrics,
    thresholds: Phase2EvaluationValidityThresholds,
) -> Phase2LayerResult:
    """构造 Layer 0 hard gate 结果。"""

    layer = "evaluation_validity"
    results = (
        _ge(
            name="num_samples",
            value=float(metrics.num_samples),
            threshold_value=float(thresholds.min_eval_samples),
            layer=layer,
            message="评估样本数需要足够支撑收益、分位数和 per-code 统计",
        ),
        _ge(
            name="valid_rollout_ratio",
            value=metrics.valid_rollout_ratio,
            threshold_value=thresholds.valid_rollout_ratio_min,
            layer=layer,
            message="所有样本都应成功完成 selector、decoder 和 execution",
        ),
        _ge(
            name="finite_reward_ratio",
            value=metrics.finite_reward_ratio,
            threshold_value=thresholds.finite_reward_ratio_min,
            layer=layer,
            message="reward、fee、turnover 不允许出现 NaN 或 inf",
        ),
        _ge(
            name="valid_selected_code_ratio",
            value=metrics.valid_selected_code_ratio,
            threshold_value=thresholds.valid_selected_code_ratio_min,
            layer=layer,
            message="selected code 必须全部落在合法 codebook 范围内",
        ),
        _eq_bool(
            name="deterministic_eval",
            value=metrics.deterministic_eval,
            expected=thresholds.deterministic_eval_required,
            layer=layer,
            message="validation/test checkpoint selection 必须使用 deterministic greedy action",
        ),
        _eq_bool(
            name="label_alignment_valid",
            value=metrics.label_alignment_valid,
            expected=thresholds.label_alignment_required,
            layer=layer,
            message="Phase I assigned label 必须和 horizon sample 对齐",
        ),
        _eq_bool(
            name="visible_state_contract_valid",
            value=metrics.visible_state_contract_valid,
            expected=thresholds.visible_state_contract_required,
            layer=layer,
            message="selector observation 不能混入当前 horizon 未来信息",
        ),
    )
    return _build_layer_result(layer_id=0, name=layer, metrics=results)


__all__ = [
    "Phase2EvaluationValidityMetrics",
    "Phase2EvaluationValidityPayload",
    "Phase2EvaluationValidityThresholds",
    "evaluate_evaluation_validity_rules",
]
