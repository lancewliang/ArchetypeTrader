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

    split_name: str
    epoch: int | None
    num_samples: int
    failed_rollout_count: int
    non_finite_reward_count: int
    invalid_selected_code_count: int
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

    num_samples: int
    valid_rollout_ratio: float
    finite_reward_ratio: float
    valid_selected_code_ratio: float
    deterministic_eval: bool
    label_alignment_valid: bool
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

    min_eval_samples: int = 500
    valid_rollout_ratio_min: float = 1.0
    finite_reward_ratio_min: float = 1.0
    valid_selected_code_ratio_min: float = 1.0
    deterministic_eval_required: bool = True
    label_alignment_required: bool = True
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
