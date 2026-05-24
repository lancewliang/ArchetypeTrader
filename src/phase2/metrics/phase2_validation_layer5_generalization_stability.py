"""Phase II layer 5 generalization, stability, and predictability metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from src.utils import _dataclass_from_mapping

from .phase2_metric_results import Phase2LayerResult
from .phase2_validation_rule_helpers import (
    _build_layer_result,
    _ge,
    _le,
)


@dataclass(frozen=True)
class Phase2PredictabilityPayload:
    """Selector 可预测性 raw metrics 计算的中间 payload。

    该 payload 保存 probe 训练诊断、confusion matrix 和随机种子。它只用于
    计算可预测性 raw metrics，不要求报告保存逐样本预测结果。
    """

    probe_train_accuracy: float
    probe_validation_accuracy: float
    probe_predictability_gap: float
    probe_confusion_matrix: tuple[tuple[int, ...], ...]
    probe_seed: int

    def __post_init__(self) -> None:
        """标准化 payload 字段。"""

        object.__setattr__(self, "probe_train_accuracy", float(self.probe_train_accuracy))
        object.__setattr__(
            self,
            "probe_validation_accuracy",
            float(self.probe_validation_accuracy),
        )
        object.__setattr__(
            self,
            "probe_predictability_gap",
            float(self.probe_predictability_gap),
        )
        object.__setattr__(
            self,
            "probe_confusion_matrix",
            tuple(
                tuple(int(value) for value in row)
                for row in self.probe_confusion_matrix
            ),
        )
        object.__setattr__(self, "probe_seed", int(self.probe_seed))

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return {
            "probe_train_accuracy": self.probe_train_accuracy,
            "probe_validation_accuracy": self.probe_validation_accuracy,
            "probe_predictability_gap": self.probe_predictability_gap,
            "probe_confusion_matrix": [
                [int(value) for value in row]
                for row in self.probe_confusion_matrix
            ],
            "probe_seed": self.probe_seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2PredictabilityPayload":
        """从 dict 恢复可预测性 payload。"""

        return cls(
            probe_train_accuracy=float(payload["probe_train_accuracy"]),
            probe_validation_accuracy=float(payload["probe_validation_accuracy"]),
            probe_predictability_gap=float(payload["probe_predictability_gap"]),
            probe_confusion_matrix=tuple(
                tuple(int(value) for value in row)
                for row in payload.get("probe_confusion_matrix", ())
            ),
            probe_seed=int(payload["probe_seed"]),
        )


@dataclass(frozen=True)
class Phase2PredictabilityMetrics:
    """Selector 可预测性 raw metrics。"""

    probe_top1_accuracy: float
    probe_top3_accuracy: float
    probe_balanced_accuracy: float
    selected_code_entropy_given_morphology: float
    selected_code_entropy: float
    mutual_information_lift: float

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2PredictabilityMetrics":
        """从 dict 恢复 metrics。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase2PredictabilityThresholds:
    """Selector 可预测性阈值配置。"""

    probe_top1_floor: float = 0.25
    probe_top1_k_factor: float = 1.5
    probe_top3_floor: float = 0.55
    probe_top3_k_factor: float = 3.0
    probe_balanced_accuracy_min: float = 0.25
    mutual_information_lift_min: float = 2.0
    entropy_given_morphology_max_ratio: float = 0.85

    def top1_threshold(self, num_archetypes: int) -> float:
        """返回 codebook size 自适应 top-1 阈值。"""

        return max(self.probe_top1_floor, self.probe_top1_k_factor / num_archetypes)

    def top3_threshold(self, num_archetypes: int) -> float:
        """返回 codebook size 自适应 top-3 阈值。"""

        return max(self.probe_top3_floor, self.probe_top3_k_factor / num_archetypes)

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2PredictabilityThresholds":
        """从 dict 恢复 thresholds。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase2GeneralizationStabilityPayload:
    """Layer 5 raw metrics 计算的中间 payload。"""

    train_score: float | None = None
    validation_score_history: tuple[float, ...] = ()
    selected_action_churn_history: tuple[float, ...] = ()
    q_value_scale_history: tuple[float, ...] = ()
    predictability_payload: Phase2PredictabilityPayload | None = None

    def __post_init__(self) -> None:
        """标准化 history 字段。"""

        for field_name in (
            "validation_score_history",
            "selected_action_churn_history",
            "q_value_scale_history",
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(float(value) for value in getattr(self, field_name)),
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return {
            "train_score": self.train_score,
            "validation_score_history": list(self.validation_score_history),
            "selected_action_churn_history": list(
                self.selected_action_churn_history
            ),
            "q_value_scale_history": list(self.q_value_scale_history),
            "predictability_payload": (
                self.predictability_payload.to_dict()
                if self.predictability_payload is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase2GeneralizationStabilityPayload":
        """从 dict 恢复 payload。"""

        predictability_payload = payload.get("predictability_payload")
        return cls(
            train_score=(
                float(score) if (score := payload.get("train_score")) is not None else None
            ),
            validation_score_history=tuple(
                float(v) for v in payload.get("validation_score_history", ())
            ),
            selected_action_churn_history=tuple(
                float(v) for v in payload.get("selected_action_churn_history", ())
            ),
            q_value_scale_history=tuple(
                float(v) for v in payload.get("q_value_scale_history", ())
            ),
            predictability_payload=(
                Phase2PredictabilityPayload.from_dict(predictability_payload)
                if isinstance(predictability_payload, Mapping)
                else None
            ),
        )


@dataclass(frozen=True)
class Phase2GeneralizationStabilityMetrics:
    """Layer 5 generalization and stability raw metrics。"""

    train_val_return_gap: float
    val_test_return_gap: float
    train_val_usage_kl: float
    validation_score_churn: float
    selected_action_churn: float
    q_value_scale_mean: float
    q_value_scale_std: float
    q_margin_mean: float
    low_confidence_selection_rate: float
    td_loss_trend: float
    imitation_loss_trend: float
    reward_mean_trend: float
    predictability: Phase2PredictabilityMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        payload = asdict(self)
        if self.predictability is not None:
            payload["predictability"] = self.predictability.to_dict()
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase2GeneralizationStabilityMetrics":
        """从 dict 恢复 metrics。"""

        base_payload = dict(payload)
        predictability_payload = base_payload.pop("predictability", None)
        return cls(
            **base_payload,
            predictability=(
                Phase2PredictabilityMetrics.from_dict(predictability_payload)
                if isinstance(predictability_payload, Mapping)
                else None
            ),
        )


@dataclass(frozen=True)
class Phase2GeneralizationStabilityThresholds:
    """Layer 5 generalization and stability 阈值配置。"""

    train_val_return_gap_warn_max: float = 0.50
    train_val_usage_kl_warn_max: float = 0.50
    validation_score_churn_warn_max: float = 0.15
    selected_action_churn_warn_max: float = 0.35
    q_value_scale_mean_warn_max: float = 100.0
    q_value_scale_std_warn_max: float = 100.0
    q_margin_mean_warn_min: float = 0.10
    low_confidence_selection_rate_warn_max: float = 0.40
    predictability_thresholds: Phase2PredictabilityThresholds = field(
        default_factory=Phase2PredictabilityThresholds
    )

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        payload = asdict(self)
        payload["predictability_thresholds"] = self.predictability_thresholds.to_dict()
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase2GeneralizationStabilityThresholds":
        """从 dict 恢复 thresholds。"""

        base_payload = dict(payload)
        predictability_payload = base_payload.pop("predictability_thresholds", None)
        return cls(
            **base_payload,
            predictability_thresholds=(
                Phase2PredictabilityThresholds.from_dict(predictability_payload)
                if isinstance(predictability_payload, Mapping)
                else Phase2PredictabilityThresholds()
            ),
        )


def evaluate_generalization_stability_rules(
    metrics: Phase2GeneralizationStabilityMetrics,
    thresholds: Phase2GeneralizationStabilityThresholds,
    *,
    num_archetypes: int,
) -> Phase2LayerResult:
    """构造 Layer 5 warning/reference 结果。"""

    layer = "generalization_stability"
    results = [
        _le(
            name="train_val_return_gap",
            value=metrics.train_val_return_gap,
            threshold_value=thresholds.train_val_return_gap_warn_max,
            layer=layer,
            message="train 明显好于 validation 时标记过拟合风险",
            severity_when_failed="warn",
        ),
        _le(
            name="train_val_usage_kl",
            value=metrics.train_val_usage_kl,
            threshold_value=thresholds.train_val_usage_kl_warn_max,
            layer=layer,
            message="train/validation selected code 分布差异过大时需要解释",
            severity_when_failed="warn",
        ),
        _le(
            name="validation_score_churn",
            value=metrics.validation_score_churn,
            threshold_value=thresholds.validation_score_churn_warn_max,
            layer=layer,
            message="validation score 波动过大时，最高点 checkpoint 可能不稳定",
            severity_when_failed="warn",
        ),
        _le(
            name="selected_action_churn",
            value=metrics.selected_action_churn,
            threshold_value=thresholds.selected_action_churn_warn_max,
            layer=layer,
            message="同一样本跨 epoch selected code 频繁变化表示决策边界不稳",
            severity_when_failed="warn",
        ),
        _le(
            name="q_value_scale_mean",
            value=metrics.q_value_scale_mean,
            threshold_value=thresholds.q_value_scale_mean_warn_max,
            layer=layer,
            message="Q value 均值尺度过大可能表示 overestimation",
            severity_when_failed="warn",
        ),
        _le(
            name="q_value_scale_std",
            value=metrics.q_value_scale_std,
            threshold_value=thresholds.q_value_scale_std_warn_max,
            layer=layer,
            message="Q value 方差过大表示估值不稳定",
            severity_when_failed="warn",
        ),
        _ge(
            name="q_margin_mean",
            value=metrics.q_margin_mean,
            threshold_value=thresholds.q_margin_mean_warn_min,
            layer=layer,
            message="top1/top2 Q margin 太低表示选择置信度不足",
            severity_when_failed="warn",
        ),
        _le(
            name="low_confidence_selection_rate",
            value=metrics.low_confidence_selection_rate,
            threshold_value=thresholds.low_confidence_selection_rate_warn_max,
            layer=layer,
            message="低置信选择比例过高会降低 checkpoint 稳定性",
            severity_when_failed="warn",
        ),
    ]
    if metrics.predictability is not None:
        results.extend(
            _predictability_results(
                metrics.predictability,
                thresholds.predictability_thresholds,
                layer=layer,
                num_archetypes=num_archetypes,
            )
        )
    return _build_layer_result(
        layer_id=5,
        name=layer,
        metrics=tuple(results),
        force_passed=True,
    )


def _predictability_results(
    metrics: Phase2PredictabilityMetrics,
    thresholds: Phase2PredictabilityThresholds,
    *,
    layer: str,
    num_archetypes: int,
) -> tuple[Any, ...]:
    """构造可预测性 reference/warn 指标结果。"""

    entropy_threshold = (
        metrics.selected_code_entropy
        * thresholds.entropy_given_morphology_max_ratio
    )
    return (
        _ge(
            name="predictability_probe_top1_accuracy",
            value=metrics.probe_top1_accuracy,
            threshold_value=thresholds.top1_threshold(num_archetypes),
            layer=layer,
            message="probe top-1 accuracy 用于参考 selector action 是否可由可见状态预测",
            severity_when_failed="warn",
        ),
        _ge(
            name="predictability_probe_top3_accuracy",
            value=metrics.probe_top3_accuracy,
            threshold_value=thresholds.top3_threshold(num_archetypes),
            layer=layer,
            message="probe top-3 accuracy 用于参考 selector 是否缩小了候选 code 范围",
            severity_when_failed="warn",
        ),
        _ge(
            name="predictability_probe_balanced_accuracy",
            value=metrics.probe_balanced_accuracy,
            threshold_value=thresholds.probe_balanced_accuracy_min,
            layer=layer,
            message="balanced accuracy 用于检查 probe 是否只预测高频 selected code",
            severity_when_failed="warn",
        ),
        _le(
            name="selected_code_entropy_given_morphology",
            value=metrics.selected_code_entropy_given_morphology,
            threshold_value=entropy_threshold,
            layer=layer,
            message="给定 morphology 后 selected code 条件熵应下降",
            severity_when_failed="warn",
        ),
        _ge(
            name="predictability_mutual_information_lift",
            value=metrics.mutual_information_lift,
            threshold_value=thresholds.mutual_information_lift_min,
            layer=layer,
            message="mutual information lift 用于参考 selected code 与可见状态的关系",
            severity_when_failed="warn",
        ),
    )


__all__ = [
    "Phase2GeneralizationStabilityMetrics",
    "Phase2GeneralizationStabilityPayload",
    "Phase2GeneralizationStabilityThresholds",
    "Phase2PredictabilityMetrics",
    "Phase2PredictabilityPayload",
    "Phase2PredictabilityThresholds",
    "evaluate_generalization_stability_rules",
]
