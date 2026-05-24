"""Phase II layer 2 baseline uplift metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.utils import _dataclass_from_mapping

from .phase2_metric_results import Phase2LayerResult
from .phase2_validation_rule_helpers import (
    _build_layer_result,
    _ge,
    _gt,
    _le,
)


@dataclass(frozen=True)
class Phase2BaselineUpliftPayload:
    """Layer 2 baseline uplift raw metrics 计算的中间 payload。"""

    selector_returns: tuple[float, ...] = ()
    assigned_label_returns: tuple[float, ...] = ()
    random_returns: tuple[float, ...] = ()
    oracle_returns: tuple[float, ...] = ()
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """标准化收益序列。"""

        for field_name in (
            "selector_returns",
            "assigned_label_returns",
            "random_returns",
            "oracle_returns",
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(float(value) for value in getattr(self, field_name)),
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return {
            "selector_returns": list(self.selector_returns),
            "assigned_label_returns": list(self.assigned_label_returns),
            "random_returns": list(self.random_returns),
            "oracle_returns": list(self.oracle_returns),
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2BaselineUpliftPayload":
        """从 dict 恢复 payload。"""

        return cls(
            selector_returns=tuple(float(v) for v in payload.get("selector_returns", ())),
            assigned_label_returns=tuple(
                float(v) for v in payload.get("assigned_label_returns", ())
            ),
            random_returns=tuple(float(v) for v in payload.get("random_returns", ())),
            oracle_returns=tuple(float(v) for v in payload.get("oracle_returns", ())),
            random_seed=(
                int(seed) if (seed := payload.get("random_seed")) is not None else None
            ),
        )


@dataclass(frozen=True)
class Phase2BaselineUpliftMetrics:
    """Layer 2 baseline uplift raw metrics。"""

    assigned_mean_return: float
    random_mean_return: float
    oracle_mean_return: float
    uplift_vs_assigned: float
    uplift_vs_random: float
    relative_uplift_vs_assigned: float
    oracle_capture_ratio: float
    regret_to_oracle: float
    beat_assigned_rate: float
    beat_random_rate: float

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2BaselineUpliftMetrics":
        """从 dict 恢复 metrics。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase2BaselineUpliftThresholds:
    """Layer 2 baseline uplift 阈值配置。"""

    uplift_vs_random_min: float = 0.0
    beat_random_rate_min: float = 0.50
    uplift_vs_assigned_min: float = -0.10
    beat_assigned_rate_warn_min: float = 0.48
    oracle_capture_ratio_warn_min: float = 0.30
    regret_to_oracle_warn_max: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2BaselineUpliftThresholds":
        """从 dict 恢复 thresholds。"""

        return _dataclass_from_mapping(cls, payload)


def evaluate_baseline_uplift_rules(
    metrics: Phase2BaselineUpliftMetrics,
    thresholds: Phase2BaselineUpliftThresholds,
) -> Phase2LayerResult:
    """构造 Layer 2 hard gate/warn 结果。"""

    layer = "baseline_uplift"
    results = (
        _gt(
            name="uplift_vs_random",
            value=metrics.uplift_vs_random,
            threshold_value=thresholds.uplift_vs_random_min,
            layer=layer,
            message="selector 必须优于 random code baseline",
        ),
        _gt(
            name="beat_random_rate",
            value=metrics.beat_random_rate,
            threshold_value=thresholds.beat_random_rate_min,
            layer=layer,
            message="样本级表现应多数优于 random baseline",
        ),
        _ge(
            name="uplift_vs_assigned",
            value=metrics.uplift_vs_assigned,
            threshold_value=thresholds.uplift_vs_assigned_min,
            layer=layer,
            message="selector 相对 KL/assigned-label baseline 不能明显退化",
        ),
        _ge(
            name="beat_assigned_rate",
            value=metrics.beat_assigned_rate,
            threshold_value=thresholds.beat_assigned_rate_warn_min,
            layer=layer,
            message="样本级 beat assigned rate 过低时，uplift 可能依赖少数尾部样本",
            severity_when_failed="warn",
        ),
        _ge(
            name="oracle_capture_ratio",
            value=metrics.oracle_capture_ratio,
            threshold_value=thresholds.oracle_capture_ratio_warn_min,
            layer=layer,
            message="selector 捕获 hindsight oracle 上界的比例过低时需要诊断",
            severity_when_failed="warn",
        ),
        _le(
            name="regret_to_oracle",
            value=metrics.regret_to_oracle,
            threshold_value=thresholds.regret_to_oracle_warn_max,
            layer=layer,
            message="selector 和 hindsight oracle 的收益差距用于排序参考",
            severity_when_failed="warn",
        ),
    )
    return _build_layer_result(layer_id=2, name=layer, metrics=results)


__all__ = [
    "Phase2BaselineUpliftMetrics",
    "Phase2BaselineUpliftPayload",
    "Phase2BaselineUpliftThresholds",
    "evaluate_baseline_uplift_rules",
]
