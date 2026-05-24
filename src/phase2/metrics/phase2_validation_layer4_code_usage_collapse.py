"""Phase II layer 4 code usage and collapse metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.utils import _dataclass_from_mapping

from .phase2_metric_results import Phase2LayerResult
from .phase2_validation_rule_helpers import (
    _build_layer_result,
    _ge,
    _le,
)


@dataclass(frozen=True)
class Phase2PerCodeUsageDiagnostic:
    """Layer 4 per-code usage/profitability diagnostic row。"""

    code_id: int
    selector_count: int
    selector_ratio: float
    kl_count: int
    kl_ratio: float
    selector_mean_return: float
    kl_mean_return: float
    uplift_vs_kl: float
    is_active: bool
    is_dead_profitable: bool

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2PerCodeUsageDiagnostic":
        """从 dict 恢复 diagnostic row。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase2CodeUsageCollapsePayload:
    """Layer 4 raw metrics 计算的中间 payload。"""

    selected_code_ids: tuple[int, ...] = ()
    assigned_code_labels: tuple[int, ...] = ()
    per_code_diagnostics: tuple[Phase2PerCodeUsageDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        """标准化 payload 字段。"""

        object.__setattr__(
            self,
            "selected_code_ids",
            tuple(int(value) for value in self.selected_code_ids),
        )
        object.__setattr__(
            self,
            "assigned_code_labels",
            tuple(int(value) for value in self.assigned_code_labels),
        )
        object.__setattr__(
            self,
            "per_code_diagnostics",
            tuple(
                item
                if isinstance(item, Phase2PerCodeUsageDiagnostic)
                else Phase2PerCodeUsageDiagnostic.from_dict(item)
                for item in self.per_code_diagnostics
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return {
            "selected_code_ids": list(self.selected_code_ids),
            "assigned_code_labels": list(self.assigned_code_labels),
            "per_code_diagnostics": [
                item.to_dict() for item in self.per_code_diagnostics
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2CodeUsageCollapsePayload":
        """从 dict 恢复 payload。"""

        return cls(
            selected_code_ids=tuple(int(v) for v in payload.get("selected_code_ids", ())),
            assigned_code_labels=tuple(
                int(v) for v in payload.get("assigned_code_labels", ())
            ),
            per_code_diagnostics=tuple(
                Phase2PerCodeUsageDiagnostic.from_dict(item)
                for item in payload.get("per_code_diagnostics", ())
            ),
        )


@dataclass(frozen=True)
class Phase2CodeUsageCollapseMetrics:
    """Layer 4 code usage and collapse raw metrics。"""

    selected_code_entropy: float
    selected_code_perplexity: float
    active_code_count: int
    max_code_usage_ratio: float
    min_code_usage_ratio: float
    usage_kl_to_train_label_distribution: float
    usage_kl_to_val_label_distribution: float
    dead_profitable_code_count: int
    min_per_code_sample_count: int

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2CodeUsageCollapseMetrics":
        """从 dict 恢复 metrics。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase2CodeUsageCollapseThresholds:
    """Layer 4 code usage and collapse 阈值配置。"""

    active_code_count_min: int = 3
    active_code_ratio_min: float = 0.40
    selected_code_entropy_min: float = 1.10
    selected_code_perplexity_min: float = 3.0
    max_code_usage_ratio_max: float = 0.60
    usage_kl_to_val_label_distribution_warn_max: float = 0.50
    dead_profitable_code_count_warn_max: int = 1
    per_code_sample_count_reference_min: int = 30

    def minimum_active_codes(self, num_archetypes: int) -> int:
        """根据 codebook size 返回 active code 下限。"""

        return max(
            self.active_code_count_min,
            int(round(float(num_archetypes) * self.active_code_ratio_min)),
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2CodeUsageCollapseThresholds":
        """从 dict 恢复 thresholds。"""

        return _dataclass_from_mapping(cls, payload)


def evaluate_code_usage_collapse_rules(
    metrics: Phase2CodeUsageCollapseMetrics,
    thresholds: Phase2CodeUsageCollapseThresholds,
    *,
    num_archetypes: int,
) -> Phase2LayerResult:
    """构造 Layer 4 hard gate/warn 结果。"""

    layer = "code_usage_collapse"
    active_code_count_min = thresholds.minimum_active_codes(num_archetypes)
    results = (
        _ge(
            name="active_code_count",
            value=float(metrics.active_code_count),
            threshold_value=float(active_code_count_min),
            layer=layer,
            message="selector 需要使用足够数量的 archetype，避免 code collapse",
        ),
        _ge(
            name="selected_code_entropy",
            value=metrics.selected_code_entropy,
            threshold_value=thresholds.selected_code_entropy_min,
            layer=layer,
            message="selected code 分布熵过低表示选择塌缩",
        ),
        _ge(
            name="selected_code_perplexity",
            value=metrics.selected_code_perplexity,
            threshold_value=thresholds.selected_code_perplexity_min,
            layer=layer,
            message="等效使用 code 数过低时需要诊断",
            severity_when_failed="warn",
        ),
        _le(
            name="max_code_usage_ratio",
            value=metrics.max_code_usage_ratio,
            threshold_value=thresholds.max_code_usage_ratio_max,
            layer=layer,
            message="单个 code 不能支配大多数样本",
        ),
        _le(
            name="usage_kl_to_val_label_distribution",
            value=metrics.usage_kl_to_val_label_distribution,
            threshold_value=thresholds.usage_kl_to_val_label_distribution_warn_max,
            layer=layer,
            message="selector code 分布相对 KL label 分布偏离过大时需要解释",
            severity_when_failed="warn",
        ),
        _le(
            name="dead_profitable_code_count",
            value=metrics.dead_profitable_code_count,
            threshold_value=thresholds.dead_profitable_code_count_warn_max,
            layer=layer,
            message="盈利 code 被 selector 忽略时需要诊断",
            severity_when_failed="warn",
        ),
        _ge(
            name="min_per_code_sample_count",
            value=float(metrics.min_per_code_sample_count),
            threshold_value=float(thresholds.per_code_sample_count_reference_min),
            layer=layer,
            message="低 support code 的 per-code return 只作为参考",
            severity_when_failed="warn",
        ),
    )
    return _build_layer_result(layer_id=4, name=layer, metrics=results)


__all__ = [
    "Phase2CodeUsageCollapseMetrics",
    "Phase2CodeUsageCollapsePayload",
    "Phase2CodeUsageCollapseThresholds",
    "Phase2PerCodeUsageDiagnostic",
    "evaluate_code_usage_collapse_rules",
]
