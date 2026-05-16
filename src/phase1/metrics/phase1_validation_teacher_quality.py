"""Phase I layer 0 teacher quality schema, thresholds, and hard gate rules."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from src.utils import _dataclass_from_mapping

if TYPE_CHECKING:
    from .phase1_metric_results import Phase1LayerResult
    from .phase1_validation_data_schema import Phase1ValidationMetrics


@dataclass(frozen=True)
class Phase1TeacherQualityPayload(Mapping[str, object]):
    """第零层 teacher quality 计算的中间 payload。

    使用场景:
        保存 DP return、flat baseline return、逐样本 advantage 以及不可计算原因。
        该对象实现 ``Mapping``，用于兼容现有 ``extra_payload["..."]`` 调用。
    """

    # 每条 horizon 的 DP teacher return。
    dp_returns: tuple[float, ...]

    # 每条 horizon 的 flat baseline return。
    flat_returns: tuple[float, ...]

    # 每条 horizon 的 DP teacher 相对 flat baseline advantage。
    advantages: tuple[float, ...]

    # 中间指标不可计算原因；输入完整时为 None。
    missing_reason: str | None

    def __post_init__(self) -> None:
        """标准化 payload 中的序列和字符串类型。"""

        object.__setattr__(
            self,
            "dp_returns",
            tuple(float(value) for value in self.dp_returns),
        )
        object.__setattr__(
            self,
            "flat_returns",
            tuple(float(value) for value in self.flat_returns),
        )
        object.__setattr__(
            self,
            "advantages",
            tuple(float(value) for value in self.advantages),
        )
        object.__setattr__(
            self,
            "missing_reason",
            None if self.missing_reason is None else str(self.missing_reason),
        )

    def _mapping(self) -> dict[str, object]:
        """返回兼容旧 ``extra_payload`` 字典访问的视图。"""

        return {
            "dp_returns": self.dp_returns,
            "flat_returns": self.flat_returns,
            "advantages": self.advantages,
            "missing_reason": self.missing_reason,
        }

    def __getitem__(self, key: str) -> object:
        """按旧 payload key 读取属性值。"""

        return self._mapping()[key]

    def __iter__(self) -> Iterator[str]:
        """迭代旧 payload key。"""

        return iter(self._mapping())

    def __len__(self) -> int:
        """返回 payload key 数量。"""

        return len(self._mapping())

    def to_dict(self) -> dict[str, Any]:
        """序列化为可落盘 dict。"""

        return {
            "dp_returns": [float(value) for value in self.dp_returns],
            "flat_returns": [float(value) for value in self.flat_returns],
            "advantages": [float(value) for value in self.advantages],
            "missing_reason": self.missing_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1TeacherQualityPayload":
        """从 dict 恢复第零层 teacher quality payload。"""

        return cls(
            dp_returns=tuple(float(value) for value in payload.get("dp_returns", ())),
            flat_returns=tuple(
                float(value) for value in payload.get("flat_returns", ())
            ),
            advantages=tuple(float(value) for value in payload.get("advantages", ())),
            missing_reason=payload.get("missing_reason"),
        )


@dataclass(frozen=True)
class Phase1TeacherQualityMetrics:
    """第零层 DP teacher 质量 raw metrics。

    使用场景:
        输入 ``evaluate_teacher_quality_rules()`` 做 hard gate 判定，并参与
        teacher quality normalized score。
    """

    # DP teacher 相对 flat baseline 的平均优势。
    dp_advantage_vs_flat: float

    # DP teacher return 大于 flat baseline 的样本比例。
    dp_win_rate_vs_flat: float

    # DP teacher 优势接近手续费噪声的样本比例。
    near_zero_opportunity_ratio: float

    # 手续费变高后 DP teacher 总优势的保留比例。
    fee_sensitivity: float

    # 非 neutral 市场形态样本覆盖率。
    morphology_coverage: float

    # 去掉收益最高 top 5% 后剩余 DP 总优势或优势保留诊断值。
    dp_return_concentration_after_top5_removed: float

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict，供 checkpoint/report 落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1TeacherQualityMetrics":
        """从 dict 恢复第零层 metrics。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase1TeacherQualityThresholds:
    """第零层 DP teacher 质量阈值配置。

    功能说明:
        保存用于判断 DP 示范数据是否值得学习的 hard gate 阈值。

    使用场景:
        由 ``evaluate_teacher_quality_rules()`` 消费。第零层失败通常表示应
        重新构造 teacher 数据，而不是继续挑选 VQ checkpoint。
    """

    # DP teacher 胜率下限。用于第零层判断 DP 示范是否广泛优于 flat baseline。
    dp_win_rate_min: float = 0.58

    # 弱机会样本比例上限。用于过滤大量收益接近手续费噪声的 teacher 数据。
    near_zero_opportunity_ratio_max: float = 0.35

    # 手续费翻倍后的收益保留比例下限。用于判断 DP 示范是否过度依赖微小价差。
    fee_sensitivity_min: float = 0.60

    # 非 neutral 市场形态覆盖率下限。用于判断样本是否有足够可学习的市场结构。
    morphology_coverage_min: float = 0.60

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供 checkpoint、report 或日志落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1TeacherQualityThresholds":
        """从 checkpoint/report 中的 dict 恢复第零层阈值配置。"""

        return _dataclass_from_mapping(cls, payload)


def evaluate_teacher_quality_rules(
    metrics: Phase1TeacherQualityMetrics,
    thresholds: Phase1TeacherQualityThresholds,
) -> Phase1LayerResult:
    """判定第零层 DP teacher 质量。

    使用场景:
        teacher 数据质量不过关时，应优先重构 demonstration 数据，而不是继续
        选择 VQ checkpoint。

    审计问题:
        VQ 模型要学习的是 DP teacher 的行为压缩。如果 teacher 本身只在少数样本
        上赚钱、优势接近手续费噪声、或只覆盖 neutral 市场，那么即使 VQ 重构很好，
        学到的 codebook 也不具备可靠交易含义。
    """

    from .phase1_validation_rule_helpers import _build_layer_result, _ge, _gt, _le

    layer = "teacher_quality"
    results = (
        _gt(
            name="dp_advantage_vs_flat",
            value=metrics.dp_advantage_vs_flat,
            threshold_value=0.0,
            layer=layer,
            message="DP teacher 平均收益必须优于 flat baseline",
        ),
        _ge(
            name="dp_win_rate_vs_flat",
            value=metrics.dp_win_rate_vs_flat,
            threshold_value=thresholds.dp_win_rate_min,
            layer=layer,
            message="DP teacher 胜率必须足够广泛，不能只依赖少数样本",
        ),
        _le(
            name="near_zero_opportunity_ratio",
            value=metrics.near_zero_opportunity_ratio,
            threshold_value=thresholds.near_zero_opportunity_ratio_max,
            layer=layer,
            message="接近手续费噪声的弱机会样本比例不能过高",
        ),
        _ge(
            name="fee_sensitivity",
            value=metrics.fee_sensitivity,
            threshold_value=thresholds.fee_sensitivity_min,
            layer=layer,
            message="手续费上升后 DP teacher 收益需要保留足够比例",
        ),
        _ge(
            name="morphology_coverage",
            value=metrics.morphology_coverage,
            threshold_value=thresholds.morphology_coverage_min,
            layer=layer,
            message="非 neutral 市场结构覆盖率需要足够高",
        ),
        _gt(
            name="dp_return_concentration_after_top5_removed",
            value=metrics.dp_return_concentration_after_top5_removed,
            threshold_value=0.0,
            layer=layer,
            message="去掉收益最高 top 5% 后 DP teacher 仍应保留正优势",
        ),
    )
    return _build_layer_result(layer_id=0, name=layer, metrics=results)


def compute_teacher_quality_score(metrics: Phase1ValidationMetrics) -> float:
    """计算第零层 teacher quality 子分数。"""

    from .phase1_validation_score_helpers import (
        _inverse_ratio_score,
        _positive_score,
        _threshold_progress,
    )

    teacher = metrics.teacher_quality
    parts = (
        _positive_score(teacher.dp_advantage_vs_flat),
        _threshold_progress(teacher.dp_win_rate_vs_flat, 0.58),
        _inverse_ratio_score(teacher.near_zero_opportunity_ratio, 0.35),
        _threshold_progress(teacher.fee_sensitivity, 0.60),
        _threshold_progress(teacher.morphology_coverage, 0.60),
        _positive_score(teacher.dp_return_concentration_after_top5_removed),
    )
    return sum(parts) / len(parts)


__all__ = [
    "compute_teacher_quality_score",
    "Phase1TeacherQualityMetrics",
    "Phase1TeacherQualityPayload",
    "Phase1TeacherQualityThresholds",
    "evaluate_teacher_quality_rules",
]
