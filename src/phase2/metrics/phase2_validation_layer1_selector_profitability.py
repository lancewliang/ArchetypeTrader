"""Phase II layer 1 selector profitability metrics."""

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
class Phase2SelectorProfitabilityPayload:
    """Layer 1 raw metrics 计算的中间 payload。"""

    selector_returns: tuple[float, ...] = ()
    selector_gross_returns: tuple[float, ...] = ()
    selector_fees: tuple[float, ...] = ()
    selector_turnover: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """标准化 tuple 字段。"""

        object.__setattr__(
            self,
            "selector_returns",
            tuple(float(value) for value in self.selector_returns),
        )
        object.__setattr__(
            self,
            "selector_gross_returns",
            tuple(float(value) for value in self.selector_gross_returns),
        )
        object.__setattr__(
            self,
            "selector_fees",
            tuple(float(value) for value in self.selector_fees),
        )
        object.__setattr__(
            self,
            "selector_turnover",
            tuple(float(value) for value in self.selector_turnover),
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return {
            "selector_returns": list(self.selector_returns),
            "selector_gross_returns": list(self.selector_gross_returns),
            "selector_fees": list(self.selector_fees),
            "selector_turnover": list(self.selector_turnover),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2SelectorProfitabilityPayload":
        """从 dict 恢复 payload。"""

        return cls(
            selector_returns=tuple(float(v) for v in payload.get("selector_returns", ())),
            selector_gross_returns=tuple(
                float(v) for v in payload.get("selector_gross_returns", ())
            ),
            selector_fees=tuple(float(v) for v in payload.get("selector_fees", ())),
            selector_turnover=tuple(
                float(v) for v in payload.get("selector_turnover", ())
            ),
        )


@dataclass(frozen=True)
class Phase2SelectorProfitabilityMetrics:
    """Layer 1 selector profitability raw metrics。"""

    mean_return: float
    median_return: float
    total_return: float
    win_rate: float
    sharpe_like: float
    downside_sharpe_like: float
    p05_return: float
    loss_rate: float
    mean_gross_return: float
    mean_fee: float
    fee_drag_ratio: float
    mean_turnover: float

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase2SelectorProfitabilityMetrics":
        """从 dict 恢复 metrics。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase2SelectorProfitabilityThresholds:
    """Layer 1 selector profitability 阈值配置。"""

    mean_return_min: float = 0.0
    median_return_min: float = -0.10
    win_rate_min: float = 0.50
    sharpe_like_min: float = 0.0
    downside_sharpe_like_min: float = 0.0
    p05_return_min: float = -10.0
    fee_drag_ratio_max: float = 0.40
    mean_turnover_max: float = 1.50

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase2SelectorProfitabilityThresholds":
        """从 dict 恢复 thresholds。"""

        return _dataclass_from_mapping(cls, payload)


def evaluate_selector_profitability_rules(
    metrics: Phase2SelectorProfitabilityMetrics,
    thresholds: Phase2SelectorProfitabilityThresholds,
) -> Phase2LayerResult:
    """构造 Layer 1 hard gate/warn 结果。"""

    layer = "selector_profitability"
    results = (
        _gt(
            name="mean_return",
            value=metrics.mean_return,
            threshold_value=thresholds.mean_return_min,
            layer=layer,
            message="selector 平均 horizon return 必须为正",
        ),
        _ge(
            name="median_return",
            value=metrics.median_return,
            threshold_value=thresholds.median_return_min,
            layer=layer,
            message="median return 明显为负时，收益可能依赖少数大盈利样本",
            severity_when_failed="warn",
        ),
        _ge(
            name="win_rate",
            value=metrics.win_rate,
            threshold_value=thresholds.win_rate_min,
            layer=layer,
            message="正收益样本比例需要达到最低要求",
        ),
        _gt(
            name="sharpe_like",
            value=metrics.sharpe_like,
            threshold_value=thresholds.sharpe_like_min,
            layer=layer,
            message="风险调整收益至少需要为正",
        ),
        _gt(
            name="downside_sharpe_like",
            value=metrics.downside_sharpe_like,
            threshold_value=thresholds.downside_sharpe_like_min,
            layer=layer,
            message="下行风险调整收益用于识别左尾亏损压力",
            severity_when_failed="warn",
        ),
        _ge(
            name="p05_return",
            value=metrics.p05_return,
            threshold_value=thresholds.p05_return_min,
            layer=layer,
            message="return 左尾 5% 分位数不能低于风险上限",
        ),
        _le(
            name="fee_drag_ratio",
            value=metrics.fee_drag_ratio,
            threshold_value=thresholds.fee_drag_ratio_max,
            layer=layer,
            message="手续费拖累比例不能过高",
        ),
        _le(
            name="mean_turnover",
            value=metrics.mean_turnover,
            threshold_value=thresholds.mean_turnover_max,
            layer=layer,
            message="平均换手过高会提高成本和滑点风险",
            severity_when_failed="warn",
        ),
    )
    return _build_layer_result(layer_id=1, name=layer, metrics=results)


__all__ = [
    "Phase2SelectorProfitabilityMetrics",
    "Phase2SelectorProfitabilityPayload",
    "Phase2SelectorProfitabilityThresholds",
    "evaluate_selector_profitability_rules",
]
