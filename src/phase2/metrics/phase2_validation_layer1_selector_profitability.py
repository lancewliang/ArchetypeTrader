"""Phase II layer 1 selector profitability metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils import PydanticBaseModel
from .phase2_validation_rule_helpers import (
    _build_layer_result,
    _ge,
    _gt,
    _le,
)

if TYPE_CHECKING:
    from .phase2_metric_results import Phase2LayerResult


class Phase2SelectorProfitabilityPayload(PydanticBaseModel):
    """Layer 1 raw metrics 计算的中间 payload。"""

    # selector greedy policy 的逐样本净 horizon return。用途：聚合 mean/median/win
    # rate/left-tail 等收益指标；方向：序列值越大越好。
    selector_returns: tuple[float, ...] = ()

    # 扣费前逐样本 gross return。用途：区分策略方向收益和手续费拖累；方向：
    # 越大越好，但需结合 fee。
    selector_gross_returns: tuple[float, ...] = ()

    # 逐样本手续费或交易成本。用途：计算 mean_fee 和 fee_drag_ratio；方向：
    # 越小越好。
    selector_fees: tuple[float, ...] = ()

    # 逐样本换手率或行为强度。用途：计算 mean_turnover 并诊断过度交易；方向：
    # 通常越小越稳，但过低可能表示不交易。
    selector_turnover: tuple[float, ...] = ()


class Phase2SelectorProfitabilityMetrics(PydanticBaseModel):
    """Layer 1 selector profitability raw metrics。"""

    # selector 平均净 horizon return。用途：直接衡量策略期望收益；方向：越大越好，
    # hard gate 要求大于 0。
    mean_return: float

    # selector 净 return 中位数。用途：检查收益是否依赖少数大盈利样本；方向：
    # 越大越好。
    median_return: float

    # selector 净 return 累计和。用途：与资金曲线方向对齐；方向：越大越好。
    total_return: float

    # return > 0 的样本比例。用途：衡量收益覆盖面；方向：越大越好。
    win_rate: float

    # mean_return / std_return 的 horizon-level 类 Sharpe。用途：风险调整收益；
    # 方向：越大越好。
    sharpe_like: float

    # mean_return / downside_std 的下行风险调整收益。用途：关注亏损侧波动；
    # 方向：越大越好。
    downside_sharpe_like: float

    # return 的 5% 分位数。用途：衡量左尾风险；方向：越大越好，越低表示尾部亏损
    # 越严重。
    p05_return: float

    # return < 0 的样本比例。用途：和 win_rate 交叉检查亏损覆盖面；方向：
    # 越小越好。
    loss_rate: float

    # 扣费前平均收益。用途：诊断策略方向是否有效；方向：越大越好。
    mean_gross_return: float

    # 平均手续费。用途：诊断交易成本；方向：越小越好。
    mean_fee: float

    # 手续费占 gross return 绝对值的比例。用途：衡量收益被成本吞噬程度；
    # 方向：越小越好。
    fee_drag_ratio: float

    # 平均换手率或行为强度。用途：诊断过度交易和成本风险；方向：通常越小越稳，
    # 但过低需结合 return 判断是否退化。
    mean_turnover: float


class Phase2SelectorProfitabilityThresholds(PydanticBaseModel):
    """Layer 1 selector profitability 阈值配置。"""

    # mean_return 下限。方向：mean_return 越大越好。
    mean_return_min: float = 0.0

    # median_return 下限。方向：median_return 越大越好。
    median_return_min: float = -0.10

    # win_rate 下限。方向：win_rate 越大越好。
    win_rate_min: float = 0.50

    # sharpe_like 下限。方向：sharpe_like 越大越好。
    sharpe_like_min: float = 0.0

    # downside_sharpe_like 下限。方向：downside_sharpe_like 越大越好。
    downside_sharpe_like_min: float = 0.0

    # p05_return 下限。方向：p05_return 越大越好。
    p05_return_min: float = -10.0

    # fee_drag_ratio 上限。方向：fee_drag_ratio 越小越好。
    fee_drag_ratio_max: float = 0.40

    # mean_turnover 上限。方向：mean_turnover 通常越小越稳。
    mean_turnover_max: float = 1.50


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
