"""Phase II layer 2 baseline uplift metrics."""

from __future__ import annotations

from src.utils import PydanticBaseModel

from .phase2_metric_results import Phase2LayerResult
from .phase2_validation_rule_helpers import (
    _build_layer_result,
    _ge,
    _gt,
    _le,
)


class Phase2BaselineUpliftPayload(PydanticBaseModel):
    """Layer 2 baseline uplift raw metrics 计算的中间 payload。"""

    # selector greedy policy 的逐样本净 return。用途：和各 baseline 做样本级/均值
    # 对比；方向：越大越好。
    selector_returns: tuple[float, ...] = ()

    # Phase I assigned-label baseline 的逐样本 return。用途：衡量“只复用 Phase I
    # label”时的收益；方向：作为参考 baseline，越大表示 baseline 越强。
    assigned_label_returns: tuple[float, ...] = ()

    # random code baseline 的逐样本 return。用途：验证 selector 是否优于随机选择；
    # 方向：作为参考 baseline，selector 应高于它。
    random_returns: tuple[float, ...] = ()

    # hindsight oracle code baseline 的逐样本 return。用途：提供未来信息上界；
    # 方向：作为不可部署参考上界，越大表示可捕获机会越多。
    oracle_returns: tuple[float, ...] = ()

    # random baseline 使用的随机种子。用途：复现实验；方向：过程数据，无好坏方向。
    random_seed: int | None = None

class Phase2BaselineUpliftMetrics(PydanticBaseModel):
    """Layer 2 baseline uplift raw metrics。"""

    # assigned-label baseline 平均 return。用途：对比 Phase I label 复用策略；方向：
    # baseline 自身越大越强，但 selector 目标是 uplift 不明显为负或为正。
    assigned_mean_return: float

    # random baseline 平均 return。用途：检查 selector 是否学到有效选择；方向：
    # baseline 自身为参考，selector 应高于它。
    random_mean_return: float

    # hindsight oracle code baseline 平均 return。用途：估计可实现机会的上界；
    # 方向：参考上界越大表示机会越多，不作为可部署目标。
    oracle_mean_return: float

    # selector_mean_return - assigned_mean_return。用途：衡量 Phase II 相对 Phase I
    # assigned label 的增益；方向：越大越好。
    uplift_vs_assigned: float

    # selector_mean_return - random_mean_return。用途：衡量 selector 是否优于随机
    # code selection；方向：越大越好，hard gate 要求为正。
    uplift_vs_random: float

    # uplift_vs_assigned / abs(assigned_mean_return)。用途：跨收益尺度比较相对增益；
    # 方向：越大越好。
    relative_uplift_vs_assigned: float

    # selector_mean_return / oracle_mean_return。用途：衡量捕获 hindsight oracle
    # 上界的比例；方向：越大越好。
    oracle_capture_ratio: float

    # oracle_mean_return - selector_mean_return。用途：衡量离 hindsight 上界的差距；
    # 方向：越小越好。
    regret_to_oracle: float

    # 样本级 selector_return > assigned_return 的比例。用途：检查 uplift 是否广泛；
    # 方向：越大越好。
    beat_assigned_rate: float

    # 样本级 selector_return > random_return 的比例。用途：检查是否多数优于随机；
    # 方向：越大越好。
    beat_random_rate: float

class Phase2BaselineUpliftThresholds(PydanticBaseModel):
    """Layer 2 baseline uplift 阈值配置。"""

    # uplift_vs_random 下限。方向：uplift_vs_random 越大越好。
    uplift_vs_random_min: float = 0.0

    # beat_random_rate 下限。方向：beat_random_rate 越大越好。
    beat_random_rate_min: float = 0.50

    # uplift_vs_assigned 下限。方向：uplift_vs_assigned 越大越好。
    uplift_vs_assigned_min: float = -0.10

    # beat_assigned_rate warning 下限。方向：beat_assigned_rate 越大越好。
    beat_assigned_rate_warn_min: float = 0.48

    # oracle_capture_ratio warning 下限。方向：oracle_capture_ratio 越大越好。
    oracle_capture_ratio_warn_min: float = 0.30

    # regret_to_oracle warning 上限。方向：regret_to_oracle 越小越好。
    regret_to_oracle_warn_max: float = 10.0

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
