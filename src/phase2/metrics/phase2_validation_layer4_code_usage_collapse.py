"""Phase II layer 4 code usage and collapse metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils import PydanticBaseModel
from .phase2_validation_rule_helpers import (
    _build_layer_result,
    _ge,
    _le,
)

if TYPE_CHECKING:
    from .phase2_metric_results import Phase2LayerResult


class Phase2PerCodeUsageDiagnostic(PydanticBaseModel):
    """Layer 4 per-code usage/profitability diagnostic row。"""

    # codebook 中的 archetype id。用途：定位具体 code；方向：无好坏方向。
    code_id: int

    # selector 选择该 code 的样本数。用途：判断 support 和活跃度；方向：诊断字段，
    # 过低表示统计不稳定，过高可能表示 collapse。
    selector_count: int

    # selector 选择该 code 的比例。用途：分析 selected code 分布；方向：诊断字段，
    # 单个 code 过高不好。
    selector_ratio: float

    # assigned-label baseline 中该 code 的样本数。用途：和 selector 使用分布对比；
    # 方向：baseline 支撑度参考，无直接好坏方向。
    kl_count: int

    # assigned-label baseline 中该 code 的比例。用途：计算 usage drift；方向：
    # baseline 分布参考，无直接好坏方向。
    kl_ratio: float

    # selector 选择该 code 时的平均 return。用途：识别该 code 的实际贡献；
    # 方向：越大越好。
    selector_mean_return: float

    # assigned-label baseline 中该 code 的平均 return。用途：判断 selector 是否忽略
    # 原本盈利 code；方向：baseline 自身越大代表该 code 更值得关注。
    kl_mean_return: float

    # selector_mean_return - kl_mean_return。用途：per-code 层面的选择增益；
    # 方向：越大越好。
    uplift_vs_kl: float

    # 该 code 是否达到 active support 阈值。用途：区分 active/inactive code；
    # 方向：True 通常更好，但所有 code 都 active 也可能表示选择缺乏区分度。
    is_active: bool

    # assigned/收益参考中表现好但 selector 几乎不用的 code 标记。用途：定位被
    # 忽略的盈利 archetype；方向：False 更好。
    is_dead_profitable: bool

class Phase2CodeUsageCollapsePayload(PydanticBaseModel):
    """Layer 4 raw metrics 计算的中间 payload。"""

    # selector 实际选择的 code id 序列。用途：计算 entropy/perplexity/usage ratio；
    # 方向：过程数据，无直接好坏方向。
    selected_code_ids: tuple[int, ...] = ()

    # Phase I assigned code label 序列。用途：作为 usage drift 的 baseline 分布；
    # 方向：过程数据，无直接好坏方向。
    assigned_code_labels: tuple[int, ...] = ()

    # per-code 使用和收益诊断。用途：report 展示和 dead profitable code 定位；
    # 方向：由每个 diagnostic 字段定义。
    per_code_diagnostics: tuple[Phase2PerCodeUsageDiagnostic, ...] = ()

class Phase2CodeUsageCollapseMetrics(PydanticBaseModel):
    """Layer 4 code usage and collapse raw metrics。"""

    # selected code 分布熵。用途：检测 code collapse；方向：越大表示使用越分散，
    # 但过高且收益差可能接近随机选择。
    selected_code_entropy: float

    # exp(selected_code_entropy)，等效使用 code 数。用途：直观表示有效 code 数；
    # 方向：通常越大越好，过低表示 collapse。
    selected_code_perplexity: float

    # selector 使用比例超过阈值的 active code 数。用途：衡量 archetype set 利用
    # 程度；方向：越大越好，至少要达到 codebook size 自适应下限。
    active_code_count: int

    # 单个 code 的最大使用比例。用途：检测是否被一个 code 支配；方向：越小越好。
    max_code_usage_ratio: float

    # 单个 code 的最小使用比例。用途：识别长期 unused code；方向：诊断字段，
    # 不直接越大越好，因为部分 code 不适合当前 split 是合理的。
    min_code_usage_ratio: float

    # selected code 分布相对 Phase I train label 分布的 KL。用途：诊断相对训练
    # 先验的使用漂移；方向：通常越小越稳，若收益提升则可接受。
    usage_kl_to_train_label_distribution: float

    # selected code 分布相对当前 validation assigned label 分布的 KL。用途：诊断
    # 相对当前 split label 先验的偏离；方向：越小越稳，偏离需由 uplift 解释。
    usage_kl_to_val_label_distribution: float

    # assigned/oracle 参考中盈利但 selector 几乎不用的 code 数。用途：发现选择
    # 偏置；方向：越小越好。
    dead_profitable_code_count: int

    # active/per-code 统计中的最小样本数。用途：判断 per-code return 可靠性；
    # 方向：越大越可靠。
    min_per_code_sample_count: int

class Phase2CodeUsageCollapseThresholds(PydanticBaseModel):
    """Layer 4 code usage and collapse 阈值配置。"""

    # active code 数固定下限。方向：active_code_count 越大越好。
    active_code_count_min: int = 3

    # active code 数占 K 的比例下限。方向：active_code_count 越大越好。
    active_code_ratio_min: float = 0.40

    # selected_code_entropy 下限。方向：entropy 越大越不容易 collapse。
    selected_code_entropy_min: float = 1.10

    # selected_code_perplexity 下限。方向：perplexity 越大越不容易 collapse。
    selected_code_perplexity_min: float = 3.0

    # max_code_usage_ratio 上限。方向：越小越好。
    max_code_usage_ratio_max: float = 0.60

    # usage_kl_to_val_label_distribution warning 上限。方向：越小越稳。
    usage_kl_to_val_label_distribution_warn_max: float = 0.50

    # dead_profitable_code_count warning 上限。方向：越小越好。
    dead_profitable_code_count_warn_max: int = 1

    # per-code 样本数参考下限。方向：min_per_code_sample_count 越大越可靠。
    per_code_sample_count_reference_min: int = 30

    def minimum_active_codes(self, num_archetypes: int) -> int:
        """根据 codebook size 返回 active code 下限。"""

        return max(
            self.active_code_count_min,
            int(round(float(num_archetypes) * self.active_code_ratio_min)),
        )

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
