"""Phase I codebook validation hard gate 规则。

本文件只负责把 evaluator 已经计算好的强类型 raw metrics 判定为 pass/fail，
不访问 model、DataLoader 或文件系统，也不重新计算 raw metrics。

设计边界:
    - ``phase1_validation_data_schema.py`` 定义指标是什么；
    - ``phase1_validation_config.py`` 定义阈值从哪里来；
    - 本文件只定义“指标和阈值如何形成 hard gate 判定”；
    - ``phase1_validation_score.py`` 只在 hard gate 全部通过后做候选 checkpoint
      之间的排序，不应该反过来影响 pass/fail。

判定原则:
    1. hard gate 是保守过滤器。任何缺失、NaN 或不可计算的必需指标都按失败处理，
       避免因为评估管线漏算而把 checkpoint 错误放入候选集；
    2. 每个 layer 只回答一个审计问题。例如 teacher 是否值得学、codebook 是否
       稳定、archetype 是否有行为语义、oracle label 是否保留盈利能力、label
       是否可由 Phase II selector 学到；
    3. 本文件中的 message 面向 report 和失败摘要，说明“为什么需要这个 gate”，
       不展开 raw metric 的计算公式。公式应保留在 calculator 或 schema 文档中。

使用场景:
    1. ``Phase1CodebookEvaluator.evaluate_checkpoint()`` 分别调用五个 layer rule；
    2. 每个 rule 返回 ``Phase1LayerResult``，供 report 和 selector 审计；
    3. ``aggregate_validation_result()`` 把五层结果、raw metrics、diagnostics、
       score 和 tie-breaker 组装为 checkpoint 级 ``Phase1ValidationResult``。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .phase1_metric_results import (
    Phase1LayerResult,
    Phase1MetricResult,
    Phase1ValidationResult,
)
from .phase1_validation_config import (
    Phase1BehaviorQualityThresholds,
    Phase1LabelPredictabilityThresholds,
    Phase1OracleProfitabilityThresholds,
    Phase1TeacherQualityThresholds,
    Phase1VQInternalThresholds,
)
from .phase1_validation_data_schema import (
    Phase1BehaviorQualityMetrics,
    Phase1CodeDiagnostic,
    Phase1LabelPredictabilityMetrics,
    Phase1OracleProfitabilityMetrics,
    Phase1TeacherQualityMetrics,
    Phase1TieBreakerMetrics,
    Phase1VQInternalMetrics,
    Phase1ValidationMetrics,
)


def _is_missing(value: Any) -> bool:
    """判断 hard gate 指标是否缺失或不可计算。

    规则层只消费已经算好的 scalar metric。这里把 ``None`` 和 ``NaN`` 都视为
    缺失，交给 ``_metric_result()`` 统一转成 ``severity="skip"`` 且
    ``passed=False``。这样 report 能区分“明确失败”和“指标没算出来”，但 selector
    仍然把二者都排除在合格 checkpoint 之外。
    """

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _metric_result(
    *,
    name: str,
    value: int | float | str | bool | None,
    threshold: str,
    passed: bool,
    layer: str,
    message: str,
) -> Phase1MetricResult:
    """创建 metric result，并把缺失 hard gate 统一标记为 skip-as-fail。

    参数说明:
        name:
            稳定的 metric 字段名，必须和 raw metrics/schema 中的字段保持一致，
            便于 report、flat dict 和失败摘要互相对齐。
        value:
            evaluator/calculator 已计算好的原始指标值。本函数不做重新计算。
        threshold:
            面向人的阈值表达式，例如 ``">= 0.85"``。真实阈值比较在调用方完成。
        passed:
            非缺失情况下的 hard gate 判定结果。
        layer:
            所属 validation layer 的稳定名称。
        message:
            report 中展示的规则解释，重点说明该 gate 想排除的风险。
    """

    if _is_missing(value):
        return Phase1MetricResult(
            name=name,
            value=None,
            threshold=threshold,
            severity="skip",
            passed=False,
            layer=layer,
            message=f"{message}；指标缺失或不可计算，按 hard gate 失败处理",
        )
    return Phase1MetricResult(
        name=name,
        value=value,
        threshold=threshold,
        severity="pass" if passed else "fail",
        passed=passed,
        layer=layer,
        message=message,
    )


def _gt(
    *,
    name: str,
    value: float,
    threshold_value: float,
    layer: str,
    message: str,
) -> Phase1MetricResult:
    """构造“必须严格大于阈值”的 hard gate 结果。

    适用于收益优势等指标：等于 0 只说明没有优于 baseline，不能证明 checkpoint
    具备增量价值，因此使用严格大于。
    """

    return _metric_result(
        name=name,
        value=value,
        threshold=f"> {threshold_value:g}",
        passed=value > threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
    )


def _ge(
    *,
    name: str,
    value: float,
    threshold_value: float,
    layer: str,
    message: str,
) -> Phase1MetricResult:
    """构造“必须大于或等于阈值”的 hard gate 结果。

    适用于准确率、覆盖率、保留比例等最低质量要求。达到阈值即认为满足该 gate，
    后续 checkpoint 之间的优劣交给 score 和 tie-breaker。
    """

    return _metric_result(
        name=name,
        value=value,
        threshold=f">= {threshold_value:g}",
        passed=value >= threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
    )


def _le(
    *,
    name: str,
    value: float | int,
    threshold_value: float | int,
    layer: str,
    message: str,
) -> Phase1MetricResult:
    """构造“必须小于或等于上限”的 hard gate 结果。

    适用于 collapse、dead code、弱 code 比例、尾部收益集中度等风险项。达到上限
    仍允许通过，超过上限说明该风险已经不可接受。
    """

    return _metric_result(
        name=name,
        value=value,
        threshold=f"<= {threshold_value:g}",
        passed=value <= threshold_value if not _is_missing(value) else False,
        layer=layer,
        message=message,
    )


def _between(
    *,
    name: str,
    value: float,
    lower: float,
    upper: float,
    layer: str,
    message: str,
) -> Phase1MetricResult:
    """构造“必须落在闭区间内”的 hard gate 结果。

    用于同时存在下限和上限的健康度指标。例如 normalized perplexity 太低通常
    表示 code 分布塌缩，太高则可能表示分配接近随机、缺少可解释结构。
    """

    return _metric_result(
        name=name,
        value=value,
        threshold=f"[{lower:g}, {upper:g}]",
        passed=lower <= value <= upper if not _is_missing(value) else False,
        layer=layer,
        message=message,
    )


def _build_layer_result(
    *,
    layer_id: int,
    name: str,
    metrics: Sequence[Phase1MetricResult],
) -> Phase1LayerResult:
    """根据本层所有 metric result 聚合 layer result。

    一层内采用全通过语义：只要任一 hard gate fail/skip，该 layer 就失败。
    这样失败层列表能直接表达 checkpoint 被排除的主要原因。
    """

    metric_tuple = tuple(metrics)
    return Phase1LayerResult(
        layer_id=layer_id,
        name=name,
        passed=all(metric.passed for metric in metric_tuple),
        metrics=metric_tuple,
    )


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

    gate 解释:
        - ``dp_advantage_vs_flat`` 要求 teacher 平均收益确实优于空仓 baseline；
        - ``dp_win_rate_vs_flat`` 检查优势是否广泛分布，而不是集中在少数 horizon；
        - ``near_zero_opportunity_ratio`` 过滤收益接近手续费噪声的弱机会样本；
        - ``fee_sensitivity`` 检查策略是否对手续费假设过于脆弱；
        - ``morphology_coverage`` 要求样本包含足够非 neutral 市场结构；
        - ``dp_return_concentration_after_top5_removed`` 检查去除头部收益后仍有正优势。
    """

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


def evaluate_vq_internal_rules(
    metrics: Phase1VQInternalMetrics,
    thresholds: Phase1VQInternalThresholds,
) -> Phase1LayerResult:
    """判定第一层 VQ 内部质量。

    审计问题:
        codebook 是否完成了“稳定且有保真的压缩”。这一层只看 VQ 自身质量，不判断
        每个 code 是否有交易语义，也不证明可预测性。

    gate 解释:
        - reconstruction/action accuracy 保证 decoded action 没有丢掉 DP 行为；
        - train/validation loss gap 控制重构过拟合；
        - active/dead/max occupancy/perplexity 检查 codebook 使用是否健康，避免
          label collapse 或近似随机分配；
        - assignment churn 和 nearest-second margin 检查 label 是否稳定、边界是否清晰；
        - direction accuracy 和 entry timing error 检查交易方向与入场时点是否保真。
    """

    layer = "vq_internal"
    results = (
        _ge(
            name="validation_action_accuracy",
            value=metrics.validation_action_accuracy,
            threshold_value=thresholds.action_accuracy_min,
            layer=layer,
            message="decoder 需要在 assigned code 条件下重构 DP 动作",
        ),
        _le(
            name="reconstruction_loss_gap",
            value=metrics.reconstruction_loss_gap,
            threshold_value=thresholds.reconstruction_loss_gap_max,
            layer=layer,
            message="validation/train 重构损失差距不能过大",
        ),
        _ge(
            name="active_code_ratio",
            value=metrics.active_code_ratio,
            threshold_value=thresholds.active_code_ratio_min,
            layer=layer,
            message="codebook 需要有足够比例的 code 被有效使用",
        ),
        _le(
            name="max_code_occupancy",
            value=metrics.max_code_occupancy,
            threshold_value=thresholds.max_code_occupancy_max,
            layer=layer,
            message="单个 code 不能占用过多样本，避免 label collapse",
        ),
        _between(
            name="normalized_code_perplexity",
            value=metrics.normalized_code_perplexity,
            lower=thresholds.normalized_perplexity_min,
            upper=thresholds.normalized_perplexity_max,
            layer=layer,
            message="code 分布既不能塌缩，也不能接近随机无结构",
        ),
        _le(
            name="dead_code_ratio",
            value=metrics.dead_code_ratio,
            threshold_value=thresholds.dead_code_ratio_max,
            layer=layer,
            message="dead code 比例不能过高",
        ),
        _le(
            name="assignment_churn_recent_mean",
            value=metrics.assignment_churn_recent_mean,
            threshold_value=thresholds.churn_recent_mean_max,
            layer=layer,
            message="近期 assignment churn 需要足够低，保证 label 语义稳定",
        ),
        _ge(
            name="code_lifetime_pass_ratio",
            value=metrics.code_lifetime_pass_ratio,
            threshold_value=thresholds.code_lifetime_pass_ratio_min,
            layer=layer,
            message="active code 需要有足够比例保持稳定生命周期",
        ),
        _ge(
            name="nearest_second_margin_median",
            value=metrics.nearest_second_margin_median,
            threshold_value=thresholds.margin_median_min,
            layer=layer,
            message="样本到最近和第二近 code 的距离 margin 需要清晰",
        ),
        _ge(
            name="direction_accuracy",
            value=metrics.direction_accuracy,
            threshold_value=thresholds.direction_accuracy_min,
            layer=layer,
            message="decoded 主方向需要和 DP demo 主方向一致",
        ),
        _le(
            name="entry_timing_error_median",
            value=metrics.entry_timing_error_median,
            threshold_value=thresholds.entry_timing_error_ratio_max,
            layer=layer,
            message="入场时点误差相对 horizon 的比例不能过大",
        ),
        _le(
            name="decoder_turnover_error",
            value=metrics.decoder_turnover_error,
            threshold_value=thresholds.decoder_turnover_error_max,
            layer=layer,
            message="decoded 不能相对 DP demo 引入过多额外换手",
        ),
    )
    return _build_layer_result(layer_id=1, name=layer, metrics=results)


def evaluate_behavior_quality_rules(
    metrics: Phase1BehaviorQualityMetrics,
    thresholds: Phase1BehaviorQualityThresholds,
) -> Phase1LayerResult:
    """判定第二层 archetype 行为质量。

    审计问题:
        active code 是否真的形成了可解释、可区分的交易 archetype。第一层通过只说明
        codebook 能重构动作；第二层进一步要求每个 code 在 support、市场形态、
        交易 motif 和 decoded action 行为上有稳定含义。

    gate 解释:
        - weak_*_code_ratio 系列限制弱 code 占比，避免大量 active code 只有样本数
          或表面分配，却缺少可解释结构；
        - intra_code_action_similarity 要求同一 code 内行为一致；
        - inter_intra_separation 要求不同 code 之间的行为中心明显分开；
        - duplicate_code_pair_count 禁止多个 code 学成几乎相同的原型；
        - profitable_code_coverage 要求足够多 active code 具备后续盈利潜力。
    """

    layer = "behavior_quality"
    duplicate_pair_count_max = (
        thresholds.duplicate_code_pair_count_max
        if thresholds.duplicate_code_pair_count_max is not None
        else (int(metrics.num_codes) if metrics.num_codes > 0 else 10)
    )
    results = (
        _le(
            name="weak_support_code_ratio",
            value=metrics.weak_support_code_ratio,
            threshold_value=thresholds.weak_support_code_ratio_max,
            layer=layer,
            message="support 不足的 active code 比例不能过高",
        ),
        _le(
            name="weak_morphology_code_ratio",
            value=metrics.weak_morphology_code_ratio,
            threshold_value=thresholds.weak_structure_code_ratio_max,
            layer=layer,
            message="市场形态不清晰的 active code 比例不能过高",
        ),
        _le(
            name="weak_motif_code_ratio",
            value=metrics.weak_motif_code_ratio,
            threshold_value=thresholds.weak_structure_code_ratio_max,
            layer=layer,
            message="交易 motif 不清晰的 active code 比例不能过高",
        ),
        _le(
            name="weak_pair_code_ratio",
            value=metrics.weak_pair_code_ratio,
            threshold_value=thresholds.weak_structure_code_ratio_max,
            layer=layer,
            message="dominant morphology-motif pair 不清晰的 active code 比例不能过高",
        ),
        _le(
            name="weak_lift_nonprofitable_code_ratio",
            value=metrics.weak_lift_nonprofitable_code_ratio,
            threshold_value=thresholds.weak_structure_code_ratio_max,
            layer=layer,
            message="缺少结构 lift 或盈利性的弱 code 比例不能过高",
        ),
        _ge(
            name="intra_code_action_similarity",
            value=metrics.intra_code_action_similarity,
            threshold_value=thresholds.intra_code_similarity_min,
            layer=layer,
            message="同一 code 内 decoded action 序列需要足够一致",
        ),
        _ge(
            name="inter_intra_separation",
            value=metrics.inter_intra_separation,
            threshold_value=thresholds.inter_intra_separation_min,
            layer=layer,
            message="不同 code 的行为中心需要和 code 内差异拉开距离",
        ),
        _ge(
            name="latent_silhouette_score",
            value=metrics.latent_silhouette_score,
            threshold_value=thresholds.latent_silhouette_score_min,
            layer=layer,
            message="latent 空间中的 assigned code 需要有清晰聚类边界",
        ),
        _le(
            name="duplicate_code_pair_count",
            value=metrics.duplicate_code_pair_count,
            threshold_value=duplicate_pair_count_max,
            layer=layer,
            message="不能存在超过重复相似度阈值的 code pair",
        ),
        _ge(
            name="profitable_code_coverage",
            value=metrics.profitable_code_coverage,
            threshold_value=thresholds.profitable_code_coverage_min,
            layer=layer,
            message="具备盈利潜力的 active code 覆盖率需要足够高",
        ),
    )
    return _build_layer_result(layer_id=2, name=layer, metrics=results)


def evaluate_oracle_profitability_rules(
    metrics: Phase1OracleProfitabilityMetrics,
    thresholds: Phase1OracleProfitabilityThresholds,
) -> Phase1LayerResult:
    """判定第三层 oracle assigned-label 盈利性。

    审计问题:
        在使用 encoder 已经分配好的 oracle label 时，decoder 执行出的策略是否仍然
        保留 DP teacher 的盈利能力。这一层证明 codebook 有可交易潜力，但不代表
        真实线上 selector 已经能提前预测这些 label。

    gate 解释:
        - decoded advantage/win rate 要求 decoded 策略优于 flat baseline 且胜率广泛；
        - random label baseline 检查 assigned label 是否有信息量，而不是 decoder
          对任意 label 都类似；
        - retention_ratio 要求压缩后保留足够 teacher 盈利能力；
        - top_5_contribution 和 trimmed advantage 检查收益不是少数尾部样本造成；
        - bad_code_ratio 和 dominant_pair_positive_ratio 控制 per-code 与结构组合层面的
          负价值风险。
    """

    layer = "oracle_profitability"
    results = (
        _gt(
            name="mean_decoded_advantage_vs_flat",
            value=metrics.mean_decoded_advantage_vs_flat,
            threshold_value=0.0,
            layer=layer,
            message="assigned-label decoded 策略平均收益必须优于 flat baseline",
        ),
        _ge(
            name="decoded_win_rate_vs_flat",
            value=metrics.decoded_win_rate_vs_flat,
            threshold_value=thresholds.decoded_win_rate_min,
            layer=layer,
            message="decoded 策略胜率需要足够广泛",
        ),
        _gt(
            name="mean_advantage_vs_random_label",
            value=metrics.mean_advantage_vs_random_label,
            threshold_value=0.0,
            layer=layer,
            message="assigned label decoded 表现必须优于 random label baseline",
        ),
        _ge(
            name="random_label_relative_lift",
            value=metrics.random_label_relative_lift,
            threshold_value=thresholds.random_label_relative_lift_min,
            layer=layer,
            message="assigned label 相对 random label 的收益提升需要足够高",
        ),
        _ge(
            name="retention_ratio",
            value=metrics.retention_ratio,
            threshold_value=thresholds.retention_ratio_min,
            layer=layer,
            message="decoded 策略需要保留足够 DP teacher 盈利能力",
        ),
        _le(
            name="downside_control",
            value=metrics.downside_control,
            threshold_value=thresholds.downside_control_max,
            layer=layer,
            message="decoded 策略相对 DP teacher 的回撤放大不能过高",
        ),
        _gt(
            name="risk_adjusted_return",
            value=metrics.risk_adjusted_return,
            threshold_value=thresholds.risk_adjusted_return_min,
            layer=layer,
            message="decoded 策略风险调整收益必须为正",
        ),
        _gt(
            name="risk_adjusted_return_vs_random",
            value=metrics.risk_adjusted_return_vs_random,
            threshold_value=0.0,
            layer=layer,
            message="decoded 策略风险调整收益必须优于 random label baseline",
        ),
        _le(
            name="top_5_contribution",
            value=metrics.top_5_contribution,
            threshold_value=thresholds.top_5_contribution_max,
            layer=layer,
            message="收益不能过度依赖收益最高 top 5% horizon",
        ),
        _gt(
            name="trimmed_decoded_advantage",
            value=metrics.trimmed_decoded_advantage,
            threshold_value=0.0,
            layer=layer,
            message="去除尾部样本后 decoded 策略仍应有正优势",
        ),
        _le(
            name="fee_drag",
            value=metrics.fee_drag,
            threshold_value=thresholds.fee_drag_max,
            layer=layer,
            message="手续费拖累比例不能过高",
        ),
        _ge(
            name="turnover_return_correlation",
            value=metrics.turnover_return_correlation,
            threshold_value=thresholds.turnover_return_correlation_min,
            layer=layer,
            message="换手与收益的相关性不能显著为负",
        ),
        _le(
            name="bad_code_ratio",
            value=metrics.bad_code_ratio,
            threshold_value=thresholds.bad_code_ratio_max,
            layer=layer,
            message="负价值 code 比例不能过高",
        ),
        _ge(
            name="dominant_pair_positive_ratio",
            value=metrics.dominant_pair_positive_ratio,
            threshold_value=thresholds.dominant_pair_positive_ratio_min,
            layer=layer,
            message="dominant morphology-motif pair 中正优势比例需要足够高",
        ),
    )
    return _build_layer_result(layer_id=3, name=layer, metrics=results)


def evaluate_label_predictability_rules(
    metrics: Phase1LabelPredictabilityMetrics,
    thresholds: Phase1LabelPredictabilityThresholds,
) -> Phase1LayerResult:
    """判定第四层 label 可预测性 / selector 可学习性。

    审计问题:
        Phase II selector 只能在 horizon 起点看到当时可用状态。如果 assigned label
        只在事后 oracle 条件下有盈利性，但无法从起点状态预测，就不能作为后续策略
        的可靠监督信号。

    gate 解释:
        - top-1/top-3 accuracy 要求 probe 明显优于随机猜测，并能缩小候选 label；
        - balanced accuracy 防止 probe 只预测高频 code；
        - mutual information lift 检查 label 和可见状态之间存在统计关系；
        - probe_return_retention 检查用 probe 预测 label 执行后仍保留足够 oracle 收益。
    """

    layer = "label_predictability"
    num_codes = max(1, int(metrics.num_codes))
    top1_threshold = max(
        thresholds.probe_top1_floor,
        thresholds.probe_top1_k_factor / num_codes,
    )
    top3_threshold = max(
        thresholds.probe_top3_floor,
        thresholds.probe_top3_k_factor / num_codes,
    )
    label_entropy_threshold = (
        metrics.label_entropy * thresholds.label_entropy_given_morphology_max_ratio
        if not _is_missing(metrics.label_entropy) and metrics.label_entropy > 0.0
        else float("nan")
    )
    results = (
        _ge(
            name="probe_top1_accuracy",
            value=metrics.probe_top1_accuracy,
            threshold_value=top1_threshold,
            layer=layer,
            message="probe top-1 accuracy 需要明显高于随机水平",
        ),
        _ge(
            name="probe_top3_accuracy",
            value=metrics.probe_top3_accuracy,
            threshold_value=top3_threshold,
            layer=layer,
            message="probe top-3 accuracy 需要能缩小 selector 候选范围",
        ),
        _ge(
            name="probe_balanced_accuracy",
            value=metrics.probe_balanced_accuracy,
            threshold_value=thresholds.probe_balanced_accuracy_min,
            layer=layer,
            message="balanced accuracy 需要避免只预测高频 code",
        ),
        _metric_result(
            name="label_entropy_given_morphology",
            value=(
                metrics.label_entropy_given_morphology
                if not _is_missing(label_entropy_threshold)
                else None
            ),
            threshold=(
                "<= "
                f"{thresholds.label_entropy_given_morphology_max_ratio:g} * H(label)"
            ),
            passed=(
                metrics.label_entropy_given_morphology <= label_entropy_threshold
                if not _is_missing(metrics.label_entropy_given_morphology)
                and not _is_missing(label_entropy_threshold)
                else False
            ),
            layer=layer,
            message="给定 morphology 后的 label 条件熵需要明显低于全局 label 熵",
        ),
        _ge(
            name="mutual_information_lift",
            value=metrics.mutual_information_lift,
            threshold_value=thresholds.mutual_information_lift_min,
            layer=layer,
            message="label 与可见状态之间需要有显著统计关系",
        ),
        _ge(
            name="probe_return_retention",
            value=metrics.probe_return_retention,
            threshold_value=thresholds.probe_return_retention_min,
            layer=layer,
            message="probe label decoded return 需要保留足够 oracle decoded return",
        ),
    )
    return _build_layer_result(layer_id=4, name=layer, metrics=results)


def aggregate_validation_result(
    *,
    checkpoint_id: str,
    stage: str,
    epoch: int,
    layers: Sequence[Phase1LayerResult],
    metrics: Phase1ValidationMetrics,
    code_diagnostics: Sequence[Phase1CodeDiagnostic],
    drift_diagnostics: Mapping[str, Phase1MetricResult],
    score: float | None,
    tie_breaker_metrics: Phase1TieBreakerMetrics,
) -> Phase1ValidationResult:
    """聚合 checkpoint 级 validation result。

    使用场景:
        ``Phase1CodebookEvaluator.evaluate_checkpoint()`` 在五层 rules 和 scoring
        都完成后调用。若任一 layer fail，则 ``passed=False`` 且 ``score`` 置为 None。

    设计说明:
        ``score`` 只用于通过 hard gate 的 checkpoint 之间排序。这里即使调用方传入了
        score，只要任一 layer 失败也会强制写成 ``None``，避免 selector 或 report
        误把失败 checkpoint 的分数当作可比较候选。
    """

    layer_tuple = tuple(layers)
    failed_layers = tuple(layer.name for layer in layer_tuple if not layer.passed)
    passed = len(failed_layers) == 0
    return Phase1ValidationResult(
        checkpoint_id=checkpoint_id,
        stage=stage,
        epoch=epoch,
        passed=passed,
        score=score if passed else None,
        failed_layers=failed_layers,
        layers=layer_tuple,
        metrics=metrics,
        code_diagnostics=tuple(code_diagnostics),
        drift_diagnostics=dict(drift_diagnostics),
        tie_breaker_metrics=tie_breaker_metrics,
    )


__all__ = [
    "aggregate_validation_result",
    "evaluate_behavior_quality_rules",
    "evaluate_label_predictability_rules",
    "evaluate_oracle_profitability_rules",
    "evaluate_teacher_quality_rules",
    "evaluate_vq_internal_rules",
]
