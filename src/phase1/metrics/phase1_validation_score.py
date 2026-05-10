"""Phase I codebook validation 综合评分和 tie-breaker 工具。

本文件只负责把已经计算好的强类型 validation metrics 转换为归一化 score。
hard gate 是否通过由 ``phase1_validation_rules.py`` 决定；调用方应只在五层
hard gate 全部通过后使用 ``compute_phase1_validation_score()`` 的结果参与排序。

设计边界:
    - rules 层回答 checkpoint 是否合格，score 层只回答合格 checkpoint 谁更优；
    - 本文件不访问模型、DataLoader、文件系统，也不重新计算 raw metrics；
    - 分数统一归一化到 [0, 1]，便于按权重线性组合；
    - 归一化函数偏保守，缺失或 NaN 的输入按 0 分处理。正常流程中 hard gate
      已经把缺失必需指标挡掉，这里的 0 分主要用于防御性处理和 report 拆分展示。

评分原则:
    1. 每个子分数内部先把不同量纲的 metric 转成可比较的 [0, 1]；
    2. 子分数内部采用简单平均，避免某个单项指标在 layer 内过度主导；
    3. 总分按 ``Phase1ValidationScoreWeights`` 加权，用于 checkpoint selector
       在多个合格候选之间排序；
    4. 分数接近时再使用 tie-breaker，优先选择风险调整收益、可学习性和盈利保留
       更好的 checkpoint。

使用场景:
    1. checkpoint 通过五层 hard gate 后计算 ``validation.score``；
    2. checkpoint score 接近时，使用 tie-breaker 指标做稳定排序；
    3. report 展示综合评分拆解时复用同一套子分数函数。
"""

from __future__ import annotations

import math

from .phase1_validation_config import Phase1ValidationScoreWeights
from .phase1_validation_data_schema import (
    Phase1TieBreakerMetrics,
    Phase1ValidationMetrics,
)


DEFAULT_TIE_SCORE_TOLERANCE = 0.03
"""默认 tie-breaker 触发阈值。

两个 score 差距小于 3% 时认为综合分无法稳定区分优劣，selector 应继续比较
``Phase1TieBreakerMetrics``。该阈值只影响候选排序，不影响 hard gate pass/fail。
"""


def _clip01(value: float) -> float:
    """把数值截断到 [0, 1]，不可计算值按 0 处理。

    使用场景:
        作为所有归一化 helper 的最后一道保护，防止异常输入或公式外推导致子分数
        超出统一评分区间。
    """

    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _positive_score(value: float, scale: float = 1.0) -> float:
    """把正向无界指标压缩到 [0, 1]。

    使用场景:
        收益优势、risk-adjusted return 等指标量纲不固定，直接线性比较容易被
        极端值主导，因此用 ``value / (abs(value) + scale)`` 做温和压缩。

    评分含义:
        非正值给 0 分；正值越大分数越接近 1，但边际提升逐渐变小。``scale`` 控制
        压缩速度，默认 1.0 表示只有明显为正的优势才会贡献较高分。
    """

    if math.isnan(value) or value <= 0:
        return 0.0
    return _clip01(value / (abs(value) + scale))


def _threshold_progress(value: float, threshold: float) -> float:
    """把“越高越好且有最低阈值”的比例指标归一化。

    评分含义:
        ``value == threshold`` 映射到 1 分，低于阈值按比例给部分分，高于阈值不再
        继续加分。hard gate 负责保证合格线，score 只表达离合格线的相对进度。

    使用场景:
        win rate、coverage、retention、accuracy 等本身有自然下限的比例类指标。
    """

    if math.isnan(value):
        return 0.0
    if threshold <= 0:
        return _clip01(value)
    return _clip01(value / threshold)


def _inverse_ratio_score(value: float, maximum: float) -> float:
    """把“越低越好且有上限”的比例指标归一化。

    评分含义:
        0 映射到 1 分，达到上限映射到 0 分，超过上限仍为 0 分。用于表达风险项
        离不可接受上限还有多少余量。

    使用场景:
        weak code ratio、dead code ratio、max occupancy、top-5 contribution 等
        越低越好的比例或风险指标。
    """

    if math.isnan(value):
        return 0.0
    if maximum <= 0:
        return 1.0 if value <= 0 else 0.0
    return _clip01(1.0 - value / maximum)


def _accuracy_window_score(value: float, lower: float, upper: float = 1.0) -> float:
    """把准确率按指定合格窗口映射到 [0, 1]。

    评分含义:
        ``lower`` 及以下给 0 分，``upper`` 及以上给 1 分，中间线性插值。适合
        validation action accuracy 这类已经有明确合格线、且上界自然为 1 的指标。
    """

    if math.isnan(value) or value <= lower:
        return 0.0
    if upper <= lower:
        return 1.0
    return _clip01((value - lower) / (upper - lower))


def compute_teacher_quality_score(metrics: Phase1ValidationMetrics) -> float:
    """计算第零层 teacher quality 子分数。

    子分数含义:
        衡量 DP teacher 是否值得被 VQ codebook 学习。该分数偏向选择优势更广泛、
        对手续费更稳健、市场结构覆盖更充分、收益不过度依赖头部样本的 teacher。

    组成项:
        - ``dp_advantage_vs_flat``: teacher 平均优势，正向无界压缩；
        - ``dp_win_rate_vs_flat``: 相对 flat 的胜率，按 0.58 合格线归一化；
        - ``near_zero_opportunity_ratio``: 弱机会样本比例，越低越好；
        - ``fee_sensitivity``: 手续费变高后的收益保留比例；
        - ``morphology_coverage``: 非 neutral 市场结构覆盖率；
        - ``dp_return_concentration_after_top5_removed``: 去掉头部收益后仍有多少优势。
    """

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


def compute_reconstruction_score(metrics: Phase1ValidationMetrics) -> float:
    """计算 reconstruction 子分数。

    子分数含义:
        单独衡量 decoder 在 assigned code 条件下还原 DP action 的能力。这里只使用
        validation action accuracy，因为 reconstruction loss 的量纲通常依赖具体
        模型实现，更适合作为 tie-breaker 或 report 指标。

    归一化:
        0.85 及以下给 0 分，1.0 给满分，中间线性插值。hard gate 已经保证合格
        checkpoint 达到最低准确率，score 只区分合格线以上的保真程度。
    """

    return _accuracy_window_score(
        metrics.vq_internal.validation_action_accuracy,
        lower=0.85,
        upper=1.0,
    )


def compute_codebook_health_score(metrics: Phase1ValidationMetrics) -> float:
    """计算 codebook health 子分数。

    子分数含义:
        衡量 VQ codebook 的使用是否健康、稳定、边界是否清楚。该分数偏向 active
        code 更多、占用更均衡、dead code 更少、assignment churn 更低、方向更保真的
        checkpoint。

    组成项:
        - ``active_code_ratio``: codebook 使用充分程度；
        - ``normalized_code_perplexity``: code 分布健康度，中心值 0.70，太低偏塌缩、
          太高偏随机；
        - ``max_code_occupancy``: 单个 code 最大占用，越低越能降低 label collapse 风险；
        - ``dead_code_ratio``: 长期无效 code 比例；
        - ``assignment_churn_recent_mean``: 近期 label 重排程度；
        - ``nearest_second_margin_median``: 最近 code 与第二近 code 的距离边界；
        - ``direction_accuracy``: decoded 主方向和 demo 主方向的一致性。
    """

    vq = metrics.vq_internal
    perplexity_center = 0.70
    perplexity_score = 1.0 - min(
        1.0,
        abs(vq.normalized_code_perplexity - perplexity_center) / 0.40,
    )
    parts = (
        _threshold_progress(vq.active_code_ratio, 0.80),
        _clip01(perplexity_score),
        _inverse_ratio_score(vq.max_code_occupancy, 0.40),
        _inverse_ratio_score(vq.dead_code_ratio, 0.20),
        _inverse_ratio_score(vq.assignment_churn_recent_mean, 0.15),
        _threshold_progress(vq.nearest_second_margin_median, 0.10),
        _threshold_progress(vq.direction_accuracy, 0.88),
    )
    return sum(parts) / len(parts)


def compute_behavior_structure_score(metrics: Phase1ValidationMetrics) -> float:
    """计算 behavior structure 子分数。

    子分数含义:
        衡量 active code 是否形成清晰、稳定、可区分的交易 archetype。该分数会奖励
        弱 code 更少、code 内行为更一致、code 间更分离、重复原型更少、盈利 code
        覆盖更高的 checkpoint。

    组成项:
        - ``weak_support_code_ratio``: support 不足的 active code 比例；
        - ``weak_morphology_code_ratio``: 市场形态不清晰的 active code 比例；
        - ``weak_motif_code_ratio``: 交易 motif 不清晰的 active code 比例；
        - ``weak_pair_code_ratio``: morphology-motif pair 不稳定的 active code 比例；
        - ``weak_lift_nonprofitable_code_ratio``: 缺少结构 lift 或盈利潜力的弱 code 比例；
        - ``intra_code_action_similarity``: 同 code 内 decoded action 一致性；
        - ``inter_intra_separation``: code 间距离相对 code 内距离的分离度；
        - ``latent_silhouette_score``: latent 聚类轮廓系数，从 [-1, 1] 映射到 [0, 1]；
        - ``duplicate_code_pair_count``: 重复 code pair 越多惩罚越大；
        - ``profitable_code_coverage``: 具备盈利潜力的 active code 覆盖率。
    """

    behavior = metrics.behavior_quality
    parts = (
        1.0 - _clip01(behavior.weak_support_code_ratio),
        1.0 - _clip01(behavior.weak_morphology_code_ratio),
        1.0 - _clip01(behavior.weak_motif_code_ratio),
        1.0 - _clip01(behavior.weak_pair_code_ratio),
        1.0 - _clip01(behavior.weak_lift_nonprofitable_code_ratio),
        _threshold_progress(behavior.intra_code_action_similarity, 0.65),
        _threshold_progress(behavior.inter_intra_separation, 1.30),
        _clip01((behavior.latent_silhouette_score + 1.0) / 2.0),
        1.0 / (1.0 + max(0, behavior.duplicate_code_pair_count)),
        _threshold_progress(behavior.profitable_code_coverage, 0.60),
    )
    return sum(parts) / len(parts)


def compute_oracle_profitability_score(metrics: Phase1ValidationMetrics) -> float:
    """计算 oracle profitability 子分数。

    子分数含义:
        衡量在 oracle assigned-label 条件下，decoder 策略是否保留了 DP teacher 的
        交易价值。该分数只评价“已知 label 后执行是否有价值”，不评价 Phase II
        selector 是否能提前预测 label。

    组成项:
        - ``mean_decoded_advantage_vs_flat``: decoded 策略相对 flat 的平均优势；
        - ``decoded_win_rate_vs_flat``: decoded 策略胜率是否广泛；
        - ``mean_advantage_vs_random_label`` 和 ``random_label_relative_lift``:
          assigned label 相对随机 label 是否有信息量；
        - ``retention_ratio``: decoded 策略保留 DP teacher 盈利能力的比例；
        - ``risk_adjusted_return``: 风险调整后的收益表现；
        - ``top_5_contribution`` 和 ``trimmed_decoded_advantage``: 收益是否依赖尾部样本；
        - ``fee_drag``: 手续费拖累越低越好；
        - ``turnover_return_correlation``: turnover 与收益的相关关系，从 [-1, 1]
          映射到 [0, 1]；
        - ``bad_code_ratio`` 和 ``dominant_pair_positive_ratio``: per-code 与结构组合
          维度的负价值风险。
    """

    oracle = metrics.oracle_profitability
    parts = (
        _positive_score(oracle.mean_decoded_advantage_vs_flat),
        _threshold_progress(oracle.decoded_win_rate_vs_flat, 0.55),
        _positive_score(oracle.mean_advantage_vs_random_label),
        _threshold_progress(oracle.random_label_relative_lift, 0.20),
        _threshold_progress(oracle.retention_ratio, 0.50),
        _positive_score(oracle.risk_adjusted_return),
        _inverse_ratio_score(oracle.top_5_contribution, 0.60),
        _positive_score(oracle.trimmed_decoded_advantage),
        1.0 - _clip01(oracle.fee_drag),
        _clip01((oracle.turnover_return_correlation + 1.0) / 2.0),
        1.0 - _clip01(oracle.bad_code_ratio),
        _threshold_progress(oracle.dominant_pair_positive_ratio, 0.60),
    )
    return sum(parts) / len(parts)


def compute_label_predictability_score(metrics: Phase1ValidationMetrics) -> float:
    """计算 label predictability 子分数。

    子分数含义:
        衡量 assigned label 是否能从 horizon 起点可见状态中学习出来。Phase I 的
        oracle label 只有在 Phase II selector 能预测时才有实际部署价值。

    组成项:
        - ``probe_top1_accuracy``: probe 直接命中 assigned label 的能力；
        - ``probe_top3_accuracy``: probe 缩小候选 label 范围的能力；
        - ``probe_balanced_accuracy``: 是否避免只预测高频 code；
        - ``mutual_information_lift``: label 与可见状态之间的统计关系强度；
        - ``probe_return_retention``: 使用 probe 预测 label 执行后保留的 oracle 收益。
    """

    label = metrics.label_predictability
    parts = (
        _threshold_progress(label.probe_top1_accuracy, 0.25),
        _threshold_progress(label.probe_top3_accuracy, 0.55),
        _threshold_progress(label.probe_balanced_accuracy, 0.25),
        _threshold_progress(label.mutual_information_lift, 2.0),
        _threshold_progress(label.probe_return_retention, 0.35),
    )
    return sum(parts) / len(parts)


def compute_phase1_validation_score(
    metrics: Phase1ValidationMetrics,
    weights: Phase1ValidationScoreWeights,
) -> float:
    """计算 Phase I checkpoint 综合评分。

    使用场景:
        仅在五层 hard gate 全部通过后调用。返回值被截断到 [0, 1]，用于合格
        checkpoint 之间排序。

    设计说明:
        总分是各子分数的加权和。权重来自 ``Phase1ValidationScoreWeights``，因此
        可以在不修改评分公式的情况下调整 selection 偏好。函数最后仍调用
        ``_clip01()``，避免权重配置异常或浮点误差导致分数越界。

    注意:
        本函数不检查 hard gate 是否通过。调用方应先运行
        ``phase1_validation_rules.py`` 中的五层规则；失败 checkpoint 的 score 应在
        ``aggregate_validation_result()`` 中置为 ``None``。
    """

    score = (
        weights.teacher_quality * compute_teacher_quality_score(metrics)
        + weights.reconstruction * compute_reconstruction_score(metrics)
        + weights.codebook_health * compute_codebook_health_score(metrics)
        + weights.behavior_structure * compute_behavior_structure_score(metrics)
        + weights.oracle_profitability * compute_oracle_profitability_score(metrics)
        + weights.label_predictability * compute_label_predictability_score(metrics)
    )
    return _clip01(score)


def build_tie_breaker_metrics(
    metrics: Phase1ValidationMetrics,
    *,
    reconstruction_loss: float,
) -> Phase1TieBreakerMetrics:
    """从五层 metrics 构造 tie-breaker 指标。

    使用场景:
        evaluator 组装 validation result 前调用，避免 selector 自己理解五层
        metrics 的内部字段路径。

    字段顺序:
        ``compare_phase1_tie_breaker()`` 会按固定优先级比较这些字段。这里集中构造
        tie-breaker payload，可以让 selector 只处理稳定 schema，不依赖五层 metrics
        的内部字段路径。

    选择理由:
        - ``risk_adjusted_return`` 优先保证真实交易质量；
        - ``probe_top3_accuracy`` 保证 Phase II selector 可学习性；
        - ``retention_ratio`` 保证压缩后仍保留 teacher 盈利能力；
        - ``active_code_ratio`` 和 ``max_code_occupancy`` 控制 codebook 使用健康度；
        - ``reconstruction_loss`` 作为最后的保真度兜底比较项。
    """

    return Phase1TieBreakerMetrics(
        risk_adjusted_return=metrics.oracle_profitability.risk_adjusted_return,
        probe_top3_accuracy=metrics.label_predictability.probe_top3_accuracy,
        retention_ratio=metrics.oracle_profitability.retention_ratio,
        active_code_ratio=metrics.vq_internal.active_code_ratio,
        max_code_occupancy=metrics.vq_internal.max_code_occupancy,
        reconstruction_loss=reconstruction_loss,
    )


def scores_are_tied(
    best_score: float,
    candidate_score: float,
    *,
    tolerance: float = DEFAULT_TIE_SCORE_TOLERANCE,
) -> bool:
    """判断两个 checkpoint score 是否足够接近，需要进入 tie-breaker。

    使用场景:
        checkpoint selector 先按 ``validation.score`` 排序。当当前最优和候选的分差
        小于 ``tolerance`` 时，认为综合分差异不足以稳定区分，继续调用
        ``compare_phase1_tie_breaker()``。

    注意:
        使用严格小于而不是小于等于，使边界值 ``tolerance`` 本身仍按 score 直接
        区分，避免 tie-breaker 触发范围比配置值更宽。
    """

    return abs(best_score - candidate_score) < tolerance


def compare_phase1_tie_breaker(
    left: Phase1TieBreakerMetrics,
    right: Phase1TieBreakerMetrics,
) -> int:
    """比较两个 tie-breaker metrics。

    返回:
        ``1`` 表示 left 更优，``-1`` 表示 right 更优，``0`` 表示无法区分。

    比较顺序:
        risk_adjusted_return、probe_top3_accuracy、retention_ratio、
        active_code_ratio 越高越好；max_code_occupancy、reconstruction_loss
        越低越好。

    设计说明:
        这是一个字典序比较：只有当前字段足够接近时才继续比较下一字段。这样能把
        业务优先级表达得很清楚，避免把不同量纲的 tie-breaker 再合成一个难审计的
        二级分数。

    相等判定:
        使用 ``math.isclose(..., rel_tol=1e-12, abs_tol=1e-12)`` 过滤浮点微小差异。
        所有字段都无法区分时返回 0，让调用方保留已有顺序或使用更外层的稳定排序键。
    """

    comparisons = (
        (left.risk_adjusted_return, right.risk_adjusted_return, True),
        (left.probe_top3_accuracy, right.probe_top3_accuracy, True),
        (left.retention_ratio, right.retention_ratio, True),
        (left.active_code_ratio, right.active_code_ratio, True),
        (left.max_code_occupancy, right.max_code_occupancy, False),
        (left.reconstruction_loss, right.reconstruction_loss, False),
    )
    for left_value, right_value, higher_is_better in comparisons:
        if math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12):
            continue
        if higher_is_better:
            return 1 if left_value > right_value else -1
        return 1 if left_value < right_value else -1
    return 0


__all__ = [
    "DEFAULT_TIE_SCORE_TOLERANCE",
    "build_tie_breaker_metrics",
    "compare_phase1_tie_breaker",
    "compute_behavior_structure_score",
    "compute_codebook_health_score",
    "compute_label_predictability_score",
    "compute_oracle_profitability_score",
    "compute_phase1_validation_score",
    "compute_reconstruction_score",
    "compute_teacher_quality_score",
    "scores_are_tied",
]
