"""Phase I codebook validation 综合评分和 tie-breaker 工具。

本文件只负责把已经计算好的强类型 validation metrics 转换为归一化 score。
hard gate 是否通过由 ``phase1_validation_rules.py`` 决定；调用方应只在四层
hard gate 全部通过后使用 ``compute_phase1_validation_score()`` 的结果参与排序。
label predictability 保留为参考值和 tie-breaker，不进入主 score。

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
    1. checkpoint 通过四层 hard gate 后计算 ``validation.score``；
    2. checkpoint score 接近时，使用 tie-breaker 指标做稳定排序；
    3. report 展示综合评分拆解时复用同一套子分数函数。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.utils import PydanticMappingModel

from .phase1_validation_config import Phase1ValidationScoreWeights
from .phase1_validation_data_schema import (
    Phase1TieBreakerMetrics,
    Phase1ValidationMetrics,
)
from .phase1_validation_behavior_quality import compute_behavior_structure_score
from .phase1_validation_label_predictability import compute_label_predictability_score
from .phase1_validation_oracle_profitability import compute_oracle_profitability_score
from .phase1_validation_score_helpers import _accuracy_window_score, _clip01
from .phase1_validation_teacher_quality import compute_teacher_quality_score
from .phase1_validation_vq_internal import compute_codebook_health_score


DEFAULT_TIE_SCORE_TOLERANCE = 0.03
"""默认 tie-breaker 触发阈值。

两个 score 差距小于 3% 时认为综合分无法稳定区分优劣，selector 应继续比较
``Phase1TieBreakerMetrics``。该阈值只影响候选排序，不影响 hard gate pass/fail。
"""


class Phase1ValidationScoreComponent(PydanticMappingModel):
    """单个 validation score 子项的可审计拆解。"""

    # 稳定 snake_case 子项名称，例如 "teacher_quality"。
    name: str

    # 加权前的归一化子分数，范围通常为 [0, 1]。
    value: float

    # 当前子项在总分中的权重。
    weight: float

    # value * weight 后的贡献值。
    weighted_value: float


class Phase1ValidationScore(PydanticMappingModel):
    """Phase I validation 综合评分及其子项拆解。"""

    # 截断到 [0, 1] 后的最终总分，selector 使用该值排序。
    total_score: float

    # 每个子项的加权前分数、权重和加权贡献。
    components: tuple[Phase1ValidationScoreComponent, ...]

    @classmethod
    def from_float(cls, value: float) -> "Phase1ValidationScore":
        """兼容历史 checkpoint 中只保存 float score 的 payload。"""

        return cls(total_score=float(value), components=())


Phase1ValidationScoreLike = Phase1ValidationScore | float | int | None
"""代码迁移期允许 selector 读取新 score 对象或历史 float。"""


def get_phase1_validation_score_value(
    score: Phase1ValidationScoreLike,
) -> float | None:
    """取出可比较的总分数值。"""

    if score is None:
        return None
    if isinstance(score, Phase1ValidationScore):
        return score.total_score
    return float(score)


def _score_component(
    *,
    name: str,
    value: float,
    weight: float,
) -> Phase1ValidationScoreComponent:
    """构造单个 score component，并统一裁剪异常输入。"""

    component_value = _clip01(value)
    return Phase1ValidationScoreComponent(
        name=name,
        value=component_value,
        weight=weight,
        weighted_value=component_value * weight,
    )


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


def compute_phase1_validation_score(
    metrics: Phase1ValidationMetrics,
    weights: Phase1ValidationScoreWeights,
) -> Phase1ValidationScore:
    """计算 Phase I checkpoint 综合评分。

    使用场景:
        仅在四层 hard gate 全部通过后调用。返回对象中的 ``total_score`` 被截断到
        [0, 1]，用于合格 checkpoint 之间排序；``components`` 保留每个子项目的
        加权前分数、权重和加权贡献，供后续报表展示。

    设计说明:
        总分是各子分数的加权和。权重来自 ``Phase1ValidationScoreWeights``，因此
        可以在不修改评分公式的情况下调整 selection 偏好。函数最后仍调用
        ``_clip01()``，避免权重配置异常或浮点误差导致分数越界。

    注意:
        本函数不检查 hard gate 是否通过。调用方应先运行
        ``phase1_validation_rules.py`` 中的 hard-gate 规则；失败 checkpoint 的 score 应在
        ``aggregate_validation_result()`` 中置为 ``None``。
    """

    components = (
        _score_component(
            name="teacher_quality",
            value=compute_teacher_quality_score(metrics),
            weight=weights.teacher_quality,
        ),
        _score_component(
            name="reconstruction",
            value=compute_reconstruction_score(metrics),
            weight=weights.reconstruction,
        ),
        _score_component(
            name="codebook_health",
            value=compute_codebook_health_score(metrics),
            weight=weights.codebook_health,
        ),
        _score_component(
            name="behavior_structure",
            value=compute_behavior_structure_score(metrics),
            weight=weights.behavior_structure,
        ),
        _score_component(
            name="oracle_profitability",
            value=compute_oracle_profitability_score(metrics),
            weight=weights.oracle_profitability + weights.label_predictability,
        ),
    )
    return Phase1ValidationScore(
        total_score=_clip01(
            sum(component.weighted_value for component in components)
        ),
        components=components,
    )


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
        - ``probe_top3_accuracy`` 只作为 Phase II selector 可学习性的参考决胜项；
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
    best_score: Phase1ValidationScoreLike,
    candidate_score: Phase1ValidationScoreLike,
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

    best_value = best_score.total_score
    candidate_value = candidate_score.total_score
    if best_value is None or candidate_value is None:
        return False
    return abs(best_value - candidate_value) < tolerance


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
    "Phase1ValidationScore",
    "Phase1ValidationScoreComponent",
    "Phase1ValidationScoreLike",
    "build_tie_breaker_metrics",
    "compare_phase1_tie_breaker",
    "compute_behavior_structure_score",
    "compute_codebook_health_score",
    "compute_label_predictability_score",
    "compute_oracle_profitability_score",
    "compute_phase1_validation_score",
    "compute_reconstruction_score",
    "compute_teacher_quality_score",
    "get_phase1_validation_score_value",
    "scores_are_tied",
]
