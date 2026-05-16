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

from collections.abc import Mapping, Sequence

from .phase1_metric_results import (
    Phase1LayerResult,
    Phase1MetricResult,
    Phase1ValidationResult,
)
from .phase1_validation_score import Phase1ValidationScoreLike
from .phase1_validation_data_schema import (
    Phase1CodeDiagnostic,
    Phase1TieBreakerMetrics,
    Phase1ValidationMetrics,
)
from .phase1_validation_behavior_quality import evaluate_behavior_quality_rules
from .phase1_validation_label_predictability import evaluate_label_predictability_rules
from .phase1_validation_oracle_profitability import evaluate_oracle_profitability_rules
from .phase1_validation_teacher_quality import evaluate_teacher_quality_rules
from .phase1_validation_vq_internal import evaluate_vq_internal_rules


def aggregate_validation_result(
    *,
    checkpoint_id: str,
    stage: str,
    epoch: int,
    layers: Sequence[Phase1LayerResult],
    metrics: Phase1ValidationMetrics,
    code_diagnostics: Sequence[Phase1CodeDiagnostic],
    drift_diagnostics: Mapping[str, Phase1MetricResult],
    score: Phase1ValidationScoreLike,
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
