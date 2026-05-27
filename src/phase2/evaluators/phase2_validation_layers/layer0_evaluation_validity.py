"""Phase II validation Layer 0: evaluation validity raw metrics."""

from __future__ import annotations

from ...metrics import (
    Phase2EvaluationValidityMetrics,
    Phase2EvaluationValidityPayload,
    Phase2Layer0EvaluationValidityComputation,
)


def compute_evaluation_validity_metrics(
    payload: Phase2EvaluationValidityPayload,
    *,
    deterministic_eval: bool,
    label_alignment_valid: bool,
    visible_state_contract_valid: bool,
) -> Phase2Layer0EvaluationValidityComputation:
    """Compute Layer 0 evaluation validity metrics from aggregate counts.

    算法:
        1. 读取 evaluator 已统计的样本总数和失败计数；
        2. 用 ``denominator = max(1, num_samples)`` 防止空 split 产生除零；
        3. 将失败计数转换成有效比例：
           ``valid_ratio = 1 - invalid_count / denominator``；
        4. 将 deterministic、label alignment、visible-state contract 三个布尔
           契约直接写入 metrics。

    公式:
        - ``valid_rollout_ratio = 1 - failed_rollout_count / N``
        - ``finite_reward_ratio = 1 - non_finite_reward_count / N``
        - ``valid_selected_code_ratio = 1 - invalid_selected_code_count / N``

    说明:
        Layer 0 不评价策略好坏，只确认评估结果是否可信。比例越接近 1 越好，
        三个布尔契约应全部为 True。
    """

    sample_count = int(payload.num_samples)
    denominator = max(1, sample_count)
    # 空 split 时 denominator 固定为 1，使 ratio 公式保持可计算；num_samples
    # 本身会在 rules 层触发样本数不足的失败。
    metrics = Phase2EvaluationValidityMetrics(
        num_samples=sample_count,
        valid_rollout_ratio=(
            1.0 - float(payload.failed_rollout_count) / float(denominator)
        ),
        finite_reward_ratio=(
            1.0 - float(payload.non_finite_reward_count) / float(denominator)
        ),
        valid_selected_code_ratio=(
            1.0 - float(payload.invalid_selected_code_count) / float(denominator)
        ),
        deterministic_eval=bool(deterministic_eval),
        label_alignment_valid=bool(label_alignment_valid),
        visible_state_contract_valid=bool(visible_state_contract_valid),
    )
    return Phase2Layer0EvaluationValidityComputation(
        metrics=metrics,
        evaluation_validity_payload=payload,
    )


__all__ = ["compute_evaluation_validity_metrics"]
