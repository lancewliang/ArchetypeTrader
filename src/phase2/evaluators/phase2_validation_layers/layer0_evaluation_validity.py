"""Phase II validation Layer 0: evaluation validity raw metrics."""

from __future__ import annotations

from ...metrics import (
    Phase2EvaluationValidityMetrics,
    Phase2EvaluationValidityPayload,
    Phase2LayerComputation,
)


def compute_evaluation_validity_metrics(
    payload: Phase2EvaluationValidityPayload,
    *,
    deterministic_eval: bool,
    label_alignment_valid: bool,
    visible_state_contract_valid: bool,
) -> Phase2LayerComputation:
    """Compute Layer 0 evaluation validity metrics from aggregate counts."""

    sample_count = int(payload.num_samples)
    denominator = max(1, sample_count)
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
    return Phase2LayerComputation(
        layer_id=0,
        layer_name="evaluation_validity",
        metrics=metrics,
        extra_payload={"evaluation_validity_payload": payload},
    )


__all__ = ["compute_evaluation_validity_metrics"]
