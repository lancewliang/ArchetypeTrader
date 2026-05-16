from __future__ import annotations

from src.phase1.metrics import (
    Phase1CodeDiagnostic,
    Phase1LayerResult,
    Phase1MetricResult,
    Phase1PerCodeProfitability,
    Phase1OracleProfitabilityPayload,
    build_phase1_risk_findings,
)


def test_build_phase1_risk_findings_uses_metric_code_and_payload_signals() -> None:
    layers = (
        Phase1LayerResult(
            layer_id=3,
            name="oracle_profitability",
            passed=False,
            metrics=(
                Phase1MetricResult(
                    name="retention_ratio",
                    value=0.42,
                    threshold=">= 0.5",
                    severity="fail",
                    passed=False,
                    layer="oracle_profitability",
                    message="retention too low",
                    threshold_value=0.5,
                    direction="greater_is_better",
                    distance_to_threshold=-0.08,
                ),
            ),
        ),
        Phase1LayerResult(
            layer_id=1,
            name="vq_internal",
            passed=True,
            metrics=(
                Phase1MetricResult(
                    name="validation_action_accuracy",
                    value=0.86,
                    threshold=">= 0.85",
                    severity="pass",
                    passed=True,
                    layer="vq_internal",
                    message="ok but close",
                    threshold_value=0.85,
                    direction="greater_is_better",
                    distance_to_threshold=0.01,
                ),
            ),
        ),
    )
    code_diagnostics = (
        Phase1CodeDiagnostic(
            code_id=8,
            support=120,
            occupancy=0.08,
            dominant_morphology="neutral",
            dominant_morphology_ratio=0.31,
            morphology_lift=1.08,
            dominant_motif="mixed + middle + switching",
            dominant_motif_ratio=0.36,
            dominant_pair="neutral:mixed + middle + switching",
            dominant_pair_ratio=0.25,
            decoded_mean_advantage=-0.05,
            decoded_win_rate=0.48,
            retention_ratio=0.2,
            fee_drag=0.45,
            status="warn",
        ),
    )
    oracle_payload = Phase1OracleProfitabilityPayload(
        per_code_profitability=(
            Phase1PerCodeProfitability(
                code_id=8,
                mean_advantage=-0.05,
                win_rate=0.48,
                retention_ratio=0.2,
                fee_drag=0.45,
                passed=False,
            ),
        ),
        decoded_returns=(-0.05,),
        dp_returns=(0.1,),
        flat_returns=(0.0,),
        random_label_returns=(0.0,),
        random_seed=7,
    )

    findings = build_phase1_risk_findings(
        layers=layers,
        code_diagnostics=code_diagnostics,
        drift_diagnostics={
            "motif_distribution_kl": Phase1MetricResult(
                name="motif_distribution_kl",
                value=0.23,
                threshold="warn if > 0.2",
                severity="warn",
                passed=True,
                layer="drift",
                message="motif drift",
                threshold_value=0.2,
                direction="less_is_better",
                distance_to_threshold=-0.03,
            )
        },
        oracle_profitability_payload=oracle_payload,
    )

    titles = {finding.title for finding in findings}
    assert "oracle_profitability.retention_ratio FAIL" in titles
    assert "Drift warning: motif_distribution_kl" in titles
    assert "Code 8 risk" in titles
    assert "Per-code profitability risk" in titles
    assert "Near-threshold metric: validation_action_accuracy" in titles
    code_finding = next(finding for finding in findings if finding.title == "Code 8 risk")
    assert code_finding.related_codes == (8,)
    assert code_finding.related_pairs == ("neutral:mixed + middle + switching",)
