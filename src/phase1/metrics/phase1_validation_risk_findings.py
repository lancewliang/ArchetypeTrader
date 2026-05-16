"""Phase I checkpoint-level cross-layer risk findings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from .phase1_metric_results import (
    Phase1LayerResult,
    Phase1MetricResult,
    Phase1RiskFinding,
)
from .phase1_validation_data_schema import Phase1CodeDiagnostic
from .phase1_validation_label_predictability import Phase1LabelPredictabilityPayload
from .phase1_validation_oracle_profitability import Phase1OracleProfitabilityPayload


def build_phase1_risk_findings(
    *,
    layers: Sequence[Phase1LayerResult],
    code_diagnostics: Sequence[Phase1CodeDiagnostic],
    drift_diagnostics: Mapping[str, Phase1MetricResult],
    oracle_profitability_payload: Phase1OracleProfitabilityPayload | None = None,
    label_predictability_payload: Phase1LabelPredictabilityPayload | None = None,
    max_findings: int = 12,
) -> tuple[Phase1RiskFinding, ...]:
    """构建 checkpoint 级风险定位。

    使用场景:
        五层 rules、score、drift diagnostics 都完成后调用。该函数只消费已有
        判定结果和 payload，不参与 hard gate，也不重新计算 raw metrics。
    """

    findings: list[Phase1RiskFinding] = []
    findings.extend(_metric_findings(layers))
    findings.extend(_drift_findings(drift_diagnostics))
    findings.extend(_code_findings(code_diagnostics))
    findings.extend(
        _payload_findings(
            oracle_profitability_payload=oracle_profitability_payload,
            label_predictability_payload=label_predictability_payload,
        )
    )
    findings.extend(_near_threshold_findings(layers))
    return tuple(_dedupe_findings(findings)[:max_findings])


def _metric_findings(
    layers: Sequence[Phase1LayerResult],
) -> list[Phase1RiskFinding]:
    """从 hard gate fail/skip/warn metric 生成风险。"""

    findings: list[Phase1RiskFinding] = []
    for layer in layers:
        for metric in layer.metrics:
            if metric.severity not in {"fail", "skip", "warn"}:
                continue
            severity = "fail" if metric.severity in {"fail", "skip"} else "warn"
            findings.append(
                Phase1RiskFinding(
                    severity=severity,
                    title=f"{layer.name}.{metric.name} {metric.severity.upper()}",
                    reason=metric.message,
                    related_metrics=(metric.name,),
                    recommended_action=(
                        "先修复该 hard gate 指标，再重新评估 checkpoint。"
                        if severity == "fail"
                        else (
                            "复查该指标的边界样本和关联 code，确认 warning 是否可解释。"
                        )
                    ),
                )
            )
    return findings


def _drift_findings(
    drift_diagnostics: Mapping[str, Phase1MetricResult],
) -> list[Phase1RiskFinding]:
    """从 train/validation drift warning 生成风险。"""

    return [
        Phase1RiskFinding(
            severity="warn",
            title=f"Drift warning: {metric.name}",
            reason=metric.message,
            related_metrics=(metric.name,),
            recommended_action=(
                "对比 train/validation 的 morphology、motif、code usage 和收益分布，"
                "确认 checkpoint 是否对验证期行情结构不稳定。"
            ),
        )
        for metric in drift_diagnostics.values()
        if metric.severity == "warn"
    ]


def _code_findings(
    code_diagnostics: Sequence[Phase1CodeDiagnostic],
) -> list[Phase1RiskFinding]:
    """从 code-level diagnostics 生成弱 code / 坏 code 风险。"""

    findings: list[Phase1RiskFinding] = []
    for code in code_diagnostics:
        status = str(code.status).lower()
        negative_return = (
            code.decoded_mean_advantage is not None
            and code.decoded_mean_advantage < 0.0
        )
        weak_status = status not in {"pass", "ok", "healthy"}
        if not weak_status and not negative_return:
            continue
        related_metrics = _code_related_metrics(code)
        reason_parts = [
            f"code status={code.status}",
            f"support={code.support}",
        ]
        if code.decoded_mean_advantage is not None:
            reason_parts.append(
                f"decoded_mean_advantage={code.decoded_mean_advantage:.6g}"
            )
        if code.dominant_motif is not None:
            reason_parts.append(f"dominant_motif={code.dominant_motif}")
        if code.dominant_pair is not None:
            reason_parts.append(f"dominant_pair={code.dominant_pair}")
        findings.append(
            Phase1RiskFinding(
                severity="warn",
                title=f"Code {code.code_id} risk",
                reason="；".join(reason_parts),
                related_metrics=related_metrics,
                related_codes=(code.code_id,),
                related_pairs=(
                    (code.dominant_pair,) if code.dominant_pair is not None else ()
                ),
                recommended_action=(
                    "检查该 code 的 decoded action 序列、dominant pair 样本和 "
                    "per-code 盈利/手续费来源。"
                ),
            )
        )
    return findings


def _payload_findings(
    *,
    oracle_profitability_payload: Phase1OracleProfitabilityPayload | None,
    label_predictability_payload: Phase1LabelPredictabilityPayload | None,
) -> list[Phase1RiskFinding]:
    """从跨层 payload 中生成无法由单个 metric 表达的风险。"""

    findings: list[Phase1RiskFinding] = []
    if oracle_profitability_payload is not None:
        bad_codes = tuple(
            item.code_id
            for item in oracle_profitability_payload.per_code_profitability
            if not item.passed or item.mean_advantage < 0.0
        )
        if bad_codes:
            findings.append(
                Phase1RiskFinding(
                    severity="warn",
                    title="Per-code profitability risk",
                    reason=(
                        "存在 per-code 盈利性未通过或 mean advantage 为负的 active code"
                    ),
                    related_metrics=("per_code_profitability",),
                    related_codes=bad_codes,
                    recommended_action=(
                        "优先复查这些 code 的 assigned samples、decoded returns、"
                        "fee drag 和 dominant morphology-motif pair。"
                    ),
                )
            )
    if (
        label_predictability_payload is not None
        and label_predictability_payload.probe_predictability_gap > 0.15
    ):
        findings.append(
            Phase1RiskFinding(
                severity="warn",
                title="Label predictability generalization gap",
                reason=(
                    "probe train accuracy 明显高于 validation accuracy，"
                    f"gap={label_predictability_payload.probe_predictability_gap:.6g}"
                ),
                related_metrics=("label_predictability_gap",),
                recommended_action=(
                    "检查低频 code 的 recall、probe confusion matrix，以及 "
                    "Phase II selector 是否可能复现同类过拟合。"
                ),
            )
        )
    return findings


def _near_threshold_findings(
    layers: Sequence[Phase1LayerResult],
) -> list[Phase1RiskFinding]:
    """为接近阈值但仍 pass 的 metric 生成 info 风险。"""

    candidates: list[tuple[float, Phase1MetricResult]] = []
    for layer in layers:
        for metric in layer.metrics:
            distance = metric.distance_to_threshold
            if metric.severity != "pass" or distance is None or distance < 0.0:
                continue
            if not _is_near_threshold(metric):
                continue
            candidates.append((distance, metric))
    candidates.sort(key=lambda item: item[0])
    return [
        Phase1RiskFinding(
            severity="info",
            title=f"Near-threshold metric: {metric.name}",
            reason=(
                f"{metric.name} 已通过但距离阈值较近，"
                f"distance_to_threshold={metric.distance_to_threshold:.6g}"
            ),
            related_metrics=(metric.name,),
            recommended_action=(
                "在候选排序时保留该边界风险，并结合相邻 checkpoint 复查稳定性。"
            ),
        )
        for _, metric in candidates[:3]
        if metric.distance_to_threshold is not None
    ]


def _is_near_threshold(metric: Phase1MetricResult) -> bool:
    """判断 pass metric 是否接近阈值。"""

    distance = metric.distance_to_threshold
    if distance is None or not math.isfinite(distance):
        return False
    threshold = metric.threshold_value
    if isinstance(threshold, tuple):
        scale = max(abs(threshold[0]), abs(threshold[1]), 1.0)
    elif isinstance(threshold, int | float):
        scale = max(abs(float(threshold)), 1.0)
    else:
        return False
    return distance <= 0.05 * scale


def _code_related_metrics(code: Phase1CodeDiagnostic) -> tuple[str, ...]:
    """根据 code diagnostic 字段推断关联 metric。"""

    metrics: list[str] = []
    if code.dominant_morphology_ratio is not None and code.dominant_morphology_ratio < 0.35:
        metrics.append("dominant_morphology_ratio")
    if code.dominant_motif_ratio is not None and code.dominant_motif_ratio < 0.40:
        metrics.append("dominant_motif_ratio")
    if code.dominant_pair_ratio is not None and code.dominant_pair_ratio < 0.30:
        metrics.append("dominant_pair_ratio")
    if code.decoded_mean_advantage is not None and code.decoded_mean_advantage < 0.0:
        metrics.append("decoded_mean_advantage")
    if code.decoded_win_rate is not None and code.decoded_win_rate < 0.52:
        metrics.append("decoded_win_rate")
    if code.fee_drag is not None and code.fee_drag > 0.40:
        metrics.append("fee_drag")
    return tuple(metrics) or ("code_diagnostic_status",)


def _dedupe_findings(
    findings: Sequence[Phase1RiskFinding],
) -> list[Phase1RiskFinding]:
    """按主要字段去重，并保持首次出现顺序。"""

    seen: set[tuple[object, ...]] = set()
    deduped: list[Phase1RiskFinding] = []
    for finding in findings:
        key = (
            finding.severity,
            finding.title,
            finding.related_metrics,
            finding.related_codes,
            finding.related_pairs,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


__all__ = ["build_phase1_risk_findings"]
