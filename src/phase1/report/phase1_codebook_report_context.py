"""Phase I codebook validation report template context builder."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from ..metrics import (
    Phase1CodeDiagnostic,
    Phase1LayerResult,
    Phase1MetricResult,
    Phase1OracleProfitabilityPayload,
    Phase1RiskFinding,
    Phase1ValidationScore,
    Phase1VQInternalPayload,
)


JsonObject = dict[str, Any]


def _format_value(value: Any) -> str:
    """格式化 HTML 表格中的指标值。"""

    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Mapping) and "total_score" in value:
        return _format_value(value.get("total_score"))
    return str(value)


def _badge_class(severity: str | bool) -> str:
    """将状态映射为 HTML badge class。"""

    if isinstance(severity, bool):
        return "pass" if severity else "fail"
    if severity in {"pass", "fail", "warn", "skip"}:
        return severity
    return "warn"


@dataclass(frozen=True)
class Phase1CodebookReportContextBuilder:
    """Phase I codebook validation report 模板上下文构建器。"""

    title: str = "Phase I Codebook Validation Report"

    def build(self, payload: Mapping[str, Any]) -> JsonObject:
        """把 report payload 转成模板使用的展示模型。"""

        validation = payload["validation"]
        summary = payload["summary"]
        report = payload["report"]
        config = payload.get("config", {})
        artifacts = payload.get("artifacts", {})
        layers = tuple(
            Phase1LayerResult.from_dict(layer)
            for layer in validation.get("layers", ())
        )
        code_diagnostics = tuple(
            Phase1CodeDiagnostic.from_dict(item)
            for item in validation.get("code_diagnostics", ())
        )
        drift_diagnostics = {
            str(name): Phase1MetricResult.from_dict(item)
            for name, item in validation.get("drift_diagnostics", {}).items()
        }
        risk_findings = tuple(
            Phase1RiskFinding.from_dict(item)
            for item in validation.get("risk_findings", ())
        )
        vq_internal_payload = self._vq_internal_payload(
            validation.get("vq_internal_payload")
        )
        oracle_profitability_payload = self._oracle_profitability_payload(
            validation.get("oracle_profitability_payload")
        )
        validation_score = self._validation_score(validation.get("score"))

        passed = bool(summary.get("passed", False))
        failed_layers = summary.get("failed_layers", [])
        failed_text = ", ".join(str(layer) for layer in failed_layers) or "-"
        return {
            "page_title": str(report.get("title", self.title)),
            "header_title": self.title,
            "header": self._build_header_context(
                summary=summary,
                report=report,
                config=config,
                vq_internal_payload=vq_internal_payload,
                oracle_profitability_payload=oracle_profitability_payload,
            ),
            "report": {
                "generated_at": str(report.get("generated_at", "-")),
                "schema": str(report.get("schema", "-")),
            },
            "summary": {
                "checkpoint_id": str(summary.get("checkpoint_id", "-")),
                "stage": str(summary.get("stage", "-")),
                "epoch": str(summary.get("epoch", "-")),
                "score": _format_value(summary.get("score")),
                "failed_layers": failed_text,
                "layer_count": str(summary.get("layer_count", len(layers))),
                "code_diagnostic_count": str(summary.get("code_diagnostic_count", 0)),
                "risk_finding_count": str(summary.get("risk_finding_count", 0)),
                "badge_class": _badge_class(passed),
                "status_label": "PASS" if passed else "FAIL",
            },
            "layers": [self._build_layer_context(layer) for layer in layers],
            "code_diagnostics": [
                self._build_code_diagnostic_context(item)
                for item in code_diagnostics
            ],
            "oracle_profitability_kpis": self._build_oracle_profitability_kpis(
                validation.get("metrics", {})
            ),
            "oracle_cumulative_return_series": (
                self._build_oracle_cumulative_return_series(
                    oracle_profitability_payload
                )
            ),
            "per_code_profit_series": self._build_per_code_profit_series(
                code_diagnostics,
                oracle_profitability_payload,
            ),
            "pair_profitability_matrix": (
                self._build_pair_profitability_matrix(
                    oracle_profitability_payload
                )
            ),
            "code_distribution": self._build_code_distribution_context(
                vq_internal_payload
            )
            if vq_internal_payload is not None
            else [],
            "tie_breaker_rows": self._build_mapping_rows(
                validation.get("tie_breaker_metrics", {})
            ),
            "score_breakdown_rows": self._build_score_breakdown_rows(
                validation_score
            )
            if validation_score is not None
            else [],
            "drift_diagnostics": [
                self._build_metric_context(metric)
                for metric in drift_diagnostics.values()
            ],
            "risk_summary": self._build_risk_summary_context(risk_findings),
            "risk_findings": [
                self._build_risk_finding_context(finding)
                for finding in risk_findings
            ],
            "config_rows": self._build_mapping_rows(config),
            "artifact_rows": self._build_mapping_rows(artifacts),
        }

    def _build_header_context(
        self,
        *,
        summary: Mapping[str, Any],
        report: Mapping[str, Any],
        config: Mapping[str, Any],
        vq_internal_payload: Phase1VQInternalPayload | None,
        oracle_profitability_payload: Phase1OracleProfitabilityPayload | None,
    ) -> JsonObject:
        """构建报告页眉元数据。

        Pair/Batch/Horizon 属于运行上下文，历史 payload 里可能来自 metadata
        或 config；K/N_val 属于 validation 结果，优先从 VQ payload 自动推导。
        """

        pair = self._first_present(
            report,
            config,
            keys=("pair", "symbol", "instrument"),
        )
        batch = self._first_present(
            report,
            config,
            keys=("batch", "batch_id", "batchid", "train_batch_id"),
        )
        horizon = self._first_present(
            report,
            config,
            keys=("horizon", "horizon_length"),
        )
        codebook_size = self._header_codebook_size(
            vq_internal_payload=vq_internal_payload,
            config=config,
            report=report,
        )
        validation_sample_count = self._header_validation_sample_count(
            vq_internal_payload=vq_internal_payload,
            oracle_profitability_payload=oracle_profitability_payload,
            report=report,
            config=config,
        )
        header = {
            "pair": _format_value(pair),
            "batch": _format_value(batch),
            "checkpoint": str(summary.get("checkpoint_id", "-")),
            "k": _format_value(codebook_size),
            "n_val": _format_value(validation_sample_count),
            "horizon": _format_value(horizon),
            "generated_at": str(report.get("generated_at", "-")),
        }
        header["meta_items"] = [
            {"label": "Pair", "value": header["pair"]},
            {"label": "Batch", "value": header["batch"]},
            {"label": "Checkpoint", "value": header["checkpoint"]},
            {"label": "K", "value": header["k"]},
            {"label": "N_val", "value": header["n_val"]},
            {"label": "Horizon", "value": header["horizon"]},
        ]
        return header

    def _first_present(
        self,
        *payloads: Mapping[str, Any],
        keys: tuple[str, ...],
    ) -> Any:
        """按候选 key 顺序从多个 mapping 中读取第一个非空值。"""

        for payload in payloads:
            for key in keys:
                value = payload.get(key)
                if value not in (None, ""):
                    return value
        return None

    def _header_codebook_size(
        self,
        *,
        vq_internal_payload: Phase1VQInternalPayload | None,
        config: Mapping[str, Any],
        report: Mapping[str, Any],
    ) -> Any:
        """读取 header 中的 K，优先使用 validation 实际 codebook size。"""

        if (
            vq_internal_payload is not None
            and vq_internal_payload.codebook_size_available
        ):
            return vq_internal_payload.codebook_size
        return self._first_present(
            report,
            config,
            keys=("k", "K", "codebook_size", "num_archetypes"),
        )

    def _header_validation_sample_count(
        self,
        *,
        vq_internal_payload: Phase1VQInternalPayload | None,
        oracle_profitability_payload: Phase1OracleProfitabilityPayload | None,
        report: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Any:
        """读取 header 中的 N_val，优先使用 code distribution 统计样本数。"""

        if vq_internal_payload is not None:
            sample_count = vq_internal_payload.code_distribution_sample_count
            if sample_count > 0:
                return sample_count
        if (
            oracle_profitability_payload is not None
            and oracle_profitability_payload.decoded_returns
        ):
            return len(oracle_profitability_payload.decoded_returns)
        return self._first_present(
            report,
            config,
            keys=("n_val", "N_val", "validation_sample_count", "val_sample_count"),
        )

    def _build_layer_context(self, layer: Phase1LayerResult) -> JsonObject:
        """构建单个 validation layer 的模板上下文。"""

        failed = sum(1 for metric in layer.metrics if not metric.passed)
        return {
            "layer_id": str(layer.layer_id),
            "name": layer.name,
            "badge_class": _badge_class(layer.passed),
            "status_label": "PASS" if layer.passed else "FAIL",
            "metric_count": str(len(layer.metrics)),
            "failed_count": str(failed),
            "metrics": [
                self._build_metric_context(metric)
                for metric in layer.metrics
            ],
        }

    def _build_metric_context(self, metric: Phase1MetricResult) -> JsonObject:
        """构建单个 metric result 的模板上下文。"""

        return {
            "name": metric.name,
            "value": _format_value(metric.value),
            "threshold": metric.threshold,
            "threshold_value": _format_value(metric.threshold_value),
            "direction": str(metric.direction or "-"),
            "distance_to_threshold": _format_value(metric.distance_to_threshold),
            "badge_class": _badge_class(metric.severity),
            "severity_label": metric.severity.upper(),
            "message": metric.message,
        }

    def _build_code_diagnostic_context(
        self,
        item: Phase1CodeDiagnostic,
    ) -> JsonObject:
        """构建单个 code diagnostic 的模板上下文。"""

        return {
            "code_id": str(item.code_id),
            "support": str(item.support),
            "occupancy": _format_value(item.occupancy),
            "dominant_morphology": str(item.dominant_morphology or "-"),
            "dominant_morphology_ratio": _format_value(
                item.dominant_morphology_ratio
            ),
            "morphology_lift": _format_value(item.morphology_lift),
            "dominant_motif": str(item.dominant_motif or "-"),
            "dominant_motif_ratio": _format_value(item.dominant_motif_ratio),
            "dominant_pair": str(item.dominant_pair or "-"),
            "dominant_pair_ratio": _format_value(item.dominant_pair_ratio),
            "decoded_mean_advantage": _format_value(item.decoded_mean_advantage),
            "decoded_win_rate": _format_value(item.decoded_win_rate),
            "retention_ratio": _format_value(item.retention_ratio),
            "fee_drag": _format_value(item.fee_drag),
            "status": item.status,
            "badge_class": _badge_class(item.status),
        }

    def _build_code_distribution_context(
        self,
        payload: Phase1VQInternalPayload,
    ) -> list[JsonObject]:
        """构建 codebook 使用分布的模板上下文。"""

        distribution = [float(value) for value in payload.code_distribution]
        active_codes = {int(code_id) for code_id in payload.active_codes}

        return [
            {
                "code_id": str(code_id),
                "occupancy": _format_value(occupancy),
                "occupancy_percent": (
                    f"{occupancy * 100:.3g}%"
                    if math.isfinite(occupancy)
                    else _format_value(occupancy)
                ),
                "bar_width": (
                    f"{max(0.0, min(100.0, occupancy * 100.0)):.3g}%"
                    if math.isfinite(occupancy)
                    else "0%"
                ),
                "active": code_id in active_codes,
                "badge_class": "pass" if code_id in active_codes else "warn",
                "status_label": "ACTIVE" if code_id in active_codes else "INACTIVE",
            }
            for code_id, occupancy in enumerate(distribution)
        ]

    def _build_risk_finding_context(
        self,
        finding: Phase1RiskFinding,
    ) -> JsonObject:
        """构建单个 risk finding 的模板上下文。"""

        return {
            "severity": finding.severity,
            "badge_class": "fail" if finding.severity == "fail" else "warn",
            "title": finding.title,
            "reason": finding.reason,
            "related_metrics": ", ".join(finding.related_metrics) or "-",
            "related_codes": (
                ", ".join(str(code_id) for code_id in finding.related_codes) or "-"
            ),
            "related_pairs": ", ".join(finding.related_pairs) or "-",
            "recommended_action": finding.recommended_action,
        }

    def _build_risk_summary_context(
        self,
        findings: tuple[Phase1RiskFinding, ...],
    ) -> JsonObject:
        """把 risk findings 聚合成报告首页的三段式风险定位。"""

        if not findings:
            return {
                "has_findings": False,
                "severity": "info",
                "badge_class": "pass",
                "finding_count": "0",
                "primary_risk": "未发现阻断或警戒级风险。",
                "inspection_target": (
                    "无需优先 drill-down；保留 hard gate、per-code 和 drift "
                    "常规审计记录。"
                ),
                "recommendation": (
                    "当前 checkpoint 可按 hard gate 和 selector 结果进入后续候选流程。"
                ),
            }

        severity_rank = {"fail": 0, "warn": 1, "info": 2}
        _, primary = min(
            enumerate(findings),
            key=lambda item: (
                severity_rank.get(item[1].severity, 3),
                item[0],
            ),
        )
        return {
            "has_findings": True,
            "severity": primary.severity,
            "badge_class": "fail" if primary.severity == "fail" else "warn",
            "finding_count": str(len(findings)),
            "primary_risk": self._risk_primary_text(primary),
            "inspection_target": self._risk_inspection_target(primary),
            "recommendation": (
                primary.recommended_action
                or "保留该 finding 的风险说明，并复查关联样本与相邻 checkpoint。"
            ),
        }

    def _risk_primary_text(self, finding: Phase1RiskFinding) -> str:
        """构建三段式中的主要风险文本。"""

        if finding.reason:
            return f"{finding.title}: {finding.reason}"
        return finding.title

    def _risk_inspection_target(self, finding: Phase1RiskFinding) -> str:
        """根据 finding 的关联对象构建优先检查目标。"""

        targets: list[str] = []
        if finding.related_codes:
            targets.append(
                "codes "
                + ", ".join(str(code_id) for code_id in finding.related_codes)
            )
        if finding.related_pairs:
            targets.append("pairs " + ", ".join(finding.related_pairs))
        if finding.related_metrics:
            targets.append("metrics " + ", ".join(finding.related_metrics))
        if targets:
            return "优先检查 " + "；".join(targets) + "。"
        return "优先检查该 finding 对应的边界样本、动作序列和验证期 trace。"

    def _build_oracle_profitability_kpis(
        self,
        metrics_payload: Any,
    ) -> list[JsonObject]:
        """构建 oracle-label 收益卡 KPI 展示字段。"""

        oracle_metrics = self._nested_value(metrics_payload, "oracle_profitability")
        if oracle_metrics in (None, ""):
            return []

        definitions = (
            (
                "mean_decoded_advantage_vs_flat",
                "mean decoded advantage",
            ),
            (
                "random_label_relative_lift",
                "vs random uplift",
            ),
            (
                "top_5_contribution",
                "top 5% contribution",
            ),
            (
                "trimmed_decoded_advantage",
                "trimmed advantage",
            ),
        )
        rows: list[JsonObject] = []
        for key, label in definitions:
            value = self._nested_value(oracle_metrics, key)
            if value in (None, ""):
                continue
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "value": _format_value(value),
                }
            )
        return rows

    def _build_per_code_profit_series(
        self,
        code_diagnostics: tuple[Phase1CodeDiagnostic, ...],
        oracle_profitability_payload: Phase1OracleProfitabilityPayload | None,
    ) -> list[JsonObject]:
        """构建 per-code 盈利图表序列。"""

        if code_diagnostics:
            return [
                {
                    "code_id": str(item.code_id),
                    "label": f"code {item.code_id}",
                    "value": _format_value(item.decoded_mean_advantage),
                    "badge_class": self._profit_badge_class(
                        item.decoded_mean_advantage
                    ),
                }
                for item in code_diagnostics
                if item.decoded_mean_advantage is not None
            ]
        if oracle_profitability_payload is None:
            return []
        return [
            {
                "code_id": str(item.code_id),
                "label": f"code {item.code_id}",
                "value": _format_value(item.mean_advantage),
                "badge_class": self._profit_badge_class(item.mean_advantage),
            }
            for item in oracle_profitability_payload.per_code_profitability
        ]

    def _build_pair_profitability_matrix(
        self,
        oracle_profitability_payload: Phase1OracleProfitabilityPayload | None,
    ) -> JsonObject:
        """构建 morphology x motif decoded advantage 矩阵上下文。"""

        if oracle_profitability_payload is None:
            return {"morphologies": [], "motifs": [], "rows": [], "cells": []}
        cells = tuple(oracle_profitability_payload.pair_profitability_matrix)
        if not cells:
            return {"morphologies": [], "motifs": [], "rows": [], "cells": []}

        morphologies = sorted({cell.morphology for cell in cells})
        motifs = sorted({cell.motif for cell in cells})
        by_pair = {
            (cell.morphology, cell.motif): cell
            for cell in cells
        }
        rows: list[JsonObject] = []
        flat_cells: list[JsonObject] = []
        for morphology in morphologies:
            row_cells: list[JsonObject] = []
            for motif in motifs:
                cell = by_pair.get((morphology, motif))
                if cell is None:
                    cell_context = {
                        "morphology": morphology,
                        "motif": motif,
                        "support": "0",
                        "mean_decoded_advantage": "-",
                        "decoded_win_rate": "-",
                        "retention_ratio": "-",
                        "fee_drag": "-",
                        "badge_class": "skip",
                    }
                else:
                    cell_context = {
                        "morphology": morphology,
                        "motif": motif,
                        "support": str(cell.support),
                        "mean_decoded_advantage": _format_value(
                            cell.mean_decoded_advantage
                        ),
                        "decoded_win_rate": _format_value(cell.decoded_win_rate),
                        "retention_ratio": _format_value(cell.retention_ratio),
                        "fee_drag": _format_value(cell.fee_drag),
                        "badge_class": self._profit_badge_class(
                            cell.mean_decoded_advantage
                        ),
                    }
                row_cells.append(cell_context)
                flat_cells.append(cell_context)
            rows.append({"morphology": morphology, "cells": row_cells})
        return {
            "morphologies": morphologies,
            "motifs": motifs,
            "rows": rows,
            "cells": flat_cells,
        }

    def _build_oracle_cumulative_return_series(
        self,
        payload: Phase1OracleProfitabilityPayload | None,
    ) -> list[JsonObject]:
        """构建 oracle-label 累计收益曲线序列。"""

        if payload is None:
            return []
        definitions = (
            ("dp", "DP", payload.dp_returns),
            ("decoded", "Decoded", payload.decoded_returns),
            ("random_label", "Random label", payload.random_label_returns),
            ("flat", "Flat", payload.flat_returns),
        )
        return [
            {
                "key": key,
                "label": label,
                "points": self._cumulative_points(returns),
            }
            for key, label, returns in definitions
            if returns
        ]

    def _cumulative_points(self, returns: tuple[float, ...]) -> list[JsonObject]:
        """把逐样本 return 转为从 0 开始的累计曲线点。"""

        total = 0.0
        points = [{"step": "0", "value": _format_value(total)}]
        for index, value in enumerate(returns, start=1):
            total += float(value)
            points.append({"step": str(index), "value": _format_value(total)})
        return points

    def _vq_internal_payload(self, payload: Any) -> Phase1VQInternalPayload | None:
        """从 report payload 恢复第一层 VQ internal payload。"""

        if isinstance(payload, Phase1VQInternalPayload):
            return payload
        if isinstance(payload, Mapping):
            return Phase1VQInternalPayload.from_dict(payload)
        return None

    def _oracle_profitability_payload(
        self,
        payload: Any,
    ) -> Phase1OracleProfitabilityPayload | None:
        """从 report payload 恢复第三层 oracle profitability payload。"""

        if isinstance(payload, Phase1OracleProfitabilityPayload):
            return payload
        if isinstance(payload, Mapping):
            return Phase1OracleProfitabilityPayload.from_dict(payload)
        return None

    def _nested_value(self, payload: Any, key: str) -> Any:
        """兼容 Mapping 和 dataclass/object 的字段读取。"""

        if isinstance(payload, Mapping):
            return payload.get(key)
        return getattr(payload, key, None)

    def _profit_badge_class(self, value: float | None) -> str:
        """将 per-code 盈利值映射为展示状态。"""

        if value is None or not math.isfinite(value):
            return "warn"
        return "pass" if value >= 0.0 else "fail"

    def _build_mapping_rows(self, payload: Mapping[str, Any]) -> list[JsonObject]:
        """构建普通 key-value 表格上下文。"""

        return [
            {"key": str(key), "value": _format_value(value)}
            for key, value in payload.items()
        ]

    def _build_score_breakdown_rows(
        self,
        score: Phase1ValidationScore,
    ) -> list[JsonObject]:
        """构建综合 score 子项拆解表格上下文。"""

        return [
            {
                "name": component.name,
                "value": _format_value(component.value),
                "weight": _format_value(component.weight),
                "weighted_value": _format_value(component.weighted_value),
            }
            for component in score.components
        ]

    def _validation_score(self, payload: Any) -> Phase1ValidationScore | None:
        """从 report payload 恢复 Phase I validation score。"""

        if payload is None:
            return None
        if isinstance(payload, Phase1ValidationScore):
            return payload
        if isinstance(payload, Mapping):
            return Phase1ValidationScore.from_dict(payload)
        if isinstance(payload, int | float):
            return Phase1ValidationScore.from_float(float(payload))
        return None


__all__ = ["Phase1CodebookReportContextBuilder"]
