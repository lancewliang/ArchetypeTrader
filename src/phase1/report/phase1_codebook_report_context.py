"""Phase I codebook validation report template context builder."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .phase1_codebook_report_schema import (
    Phase1CodebookReportDocument,
    Phase1CodebookReportHtmlContext,
    Phase1ReportChartGridLine,
    Phase1ReportChartSeries,
    Phase1ReportCodeDiagnosticView,
    Phase1ReportCodeDistributionRow,
    Phase1ReportHeader,
    Phase1ReportHeaderItem,
    Phase1ReportKpiRow,
    Phase1ReportLayerView,
    Phase1ReportLineChart,
    Phase1ReportMappingRow,
    Phase1ReportMetricView,
    Phase1ReportPairProfitabilityCell,
    Phase1ReportPairProfitabilityMatrix,
    Phase1ReportPairProfitabilityRow,
    Phase1ReportProfitSeriesRow,
    Phase1ReportRiskFindingView,
    Phase1ReportRiskSummaryView,
    Phase1ReportScoreBreakdownRow,
    Phase1ReportSeries,
    Phase1ReportSeriesPoint,
    Phase1ReportSummaryView,
)
from ..metrics import (
    Phase1CodeDiagnostic,
    Phase1LayerResult,
    Phase1MetricResult,
    Phase1OracleProfitabilityPayload,
    Phase1RiskFinding,
    Phase1ValidationResult,
    Phase1ValidationScore,
    Phase1VQInternalPayload,
    get_phase1_validation_score_value,
)


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

    def build(
        self,
        payload: Phase1CodebookReportDocument,
    ) -> Phase1CodebookReportHtmlContext:
        """把 report payload 转成模板使用的展示模型。"""

        validation = payload.validation
        report = payload.report.to_dict()
        config = payload.config
        layers = tuple(validation.layers)
        code_diagnostics = tuple(validation.code_diagnostics)
        drift_diagnostics = dict(validation.drift_diagnostics)
        risk_findings = tuple(validation.risk_findings)
        vq_internal_payload = validation.vq_internal_payload
        oracle_profitability_payload = validation.oracle_profitability_payload
        validation_score = validation.score
        oracle_return_series = self._build_oracle_cumulative_return_series(
            oracle_profitability_payload
        )
        return Phase1CodebookReportHtmlContext(
            page_title=str(report.get("title", self.title)),
            header_title=self.title,
            header=self._build_header_context(
                validation=validation,
                report=report,
                config=config,
                vq_internal_payload=vq_internal_payload,
                oracle_profitability_payload=oracle_profitability_payload,
            ),
            report={
                "generated_at": str(report.get("generated_at", "-")),
                "schema": str(report.get("schema", "-")),
            },
            summary=self._build_summary_context(validation),
            layers=tuple(self._build_layer_context(layer) for layer in layers),
            code_diagnostics=tuple(
                self._build_code_diagnostic_context(item)
                for item in code_diagnostics
            ),
            oracle_profitability_kpis=self._build_oracle_profitability_kpis(
                validation.metrics
            ),
            oracle_cumulative_return_series=oracle_return_series,
            oracle_cumulative_return_chart=(
                self._build_oracle_cumulative_return_chart(
                    oracle_profitability_payload
                )
            ),
            per_code_profit_series=self._build_per_code_profit_series(
                code_diagnostics,
                oracle_profitability_payload,
            ),
            pair_profitability_matrix=(
                self._build_pair_profitability_matrix(
                    oracle_profitability_payload
                )
            ),
            code_distribution=self._build_code_distribution_context(
                vq_internal_payload
            )
            if vq_internal_payload is not None
            else (),
            tie_breaker_rows=self._build_mapping_rows(
                validation.tie_breaker_metrics.to_dict()
            ),
            score_breakdown_rows=self._build_score_breakdown_rows(
                validation_score
            )
            if validation_score is not None
            else (),
            drift_diagnostics=tuple(
                self._build_metric_context(metric)
                for metric in drift_diagnostics.values()
            ),
            risk_summary=self._build_risk_summary_context(risk_findings),
            risk_findings=tuple(
                self._build_risk_finding_context(finding)
                for finding in risk_findings
            ),
            config_rows=self._build_mapping_rows(config),
            artifact_rows=self._build_mapping_rows(payload.artifacts),
        )

    def _build_header_context(
        self,
        *,
        validation: Phase1ValidationResult,
        report: Mapping[str, Any],
        config: Mapping[str, Any],
        vq_internal_payload: Phase1VQInternalPayload | None,
        oracle_profitability_payload: Phase1OracleProfitabilityPayload | None,
    ) -> Phase1ReportHeader:
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
        pair_text = _format_value(pair)
        batch_text = _format_value(batch)
        checkpoint_text = validation.checkpoint_id
        k_text = _format_value(codebook_size)
        n_val_text = _format_value(validation_sample_count)
        horizon_text = _format_value(horizon)
        generated_at_text = str(report.get("generated_at", "-"))
        return Phase1ReportHeader(
            pair=pair_text,
            batch=batch_text,
            checkpoint=checkpoint_text,
            k=k_text,
            n_val=n_val_text,
            horizon=horizon_text,
            generated_at=generated_at_text,
            meta_items=(
                Phase1ReportHeaderItem(label="Pair", value=pair_text),
                Phase1ReportHeaderItem(label="Batch", value=batch_text),
                Phase1ReportHeaderItem(label="Checkpoint", value=checkpoint_text),
                Phase1ReportHeaderItem(label="K", value=k_text),
                Phase1ReportHeaderItem(label="N_val", value=n_val_text),
                Phase1ReportHeaderItem(label="Horizon", value=horizon_text),
            ),
        )

    def _build_summary_context(
        self,
        validation: Phase1ValidationResult,
    ) -> Phase1ReportSummaryView:
        """从 validation result 直接构建报告摘要展示模型。"""

        failed_text = ", ".join(str(layer) for layer in validation.failed_layers) or "-"
        passed = validation.passed
        return Phase1ReportSummaryView(
            checkpoint_id=validation.checkpoint_id,
            stage=validation.stage,
            epoch=str(validation.epoch),
            score=_format_value(get_phase1_validation_score_value(validation.score)),
            failed_layers=failed_text,
            layer_count=str(len(validation.layers)),
            code_diagnostic_count=str(len(validation.code_diagnostics)),
            risk_finding_count=str(len(validation.risk_findings)),
            badge_class=_badge_class(passed),
            status_label="PASS" if passed else "FAIL",
        )

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

    def _build_layer_context(self, layer: Phase1LayerResult) -> Phase1ReportLayerView:
        """构建单个 validation layer 的模板上下文。"""

        failed = sum(1 for metric in layer.metrics if not metric.passed)
        return Phase1ReportLayerView(
            layer_id=str(layer.layer_id),
            name=layer.name,
            badge_class=_badge_class(layer.passed),
            status_label="PASS" if layer.passed else "FAIL",
            metric_count=str(len(layer.metrics)),
            failed_count=str(failed),
            metrics=tuple(
                self._build_metric_context(metric)
                for metric in layer.metrics
            ),
        )

    def _build_metric_context(
        self,
        metric: Phase1MetricResult,
    ) -> Phase1ReportMetricView:
        """构建单个 metric result 的模板上下文。"""

        return Phase1ReportMetricView(
            name=metric.name,
            value=_format_value(metric.value),
            threshold=metric.threshold,
            threshold_value=_format_value(metric.threshold_value),
            direction=str(metric.direction or "-"),
            distance_to_threshold=_format_value(metric.distance_to_threshold),
            badge_class=_badge_class(metric.severity),
            severity_label=metric.severity.upper(),
            message=metric.message,
        )

    def _build_code_diagnostic_context(
        self,
        item: Phase1CodeDiagnostic,
    ) -> Phase1ReportCodeDiagnosticView:
        """构建单个 code diagnostic 的模板上下文。"""

        return Phase1ReportCodeDiagnosticView(
            code_id=str(item.code_id),
            support=str(item.support),
            occupancy=_format_value(item.occupancy),
            dominant_morphology=str(item.dominant_morphology or "-"),
            dominant_morphology_ratio=_format_value(
                item.dominant_morphology_ratio
            ),
            morphology_lift=_format_value(item.morphology_lift),
            dominant_motif=str(item.dominant_motif or "-"),
            dominant_motif_ratio=_format_value(item.dominant_motif_ratio),
            dominant_pair=str(item.dominant_pair or "-"),
            dominant_pair_ratio=_format_value(item.dominant_pair_ratio),
            decoded_mean_advantage=_format_value(item.decoded_mean_advantage),
            decoded_win_rate=_format_value(item.decoded_win_rate),
            retention_ratio=_format_value(item.retention_ratio),
            fee_drag=_format_value(item.fee_drag),
            status=item.status,
            badge_class=_badge_class(item.status),
        )

    def _build_code_distribution_context(
        self,
        payload: Phase1VQInternalPayload,
    ) -> tuple[Phase1ReportCodeDistributionRow, ...]:
        """构建 codebook 使用分布的模板上下文。"""

        distribution = [float(value) for value in payload.code_distribution]
        active_codes = {int(code_id) for code_id in payload.active_codes}

        return tuple(
            Phase1ReportCodeDistributionRow(
                code_id=str(code_id),
                occupancy=_format_value(occupancy),
                occupancy_percent=(
                    f"{occupancy * 100:.3g}%"
                    if math.isfinite(occupancy)
                    else _format_value(occupancy)
                ),
                bar_width=(
                    f"{max(0.0, min(100.0, occupancy * 100.0)):.3g}%"
                    if math.isfinite(occupancy)
                    else "0%"
                ),
                active=code_id in active_codes,
                badge_class="pass" if code_id in active_codes else "warn",
                status_label="ACTIVE" if code_id in active_codes else "INACTIVE",
            )
            for code_id, occupancy in enumerate(distribution)
        )

    def _build_risk_finding_context(
        self,
        finding: Phase1RiskFinding,
    ) -> Phase1ReportRiskFindingView:
        """构建单个 risk finding 的模板上下文。"""

        return Phase1ReportRiskFindingView(
            severity=finding.severity,
            badge_class="fail" if finding.severity == "fail" else "warn",
            title=finding.title,
            reason=finding.reason,
            related_metrics=", ".join(finding.related_metrics) or "-",
            related_codes=(
                ", ".join(str(code_id) for code_id in finding.related_codes) or "-"
            ),
            related_pairs=", ".join(finding.related_pairs) or "-",
            recommended_action=finding.recommended_action,
        )

    def _build_risk_summary_context(
        self,
        findings: tuple[Phase1RiskFinding, ...],
    ) -> Phase1ReportRiskSummaryView:
        """把 risk findings 聚合成报告首页的三段式风险定位。"""

        if not findings:
            return Phase1ReportRiskSummaryView(
                has_findings=False,
                severity="info",
                badge_class="pass",
                finding_count="0",
                primary_risk="未发现阻断或警戒级风险。",
                inspection_target=(
                    "无需优先 drill-down；保留 hard gate、per-code 和 drift "
                    "常规审计记录。"
                ),
                recommendation=(
                    "当前 checkpoint 可按 hard gate 和 selector 结果进入后续候选流程。"
                ),
            )

        severity_rank = {"fail": 0, "warn": 1, "info": 2}
        _, primary = min(
            enumerate(findings),
            key=lambda item: (
                severity_rank.get(item[1].severity, 3),
                item[0],
            ),
        )
        return Phase1ReportRiskSummaryView(
            has_findings=True,
            severity=primary.severity,
            badge_class="fail" if primary.severity == "fail" else "warn",
            finding_count=str(len(findings)),
            primary_risk=self._risk_primary_text(primary),
            inspection_target=self._risk_inspection_target(primary),
            recommendation=(
                primary.recommended_action
                or "保留该 finding 的风险说明，并复查关联样本与相邻 checkpoint。"
            ),
        )

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
    ) -> tuple[Phase1ReportKpiRow, ...]:
        """构建 oracle-label 收益卡 KPI 展示字段。"""

        oracle_metrics = self._nested_value(metrics_payload, "oracle_profitability")
        if oracle_metrics in (None, ""):
            return ()

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
        rows: list[Phase1ReportKpiRow] = []
        for key, label in definitions:
            value = self._nested_value(oracle_metrics, key)
            if value in (None, ""):
                continue
            rows.append(
                Phase1ReportKpiRow(
                    key=key,
                    label=label,
                    value=_format_value(value),
                )
            )
        return tuple(rows)

    def _build_per_code_profit_series(
        self,
        code_diagnostics: tuple[Phase1CodeDiagnostic, ...],
        oracle_profitability_payload: Phase1OracleProfitabilityPayload | None,
    ) -> tuple[Phase1ReportProfitSeriesRow, ...]:
        """构建 per-code 盈利图表序列。"""

        if code_diagnostics:
            return tuple(
                Phase1ReportProfitSeriesRow(
                    code_id=str(item.code_id),
                    label=f"code {item.code_id}",
                    value=_format_value(item.decoded_mean_advantage),
                    badge_class=self._profit_badge_class(
                        item.decoded_mean_advantage
                    ),
                )
                for item in code_diagnostics
                if item.decoded_mean_advantage is not None
            )
        if oracle_profitability_payload is None:
            return ()
        return tuple(
            Phase1ReportProfitSeriesRow(
                code_id=str(item.code_id),
                label=f"code {item.code_id}",
                value=_format_value(item.mean_advantage),
                badge_class=self._profit_badge_class(item.mean_advantage),
            )
            for item in oracle_profitability_payload.per_code_profitability
        )

    def _build_pair_profitability_matrix(
        self,
        oracle_profitability_payload: Phase1OracleProfitabilityPayload | None,
    ) -> Phase1ReportPairProfitabilityMatrix:
        """构建 morphology x motif decoded advantage 矩阵上下文。"""

        if oracle_profitability_payload is None:
            return Phase1ReportPairProfitabilityMatrix(
                morphologies=(),
                motifs=(),
                rows=(),
                cells=(),
            )
        cells = tuple(oracle_profitability_payload.pair_profitability_matrix)
        if not cells:
            return Phase1ReportPairProfitabilityMatrix(
                morphologies=(),
                motifs=(),
                rows=(),
                cells=(),
            )

        morphologies = tuple(sorted({cell.morphology for cell in cells}))
        motifs = tuple(sorted({cell.motif for cell in cells}))
        by_pair = {
            (cell.morphology, cell.motif): cell
            for cell in cells
        }
        max_abs_advantage = max(
            (
                abs(cell.mean_decoded_advantage)
                for cell in cells
                if math.isfinite(cell.mean_decoded_advantage)
            ),
            default=1.0,
        )
        if max_abs_advantage <= 0.0:
            max_abs_advantage = 1.0
        rows: list[Phase1ReportPairProfitabilityRow] = []
        flat_cells: list[Phase1ReportPairProfitabilityCell] = []
        for morphology in morphologies:
            row_cells: list[Phase1ReportPairProfitabilityCell] = []
            for motif in motifs:
                cell = by_pair.get((morphology, motif))
                if cell is None:
                    cell_context = Phase1ReportPairProfitabilityCell(
                        morphology=morphology,
                        motif=motif,
                        support="0",
                        mean_decoded_advantage="-",
                        decoded_win_rate="-",
                        retention_ratio="-",
                        fee_drag="-",
                        badge_class="skip",
                        display_value="-",
                        background_color="#eef1f5",
                        text_color="#525866",
                        tooltip=f"{morphology} / {motif}: no validation samples.",
                    )
                else:
                    mean_advantage = cell.mean_decoded_advantage
                    cell_context = Phase1ReportPairProfitabilityCell(
                        morphology=morphology,
                        motif=motif,
                        support=str(cell.support),
                        mean_decoded_advantage=_format_value(
                            mean_advantage
                        ),
                        decoded_win_rate=_format_value(cell.decoded_win_rate),
                        retention_ratio=_format_value(cell.retention_ratio),
                        fee_drag=_format_value(cell.fee_drag),
                        badge_class=self._profit_badge_class(
                            mean_advantage
                        ),
                        display_value=self._format_signed_value(mean_advantage),
                        background_color=self._heatmap_background(
                            mean_advantage,
                            max_abs_advantage,
                        ),
                        text_color=self._heatmap_text_color(
                            mean_advantage,
                            max_abs_advantage,
                        ),
                        tooltip=(
                            f"{morphology} / {motif}: "
                            f"mean decoded advantage "
                            f"{self._format_signed_value(mean_advantage)}, "
                            f"support {cell.support}, win rate "
                            f"{_format_value(cell.decoded_win_rate)}, "
                            f"retention {_format_value(cell.retention_ratio)}, "
                            f"fee drag {_format_value(cell.fee_drag)}."
                        ),
                    )
                row_cells.append(cell_context)
                flat_cells.append(cell_context)
            rows.append(
                Phase1ReportPairProfitabilityRow(
                    morphology=morphology,
                    cells=tuple(row_cells),
                )
            )
        return Phase1ReportPairProfitabilityMatrix(
            morphologies=morphologies,
            motifs=motifs,
            rows=tuple(rows),
            cells=tuple(flat_cells),
            grid_template_columns=(
                f"minmax(132px, 1fr) repeat({len(motifs)}, minmax(118px, 1fr))"
            ),
            legend_min=f"-{_format_value(max_abs_advantage)}",
            legend_max=f"+{_format_value(max_abs_advantage)}",
        )

    def _build_oracle_cumulative_return_series(
        self,
        payload: Phase1OracleProfitabilityPayload | None,
    ) -> tuple[Phase1ReportSeries, ...]:
        """构建 oracle-label 累计收益曲线序列。"""

        if payload is None:
            return ()
        definitions = (
            ("dp", "DP", payload.dp_returns),
            ("decoded", "Decoded", payload.decoded_returns),
            ("random_label", "Random label", payload.random_label_returns),
            ("flat", "Flat", payload.flat_returns),
        )
        return tuple(
            Phase1ReportSeries(
                key=key,
                label=label,
                points=self._cumulative_points(returns),
            )
            for key, label, returns in definitions
            if returns
        )

    def _build_oracle_cumulative_return_chart(
        self,
        payload: Phase1OracleProfitabilityPayload | None,
    ) -> Phase1ReportLineChart:
        """构建 oracle-label 累计收益静态 SVG 折线图。"""

        if payload is None:
            return Phase1ReportLineChart()
        definitions = (
            (
                "dp",
                "DP",
                "var(--blue)",
                payload.dp_returns,
                "DP teacher 累计收益，是第一阶段示范轨迹的收益参照。",
            ),
            (
                "decoded",
                "Decoded",
                "var(--pass)",
                payload.decoded_returns,
                "assigned label 经 frozen decoder 执行后的累计收益。",
            ),
            (
                "random_label",
                "Random label",
                "var(--rose)",
                payload.random_label_returns,
                "随机 label 基准累计收益，用于检查 label assignment 的信息量。",
            ),
            (
                "flat",
                "Flat",
                "var(--skip)",
                payload.flat_returns,
                "空仓基准累计收益。",
            ),
        )
        raw_series = [
            (key, label, color, self._cumulative_values(returns), tooltip)
            for key, label, color, returns, tooltip in definitions
            if returns
        ]
        finite_values = [
            value
            for _, _, _, values, _ in raw_series
            for value in values
            if math.isfinite(value)
        ]
        if not raw_series or not finite_values:
            return Phase1ReportLineChart()

        y_min = min(finite_values)
        y_max = max(finite_values)
        if y_min == y_max:
            padding = max(1.0, abs(y_min) * 0.05)
            y_min -= padding
            y_max += padding
        else:
            padding = (y_max - y_min) * 0.06
            y_min -= padding
            y_max += padding

        width = 820.0
        height = 330.0
        left = 52.0
        right = 22.0
        top = 24.0
        bottom = 38.0
        plot_width = width - left - right
        plot_height = height - top - bottom

        def x_coord(step: int, max_step: int) -> float:
            if max_step <= 0:
                return left
            return left + step * plot_width / max_step

        def y_coord(value: float) -> float:
            return top + (y_max - value) * plot_height / (y_max - y_min)

        grid_lines = tuple(
            Phase1ReportChartGridLine(
                y=self._format_svg_number(
                    top + index * plot_height / 4.0
                ),
                label=_format_value(y_max - index * (y_max - y_min) / 4.0),
            )
            for index in range(5)
        )
        chart_series: list[Phase1ReportChartSeries] = []
        for key, label, color, values, tooltip in raw_series:
            max_step = len(values) - 1
            sampled_indices = self._chart_sample_indices(len(values))
            points = " ".join(
                (
                    f"{self._format_svg_number(x_coord(index, max_step))},"
                    f"{self._format_svg_number(y_coord(values[index]))}"
                )
                for index in sampled_indices
                if math.isfinite(values[index])
            )
            if not points:
                continue
            chart_series.append(
                Phase1ReportChartSeries(
                    key=key,
                    label=label,
                    color=color,
                    points=points,
                    end_value=_format_value(values[-1]),
                    tooltip=tooltip,
                )
            )

        return Phase1ReportLineChart(
            width=str(int(width)),
            height=str(int(height)),
            grid_lines=grid_lines,
            series=tuple(chart_series),
            y_min=_format_value(y_min),
            y_max=_format_value(y_max),
            x_axis_label="validation horizon order",
        )

    def _cumulative_points(
        self,
        returns: tuple[float, ...],
    ) -> tuple[Phase1ReportSeriesPoint, ...]:
        """把逐样本 return 转为从 0 开始的累计曲线点。"""

        total = 0.0
        points = [Phase1ReportSeriesPoint(step="0", value=_format_value(total))]
        for index, value in enumerate(returns, start=1):
            total += float(value)
            points.append(
                Phase1ReportSeriesPoint(
                    step=str(index),
                    value=_format_value(total),
                )
            )
        return tuple(points)

    def _cumulative_values(self, returns: tuple[float, ...]) -> tuple[float, ...]:
        """把逐样本 return 转成累计数值序列，供 SVG 图表使用。"""

        total = 0.0
        values = [total]
        for value in returns:
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                total += numeric_value
            values.append(total)
        return tuple(values)

    def _chart_sample_indices(self, value_count: int) -> tuple[int, ...]:
        """限制 SVG 点数，避免长验证集把 HTML 膨胀成全量点位表。"""

        if value_count <= 0:
            return ()
        max_points = 320
        if value_count <= max_points:
            return tuple(range(value_count))
        last_index = value_count - 1
        step = last_index / (max_points - 1)
        indices = {
            0,
            last_index,
            *(round(index * step) for index in range(max_points)),
        }
        return tuple(sorted(indices))

    def _heatmap_background(self, value: float | None, max_abs: float) -> str:
        """按收益正负和强度生成热力图背景色。"""

        if value is None or not math.isfinite(value):
            return "#eef1f5"
        alpha = min(0.92, max(0.16, abs(value) / max_abs * 0.82 + 0.10))
        if value >= 0.0:
            return f"rgba(20, 122, 77, {alpha:.3g})"
        return f"rgba(180, 35, 24, {alpha:.3g})"

    def _heatmap_text_color(self, value: float | None, max_abs: float) -> str:
        """按背景强度选择热力图文字颜色。"""

        if value is None or not math.isfinite(value):
            return "#525866"
        intensity = abs(value) / max_abs
        return "#ffffff" if intensity >= 0.62 else "#0f172a"

    def _format_signed_value(self, value: float | None) -> str:
        """格式化带正号的收益值。"""

        if value is None:
            return "-"
        if not math.isfinite(value):
            return _format_value(value)
        prefix = "+" if value >= 0.0 else ""
        return f"{prefix}{_format_value(value)}"

    def _format_svg_number(self, value: float) -> str:
        """格式化 SVG 坐标，减少无意义小数。"""

        return f"{value:.2f}".rstrip("0").rstrip(".")

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

    def _build_mapping_rows(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[Phase1ReportMappingRow, ...]:
        """构建普通 key-value 表格上下文。"""

        return tuple(
            Phase1ReportMappingRow(key=str(key), value=_format_value(value))
            for key, value in payload.items()
        )

    def _build_score_breakdown_rows(
        self,
        score: Phase1ValidationScore,
    ) -> tuple[Phase1ReportScoreBreakdownRow, ...]:
        """构建综合 score 子项拆解表格上下文。"""

        return tuple(
            Phase1ReportScoreBreakdownRow(
                name=component.name,
                value=_format_value(component.value),
                weight=_format_value(component.weight),
                weighted_value=_format_value(component.weighted_value),
            )
            for component in score.components
        )


__all__ = ["Phase1CodebookReportContextBuilder"]
