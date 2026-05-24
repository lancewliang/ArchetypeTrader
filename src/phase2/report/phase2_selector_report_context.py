"""Phase II selector report template context builder."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .phase2_selector_report_schema import (
    DEFAULT_PHASE2_REPORT_TITLE,
    Phase2ReportBaselineRow,
    Phase2ReportChartGridLine,
    Phase2ReportChartSeries,
    Phase2ReportCodeUsageRow,
    Phase2ReportDocument,
    Phase2ReportHeader,
    Phase2ReportHeaderItem,
    Phase2ReportHtmlContext,
    Phase2ReportLayerView,
    Phase2ReportLineChart,
    Phase2ReportMappingRow,
    Phase2ReportPerCodeProfitabilityRow,
    Phase2ReportSeries,
    Phase2ReportSeriesPoint,
    Phase2ReportSummaryView,
    ensure_phase2_report_document,
    layer_result_to_view,
)
from ..metrics import Phase2LayerComputation, Phase2ValidationResult


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
    if isinstance(value, tuple | list):
        return ", ".join(_format_value(item) for item in value) or "-"
    return str(value)


def _format_percent(value: Any) -> str:
    """把 0-1 比例格式化为百分比文本。"""

    number = _as_float(value)
    if number is None:
        return _format_value(value)
    return f"{number * 100:.2f}%"


def _badge_class(status: str | bool | None) -> str:
    """将状态映射为 HTML badge class。"""

    if isinstance(status, bool):
        return "pass" if status else "fail"
    if status in {"pass", "fail", "warn", "skip"}:
        return status
    if status in {"active", "selected"}:
        return "pass"
    if status in {"dead", "inactive", "reference"}:
        return "skip"
    return "warn"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Best-effort conversion of dataclass-like report inputs to mappings."""

    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        return payload if isinstance(payload, Mapping) else {}
    if isinstance(value, Mapping):
        return value
    return {}


def _as_float(value: Any) -> float | None:
    """Best-effort float conversion."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _get_first(mapping: Mapping[str, Any], *keys: str) -> Any:
    """按候选 key 顺序读取第一个非空值。"""

    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


_CORE_METRIC_DESCRIPTIONS: dict[str, str] = {
    "mean_return": "selector greedy action 的平均 horizon return。",
    "median_return": "selector greedy action 的 return 中位数。",
    "sharpe_like": "horizon-level 风险调整收益指标。",
    "win_rate": "horizon return 大于 0 的样本比例。",
    "mean_turnover": "平均换手率或行为强度指标。",
}

_SERIES_LABELS: dict[str, str] = {
    "selector": "Selector",
    "kl": "Assigned Label",
    "assigned": "Assigned Label",
    "assigned_label": "Assigned Label",
    "random": "Random",
    "oracle": "Oracle",
    "hold": "Hold",
}

_SERIES_COLORS: dict[str, str] = {
    "selector": "#2563eb",
    "kl": "#0f766e",
    "assigned": "#0f766e",
    "assigned_label": "#0f766e",
    "random": "#a15c00",
    "oracle": "#7c3aed",
    "hold": "#be123c",
}


@dataclass(frozen=True)
class Phase2SelectorReportContextBuilder:
    """Build a static HTML template context from ``Phase2ReportDocument``."""

    title: str = DEFAULT_PHASE2_REPORT_TITLE

    def build(
        self,
        payload: Phase2ReportDocument | Mapping[str, Any],
    ) -> Phase2ReportHtmlContext:
        """将 report document 转换为模板展示模型。"""

        document = ensure_phase2_report_document(payload)
        validation = document.validation
        report_payload = self._report_payload(validation)
        cumulative_series = self._build_cumulative_return_series(report_payload)
        return Phase2ReportHtmlContext(
            page_title=document.report.title,
            header_title=self.title,
            header=self._build_header(document, validation),
            report={
                "generated_at": document.report.generated_at,
                "schema": document.report.schema,
            },
            summary=self._build_summary(document, validation),
            layers=self._build_layers(validation),
            core_metric_rows=self._build_core_metric_rows(validation),
            baseline_rows=self._build_baseline_rows(validation),
            per_code_profitability_rows=self._build_per_code_profitability_rows(
                report_payload
            ),
            code_usage_rows=self._build_code_usage_rows(report_payload),
            cumulative_return_series=cumulative_series,
            config_rows=self._build_mapping_rows(document.config),
            artifact_rows=self._build_mapping_rows(document.artifacts),
            cumulative_return_chart=self._build_cumulative_return_chart(
                cumulative_series
            ),
        )

    def _build_header(
        self,
        document: Phase2ReportDocument,
        validation: Phase2ValidationResult | None,
    ) -> Phase2ReportHeader:
        """构建页眉元数据。"""

        metadata = document.report.metadata
        config = document.config
        selection = document.selection
        layer0_payload = self._payload_mapping(validation, "evaluation_validity_payload")
        layer0_metrics = self._layer_metrics_mapping(validation, 0, "evaluation_validity")
        checkpoint = _get_first(
            selection,
            "selected_checkpoint_id",
            "checkpoint_id",
        )
        pair = _get_first(metadata, "pair", "symbol", "instrument")
        batch = _get_first(metadata, "batch", "batch_id", "train_batch_id", "run_id")
        k_value = _get_first(
            metadata,
            "k",
            "num_archetypes",
        ) or _get_first(config, "k", "num_archetypes", "codebook_size")
        if k_value is None:
            k_value = layer0_payload.get("num_archetypes")
        n_val = _get_first(metadata, "n_val", "num_samples")
        if n_val is None:
            n_val = layer0_metrics.get("num_samples") or layer0_payload.get("num_samples")
        horizon = _get_first(
            metadata,
            "horizon",
            "horizon_steps",
        ) or _get_first(config, "horizon", "horizon_steps", "sequence_length")
        meta_items = (
            Phase2ReportHeaderItem("epoch", _format_value(selection.get("selected_epoch"))),
            Phase2ReportHeaderItem(
                "split",
                _format_value(layer0_payload.get("split_name")),
            ),
            Phase2ReportHeaderItem(
                "schema",
                document.report.schema,
            ),
        )
        return Phase2ReportHeader(
            pair=_format_value(pair),
            batch=_format_value(batch),
            checkpoint=_format_value(checkpoint),
            k=_format_value(k_value),
            n_val=_format_value(n_val),
            horizon=_format_value(horizon),
            generated_at=document.report.generated_at,
            meta_items=meta_items,
        )

    def _build_summary(
        self,
        document: Phase2ReportDocument,
        validation: Phase2ValidationResult | None,
    ) -> Phase2ReportSummaryView:
        """构建首页摘要卡片。"""

        summary = document.summary
        selection = document.selection
        metrics = validation.metrics if validation is not None else None
        failed_layers = summary.get("failed_layers", ())
        passed = bool(summary.get("passed", validation is not None))
        selected_score = selection.get("selected_score")
        if selected_score is None and metrics is not None:
            selected_score = metrics.mean_return
        return Phase2ReportSummaryView(
            checkpoint_id=_format_value(
                selection.get("selected_checkpoint_id")
                or summary.get("selected_checkpoint_id")
            ),
            epoch=_format_value(
                selection.get("selected_epoch") or summary.get("selected_epoch")
            ),
            score=_format_value(selected_score),
            mean_return=_format_value(
                summary.get("mean_return")
                if summary.get("mean_return") is not None
                else getattr(metrics, "mean_return", None)
            ),
            sharpe_like=_format_value(
                summary.get("sharpe_like")
                if summary.get("sharpe_like") is not None
                else getattr(metrics, "sharpe_like", None)
            ),
            win_rate=_format_percent(
                summary.get("win_rate")
                if summary.get("win_rate") is not None
                else getattr(metrics, "win_rate", None)
            ),
            failed_layers=_format_value(failed_layers),
            layer_count=_format_value(
                summary.get("layer_count")
                if summary.get("layer_count") is not None
                else len(validation.layers) if validation is not None else None
            ),
            badge_class=_badge_class(summary.get("status") or passed),
            status_label="PASS" if passed else "FAIL",
            reason=str(summary.get("reason", "selected" if passed else "blocked")),
        )

    def _build_layers(
        self,
        validation: Phase2ValidationResult | None,
    ) -> tuple[Phase2ReportLayerView, ...]:
        """构建 validation layer 视图。"""

        if validation is None:
            return ()
        return tuple(layer_result_to_view(layer) for layer in validation.layers)

    def _build_core_metric_rows(
        self,
        validation: Phase2ValidationResult | None,
    ) -> tuple[Phase2ReportMappingRow, ...]:
        """构建 selector 核心指标行。"""

        if validation is None:
            return ()
        metrics = validation.metrics.to_dict()
        return tuple(
            Phase2ReportMappingRow(
                key=str(key),
                value=_format_value(value),
                description=_CORE_METRIC_DESCRIPTIONS.get(str(key), ""),
            )
            for key, value in metrics.items()
        )

    def _build_baseline_rows(
        self,
        validation: Phase2ValidationResult | None,
    ) -> tuple[Phase2ReportBaselineRow, ...]:
        """构建 selector 与 baseline 对比行。"""

        if validation is None:
            return ()
        baseline_metrics = self._layer_metrics_mapping(validation, 2, "baseline_uplift")
        if not baseline_metrics:
            return ()
        rows = (
            self._baseline_row(
                baseline="Assigned Label",
                mean_return=baseline_metrics.get("assigned_mean_return"),
                uplift=baseline_metrics.get("uplift_vs_assigned"),
                beat_rate=baseline_metrics.get("beat_assigned_rate"),
                status_metric=baseline_metrics.get("uplift_vs_assigned"),
            ),
            self._baseline_row(
                baseline="Random",
                mean_return=baseline_metrics.get("random_mean_return"),
                uplift=baseline_metrics.get("uplift_vs_random"),
                beat_rate=baseline_metrics.get("beat_random_rate"),
                status_metric=baseline_metrics.get("uplift_vs_random"),
            ),
            self._baseline_row(
                baseline="Oracle",
                mean_return=baseline_metrics.get("oracle_mean_return"),
                uplift=(
                    -float(regret)
                    if (regret := _as_float(baseline_metrics.get("regret_to_oracle")))
                    is not None
                    else None
                ),
                beat_rate=baseline_metrics.get("oracle_capture_ratio"),
                status_metric=None,
                reference=True,
            ),
        )
        return tuple(row for row in rows if row.mean_return != "-")

    def _baseline_row(
        self,
        *,
        baseline: str,
        mean_return: Any,
        uplift: Any,
        beat_rate: Any,
        status_metric: Any,
        reference: bool = False,
    ) -> Phase2ReportBaselineRow:
        """构建单个 baseline 行。"""

        status_value = _as_float(status_metric)
        if reference:
            badge = "skip"
            status = "REFERENCE"
        elif status_value is None:
            badge = "warn"
            status = "WARN"
        elif status_value >= 0.0:
            badge = "pass"
            status = "PASS"
        else:
            badge = "fail"
            status = "FAIL"
        return Phase2ReportBaselineRow(
            baseline=baseline,
            mean_return=_format_value(mean_return),
            uplift=_format_value(uplift),
            beat_rate=_format_percent(beat_rate),
            status=status,
            badge_class=badge,
        )

    def _build_per_code_profitability_rows(
        self,
        report_payload: Mapping[str, Any],
    ) -> tuple[Phase2ReportPerCodeProfitabilityRow, ...]:
        """构建 per-code 盈利对比行。"""

        rows = report_payload.get("per_code_profitability_comparison", ())
        if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
            return ()
        result: list[Phase2ReportPerCodeProfitabilityRow] = []
        for item in rows:
            row = _as_mapping(item)
            if not row:
                continue
            status = str(row.get("status") or ("active" if row.get("is_active") else "reference"))
            result.append(
                Phase2ReportPerCodeProfitabilityRow(
                    code_id=_format_value(row.get("code_id")),
                    selector_support=_format_value(
                        row.get("selector_support", row.get("selector_count"))
                    ),
                    kl_support=_format_value(row.get("kl_support", row.get("kl_count"))),
                    selector_mean_return=_format_value(
                        row.get("selector_mean_return")
                    ),
                    kl_mean_return=_format_value(row.get("kl_mean_return")),
                    selector_win_rate=_format_percent(
                        row.get("selector_win_rate")
                    ),
                    kl_win_rate=_format_percent(row.get("kl_win_rate")),
                    uplift_vs_kl=_format_value(row.get("uplift_vs_kl")),
                    badge_class=_badge_class(status),
                    status_label=status.upper(),
                )
            )
        return tuple(result)

    def _build_code_usage_rows(
        self,
        report_payload: Mapping[str, Any],
    ) -> tuple[Phase2ReportCodeUsageRow, ...]:
        """构建 code usage 分布行。"""

        comparison = report_payload.get("codebook_usage_comparison")
        if isinstance(comparison, Sequence) and not isinstance(comparison, str | bytes):
            return tuple(
                row
                for item in comparison
                if (row := self._code_usage_row_from_mapping(_as_mapping(item)))
                is not None
            )

        distribution = _as_mapping(report_payload.get("codebook_usage_distribution"))
        selector_counts = self._count_map(distribution.get("selector"))
        kl_counts = self._count_map(distribution.get("kl"))
        if not selector_counts and not kl_counts:
            return ()
        selector_total = sum(selector_counts.values())
        kl_total = sum(kl_counts.values())
        code_ids = sorted(set(selector_counts) | set(kl_counts))
        max_selector_count = max(selector_counts.values(), default=0)
        rows: list[Phase2ReportCodeUsageRow] = []
        for code_id in code_ids:
            selector_count = selector_counts.get(code_id, 0)
            kl_count = kl_counts.get(code_id, 0)
            selector_ratio = (
                selector_count / selector_total if selector_total > 0 else 0.0
            )
            kl_ratio = kl_count / kl_total if kl_total > 0 else 0.0
            rows.append(
                Phase2ReportCodeUsageRow(
                    code_id=str(code_id),
                    selector_count=str(selector_count),
                    selector_ratio=_format_percent(selector_ratio),
                    kl_count=str(kl_count),
                    kl_ratio=_format_percent(kl_ratio),
                    ratio_delta=_format_percent(selector_ratio - kl_ratio),
                    bar_width=(
                        f"{selector_count / max_selector_count * 100:.2f}%"
                        if max_selector_count > 0
                        else "0%"
                    ),
                    badge_class="pass" if selector_count > 0 else "skip",
                    status_label="ACTIVE" if selector_count > 0 else "INACTIVE",
                )
            )
        return tuple(rows)

    def _code_usage_row_from_mapping(
        self,
        row: Mapping[str, Any],
    ) -> Phase2ReportCodeUsageRow | None:
        """从聚合 usage comparison row 构建视图行。"""

        if not row:
            return None
        selector_ratio = _as_float(row.get("selector_ratio"))
        status = str(row.get("status") or ("active" if selector_ratio else "inactive"))
        return Phase2ReportCodeUsageRow(
            code_id=_format_value(row.get("code_id")),
            selector_count=_format_value(row.get("selector_count")),
            selector_ratio=_format_percent(row.get("selector_ratio")),
            kl_count=_format_value(row.get("kl_count")),
            kl_ratio=_format_percent(row.get("kl_ratio")),
            ratio_delta=_format_percent(row.get("ratio_delta")),
            bar_width=(
                f"{selector_ratio * 100:.2f}%"
                if selector_ratio is not None
                else "0%"
            ),
            badge_class=_badge_class(status),
            status_label=status.upper(),
        )

    def _build_cumulative_return_series(
        self,
        report_payload: Mapping[str, Any],
    ) -> tuple[Phase2ReportSeries, ...]:
        """构建累计收益序列数据。"""

        curves = _as_mapping(
            report_payload.get("cumulative_return_curves")
            or report_payload.get("oracle_label_cumulative_returns")
        )
        series: list[Phase2ReportSeries] = []
        for key, values in curves.items():
            if isinstance(values, str | bytes) or not isinstance(values, Sequence):
                continue
            points = tuple(
                Phase2ReportSeriesPoint(
                    step=str(index),
                    value=_format_value(value),
                )
                for index, value in enumerate(values)
            )
            if not points:
                continue
            key_text = str(key)
            series.append(
                Phase2ReportSeries(
                    key=key_text,
                    label=_SERIES_LABELS.get(key_text, key_text),
                    points=points,
                )
            )
        return tuple(series)

    def _build_cumulative_return_chart(
        self,
        series: tuple[Phase2ReportSeries, ...],
    ) -> Phase2ReportLineChart:
        """构建静态 SVG 累计收益曲线。"""

        chart_width = 820
        chart_height = 330
        left = 48.0
        right = 790.0
        top = 24.0
        bottom = 300.0
        numeric_series: list[tuple[Phase2ReportSeries, list[float]]] = []
        all_values: list[float] = []
        for item in series:
            values = [
                value
                for point in item.points
                if (value := _as_float(point.value)) is not None
            ]
            if not values:
                continue
            numeric_series.append((item, values))
            all_values.extend(values)
        if not numeric_series or not all_values:
            return Phase2ReportLineChart(title="累计收益曲线")

        y_min = min(all_values)
        y_max = max(all_values)
        if math.isclose(y_min, y_max):
            padding = abs(y_min) * 0.1 or 1.0
            y_min -= padding
            y_max += padding
        else:
            padding = (y_max - y_min) * 0.08
            y_min -= padding
            y_max += padding

        def x_at(index: int, length: int) -> float:
            if length <= 1:
                return left
            return left + (right - left) * index / (length - 1)

        def y_at(value: float) -> float:
            return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

        chart_series: list[Phase2ReportChartSeries] = []
        for item, values in numeric_series:
            points = " ".join(
                f"{x_at(index, len(values)):.2f},{y_at(value):.2f}"
                for index, value in enumerate(values)
            )
            chart_series.append(
                Phase2ReportChartSeries(
                    key=item.key,
                    label=item.label,
                    color=_SERIES_COLORS.get(item.key, "#475467"),
                    points=points,
                    end_value=_format_value(values[-1]),
                    tooltip=f"{item.label}: {_format_value(values[-1])}",
                )
            )

        grid_lines = tuple(
            Phase2ReportChartGridLine(
                y=f"{y_at(value):.2f}",
                label=_format_value(value),
            )
            for value in self._grid_values(y_min, y_max)
        )
        return Phase2ReportLineChart(
            title="累计收益曲线",
            width=str(chart_width),
            height=str(chart_height),
            grid_lines=grid_lines,
            series=tuple(chart_series),
            y_min=_format_value(y_min),
            y_max=_format_value(y_max),
        )

    def _build_mapping_rows(
        self,
        mapping: Mapping[str, Any],
    ) -> tuple[Phase2ReportMappingRow, ...]:
        """构建通用 mapping rows。"""

        return tuple(
            Phase2ReportMappingRow(
                key=str(key),
                value=_format_value(value),
            )
            for key, value in mapping.items()
        )

    def _report_payload(
        self,
        validation: Phase2ValidationResult | None,
    ) -> Mapping[str, Any]:
        """读取 evaluator 写入的 report 聚合 payload。"""

        if validation is None or validation.payloads is None:
            return {}
        return _as_mapping(validation.payloads.report_payload)

    def _payload_mapping(
        self,
        validation: Phase2ValidationResult | None,
        attr_name: str,
    ) -> Mapping[str, Any]:
        """读取 validation payloads 中的单个 payload。"""

        if validation is None or validation.payloads is None:
            return {}
        return _as_mapping(getattr(validation.payloads, attr_name, None))

    def _layer_metrics_mapping(
        self,
        validation: Phase2ValidationResult | None,
        layer_id: int,
        layer_name: str,
    ) -> Mapping[str, Any]:
        """读取 layer_computations 中的 raw metrics。"""

        if validation is None:
            return {}
        for computation in validation.layer_computations:
            if self._matches_layer(computation, layer_id, layer_name):
                return _as_mapping(computation.metrics)
        return {}

    @staticmethod
    def _matches_layer(
        computation: Phase2LayerComputation,
        layer_id: int,
        layer_name: str,
    ) -> bool:
        """判断 layer computation 是否匹配目标层。"""

        return computation.layer_id == layer_id or computation.layer_name == layer_name

    @staticmethod
    def _count_map(value: Any) -> dict[int, int]:
        """把 report count distribution 转换为 code_id -> count。"""

        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            return {}
        counts: dict[int, int] = {}
        for item in value:
            row = _as_mapping(item)
            if not row:
                continue
            try:
                code_id = int(row.get("code_id"))
                count = int(row.get("count", 0))
            except (TypeError, ValueError):
                continue
            counts[code_id] = count
        return counts

    @staticmethod
    def _grid_values(y_min: float, y_max: float) -> tuple[float, ...]:
        """返回图表 y 轴网格线数值。"""

        if not math.isfinite(y_min) or not math.isfinite(y_max):
            return ()
        if math.isclose(y_min, y_max):
            return (y_min,)
        step = (y_max - y_min) / 4.0
        return tuple(y_min + step * index for index in range(5))


__all__ = ["Phase2SelectorReportContextBuilder"]
