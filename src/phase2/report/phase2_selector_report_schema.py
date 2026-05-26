"""Phase II selector report 的强类型数据结构。

本文件只定义 report payload 和 HTML template context 需要的数据类型：

1. Report payload 类型承载机器可读报告数据，并通过 ``to_dict()`` 输出稳定
   JSON 结构；
2. HTML context/view 类型承载模板渲染前的展示模型，并通过 ``to_dict()`` 转成
   模板引擎可消费的普通 dict；
3. 本文件不读取文件、不渲染 HTML、不重新计算 validation metrics，也不选择
   checkpoint。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

from ..checkpoint.phase2_checkpoint_selector import Phase2CheckpointSelectionResult
from ..metrics import (
    Phase2LayerResult,
    Phase2MetricResult,
    Phase2ValidationResult,
)


JsonScalar: TypeAlias = str | int | float | bool | None
"""JSON payload 中允许出现的标量类型。"""

JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
"""JSON payload 中允许出现的递归值类型。"""

JsonObject: TypeAlias = dict[str, JsonValue]
"""JSON object 类型别名，用于 report payload 的顶层和嵌套 mapping。"""

PHASE2_REPORT_SCHEMA = "phase2_selection_report.v1"
"""Phase II selector report 的 payload schema 版本。"""

DEFAULT_PHASE2_REPORT_TITLE = "Phase II Selector Validation Report"
"""未显式传入标题时使用的默认 report 标题。"""


def json_safe(value: Any) -> JsonValue:
    """把常见 Python/report 对象转换为 JSON-friendly 值。"""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_safe(item) for item in value]
    return cast(JsonValue, value)


def template_safe(value: Any) -> Any:
    """把 HTML context dataclass 递归转换为模板引擎可消费的普通 Python 值。"""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: template_safe(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): template_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [template_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class Phase2ReportMeta:
    """报告元信息。

    含义：
        描述一份 report 本身，例如标题、生成时间和 schema 版本。

    作用：
        生成 payload 中的 ``report`` 节点；额外 metadata 会展开到该节点顶层，
        以兼容后续 JSON/HTML 消费方。
    """

    # 报告标题，用于 payload 和 HTML 页面标题。
    title: str

    # 报告生成时间，建议使用 UTC ISO-8601 字符串。
    generated_at: str

    # payload schema 版本，用于后续兼容迁移。
    schema: str = PHASE2_REPORT_SCHEMA

    # 运行侧额外元数据，例如 pair、batch、git sha、run id。
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """标准化字段，确保 metadata 已经 JSON-safe。"""

        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "generated_at", str(self.generated_at))
        object.__setattr__(self, "schema", str(self.schema))
        object.__setattr__(self, "metadata", _json_object(self.metadata))

    @classmethod
    def generated(
        cls,
        *,
        title: str = DEFAULT_PHASE2_REPORT_TITLE,
        metadata: Mapping[str, object] | None = None,
        generated_at: str | None = None,
    ) -> "Phase2ReportMeta":
        """为新生成的 report 创建元信息。"""

        return cls(
            title=title,
            generated_at=generated_at or datetime.now(UTC).isoformat(),
            metadata=_json_object(metadata or {}),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2ReportMeta":
        """从 payload 的 ``report`` 节点恢复强类型元信息。"""

        reserved_keys = {"title", "generated_at", "schema"}
        metadata = {
            str(key): value
            for key, value in payload.items()
            if str(key) not in reserved_keys
        }
        return cls(
            title=str(payload.get("title", DEFAULT_PHASE2_REPORT_TITLE)),
            generated_at=str(payload.get("generated_at", "-")),
            schema=str(payload.get("schema", PHASE2_REPORT_SCHEMA)),
            metadata=_json_object(metadata),
        )

    def to_dict(self) -> JsonObject:
        """转换为 payload 兼容的 ``report`` dict。"""

        payload: JsonObject = {
            "title": self.title,
            "generated_at": self.generated_at,
            "schema": self.schema,
        }
        payload.update(dict(self.metadata))
        return payload


@dataclass(frozen=True)
class Phase2ReportDocument:
    """完整机器可读 report payload。

    含义：
        一份 Phase II selector report 的强类型顶层容器。

    作用：
        统一承载 checkpoint selection 摘要、validation result、config 和 artifact
        索引；业务代码先填充该对象，再通过 ``to_dict()`` 生成 JSON/HTML 共用
        payload。
    """

    # 报告元信息，对应 payload["report"]。
    report: Phase2ReportMeta

    # 选择摘要，对应 payload["selection"]。
    selection: Mapping[str, JsonValue] = field(default_factory=dict)

    # 首页摘要，对应 payload["summary"]。
    summary: Mapping[str, JsonValue] = field(default_factory=dict)

    # 完整 validation result，对应 payload["validation"]；无合格选择时可为空。
    validation: Phase2ValidationResult | None = None

    # Phase II 配置快照，对应 payload["config"]。
    config: Mapping[str, JsonValue] = field(default_factory=dict)

    # 产物路径索引，对应 payload["artifacts"]。
    artifacts: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """标准化 mapping 字段，使其可以直接落盘为 JSON。"""

        object.__setattr__(self, "selection", _json_object(self.selection))
        object.__setattr__(self, "summary", _json_object(self.summary))
        object.__setattr__(self, "config", _json_object(self.config))
        object.__setattr__(self, "artifacts", _json_object(self.artifacts))

    @classmethod
    def from_validation_result(
        cls,
        *,
        validation_result: Phase2ValidationResult,
        title: str = DEFAULT_PHASE2_REPORT_TITLE,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
        generated_at: str | None = None,
        selection: Mapping[str, object] | None = None,
    ) -> "Phase2ReportDocument":
        """从 validation result 构建完整 report document。"""

        return cls(
            report=Phase2ReportMeta.generated(
                title=title,
                generated_at=generated_at,
                metadata=metadata,
            ),
            selection=_json_object(selection or {}),
            summary=_build_validation_summary(validation_result),
            validation=validation_result,
            config=_json_object(config or {}),
            artifacts=_json_object(artifacts or {}),
        )

    @classmethod
    def from_selection_result(
        cls,
        *,
        selection_result: Phase2CheckpointSelectionResult,
        title: str = DEFAULT_PHASE2_REPORT_TITLE,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
        generated_at: str | None = None,
    ) -> "Phase2ReportDocument":
        """从 checkpoint selection result 构建 report document。"""

        validation = (
            selection_result.checkpoint.validation_result
            if selection_result.checkpoint is not None
            else None
        )
        selection_payload = _selection_result_to_dict(selection_result)
        summary = (
            _build_validation_summary(validation)
            if validation is not None
            else _build_blocked_summary(selection_payload)
        )
        return cls(
            report=Phase2ReportMeta.generated(
                title=title,
                generated_at=generated_at,
                metadata=metadata,
            ),
            selection=selection_payload,
            summary=summary,
            validation=validation,
            config=_json_object(config or {}),
            artifacts=_json_object(artifacts or {}),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2ReportDocument":
        """从 public payload dict 恢复 report document。"""

        report_node = _require_mapping(payload.get("report"), "report")
        validation_payload = payload.get("validation")
        return cls(
            report=Phase2ReportMeta.from_dict(report_node),
            selection=_json_object(_optional_mapping(payload.get("selection"))),
            summary=_json_object(_optional_mapping(payload.get("summary"))),
            validation=(
                Phase2ValidationResult.from_dict(validation_payload)
                if isinstance(validation_payload, Mapping)
                else None
            ),
            config=_json_object(_optional_mapping(payload.get("config"))),
            artifacts=_json_object(_optional_mapping(payload.get("artifacts"))),
        )

    def to_dict(self) -> JsonObject:
        """转换为对外兼容的 report payload dict。"""

        return {
            "report": self.report.to_dict(),
            "selection": dict(self.selection),
            "summary": dict(self.summary),
            "validation": (
                _json_object(self.validation.to_dict())
                if self.validation is not None
                else None
            ),
            "config": dict(self.config),
            "artifacts": dict(self.artifacts),
        }


@dataclass(frozen=True)
class Phase2ReportHeaderItem:
    """HTML header 中单个元数据项。"""

    label: str
    value: str


@dataclass(frozen=True)
class Phase2ReportHeader:
    """HTML 报告页眉视图模型。"""

    pair: str
    batch: str
    checkpoint: str
    k: str
    n_val: str
    horizon: str
    generated_at: str
    meta_items: tuple[Phase2ReportHeaderItem, ...]


@dataclass(frozen=True)
class Phase2ReportSummaryView:
    """HTML summary 区视图模型。"""

    checkpoint_id: str
    epoch: str
    score: str
    mean_return: str
    sharpe_like: str
    win_rate: str
    failed_layers: str
    layer_count: str
    badge_class: str
    status_label: str
    reason: str = ""


@dataclass(frozen=True)
class Phase2ReportMetricView:
    """单个 Phase II metric 的 HTML 表格行视图。"""

    name: str
    value: str
    threshold: str
    threshold_value: str
    direction: str
    distance_to_threshold: str
    badge_class: str
    severity_label: str
    message: str
    description: str = ""


@dataclass(frozen=True)
class Phase2ReportLayerView:
    """单个 Phase II validation layer 的 HTML 视图。"""

    layer_id: str
    name: str
    badge_class: str
    status_label: str
    metric_count: str
    failed_count: str
    metrics: tuple[Phase2ReportMetricView, ...]


@dataclass(frozen=True)
class Phase2ReportMappingRow:
    """通用 key-value 表格行。"""

    key: str
    value: str
    description: str = ""


@dataclass(frozen=True)
class Phase2ReportKpiRow:
    """KPI 展示行。"""

    key: str
    label: str
    value: str


@dataclass(frozen=True)
class Phase2ReportBaselineRow:
    """selector 与 baseline 对比行。"""

    baseline: str
    mean_return: str
    uplift: str
    beat_rate: str
    status: str
    badge_class: str


@dataclass(frozen=True)
class Phase2ReportPerCodeProfitabilityRow:
    """per-code selector/assigned-label 盈利对比行。"""

    code_id: str
    selector_support: str
    kl_support: str
    selector_mean_return: str
    kl_mean_return: str
    selector_win_rate: str
    kl_win_rate: str
    uplift_vs_kl: str
    badge_class: str
    status_label: str


@dataclass(frozen=True)
class Phase2ReportCodeUsageRow:
    """codebook 使用分布对比行。"""

    code_id: str
    selector_count: str
    selector_ratio: str
    kl_count: str
    kl_ratio: str
    ratio_delta: str
    bar_width: str
    badge_class: str
    status_label: str


@dataclass(frozen=True)
class Phase2ReportPairProfitabilityCell:
    """Dominant Pair 热力图中的单个 morphology/motif cell。"""

    morphology: str
    motif: str
    support: str
    selector_mean_return: str
    kl_mean_return: str
    random_mean_return: str
    mean_advantage_vs_kl: str
    mean_advantage_vs_random: str
    win_rate: str
    fee_drag_ratio: str
    dominant_selected_code: str
    dominant_selected_code_ratio: str
    background_color: str = "#ffffff"
    text_color: str = "#344054"
    display_value: str = "-"
    tooltip: str = ""


@dataclass(frozen=True)
class Phase2ReportPairProfitabilityRow:
    """Dominant Pair 热力图中的一个 morphology 行。"""

    morphology: str
    cells: tuple[Phase2ReportPairProfitabilityCell, ...]


@dataclass(frozen=True)
class Phase2ReportPairProfitabilityMatrix:
    """Dominant Pair 热力图展示模型。"""

    motifs: tuple[str, ...] = ()
    motif_headers: tuple[Phase2ReportMappingRow, ...] = ()
    rows: tuple[Phase2ReportPairProfitabilityRow, ...] = ()
    cells: tuple[Phase2ReportPairProfitabilityCell, ...] = ()
    grid_template_columns: str = "minmax(118px, 1.2fr)"
    legend_min: str = "-"
    legend_max: str = "-"
    legend_label: str = "mean advantage vs KL"


@dataclass(frozen=True)
class Phase2ReportCodeDiagnosticRow:
    """Phase II selector code 级诊断表行。"""

    code_id: str
    status: str
    badge_class: str
    selector_support: str
    selector_usage_ratio: str
    kl_support: str
    kl_usage_ratio: str
    usage_delta: str
    selector_mean_return: str
    kl_mean_return: str
    uplift_vs_kl: str
    selector_win_rate: str
    selector_fee_drag_ratio: str
    selector_turnover: str
    dominant_morphology: str
    dominant_morphology_ratio: str
    dominant_motif: str
    dominant_motif_ratio: str
    dominant_pair: str
    dominant_pair_ratio: str
    mean_q_margin: str
    low_confidence_ratio: str
    profitable_deviation_count: str
    unprofitable_deviation_count: str
    unprofitable_deviation_rate: str
    risk_reason: str


@dataclass(frozen=True)
class Phase2ReportSeriesPoint:
    """图表序列中的单个点。"""

    step: str
    value: str


@dataclass(frozen=True)
class Phase2ReportSeries:
    """图表序列。"""

    key: str
    label: str
    points: tuple[Phase2ReportSeriesPoint, ...]


@dataclass(frozen=True)
class Phase2ReportChartGridLine:
    """SVG 图表中的横向网格线。"""

    y: str
    label: str


@dataclass(frozen=True)
class Phase2ReportChartSeries:
    """SVG 折线图中的一条序列。"""

    key: str
    label: str
    color: str
    points: str
    end_value: str
    tooltip: str


@dataclass(frozen=True)
class Phase2ReportLineChart:
    """静态 SVG 折线图视图模型。"""

    title: str = ""
    width: str = "820"
    height: str = "330"
    grid_lines: tuple[Phase2ReportChartGridLine, ...] = ()
    series: tuple[Phase2ReportChartSeries, ...] = ()
    y_min: str = "-"
    y_max: str = "-"
    x_axis_label: str = "validation horizon order"


@dataclass(frozen=True)
class Phase2ReportHtmlContext:
    """完整 HTML 模板上下文。"""

    page_title: str
    header_title: str
    header: Phase2ReportHeader
    report: Mapping[str, str]
    summary: Phase2ReportSummaryView
    layers: tuple[Phase2ReportLayerView, ...]
    core_metric_rows: tuple[Phase2ReportMappingRow, ...]
    baseline_rows: tuple[Phase2ReportBaselineRow, ...]
    per_code_profitability_rows: tuple[Phase2ReportPerCodeProfitabilityRow, ...]
    code_usage_rows: tuple[Phase2ReportCodeUsageRow, ...]
    cumulative_return_series: tuple[Phase2ReportSeries, ...]
    config_rows: tuple[Phase2ReportMappingRow, ...]
    artifact_rows: tuple[Phase2ReportMappingRow, ...]
    cumulative_return_chart: Phase2ReportLineChart = field(
        default_factory=Phase2ReportLineChart
    )
    pair_profitability_matrix: Phase2ReportPairProfitabilityMatrix = field(
        default_factory=Phase2ReportPairProfitabilityMatrix
    )
    code_diagnostic_rows: tuple[Phase2ReportCodeDiagnosticRow, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """转换为模板引擎可消费的普通 dict。"""

        return cast(dict[str, Any], template_safe(self))

    def __getitem__(self, key: str) -> Any:
        """兼容测试和调用方对 HTML context 的 dict-style 读取。"""

        return self.to_dict()[key]


def ensure_phase2_report_document(
    payload: Phase2ReportDocument | Mapping[str, Any],
) -> Phase2ReportDocument:
    """确保输入是强类型 report document。"""

    if isinstance(payload, Phase2ReportDocument):
        return payload
    return Phase2ReportDocument.from_dict(payload)


def metric_result_to_view(metric: Phase2MetricResult) -> Phase2ReportMetricView:
    """把 Phase II metric result 转换为模板友好的行视图。"""

    return Phase2ReportMetricView(
        name=metric.name,
        value=str(metric.value) if metric.value is not None else "-",
        threshold=metric.threshold,
        threshold_value=(
            str(metric.threshold_value)
            if metric.threshold_value is not None
            else "-"
        ),
        direction=metric.direction or "-",
        distance_to_threshold=(
            str(metric.distance_to_threshold)
            if metric.distance_to_threshold is not None
            else "-"
        ),
        badge_class=metric.severity,
        severity_label=metric.severity.upper(),
        message=metric.message,
    )


def layer_result_to_view(layer: Phase2LayerResult) -> Phase2ReportLayerView:
    """把 Phase II layer result 转换为模板友好的 layer 视图。"""

    failed_count = sum(1 for metric in layer.metrics if not metric.passed)
    return Phase2ReportLayerView(
        layer_id=str(layer.layer_id),
        name=layer.name,
        badge_class="pass" if layer.passed else "fail",
        status_label="PASS" if layer.passed else "FAIL",
        metric_count=str(len(layer.metrics)),
        failed_count=str(failed_count),
        metrics=tuple(metric_result_to_view(metric) for metric in layer.metrics),
    )


def _build_validation_summary(
    validation_result: Phase2ValidationResult,
) -> JsonObject:
    """从 validation result 生成 report 首页摘要。"""

    metrics = validation_result.metrics
    failed_layers = tuple(
        layer.name for layer in validation_result.layers if not layer.passed
    )
    passed = len(failed_layers) == 0
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "mean_return": metrics.mean_return,
        "median_return": metrics.median_return,
        "sharpe_like": metrics.sharpe_like,
        "win_rate": metrics.win_rate,
        "mean_turnover": metrics.mean_turnover,
        "layer_count": len(validation_result.layers),
        "failed_layers": list(failed_layers),
    }


def _build_blocked_summary(selection: Mapping[str, JsonValue]) -> JsonObject:
    """为无合格 checkpoint 的 selection result 生成摘要。"""

    return {
        "passed": False,
        "status": "fail",
        "reason": "no eligible Phase II validation checkpoint",
        "selected_checkpoint_id": selection.get("selected_checkpoint_id"),
        "selected_epoch": selection.get("selected_epoch"),
        "selected_score": selection.get("selected_score"),
    }


def _selection_result_to_dict(
    selection_result: Phase2CheckpointSelectionResult,
) -> JsonObject:
    """把 checkpoint selection result 转换为 report payload。"""

    return {
        "has_selection": selection_result.has_selection,
        "selected_checkpoint_id": selection_result.selected_checkpoint_id,
        "selected_epoch": selection_result.selected_epoch,
        "selected_score": selection_result.selected_score,
    }


def _json_object(value: Mapping[str, Any]) -> JsonObject:
    """确保 mapping 已转换为 JSON object。"""

    safe_value = json_safe(value)
    if not isinstance(safe_value, dict):
        raise TypeError("expected a JSON object mapping")
    return safe_value


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """读取必填 mapping 字段。"""

    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a mapping")


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    """读取可选 mapping 字段，缺失时返回空 mapping。"""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise TypeError("optional report payload sections must be mappings")


__all__ = [
    "DEFAULT_PHASE2_REPORT_TITLE",
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "PHASE2_REPORT_SCHEMA",
    "Phase2ReportBaselineRow",
    "Phase2ReportChartGridLine",
    "Phase2ReportChartSeries",
    "Phase2ReportCodeUsageRow",
    "Phase2ReportDocument",
    "Phase2ReportHeader",
    "Phase2ReportHeaderItem",
    "Phase2ReportHtmlContext",
    "Phase2ReportKpiRow",
    "Phase2ReportLayerView",
    "Phase2ReportLineChart",
    "Phase2ReportMappingRow",
    "Phase2ReportMeta",
    "Phase2ReportMetricView",
    "Phase2ReportPairProfitabilityCell",
    "Phase2ReportPairProfitabilityMatrix",
    "Phase2ReportPairProfitabilityRow",
    "Phase2ReportPerCodeProfitabilityRow",
    "Phase2ReportSeries",
    "Phase2ReportSeriesPoint",
    "Phase2ReportSummaryView",
    "Phase2ReportCodeDiagnosticRow",
    "ensure_phase2_report_document",
    "json_safe",
    "layer_result_to_view",
    "metric_result_to_view",
    "template_safe",
]
