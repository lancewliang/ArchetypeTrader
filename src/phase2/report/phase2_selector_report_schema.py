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

from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias
from pydantic import Field
from src.utils import PydanticMappingModel
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

class Phase2ReportMeta(PydanticMappingModel):
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
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)
 
class Phase2ReportDocument(PydanticMappingModel):
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
    selection: Mapping[str, object] = Field(default_factory=dict)

    # 首页摘要，对应 payload["summary"]。
    summary: Mapping[str, object] = Field(default_factory=dict)

    # 完整 validation result，对应 payload["validation"]；无合格选择时可为空。
    validation: Phase2ValidationResult | None = None

    # Phase II 配置快照，对应 payload["config"]。
    config: Mapping[str, object] = Field(default_factory=dict)

    # 产物路径索引，对应 payload["artifacts"]。
    artifacts: Mapping[str, object] = Field(default_factory=dict)
 
 

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


class Phase2ReportHtmlContext(PydanticMappingModel):
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
    cumulative_return_chart: Phase2ReportLineChart = Field(
        default_factory=Phase2ReportLineChart
    )
    pair_profitability_matrix: Phase2ReportPairProfitabilityMatrix = Field(
        default_factory=Phase2ReportPairProfitabilityMatrix
    )
    code_diagnostic_rows: tuple[Phase2ReportCodeDiagnosticRow, ...] = ()


   

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
 
]
