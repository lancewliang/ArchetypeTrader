"""Phase I codebook validation report 的强类型数据结构。

本文件只定义 report payload 和 HTML template context 需要的数据类型：

1. Report payload 类型负责承载机器可读报告数据，最终通过 ``to_dict()`` 保持现有
   JSON payload 结构；
2. HTML context/view 类型负责承载模板渲染前的展示模型，最终通过
   ``to_dict()`` 转成模板引擎可消费的普通 dict；
3. 本文件不读取文件、不渲染 HTML、不重新计算 validation metrics。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

from pydantic import Field

from phase1.evaluators.phase1_validation_layers.analysis_code_distribution import CodeDistributionView
from src.utils import PydanticMappingModel

from ..metrics import (
    Phase1ValidationResult,
)


JsonScalar: TypeAlias = str | int | float | bool | None
"""JSON payload 中允许出现的标量类型。"""

JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
"""JSON payload 中允许出现的递归值类型。"""

JsonObject: TypeAlias = dict[str, JsonValue]
"""JSON object 类型别名，用于 report payload 的顶层和嵌套 mapping。"""

PHASE1_CODEBOOK_REPORT_SCHEMA = "phase1_codebook_validation_report.v1"
"""Phase I codebook validation report 的 payload schema 版本。"""

DEFAULT_PHASE1_CODEBOOK_REPORT_TITLE = "Phase I Codebook Validation Report"
"""未显式传入标题时使用的默认 report 标题。"""


class Phase1CodebookReportMeta(PydanticMappingModel):
    """报告元信息。

    含义：
        描述一份 report 本身，例如标题、生成时间和 schema 版本。

    作用：
        生成 payload 中的 ``report`` 节点；额外 metadata 会展开到该节点顶层，
        以兼容现有 payload 消费方。
    """

    # 报告标题，用于 payload 和 HTML 页面标题。
    title: str

    # 报告生成时间，建议使用 UTC ISO-8601 字符串。
    generated_at: str

    # payload schema 版本，用于后续兼容迁移。
    schema: str = PHASE1_CODEBOOK_REPORT_SCHEMA

    # 运行侧额外元数据，例如 pair、batch、git sha、run id。
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)

class Phase1CodebookReportDocument(PydanticMappingModel):
    """完整机器可读 report payload。

    含义：
        一份 codebook validation report 的强类型顶层容器。

    作用：
        替代在 ``phase1_codebook_report.py`` 中手写嵌套 dict 的方式；业务代码先填充
        该对象，再通过 ``to_dict()`` 生成 JSON/HTML 共用 payload。
    """

    # 报告元信息，对应 payload["report"]。
    report: Phase1CodebookReportMeta

    # 完整 validation result，对应 payload["validation"]。
    validation: Phase1ValidationResult

    # Phase I 配置快照，对应 payload["config"]。
    config: Mapping[str, JsonValue] = Field(default_factory=dict)

    # 产物路径索引，对应 payload["artifacts"]。
    artifacts: Mapping[str, JsonValue] = Field(default_factory=dict)

@dataclass(frozen=True)
class Phase1ReportHeaderItem:
    """HTML header 中单个元数据项。

    作用：
        用于 header 的横向 key-value 信息展示，例如 Pair、Batch、Checkpoint。
    """

    # 展示标签。
    label: str

    # 已格式化展示值。
    value: str


@dataclass(frozen=True)
class Phase1ReportHeader:
    """HTML 报告页眉视图模型。

    作用：
        汇总 pair、batch、checkpoint、K、N_val、horizon 等关键信息，用于页面顶部
        快速定位本报告对应的数据上下文。
    """

    # 交易对或标的。
    pair: str

    # 数据批次或训练批次。
    batch: str

    # checkpoint ID。
    checkpoint: str

    # codebook size。
    k: str

    # validation 样本数。
    n_val: str

    # 预测/持有 horizon。
    horizon: str

    # 报告生成时间。
    generated_at: str

    # 页面中循环展示的 header 元数据项。
    meta_items: tuple[Phase1ReportHeaderItem, ...]


@dataclass(frozen=True)
class Phase1ReportSummaryView:
    """HTML summary 区视图模型。

    作用：
        用于报告首页摘要卡片，展示 checkpoint 通过状态、score 和失败层概览。
    """

    # checkpoint ID。
    checkpoint_id: str

    # checkpoint 阶段。
    stage: str

    # epoch 文本。
    epoch: str

    # 已格式化 score。
    score: str

    # 失败 layer 列表文本。
    failed_layers: str

    # layer 总数文本。
    layer_count: str

    # code diagnostic 数量文本。
    code_diagnostic_count: str

    # risk finding 数量文本。
    risk_finding_count: str

    # PASS/FAIL badge CSS class。
    badge_class: str

    # PASS/FAIL 展示文本。
    status_label: str


@dataclass(frozen=True)
class Phase1ReportMetricView:
    """单个 metric 的 HTML 表格行视图。

    作用：
        统一展示 hard gate metric、drift diagnostic 等指标的值、阈值和状态。
    """

    # metric 稳定名称。
    name: str

    # 已格式化实际值。
    value: str

    # 人类可读阈值表达式。
    threshold: str

    # 机器阈值的展示文本。
    threshold_value: str

    # 指标方向，例如 higher_is_better/lower_is_better。
    direction: str

    # 到阈值的距离展示文本。
    distance_to_threshold: str

    # pass/warn/fail/skip badge CSS class。
    badge_class: str

    # 严重级别展示文本。
    severity_label: str

    # 指标说明或失败原因。
    message: str

    # 指标含义和判读说明，用于 hover tooltip。
    description: str = ""


@dataclass(frozen=True)
class Phase1ReportLayerView:
    """单个 validation layer 的 HTML 视图。

    作用：
        展示某一层 hard gate 的整体状态及其内部 metric 明细。
    """

    # layer 数字编号。
    layer_id: str

    # layer 稳定名称。
    name: str

    # layer 状态 badge CSS class。
    badge_class: str

    # PASS/FAIL 展示文本。
    status_label: str

    # metric 总数文本。
    metric_count: str

    # 失败 metric 数量文本。
    failed_count: str

    # layer 内所有 metric 行。
    metrics: tuple[Phase1ReportMetricView, ...]


@dataclass(frozen=True)
class Phase1ReportCodeDiagnosticView:
    """单个 code 的诊断视图。

    作用：
        展示 code-level 行为归因、分布占比和 decoded 盈利质量。
    """

    # codebook 中的 code id。
    code_id: str

    # 该 code 覆盖的样本数。
    support: str

    # 该 code occupancy。
    occupancy: str

    # dominant morphology 标签。
    dominant_morphology: str

    # dominant morphology 占比。
    dominant_morphology_ratio: str

    # morphology lift。
    morphology_lift: str

    # dominant motif 标签。
    dominant_motif: str

    # dominant motif 占比。
    dominant_motif_ratio: str

    # dominant morphology-motif pair。
    dominant_pair: str

    # dominant pair 占比。
    dominant_pair_ratio: str

    # decoded mean advantage。
    decoded_mean_advantage: str

    # decoded win rate。
    decoded_win_rate: str

    # decoded return 相对 DP return 的保留比例。
    retention_ratio: str

    # 手续费拖累。
    fee_drag: str

    # code 诊断状态。
    status: str

    # 状态 badge CSS class。
    badge_class: str


@dataclass(frozen=True)
class Phase1ReportRiskSummaryView:
    """风险摘要视图。

    作用：
        在报告首页用三段式说明主要风险、优先检查目标和建议动作。
    """

    # 是否存在 risk findings。
    has_findings: bool

    # 最高优先级风险等级。
    severity: str

    # 风险 badge CSS class。
    badge_class: str

    # finding 数量文本。
    finding_count: str

    # 主要风险描述。
    primary_risk: str

    # 建议优先检查的 code/pair/metric。
    inspection_target: str

    # 建议处置动作。
    recommendation: str


@dataclass(frozen=True)
class Phase1ReportRiskFindingView:
    """单条风险 finding 视图。

    作用：
        展示 checkpoint 级跨层风险定位结果，方便人工审计追踪证据。
    """

    # 风险等级。
    severity: str

    # 风险 badge CSS class。
    badge_class: str

    # 风险标题。
    title: str

    # 风险原因。
    reason: str

    # 关联 metric 列表文本。
    related_metrics: str

    # 关联 code 列表文本。
    related_codes: str

    # 关联 pair 列表文本。
    related_pairs: str

    # 建议动作。
    recommended_action: str


@dataclass(frozen=True)
class Phase1ReportMappingRow:
    """通用 key-value 表格行。

    作用：
        用于 config、artifacts、tie-breaker metrics 等简单 mapping 展示。
    """

    # 字段名。
    key: str

    # 已格式化字段值。
    value: str

    # 可选 tooltip 说明。
    description: str = ""


@dataclass(frozen=True)
class Phase1ReportKpiRow:
    """KPI 展示行。

    作用：
        用于 oracle profitability KPI 等带稳定 key、展示 label 和 value 的区块。
    """

    # 稳定机器 key。
    key: str

    # 人类可读 label。
    label: str

    # 已格式化值。
    value: str


@dataclass(frozen=True)
class Phase1ReportProfitSeriesRow:
    """per-code 盈利序列行。

    作用：
        用于展示每个 code 的 decoded profitability bar/list。
    """

    # code id。
    code_id: str

    # 展示 label，例如 code 3。
    label: str

    # 已格式化盈利值。
    value: str

    # 盈利状态 badge CSS class。
    badge_class: str

    # 条形图宽度。
    bar_width: str = "0%"


@dataclass(frozen=True)
class Phase1ReportLabelTip:
    """带 tooltip 说明的短标签。"""

    # 展示标签。
    label: str

    # tooltip 说明。
    description: str


@dataclass(frozen=True)
class Phase1ReportScoreBreakdownRow:
    """综合分拆解行。

    作用：
        展示 validation.score 的每个 component、权重和加权贡献。
    """

    # score component 名称。
    name: str

    # component 原始分值。
    value: str

    # component 权重。
    weight: str

    # component 加权贡献。
    weighted_value: str


@dataclass(frozen=True)
class Phase1ReportSeriesPoint:
    """图表序列中的单个点。

    作用：
        用于累计收益曲线等 step-value 图表数据。
    """

    # 横轴 step。
    step: str

    # 纵轴值。
    value: str


@dataclass(frozen=True)
class Phase1ReportSeries:
    """图表序列。

    作用：
        承载一条累计收益曲线，例如 DP、Decoded、Random label 或 Flat。
    """

    # 序列稳定 key。
    key: str

    # 序列展示 label。
    label: str

    # 序列点。
    points: tuple[Phase1ReportSeriesPoint, ...]


@dataclass(frozen=True)
class Phase1ReportChartGridLine:
    """SVG 图表中的横向网格线。"""

    # SVG y 坐标。
    y: str

    # 网格线数值标签。
    label: str


@dataclass(frozen=True)
class Phase1ReportChartSeries:
    """SVG 折线图中的一条序列。"""

    # 序列稳定 key。
    key: str

    # 序列展示 label。
    label: str

    # CSS 颜色值。
    color: str

    # SVG polyline points 字符串。
    points: str

    # 序列末值。
    end_value: str

    # hover title 文案。
    tooltip: str


@dataclass(frozen=True)
class Phase1ReportLineChart:
    """静态 SVG 折线图视图模型。"""

    # 图表小标题。
    title: str = ""

    # SVG viewBox 宽度。
    width: str = "820"

    # SVG viewBox 高度。
    height: str = "330"

    # 横向网格线。
    grid_lines: tuple[Phase1ReportChartGridLine, ...] = ()

    # 折线序列。
    series: tuple[Phase1ReportChartSeries, ...] = ()

    # y 轴最小值。
    y_min: str = "-"

    # y 轴最大值。
    y_max: str = "-"

    # x 轴说明。
    x_axis_label: str = "validation horizon order"

    # 同一区块内的补充子图。
    detail_charts: tuple["Phase1ReportLineChart", ...] = ()


@dataclass(frozen=True)
class Phase1ReportPairProfitabilityCell:
    """morphology-motif 盈利矩阵单元格。

    作用：
        展示某个 morphology 和 motif 组合下的 support、收益和风险状态。
    """

    # morphology 标签。
    morphology: str

    # motif 标签。
    motif: str

    # 样本数。
    support: str

    # mean decoded advantage。
    mean_decoded_advantage: str

    # decoded win rate。
    decoded_win_rate: str

    # return retention ratio。
    retention_ratio: str

    # fee drag。
    fee_drag: str

    # 盈利状态 badge CSS class。
    badge_class: str

    # 热力图中展示的单元格文本。
    display_value: str = "-"

    # 热力图单元格背景色。
    background_color: str = "#eef1f5"

    # 热力图单元格文字色。
    text_color: str = "#0f172a"

    # 单元格 hover title 文案。
    tooltip: str = ""


@dataclass(frozen=True)
class Phase1ReportPairProfitabilityRow:
    """morphology-motif 盈利矩阵中的一行。

    作用：
        以 morphology 为行维度，持有该 morphology 下所有 motif 单元格。
    """

    # 行对应的 morphology。
    morphology: str

    # morphology tooltip 说明。
    morphology_description: str

    # 该 morphology 下的所有 motif 单元格。
    cells: tuple[Phase1ReportPairProfitabilityCell, ...]


@dataclass(frozen=True)
class Phase1ReportPairProfitabilityMatrix:
    """morphology x motif 盈利矩阵视图。

    作用：
        为 HTML 模板提供矩阵表头、行结构和扁平 cell 列表。
    """

    # 矩阵行维度。
    morphologies: tuple[str, ...]

    # 矩阵列维度。
    motifs: tuple[str, ...]

    # 带 tooltip 的矩阵列头。
    motif_headers: tuple[Phase1ReportLabelTip, ...]

    # 按 morphology 分组的矩阵行。
    rows: tuple[Phase1ReportPairProfitabilityRow, ...]

    # 扁平化单元格列表，便于模板或后续图表复用。
    cells: tuple[Phase1ReportPairProfitabilityCell, ...]

    # 热力图 CSS grid-template-columns。
    grid_template_columns: str = ""

    # 热力图图例负向端点。
    legend_min: str = "-"

    # 热力图图例正向端点。
    legend_max: str = "-"


class Phase1CodebookReportHtmlContext(PydanticMappingModel):
    """完整 HTML 模板上下文。

    含义：
        HTML 模板渲染前的强类型展示模型。

    作用：
        替代 context builder 直接返回裸 dict 的弱类型方式；builder 先构造该对象，
        再通过 ``to_dict()`` 交给模板引擎。
    """

    # 浏览器页面标题。
    page_title: str

    # 报告主标题。
    header_title: str

    # 页眉元数据。
    header: Phase1ReportHeader

    # report schema/generated_at 等基础信息。
    report: Mapping[str, str]

    # 首页摘要。
    summary: Phase1ReportSummaryView

    # validation layer 明细。
    layers: tuple[Phase1ReportLayerView, ...]

    # code-level diagnostics。
    code_diagnostics: tuple[Phase1ReportCodeDiagnosticView, ...]

    # oracle profitability KPI。
    oracle_profitability_kpis: tuple[Phase1ReportKpiRow, ...]

    # 累计收益曲线序列。
    oracle_cumulative_return_series: tuple[Phase1ReportSeries, ...]

    # per-code 盈利序列。
    per_code_profit_series: tuple[Phase1ReportProfitSeriesRow, ...]

    # morphology x motif 盈利矩阵。
    pair_profitability_matrix: Phase1ReportPairProfitabilityMatrix

    # codebook 使用分布视图。
    code_distribution_view: CodeDistributionView | None = None

    # tie-breaker metric 行。
    tie_breaker_rows: tuple[Phase1ReportMappingRow, ...]

    # validation score component 拆解行。
    score_breakdown_rows: tuple[Phase1ReportScoreBreakdownRow, ...]

    # drift diagnostic metric 行。
    drift_diagnostics: tuple[Phase1ReportMetricView, ...]

    # 风险摘要。
    risk_summary: Phase1ReportRiskSummaryView

    # 风险 finding 明细。
    risk_findings: tuple[Phase1ReportRiskFindingView, ...]

    # config snapshot 行。
    config_rows: tuple[Phase1ReportMappingRow, ...]

    # artifact 路径行。
    artifact_rows: tuple[Phase1ReportMappingRow, ...]

    # 累计收益 SVG 折线图。
    oracle_cumulative_return_chart: Phase1ReportLineChart = Field(
        default_factory=Phase1ReportLineChart
    )

    def __getitem__(self, key: str) -> Any:
        """兼容历史测试和调用方对 HTML context 的 dict-style 读取。"""

        return self.model_dump(mode="python")[key]





__all__ = [
    "DEFAULT_PHASE1_CODEBOOK_REPORT_TITLE",
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "PHASE1_CODEBOOK_REPORT_SCHEMA",
    "Phase1CodebookReportDocument",
    "Phase1CodebookReportMeta",
    "Phase1CodebookReportHtmlContext",
    "Phase1ReportHeader",
    "Phase1ReportHeaderItem",
    "Phase1ReportSummaryView",
    "Phase1ReportLayerView",
    "Phase1ReportMetricView",
    "Phase1ReportCodeDiagnosticView",
    "Phase1ReportRiskSummaryView",
    "Phase1ReportRiskFindingView",
    "Phase1ReportMappingRow",
    "Phase1ReportLabelTip",
    "Phase1ReportKpiRow",
    "Phase1ReportProfitSeriesRow",
    "Phase1ReportScoreBreakdownRow",
    "Phase1ReportChartGridLine",
    "Phase1ReportChartSeries",
    "Phase1ReportLineChart",
    "Phase1ReportSeries",
    "Phase1ReportSeriesPoint",
    "Phase1ReportPairProfitabilityCell",
    "Phase1ReportPairProfitabilityRow",
    "Phase1ReportPairProfitabilityMatrix",
    "json_safe",
    "template_safe",
]
