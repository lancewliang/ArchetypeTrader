"""Phase I codebook validation report 的强类型数据结构。

本文件只定义 report payload 和 HTML template context 需要的数据类型：

1. Report payload 类型负责承载机器可读报告数据，最终通过 ``to_dict()`` 保持现有
   JSON payload 结构；
2. HTML context/view 类型负责承载模板渲染前的展示模型，最终通过
   ``to_dict()`` 转成模板引擎可消费的普通 dict；
3. 本文件不读取文件、不渲染 HTML、不重新计算 validation metrics。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

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


def json_safe(value: Any) -> JsonValue:
    """把常见 Python/report 对象转换为 JSON-friendly 值。

    作用：
        统一处理 ``Path``、``tuple``、``Mapping`` 以及带 ``to_dict()`` 的强类型对象，
        避免 report payload 入口重复写弱类型转换逻辑。
    """

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
    """把 HTML context dataclass 递归转换为模板引擎可消费的普通 Python 值。

    作用：
        report 模板引擎只理解 ``Mapping``、序列和标量；HTML view 层内部使用强类型
        dataclass，最后由该函数在渲染边界统一降级为 dict/list。
    """

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
class Phase1CodebookReportMeta:
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
        title: str = DEFAULT_PHASE1_CODEBOOK_REPORT_TITLE,
        metadata: Mapping[str, object] | None = None,
        generated_at: str | None = None,
    ) -> "Phase1CodebookReportMeta":
        """为新生成的 report 创建元信息。"""

        return cls(
            title=title,
            generated_at=generated_at or datetime.now(UTC).isoformat(),
            metadata=_json_object(metadata or {}),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase1CodebookReportMeta":
        """从 payload 的 ``report`` 节点恢复强类型元信息。"""

        reserved_keys = {"title", "generated_at", "schema"}
        metadata = {
            str(key): value
            for key, value in payload.items()
            if str(key) not in reserved_keys
        }
        return cls(
            title=str(payload.get("title", DEFAULT_PHASE1_CODEBOOK_REPORT_TITLE)),
            generated_at=str(payload.get("generated_at", "-")),
            schema=str(payload.get("schema", PHASE1_CODEBOOK_REPORT_SCHEMA)),
            metadata=_json_object(metadata),
        )

    def to_dict(self) -> JsonObject:
        """转换为现有 payload 兼容的 ``report`` dict。"""

        payload: JsonObject = {
            "title": self.title,
            "generated_at": self.generated_at,
            "schema": self.schema,
        }
        payload.update(dict(self.metadata))
        return payload


@dataclass(frozen=True)
class Phase1CodebookReportDocument:
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
    config: Mapping[str, JsonValue] = field(default_factory=dict)

    # 产物路径索引，对应 payload["artifacts"]。
    artifacts: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """标准化 config/artifacts，使其可以直接落盘为 JSON。"""

        object.__setattr__(self, "config", _json_object(self.config))
        object.__setattr__(self, "artifacts", _json_object(self.artifacts))

    @classmethod
    def from_validation_result(
        cls,
        *,
        validation_result: Phase1ValidationResult,
        title: str = DEFAULT_PHASE1_CODEBOOK_REPORT_TITLE,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
        generated_at: str | None = None,
    ) -> "Phase1CodebookReportDocument":
        """从 validation result 构建完整 report document。"""

        return cls(
            report=Phase1CodebookReportMeta.generated(
                title=title,
                generated_at=generated_at,
                metadata=metadata,
            ),
            validation=validation_result,
            config=_json_object(config or {}),
            artifacts=_json_object(artifacts or {}),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase1CodebookReportDocument":
        """从 public payload dict 恢复 report document。"""

        report_payload = _require_mapping(payload.get("report"), "report")
        validation_payload = _require_mapping(
            payload.get("validation"),
            "validation",
        )
        return cls(
            report=Phase1CodebookReportMeta.from_dict(report_payload),
            validation=Phase1ValidationResult.from_dict(validation_payload),
            config=_json_object(_optional_mapping(payload.get("config"))),
            artifacts=_json_object(_optional_mapping(payload.get("artifacts"))),
        )

    def to_dict(self) -> JsonObject:
        """转换为现有对外兼容的 report payload dict。"""

        return {
            "report": self.report.to_dict(),
            "validation": _json_object(self.validation.to_dict()),
            "config": dict(self.config),
            "artifacts": dict(self.artifacts),
        }


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


@dataclass(frozen=True)
class Phase1ReportCodeDistributionRow:
    """codebook 使用分布行。

    作用：
        用于展示每个 code 的 occupancy、active/inactive 状态和条形宽度。
    """

    # code id。
    code_id: str

    # occupancy 原始展示值。
    occupancy: str

    # occupancy 百分比展示值。
    occupancy_percent: str

    # HTML/CSS 条形图宽度。
    bar_width: str

    # 该 code 是否 active。
    active: bool

    # active 状态 badge CSS class。
    badge_class: str

    # ACTIVE/INACTIVE 展示文本。
    status_label: str


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


@dataclass(frozen=True)
class Phase1ReportPairProfitabilityRow:
    """morphology-motif 盈利矩阵中的一行。

    作用：
        以 morphology 为行维度，持有该 morphology 下所有 motif 单元格。
    """

    # 行对应的 morphology。
    morphology: str

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

    # 按 morphology 分组的矩阵行。
    rows: tuple[Phase1ReportPairProfitabilityRow, ...]

    # 扁平化单元格列表，便于模板或后续图表复用。
    cells: tuple[Phase1ReportPairProfitabilityCell, ...]


@dataclass(frozen=True)
class Phase1CodebookReportHtmlContext:
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

    # codebook occupancy 分布。
    code_distribution: tuple[Phase1ReportCodeDistributionRow, ...]

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

    def to_dict(self) -> dict[str, Any]:
        """转换为模板引擎可消费的普通 dict。"""

        return cast(dict[str, Any], template_safe(self))

    def __getitem__(self, key: str) -> Any:
        """兼容历史测试和调用方对 HTML context 的 dict-style 读取。"""

        return self.to_dict()[key]


def ensure_phase1_codebook_report_document(
    payload: Phase1CodebookReportDocument | Mapping[str, Any],
) -> Phase1CodebookReportDocument:
    """确保输入是强类型 report document。

    作用：
        给后续 context builder 提供统一入口；新代码可直接传 document，历史调用方传
        dict 时在这里恢复成强类型对象。
    """

    if isinstance(payload, Phase1CodebookReportDocument):
        return payload
    return Phase1CodebookReportDocument.from_dict(payload)


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
    "Phase1ReportKpiRow",
    "Phase1ReportProfitSeriesRow",
    "Phase1ReportCodeDistributionRow",
    "Phase1ReportScoreBreakdownRow",
    "Phase1ReportSeries",
    "Phase1ReportSeriesPoint",
    "Phase1ReportPairProfitabilityCell",
    "Phase1ReportPairProfitabilityRow",
    "Phase1ReportPairProfitabilityMatrix",
    "ensure_phase1_codebook_report_document",
    "json_safe",
    "template_safe",
]
