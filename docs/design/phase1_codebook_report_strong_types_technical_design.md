# Phase I Codebook Report Strong Types Technical Design

本文档定义 `phase1_codebook_report.py` 和
`phase1_codebook_report_context.py` 的强类型重构方案。目标是先把
report HTML 需要的数据填充到强类型对象中，再由强类型对象生成普通 `dict`，
最后交给 HTML 模板渲染，避免当前实现中大量 `dict[str, Any]`、隐式 key 和运行期
反序列化造成的弱类型边界。

## 1. 当前问题

当前数据流为：

```text
Phase1ValidationResult
  -> Phase1CodebookReport.build_payload() 拼 dict
  -> Phase1CodebookReportContextBuilder.build(payload: Mapping[str, Any])
  -> context dict
  -> render_template_file()
```

主要问题：

1. `build_payload()` 手工拼接嵌套 dict，字段名、字段类型和必填关系没有集中定义。
2. `phase1_codebook_report_context.py` 入口只接收 `Mapping[str, Any]`，内部再从 dict
   恢复 `Phase1LayerResult`、`Phase1MetricResult`、`Phase1RiskFinding` 等对象。
3. 模板上下文字段也是普通 dict，模板需要哪些字段、哪些字段可以为空，只能从实现和
   测试里反推。
4. 新增 report 区块时容易出现字段漏填、key 拼写错误、历史 payload 兼容逻辑散落在
   context builder 中。

## 2. 设计目标

本次重构目标：

1. 新增一个 schema 文件，集中定义 report payload 和 HTML context 的强类型数据结构。
2. `phase1_codebook_report.py` 先构建强类型 report payload 对象，再调用
   `to_dict()` 生成对外兼容的 payload dict。
3. `phase1_codebook_report_context.py` 先构建强类型 HTML context 对象，再调用
   `to_dict()` 生成模板渲染所需 dict。
4. 对外保留现有 `build_payload()`、`build_html()`、`render_html()` 行为，降低调用方
   迁移成本。
5. HTML 模板最终仍只消费普通 dict，不改 `_template.py` 的模板引擎。

非目标：

1. 不重新设计 Phase I validation 指标计算。
2. 不修改 hard gate、checkpoint selector 或 tie-breaker 策略。
3. 不引入 Pydantic 等运行时依赖；优先使用 `dataclass(frozen=True)` 和现有
   `to_dict()` / `from_dict()` 风格。
4. 不在本次重构中改变 HTML 视觉样式。

## 3. 新文件位置

新增文件：

```text
src/phase1/report/phase1_codebook_report_schema.py
```

该文件只定义 report 层数据类型和轻量序列化 helper，不负责读取文件、不渲染 HTML、
不重新计算 metrics。

建议导出：

```python
__all__ = [
    "JsonScalar",
    "JsonValue",
    "Phase1CodebookReportDocument",
    "Phase1CodebookReportMeta",
    "Phase1CodebookReportSummary",
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
]
```

## 4. 数据流设计

重构后的数据流：

```text
Phase1ValidationResult
  -> Phase1CodebookReportDocument.from_validation_result(...)
  -> document.to_dict()
  -> Phase1CodebookReportContextBuilder.build(document 或 document_dict)
  -> Phase1CodebookReportHtmlContext
  -> html_context.to_dict()
  -> render_template_file()
```

`build_payload()` 仍返回 dict，但内部不再直接拼 dict：

```python
def build_payload(...) -> dict[str, JsonValue]:
    document = Phase1CodebookReportDocument.from_validation_result(...)
    return document.to_dict()
```

`render_html()` 支持历史 dict payload，但会尽早恢复成强类型 document：

```python
def render_html(self, payload: Mapping[str, Any] | Phase1CodebookReportDocument) -> str:
    context = Phase1CodebookReportContextBuilder(title=self.title).build(payload)
    return render_template_file(_TEMPLATE_PATH, context.to_dict())
```

## 5. Report Payload 强类型

### 5.1 JSON 类型别名

```python
JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
```

`phase1_codebook_report.py` 中现有 `_json_safe()` 迁移到新 schema 文件，作为
`json_safe(value: Any) -> JsonValue` 的内部 helper。它继续支持：

1. 具备 `to_dict()` 的对象；
2. `Path`；
3. `Mapping`；
4. `tuple` / `list`。

### 5.2 `Phase1CodebookReportMeta`

```python
@dataclass(frozen=True)
class Phase1CodebookReportMeta:
    title: str
    generated_at: str
    schema: str = "phase1_codebook_validation_report.v1"
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> JsonObject: ...
```

说明：

1. 固定字段 `title`、`generated_at`、`schema` 保持当前 payload 结构。
2. 原 `metadata` 不再通过 `**dict(metadata)` 直接混入弱类型 dict，而是明确保存在
   `metadata` 字段中。
3. 为保持历史 payload 兼容，`to_dict()` 可以继续把 metadata 展开到 `report` dict
   顶层，但内部类型仍集中在该类中。

### 5.3 `Phase1CodebookReportSummary`

```python
@dataclass(frozen=True)
class Phase1CodebookReportSummary:
    checkpoint_id: str
    stage: str
    epoch: int
    passed: bool
    score: float | int | None
    failed_layers: tuple[str, ...]
    failed_layer_count: int
    layer_count: int
    code_diagnostic_count: int
    risk_finding_count: int

    @classmethod
    def from_validation_result(
        cls,
        validation_result: Phase1ValidationResult,
    ) -> "Phase1CodebookReportSummary": ...

    def to_dict(self) -> JsonObject: ...
```

`score` 读取规则：

1. 如果 `validation_result.score` 是 `Phase1ValidationScore`，使用
   `total_score`。
2. 如果是 `float` / `int`，直接转成数值。
3. 如果是 `None`，保留 `None`。

### 5.4 `Phase1CodebookReportDocument`

```python
@dataclass(frozen=True)
class Phase1CodebookReportDocument:
    report: Phase1CodebookReportMeta
    summary: Phase1CodebookReportSummary
    validation: Phase1ValidationResult
    config: Mapping[str, JsonValue] = field(default_factory=dict)
    artifacts: Mapping[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def from_validation_result(
        cls,
        *,
        validation_result: Phase1ValidationResult,
        title: str,
        generated_at: str,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "Phase1CodebookReportDocument": ...

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase1CodebookReportDocument": ...

    def to_dict(self) -> JsonObject: ...
```

`to_dict()` 输出结构保持当前兼容格式：

```python
{
    "report": report.to_dict(),
    "summary": summary.to_dict(),
    "validation": validation.to_dict(),
    "config": dict(config),
    "artifacts": dict(artifacts),
}
```

`from_dict()` 用于兼容现有测试和历史 payload。它负责把
`payload["validation"]` 恢复为 `Phase1ValidationResult`。如果历史测试 payload
缺少 `metrics` 或 `tie_breaker_metrics` 等新必填字段，可以提供一个私有兼容路径：

```python
_validation_result_from_report_payload(payload: Mapping[str, Any])
```

该兼容路径只应该留在 schema 文件中，避免 context builder 继续散落反序列化逻辑。

## 6. HTML Context 强类型

HTML context 强类型的字段名应和模板变量保持一致。这样 `to_dict()` 的输出可以直接
传给 `render_template_file()`，模板无需改语法。

### 6.1 基础视图类型

```python
@dataclass(frozen=True)
class Phase1ReportHeaderItem:
    label: str
    value: str

@dataclass(frozen=True)
class Phase1ReportHeader:
    pair: str
    batch: str
    checkpoint: str
    k: str
    n_val: str
    horizon: str
    generated_at: str
    meta_items: tuple[Phase1ReportHeaderItem, ...]

@dataclass(frozen=True)
class Phase1ReportSummaryView:
    checkpoint_id: str
    stage: str
    epoch: str
    score: str
    failed_layers: str
    layer_count: str
    code_diagnostic_count: str
    risk_finding_count: str
    badge_class: str
    status_label: str
```

Risk summary 和 finding 也属于首页/审计区的固定模板契约：

```python
@dataclass(frozen=True)
class Phase1ReportRiskSummaryView:
    has_findings: bool
    severity: str
    badge_class: str
    finding_count: str
    primary_risk: str
    inspection_target: str
    recommendation: str

@dataclass(frozen=True)
class Phase1ReportRiskFindingView:
    severity: str
    badge_class: str
    title: str
    reason: str
    related_metrics: str
    related_codes: str
    related_pairs: str
    recommended_action: str
```

### 6.2 指标和 layer 视图

```python
@dataclass(frozen=True)
class Phase1ReportMetricView:
    name: str
    value: str
    threshold: str
    threshold_value: str
    direction: str
    distance_to_threshold: str
    badge_class: str
    severity_label: str
    message: str

@dataclass(frozen=True)
class Phase1ReportLayerView:
    layer_id: str
    name: str
    badge_class: str
    status_label: str
    metric_count: str
    failed_count: str
    metrics: tuple[Phase1ReportMetricView, ...]
```

### 6.3 Code diagnostics 视图

```python
@dataclass(frozen=True)
class Phase1ReportCodeDiagnosticView:
    code_id: str
    support: str
    occupancy: str
    dominant_morphology: str
    dominant_morphology_ratio: str
    morphology_lift: str
    dominant_motif: str
    dominant_motif_ratio: str
    dominant_pair: str
    dominant_pair_ratio: str
    decoded_mean_advantage: str
    decoded_win_rate: str
    retention_ratio: str
    fee_drag: str
    status: str
    badge_class: str
```

### 6.4 通用 row、series、matrix 视图

```python
@dataclass(frozen=True)
class Phase1ReportMappingRow:
    key: str
    value: str

@dataclass(frozen=True)
class Phase1ReportKpiRow:
    key: str
    label: str
    value: str

@dataclass(frozen=True)
class Phase1ReportProfitSeriesRow:
    code_id: str
    label: str
    value: str
    badge_class: str

@dataclass(frozen=True)
class Phase1ReportCodeDistributionRow:
    code_id: str
    occupancy: str
    occupancy_percent: str
    bar_width: str
    active: bool
    badge_class: str
    status_label: str

@dataclass(frozen=True)
class Phase1ReportScoreBreakdownRow:
    name: str
    value: str
    weight: str
    weighted_value: str

@dataclass(frozen=True)
class Phase1ReportSeriesPoint:
    step: str
    value: str

@dataclass(frozen=True)
class Phase1ReportSeries:
    key: str
    label: str
    points: tuple[Phase1ReportSeriesPoint, ...]
```

Pair profitability matrix 建议保留一个专门类型，避免 template 侧依赖嵌套 dict：

```python
@dataclass(frozen=True)
class Phase1ReportPairProfitabilityCell:
    morphology: str
    motif: str
    support: str
    mean_decoded_advantage: str
    decoded_win_rate: str
    retention_ratio: str
    fee_drag: str
    badge_class: str

@dataclass(frozen=True)
class Phase1ReportPairProfitabilityRow:
    morphology: str
    cells: tuple[Phase1ReportPairProfitabilityCell, ...]

@dataclass(frozen=True)
class Phase1ReportPairProfitabilityMatrix:
    morphologies: tuple[str, ...]
    motifs: tuple[str, ...]
    rows: tuple[Phase1ReportPairProfitabilityRow, ...]
    cells: tuple[Phase1ReportPairProfitabilityCell, ...]
```

### 6.5 `Phase1CodebookReportHtmlContext`

```python
@dataclass(frozen=True)
class Phase1CodebookReportHtmlContext:
    page_title: str
    header_title: str
    header: Phase1ReportHeader
    report: Mapping[str, str]
    summary: Phase1ReportSummaryView
    layers: tuple[Phase1ReportLayerView, ...]
    code_diagnostics: tuple[Phase1ReportCodeDiagnosticView, ...]
    oracle_profitability_kpis: tuple[Phase1ReportKpiRow, ...]
    oracle_cumulative_return_series: tuple[Phase1ReportSeries, ...]
    per_code_profit_series: tuple[Phase1ReportProfitSeriesRow, ...]
    pair_profitability_matrix: Phase1ReportPairProfitabilityMatrix
    code_distribution: tuple[Phase1ReportCodeDistributionRow, ...]
    tie_breaker_rows: tuple[Phase1ReportMappingRow, ...]
    score_breakdown_rows: tuple[Phase1ReportScoreBreakdownRow, ...]
    drift_diagnostics: tuple[Phase1ReportMetricView, ...]
    risk_summary: Phase1ReportRiskSummaryView
    risk_findings: tuple[Phase1ReportRiskFindingView, ...]
    config_rows: tuple[Phase1ReportMappingRow, ...]
    artifact_rows: tuple[Phase1ReportMappingRow, ...]

    def to_dict(self) -> dict[str, Any]: ...
```

`to_dict()` 可以使用一个统一 helper 递归展开 dataclass、tuple 和 mapping：

```python
def dataclass_to_template_dict(value: Any) -> Any:
    ...
```

注意：这里的 `dict[str, Any]` 是模板引擎入口要求，不再是业务层随意构造的弱类型
数据。弱类型边界被限制在最后一步。

## 7. `phase1_codebook_report.py` 修改方案

职责调整：

1. 保留 `Phase1CodebookReport` 作为外部入口。
2. 删除本文件内 `_json_safe()`，改从
   `phase1_codebook_report_schema.py` 导入 report document。
3. 新增私有 `_build_document()` 或公开 `build_document()`，用于测试强类型对象。
4. `build_payload()` 只调用 `document.to_dict()`。
5. `render_html()` 只接收 document 或 dict，交给 context builder，最后把
   `html_context.to_dict()` 传给模板。

建议接口：

```python
def build_document(
    self,
    *,
    validation_result: Phase1ValidationResult,
    config: Mapping[str, object] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Phase1CodebookReportDocument:
    ...
```

兼容接口：

```python
def build_payload(...) -> dict[str, JsonValue]:
    return self.build_document(...).to_dict()
```

## 8. `phase1_codebook_report_context.py` 修改方案

职责调整：

1. `build()` 返回 `Phase1CodebookReportHtmlContext`，不再返回普通 dict。
2. `build()` 入口支持 `Phase1CodebookReportDocument | Mapping[str, Any]`，历史
   dict 在入口第一步统一转换成 document。
3. 内部 helper 不再返回 `JsonObject`，而是返回对应强类型 view。
4. `_vq_internal_payload()`、`_oracle_profitability_payload()`、
   `_validation_score()` 这类反序列化 helper 尽量迁移到 schema/document 层；context
   builder 只消费强类型 `Phase1ValidationResult`。
5. `_format_value()`、`_badge_class()` 仍可留在 context builder，因为它们属于展示层。

示例：

```python
def build(
    self,
    payload: Phase1CodebookReportDocument | Mapping[str, Any],
) -> Phase1CodebookReportHtmlContext:
    document = ensure_report_document(payload)
    validation_result = document.validation
    ...
    return Phase1CodebookReportHtmlContext(...)
```

## 9. 兼容性策略

保留兼容行为：

1. `Phase1CodebookReport.build_payload()` 返回结构与当前一致。
2. `Phase1CodebookReport.render_html(payload_dict)` 仍可渲染旧测试 payload。
3. `metadata` 在 `report` dict 中继续以顶层 key 展开，避免已有消费者读取失败。
4. `score` 同时兼容 `Phase1ValidationScore`、数字和 `None`。
5. `config`、`artifacts` 仍允许任意 JSON-safe 值，但在 document 内先经过
   `json_safe()` 标准化。

可以收紧的行为：

1. 新代码路径优先使用 `Phase1CodebookReportDocument`，不鼓励业务代码直接手写
   payload dict。
2. 新增模板字段必须先新增 dataclass 字段，再在 builder 中填充，最后由 `to_dict()`
   暴露给模板。
3. 测试应覆盖强类型对象，而不是只断言最终 HTML 字符串。

## 10. 测试计划

需要更新或新增 `tests/test_phase1_codebook_report.py` 中的测试：

1. `test_build_document_returns_strong_report_document`
   验证 `build_document()` 返回 `Phase1CodebookReportDocument`，summary 数量和 score
   正确。
2. `test_build_payload_preserves_existing_dict_contract`
   验证 `build_payload()` 输出 key 与当前兼容。
3. `test_context_builder_returns_strong_html_context`
   验证 `Phase1CodebookReportContextBuilder.build()` 返回
   `Phase1CodebookReportHtmlContext`，不是裸 dict。
4. `test_html_context_to_dict_matches_template_contract`
   验证 `to_dict()` 包含模板需要的关键字段：`page_title`、`header`、`summary`、
   `layers`、`code_diagnostics` 等。
5. `test_render_html_accepts_legacy_payload_dict`
   保留历史 payload dict 渲染能力，防止外部调用方立刻破坏。
6. `test_json_safe_normalizes_path_tuple_and_dataclass`
   覆盖 Path、tuple、Mapping、`to_dict()` 对象。

运行命令使用项目约定的 conda 环境：

```bash
conda run -n ArachetypeTrade pytest tests/test_phase1_codebook_report.py
```

## 11. 实施步骤

建议分三步落地，降低回归风险：

1. 新增 `phase1_codebook_report_schema.py`，只引入强类型和 `to_dict()`，暂不改 HTML
   行为。
2. 改 `phase1_codebook_report.py`，让 `build_payload()` 通过
   `Phase1CodebookReportDocument` 生成 dict，并补充 document 级测试。
3. 改 `phase1_codebook_report_context.py`，让 context builder 返回
   `Phase1CodebookReportHtmlContext`，最后在 `render_html()` 调用 `to_dict()`。

每一步都保持测试可运行，避免一次性同时修改 payload、context 和模板导致问题难以
定位。

## 12. 验收标准

重构完成后应满足：

1. `phase1_codebook_report.py` 不再直接手写完整嵌套 payload dict。
2. `phase1_codebook_report_context.py` 的 public `build()` 不再返回裸 dict。
3. 新增的 report HTML 字段必须先存在于强类型 schema 中。
4. `build_payload()` 输出结构与历史调用兼容。
5. `render_html()` 仍可接受当前测试中的 legacy payload dict。
6. `tests/test_phase1_codebook_report.py` 全部通过。
