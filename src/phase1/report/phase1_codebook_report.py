"""Phase I codebook validation report JSON/HTML 呈现入口。

文件功能说明:
    本文件实现 ``Phase1CodebookReport``，负责把已经计算好的
    ``Phase1ValidationResult`` 转换为可落盘的 JSON payload 和静态 HTML report。

设计边界:
    - 只消费 validation result、config 和 artifacts；
    - 不重新计算 raw metrics；
    - 不重新执行 hard gate rules；
    - 不参与 checkpoint selection；
    - HTML 是静态审计视图，不依赖外部 JS/CSS 文件。

使用场景:
    ``Phase1Report.write_report()`` 或训练主流程在拿到
    ``Phase1ValidationResult`` 后，调用本类写出 JSON/HTML，供人工审计、
    实验复盘和后续 Phase II/III 读取摘要。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..metrics import (
    Phase1CodeDiagnostic,
    Phase1LayerResult,
    Phase1MetricResult,
    Phase1ValidationResult,
)


JsonObject = dict[str, Any]


def _json_safe(value: Any) -> Any:
    """把常见非 JSON 原生对象转换为可序列化值。

    输入参数:
        value: 任意 report payload 值，可包含 Path、tuple、list、mapping 或具备
            ``to_dict()`` 方法的对象。

    输出:
        JSON 可序列化的 Python 值。

    使用场景:
        写 JSON 前统一处理 artifacts、config 和 validation result 中的路径或对象。
    """

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value


def _format_value(value: Any) -> str:
    """格式化 HTML 表格中的指标值。

    输入参数:
        value: 原始指标值。

    输出:
        面向 HTML 展示的字符串。

    使用场景:
        metric、tie-breaker、config 和 artifact 表格渲染。
    """

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
    return str(value)


def _badge_class(severity: str | bool) -> str:
    """将状态映射为 HTML badge class。

    输入参数:
        severity: metric severity 字符串，或 layer/checkpoint passed 布尔值。

    输出:
        ``pass``、``fail``、``warn`` 或 ``skip``。

    使用场景:
        report 中统一渲染 checkpoint、layer 和 metric 状态。
    """

    if isinstance(severity, bool):
        return "pass" if severity else "fail"
    if severity in {"pass", "fail", "warn", "skip"}:
        return severity
    return "warn"


@dataclass(frozen=True)
class Phase1CodebookReport:
    """Phase I codebook validation 报告渲染器。

    功能说明:
        提供 ``build_payload()``、``write_json()`` 和 ``write_html()`` 三个入口。
        payload 保留完整机器可读结构，HTML 提供人工审计视图。

    使用场景:
        full validation 完成后，把 ``Phase1ValidationResult`` 输出为
        ``phase1_codebook_validation.json`` 和 ``phase1_codebook_validation.html``。
    """

    title: str = "Phase I Codebook Validation Report"

    def build_payload(
        self,
        *,
        validation_result: Phase1ValidationResult,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> JsonObject:
        """构建 JSON/HTML 共用 report payload。

        输入参数:
            validation_result: 五层 validation 完整结果。
            config: Phase I 配置快照，可为空。
            artifacts: 产物路径索引，可为空。
            metadata: 额外 report 元数据，例如 run id、git sha 或数据批次。

        输出:
            普通 dict，包含 report 元信息、summary、validation、config 和 artifacts。

        使用场景:
            ``write_json()`` 和 ``write_html()`` 的共同前置步骤；外部系统也可直接
            调用本方法拿到机器可读 payload。
        """

        generated_at = datetime.now(UTC).isoformat()
        validation_payload = validation_result.to_dict()
        summary = {
            "checkpoint_id": validation_result.checkpoint_id,
            "stage": validation_result.stage,
            "epoch": validation_result.epoch,
            "passed": validation_result.passed,
            "score": validation_result.score,
            "failed_layers": list(validation_result.failed_layers),
            "failed_layer_count": len(validation_result.failed_layers),
            "layer_count": len(validation_result.layers),
            "code_diagnostic_count": len(validation_result.code_diagnostics),
        }
        payload = {
            "report": {
                "title": self.title,
                "generated_at": generated_at,
                "schema": "phase1_codebook_validation_report.v1",
                **dict(metadata or {}),
            },
            "summary": summary,
            "validation": validation_payload,
            "config": dict(config or {}),
            "artifacts": dict(artifacts or {}),
        }
        return _json_safe(payload)

    def write_json(
        self,
        *,
        validation_result: Phase1ValidationResult,
        output_path: str | Path,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """写出 codebook validation JSON report。

        输入参数:
            validation_result: 五层 validation 完整结果。
            output_path: JSON 输出路径。
            config: Phase I 配置快照，可为空。
            artifacts: 产物路径索引，可为空。
            metadata: 额外 report 元数据。

        输出:
            实际写出的 ``Path``。

        使用场景:
            checkpoint validation 完成后保存机器可读 report，供后续脚本、selector
            或实验审计读取。
        """

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_payload(
            validation_result=validation_result,
            config=config,
            artifacts=artifacts,
            metadata=metadata,
        )
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def write_html(
        self,
        *,
        validation_result: Phase1ValidationResult,
        output_path: str | Path,
        config: Mapping[str, object] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """写出 codebook validation HTML report。

        输入参数:
            validation_result: 五层 validation 完整结果。
            output_path: HTML 输出路径。
            config: Phase I 配置快照，可为空。
            artifacts: 产物路径索引，可为空。
            metadata: 额外 report 元数据。

        输出:
            实际写出的 ``Path``。

        使用场景:
            checkpoint validation 完成后保存人工审计报告，展示五层 hard gate、
            raw metric、code diagnostics 和 tie-breaker。
        """

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build_payload(
            validation_result=validation_result,
            config=config,
            artifacts=artifacts,
            metadata=metadata,
        )
        path.write_text(self.render_html(payload), encoding="utf-8")
        return path

    def render_html(self, payload: Mapping[str, Any]) -> str:
        """将 report payload 渲染为静态 HTML 字符串。

        输入参数:
            payload: ``build_payload()`` 返回的 report payload。

        输出:
            完整 HTML 文本。

        使用场景:
            ``write_html()`` 写文件前调用；测试中也可直接校验 HTML 片段。
        """

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

        return "\n".join(
            [
                "<!doctype html>",
                '<html lang="zh-CN">',
                "<head>",
                '  <meta charset="utf-8">',
                '  <meta name="viewport" content="width=device-width, initial-scale=1">',
                f"  <title>{escape(str(report.get('title', self.title)))}</title>",
                f"  {self._style_block()}",
                "</head>",
                "<body>",
                self._render_header(summary, report),
                "  <main class=\"wrap\">",
                self._render_summary(summary),
                self._render_layers(layers),
                self._render_metric_tables(layers),
                self._render_code_diagnostics(code_diagnostics),
                self._render_tie_breaker(validation.get("tie_breaker_metrics", {})),
                self._render_drift_diagnostics(drift_diagnostics),
                self._render_mapping_table("Config Snapshot", config),
                self._render_mapping_table("Artifacts", artifacts),
                "  </main>",
                "</body>",
                "</html>",
            ]
        )

    def _render_header(
        self,
        summary: Mapping[str, Any],
        report: Mapping[str, Any],
    ) -> str:
        """渲染 HTML 顶部标题区。

        输入参数:
            summary: report summary payload。
            report: report metadata payload。

        输出:
            header HTML 字符串。

        使用场景:
            ``render_html()`` 内部组装报告页面。
        """

        return f"""
  <header>
    <div class="wrap hero">
      <div>
        <h1>{escape(self.title)}</h1>
        <p class="subtitle">Checkpoint {escape(str(summary.get("checkpoint_id", "-")))} · Stage {escape(str(summary.get("stage", "-")))} · Epoch {escape(str(summary.get("epoch", "-")))}</p>
      </div>
      <div class="meta">
        <span>Generated {escape(str(report.get("generated_at", "-")))}</span>
        <span>Schema {escape(str(report.get("schema", "-")))}</span>
      </div>
    </div>
  </header>"""

    def _render_summary(self, summary: Mapping[str, Any]) -> str:
        """渲染 checkpoint validation 摘要。

        输入参数:
            summary: report summary payload。

        输出:
            summary section HTML。

        使用场景:
            HTML 首屏展示 checkpoint 是否通过、score 和失败层。
        """

        passed = bool(summary.get("passed", False))
        badge = _badge_class(passed)
        failed_layers = summary.get("failed_layers", [])
        failed_text = ", ".join(str(layer) for layer in failed_layers) or "-"
        score = _format_value(summary.get("score"))
        return f"""
    <section class="grid summary">
      <article class="panel score-card">
        <div class="label">Validation Score</div>
        <div class="score">{escape(score)}</div>
        <span class="badge {badge}">{'PASS' if passed else 'FAIL'}</span>
      </article>
      <article class="panel"><div class="label">Checkpoint</div><div class="value">{escape(str(summary.get("checkpoint_id", "-")))}</div></article>
      <article class="panel"><div class="label">Stage</div><div class="value">{escape(str(summary.get("stage", "-")))}</div></article>
      <article class="panel"><div class="label">Epoch</div><div class="value">{escape(str(summary.get("epoch", "-")))}</div></article>
      <article class="panel"><div class="label">Failed Layers</div><div class="value">{escape(failed_text)}</div></article>
      <article class="panel"><div class="label">Code Diagnostics</div><div class="value">{escape(str(summary.get("code_diagnostic_count", 0)))}</div></article>
    </section>"""

    def _render_layers(self, layers: tuple[Phase1LayerResult, ...]) -> str:
        """渲染五层 hard gate 总览。

        输入参数:
            layers: 五层 layer result。

        输出:
            layer cards HTML。

        使用场景:
            HTML 摘要区之后展示每层是否通过、指标数量和失败数量。
        """

        cards = []
        for layer in layers:
            failed = sum(1 for metric in layer.metrics if not metric.passed)
            badge = _badge_class(layer.passed)
            cards.append(
                f"""
      <article class="panel layer-card">
        <div class="layer-head">
          <div><div class="label">Layer {layer.layer_id}</div><h3>{escape(layer.name)}</h3></div>
          <span class="badge {badge}">{'PASS' if layer.passed else 'FAIL'}</span>
        </div>
        <div class="layer-stats">
          <div><strong>{len(layer.metrics)}</strong><span>metrics</span></div>
          <div><strong>{failed}</strong><span>failed</span></div>
        </div>
      </article>"""
            )
        return f"""
    <section>
      <h2>Layer Gates</h2>
      <div class="grid layers">{''.join(cards)}</div>
    </section>"""

    def _render_metric_tables(self, layers: tuple[Phase1LayerResult, ...]) -> str:
        """渲染每层 metric 明细表。

        输入参数:
            layers: 五层 layer result。

        输出:
            metric tables HTML。

        使用场景:
            人工审计具体哪个 raw metric 触发 fail/skip/warn。
        """

        sections = []
        for layer in layers:
            rows = "\n".join(self._render_metric_row(metric) for metric in layer.metrics)
            sections.append(
                f"""
      <article class="panel">
        <div class="panel-head"><h3>Layer {layer.layer_id}: {escape(layer.name)}</h3></div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th><th>Message</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
      </article>"""
            )
        return f"""
    <section>
      <h2>Metric Details</h2>
      <div class="stack">{''.join(sections)}</div>
    </section>"""

    def _render_metric_row(self, metric: Phase1MetricResult) -> str:
        """渲染单条 metric result 表格行。

        输入参数:
            metric: 单个 hard gate metric result。

        输出:
            ``<tr>`` HTML 字符串。

        使用场景:
            ``_render_metric_tables()`` 和 drift diagnostics 表格复用。
        """

        badge = _badge_class(metric.severity)
        return f"""
              <tr>
                <td>{escape(metric.name)}</td>
                <td>{escape(_format_value(metric.value))}</td>
                <td>{escape(metric.threshold)}</td>
                <td><span class="badge {badge}">{escape(metric.severity.upper())}</span></td>
                <td>{escape(metric.message)}</td>
              </tr>"""

    def _render_code_diagnostics(
        self,
        diagnostics: tuple[Phase1CodeDiagnostic, ...],
    ) -> str:
        """渲染 code-level diagnostics 表。

        输入参数:
            diagnostics: per-code diagnostics。

        输出:
            code diagnostics section HTML；没有数据时返回空字符串。

        使用场景:
            展示每个 active code 的 support、morphology、motif 和 profitability 摘要。
        """

        if not diagnostics:
            return ""
        rows = []
        for item in diagnostics:
            rows.append(
                f"""
              <tr>
                <td>{item.code_id}</td>
                <td>{item.support}</td>
                <td>{escape(_format_value(item.occupancy))}</td>
                <td>{escape(str(item.dominant_morphology or "-"))}</td>
                <td>{escape(_format_value(item.dominant_morphology_ratio))}</td>
                <td>{escape(str(item.dominant_motif or "-"))}</td>
                <td>{escape(_format_value(item.dominant_motif_ratio))}</td>
                <td>{escape(str(item.dominant_pair or "-"))}</td>
                <td>{escape(_format_value(item.decoded_mean_advantage))}</td>
                <td>{escape(_format_value(item.retention_ratio))}</td>
                <td>{escape(item.status)}</td>
              </tr>"""
            )
        return f"""
    <section>
      <h2>Code Diagnostics</h2>
      <article class="panel table-wrap">
        <table>
          <thead><tr><th>Code</th><th>Support</th><th>Occupancy</th><th>Morphology</th><th>Morph Ratio</th><th>Motif</th><th>Motif Ratio</th><th>Pair</th><th>Decoded Adv</th><th>Retention</th><th>Status</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </article>
    </section>"""

    def _render_tie_breaker(self, tie_breaker: Mapping[str, Any]) -> str:
        """渲染 tie-breaker 指标表。

        输入参数:
            tie_breaker: ``Phase1TieBreakerMetrics.to_dict()`` payload。

        输出:
            tie-breaker section HTML。

        使用场景:
            分数接近时解释 checkpoint selector 的二级排序依据。
        """

        return self._render_mapping_table("Tie Breaker Metrics", tie_breaker)

    def _render_drift_diagnostics(
        self,
        diagnostics: Mapping[str, Phase1MetricResult],
    ) -> str:
        """渲染 drift diagnostics 表。

        输入参数:
            diagnostics: drift diagnostic name 到 metric result 的映射。

        输出:
            drift diagnostics section HTML；没有数据时返回空字符串。

        使用场景:
            后续加入跨 epoch drift metric 后，在 report 中统一呈现。
        """

        if not diagnostics:
            return ""
        rows = "\n".join(self._render_metric_row(metric) for metric in diagnostics.values())
        return f"""
    <section>
      <h2>Drift Diagnostics</h2>
      <article class="panel table-wrap">
        <table>
          <thead><tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th><th>Message</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </article>
    </section>"""

    def _render_mapping_table(
        self,
        title: str,
        payload: Mapping[str, Any],
    ) -> str:
        """渲染普通 key-value 表。

        输入参数:
            title: section 标题。
            payload: key-value 数据。

        输出:
            mapping section HTML；payload 为空时返回空字符串。

        使用场景:
            渲染 config、artifacts、tie-breaker 等普通字典。
        """

        if not payload:
            return ""
        rows = "\n".join(
            f"<tr><td>{escape(str(key))}</td><td>{escape(_format_value(value))}</td></tr>"
            for key, value in payload.items()
        )
        return f"""
    <section>
      <h2>{escape(title)}</h2>
      <article class="panel table-wrap">
        <table>
          <thead><tr><th>Key</th><th>Value</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </article>
    </section>"""

    def _style_block(self) -> str:
        """返回 HTML 内联样式。

        输入参数:
            无。

        输出:
            ``<style>`` HTML 字符串。

        使用场景:
            ``render_html()`` 构建自包含静态 report。
        """

        return """<style>
    :root { color-scheme: light; --bg:#f6f7fb; --panel:#fff; --ink:#172033; --muted:#667085; --line:#d8dee8; --soft:#edf1f6; --pass:#147a4d; --pass-bg:#e8f6ef; --fail:#b42318; --fail-bg:#ffe8e5; --warn:#a15c00; --warn-bg:#fff4df; --skip:#475467; --skip-bg:#eef2f6; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .wrap { width:min(1440px, calc(100vw - 40px)); margin:0 auto; }
    header { background:#101828; color:#fff; border-bottom:1px solid #0b1220; }
    .hero { display:flex; justify-content:space-between; gap:24px; padding:28px 0; align-items:flex-end; }
    h1,h2,h3,p { margin:0; }
    h1 { font-size:28px; line-height:1.2; }
    h2 { font-size:18px; margin:22px 0 10px; }
    h3 { font-size:14px; }
    .subtitle { margin-top:8px; color:#cbd5e1; }
    .meta { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; color:#d1d5db; font-size:12px; }
    .meta span { border:1px solid rgba(255,255,255,.18); border-radius:6px; padding:6px 8px; }
    main { padding:22px 0 48px; }
    .grid { display:grid; gap:14px; }
    .summary { grid-template-columns:280px repeat(5, minmax(130px, 1fr)); align-items:stretch; }
    .layers { grid-template-columns:repeat(5, minmax(0, 1fr)); }
    .stack { display:grid; gap:14px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; box-shadow:0 8px 24px rgba(16,24,40,.06); overflow:hidden; padding:14px; }
    .panel-head { padding-bottom:10px; border-bottom:1px solid var(--soft); margin-bottom:8px; }
    .score-card { display:flex; flex-direction:column; gap:10px; }
    .score { font-size:40px; font-weight:760; line-height:1; }
    .label { color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; }
    .value { margin-top:8px; font-size:18px; font-weight:720; overflow-wrap:anywhere; }
    .badge { border-radius:999px; padding:3px 8px; font-size:12px; font-weight:800; display:inline-flex; width:max-content; }
    .badge.pass { color:var(--pass); background:var(--pass-bg); }
    .badge.fail { color:var(--fail); background:var(--fail-bg); }
    .badge.warn { color:var(--warn); background:var(--warn-bg); }
    .badge.skip { color:var(--skip); background:var(--skip-bg); }
    .layer-card { min-height:132px; display:flex; flex-direction:column; justify-content:space-between; }
    .layer-head { display:flex; justify-content:space-between; gap:10px; align-items:flex-start; }
    .layer-stats { display:grid; grid-template-columns:1fr 1fr; gap:8px; border-top:1px solid var(--soft); padding-top:10px; }
    .layer-stats strong { display:block; font-size:20px; }
    .layer-stats span { color:var(--muted); font-size:12px; }
    .table-wrap { overflow:auto; padding:0; }
    table { width:100%; border-collapse:collapse; min-width:760px; }
    th,td { text-align:left; vertical-align:top; border-bottom:1px solid var(--soft); padding:9px 10px; }
    th { color:#475467; font-size:12px; text-transform:uppercase; background:#f9fafb; }
    td { overflow-wrap:anywhere; }
    tr:last-child td { border-bottom:0; }
    @media (max-width: 980px) { .hero { align-items:flex-start; flex-direction:column; } .summary, .layers { grid-template-columns:1fr; } .wrap { width:min(100vw - 24px, 1440px); } }
  </style>"""


__all__ = ["Phase1CodebookReport"]
