"""Phase I codebook validation report template context builder."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from phase1.evaluators.phase1_validation_layers.analysis_code_distribution import build_code_distribution_context

from .phase1_codebook_report_schema import (
    Phase1CodebookReportDocument,
    Phase1CodebookReportHtmlContext,
    Phase1ReportChartGridLine,
    Phase1ReportChartSeries,
    Phase1ReportCodeDiagnosticView,
    Phase1ReportHeader,
    Phase1ReportHeaderItem,
    Phase1ReportKpiRow,
    Phase1ReportLabelTip,
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


_METRIC_DESCRIPTIONS: dict[str, str] = {
    "dp_advantage_vs_flat": (
        "DP teacher 每条 horizon 相对 flat baseline 的平均收益优势；"
        "用于确认示范轨迹整体是否有正交易价值。"
    ),
    "dp_win_rate_vs_flat": (
        "DP teacher 收益超过 flat baseline 的 horizon 比例；"
        "用于检查收益是否广泛存在，而不是只依赖少数样本。"
    ),
    "near_zero_opportunity_ratio": (
        "收益优势接近手续费噪声的样本比例；比例越高，说明训练集中"
        "可学习的清晰交易机会越少。"
    ),
    "fee_sensitivity": (
        "手续费上升后 DP teacher 总优势的保留比例；用于识别"
        "过度依赖微小价差、成本稍变即失效的示范。"
    ),
    "morphology_coverage": (
        "非 neutral 市场形态在 validation 中的覆盖率；用于确认样本包含"
        "足够趋势、反转或波动结构，而不是多数为无结构行情。"
    ),
    "dp_return_concentration_after_top5_removed": (
        "去掉收益最高 5% horizon 后的 DP 总优势；用于检查 teacher 收益"
        "是否被极少数尾部行情支撑。"
    ),
    "validation_action_accuracy": (
        "decoder 在 assigned code 条件下重构 DP 示范动作的准确率；"
        "衡量 code 是否保留基本动作信息。"
    ),
    "reconstruction_loss_gap": (
        "validation reconstruction loss 相对 train reconstruction loss 的泛化差距；"
        "差距过大表示 decoder 泛化不稳。"
    ),
    "active_code_ratio": (
        "validation 中被有效使用的 code 占 codebook 的比例；用于判断"
        "codebook 是否充分利用。"
    ),
    "max_code_occupancy": (
        "单个 code 的最大样本占用率；过高表示 label collapse 或多个行为模式"
        "被混到同一个 code。"
    ),
    "normalized_code_perplexity": (
        "code 使用分布的归一化 perplexity；越高表示使用越均衡，过低说明"
        "codebook 有塌缩风险。"
    ),
    "dead_code_ratio": (
        "validation 中几乎没有样本支持的 code 比例；比例高说明 codebook 容量"
        "未被有效使用。"
    ),
    "assignment_churn_recent_mean": (
        "相邻或近期 checkpoint 之间 label assignment 的平均变动率；用于检查"
        "code 语义是否稳定。"
    ),
    "code_lifetime_pass_ratio": (
        "active code 中生命周期达到稳定要求的比例；用于过滤短暂出现、"
        "不可复用的 code。"
    ),
    "nearest_second_margin_median": (
        "最近 code 与第二近 code 距离差的中位数；margin 低表示样本分配"
        "边界模糊、label 容易抖动。"
    ),
    "direction_accuracy": (
        "decoded 动作方向与 DP 示范方向一致的比例；用于检查 long/short/flat"
        "方向是否被正确保留。"
    ),
    "entry_timing_error_median": (
        "decoded 入场时点相对 DP 示范入场时点的中位误差；误差大表示"
        "交易时机被重构偏移。"
    ),
    "decoder_turnover_error": (
        "decoded 换手强度相对 DP 示范的误差；用于检查 decoder 是否过度交易"
        "或过度平滑动作。"
    ),
    "weak_support_code_ratio": (
        "support 不足的 active code 比例；比例高说明大量 code 级统计不稳定。"
    ),
    "weak_morphology_code_ratio": (
        "dominant morphology 不清晰的 active code 比例；比例高表示 code 与"
        "市场结构关系弱。"
    ),
    "weak_motif_code_ratio": (
        "dominant trading motif 不清晰的 active code 比例；比例高表示 code 内部"
        "交易意图混杂。"
    ),
    "weak_pair_code_ratio": (
        "dominant morphology-motif pair 不清晰的 active code 比例；用于检查"
        "市场形态与交易行为是否稳定绑定。"
    ),
    "weak_lift_nonprofitable_code_ratio": (
        "同时缺少 morphology lift 或盈利性的弱 code 比例；用于识别"
        "既不可解释又不可交易的原型。"
    ),
    "intra_code_action_similarity": (
        "同一 code 内 decoded action 序列的一致性；越高表示该 code 行为语义"
        "越集中。"
    ),
    "inter_intra_separation": (
        "code 间行为中心距离与 code 内差异的比值；越高表示不同 archetype"
        "分得越开。"
    ),
    "latent_silhouette_score": (
        "latent 空间中 assigned code 的 silhouette 聚类分数；用于判断 label"
        "边界是否清晰。"
    ),
    "duplicate_code_pair_count": (
        "行为相似度超过重复阈值的 code pair 数量；数量高说明 codebook 中"
        "存在冗余原型。"
    ),
    "profitable_code_coverage": (
        "具备正 decoded advantage 的 active code 覆盖率；用于确认盈利能力"
        "不是只集中在少数 code。"
    ),
    "mean_decoded_advantage_vs_flat": (
        "assigned-label decoded 策略相对 flat baseline 的平均收益优势；"
        "这是 codebook oracle 盈利性的最低要求。"
    ),
    "decoded_win_rate_vs_flat": (
        "decoded 收益超过 flat baseline 的 horizon 比例；用于判断盈利是否"
        "稳定分布。"
    ),
    "mean_advantage_vs_random_label": (
        "assigned label decoded return 相对 random label decoded return 的平均优势；"
        "用于验证 encoder 分配的 label 不是随机 ID。"
    ),
    "random_label_relative_lift": (
        "assigned label 相对 random label 的收益提升比例；越高说明 label assignment"
        "携带越多交易信息。"
    ),
    "retention_ratio": (
        "decoded 总优势相对 DP teacher 总优势的保留比例；用于衡量压缩后"
        "保留了多少 teacher 盈利能力。"
    ),
    "downside_control": (
        "decoded 策略累计收益曲线的最大回撤；用于避免平均收益为正但"
        "自身回撤结构过差。"
    ),
    "risk_adjusted_return": (
        "decoded return 的风险调整收益；用于把收益和波动/下行风险一起评估。"
    ),
    "risk_adjusted_return_vs_random": (
        "decoded 策略风险调整收益相对 random label baseline 的优势；用于确认"
        "收益质量也优于随机标签。"
    ),
    "top_5_contribution": (
        "收益最高 5% horizon 对总 decoded profit 的贡献比例；比例高说明"
        "盈利过度依赖尾部样本。"
    ),
    "trimmed_decoded_advantage": (
        "去掉收益最高和最低尾部样本后的 decoded 平均优势；用于检查主体样本"
        "是否仍有正期望。"
    ),
    "fee_drag": (
        "手续费占 gross profit 的比例；过高通常表示 decoder 交易过碎或收益空间"
        "被成本吞噬。"
    ),
    "turnover_return_correlation": (
        "换手强度与收益的相关性；显著为负表示越交易越亏，可能存在"
        "过度交易问题。"
    ),
    "bad_code_ratio": (
        "decoded profitability 或行为质量不达标的坏 code 比例；用于定位"
        "整体通过但局部失效的 codebook。"
    ),
    "dominant_pair_positive_ratio": (
        "dominant morphology-motif pair 中 mean decoded advantage 为正的比例；"
        "用于检查主要形态-行为组合是否大多可盈利。"
    ),
    "probe_top1_accuracy": (
        "用 horizon 起点可见状态预测 assigned label 的 top-1 accuracy；"
        "衡量 Phase II selector 是否可能学到这些 label。"
    ),
    "probe_top3_accuracy": (
        "label probe 的 top-3 accuracy；用于判断 selector 是否能把候选 code"
        "缩小到少数几个。"
    ),
    "probe_balanced_accuracy": (
        "按 code 平衡后的 probe accuracy；用于避免模型只预测高频 code。"
    ),
    "label_entropy_given_morphology": (
        "给定市场形态后的 label 条件熵；值越低表示市场结构对 label"
        "有更强约束。"
    ),
    "mutual_information_lift": (
        "label 与可见状态之间的互信息提升；用于衡量 label 是否可由当前状态"
        "预测，而不是主要依赖未来路径。"
    ),
    "probe_return_retention": (
        "probe 预测 label 的 decoded return 相对 oracle assigned label decoded return"
        "的保留比例；用于估计 selector 学成后的收益上限。"
    ),
    "morphology_distribution_kl": (
        "validation 市场形态分布相对 train 的 KL divergence；值高说明"
        "验证期行情结构发生迁移。"
    ),
    "code_usage_kl": (
        "validation code usage 分布相对 train 的 KL divergence；值高说明"
        "encoder assignment 或市场结构在验证期发生漂移。"
    ),
    "motif_distribution_kl": (
        "validation 行为 motif 分布相对 train 的 KL divergence；值高说明"
        "decoder 行为结构或样本结构发生迁移。"
    ),
    "reconstruction_generalization_gap": (
        "validation 与 train reconstruction quality 的泛化差距；用于解释"
        "验证集重构或收益变差是否来自泛化问题。"
    ),
    "label_predictability_gap": (
        "train/validation label predictability 的差距；值高说明 selector 可学习性"
        "在验证期下降。"
    ),
    "per_code_return_gap": (
        "train/validation per-code return 的差距；用于定位某些 code 在验证期"
        "盈利结构是否失效。"
    ),
    "reconstruction_loss": (
        "validation reconstruction loss；越低表示 decoder 越能重构 DP 示范动作，"
        "常作为分数接近时的稳定性决胜字段。"
    ),
}


_MORPHOLOGY_DESCRIPTIONS: dict[str, str] = {
    "downtrend": "下跌趋势行情，价格整体向下推进；通常更适合观察 short 或防守型行为。",
    "uptrend": "上涨趋势行情，价格整体向上推进；通常更适合观察 long 或顺势行为。",
    "neutral": "中性或无明显方向行情，趋势信号弱；该类过多时可能降低 label 的结构可解释性。",
    "range-high-vol": "高波动震荡行情，价格没有稳定方向但振幅较大；需要关注换手和手续费拖累。",
    "range-low-vol": "低波动震荡行情，价格方向和振幅都较弱；交易机会可能更接近成本噪声。",
    "reversal-down": "由上行或强势状态转向下行的反转结构；用于识别潜在做空或止盈切换场景。",
    "reversal-up": "由下行或弱势状态转向上行的反转结构；用于识别潜在做多或空头回补场景。",
    "volatile-mixed": "高波动且方向混杂的行情，趋势与反转信号交错；对 selector 和 decoder 稳定性要求更高。",
}


def _describe_morphology(label: str) -> str:
    """返回 morphology 标签的中文说明。"""

    return _MORPHOLOGY_DESCRIPTIONS.get(
        label,
        f"{label} 市场形态；表示由行情结构分类器归纳出的 validation 样本状态。",
    )


def _describe_motif(label: str) -> str:
    """返回 trading motif 标签的中文说明。"""

    parts = tuple(part.strip() for part in label.split("+"))
    descriptions: list[str] = []
    for part in parts:
        if part == "long":
            descriptions.append("long 表示以做多或持有多头为主")
        elif part == "short":
            descriptions.append("short 表示以做空或持有空头为主")
        elif part == "flat":
            descriptions.append("flat 表示以空仓或降低风险暴露为主")
        elif part == "early":
            descriptions.append("early 表示动作主要出现在 horizon 前段")
        elif part == "mid":
            descriptions.append("mid 表示动作主要出现在 horizon 中段")
        elif part == "late":
            descriptions.append("late 表示动作主要出现在 horizon 后段")
        elif part == "hold":
            descriptions.append("hold 表示持仓相对连续，换手较少")
        elif part == "switch":
            descriptions.append("switch 表示方向切换或换手较多")
        elif part == "with-recent-move":
            descriptions.append("with-recent-move 表示顺近期价格运动方向")
        elif part == "against-recent-move":
            descriptions.append("against-recent-move 表示逆近期价格运动方向")
        else:
            descriptions.append(f"{part} 是 motif 的一个行为标签")
    return "；".join(descriptions) + "。"


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
            code_distribution_view=build_code_distribution_context(
                vq_internal_payload.code_distributions
            )
            if vq_internal_payload is not None
            else None,
            tie_breaker_rows=self._build_mapping_rows(
                validation.tie_breaker_metrics.to_dict(),
                descriptions=_METRIC_DESCRIPTIONS,
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
            sample_count = (
                vq_internal_payload
                .code_distributions
                .code_distribution_total_sample_count
            )
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
        is_reference_layer = layer.name == "label_predictability"
        return Phase1ReportLayerView(
            layer_id=str(layer.layer_id),
            name=layer.name,
            badge_class="warn" if is_reference_layer else _badge_class(layer.passed),
            status_label=(
                "REF"
                if is_reference_layer
                else "PASS" if layer.passed else "FAIL"
            ),
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

        description = self._metric_description(metric)
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
            description=description,
        )

    def _metric_description(self, metric: Phase1MetricResult) -> str:
        """构建 metric hover 中的指标含义和判定说明。"""

        description = _METRIC_DESCRIPTIONS.get(
            metric.name,
            "该指标用于 Phase I codebook hard gate 或 drift 诊断；"
            "需要结合当前值、阈值方向和风险 findings 一起判断。",
        )
        if metric.message:
            return f"指标含义: {description} 当前判定: {metric.message}"
        return f"指标含义: {description}"

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
            max_abs_advantage = self._max_abs_profit_value(
                item.decoded_mean_advantage
                for item in code_diagnostics
                if item.decoded_mean_advantage is not None
            )
            return tuple(
                Phase1ReportProfitSeriesRow(
                    code_id=str(item.code_id),
                    label=f"code {item.code_id}",
                    value=self._format_signed_value(
                        item.decoded_mean_advantage
                    ),
                    badge_class=self._profit_badge_class(
                        item.decoded_mean_advantage
                    ),
                    bar_width=self._profit_bar_width(
                        item.decoded_mean_advantage,
                        max_abs_advantage,
                    ),
                )
                for item in code_diagnostics
                if item.decoded_mean_advantage is not None
            )
        if oracle_profitability_payload is None:
            return ()
        max_abs_advantage = self._max_abs_profit_value(
            item.mean_advantage
            for item in oracle_profitability_payload.per_code_profitability
        )
        return tuple(
            Phase1ReportProfitSeriesRow(
                code_id=str(item.code_id),
                label=f"code {item.code_id}",
                value=self._format_signed_value(item.mean_advantage),
                badge_class=self._profit_badge_class(item.mean_advantage),
                bar_width=self._profit_bar_width(
                    item.mean_advantage,
                    max_abs_advantage,
                ),
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
                motif_headers=(),
                rows=(),
                cells=(),
            )
        cells = tuple(oracle_profitability_payload.pair_profitability_matrix)
        if not cells:
            return Phase1ReportPairProfitabilityMatrix(
                morphologies=(),
                motifs=(),
                motif_headers=(),
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
                    morphology_description=_describe_morphology(morphology),
                    cells=tuple(row_cells),
                )
            )
        return Phase1ReportPairProfitabilityMatrix(
            morphologies=morphologies,
            motifs=motifs,
            motif_headers=tuple(
                Phase1ReportLabelTip(
                    label=motif,
                    description=_describe_motif(motif),
                )
                for motif in motifs
            ),
            rows=tuple(rows),
            cells=tuple(flat_cells),
            grid_template_columns=(
                f"minmax(86px, 1.15fr) repeat({len(motifs)}, minmax(0, 1fr))"
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
        chart = self._build_line_chart_from_series(
            raw_series,
            title="全部累计收益",
        )
        if not chart.series:
            return chart

        detail_series = tuple(
            item for item in raw_series if item[0] != "dp"
        )
        if self._needs_decoder_detail_chart(raw_series, detail_series):
            detail_chart = self._build_line_chart_from_series(
                detail_series,
                title="Decoded / Random / Flat 明细",
            )
            if detail_chart.series:
                return Phase1ReportLineChart(
                    title=chart.title,
                    width=chart.width,
                    height=chart.height,
                    grid_lines=chart.grid_lines,
                    series=chart.series,
                    y_min=chart.y_min,
                    y_max=chart.y_max,
                    x_axis_label=chart.x_axis_label,
                    detail_charts=(detail_chart,),
                )
        return chart

    def _build_line_chart_from_series(
        self,
        raw_series: list[tuple[str, str, str, tuple[float, ...], str]]
        | tuple[tuple[str, str, str, tuple[float, ...], str], ...],
        *,
        title: str,
    ) -> Phase1ReportLineChart:
        """把累计数值序列转换成静态 SVG 折线图坐标。"""

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
            title=title,
            width=str(int(width)),
            height=str(int(height)),
            grid_lines=grid_lines,
            series=tuple(chart_series),
            y_min=_format_value(y_min),
            y_max=_format_value(y_max),
            x_axis_label="validation horizon order",
        )

    def _needs_decoder_detail_chart(
        self,
        raw_series: list[tuple[str, str, str, tuple[float, ...], str]],
        detail_series: tuple[tuple[str, str, str, tuple[float, ...], str], ...],
    ) -> bool:
        """判断 decoded/random 是否会被 DP 量级压扁。"""

        if not detail_series:
            return False
        full_range = self._series_value_range(tuple(raw_series))
        detail_range = self._series_value_range(detail_series)
        if detail_range <= 0.0:
            return False
        return full_range / detail_range >= 20.0

    def _series_value_range(
        self,
        raw_series: tuple[tuple[str, str, str, tuple[float, ...], str], ...],
    ) -> float:
        """计算一组序列的有限值范围。"""

        finite_values = [
            value
            for _, _, _, values, _ in raw_series
            for value in values
            if math.isfinite(value)
        ]
        if not finite_values:
            return 0.0
        return max(finite_values) - min(finite_values)

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

    def _max_abs_profit_value(self, values: Any) -> float:
        """计算 per-code 盈利条形图的归一化尺度。"""

        finite_values = [
            abs(float(value))
            for value in values
            if value is not None and math.isfinite(float(value))
        ]
        return max(finite_values, default=1.0) or 1.0

    def _profit_bar_width(
        self,
        value: float | None,
        max_abs_value: float,
    ) -> str:
        """按绝对收益生成 per-code 条形宽度。"""

        if value is None or not math.isfinite(value):
            return "0%"
        width = abs(value) / max_abs_value * 100.0
        return f"{max(2.0, min(100.0, width)):.3g}%"

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
        *,
        descriptions: Mapping[str, str] | None = None,
    ) -> tuple[Phase1ReportMappingRow, ...]:
        """构建普通 key-value 表格上下文。"""

        description_map = descriptions or {}
        return tuple(
            Phase1ReportMappingRow(
                key=str(key),
                value=_format_value(value),
                description=description_map.get(str(key), ""),
            )
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
