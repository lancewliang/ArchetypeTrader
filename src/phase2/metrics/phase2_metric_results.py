"""Phase II metrics result payload 骨架。

文件功能说明:
    本文件定义 Phase II evaluator 已经计算完成的指标结果对象。它们是
    checkpoint selector、artifact store 和 report 之间共享的结果 payload。

设计边界:
    - 只承载 evaluator 产出的 metrics、layer results 和 report payloads；
    - 不计算指标、不读取模型、不访问训练数据；
    - 不负责 checkpoint 模型权重保存；
    - 不判断 best checkpoint，也不应用 hard gate 或 tie-breaker。

使用场景:
    ``Phase2Evaluator`` 评估 validation/test split 后生成这些对象；
    ``Phase2ArtifactStore`` 负责保存/读取它们；
    ``Phase2CheckpointSelector`` 和 report 只消费其中的稳定字段。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, TypeAlias

from pydantic import Field, field_validator, model_validator

from src.utils import PydanticMappingModel


MetricSeverity = Literal["pass", "warn", "fail", "skip"]
MetricDirection = Literal["greater_is_better", "less_is_better", "between", "equal"]
MetricThresholdValue = float | tuple[float, float] | bool | None

if TYPE_CHECKING:
    from .phase2_validation_layer0_evaluation_validity import (
        Phase2EvaluationValidityPayload,
        Phase2EvaluationValidityMetrics,
    )
    from .phase2_validation_layer1_selector_profitability import (
        Phase2SelectorProfitabilityPayload,
        Phase2SelectorProfitabilityMetrics,
    )
    from .phase2_validation_layer2_baseline_uplift import (
        Phase2BaselineUpliftPayload,
        Phase2BaselineUpliftMetrics,
    )
    from .phase2_validation_layer3_demonstration_consistency import (
        Phase2DemonstrationConsistencyPayload,
        Phase2DemonstrationConsistencyMetrics,
    )
    from .phase2_validation_layer4_code_usage_collapse import (
        Phase2CodeUsageCollapsePayload,
        Phase2CodeUsageCollapseMetrics,
        Phase2PerCodeUsageDiagnostic,
    )
    from .phase2_validation_layer5_generalization_stability import (
        Phase2GeneralizationStabilityPayload,
        Phase2PredictabilityPayload,
        Phase2GeneralizationStabilityMetrics,
    )

    Phase2LayerMetrics: TypeAlias = (
        Phase2EvaluationValidityMetrics
        | Phase2SelectorProfitabilityMetrics
        | Phase2BaselineUpliftMetrics
        | Phase2DemonstrationConsistencyMetrics
        | Phase2CodeUsageCollapseMetrics
        | Phase2GeneralizationStabilityMetrics
    )
    Phase2LayerPayload: TypeAlias = (
        Phase2EvaluationValidityPayload
        | Phase2SelectorProfitabilityPayload
        | Phase2BaselineUpliftPayload
        | Phase2DemonstrationConsistencyPayload
        | Phase2CodeUsageCollapsePayload
        | Phase2GeneralizationStabilityPayload
        | Phase2PredictabilityPayload
    )
else:
    Phase2LayerMetrics: TypeAlias = object
    Phase2LayerPayload: TypeAlias = object


class Phase2MetricResult(PydanticMappingModel):
    """单个 Phase II validation metric 的判定结果。"""

    # 指标稳定名称。用途：report、JSON 和 rule 追踪；方向：无好坏方向。
    name: str

    # 指标实际值。用途：展示和审计 hard gate 判定；方向：由 direction 字段定义。
    value: int | float | str | bool | None

    # 人类可读阈值表达式，例如 ">= 0.5"。用途：report 展示；方向：由表达式定义。
    threshold: str

    # 判定严重级别。用途：区分 pass/warn/fail/skip；方向：pass 最好，fail 最差。
    severity: MetricSeverity

    # 该指标是否通过规则。用途：聚合 layer passed；方向：True 更好。
    passed: bool

    # 指标所属 layer 稳定名称。用途：分组展示和审计；方向：无好坏方向。
    layer: str

    # 指标解释、失败原因或诊断建议。用途：report 文案；方向：无好坏方向。
    message: str = ""

    # 机器可读阈值。用途：后续重放或结构化审计；方向：由 direction 字段定义。
    threshold_value: MetricThresholdValue = None

    # 指标方向：越大越好、越小越好、区间约束或等值约束。
    direction: MetricDirection | None = None

    # 当前值到阈值的距离。用途：排序风险程度或展示裕量；方向：通常越大代表
    # 离通过边界越安全，具体含义由 rule helper 生成。
    distance_to_threshold: float | None = None

    @field_validator("threshold_value", mode="before")
    @classmethod
    def _restore_threshold_value(cls, value: Any) -> MetricThresholdValue:
        """Restore legacy list/tuple threshold payloads."""

        return _threshold_value_from_payload(value)


class Phase2LayerResult(PydanticMappingModel):
    """单个 Phase II validation layer 的判定结果。"""

    # layer 数字编号，0-5。用途：固定展示顺序；方向：无好坏方向。
    layer_id: int

    # layer 稳定名称。用途：report 分组和规则追踪；方向：无好坏方向。
    name: str

    # 本层是否通过。用途：checkpoint selector hard gate 聚合；方向：True 更好。
    passed: bool

    # 本层下属 metric 判定结果。用途：报告阈值细节；方向：由每个 metric 决定。
    metrics: tuple[Phase2MetricResult, ...]


def _threshold_value_from_payload(value: Any) -> MetricThresholdValue:
    """恢复机器可读阈值。"""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, tuple | list):
        if len(value) != 2:
            return None
        return (float(value[0]), float(value[1]))
    return float(value)


class Phase2ValidationMetrics(PydanticMappingModel):
    """Phase II validation 核心指标 payload。

    功能说明:
        保存 Phase II selector validation/test split 上的核心可排序指标。指标同时
        覆盖 selector 收益、交易行为、assigned-label baseline、random baseline
        和 code usage 诊断。

    设计边界:
        本类只承载 evaluator 已经计算好的数值，不负责收益计算、baseline 执行、
        阈值判断或 checkpoint 选择。

    使用场景:
        ``Phase2Evaluator.evaluate()`` 生成该对象，并放入
        ``Phase2ValidationResult.metrics``；checkpoint selector 和 report 读取该
        对象中的稳定字段进行排序和展示。
    """

    # selector greedy action 的平均 horizon return。用途：checkpoint selector 主排序
    # 指标；方向：越大越好，必须结合 risk 和 baseline uplift 审计。
    mean_return: float

    # selector greedy action 的 return 中位数。用途：降低极端收益样本对平均值的
    # 干扰；方向：越大越好，明显低于 0 表示收益可能依赖少数尾部样本。
    median_return: float

    # 类 Sharpe 风险调整收益指标。用途：衡量单位波动下的平均收益质量；方向：
    # 越大越好，低于或接近 0 表示风险调整收益不足。
    sharpe_like: float

    # horizon return 大于 0 的比例。用途：衡量正收益样本覆盖面；方向：越大越好，
    # 但不能替代收益幅度。
    win_rate: float

    # 平均换手率或行为强度指标。用途：诊断交易成本和过度交易风险；方向：通常
    # 越小越稳，但过低也可能表示策略退化为不交易，需要结合 return。
    mean_turnover: float


class Phase2ReportPairProfitabilityPayloadRow(PydanticMappingModel):
    """Report payload 中 Dominant Pair heatmap 的单个 cell 聚合行。"""

    morphology: str
    motif: str
    support: int
    selector_mean_return: float
    kl_mean_return: float
    random_mean_return: float
    mean_advantage_vs_kl: float
    mean_advantage_vs_random: float
    win_rate: float
    fee_drag_ratio: float
    dominant_selected_code: int | None
    dominant_selected_code_ratio: float


class Phase2ReportCodeDiagnosticPayloadRow(PydanticMappingModel):
    """Report payload 中 code 级诊断表的单个聚合行。"""

    code_id: int
    selector_support: int
    selector_usage_ratio: float
    kl_support: int
    kl_usage_ratio: float
    usage_delta: float
    selector_mean_return: float
    kl_mean_return: float
    uplift_vs_kl: float
    selector_win_rate: float
    selector_fee_drag_ratio: float
    selector_turnover: float
    dominant_morphology: str | None
    dominant_morphology_ratio: float
    dominant_motif: str | None
    dominant_motif_ratio: float
    dominant_pair: str | None
    dominant_pair_ratio: float
    mean_q_margin: float
    low_confidence_ratio: float
    profitable_deviation_count: int
    unprofitable_deviation_count: int
    unprofitable_deviation_rate: float
    status: str
    risk_reason: str

    @field_validator(
        "dominant_morphology",
        "dominant_motif",
        "dominant_pair",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, value: Any) -> str | None:
        """Keep legacy optional string normalization."""

        return None if value in (None, "") else str(value)


class Phase2ReportCodeCount(PydanticMappingModel):
    """Report payload 中某个 code 的样本数。"""

    code_id: int
    count: int


class Phase2ReportCodeUsageDistribution(PydanticMappingModel):
    """Report payload 中 selector 和 assigned-label 的 code 使用分布。"""

    selector: tuple[Phase2ReportCodeCount, ...] = ()
    kl: tuple[Phase2ReportCodeCount, ...] = ()

    @field_validator("selector", "kl", mode="before")
    @classmethod
    def _restore_code_counts(cls, value: Any) -> tuple[Phase2ReportCodeCount, ...]:
        """Keep legacy support for dict and object-like code count rows."""

        if value is None:
            return ()
        return tuple(_code_count_from_value(item) for item in value)


class Phase2ReportCumulativeReturns(PydanticMappingModel):
    """Report payload 中各 baseline 的累计收益曲线。"""

    selector: tuple[float, ...] = ()
    kl: tuple[float, ...] = ()
    random: tuple[float, ...] = ()
    oracle: tuple[float, ...] = ()
    hold: tuple[float, ...] = ()

    @field_validator("selector", "kl", "random", "oracle", "hold", mode="before")
    @classmethod
    def _restore_float_tuple(cls, value: Any) -> tuple[float, ...]:
        """Keep legacy cumulative-return sequence normalization."""

        if value is None or isinstance(value, str | bytes):
            return ()
        try:
            return tuple(float(item) for item in value)
        except TypeError:
            return ()


def _code_count_from_value(value: Any) -> Phase2ReportCodeCount:
    """把 dict 或 dataclass-like code count 转成强类型对象。"""

    if isinstance(value, Phase2ReportCodeCount):
        return value
    if isinstance(value, Mapping):
        return Phase2ReportCodeCount.model_validate(value)
    return Phase2ReportCodeCount(
        code_id=int(getattr(value, "code_id")),
        count=int(getattr(value, "count")),
    )


class Phase2ValidationPayloads(PydanticMappingModel):
    """Phase II validation/report 需要复用的聚合 payload。

    本对象只保存 evaluator 已经聚合好的过程数据和诊断数据，不保存完整逐样本
    ``selection_trace``。
    """

    # Layer 0 评估可信度过程数据。用途：审计 split、epoch、样本数和失败计数；
    # 方向：过程数据本身无排序方向，由 Layer 0 metrics 转换为好坏判定。
    evaluation_validity_payload: Any | None = None

    # Layer 1 selector 收益过程数据。用途：保存收益、gross return、fee、turnover
    # 序列以便复查聚合指标；方向：过程数据无直接方向。
    selector_profitability_payload: Any | None = None

    # Layer 2 baseline 对比过程数据。用途：保存 selector/assigned/random/oracle
    # return 序列；方向：过程数据无直接方向。
    baseline_uplift_payload: Any | None = None

    # Layer 3 demonstration consistency 过程数据。用途：保存 selected/assigned code、
    # return 和 Q value 序列；方向：过程数据无直接方向。
    demonstration_consistency_payload: Any | None = None

    # Layer 4 code usage 过程数据。用途：保存 selected code 分布和 per-code 诊断；
    # 方向：过程数据无直接方向。
    code_usage_collapse_payload: Any | None = None

    # Layer 5 泛化稳定性过程数据。用途：保存 score/churn/Q scale 历史和 probe
    # payload；方向：过程数据无直接方向。
    generalization_stability_payload: Any | None = None

    # Report per-code 盈利对比行。默认从 code_usage_collapse_payload 的
    # per_code_diagnostics 复用，避免重复组装同一对象。
    per_code_profitability_comparison: tuple[Any, ...] = ()

    # Report Dominant Pair heatmap 行。用途：展示 morphology/motif 组合收益。
    selector_pair_profitability_matrix: tuple[
        Phase2ReportPairProfitabilityPayloadRow,
        ...,
    ] = ()

    # Report code 级诊断表行。用途：展示 code 行为归因、风险原因和偏离质量。
    code_diagnostics: tuple[Phase2ReportCodeDiagnosticPayloadRow, ...] = ()

    # Report code usage 分布。用途：展示 selector 与 assigned-label 使用差异。
    codebook_usage_distribution: Phase2ReportCodeUsageDistribution = Field(
        default_factory=Phase2ReportCodeUsageDistribution
    )

    # Report 各 baseline 累计收益曲线。用途：HTML 静态曲线和 JSON payload。
    oracle_label_cumulative_returns: Phase2ReportCumulativeReturns = Field(
        default_factory=Phase2ReportCumulativeReturns
    )

    @field_validator("codebook_usage_distribution", mode="before")
    @classmethod
    def _default_codebook_usage_distribution(
        cls,
        value: Any,
    ) -> Phase2ReportCodeUsageDistribution | Any:
        """Use an empty distribution for legacy null payloads."""

        return Phase2ReportCodeUsageDistribution() if value is None else value

    @field_validator("oracle_label_cumulative_returns", mode="before")
    @classmethod
    def _default_cumulative_returns(
        cls,
        value: Any,
    ) -> Phase2ReportCumulativeReturns | Any:
        """Use empty cumulative returns for legacy null payloads."""

        return Phase2ReportCumulativeReturns() if value is None else value

    @model_validator(mode="before")
    @classmethod
    def _restore_payloads_payload(cls, payload: Any) -> Any:
        """Restore nested validation/report payload objects."""

        if not isinstance(payload, Mapping):
            return payload

        from .phase2_validation_layer0_evaluation_validity import (
            Phase2EvaluationValidityPayload,
        )
        from .phase2_validation_layer1_selector_profitability import (
            Phase2SelectorProfitabilityPayload,
        )
        from .phase2_validation_layer2_baseline_uplift import (
            Phase2BaselineUpliftPayload,
        )
        from .phase2_validation_layer3_demonstration_consistency import (
            Phase2DemonstrationConsistencyPayload,
        )
        from .phase2_validation_layer4_code_usage_collapse import (
            Phase2CodeUsageCollapsePayload,
            Phase2PerCodeUsageDiagnostic,
        )
        from .phase2_validation_layer5_generalization_stability import (
            Phase2GeneralizationStabilityPayload,
        )

        cumulative_payload = (
            payload.get("oracle_label_cumulative_returns")
            or payload.get("cumulative_return_curves")
            or {}
        )
        restored = dict(payload)
        nested_payload_builders = {
            "evaluation_validity_payload": Phase2EvaluationValidityPayload,
            "selector_profitability_payload": Phase2SelectorProfitabilityPayload,
            "baseline_uplift_payload": Phase2BaselineUpliftPayload,
            "demonstration_consistency_payload": Phase2DemonstrationConsistencyPayload,
            "code_usage_collapse_payload": Phase2CodeUsageCollapsePayload,
            "generalization_stability_payload": Phase2GeneralizationStabilityPayload,
        }
        for key, builder in nested_payload_builders.items():
            value = restored.get(key)
            if isinstance(value, Mapping):
                restored[key] = builder.model_validate(value)
            elif value is not None and not isinstance(value, builder):
                restored[key] = None

        restored["per_code_profitability_comparison"] = tuple(
            item
            if isinstance(item, Phase2PerCodeUsageDiagnostic)
            else Phase2PerCodeUsageDiagnostic.model_validate(item)
            for item in (
                payload.get("per_code_profitability_comparison", ()) or ()
            )
            if isinstance(item, Phase2PerCodeUsageDiagnostic | Mapping)
        )
        restored["selector_pair_profitability_matrix"] = tuple(
            item
            if isinstance(item, Phase2ReportPairProfitabilityPayloadRow)
            else Phase2ReportPairProfitabilityPayloadRow.model_validate(item)
            for item in (
                payload.get("selector_pair_profitability_matrix", ()) or ()
            )
            if isinstance(item, Phase2ReportPairProfitabilityPayloadRow | Mapping)
        )
        restored["code_diagnostics"] = tuple(
            item
            if isinstance(item, Phase2ReportCodeDiagnosticPayloadRow)
            else Phase2ReportCodeDiagnosticPayloadRow.model_validate(item)
            for item in (payload.get("code_diagnostics", ()) or ())
            if isinstance(item, Phase2ReportCodeDiagnosticPayloadRow | Mapping)
        )
        distribution = payload.get("codebook_usage_distribution")
        restored["codebook_usage_distribution"] = (
            distribution
            if isinstance(distribution, Phase2ReportCodeUsageDistribution)
            else Phase2ReportCodeUsageDistribution.model_validate(distribution)
            if isinstance(distribution, Mapping)
            else Phase2ReportCodeUsageDistribution()
        )
        restored["oracle_label_cumulative_returns"] = (
            cumulative_payload
            if isinstance(cumulative_payload, Phase2ReportCumulativeReturns)
            else Phase2ReportCumulativeReturns.model_validate(cumulative_payload)
            if isinstance(cumulative_payload, Mapping)
            else Phase2ReportCumulativeReturns()
        )
        return restored

    @model_validator(mode="after")
    def _normalize_nested_payloads(self) -> "Phase2ValidationPayloads":
        """标准化 validation/report payload 的嵌套强类型字段。"""

        from .phase2_validation_layer4_code_usage_collapse import (
            Phase2PerCodeUsageDiagnostic,
        )

        per_code_rows = self.per_code_profitability_comparison or (
            self.code_usage_collapse_payload.per_code_diagnostics
            if self.code_usage_collapse_payload is not None
            else ()
        )
        object.__setattr__(
            self,
            "per_code_profitability_comparison",
            tuple(
                item
                if isinstance(item, Phase2PerCodeUsageDiagnostic)
                else Phase2PerCodeUsageDiagnostic.model_validate(item)
                for item in per_code_rows
            ),
        )
        object.__setattr__(
            self,
            "selector_pair_profitability_matrix",
            tuple(
                item
                if isinstance(item, Phase2ReportPairProfitabilityPayloadRow)
                else Phase2ReportPairProfitabilityPayloadRow.model_validate(item)
                for item in (self.selector_pair_profitability_matrix or ())
            ),
        )
        object.__setattr__(
            self,
            "code_diagnostics",
            tuple(
                item
                if isinstance(item, Phase2ReportCodeDiagnosticPayloadRow)
                else Phase2ReportCodeDiagnosticPayloadRow.model_validate(item)
                for item in (self.code_diagnostics or ())
            ),
        )
        if isinstance(self.codebook_usage_distribution, Mapping):
            object.__setattr__(
                self,
                "codebook_usage_distribution",
                Phase2ReportCodeUsageDistribution.model_validate(
                    self.codebook_usage_distribution
                ),
            )
        elif self.codebook_usage_distribution is None:
            object.__setattr__(
                self,
                "codebook_usage_distribution",
                Phase2ReportCodeUsageDistribution(),
            )
        if isinstance(self.oracle_label_cumulative_returns, Mapping):
            object.__setattr__(
                self,
                "oracle_label_cumulative_returns",
                Phase2ReportCumulativeReturns.model_validate(
                    self.oracle_label_cumulative_returns
                ),
            )
        elif self.oracle_label_cumulative_returns is None:
            object.__setattr__(
                self,
                "oracle_label_cumulative_returns",
                Phase2ReportCumulativeReturns(),
            )
        return self


class Phase2ValidationResult(PydanticMappingModel):
    """Phase II validation 结果摘要。

    功能说明:
        保存 evaluator 已经计算好的 selection metrics 和诊断信息，作为
        validation result、report 和 checkpoint selector 的共享输入。

    设计边界:
        本类只承载结果，不负责计算指标、应用阈值或决定 best checkpoint。
        ``metrics`` 应保存可排序、可报告的稳定字段；``layers`` 保存分层判定；
        ``payloads`` 保存报表卡片需要复用的聚合过程数据。

    使用场景:
        ``Phase2Evaluator`` 评估某个 epoch 后返回该对象，再由 artifact store
        保存为 Phase II validation result payload。
    """

    # checkpoint selector 直接消费的核心摘要指标。用途：排序和摘要展示；方向：
    # 由 Phase2ValidationMetrics 各字段定义。
    metrics: Phase2ValidationMetrics

    # hard-gate/reference layer 判定结果。用途：过滤不可用 checkpoint、展示风险；
    # 方向：通过层越多越好，但 Layer 5 当前主要是 warn/reference。
    layers: tuple[Phase2LayerResult, ...] = ()

    # Layer 0-5 强类型 raw metrics 和本层中间 payload。用途：完整审计每层聚合
    # 指标；方向：由具体 metrics 字段定义。
    layer_computations: tuple[Any, ...] = ()

    # 报表和诊断卡片复用的聚合 payload。用途：HTML/JSON report 展示；方向：
    # 展示数据，无直接排序方向。
    payloads: Phase2ValidationPayloads | None = None

    @model_validator(mode="before")
    @classmethod
    def _restore_validation_result_payload(cls, payload: Any) -> Any:
        """Restore nested validation result fields from dict payloads."""

        if not isinstance(payload, Mapping):
            return payload
        metrics_payload = payload.get("metrics")
        if isinstance(metrics_payload, Phase2ValidationMetrics):
            restored_metrics = metrics_payload
        elif isinstance(metrics_payload, Mapping):
            restored_metrics = Phase2ValidationMetrics.model_validate(metrics_payload)
        else:
            raise ValueError("invalid phase2 validation result payload: missing metrics")
        payloads_payload = payload.get("payloads")
        return {
            **dict(payload),
            "metrics": restored_metrics,
            "layers": tuple(
                layer
                if isinstance(layer, Phase2LayerResult)
                else Phase2LayerResult.model_validate(layer)
                for layer in payload.get("layers", ())
                if isinstance(layer, Phase2LayerResult | Mapping)
            ),
            "layer_computations": tuple(
                computation
                if isinstance(computation, Phase2LayerComputation)
                else Phase2LayerComputation.model_validate(computation)
                for computation in payload.get("layer_computations", ())
                if isinstance(computation, Phase2LayerComputation | Mapping)
            ),
            "payloads": (
                payloads_payload
                if isinstance(payloads_payload, Phase2ValidationPayloads)
                else Phase2ValidationPayloads.model_validate(payloads_payload)
                if isinstance(payloads_payload, Mapping)
                else None
            ),
        }


class Phase2LayerComputation(PydanticMappingModel):
    """单个 Phase II validation layer 的 raw metric 计算结果。

    各 ``phase2_validation_layers/layer*.py`` 文件只负责 raw metric 计算，不做
    hard gate pass/fail 判定。rules 层后续读取 ``metrics`` 并生成
    ``Phase2LayerResult``。
    """

    # layer 数字编号，0 到 5。用途：固定 layer 顺序；方向：无好坏方向。
    layer_id: int

    # layer 稳定名称，例如 "selector_profitability"。用途：反序列化和 report
    # 分组；方向：无好坏方向。
    layer_name: str

    # 本层强类型 raw metrics。用途：保存 evaluator 已计算的原始聚合指标；
    # 方向：由具体 layer metrics 字段定义。
    metrics: Phase2LayerMetrics

    # 可选额外中间产物，例如 per-code diagnostics 或 predictability payload。
    # 用途：补充 report 细节；方向：过程数据无直接排序方向。
    extra_payload: Mapping[str, object] | None = None

    @model_validator(mode="before")
    @classmethod
    def _restore_layer_computation_payload(cls, payload: Any) -> Any:
        """Restore layer-specific metrics and extra payloads."""

        if not isinstance(payload, Mapping):
            return payload
        layer_id = int(payload["layer_id"])
        layer_name = str(payload["layer_name"])
        metrics_payload = payload.get("metrics", {})
        extra_payload = payload.get("extra_payload")
        return {
            **dict(payload),
            "layer_id": layer_id,
            "layer_name": layer_name,
            "metrics": _layer_metrics_from_payload(layer_id, layer_name, metrics_payload),
            "extra_payload": (
                _layer_extra_payload_from_payload(layer_name, extra_payload)
                if isinstance(extra_payload, Mapping)
                else None
            ),
        }


def _layer_metrics_from_payload(
    layer_id: int,
    layer_name: str,
    payload: Any,
) -> Phase2LayerMetrics:
    """按 layer 恢复强类型 raw metrics。"""

    if not isinstance(payload, Mapping):
        return payload
    if layer_id == 0 or layer_name == "evaluation_validity":
        from .phase2_validation_layer0_evaluation_validity import (
            Phase2EvaluationValidityMetrics,
        )

        return Phase2EvaluationValidityMetrics.model_validate(payload)
    if layer_id == 1 or layer_name == "selector_profitability":
        from .phase2_validation_layer1_selector_profitability import (
            Phase2SelectorProfitabilityMetrics,
        )

        return Phase2SelectorProfitabilityMetrics.model_validate(payload)
    if layer_id == 2 or layer_name == "baseline_uplift":
        from .phase2_validation_layer2_baseline_uplift import (
            Phase2BaselineUpliftMetrics,
        )

        return Phase2BaselineUpliftMetrics.model_validate(payload)
    if layer_id == 3 or layer_name == "demonstration_consistency":
        from .phase2_validation_layer3_demonstration_consistency import (
            Phase2DemonstrationConsistencyMetrics,
        )

        return Phase2DemonstrationConsistencyMetrics.model_validate(payload)
    if layer_id == 4 or layer_name == "code_usage_collapse":
        from .phase2_validation_layer4_code_usage_collapse import (
            Phase2CodeUsageCollapseMetrics,
        )

        return Phase2CodeUsageCollapseMetrics.model_validate(payload)
    if layer_id == 5 or layer_name == "generalization_stability":
        from .phase2_validation_layer5_generalization_stability import (
            Phase2GeneralizationStabilityMetrics,
        )

        return Phase2GeneralizationStabilityMetrics.model_validate(payload)
    return dict(payload)


def _layer_extra_payload_from_payload(
    layer_name: str,
    payload: Mapping[str, Any],
) -> Mapping[str, object]:
    """恢复 layer computation 中挂载的常见 payload。"""

    from .phase2_validation_layer0_evaluation_validity import (
        Phase2EvaluationValidityPayload,
    )
    from .phase2_validation_layer1_selector_profitability import (
        Phase2SelectorProfitabilityPayload,
    )
    from .phase2_validation_layer2_baseline_uplift import (
        Phase2BaselineUpliftPayload,
    )
    from .phase2_validation_layer3_demonstration_consistency import (
        Phase2DemonstrationConsistencyPayload,
    )
    from .phase2_validation_layer4_code_usage_collapse import (
        Phase2CodeUsageCollapsePayload,
        Phase2PerCodeUsageDiagnostic,
    )
    from .phase2_validation_layer5_generalization_stability import (
        Phase2GeneralizationStabilityPayload,
    )

    restored: dict[str, object] = dict(payload)
    payload_key_builders = {
        "evaluation_validity_payload": Phase2EvaluationValidityPayload,
        "selector_profitability_payload": Phase2SelectorProfitabilityPayload,
        "baseline_uplift_payload": Phase2BaselineUpliftPayload,
        "demonstration_consistency_payload": Phase2DemonstrationConsistencyPayload,
        "code_usage_collapse_payload": Phase2CodeUsageCollapsePayload,
        "generalization_stability_payload": Phase2GeneralizationStabilityPayload,
    }
    for key, builder in payload_key_builders.items():
        value = restored.get(key)
        if isinstance(value, Mapping):
            restored[key] = builder.model_validate(value)

    diagnostics = restored.get("per_code_diagnostics")
    if isinstance(diagnostics, list | tuple):
        restored["per_code_diagnostics"] = tuple(
            Phase2PerCodeUsageDiagnostic.model_validate(item)
            if isinstance(item, Mapping)
            else item
            for item in diagnostics
        )

    return restored


__all__ = [
    "MetricDirection",
    "MetricSeverity",
    "MetricThresholdValue",
    "Phase2LayerComputation",
    "Phase2LayerMetrics",
    "Phase2LayerPayload",
    "Phase2LayerResult",
    "Phase2MetricResult",
    "Phase2ReportCodeCount",
    "Phase2ReportCodeDiagnosticPayloadRow",
    "Phase2ReportCodeUsageDistribution",
    "Phase2ReportCumulativeReturns",
    "Phase2ReportPairProfitabilityPayloadRow",
    "Phase2ValidationMetrics",
    "Phase2ValidationPayloads",
    "Phase2ValidationResult",
]
