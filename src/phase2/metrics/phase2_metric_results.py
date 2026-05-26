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

from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypeAlias


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


@dataclass(frozen=True)
class Phase2MetricResult:
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

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2MetricResult":
        """从 dict 恢复 metric result。"""

        return cls(
            name=str(payload["name"]),
            value=payload.get("value"),
            threshold=str(payload["threshold"]),
            severity=payload["severity"],  # type: ignore[arg-type]
            passed=bool(payload["passed"]),
            layer=str(payload["layer"]),
            message=str(payload.get("message", "")),
            threshold_value=_threshold_value_from_payload(
                payload.get("threshold_value")
            ),
            direction=(
                str(direction)
                if (direction := payload.get("direction")) is not None
                else None
            ),  # type: ignore[arg-type]
            distance_to_threshold=(
                float(distance)
                if (distance := payload.get("distance_to_threshold")) is not None
                else None
            ),
        )


@dataclass(frozen=True)
class Phase2LayerResult:
    """单个 Phase II validation layer 的判定结果。"""

    # layer 数字编号，0-5。用途：固定展示顺序；方向：无好坏方向。
    layer_id: int

    # layer 稳定名称。用途：report 分组和规则追踪；方向：无好坏方向。
    name: str

    # 本层是否通过。用途：checkpoint selector hard gate 聚合；方向：True 更好。
    passed: bool

    # 本层下属 metric 判定结果。用途：报告阈值细节；方向：由每个 metric 决定。
    metrics: tuple[Phase2MetricResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return {
            "layer_id": self.layer_id,
            "name": self.name,
            "passed": self.passed,
            "metrics": [metric.to_dict() for metric in self.metrics],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2LayerResult":
        """从 dict 恢复 layer result。"""

        return cls(
            layer_id=int(payload["layer_id"]),
            name=str(payload["name"]),
            passed=bool(payload["passed"]),
            metrics=tuple(
                Phase2MetricResult.from_dict(metric)
                for metric in payload.get("metrics", ())
            ),
        )


def _threshold_value_from_payload(value: Any) -> MetricThresholdValue:
    """恢复机器可读阈值。"""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, tuple | list):
        if len(value) != 2:
            return None
        return (float(value[0]), float(value[1]))
    return float(value)


def _dataclass_from_payload(
    dataclass_type: type[Any],
    payload: Mapping[str, Any],
) -> Any:
    """用 payload 中匹配 dataclass 字段的键恢复对象。"""

    allowed_fields = {field.name for field in fields(dataclass_type)}
    return dataclass_type(
        **{key: value for key, value in payload.items() if key in allowed_fields}
    )


def _payload_to_dict(value: Any) -> Any:
    """递归序列化 metrics/payload 中的 dataclass 和容器。"""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _payload_to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_payload_to_dict(item) for item in value]
    return value


@dataclass(frozen=True)
class Phase2ValidationMetrics:
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

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2ValidationMetrics":
        """从 dict 恢复 validation 核心指标。"""

        return _dataclass_from_payload(cls, payload)


@dataclass(frozen=True)
class Phase2ReportPairProfitabilityPayloadRow:
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

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase2ReportPairProfitabilityPayloadRow":
        """从 dict 恢复 pair profitability payload row。"""

        return cls(
            morphology=str(payload.get("morphology", "")),
            motif=str(payload.get("motif", "")),
            support=int(payload.get("support", 0)),
            selector_mean_return=float(payload.get("selector_mean_return", 0.0)),
            kl_mean_return=float(payload.get("kl_mean_return", 0.0)),
            random_mean_return=float(payload.get("random_mean_return", 0.0)),
            mean_advantage_vs_kl=float(payload.get("mean_advantage_vs_kl", 0.0)),
            mean_advantage_vs_random=float(
                payload.get("mean_advantage_vs_random", 0.0)
            ),
            win_rate=float(payload.get("win_rate", 0.0)),
            fee_drag_ratio=float(payload.get("fee_drag_ratio", 0.0)),
            dominant_selected_code=(
                int(value)
                if (value := payload.get("dominant_selected_code")) is not None
                else None
            ),
            dominant_selected_code_ratio=float(
                payload.get("dominant_selected_code_ratio", 0.0)
            ),
        )


@dataclass(frozen=True)
class Phase2ReportCodeDiagnosticPayloadRow:
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

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase2ReportCodeDiagnosticPayloadRow":
        """从 dict 恢复 code diagnostic payload row。"""

        return cls(
            code_id=int(payload.get("code_id", 0)),
            selector_support=int(payload.get("selector_support", 0)),
            selector_usage_ratio=float(payload.get("selector_usage_ratio", 0.0)),
            kl_support=int(payload.get("kl_support", 0)),
            kl_usage_ratio=float(payload.get("kl_usage_ratio", 0.0)),
            usage_delta=float(payload.get("usage_delta", 0.0)),
            selector_mean_return=float(payload.get("selector_mean_return", 0.0)),
            kl_mean_return=float(payload.get("kl_mean_return", 0.0)),
            uplift_vs_kl=float(payload.get("uplift_vs_kl", 0.0)),
            selector_win_rate=float(payload.get("selector_win_rate", 0.0)),
            selector_fee_drag_ratio=float(
                payload.get("selector_fee_drag_ratio", 0.0)
            ),
            selector_turnover=float(payload.get("selector_turnover", 0.0)),
            dominant_morphology=_optional_str(payload.get("dominant_morphology")),
            dominant_morphology_ratio=float(
                payload.get("dominant_morphology_ratio", 0.0)
            ),
            dominant_motif=_optional_str(payload.get("dominant_motif")),
            dominant_motif_ratio=float(payload.get("dominant_motif_ratio", 0.0)),
            dominant_pair=_optional_str(payload.get("dominant_pair")),
            dominant_pair_ratio=float(payload.get("dominant_pair_ratio", 0.0)),
            mean_q_margin=float(payload.get("mean_q_margin", 0.0)),
            low_confidence_ratio=float(payload.get("low_confidence_ratio", 0.0)),
            profitable_deviation_count=int(
                payload.get("profitable_deviation_count", 0)
            ),
            unprofitable_deviation_count=int(
                payload.get("unprofitable_deviation_count", 0)
            ),
            unprofitable_deviation_rate=float(
                payload.get("unprofitable_deviation_rate", 0.0)
            ),
            status=str(payload.get("status", "warn")),
            risk_reason=str(payload.get("risk_reason", "")),
        )


@dataclass(frozen=True)
class Phase2ReportCodeCount:
    """Report payload 中某个 code 的样本数。"""

    code_id: int
    count: int

    def to_dict(self) -> dict[str, int]:
        """序列化为普通 dict。"""

        return {"code_id": self.code_id, "count": self.count}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2ReportCodeCount":
        """从 dict 恢复 code count。"""

        return cls(
            code_id=int(payload.get("code_id", 0)),
            count=int(payload.get("count", 0)),
        )


@dataclass(frozen=True)
class Phase2ReportCodeUsageDistribution:
    """Report payload 中 selector 和 assigned-label 的 code 使用分布。"""

    selector: tuple[Phase2ReportCodeCount, ...] = ()
    kl: tuple[Phase2ReportCodeCount, ...] = ()

    def __post_init__(self) -> None:
        """标准化 code count 行。"""

        object.__setattr__(
            self,
            "selector",
            tuple(_code_count_from_value(item) for item in (self.selector or ())),
        )
        object.__setattr__(
            self,
            "kl",
            tuple(_code_count_from_value(item) for item in (self.kl or ())),
        )

    def to_dict(self) -> dict[str, list[dict[str, int]]]:
        """序列化为普通 dict。"""

        return {
            "selector": [item.to_dict() for item in self.selector],
            "kl": [item.to_dict() for item in self.kl],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Phase2ReportCodeUsageDistribution":
        """从 dict 恢复 code usage distribution。"""

        return cls(
            selector=tuple(
                Phase2ReportCodeCount.from_dict(item)
                for item in (payload.get("selector", ()) or ())
                if isinstance(item, Mapping)
            ),
            kl=tuple(
                Phase2ReportCodeCount.from_dict(item)
                for item in (payload.get("kl", ()) or ())
                if isinstance(item, Mapping)
            ),
        )


@dataclass(frozen=True)
class Phase2ReportCumulativeReturns:
    """Report payload 中各 baseline 的累计收益曲线。"""

    selector: tuple[float, ...] = ()
    kl: tuple[float, ...] = ()
    random: tuple[float, ...] = ()
    oracle: tuple[float, ...] = ()
    hold: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """标准化累计收益序列。"""

        for name in ("selector", "kl", "random", "oracle", "hold"):
            object.__setattr__(
                self,
                name,
                tuple(float(value) for value in getattr(self, name)),
            )

    def to_dict(self) -> dict[str, list[float]]:
        """序列化为普通 dict。"""

        return {
            "selector": list(self.selector),
            "kl": list(self.kl),
            "random": list(self.random),
            "oracle": list(self.oracle),
            "hold": list(self.hold),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2ReportCumulativeReturns":
        """从 dict 恢复累计收益曲线。"""

        return cls(
            selector=_float_tuple(payload.get("selector", ())),
            kl=_float_tuple(payload.get("kl", ())),
            random=_float_tuple(payload.get("random", ())),
            oracle=_float_tuple(payload.get("oracle", ())),
            hold=_float_tuple(payload.get("hold", ())),
        )


@dataclass(frozen=True)
class Phase2ReportPayload:
    """HTML/JSON report 复用的强类型聚合 payload。"""

    per_code_profitability_comparison: tuple[
        "Phase2PerCodeUsageDiagnostic",
        ...,
    ] = ()
    selector_pair_profitability_matrix: tuple[
        Phase2ReportPairProfitabilityPayloadRow,
        ...,
    ] = ()
    code_diagnostics: tuple[Phase2ReportCodeDiagnosticPayloadRow, ...] = ()
    codebook_usage_distribution: Phase2ReportCodeUsageDistribution = field(
        default_factory=Phase2ReportCodeUsageDistribution
    )
    oracle_label_cumulative_returns: Phase2ReportCumulativeReturns = field(
        default_factory=Phase2ReportCumulativeReturns
    )

    def __post_init__(self) -> None:
        """标准化 report payload 的嵌套行类型。"""

        from .phase2_validation_layer4_code_usage_collapse import (
            Phase2PerCodeUsageDiagnostic,
        )

        object.__setattr__(
            self,
            "per_code_profitability_comparison",
            tuple(
                item
                if isinstance(item, Phase2PerCodeUsageDiagnostic)
                else Phase2PerCodeUsageDiagnostic.from_dict(item)
                for item in (self.per_code_profitability_comparison or ())
            ),
        )
        object.__setattr__(
            self,
            "selector_pair_profitability_matrix",
            tuple(
                item
                if isinstance(item, Phase2ReportPairProfitabilityPayloadRow)
                else Phase2ReportPairProfitabilityPayloadRow.from_dict(item)
                for item in (self.selector_pair_profitability_matrix or ())
            ),
        )
        object.__setattr__(
            self,
            "code_diagnostics",
            tuple(
                item
                if isinstance(item, Phase2ReportCodeDiagnosticPayloadRow)
                else Phase2ReportCodeDiagnosticPayloadRow.from_dict(item)
                for item in (self.code_diagnostics or ())
            ),
        )
        if isinstance(self.codebook_usage_distribution, Mapping):
            object.__setattr__(
                self,
                "codebook_usage_distribution",
                Phase2ReportCodeUsageDistribution.from_dict(
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
                Phase2ReportCumulativeReturns.from_dict(
                    self.oracle_label_cumulative_returns
                ),
            )
        elif self.oracle_label_cumulative_returns is None:
            object.__setattr__(
                self,
                "oracle_label_cumulative_returns",
                Phase2ReportCumulativeReturns(),
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为对外兼容的 report_payload dict。"""

        return {
            "per_code_profitability_comparison": [
                item.to_dict() for item in self.per_code_profitability_comparison
            ],
            "selector_pair_profitability_matrix": [
                item.to_dict() for item in self.selector_pair_profitability_matrix
            ],
            "code_diagnostics": [item.to_dict() for item in self.code_diagnostics],
            "codebook_usage_distribution": self.codebook_usage_distribution.to_dict(),
            "oracle_label_cumulative_returns": (
                self.oracle_label_cumulative_returns.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2ReportPayload":
        """从 dict 恢复强类型 report payload。"""

        from .phase2_validation_layer4_code_usage_collapse import (
            Phase2PerCodeUsageDiagnostic,
        )

        cumulative_payload = (
            payload.get("oracle_label_cumulative_returns")
            or payload.get("cumulative_return_curves")
            or {}
        )
        return cls(
            per_code_profitability_comparison=tuple(
                Phase2PerCodeUsageDiagnostic.from_dict(item)
                for item in (
                    payload.get("per_code_profitability_comparison", ()) or ()
                )
                if isinstance(item, Mapping)
            ),
            selector_pair_profitability_matrix=tuple(
                Phase2ReportPairProfitabilityPayloadRow.from_dict(item)
                for item in (
                    payload.get("selector_pair_profitability_matrix", ()) or ()
                )
                if isinstance(item, Mapping)
            ),
            code_diagnostics=tuple(
                Phase2ReportCodeDiagnosticPayloadRow.from_dict(item)
                for item in (payload.get("code_diagnostics", ()) or ())
                if isinstance(item, Mapping)
            ),
            codebook_usage_distribution=(
                Phase2ReportCodeUsageDistribution.from_dict(distribution)
                if isinstance(
                    distribution := payload.get("codebook_usage_distribution"),
                    Mapping,
                )
                else Phase2ReportCodeUsageDistribution()
            ),
            oracle_label_cumulative_returns=(
                Phase2ReportCumulativeReturns.from_dict(cumulative_payload)
                if isinstance(cumulative_payload, Mapping)
                else Phase2ReportCumulativeReturns()
            ),
        )


def _optional_str(value: Any) -> str | None:
    """把空值标准化为 None，其他值转为 str。"""

    return None if value in (None, "") else str(value)


def _float_tuple(values: Any) -> tuple[float, ...]:
    """把序列值标准化为 float tuple。"""

    if isinstance(values, str | bytes):
        return ()
    try:
        return tuple(float(value) for value in values)
    except TypeError:
        return ()


def _code_count_from_value(value: Any) -> Phase2ReportCodeCount:
    """把 dict 或 dataclass-like code count 转成强类型对象。"""

    if isinstance(value, Phase2ReportCodeCount):
        return value
    if isinstance(value, Mapping):
        return Phase2ReportCodeCount.from_dict(value)
    return Phase2ReportCodeCount(
        code_id=int(getattr(value, "code_id")),
        count=int(getattr(value, "count")),
    )


@dataclass(frozen=True)
class Phase2ValidationPayloads:
    """Phase II validation/report 需要复用的聚合 payload。

    本对象只保存 evaluator 已经聚合好的过程数据和诊断数据，不保存完整逐样本
    ``selection_trace``。
    """

    # Layer 0 评估可信度过程数据。用途：审计 split、epoch、样本数和失败计数；
    # 方向：过程数据本身无排序方向，由 Layer 0 metrics 转换为好坏判定。
    evaluation_validity_payload: Phase2EvaluationValidityPayload | None = None

    # Layer 1 selector 收益过程数据。用途：保存收益、gross return、fee、turnover
    # 序列以便复查聚合指标；方向：过程数据无直接方向。
    selector_profitability_payload: Phase2SelectorProfitabilityPayload | None = None

    # Layer 2 baseline 对比过程数据。用途：保存 selector/assigned/random/oracle
    # return 序列；方向：过程数据无直接方向。
    baseline_uplift_payload: Phase2BaselineUpliftPayload | None = None

    # Layer 3 demonstration consistency 过程数据。用途：保存 selected/assigned code、
    # return 和 Q value 序列；方向：过程数据无直接方向。
    demonstration_consistency_payload: Phase2DemonstrationConsistencyPayload | None = None

    # Layer 4 code usage 过程数据。用途：保存 selected code 分布和 per-code 诊断；
    # 方向：过程数据无直接方向。
    code_usage_collapse_payload: Phase2CodeUsageCollapsePayload | None = None

    # Layer 5 泛化稳定性过程数据。用途：保存 score/churn/Q scale 历史和 probe
    # payload；方向：过程数据无直接方向。
    generalization_stability_payload: Phase2GeneralizationStabilityPayload | None = None

    # HTML/report 复用的强类型聚合 payload。用途：避免报表重新执行 evaluator；
    # 方向：展示数据，无直接排序方向。
    report_payload: Phase2ReportPayload | None = None

    def __post_init__(self) -> None:
        """兼容旧 dict 输入，并在对象边界恢复强类型 report payload。"""

        if isinstance(self.report_payload, Mapping):
            object.__setattr__(
                self,
                "report_payload",
                Phase2ReportPayload.from_dict(self.report_payload),
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return {
            "evaluation_validity_payload": _payload_to_dict(
                self.evaluation_validity_payload
            ),
            "selector_profitability_payload": _payload_to_dict(
                self.selector_profitability_payload
            ),
            "baseline_uplift_payload": _payload_to_dict(self.baseline_uplift_payload),
            "demonstration_consistency_payload": _payload_to_dict(
                self.demonstration_consistency_payload
            ),
            "code_usage_collapse_payload": _payload_to_dict(
                self.code_usage_collapse_payload
            ),
            "generalization_stability_payload": _payload_to_dict(
                self.generalization_stability_payload
            ),
            "report_payload": _payload_to_dict(self.report_payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2ValidationPayloads":
        """从 dict 恢复 validation/report payload 聚合。"""

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
        )
        from .phase2_validation_layer5_generalization_stability import (
            Phase2GeneralizationStabilityPayload,
        )

        evaluation_payload = payload.get("evaluation_validity_payload")
        selector_payload = payload.get("selector_profitability_payload")
        baseline_payload = payload.get("baseline_uplift_payload")
        demonstration_payload = payload.get("demonstration_consistency_payload")
        code_usage_payload = payload.get("code_usage_collapse_payload")
        stability_payload = payload.get("generalization_stability_payload")
        report_payload = payload.get("report_payload")
        return cls(
            evaluation_validity_payload=(
                Phase2EvaluationValidityPayload.from_dict(evaluation_payload)
                if isinstance(evaluation_payload, Mapping)
                else None
            ),
            selector_profitability_payload=(
                Phase2SelectorProfitabilityPayload.from_dict(selector_payload)
                if isinstance(selector_payload, Mapping)
                else None
            ),
            baseline_uplift_payload=(
                Phase2BaselineUpliftPayload.from_dict(baseline_payload)
                if isinstance(baseline_payload, Mapping)
                else None
            ),
            demonstration_consistency_payload=(
                Phase2DemonstrationConsistencyPayload.from_dict(demonstration_payload)
                if isinstance(demonstration_payload, Mapping)
                else None
            ),
            code_usage_collapse_payload=(
                Phase2CodeUsageCollapsePayload.from_dict(code_usage_payload)
                if isinstance(code_usage_payload, Mapping)
                else None
            ),
            generalization_stability_payload=(
                Phase2GeneralizationStabilityPayload.from_dict(stability_payload)
                if isinstance(stability_payload, Mapping)
                else None
            ),
            report_payload=(
                Phase2ReportPayload.from_dict(report_payload)
                if isinstance(report_payload, Mapping)
                else None
            ),
        )


@dataclass(frozen=True)
class Phase2ValidationResult:
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
    layer_computations: tuple[Phase2LayerComputation, ...] = ()

    # 报表和诊断卡片复用的聚合 payload。用途：HTML/JSON report 展示；方向：
    # 展示数据，无直接排序方向。
    payloads: Phase2ValidationPayloads | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return {
            "metrics": self.metrics.to_dict(),
            "layers": [layer.to_dict() for layer in self.layers],
            "layer_computations": [
                computation.to_dict() for computation in self.layer_computations
            ],
            "payloads": (
                self.payloads.to_dict() if self.payloads is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2ValidationResult":
        """从 dict 恢复 validation result。"""

        metrics_payload = payload.get("metrics")
        if not isinstance(metrics_payload, Mapping):
            raise ValueError("invalid phase2 validation result payload: missing metrics")
        payloads_payload = payload.get("payloads")
        return cls(
            metrics=Phase2ValidationMetrics.from_dict(metrics_payload),
            layers=tuple(
                Phase2LayerResult.from_dict(layer)
                for layer in payload.get("layers", ())
                if isinstance(layer, Mapping)
            ),
            layer_computations=tuple(
                Phase2LayerComputation.from_dict(computation)
                for computation in payload.get("layer_computations", ())
                if isinstance(computation, Mapping)
            ),
            payloads=(
                Phase2ValidationPayloads.from_dict(payloads_payload)
                if isinstance(payloads_payload, Mapping)
                else None
            ),
        )


@dataclass(frozen=True)
class Phase2LayerComputation:
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

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return {
            "layer_id": self.layer_id,
            "layer_name": self.layer_name,
            "metrics": _payload_to_dict(self.metrics),
            "extra_payload": _payload_to_dict(self.extra_payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2LayerComputation":
        """从 dict 恢复单层 raw metric 计算结果。"""

        layer_id = int(payload["layer_id"])
        layer_name = str(payload["layer_name"])
        metrics_payload = payload.get("metrics", {})
        extra_payload = payload.get("extra_payload")
        return cls(
            layer_id=layer_id,
            layer_name=layer_name,
            metrics=_layer_metrics_from_dict(layer_id, layer_name, metrics_payload),
            extra_payload=(
                _layer_extra_payload_from_dict(layer_name, extra_payload)
                if isinstance(extra_payload, Mapping)
                else None
            ),
        )


def _layer_metrics_from_dict(
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

        return Phase2EvaluationValidityMetrics.from_dict(payload)
    if layer_id == 1 or layer_name == "selector_profitability":
        from .phase2_validation_layer1_selector_profitability import (
            Phase2SelectorProfitabilityMetrics,
        )

        return Phase2SelectorProfitabilityMetrics.from_dict(payload)
    if layer_id == 2 or layer_name == "baseline_uplift":
        from .phase2_validation_layer2_baseline_uplift import (
            Phase2BaselineUpliftMetrics,
        )

        return Phase2BaselineUpliftMetrics.from_dict(payload)
    if layer_id == 3 or layer_name == "demonstration_consistency":
        from .phase2_validation_layer3_demonstration_consistency import (
            Phase2DemonstrationConsistencyMetrics,
        )

        return Phase2DemonstrationConsistencyMetrics.from_dict(payload)
    if layer_id == 4 or layer_name == "code_usage_collapse":
        from .phase2_validation_layer4_code_usage_collapse import (
            Phase2CodeUsageCollapseMetrics,
        )

        return Phase2CodeUsageCollapseMetrics.from_dict(payload)
    if layer_id == 5 or layer_name == "generalization_stability":
        from .phase2_validation_layer5_generalization_stability import (
            Phase2GeneralizationStabilityMetrics,
        )

        return Phase2GeneralizationStabilityMetrics.from_dict(payload)
    return dict(payload)


def _layer_extra_payload_from_dict(
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
            restored[key] = builder.from_dict(value)

    diagnostics = restored.get("per_code_diagnostics")
    if isinstance(diagnostics, list | tuple):
        restored["per_code_diagnostics"] = tuple(
            Phase2PerCodeUsageDiagnostic.from_dict(item)
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
    "Phase2ReportPayload",
    "Phase2ValidationMetrics",
    "Phase2ValidationPayloads",
    "Phase2ValidationResult",
]
