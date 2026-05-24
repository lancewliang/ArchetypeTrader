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

from dataclasses import asdict, dataclass, fields
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

    name: str
    value: int | float | str | bool | None
    threshold: str
    severity: MetricSeverity
    passed: bool
    layer: str
    message: str = ""
    threshold_value: MetricThresholdValue = None
    direction: MetricDirection | None = None
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

    layer_id: int
    name: str
    passed: bool
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

    # selector greedy action 的平均 horizon return。
    mean_return: float

    # selector greedy action 的 return 中位数。
    median_return: float

    # 类 Sharpe 风险调整收益指标。
    sharpe_like: float

    # horizon return 大于 0 的比例。
    win_rate: float

    # 平均换手率或行为强度指标。
    mean_turnover: float

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase2ValidationMetrics":
        """从 dict 恢复 validation 核心指标。"""

        return _dataclass_from_payload(cls, payload)


@dataclass(frozen=True)
class Phase2ValidationPayloads:
    """Phase II validation/report 需要复用的聚合 payload。

    本对象只保存 evaluator 已经聚合好的过程数据和诊断数据，不保存完整逐样本
    ``selection_trace``。
    """

    evaluation_validity_payload: Phase2EvaluationValidityPayload | None = None
    selector_profitability_payload: Phase2SelectorProfitabilityPayload | None = None
    baseline_uplift_payload: Phase2BaselineUpliftPayload | None = None
    demonstration_consistency_payload: Phase2DemonstrationConsistencyPayload | None = None
    code_usage_collapse_payload: Phase2CodeUsageCollapsePayload | None = None
    generalization_stability_payload: Phase2GeneralizationStabilityPayload | None = None
    report_payload: Mapping[str, object] | None = None

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
                dict(report_payload) if isinstance(report_payload, Mapping) else None
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

    # checkpoint selector 直接消费的核心摘要指标，例如 mean_return、risk。
    metrics: Phase2ValidationMetrics

    # hard-gate/reference layer 判定结果。
    layers: tuple[Phase2LayerResult, ...] = ()

    # Layer 0-5 强类型 raw metrics 和本层中间 payload。
    layer_computations: tuple[Phase2LayerComputation, ...] = ()

    # 报表和诊断卡片复用的聚合 payload。
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

    # layer 数字编号，0 到 5。
    layer_id: int

    # layer 稳定名称，例如 "selector_profitability"。
    layer_name: str

    # 本层强类型 raw metrics。
    metrics: Phase2LayerMetrics

    # 可选额外中间产物，例如 per-code diagnostics 或 predictability payload。
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
    "Phase2ValidationMetrics",
    "Phase2ValidationPayloads",
    "Phase2ValidationResult",
]
