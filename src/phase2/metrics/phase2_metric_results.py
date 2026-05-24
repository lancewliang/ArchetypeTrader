"""Phase II metrics result payload 骨架。

文件功能说明:
    本文件定义 Phase II evaluator 已经计算完成的指标结果对象。它们是
    checkpoint selector、artifact store 和 report 之间共享的结果 payload。

设计边界:
    - 只承载 evaluator 产出的 metrics 和 diagnostics；
    - 不计算指标、不读取模型、不访问训练数据；
    - 不负责 checkpoint 模型权重保存；
    - 不判断 best checkpoint，也不应用 hard gate 或 tie-breaker。

使用场景:
    ``Phase2Evaluator`` 评估 validation/test split 后生成这些对象；
    ``Phase2ArtifactStore`` 负责保存/读取它们；
    ``Phase2CheckpointSelector`` 和 report 只消费其中的稳定字段。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


MetricSeverity = Literal["pass", "warn", "fail", "skip"]
MetricDirection = Literal["greater_is_better", "less_is_better", "between", "equal"]
MetricThresholdValue = float | tuple[float, float] | bool | None


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


@dataclass(frozen=True)
class Phase2ValidationResult:
    """Phase II validation 结果摘要。

    功能说明:
        保存 evaluator 已经计算好的 selection metrics 和诊断信息，作为
        validation result、report 和 checkpoint selector 的共享输入。

    设计边界:
        本类只承载结果，不负责计算指标、应用阈值或决定 best checkpoint。
        ``metrics`` 应保存可排序、可报告的稳定字段；``diagnostics`` 保存解释性
        信息，例如 code usage、label consistency 或 reward distribution。

    使用场景:
        ``Phase2Evaluator`` 评估某个 epoch 后返回该对象，再由 artifact store
        保存为 Phase II validation result payload。
    """

    # evaluator 产出的核心指标，例如 mean_return、risk 和 consistency。
    metrics: Phase2ValidationMetrics
