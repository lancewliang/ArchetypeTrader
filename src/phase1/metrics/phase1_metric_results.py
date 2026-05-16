"""Phase I codebook validation 判定结果 schema。

本文件定义 rules 层输出给 checkpoint selector、report 和 checkpoint payload 的
稳定结果结构。它不计算 raw metrics，也不持有 hard gate 阈值；raw metrics 来自
``phase1_validation_data_schema.py``，阈值判定由 ``phase1_validation_rules.py``
完成。

使用场景:
    1. 单个 hard gate 指标被包装成 ``Phase1MetricResult``；
    2. 同一 validation layer 的指标结果聚合为 ``Phase1LayerResult``；
    3. 一个 checkpoint 的五层判定、综合分、诊断表和 tie-breaker 指标聚合为
       ``Phase1ValidationResult``；
    4. checkpoint/report 通过 ``to_dict()`` / ``from_dict()`` 做可审计落盘。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from .phase1_validation_data_schema import (
    Phase1CodeDiagnostic,
    Phase1TieBreakerMetrics,
    Phase1ValidationMetrics,
)
from .phase1_validation_behavior_quality import Phase1BehaviorQualityPayload
from .phase1_validation_label_predictability import Phase1LabelPredictabilityPayload
from .phase1_validation_oracle_profitability import Phase1OracleProfitabilityPayload
from .phase1_validation_teacher_quality import Phase1TeacherQualityPayload
from .phase1_validation_vq_internal import Phase1VQInternalPayload
from .phase1_validation_score import (
    Phase1ValidationScore,
    Phase1ValidationScoreLike,
)


MetricSeverity = Literal["pass", "warn", "fail", "skip"]
"""单个 metric 的判定严重级别。

含义:
    - ``pass``: 指标通过；
    - ``warn``: 指标触发警戒但不阻断 checkpoint；
    - ``fail``: hard gate 失败，checkpoint 不能进入候选；
    - ``skip``: 缺少必要输入无法计算。hard gate 指标默认 skip-as-fail。
"""


@dataclass(frozen=True)
class Phase1MetricResult:
    """单个 validation metric 的判定结果。

    功能说明:
        记录 metric name、实际值、人类可读阈值、严重级别、是否通过 hard gate
        以及 report 展示用说明。

    使用场景:
        由 ``phase1_validation_rules.py`` 中的每条 hard gate 规则创建，直接供
        layer result、report 表格和失败摘要消费。
    """

    # 稳定 snake_case 指标名，例如 "validation_action_accuracy"。
    name: str

    # 指标实际值。缺失或不可计算时允许为 None。
    value: int | float | str | bool | None

    # 人类可读阈值表达式，例如 ">= 0.85" 或 "> 0"。
    threshold: str

    # 判定严重级别，取值为 pass/warn/fail/skip。
    severity: MetricSeverity

    # 是否满足 hard gate。warn 可以 passed=True；skip 对 hard gate 默认 passed=False。
    passed: bool

    # 所属 layer 稳定名称，例如 "teacher_quality"。
    layer: str

    # report 展示用解释文本。
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供 checkpoint/report 落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1MetricResult":
        """从 dict 恢复 metric result。"""

        return cls(
            name=str(payload["name"]),
            value=payload.get("value"),
            threshold=str(payload["threshold"]),
            severity=payload["severity"],  # type: ignore[arg-type]
            passed=bool(payload["passed"]),
            layer=str(payload["layer"]),
            message=str(payload.get("message", "")),
        )


@dataclass(frozen=True)
class Phase1LayerResult:
    """单个 validation layer 的判定结果。

    功能说明:
        聚合一层内的多个 ``Phase1MetricResult``，并给出该层是否整体通过。

    使用场景:
        五个 rule 函数分别返回一个 layer result；最终由
        ``aggregate_validation_result()`` 聚合成 checkpoint 级 validation result。
    """

    # layer 数字编号，0 到 4。
    layer_id: int

    # layer 稳定名称，例如 "vq_internal"。
    name: str

    # 该层所有 hard gate 是否通过。
    passed: bool

    # 该层所有 metric 判定结果。
    metrics: tuple[Phase1MetricResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供 checkpoint/report 落盘。"""

        return {
            "layer_id": self.layer_id,
            "name": self.name,
            "passed": self.passed,
            "metrics": [metric.to_dict() for metric in self.metrics],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1LayerResult":
        """从 dict 恢复 layer result。"""

        return cls(
            layer_id=int(payload["layer_id"]),
            name=str(payload["name"]),
            passed=bool(payload["passed"]),
            metrics=tuple(
                Phase1MetricResult.from_dict(metric)
                for metric in payload.get("metrics", ())
            ),
        )


@dataclass(frozen=True)
class Phase1ValidationResult:
    """单个 checkpoint 的完整 Phase I validation 结果。

    功能说明:
        汇总五层 hard gate 判定、综合评分、失败层列表、强类型 raw metrics、
        code-level diagnostics、drift diagnostics 和 tie-breaker 指标。

    使用场景:
        作为 checkpoint selector 的主要输入，也作为 report 的核心 payload。
        ``to_flat_dict()`` 可生成 selector 快速读取的 top-level scalar。
    """

    # checkpoint 稳定 ID 或文件名。
    checkpoint_id: str

    # 所属训练阶段，例如 "train" 或 "validation"。
    stage: str

    # checkpoint 对应 epoch。
    epoch: int

    # 五层 hard gate 是否全部通过。
    passed: bool

    # 综合评分对象。只有 passed=True 时通常才有值；失败 checkpoint 推荐为 None。
    score: Phase1ValidationScoreLike

    # 失败 layer 名称列表。
    failed_layers: tuple[str, ...]

    # 五层判定结果。
    layers: tuple[Phase1LayerResult, ...]

    # 五层强类型 raw metrics。
    metrics: Phase1ValidationMetrics

    # code-level 诊断表。
    code_diagnostics: tuple[Phase1CodeDiagnostic, ...]

    # 横向 drift 诊断。key 为稳定诊断名，value 为 metric result。
    drift_diagnostics: Mapping[str, Phase1MetricResult]

    # score 接近时的决胜指标。
    tie_breaker_metrics: Phase1TieBreakerMetrics

    # 第零层 teacher quality 中间 payload，用于审计 DP/flat return 明细。
    teacher_quality_payload: Phase1TeacherQualityPayload | None = None

    # 第一层 VQ 内部中间 payload，用于 report 展示 code distribution 等非 gate 字段。
    vq_internal_payload: Phase1VQInternalPayload | None = None

    # 第二层 behavior quality 中间 payload，用于审计 morphology/motif 标签。
    behavior_quality_payload: Phase1BehaviorQualityPayload | None = None

    # 第三层 oracle profitability 中间 payload，用于审计收益曲线和 per-code 盈利。
    oracle_profitability_payload: Phase1OracleProfitabilityPayload | None = None

    # 第四层 label predictability 中间 payload，用于审计 probe 诊断。
    label_predictability_payload: Phase1LabelPredictabilityPayload | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为 checkpoint/report 可保存的嵌套 dict。"""

        return {
            "checkpoint_id": self.checkpoint_id,
            "stage": self.stage,
            "epoch": self.epoch,
            "passed": self.passed,
            "score": (
                self.score.to_dict()
                if isinstance(self.score, Phase1ValidationScore)
                else self.score
            ),
            "failed_layers": list(self.failed_layers),
            "layers": [layer.to_dict() for layer in self.layers],
            "metrics": self.metrics.to_dict(),
            "code_diagnostics": [
                diagnostic.to_dict() for diagnostic in self.code_diagnostics
            ],
            "drift_diagnostics": {
                name: result.to_dict()
                for name, result in self.drift_diagnostics.items()
            },
            "tie_breaker_metrics": self.tie_breaker_metrics.to_dict(),
            "teacher_quality_payload": (
                self.teacher_quality_payload.to_dict()
                if self.teacher_quality_payload is not None
                else None
            ),
            "vq_internal_payload": (
                self.vq_internal_payload.to_dict()
                if self.vq_internal_payload is not None
                else None
            ),
            "behavior_quality_payload": (
                self.behavior_quality_payload.to_dict()
                if self.behavior_quality_payload is not None
                else None
            ),
            "oracle_profitability_payload": (
                self.oracle_profitability_payload.to_dict()
                if self.oracle_profitability_payload is not None
                else None
            ),
            "label_predictability_payload": (
                self.label_predictability_payload.to_dict()
                if self.label_predictability_payload is not None
                else None
            ),
        }

    def to_flat_dict(self) -> dict[str, int | float | bool | None]:
        """生成 checkpoint selector 快速读取的扁平视图。

        使用场景:
            selector 不需要理解每个 metric 的计算细节，只读取通过状态、score、
            失败层数量和 tie-breaker 字段即可。
        """

        payload: dict[str, int | float | bool | None] = {
            "validation.passed": self.passed,
            "validation.score": self.score.total_score,
            "validation.failed_layer_count": len(self.failed_layers),
        }
        for layer in self.layers:
            payload[f"validation.layer{layer.layer_id}.{layer.name}.passed"] = layer.passed
        for key, value in self.tie_breaker_metrics.to_dict().items():
            payload[f"validation.tie_breaker.{key}"] = value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1ValidationResult":
        """从 checkpoint/report payload 恢复 validation result。"""

        return cls(
            checkpoint_id=str(payload["checkpoint_id"]),
            stage=str(payload["stage"]),
            epoch=int(payload["epoch"]),
            passed=bool(payload["passed"]),
            score=_phase1_validation_score_from_payload(payload.get("score")),
            failed_layers=tuple(str(name) for name in payload.get("failed_layers", ())),
            layers=tuple(
                Phase1LayerResult.from_dict(layer)
                for layer in payload.get("layers", ())
            ),
            metrics=Phase1ValidationMetrics.from_dict(payload["metrics"]),
            code_diagnostics=tuple(
                Phase1CodeDiagnostic.from_dict(diagnostic)
                for diagnostic in payload.get("code_diagnostics", ())
            ),
            drift_diagnostics={
                str(name): Phase1MetricResult.from_dict(result)
                for name, result in payload.get("drift_diagnostics", {}).items()
            },
            tie_breaker_metrics=Phase1TieBreakerMetrics.from_dict(
                payload["tie_breaker_metrics"]
            ),
            teacher_quality_payload=(
                Phase1TeacherQualityPayload.from_dict(teacher_payload)
                if (teacher_payload := payload.get("teacher_quality_payload"))
                is not None
                else None
            ),
            vq_internal_payload=(
                Phase1VQInternalPayload.from_dict(vq_payload)
                if (vq_payload := payload.get("vq_internal_payload")) is not None
                else None
            ),
            behavior_quality_payload=(
                Phase1BehaviorQualityPayload.from_dict(behavior_payload)
                if (behavior_payload := payload.get("behavior_quality_payload"))
                is not None
                else None
            ),
            oracle_profitability_payload=(
                Phase1OracleProfitabilityPayload.from_dict(oracle_payload)
                if (oracle_payload := payload.get("oracle_profitability_payload"))
                is not None
                else None
            ),
            label_predictability_payload=(
                Phase1LabelPredictabilityPayload.from_dict(label_payload)
                if (label_payload := payload.get("label_predictability_payload"))
                is not None
                else None
            ),
        )


__all__ = [
    "MetricSeverity",
    "Phase1LayerResult",
    "Phase1MetricResult",
    "Phase1ValidationResult",
]


def _phase1_validation_score_from_payload(
    payload: Phase1ValidationScoreLike | Mapping[str, Any],
) -> Phase1ValidationScore | None:
    """恢复新 score 对象，同时兼容历史 float payload。"""

    if payload is None:
        return None
    if isinstance(payload, Phase1ValidationScore):
        return payload
    if isinstance(payload, Mapping):
        return Phase1ValidationScore.from_dict(payload)
    return Phase1ValidationScore.from_float(float(payload))
