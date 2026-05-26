"""Phase I codebook validation 判定结果 schema。

本文件定义 rules 层输出给 checkpoint selector、report 和 checkpoint payload 的
稳定结果结构。它不计算 raw metrics，也不持有 hard gate 阈值；raw metrics 来自
``phase1_validation_data_schema.py``，阈值判定由 ``phase1_validation_rules.py``
完成。

使用场景:
    1. 单个 hard gate 指标被包装成 ``Phase1MetricResult``；
    2. 同一 validation layer 的指标结果聚合为 ``Phase1LayerResult``；
    3. 一个 checkpoint 的 hard-gate/reference 判定、综合分、诊断表和 tie-breaker 指标聚合为
       ``Phase1ValidationResult``；
    4. checkpoint/report 通过 ``to_dict()`` / ``from_dict()`` 做可审计落盘。
"""

from __future__ import annotations

from typing import  Literal, Mapping
from src.utils import PydanticBaseModel

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
)


MetricSeverity = Literal["pass", "warn", "fail", "skip"]
"""单个 metric 的判定严重级别。

含义:
    - ``pass``: 指标通过；
    - ``warn``: 指标触发警戒但不阻断 checkpoint；
    - ``fail``: hard gate 失败，checkpoint 不能进入候选；
    - ``skip``: 缺少必要输入无法计算。hard gate 指标默认 skip-as-fail。
"""

MetricDirection = Literal["greater_is_better", "less_is_better", "between"]
"""单个 metric 的阈值方向。

含义:
    - ``greater_is_better``: 指标值越大越安全，margin = value - threshold；
    - ``less_is_better``: 指标值越小越安全，margin = threshold - value；
    - ``between``: 指标值应落在闭区间，margin = min(value - lower, upper - value)。
"""

MetricThresholdValue = float | tuple[float, float] | None
"""机器可读阈值；区间阈值使用 ``(lower, upper)``。"""

RiskSeverity = Literal["info", "warn", "fail"]
"""checkpoint 级跨层风险定位严重级别。"""


class Phase1MetricResult(PydanticBaseModel):
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

    # 机器可读阈值。单侧阈值为 float，区间阈值为 (lower, upper)。
    threshold_value: MetricThresholdValue = None

    # 指标方向，用于 report 计算阈值距离和排序解释。
    direction: MetricDirection | None = None

    # 到通过边界的有符号距离。>= 0 表示在阈值安全侧，< 0 表示越界。
    distance_to_threshold: float | None = None
  


class Phase1RiskFinding(PydanticBaseModel):
    """checkpoint 级跨层风险定位结果。

    功能说明:
        把 hard gate metric、drift diagnostic、code diagnostic 和 layer payload
        中的异常信号合并成 report/审计可直接消费的行动项。

    设计边界:
        - 不参与 hard gate pass/fail；
        - 不重新计算底层 raw metrics；
        - 只引用已经存在的 metric、code 和 morphology-motif pair 证据。
    """

    # 风险等级。fail 表示已经阻断 hard gate，warn 表示需要排查，info 表示边界风险。
    severity: RiskSeverity

    # 简短风险标题。
    title: str

    # 风险主因说明。
    reason: str

    # 关联 metric 稳定名称。
    related_metrics: tuple[str, ...] = ()

    # 关联 code id。
    related_codes: tuple[int, ...] = ()

    # 关联 morphology-motif pair 字符串。
    related_pairs: tuple[str, ...] = ()

    # 建议动作。
    recommended_action: str = ""


class Phase1LayerResult(PydanticBaseModel):
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


class Phase1ValidationResult(PydanticBaseModel):
    """单个 checkpoint 的完整 Phase I validation 结果。

    功能说明:
        汇总 hard gate/reference 判定、综合评分、失败层列表、强类型 raw metrics、
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

    # 四层 hard gate 是否全部通过。
    passed: bool

    # 综合评分对象。只有 passed=True 时通常才有值；失败 checkpoint 推荐为 None。
    score: Phase1ValidationScore | None

    # 失败 layer 名称列表。
    failed_layers: tuple[str, ...]

    # hard-gate/reference layer 判定结果。
    layers: tuple[Phase1LayerResult, ...]

    # 五层强类型 raw metrics。
    metrics: Phase1ValidationMetrics

    # code-level 诊断表。
    code_diagnostics: tuple[Phase1CodeDiagnostic, ...]

    # 横向 drift 诊断。key 为稳定诊断名，value 为 metric result。
    drift_diagnostics: Mapping[str, Phase1MetricResult]

    # score 接近时的决胜指标。
    tie_breaker_metrics: Phase1TieBreakerMetrics

    # checkpoint 级跨层风险定位。
    risk_findings: tuple[Phase1RiskFinding, ...] = ()

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

    


__all__ = [
    "MetricDirection",
    "MetricSeverity",
    "MetricThresholdValue",
    "Phase1LayerResult",
    "Phase1MetricResult",
    "Phase1RiskFinding",
    "Phase1ValidationResult",
    "RiskSeverity",
]

