"""Phase I codebook validation 中间数据与强类型指标 schema。

本文件只定义 evaluator、rules、scoring、report 之间传递的数据结构，不包含
模型调用、DataLoader 遍历、指标公式计算或文件系统读写逻辑。

使用场景:
    1. ``Phase1CodebookEvaluator`` 收集 split snapshot；
    2. 五个 validation layer calculator 基于 snapshot 计算 raw metrics；
    3. ``phase1_validation_rules.py`` 读取强类型 metrics 并输出判定结果；
    4. ``phase1_validation_score.py`` 读取强类型 metrics 并计算综合评分；
    5. report/checkpoint selector 通过 ``to_dict()`` 或 ``to_flat_dict()`` 消费
       稳定的序列化结果。

设计约束:
    - 代码内部访问指标时使用 Pydantic 字段，不通过字符串 key 访问；
    - 字符串 key 只出现在 ``to_dict()`` / ``to_flat_dict()`` 的序列化结果中；
    - 缺失或不可计算的数值指标使用 ``float("nan")`` 或 ``None``，不要省略字段；
    - 本模块不依赖 torch、模型类、dataloader 或文件系统。
"""

from __future__ import annotations

from typing import Mapping, TypeAlias

import numpy as np
from pydantic import Field

from src.utils import PydanticBaseModel
from .phase1_validation_behavior_quality import (
    Phase1BehaviorQualityMetrics,
    Phase1BehaviorQualityPayload,
)
from .phase1_validation_label_predictability import (
    Phase1LabelPredictabilityMetrics,
    Phase1LabelPredictabilityPayload,
)
from .phase1_validation_oracle_profitability import (
    Phase1OracleProfitabilityMetrics,
    Phase1OracleProfitabilityPayload,
)
from .phase1_validation_teacher_quality import (
    Phase1TeacherQualityMetrics,
    Phase1TeacherQualityPayload,
)
from .phase1_validation_vq_internal import Phase1VQInternalMetrics
 
 

class Phase1EvaluationSnapshot(PydanticBaseModel):
    """单个 split 在某个 checkpoint 下的完整可计算状态。

    功能说明:
        由 evaluator 遍历 DataLoader 后生成，保存计算五层 validation raw metrics
        所需的中间数组和基础重构指标。

    使用场景:
        layer0 到 layer4 的 metric calculator 读取该对象计算 teacher quality、
        VQ internal、behavior quality、oracle profitability 和 label predictability。
        该对象通常只在内存中流转，不建议完整写入常规 checkpoint。

    数组形状约定:
        ``N`` 表示 split 中 horizon 样本数；
        ``H`` 表示每条 horizon 的时间步长度；
        ``F`` 表示单步状态特征维度；
        ``A`` 表示 action 类别数；
        ``K`` 表示 codebook size；
        ``D`` 表示 latent/code embedding 维度。
    """

    # 数据 split 名称，例如 "train"、"val" 或 "test"。用于区分指标来源。
    split: str

    # 当前 checkpoint 或 epoch 编号。用于关联训练过程和 assignment history。
    epoch: int

    # horizon 样本稳定 ID，shape=[N]。用于跨 epoch 对齐同一样本并计算 assignment churn。
    sample_ids: np.ndarray

    # horizon 状态特征数组，shape=[N, H, F]。用于 label predictability probe 和部分 morphology 诊断。
    states: np.ndarray

    # horizon 相对状态特征数组，shape=[N, H, F_relative]。用于 decoder 条件输入。
    relative_states: np.ndarray

    # horizon 趋势状态特征数组，shape=[N, H, F_trend]。用于 decoder 条件输入。
    trend_states: np.ndarray

    # horizon 价格序列，shape=[N, H]。用于收益、fee drag、morphology 和 oracle profitability 计算；缺失时可为 None。
    prices: np.ndarray | None

    # DP teacher 动作序列，shape=[N, H]。用于重构准确率、direction accuracy、teacher return 计算。
    demo_actions: np.ndarray

    # DP teacher reward 序列，shape=[N, H]。用于 teacher quality 和 retention ratio 计算。
    demo_rewards: np.ndarray

    # decoder 在 assigned code 条件下输出的离散动作序列，shape=[N, H]，通常为 decoded_logits 的 argmax。
    decoded_actions: np.ndarray

    # decoder 原始 logits，shape=[N, H, A]。用于 action accuracy、top-k 诊断和后续更细粒度 report。
    decoded_logits: np.ndarray

    # encoder/quantizer 分配的 code id，shape=[N]。用于 code occupancy、per-code metrics 和 label probe。
    code_ids: np.ndarray

    # encoder 连续 latent，shape=[N, D]。用于 latent silhouette、quantization distance 和聚类诊断。
    z_e: np.ndarray

    # quantized latent，shape=[N, D]。用于量化误差和 decoder 条件输入诊断。
    z_q: np.ndarray

    # 每个样本到 codebook 各 code 的距离，shape=[N, K]。用于 nearest/second-nearest margin 和分配置信度诊断。
    distances: np.ndarray

    # 当前 split 的平均 reconstruction loss。用于泛化 gap 和 tie-breaker。
    reconstruction_loss: float

    # 当前 split 的动作重构准确率。用于第一层 VQ internal hard gate。
    action_accuracy: float

    # horizon LOB 深度行情，shape=[N, H, 20]。用于执行收益中的盘口滑点计算；缺失时可为 None。
    depthprices: np.ndarray | None = Field(default=None, exclude=True)

class CodeAssignmentSnapshot(PydanticBaseModel):
    """某个 epoch 的 code assignment 快照。

    功能说明:
        保存样本到 code 的离散分配结果，以及当前 active code 集合。

    使用场景:
        计算相邻 epoch assignment churn、active code lifetime，并诊断 codebook
        是否仍在重排。

    数组形状约定:
        ``N`` 表示该 assignment snapshot 覆盖的 horizon 样本数。
    """

    # 当前 assignment 所属 epoch。
    epoch: int

    # 当前 assignment 所属 split，通常使用 validation split 计算 churn。
    split: str

    # 与 code_ids 一一对应的稳定样本 ID，shape=[N]。用于跨 epoch 对齐同一样本。
    sample_ids: np.ndarray

    # 每个样本在当前 epoch 被分配到的 code id，shape=[N]。
    code_ids: np.ndarray

    # 当前 epoch 满足 active 标准的 code id 集合。
    active_codes: tuple[int, ...]

    # 每个 code 的 latent/code embedding 原型，shape=[K, D]；无样本 code 行为 NaN。
    code_prototypes: np.ndarray | None = None

    # 每个 code 的 decoded action/position 原型，shape=[K, H]；无样本 code 行为 NaN。
    action_prototypes: np.ndarray | None = None
 
 


class Phase1CodeDiagnostic(PydanticBaseModel):
    """单个 code 的 report 级诊断数据。

    功能说明:
        汇总 code support、occupancy、dominant morphology/motif/pair 和 decoded
        profitability 等信息。

    使用场景:
        直接供 report 渲染 code-level 表格，也可用于定位弱 code、坏 code 或重复 code。
    """

    # codebook 中的 code id。
    code_id: int

    # validation split 中分配到该 code 的样本数量。
    support: int

    # 该 code 的样本占比，等于 support / N。
    occupancy: float

    # 该 code 内占比最高的市场形态；不可计算时为 None。
    dominant_morphology: str | None

    # dominant morphology 在该 code 内的占比；不可计算时为 None。
    dominant_morphology_ratio: float | None

    # dominant morphology 相对全体验证集分布的 lift；不可计算时为 None。
    morphology_lift: float | None

    # 该 code 内占比最高的交易 motif；不可计算时为 None。
    dominant_motif: str | None

    # dominant motif 在该 code 内的占比；不可计算时为 None。
    dominant_motif_ratio: float | None

    # 该 code 内占比最高的 morphology-motif 组合；不可计算时为 None。
    dominant_pair: str | None

    # dominant pair 在该 code 内的占比；不可计算时为 None。
    dominant_pair_ratio: float | None

    # 该 code 的 decoded mean advantage vs flat；不可计算时为 None。
    decoded_mean_advantage: float | None

    # 该 code 的 decoded win rate vs flat；不可计算时为 None。
    decoded_win_rate: float | None

    # 该 code 的 decoded return 相对 DP teacher return 的保留比例；不可计算时为 None。
    retention_ratio: float | None

    # 该 code 的手续费拖累比例；不可计算时为 None。
    fee_drag: float | None

    # 该 code 的综合诊断状态，由 support、结构清晰度、pair 稳定性和盈利辅助证据共同决定。
    status: str


class Phase1TieBreakerMetrics(PydanticBaseModel):
    """checkpoint 综合分接近时使用的决胜指标。

    使用场景:
        当多个 checkpoint 都通过 hard gate 且 ``validation.score`` 差距小于
        tie tolerance 时，checkpoint selector 按这些字段顺序做稳定排序。
    """

    # 风险调整收益，越高越优先。
    risk_adjusted_return: float

    # probe top-3 accuracy，越高越说明 selector 可学习性更好。
    probe_top3_accuracy: float

    # decoded 盈利保留比例，越高越说明压缩后保留 DP 盈利能力更好。
    retention_ratio: float

    # active code ratio，越高越说明 codebook 使用更充分。
    active_code_ratio: float

    # max code occupancy，越低越说明 label collapse 风险更低。
    max_code_occupancy: float

    # validation reconstruction loss，越低越说明基础重构质量更好。
    reconstruction_loss: float


class Phase1PerCodeProfitability(PydanticBaseModel):
    """单个 code 的盈利性判定结果。

    使用场景:
        第三层 oracle profitability 计算 per-code bad ratio，第二层 behavior
        quality 可复用该对象计算 profitable-code coverage。
    """

    # codebook 中的 code id。
    code_id: int

    # 该 code 在其 assigned validation samples 上的 mean advantage vs flat。
    mean_advantage: float

    # 该 code 的 win rate vs flat。
    win_rate: float

    # 该 code 的 decoded return 相对 DP teacher return 的保留比例。
    retention_ratio: float

    # 该 code 的手续费拖累比例。
    fee_drag: float

    # 该 code 是否通过 per-code 盈利条件。
    passed: bool


Phase1LayerMetrics: TypeAlias = (
    Phase1TeacherQualityMetrics
    | Phase1VQInternalMetrics
    | Phase1BehaviorQualityMetrics
    | Phase1OracleProfitabilityMetrics
    | Phase1LabelPredictabilityMetrics
)
"""单个 validation layer calculator 输出的强类型 metrics 类型。"""


class Phase1LayerComputation(PydanticBaseModel):
    """单个 validation layer 的 raw metric 计算结果。

    功能说明:
        五个 ``phase1_validation_layers/layer*.py`` 文件只负责 raw metric 计算，
        不做 hard gate pass/fail 判定。该对象把本层强类型 metrics、可选
        code-level diagnostics 和后续 layer 需要复用的额外 payload 打包返回。

    使用场景:
        ``Phase1CodebookEvaluator`` 调用 layer calculator 后，读取 ``metrics``
        交给 ``phase1_validation_rules.py``，并把 ``code_diagnostics`` 和
        ``extra_payload`` 合并进 checkpoint/report payload。
    """

    # layer 数字编号，0 到 4。
    layer_id: int

    # layer 稳定名称，例如 "teacher_quality"。
    layer_name: str

    # 本层强类型 raw metrics。
    metrics: Phase1LayerMetrics

    # 可选 code-level 诊断表，主要由 layer2/layer3 填充。
    code_diagnostics: tuple[Phase1CodeDiagnostic, ...] = ()

    # 可选额外中间产物，例如 per-code profitability 或 probe diagnostics。
    extra_payload: Mapping[str, object] = Field(default_factory=dict)


class Phase1ValidationMetrics(PydanticBaseModel):
    """五层 validation raw metrics 聚合对象。

    功能说明:
        统一承载五层强类型 metrics，是 rules、scoring、report 和 checkpoint
        selector 的核心输入。

    使用场景:
        ``Phase1ValidationResult`` 持有该对象；``to_flat_dict()`` 为 selector
        提供扁平 key 快速读取，但业务代码内部仍应使用强类型字段访问。
    """

    # 第零层 DP teacher 质量 metrics。
    teacher_quality: Phase1TeacherQualityMetrics

    # 第一层 VQ 内部质量 metrics。
    vq_internal: Phase1VQInternalMetrics

    # 第二层 archetype 行为质量 metrics。
    behavior_quality: Phase1BehaviorQualityMetrics

    # 第三层 oracle assigned-label 盈利性 metrics。
    oracle_profitability: Phase1OracleProfitabilityMetrics

    # 第四层 label 可预测性 metrics。
    label_predictability: Phase1LabelPredictabilityMetrics
 
__all__ = [
    "CodeAssignmentSnapshot",
    "Phase1BehaviorQualityMetrics",
    "Phase1BehaviorQualityPayload",
    "Phase1CodeDiagnostic",
    "Phase1EvaluationSnapshot",
    "Phase1LabelPredictabilityMetrics",
    "Phase1LabelPredictabilityPayload",
    "Phase1LayerComputation",
    "Phase1LayerMetrics",
    "Phase1OracleProfitabilityMetrics",
    "Phase1OracleProfitabilityPayload",
    "Phase1PerCodeProfitability",
    "Phase1TeacherQualityMetrics",
    "Phase1TeacherQualityPayload",
    "Phase1TieBreakerMetrics",
    "Phase1VQInternalMetrics",
    "Phase1ValidationMetrics",
]
