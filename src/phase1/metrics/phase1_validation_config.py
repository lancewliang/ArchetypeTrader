"""Phase I codebook validation 配置定义。

本文件只放配置对象，不放指标计算、规则判定或文件读写逻辑。
配置按职责拆成三类:

1. 五个 layer thresholds dataclass: 五层 hard gate 各自使用的阈值；
2. ``Phase1ValidationScoreWeights``: checkpoint 通过 hard gate 后的综合评分权重；
3. ``Phase1ValidationRuntimeConfig``: evaluator 和各层 metric calculator 的运行参数。

设计上不定义阈值总配置类，也不定义第四个总配置类。调用方应显式持有五个
layer thresholds 对象、一个评分权重对象和一个运行参数对象，避免不同层阈值、
权重和运行参数混在一起后难以审计。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, TypeVar


TConfig = TypeVar("TConfig")


def _dataclass_from_mapping(config_type: type[TConfig], payload: Mapping[str, Any]) -> TConfig:
    """从 mapping 反序列化配置对象。

    功能说明:
        只读取目标 dataclass 已声明的字段，忽略 payload 中多余的 key。这样旧
        checkpoint 或 report 中带有扩展字段时，当前代码仍可以加载默认配置。

    使用场景:
        checkpoint/report 读取配置快照时调用各配置类的 ``from_dict()``，由该
        helper 过滤字段并构造 frozen dataclass。
    """

    field_names = {field.name for field in fields(config_type)}
    values = {key: value for key, value in payload.items() if key in field_names}
    return config_type(**values)


@dataclass(frozen=True)
class Phase1TeacherQualityThresholds:
    """第零层 DP teacher 质量阈值配置。

    功能说明:
        保存用于判断 DP 示范数据是否值得学习的 hard gate 阈值。

    使用场景:
        由 ``evaluate_teacher_quality_rules()`` 消费。第零层失败通常表示应
        重新构造 teacher 数据，而不是继续挑选 VQ checkpoint。
    """

    # DP teacher 胜率下限。用于第零层判断 DP 示范是否广泛优于 flat baseline。
    dp_win_rate_min: float = 0.58

    # 弱机会样本比例上限。用于过滤大量收益接近手续费噪声的 teacher 数据。
    near_zero_opportunity_ratio_max: float = 0.35

    # 手续费翻倍后的收益保留比例下限。用于判断 DP 示范是否过度依赖微小价差。
    fee_sensitivity_min: float = 0.60

    # 非 neutral 市场形态覆盖率下限。用于判断样本是否有足够可学习的市场结构。
    morphology_coverage_min: float = 0.60

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供 checkpoint、report 或日志落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1TeacherQualityThresholds":
        """从 checkpoint/report 中的 dict 恢复第零层阈值配置。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase1VQInternalThresholds:
    """第一层 VQ 内部质量阈值配置。

    功能说明:
        保存用于判断 VQ codebook 是否稳定、未塌缩、可用的 hard gate 阈值。

    使用场景:
        由 ``evaluate_vq_internal_rules()`` 消费。该层重点检查重构保真、
        code 使用健康度、assignment 稳定性和量化边界清晰度。
    """

    # validation 动作重构准确率下限。用于第一层判断 decoder 是否保留 DP 动作信息。
    action_accuracy_min: float = 0.85

    # validation/train 重构损失比值上限。用于检查重构能力是否明显过拟合训练集。
    reconstruction_loss_gap_max: float = 1.25

    # active code 占 codebook 的比例下限。用于检查 codebook 是否被充分使用。
    active_code_ratio_min: float = 0.80

    # 单个 code 最大占用比例上限。用于检测 label collapse 或单 code 吃掉过多样本。
    max_code_occupancy_max: float = 0.40

    # 归一化 perplexity 下限。用于防止 code 分布过度塌缩。
    normalized_perplexity_min: float = 0.50

    # 归一化 perplexity 上限。用于防止分配过于接近随机、缺少结构。
    normalized_perplexity_max: float = 0.90

    # dead code 比例上限。用于限制长期几乎没有样本分配的无效 code 数量。
    dead_code_ratio_max: float = 0.20

    # 最近若干 epoch assignment churn 均值上限。用于判断 label 语义是否稳定。
    churn_recent_mean_max: float = 0.15

    # 最近 code 距离和第二近 code 距离的 margin 中位数下限。用于判断分配边界是否清晰。
    margin_median_min: float = 0.10

    # decoded 主方向和 demo 主方向一致率下限。用于检查 long/short/flat 大方向是否保真。
    direction_accuracy_min: float = 0.88

    # 入场时点误差占 horizon 长度的比例上限。用于判断 decoded entry timing 是否偏移过大。
    entry_timing_error_ratio_max: float = 0.15

    # active code 中 lifetime 达标的比例下限。用于判断 code 是否已经形成稳定语义。
    code_lifetime_pass_ratio_min: float = 0.80

    # decoded 与 demo 的换手误差上限，单位为 horizon 归一化换手次数。
    decoder_turnover_error_max: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供 checkpoint、report 或日志落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1VQInternalThresholds":
        """从 checkpoint/report 中的 dict 恢复第一层阈值配置。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase1BehaviorQualityThresholds:
    """第二层 archetype 行为质量阈值配置。

    功能说明:
        保存用于判断每个 code 是否具备清晰、稳定、可区分交易行为含义的
        hard gate 阈值。

    使用场景:
        由 ``evaluate_behavior_quality_rules()`` 消费。该层重点检查 per-code
        support、morphology/motif/pair 纯度、行为一致性和 code 间分离度。
    """

    # 单个 active code 的绝对最小样本数。用于保证 per-code 诊断具备统计支撑。
    min_code_support_abs: int = 100

    # 单个 active code 的相对最小样本比例。实际 support 阈值取 max(绝对阈值, 相对阈值 * N)。
    min_code_support_ratio: float = 0.02

    # support 不足的弱 code 比例上限。设计标准要求超过 20% 时淘汰。
    weak_support_code_ratio_max: float = 0.20

    # morphology/motif/pair/lift 等结构弱 code 比例上限。设计标准要求超过 40% 时淘汰。
    weak_structure_code_ratio_max: float = 0.40

    # 兼容旧配置快照的遗留字段；新规则不再读取该字段。
    weak_code_ratio_max: float = 0.20

    # dominant morphology 占比下限。用于判断 code 是否集中对应某类市场形态。
    dominant_morphology_ratio_min: float = 0.35

    # dominant motif 占比下限。用于判断 code 是否对应清晰的交易行为意图。
    dominant_motif_ratio_min: float = 0.40

    # dominant morphology-motif pair 占比下限。用于判断市场形态和行为是否形成稳定组合。
    dominant_pair_ratio_min: float = 0.30

    # morphology lift 下限。用于判断某 code 是否真的富集某类市场结构。
    morphology_lift_min: float = 1.25

    # 同一 code 内 decoded action sequence 相似度下限。用于衡量 archetype 内部一致性。
    intra_code_similarity_min: float = 0.65

    # code 间距离 / code 内距离的分离度下限。用于判断不同 archetype 是否足够可区分。
    inter_intra_separation_min: float = 1.30

    # 任意两个 code 原型 decoded action 相似度上限。用于识别重复 code。
    duplicate_code_similarity_max: float = 0.85

    # latent silhouette score 下限。用于判断 latent 空间中的 code assignment 是否清晰。
    latent_silhouette_score_min: float = 0.10

    # 具备 per-code 盈利能力的 active code 覆盖率下限。
    profitable_code_coverage_min: float = 0.60

    # 重复 code pair 数量上限。None 表示按当前 codebook size K 动态判定。
    duplicate_code_pair_count_max: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供 checkpoint、report 或日志落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1BehaviorQualityThresholds":
        """从 checkpoint/report 中的 dict 恢复第二层阈值配置。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase1OracleProfitabilityThresholds:
    """第三层 oracle assigned-label 盈利性阈值配置。

    功能说明:
        保存用于判断 assigned label 经 decoder 执行后是否保留 DP 盈利能力的
        hard gate 阈值。

    使用场景:
        由 ``evaluate_oracle_profitability_rules()`` 消费。该层只证明 codebook
        在 oracle label 条件下有可交易潜力，不替代 Phase II selector validation。
    """

    # oracle assigned-label decoded 策略胜率下限。用于第三层判断盈利是否足够广泛。
    decoded_win_rate_min: float = 0.55

    # decoded 策略相对 DP teacher 的收益保留比例下限。用于判断压缩后是否保留盈利能力。
    retention_ratio_min: float = 0.50

    # assigned label 相对随机 label 的收益提升下限。用于判断 encoder label 是否有信息量。
    random_label_relative_lift_min: float = 0.20

    # 负价值 code 比例上限。用于限制 codebook 中明显有害 archetype 的数量。
    bad_code_ratio_max: float = 0.30

    # 收益最高 5% horizon 的贡献比例上限。用于检测收益是否过度依赖少数尾部样本。
    top_5_contribution_max: float = 0.60

    # dominant pair 中正优势 pair 的比例下限。用于判断主要市场-行为组合是否真正盈利。
    dominant_pair_positive_ratio_min: float = 0.60

    # decoded 策略相对 DP teacher 的最大回撤比例上限。
    downside_control_max: float = 1.50

    # decoded 风险调整收益下限。0 表示至少不能为负。
    risk_adjusted_return_min: float = 0.0

    # 全局手续费拖累比例上限。
    fee_drag_max: float = 0.35

    # horizon turnover 与 decoded return 相关性下限。
    turnover_return_correlation_min: float = -0.10

    # per-code 胜率下限。用于构造 Layer 3 per-code profitability passed 字段。
    per_code_win_rate_min: float = 0.52

    # per-code retention ratio 下限。
    per_code_retention_ratio_min: float = 0.40

    # per-code fee drag 上限。
    per_code_fee_drag_max: float = 0.40

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供 checkpoint、report 或日志落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1OracleProfitabilityThresholds":
        """从 checkpoint/report 中的 dict 恢复第三层阈值配置。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase1LabelPredictabilityThresholds:
    """第四层 label 可预测性阈值配置。

    功能说明:
        保存用于判断 assigned label 是否能从 horizon 起点可见状态中学习出来的
        hard gate 阈值。

    使用场景:
        由 ``evaluate_label_predictability_rules()`` 消费。该层用于估计 Phase II
        selector 的可学习性，避免选择 oracle label 盈利但起点状态不可预测的 checkpoint。
    """

    # probe top-1 accuracy 的固定下限。实际阈值取 max(floor, k_factor / K)。
    probe_top1_floor: float = 0.25

    # probe top-1 accuracy 的 codebook size 自适应倍数。用于保证表现明显优于随机猜测。
    probe_top1_k_factor: float = 1.5

    # probe top-3 accuracy 的固定下限。实际阈值取 max(floor, k_factor / K)。
    probe_top3_floor: float = 0.55

    # probe top-3 accuracy 的 codebook size 自适应倍数。用于判断 selector 是否能缩小候选范围。
    probe_top3_k_factor: float = 3.0

    # probe balanced accuracy 下限。用于避免 probe 只学会预测高频 code。
    probe_balanced_accuracy_min: float = 0.25

    # label 和可见状态之间互信息相对随机置换 baseline 的提升下限。
    mutual_information_lift_min: float = 2.0

    # probe label decoded return 相对 oracle assigned-label return 的保留比例下限。
    probe_return_retention_min: float = 0.35

    # H(label | morphology) 相对 H(label) 的上限，用于判断 morphology 是否解释 label 结构。
    label_entropy_given_morphology_max_ratio: float = 0.80

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供 checkpoint、report 或日志落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1LabelPredictabilityThresholds":
        """从 checkpoint/report 中的 dict 恢复第四层阈值配置。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase1ValidationScoreWeights:
    """Phase I validation 综合评分权重配置。

    功能说明:
        定义 checkpoint 通过全部 hard gate 后，各类 normalized score 在最终
        ``validation.score`` 中的权重。

    使用场景:
        只由 ``phase1_validation_score.py`` 消费。失败 checkpoint 不应因为本
        权重获得排序资格；权重只用于合格 checkpoint 之间的排序。
    """

    # DP teacher 质量子分数权重。用于让 teacher 稳定性参与最终 checkpoint 排序。
    teacher_quality: float = 0.10

    # 重构质量子分数权重。用于强调 decoder 对 DP 动作的保真能力。
    reconstruction: float = 0.20

    # codebook 健康度子分数权重。用于奖励高 active ratio、低 collapse、低 churn 的 checkpoint。
    codebook_health: float = 0.15

    # 行为结构子分数权重。用于奖励 morphology/motif/pair 清晰、code 间分离度高的 checkpoint。
    behavior_structure: float = 0.20

    # oracle 盈利性子分数权重。用于优先选择 assigned-label decoder 收益质量更好的 checkpoint。
    oracle_profitability: float = 0.25

    # label 可预测性子分数权重。用于让 Phase II selector 可学习性进入最终排序。
    label_predictability: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供 checkpoint、report 或日志落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1ValidationScoreWeights":
        """从 checkpoint/report 中的 dict 恢复评分权重配置。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase1ValidationRuntimeConfig:
    """Phase I codebook evaluator 运行参数配置。

    功能说明:
        保存 evaluator 和五层 metric calculator 在计算 raw metrics 时需要的运行
        参数，例如手续费、随机 label baseline 次数、probe 训练参数和随机种子。

    使用场景:
        由 ``Phase1CodebookEvaluator`` 以及
        ``phase1_validation_layers/layer*.py`` 消费。本类不表达 hard gate
        标准，也不参与 checkpoint score 权重配置。
    """

    # 单边手续费率。用于计算扣费收益、near-zero opportunity 和 fee drag。
    fee_rate: float = 0.0002

    # 随机 label baseline 重复次数。用于降低 random-label profitability 估计方差。
    random_label_trials: int = 3

    # assignment churn 的最近 epoch 窗口长度。用于衡量 code 语义近期稳定性。
    churn_window_epochs: int = 5

    # active code 的最小占用比例。用于 evaluator 统计 active_code_ratio。
    active_code_min_occupancy: float = 0.01

    # dead code 的最大占用比例。用于 evaluator 统计 dead_code_ratio。
    dead_code_max_occupancy: float = 0.001

    # 收益集中度统计中的 top contribution 比例。默认检查收益最高 5% horizon。
    top_contribution_ratio: float = 0.05

    # label predictability probe 的训练 epoch 数。用于第四层轻量 selector 可学习性验证。
    probe_epochs: int = 20

    # label predictability probe 的学习率。用于训练 logistic/shallow MLP 等轻量模型。
    probe_learning_rate: float = 1e-3

    # label predictability probe 的 batch size。用于控制 probe 训练吞吐和显存占用。
    probe_batch_size: int = 256

    # validation 过程随机种子。用于 random label baseline 和 probe 训练可复现。
    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """序列化为普通 dict，供 checkpoint、report 或日志落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1ValidationRuntimeConfig":
        """从 checkpoint/report 中的 dict 恢复 evaluator 运行参数配置。"""

        return _dataclass_from_mapping(cls, payload)
