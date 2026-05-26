"""Phase I codebook validation 配置定义。

本文件只放配置对象，不放指标计算、规则判定或文件读写逻辑。
配置按职责拆成三类:

1. 五个 layer thresholds schema: 四层 hard gate 和一层 reference 各自使用的阈值；
2. ``Phase1ValidationScoreWeights``: checkpoint 通过 hard gate 后的综合评分权重；
3. ``Phase1ValidationRuntimeConfig``: evaluator 和各层 metric calculator 的运行参数。

设计上不定义阈值总配置类，也不定义第四个总配置类。调用方应显式持有五个
layer thresholds 对象、一个评分权重对象和一个运行参数对象，避免不同层阈值、
权重和运行参数混在一起后难以审计。
"""

from __future__ import annotations

from pydantic import model_validator

from src.utils import PydanticMappingModel
from .phase1_validation_behavior_quality import Phase1BehaviorQualityThresholds
from .phase1_validation_label_predictability import Phase1LabelPredictabilityThresholds
from .phase1_validation_oracle_profitability import Phase1OracleProfitabilityThresholds
from .phase1_validation_teacher_quality import Phase1TeacherQualityThresholds
from .phase1_validation_vq_internal import Phase1VQInternalThresholds


class Phase1ValidationScoreWeights(PydanticMappingModel):
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
    oracle_profitability: float = 0.35

    # 保留兼容旧配置；label predictability 不再读取指标进入主 score。
    # 若旧配置传入非零值，score 层会把该权重并入 oracle_profitability。
    label_predictability: float = 0.0


class Phase1ValidationRuntimeConfig(PydanticMappingModel):
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
    fee_rate: float = 0.0004

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
    probe_epochs: int = 100

    # label predictability probe 的学习率。用于训练 logistic/shallow MLP 等轻量模型。
    probe_learning_rate: float = 1e-3

    # label predictability probe 的 batch size。用于控制 probe 训练吞吐和显存占用。
    probe_batch_size: int = 256

    # validation 过程随机种子。用于 random label baseline 和 probe 训练可复现。
    random_seed: int = 42

    # codebook size。设置时优先用于 Layer 1 occupancy/perplexity 统计；
    # 未设置时由 snapshot.distances.shape[-1] 推断。
    codebook_size: int | None = None

    # code id 对齐使用的 prototype 类型。auto/action 优先 decoded action prototype，
    # code 优先 latent/code embedding prototype；缺失时回退到 raw id。
    code_alignment_prototype: str = "auto"

    @model_validator(mode="after")
    def _validate_code_alignment_prototype(self) -> "Phase1ValidationRuntimeConfig":
        """校验 runtime config 枚举字段。"""

        if self.code_alignment_prototype not in {"auto", "action", "code"}:
            raise ValueError(
                "code_alignment_prototype must be one of auto, action, code"
            )
        return self
