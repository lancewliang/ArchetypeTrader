"""Phase II layer 5 generalization, stability, and predictability metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from src.utils import PydanticBaseModel

from .phase2_layer_computation import Phase2LayerComputationBase
from .phase2_validation_rule_helpers import (
    _build_layer_result,
    _ge,
    _le,
)

if TYPE_CHECKING:
    from .phase2_metric_results import Phase2LayerResult


class Phase2PredictabilityPayload(PydanticBaseModel):
    """Selector 可预测性 raw metrics 计算的中间 payload。

    该 payload 保存 probe 训练诊断、confusion matrix 和随机种子。它只用于
    计算可预测性 raw metrics，不要求报告保存逐样本预测结果。
    """

    # probe 在 train split 上预测 selected/assigned code 的 accuracy。用途：判断
    # 可见状态是否包含可学习信号；方向：越大越好，但 train 高而 validation 低表示过拟合。
    probe_train_accuracy: float

    # probe 在 validation split 上的 accuracy。用途：评估可预测性泛化；方向：
    # 越大越好。
    probe_validation_accuracy: float

    # probe_train_accuracy - probe_validation_accuracy。用途：诊断 probe 过拟合；
    # 方向：绝对值越小越稳，正值过大表示 train-only 记忆。
    probe_predictability_gap: float

    # probe confusion matrix。用途：定位哪些 code 可预测/混淆；方向：过程诊断，
    # 无单一好坏方向。
    probe_confusion_matrix: tuple[tuple[int, ...], ...]

    # probe 随机种子。用途：复现实验；方向：过程数据，无好坏方向。
    probe_seed: int


class Phase2PredictabilityMetrics(PydanticBaseModel):
    """Selector 可预测性 raw metrics。"""

    # 用可见状态预测 selected code 的 top-1 accuracy。用途：验证 selector 决策是否
    # 能由在线可见状态解释；方向：越大越好。
    probe_top1_accuracy: float

    # 用可见状态预测 selected code 的 top-3 accuracy。用途：判断是否能缩小候选
    # code 范围；方向：越大越好。
    probe_top3_accuracy: float

    # 按 code 平衡后的 accuracy。用途：避免 probe 只预测高频 code；方向：越大越好。
    probe_balanced_accuracy: float

    # 给定 morphology 后 selected code 的条件熵。用途：检查市场形态是否约束
    # code 选择；方向：越小越好，至少应低于总体 entropy 的一定比例。
    selected_code_entropy_given_morphology: float

    # selected code 的总体熵。用途：作为条件熵参考基准；方向：诊断字段，过低
    # 表示 collapse，过高可能表示接近随机。
    selected_code_entropy: float

    # selected code 与可见状态/morphology 的 mutual information lift。用途：衡量
    # 状态对 code 选择的信息增益；方向：越大越好。
    mutual_information_lift: float


class Phase2PredictabilityThresholds(PydanticBaseModel):
    """Selector 可预测性阈值配置。"""

    # top-1 accuracy 固定下限。方向：probe_top1_accuracy 越大越好。
    probe_top1_floor: float = 0.25

    # top-1 accuracy 相对随机 1/K 的倍率下限。方向：倍率越高要求越严格。
    probe_top1_k_factor: float = 1.5

    # top-3 accuracy 固定下限。方向：probe_top3_accuracy 越大越好。
    probe_top3_floor: float = 0.55

    # top-3 accuracy 相对随机 3/K 的倍率下限。方向：倍率越高要求越严格。
    probe_top3_k_factor: float = 3.0

    # balanced accuracy 下限。方向：probe_balanced_accuracy 越大越好。
    probe_balanced_accuracy_min: float = 0.25

    # mutual_information_lift 下限。方向：越大越好。
    mutual_information_lift_min: float = 2.0

    # 条件熵相对总体熵的上限比例。方向：selected_code_entropy_given_morphology
    # 越小越好。
    entropy_given_morphology_max_ratio: float = 0.85

    def top1_threshold(self, num_archetypes: int) -> float:
        """返回 codebook size 自适应 top-1 阈值。"""

        return max(self.probe_top1_floor, self.probe_top1_k_factor / num_archetypes)

    def top3_threshold(self, num_archetypes: int) -> float:
        """返回 codebook size 自适应 top-3 阈值。"""

        return max(self.probe_top3_floor, self.probe_top3_k_factor / num_archetypes)


class Phase2GeneralizationStabilityPayload(PydanticBaseModel):
    """Layer 5 raw metrics 计算的中间 payload。"""

    # train split 上的 selector score/return 摘要。用途：计算 train-val gap；
    # 方向：过程数据，单独越大不代表泛化更好。
    train_score: float | None = None

    # 历史 validation score 序列。用途：计算 validation_score_churn；方向：
    # 序列本身无好坏方向，波动越小越稳。
    validation_score_history: tuple[float, ...] = ()

    # 同一样本跨 epoch selected action/code 的变化率历史。用途：诊断决策边界稳定性；
    # 方向：越小越稳定。
    selected_action_churn_history: tuple[float, ...] = ()

    # Q value 尺度历史。用途：诊断 Q overestimation 或发散；方向：过程数据，
    # 尺度过大或波动过大不好。
    q_value_scale_history: tuple[float, ...] = ()

    # 可预测性 probe 过程数据。用途：计算 predictability metrics；方向：由其中
    # 字段定义。
    predictability_payload: Phase2PredictabilityPayload | None = None


class Phase2GeneralizationStabilityMetrics(PydanticBaseModel):
    """Layer 5 generalization and stability raw metrics。"""

    # train return/score 与 validation return/score 的差距。用途：识别过拟合；
    # 方向：越小越好。
    train_val_return_gap: float

    # validation return 与 test return 的差距。用途：最终泛化诊断；方向：越小越好，
    # 但 test 不应回流参与 checkpoint selection。
    val_test_return_gap: float

    # train 和 validation selected code 分布的 KL。用途：检查使用分布泛化；
    # 方向：越小越稳。
    train_val_usage_kl: float

    # validation score 历史波动。用途：判断 checkpoint 高点是否偶然；方向：
    # 越小越稳。
    validation_score_churn: float

    # selected action/code 跨 epoch 变化率。用途：判断决策边界稳定性；方向：
    # 越小越稳。
    selected_action_churn: float

    # Q value 绝对尺度均值。用途：检测 overestimation 或数值发散；方向：
    # 不是越大越好，过大不好。
    q_value_scale_mean: float

    # Q value 尺度标准差。用途：检测估值波动；方向：越小越稳。
    q_value_scale_std: float

    # top1 与 top2 Q value margin 均值。用途：衡量选择置信度；方向：越大越好。
    q_margin_mean: float

    # Q margin 低于置信阈值的样本比例。用途：衡量低置信选择覆盖面；方向：
    # 越小越好。
    low_confidence_selection_rate: float

    # TD loss 随 epoch 的趋势。用途：诊断 value learning 是否恶化；方向：
    # 越小/越不升高越好，正向上升表示风险。
    td_loss_trend: float

    # imitation loss 随 epoch 的趋势。用途：诊断 demonstration 约束是否失效；
    # 方向：越小/越不升高越稳，但过低也可能表示只复制 label。
    imitation_loss_trend: float

    # reward_mean 随 epoch 的趋势。用途：观察训练 reward 是否改善；方向：
    # 越大通常越好，但需结合 validation return。
    reward_mean_trend: float

    # 可选可预测性 metrics。用途：说明 selector 选择是否可由可见状态解释；
    # 方向：由 Phase2PredictabilityMetrics 字段定义。
    predictability: Phase2PredictabilityMetrics | None = None


class Phase2Layer5GeneralizationStabilityComputation(
    Phase2LayerComputationBase
):
    """Layer 5 generalization/stability 的 raw metrics 和过程 payload。"""

    layer_id: Literal[5] = 5
    layer_name: Literal["generalization_stability"] = (
        "generalization_stability"
    )
    metrics: Phase2GeneralizationStabilityMetrics
    generalization_stability_payload: Phase2GeneralizationStabilityPayload


class Phase2GeneralizationStabilityThresholds(PydanticBaseModel):
    """Layer 5 generalization and stability 阈值配置。"""

    # train_val_return_gap warning 上限。方向：越小越好。
    train_val_return_gap_warn_max: float = 0.50

    # train_val_usage_kl warning 上限。方向：越小越稳。
    train_val_usage_kl_warn_max: float = 0.50

    # validation_score_churn warning 上限。方向：越小越稳。
    validation_score_churn_warn_max: float = 0.15

    # selected_action_churn warning 上限。方向：越小越稳。
    selected_action_churn_warn_max: float = 0.35

    # q_value_scale_mean warning 上限。方向：Q 尺度过大不好。
    q_value_scale_mean_warn_max: float = 100.0

    # q_value_scale_std warning 上限。方向：越小越稳。
    q_value_scale_std_warn_max: float = 100.0

    # q_margin_mean warning 下限。方向：越大越好。
    q_margin_mean_warn_min: float = 0.10

    # low_confidence_selection_rate warning 上限。方向：越小越好。
    low_confidence_selection_rate_warn_max: float = 0.40

    # 可预测性阈值集合。用途：生成 predictability warning/reference 指标；
    # 方向：由子阈值字段定义。
    predictability_thresholds: Phase2PredictabilityThresholds = Field(
        default_factory=Phase2PredictabilityThresholds
    )


def evaluate_generalization_stability_rules(
    metrics: Phase2GeneralizationStabilityMetrics,
    thresholds: Phase2GeneralizationStabilityThresholds,
    *,
    num_archetypes: int,
) -> Phase2LayerResult:
    """构造 Layer 5 warning/reference 结果。"""

    layer = "generalization_stability"
    results = [
        _le(
            name="train_val_return_gap",
            value=metrics.train_val_return_gap,
            threshold_value=thresholds.train_val_return_gap_warn_max,
            layer=layer,
            message="train 明显好于 validation 时标记过拟合风险",
            severity_when_failed="warn",
        ),
        _le(
            name="train_val_usage_kl",
            value=metrics.train_val_usage_kl,
            threshold_value=thresholds.train_val_usage_kl_warn_max,
            layer=layer,
            message="train/validation selected code 分布差异过大时需要解释",
            severity_when_failed="warn",
        ),
        _le(
            name="validation_score_churn",
            value=metrics.validation_score_churn,
            threshold_value=thresholds.validation_score_churn_warn_max,
            layer=layer,
            message="validation score 波动过大时，最高点 checkpoint 可能不稳定",
            severity_when_failed="warn",
        ),
        _le(
            name="selected_action_churn",
            value=metrics.selected_action_churn,
            threshold_value=thresholds.selected_action_churn_warn_max,
            layer=layer,
            message="同一样本跨 epoch selected code 频繁变化表示决策边界不稳",
            severity_when_failed="warn",
        ),
        _le(
            name="q_value_scale_mean",
            value=metrics.q_value_scale_mean,
            threshold_value=thresholds.q_value_scale_mean_warn_max,
            layer=layer,
            message="Q value 均值尺度过大可能表示 overestimation",
            severity_when_failed="warn",
        ),
        _le(
            name="q_value_scale_std",
            value=metrics.q_value_scale_std,
            threshold_value=thresholds.q_value_scale_std_warn_max,
            layer=layer,
            message="Q value 方差过大表示估值不稳定",
            severity_when_failed="warn",
        ),
        _ge(
            name="q_margin_mean",
            value=metrics.q_margin_mean,
            threshold_value=thresholds.q_margin_mean_warn_min,
            layer=layer,
            message="top1/top2 Q margin 太低表示选择置信度不足",
            severity_when_failed="warn",
        ),
        _le(
            name="low_confidence_selection_rate",
            value=metrics.low_confidence_selection_rate,
            threshold_value=thresholds.low_confidence_selection_rate_warn_max,
            layer=layer,
            message="低置信选择比例过高会降低 checkpoint 稳定性",
            severity_when_failed="warn",
        ),
    ]
    if metrics.predictability is not None:
        results.extend(
            _predictability_results(
                metrics.predictability,
                thresholds.predictability_thresholds,
                layer=layer,
                num_archetypes=num_archetypes,
            )
        )
    return _build_layer_result(
        layer_id=5,
        name=layer,
        metrics=tuple(results),
        force_passed=True,
    )


def _predictability_results(
    metrics: Phase2PredictabilityMetrics,
    thresholds: Phase2PredictabilityThresholds,
    *,
    layer: str,
    num_archetypes: int,
) -> tuple[Any, ...]:
    """构造可预测性 reference/warn 指标结果。"""

    entropy_threshold = (
        metrics.selected_code_entropy
        * thresholds.entropy_given_morphology_max_ratio
    )
    return (
        _ge(
            name="predictability_probe_top1_accuracy",
            value=metrics.probe_top1_accuracy,
            threshold_value=thresholds.top1_threshold(num_archetypes),
            layer=layer,
            message="probe top-1 accuracy 用于参考 selector action 是否可由可见状态预测",
            severity_when_failed="warn",
        ),
        _ge(
            name="predictability_probe_top3_accuracy",
            value=metrics.probe_top3_accuracy,
            threshold_value=thresholds.top3_threshold(num_archetypes),
            layer=layer,
            message="probe top-3 accuracy 用于参考 selector 是否缩小了候选 code 范围",
            severity_when_failed="warn",
        ),
        _ge(
            name="predictability_probe_balanced_accuracy",
            value=metrics.probe_balanced_accuracy,
            threshold_value=thresholds.probe_balanced_accuracy_min,
            layer=layer,
            message="balanced accuracy 用于检查 probe 是否只预测高频 selected code",
            severity_when_failed="warn",
        ),
        _le(
            name="selected_code_entropy_given_morphology",
            value=metrics.selected_code_entropy_given_morphology,
            threshold_value=entropy_threshold,
            layer=layer,
            message="给定 morphology 后 selected code 条件熵应下降",
            severity_when_failed="warn",
        ),
        _ge(
            name="predictability_mutual_information_lift",
            value=metrics.mutual_information_lift,
            threshold_value=thresholds.mutual_information_lift_min,
            layer=layer,
            message="mutual information lift 用于参考 selected code 与可见状态的关系",
            severity_when_failed="warn",
        ),
    )


__all__ = [
    "Phase2Layer5GeneralizationStabilityComputation",
    "Phase2GeneralizationStabilityMetrics",
    "Phase2GeneralizationStabilityPayload",
    "Phase2GeneralizationStabilityThresholds",
    "Phase2PredictabilityMetrics",
    "Phase2PredictabilityPayload",
    "Phase2PredictabilityThresholds",
    "evaluate_generalization_stability_rules",
]
