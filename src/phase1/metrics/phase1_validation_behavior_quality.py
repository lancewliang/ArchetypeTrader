"""Phase I layer 2 behavior quality schema, thresholds, and hard gate rules."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from src.utils import _dataclass_from_mapping

if TYPE_CHECKING:
    from .phase1_metric_results import Phase1LayerResult
    from .phase1_validation_data_schema import Phase1ValidationMetrics


@dataclass(frozen=True)
class Phase1BehaviorQualityPayload(Mapping[str, object]):
    """第二层 behavior quality 计算的中间 payload。

    使用场景:
        保存每条样本的 morphology/motif 标签，以及当前 validation split 的
        active code 列表。该对象实现 ``Mapping``，用于兼容现有
        ``extra_payload["..."]`` 调用。
    """

    # 每条 horizon 的市场形态标签。
    morphology_labels: tuple[str, ...]

    # 每条 horizon 的交易 motif 标签。
    motif_labels: tuple[str, ...]

    # 当前 validation split 中满足 active occupancy 阈值的 code id。
    active_codes: tuple[int, ...]

    def __post_init__(self) -> None:
        """标准化 payload 中的标签和 code id 类型。"""

        object.__setattr__(
            self,
            "morphology_labels",
            tuple(str(label) for label in self.morphology_labels),
        )
        object.__setattr__(
            self,
            "motif_labels",
            tuple(str(label) for label in self.motif_labels),
        )
        object.__setattr__(
            self,
            "active_codes",
            tuple(int(code_id) for code_id in self.active_codes),
        )

    def _mapping(self) -> dict[str, object]:
        """返回兼容旧 ``extra_payload`` 字典访问的视图。"""

        return {
            "morphology_labels": self.morphology_labels,
            "motif_labels": self.motif_labels,
            "active_codes": self.active_codes,
        }

    def __getitem__(self, key: str) -> object:
        """按旧 payload key 读取属性值。"""

        return self._mapping()[key]

    def __iter__(self) -> Iterator[str]:
        """迭代旧 payload key。"""

        return iter(self._mapping())

    def __len__(self) -> int:
        """返回 payload key 数量。"""

        return len(self._mapping())

    def to_dict(self) -> dict[str, Any]:
        """序列化为可落盘 dict。"""

        return {
            "morphology_labels": list(self.morphology_labels),
            "motif_labels": list(self.motif_labels),
            "active_codes": [int(code_id) for code_id in self.active_codes],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1BehaviorQualityPayload":
        """从 dict 恢复第二层 behavior quality payload。"""

        return cls(
            morphology_labels=tuple(
                str(label) for label in payload.get("morphology_labels", ())
            ),
            motif_labels=tuple(str(label) for label in payload.get("motif_labels", ())),
            active_codes=tuple(
                int(code_id) for code_id in payload.get("active_codes", ())
            ),
        )


@dataclass(frozen=True)
class Phase1BehaviorQualityMetrics:
    """第二层 archetype 行为质量 raw metrics。"""

    # support 不达标的 active code 比例。
    weak_support_code_ratio: float

    # dominant morphology 或 morphology purity 不达标的 active code 比例。
    weak_morphology_code_ratio: float

    # dominant motif 或 motif purity 不达标的 active code 比例。
    weak_motif_code_ratio: float

    # dominant morphology-motif pair 不达标的 active code 比例。
    weak_pair_code_ratio: float

    # morphology lift 或盈利性辅助判断较弱的 active code 比例。
    weak_lift_nonprofitable_code_ratio: float

    # 同一 code 内 decoded action sequence 的平均相似度。
    intra_code_action_similarity: float

    # code 间动作中心距离 / code 内动作距离。
    inter_intra_separation: float

    # 基于 z_e 和 assigned code 的 latent silhouette score。
    latent_silhouette_score: float

    # decoded action 原型相似度超过阈值的 code pair 数量。
    duplicate_code_pair_count: int

    # 具备第三层 per-code 盈利能力的 active code 覆盖率。
    profitable_code_coverage: float

    # 当前 codebook size K，用于 duplicate code pair 上限按 K 动态判定。
    num_codes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict，供 checkpoint/report 落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1BehaviorQualityMetrics":
        """从 dict 恢复第二层 metrics。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase1BehaviorQualityThresholds:
    """第二层 archetype 行为质量阈值配置。"""

    # 单个 active code 的绝对最小样本数。用于保证 per-code 诊断具备统计支撑。
    min_code_support_abs: int = 100

    # 单个 active code 的相对最小样本比例。实际 support 阈值取 max(绝对阈值, 相对阈值 * N)。
    min_code_support_ratio: float = 0.02

    # support 不足的弱 code 比例上限。设计标准要求超过 20% 时淘汰。
    weak_support_code_ratio_max: float = 0.40

    # morphology/motif/pair/lift 等结构弱 code 比例上限。设计标准要求超过 40% 时淘汰。
    weak_structure_code_ratio_max: float = 0.90

    # 兼容旧配置快照的遗留字段；新规则不再读取该字段。
    weak_code_ratio_max: float = 0.20

    # dominant morphology 占比下限。用于判断 code 是否集中对应某类市场形态。
    dominant_morphology_ratio_min: float = 0.35

    # dominant motif 占比下限。用于判断 code 是否对应清晰的交易行为意图。
    dominant_motif_ratio_min: float = 0.40

    # motif entropy purity 下限。dominant motif 未达标但 purity 达标时仍可视为结构清晰。
    motif_purity_min: float = 0.35

    # dominant morphology-motif pair 占比下限。用于判断市场形态和行为是否形成稳定组合。
    dominant_pair_ratio_min: float = 0.30

    # morphology lift 下限。用于判断某 code 是否真的富集某类市场结构。
    morphology_lift_min: float = 1.25

    # morphology entropy purity 下限。dominant morphology 未达标但 purity 达标时仍可视为结构清晰。
    morphology_purity_min: float = 0.30

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


def evaluate_behavior_quality_rules(
    metrics: Phase1BehaviorQualityMetrics,
    thresholds: Phase1BehaviorQualityThresholds,
) -> Phase1LayerResult:
    """判定第二层 archetype 行为质量。"""

    from .phase1_validation_rule_helpers import _build_layer_result, _ge, _le

    layer = "behavior_quality"
    duplicate_pair_count_max = (
        thresholds.duplicate_code_pair_count_max
        if thresholds.duplicate_code_pair_count_max is not None
        else (int(metrics.num_codes) if metrics.num_codes > 0 else 10)
    )
    results = (
        _le(
            name="weak_support_code_ratio",
            value=metrics.weak_support_code_ratio,
            threshold_value=thresholds.weak_support_code_ratio_max,
            layer=layer,
            message="support 不足的 active code 比例不能过高",
        ),
        _le(
            name="weak_morphology_code_ratio",
            value=metrics.weak_morphology_code_ratio,
            threshold_value=thresholds.weak_structure_code_ratio_max,
            layer=layer,
            message="市场形态不清晰的 active code 比例不能过高",
        ),
        _le(
            name="weak_motif_code_ratio",
            value=metrics.weak_motif_code_ratio,
            threshold_value=thresholds.weak_structure_code_ratio_max,
            layer=layer,
            message="交易 motif 不清晰的 active code 比例不能过高",
        ),
        _le(
            name="weak_pair_code_ratio",
            value=metrics.weak_pair_code_ratio,
            threshold_value=thresholds.weak_structure_code_ratio_max,
            layer=layer,
            message="dominant morphology-motif pair 不清晰的 active code 比例不能过高",
        ),
        _le(
            name="weak_lift_nonprofitable_code_ratio",
            value=metrics.weak_lift_nonprofitable_code_ratio,
            threshold_value=thresholds.weak_structure_code_ratio_max,
            layer=layer,
            message="缺少结构 lift 或盈利性的弱 code 比例不能过高",
        ),
        _ge(
            name="intra_code_action_similarity",
            value=metrics.intra_code_action_similarity,
            threshold_value=thresholds.intra_code_similarity_min,
            layer=layer,
            message="同一 code 内 decoded action 序列需要足够一致",
        ),
        _ge(
            name="inter_intra_separation",
            value=metrics.inter_intra_separation,
            threshold_value=thresholds.inter_intra_separation_min,
            layer=layer,
            message="不同 code 的行为中心需要和 code 内差异拉开距离",
        ),
        _ge(
            name="latent_silhouette_score",
            value=metrics.latent_silhouette_score,
            threshold_value=thresholds.latent_silhouette_score_min,
            layer=layer,
            message="latent 空间中的 assigned code 需要有清晰聚类边界",
        ),
        _le(
            name="duplicate_code_pair_count",
            value=metrics.duplicate_code_pair_count,
            threshold_value=duplicate_pair_count_max,
            layer=layer,
            message="不能存在超过重复相似度阈值的 code pair",
        ),
        _ge(
            name="profitable_code_coverage",
            value=metrics.profitable_code_coverage,
            threshold_value=thresholds.profitable_code_coverage_min,
            layer=layer,
            message="具备盈利潜力的 active code 覆盖率需要足够高",
        ),
    )
    return _build_layer_result(layer_id=2, name=layer, metrics=results)


def compute_behavior_structure_score(metrics: Phase1ValidationMetrics) -> float:
    """计算 behavior structure 子分数。"""

    from .phase1_validation_score_helpers import _clip01, _threshold_progress

    behavior = metrics.behavior_quality
    parts = (
        1.0 - _clip01(behavior.weak_support_code_ratio),
        1.0 - _clip01(behavior.weak_morphology_code_ratio),
        1.0 - _clip01(behavior.weak_motif_code_ratio),
        1.0 - _clip01(behavior.weak_pair_code_ratio),
        1.0 - _clip01(behavior.weak_lift_nonprofitable_code_ratio),
        _threshold_progress(behavior.intra_code_action_similarity, 0.65),
        _threshold_progress(behavior.inter_intra_separation, 1.30),
        _clip01((behavior.latent_silhouette_score + 1.0) / 2.0),
        1.0 / (1.0 + max(0, behavior.duplicate_code_pair_count)),
        _threshold_progress(behavior.profitable_code_coverage, 0.60),
    )
    return sum(parts) / len(parts)


__all__ = [
    "compute_behavior_structure_score",
    "Phase1BehaviorQualityMetrics",
    "Phase1BehaviorQualityPayload",
    "Phase1BehaviorQualityThresholds",
    "evaluate_behavior_quality_rules",
]
