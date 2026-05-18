"""Phase I layer 3 oracle profitability schema, thresholds, and hard gate rules."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from src.utils import _dataclass_from_mapping

if TYPE_CHECKING:
    from .phase1_metric_results import Phase1LayerResult
    from .phase1_validation_data_schema import (
        Phase1PerCodeProfitability,
        Phase1ValidationMetrics,
    )


@dataclass(frozen=True)
class Phase1PairProfitabilityCell:
    """单个 morphology-motif pair 的 oracle 盈利性摘要。"""

    # 市场形态标签，例如 uptrend、downtrend、range-high-vol。
    morphology: str

    # 行为 motif 标签，例如 long-hold、short-hold、flat。
    motif: str

    # validation 中落入该 pair 的 horizon 数量。
    support: int

    # 该 pair 上 decoded return 相对 flat baseline 的平均优势。
    mean_decoded_advantage: float

    # 该 pair 上 decoded return 超过 flat baseline 的比例。
    decoded_win_rate: float

    # 该 pair 上 decoded advantage 相对 DP advantage 的保留比例。
    retention_ratio: float

    # 该 pair 上 total fee / gross profit。
    fee_drag: float

    def to_dict(self) -> dict[str, Any]:
        """序列化为 report/checkpoint 可保存的 dict。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1PairProfitabilityCell":
        """从 dict 恢复 pair profitability cell。"""

        return cls(
            morphology=str(payload["morphology"]),
            motif=str(payload["motif"]),
            support=int(payload["support"]),
            mean_decoded_advantage=float(payload["mean_decoded_advantage"]),
            decoded_win_rate=float(payload["decoded_win_rate"]),
            retention_ratio=float(payload["retention_ratio"]),
            fee_drag=float(payload["fee_drag"]),
        )


@dataclass(frozen=True)
class Phase1OracleProfitabilityPayload(Mapping[str, object]):
    """第三层 oracle profitability 计算的中间 payload。

    使用场景:
        保存 per-code 盈利性摘要、decoded/DP/flat/random label returns 以及
        random seed。该对象实现 ``Mapping``，用于兼容现有
        ``extra_payload["..."]`` 调用。
    """

    # Layer 3 输出的 per-code 盈利性摘要，供 Layer 2 复用。
    per_code_profitability: tuple["Phase1PerCodeProfitability", ...]

    # 每条 horizon 的 assigned-label decoded return。
    decoded_returns: tuple[float, ...]

    # 每条 horizon 的 DP teacher return。
    dp_returns: tuple[float, ...]

    # 每条 horizon 的 flat baseline return。
    flat_returns: tuple[float, ...]

    # 每条 horizon 的 random-label decoded return。
    random_label_returns: tuple[float, ...]

    # random label baseline 使用的随机种子。
    random_seed: int

    # morphology x motif 的 oracle decoded profitability 矩阵 cell。
    pair_profitability_matrix: tuple[Phase1PairProfitabilityCell, ...] = ()

    def __post_init__(self) -> None:
        """标准化 payload 中的序列和标量类型。"""

        pair_cells = []
        for item in self.pair_profitability_matrix:
            if isinstance(item, Phase1PairProfitabilityCell):
                pair_cells.append(item)
            else:
                pair_cells.append(Phase1PairProfitabilityCell.from_dict(item))
        object.__setattr__(
            self,
            "pair_profitability_matrix",
            tuple(pair_cells),
        )
        object.__setattr__(
            self,
            "per_code_profitability",
            tuple(self.per_code_profitability),
        )
        object.__setattr__(
            self,
            "decoded_returns",
            tuple(float(value) for value in self.decoded_returns),
        )
        object.__setattr__(
            self,
            "dp_returns",
            tuple(float(value) for value in self.dp_returns),
        )
        object.__setattr__(
            self,
            "flat_returns",
            tuple(float(value) for value in self.flat_returns),
        )
        object.__setattr__(
            self,
            "random_label_returns",
            tuple(float(value) for value in self.random_label_returns),
        )
        object.__setattr__(self, "random_seed", int(self.random_seed))

    def _mapping(self) -> dict[str, object]:
        """返回兼容旧 ``extra_payload`` 字典访问的视图。"""

        return {
            "per_code_profitability": self.per_code_profitability,
            "decoded_returns": self.decoded_returns,
            "dp_returns": self.dp_returns,
            "flat_returns": self.flat_returns,
            "random_label_returns": self.random_label_returns,
            "random_seed": self.random_seed,
            "pair_profitability_matrix": self.pair_profitability_matrix,
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
            "per_code_profitability": [
                item.to_dict() for item in self.per_code_profitability
            ],
            "decoded_returns": [float(value) for value in self.decoded_returns],
            "dp_returns": [float(value) for value in self.dp_returns],
            "flat_returns": [float(value) for value in self.flat_returns],
            "random_label_returns": [
                float(value) for value in self.random_label_returns
            ],
            "random_seed": self.random_seed,
            "pair_profitability_matrix": [
                item.to_dict() for item in self.pair_profitability_matrix
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1OracleProfitabilityPayload":
        """从 dict 恢复第三层 oracle profitability payload。"""

        from .phase1_validation_data_schema import Phase1PerCodeProfitability

        per_code = []
        for item in payload.get("per_code_profitability", ()):
            if isinstance(item, Phase1PerCodeProfitability):
                per_code.append(item)
            else:
                per_code.append(Phase1PerCodeProfitability.from_dict(item))
        return cls(
            per_code_profitability=tuple(per_code),
            decoded_returns=tuple(
                float(value) for value in payload.get("decoded_returns", ())
            ),
            dp_returns=tuple(float(value) for value in payload.get("dp_returns", ())),
            flat_returns=tuple(
                float(value) for value in payload.get("flat_returns", ())
            ),
            random_label_returns=tuple(
                float(value) for value in payload.get("random_label_returns", ())
            ),
            random_seed=int(payload["random_seed"]),
            pair_profitability_matrix=tuple(
                item
                if isinstance(item, Phase1PairProfitabilityCell)
                else Phase1PairProfitabilityCell.from_dict(item)
                for item in payload.get("pair_profitability_matrix", ())
            ),
        )


@dataclass(frozen=True)
class Phase1OracleProfitabilityMetrics:
    """第三层 oracle assigned-label 盈利性 raw metrics。"""

    # decoded return 相对 flat baseline 的平均优势。
    mean_decoded_advantage_vs_flat: float

    # decoded return 大于 flat baseline 的样本比例。
    decoded_win_rate_vs_flat: float

    # assigned label decoded return 相对 random label decoded return 的平均优势。
    mean_advantage_vs_random_label: float

    # assigned label 相对 random label 的收益提升比例。
    random_label_relative_lift: float

    # decoded 总优势 / DP teacher 总优势。
    retention_ratio: float

    # decoded 策略累计收益曲线的最大回撤。
    downside_control: float

    # decoded return 的风险调整收益指标。
    risk_adjusted_return: float

    # 收益最高 5% horizon 对总 decoded profit 的贡献比例。
    top_5_contribution: float

    # 去掉两端尾部样本后的 decoded mean advantage。
    trimmed_decoded_advantage: float

    # total fee / gross profit。
    fee_drag: float

    # horizon turnover 与 decoded return 的相关性。
    turnover_return_correlation: float

    # per-code mean advantage 为负的 active code 比例。
    bad_code_ratio: float

    # dominant pair 中 decoded advantage 为正的比例。
    dominant_pair_positive_ratio: float

    # random label baseline 的风险调整收益。
    random_label_risk_adjusted_return: float = float("nan")

    # decoded 风险调整收益相对 random label baseline 的差值。
    risk_adjusted_return_vs_random: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict，供 checkpoint/report 落盘。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase1OracleProfitabilityMetrics":
        """从 dict 恢复第三层 metrics。"""

        return _dataclass_from_mapping(cls, payload)


@dataclass(frozen=True)
class Phase1OracleProfitabilityThresholds:
    """第三层 oracle assigned-label 盈利性阈值配置。"""

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

    # decoded 策略累计收益曲线的最大回撤上限。
    downside_control_max: float = 20.00

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


def evaluate_oracle_profitability_rules(
    metrics: Phase1OracleProfitabilityMetrics,
    thresholds: Phase1OracleProfitabilityThresholds,
) -> Phase1LayerResult:
    """判定第三层 oracle assigned-label 盈利性。"""

    from .phase1_validation_rule_helpers import _build_layer_result, _ge, _gt, _le

    layer = "oracle_profitability"
    results = (
        _gt(
            name="mean_decoded_advantage_vs_flat",
            value=metrics.mean_decoded_advantage_vs_flat,
            threshold_value=0.0,
            layer=layer,
            message="assigned-label decoded 策略平均收益必须优于 flat baseline",
        ),
        _ge(
            name="decoded_win_rate_vs_flat",
            value=metrics.decoded_win_rate_vs_flat,
            threshold_value=thresholds.decoded_win_rate_min,
            layer=layer,
            message="decoded 策略胜率需要足够广泛",
        ),
        _gt(
            name="mean_advantage_vs_random_label",
            value=metrics.mean_advantage_vs_random_label,
            threshold_value=0.0,
            layer=layer,
            message="assigned label decoded 表现必须优于 random label baseline",
        ),
        _ge(
            name="random_label_relative_lift",
            value=metrics.random_label_relative_lift,
            threshold_value=thresholds.random_label_relative_lift_min,
            layer=layer,
            message="assigned label 相对 random label 的收益提升需要足够高",
        ),
        _ge(
            name="retention_ratio",
            value=metrics.retention_ratio,
            threshold_value=thresholds.retention_ratio_min,
            layer=layer,
            message="decoded 策略需要保留足够 DP teacher 盈利能力",
        ),
        _le(
            name="downside_control",
            value=metrics.downside_control,
            threshold_value=thresholds.downside_control_max,
            layer=layer,
            message="decoded 策略累计收益曲线的最大回撤不能过高",
        ),
        _gt(
            name="risk_adjusted_return",
            value=metrics.risk_adjusted_return,
            threshold_value=thresholds.risk_adjusted_return_min,
            layer=layer,
            message="decoded 策略风险调整收益必须为正",
        ),
        _gt(
            name="risk_adjusted_return_vs_random",
            value=metrics.risk_adjusted_return_vs_random,
            threshold_value=0.0,
            layer=layer,
            message="decoded 策略风险调整收益必须优于 random label baseline",
        ),
        _le(
            name="top_5_contribution",
            value=metrics.top_5_contribution,
            threshold_value=thresholds.top_5_contribution_max,
            layer=layer,
            message="收益不能过度依赖收益最高 top 5% horizon",
        ),
        _gt(
            name="trimmed_decoded_advantage",
            value=metrics.trimmed_decoded_advantage,
            threshold_value=0.0,
            layer=layer,
            message="去除尾部样本后 decoded 策略仍应有正优势",
        ),
        _le(
            name="fee_drag",
            value=metrics.fee_drag,
            threshold_value=thresholds.fee_drag_max,
            layer=layer,
            message="手续费拖累比例不能过高",
        ),
        _ge(
            name="turnover_return_correlation",
            value=metrics.turnover_return_correlation,
            threshold_value=thresholds.turnover_return_correlation_min,
            layer=layer,
            message="换手与收益的相关性不能显著为负",
        ),
        _le(
            name="bad_code_ratio",
            value=metrics.bad_code_ratio,
            threshold_value=thresholds.bad_code_ratio_max,
            layer=layer,
            message="负价值 code 比例不能过高",
        ),
        _ge(
            name="dominant_pair_positive_ratio",
            value=metrics.dominant_pair_positive_ratio,
            threshold_value=thresholds.dominant_pair_positive_ratio_min,
            layer=layer,
            message="dominant morphology-motif pair 中正优势比例需要足够高",
        ),
    )
    return _build_layer_result(layer_id=3, name=layer, metrics=results)


def compute_oracle_profitability_score(metrics: Phase1ValidationMetrics) -> float:
    """计算 oracle profitability 子分数。"""

    from .phase1_validation_score_helpers import (
        _clip01,
        _inverse_ratio_score,
        _positive_score,
        _threshold_progress,
    )

    oracle = metrics.oracle_profitability
    parts = (
        _positive_score(oracle.mean_decoded_advantage_vs_flat),
        _threshold_progress(oracle.decoded_win_rate_vs_flat, 0.55),
        _positive_score(oracle.mean_advantage_vs_random_label),
        _threshold_progress(oracle.random_label_relative_lift, 0.20),
        _threshold_progress(oracle.retention_ratio, 0.50),
        _positive_score(oracle.risk_adjusted_return),
        _positive_score(oracle.risk_adjusted_return_vs_random),
        _inverse_ratio_score(oracle.top_5_contribution, 0.60),
        _positive_score(oracle.trimmed_decoded_advantage),
        1.0 - _clip01(oracle.fee_drag),
        _clip01((oracle.turnover_return_correlation + 1.0) / 2.0),
        1.0 - _clip01(oracle.bad_code_ratio),
        _threshold_progress(oracle.dominant_pair_positive_ratio, 0.60),
    )
    return sum(parts) / len(parts)


__all__ = [
    "compute_oracle_profitability_score",
    "Phase1OracleProfitabilityMetrics",
    "Phase1OracleProfitabilityPayload",
    "Phase1OracleProfitabilityThresholds",
    "Phase1PairProfitabilityCell",
    "evaluate_oracle_profitability_rules",
]
