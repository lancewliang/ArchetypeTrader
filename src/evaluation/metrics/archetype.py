"""Per-archetype 诊断 + DP teacher 质量指标."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .risk import sharpe_ratio


@dataclass
class PerCodeStats:
    code_id: int
    count: int
    avg_return: float
    win_rate: float
    no_trade_ratio: float
    switch_point_distribution: Dict[str, float]


@dataclass
class ArchetypeDiagnostics:
    per_code: List[PerCodeStats] = field(default_factory=list)
    no_trade_code_concentration: Dict[str, float] = field(default_factory=dict)
    active_trade_code_count: int = 0


def per_code_summary(
    horizon_returns: Sequence[float],
    code_ids: Sequence[int],
    no_trade_flags: Sequence[bool],
    switch_points: Sequence[int],
    no_trade_code_ratio_threshold: float = 0.8,
) -> ArchetypeDiagnostics:
    """按 code 分组计算指标。

    实现要点
    --------
    - 每个 code 汇总样本数、平均收益、胜率、no-trade 占比、切换点四等分布。
    - ``no_trade_code_concentration``: no-trade 样本在 top-1 / top-2 code 上的集中度，
      用于检测"少数 code 退化为 no-trade archetype"问题（设计 §5.4 no-trade 容量监控）。
    - ``active_trade_code_count``: ``no_trade_ratio < no_trade_code_ratio_threshold``
      的 code 数；过低提示 no-trade 占据 codebook 容量。

    Parameters
    ----------
    horizon_returns : 每个 horizon 的 student 净收益（也可传 demo_return 做 hindsight 对比）。
    code_ids : 与 ``horizon_returns`` 顺序一致的 archetype id。
    no_trade_flags : True 表示该 horizon 全程 flat。
    switch_points : 切换发生的 timestep；-1 表示无切换。
    """
    by_code: Dict[int, dict] = defaultdict(lambda: {
        "returns": [], "no_trades": [], "switch_points": []
    })
    for ret, cid, nt, sw in zip(horizon_returns, code_ids, no_trade_flags, switch_points):
        bucket = by_code[int(cid)]
        bucket["returns"].append(float(ret))
        bucket["no_trades"].append(bool(nt))
        bucket["switch_points"].append(int(sw))

    per_code: List[PerCodeStats] = []
    for cid in sorted(by_code):
        rets = by_code[cid]["returns"]
        nts = by_code[cid]["no_trades"]
        sps = by_code[cid]["switch_points"]
        avg = sum(rets) / max(len(rets), 1)
        win_rate = sum(1 for r in rets if r > 0) / max(len(rets), 1)
        no_trade_ratio = sum(1 for n in nts if n) / max(len(nts), 1)
        # switch point 分桶到 horizon 的几等分
        if sps:
            quartile_counts = [0, 0, 0, 0]
            max_h = max([s for s in sps if s >= 0], default=0)
            for s in sps:
                if s < 0:
                    continue
                # 0 ~ max_h 分四等
                bucket = min(3, int(s / max(max_h / 4, 1)))
                quartile_counts[bucket] += 1
            total_sw = sum(quartile_counts)
            distribution = {
                f"q{i+1}": quartile_counts[i] / max(total_sw, 1) for i in range(4)
            }
        else:
            distribution = {f"q{i+1}": 0.0 for i in range(4)}

        per_code.append(
            PerCodeStats(
                code_id=cid,
                count=len(rets),
                avg_return=avg,
                win_rate=win_rate,
                no_trade_ratio=no_trade_ratio,
                switch_point_distribution=distribution,
            )
        )

    # no-trade 集中度
    sorted_by_no_trade_share = sorted(
        per_code,
        key=lambda s: -(s.no_trade_ratio * s.count),
    )
    total_no_trade = sum(s.no_trade_ratio * s.count for s in per_code)
    if total_no_trade > 0:
        top1 = (sorted_by_no_trade_share[0].no_trade_ratio * sorted_by_no_trade_share[0].count) / total_no_trade
        top2 = top1
        if len(sorted_by_no_trade_share) >= 2:
            top2 += (sorted_by_no_trade_share[1].no_trade_ratio * sorted_by_no_trade_share[1].count) / total_no_trade
    else:
        top1 = 0.0
        top2 = 0.0
    concentration = {"top1": top1, "top2": top2}

    active = sum(
        1 for s in per_code
        if s.no_trade_ratio < no_trade_code_ratio_threshold and s.count > 0
    )
    return ArchetypeDiagnostics(
        per_code=per_code,
        no_trade_code_concentration=concentration,
        active_trade_code_count=active,
    )


@dataclass
class TeacherQuality:
    teacher_val_dp_teacher_sharpe: float
    teacher_val_dp_teacher_profitable_ratio: float
    return_distribution: Dict[str, float]


def dp_teacher_quality(
    dp_horizon_returns: Sequence[float],
    dp_step_returns: Sequence[float],
    annualization_factor: int = 525_600,
) -> TeacherQuality:
    """DP teacher 在 validation 上的质量。

    用途
    ----
    - 当 ``teacher_val_dp_teacher_profitable_ratio`` 过低时，``teacher_val_return_capture_ratio``
      不可单独解读为学生学得好（设计 §9.5 / §10）。
    - ``return_distribution`` 含 mean/min/p25/p50/p75/max，便于 audit 老师收益分布。
    """
    horizons = list(dp_horizon_returns)
    profitable = sum(1 for r in horizons if r > 0)
    profitable_ratio = profitable / max(len(horizons), 1)
    sharpe = sharpe_ratio(dp_step_returns, annualization_factor=annualization_factor)
    if horizons:
        sorted_h = sorted(horizons)
        n = len(sorted_h)
        distribution = {
            "mean": sum(horizons) / n,
            "min": float(sorted_h[0]),
            "p25": float(sorted_h[max(0, int(0.25 * (n - 1)))]),
            "p50": float(sorted_h[max(0, int(0.5 * (n - 1)))]),
            "p75": float(sorted_h[max(0, int(0.75 * (n - 1)))]),
            "max": float(sorted_h[-1]),
        }
    else:
        distribution = {"mean": 0.0, "min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    return TeacherQuality(
        teacher_val_dp_teacher_sharpe=sharpe,
        teacher_val_dp_teacher_profitable_ratio=profitable_ratio,
        return_distribution=distribution,
    )
