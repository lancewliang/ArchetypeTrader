"""Demo 批生成器（含 reject_transition 监控）.

设计文档锚点: §5.3 与 §6.8.1。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.config.phase1_config import RejectTransitionHealthConfig
from src.data.horizon_builder import HorizonRecord

from .single_trade_dp import DPInputs, DPResult, SingleTradeDPPlanner


class RejectTransitionExceeded(RuntimeError):
    """``dataset_reject_rate`` / ``per_horizon_reject_rate`` 超阈时抛出。"""


@dataclass
class RejectStats:
    dataset_reject_rate: float = 0.0
    per_horizon_reject_count: List[int] = field(default_factory=list)
    per_horizon_reject_rate: List[float] = field(default_factory=list)
    worst_reject_horizons: List[dict] = field(default_factory=list)
    reject_by_action_pair: Dict[str, int] = field(default_factory=dict)


class Phase1DemoGenerator:
    """批生成 demonstration。

    使用方式::

        gen = Phase1DemoGenerator(planner, health=RejectTransitionHealthConfig(...))
        horizons, stats = gen.generate(sampled_horizons)

    并行性:
    - horizon 之间相互独立；当前实现是顺序的（便于调试与 deterministic）。
      实际生产可以替换为 ``joblib.Parallel`` 并行 ``planner.plan``，
      但必须固定 seed、固定输出顺序。
    """

    def __init__(
        self,
        planner: SingleTradeDPPlanner,
        health: RejectTransitionHealthConfig,
        worst_top_k: int = 10,
    ) -> None:
        self.planner = planner
        self.health = health
        self.worst_top_k = worst_top_k

    def generate(
        self,
        horizons: List[HorizonRecord],
    ) -> Tuple[List[HorizonRecord], RejectStats]:
        """对每个 horizon 跑 DP 并汇总 reject 统计。

        Steps
        -----
        1. 顺序调 ``planner.plan(DPInputs)``，得到 ``DPResult``。
        2. 把 ``actions``、``rewards`` 写回 ``HorizonRecord``（in-place）。
        3. 累加每个 horizon 的 reject 计数 → ``RejectStats``。
        4. 触发 ``RejectTransitionExceeded`` 或返回。

        Returns
        -------
        ``(horizons, stats)`` : 输入列表（已填好 actions/rewards）+ reject 统计。

        Raises
        ------
        RejectTransitionExceeded : 任一 horizon 的 ``per_horizon_reject_rate`` 或
                                   全局 ``dataset_reject_rate`` 超阈，
                                   且 ``health.fail_when_exceeded=True``。
        ValueError : 任一 horizon 的 ``execution_books`` 为空（HorizonBuilder 漏配）。
        """
        if not horizons:
            return [], RejectStats()

        total_evaluated_transitions = 0
        total_rejected_transitions = 0
        per_horizon_reject_count: List[int] = []
        per_horizon_reject_rate: List[float] = []
        action_pair_counter: Counter = Counter()
        per_horizon_meta: List[dict] = []

        for rec in horizons:
            if not rec.execution_books:
                raise ValueError(
                    f"horizon {rec.sample_id} 的 execution_books 为空; "
                    "构建 horizon 时漏掉了盘口列。"
                )
            inputs = DPInputs(
                prices=list(rec.prices),
                execution_books=list(rec.execution_books),
                horizon=len(rec.execution_books),
            )
            result: DPResult = self.planner.plan(inputs)

            # 把 actions / rewards 写回 record
            rec.actions = list(result.actions)
            rec.rewards = list(result.rewards)

            evaluated_per_horizon = result.precompute_evaluated_count
            rejected = result.precompute_rejected_count
            total_evaluated_transitions += evaluated_per_horizon
            total_rejected_transitions += rejected
            per_horizon_reject_count.append(rejected)
            rate = rejected / max(evaluated_per_horizon, 1)
            per_horizon_reject_rate.append(rate)

            for key, count in (result.precompute_rejected_by_pair or {}).items():
                action_pair_counter[key] += count

            per_horizon_meta.append(
                {
                    "sample_id": rec.sample_id,
                    "rate": rate,
                    "rejected": rejected,
                    "window_start": rec.start_index,
                    "strata": rec.strata_label,
                }
            )

        stats = RejectStats()
        if total_evaluated_transitions > 0:
            stats.dataset_reject_rate = total_rejected_transitions / total_evaluated_transitions
        stats.per_horizon_reject_count = per_horizon_reject_count
        stats.per_horizon_reject_rate = per_horizon_reject_rate
        stats.reject_by_action_pair = dict(action_pair_counter)
        stats.worst_reject_horizons = sorted(
            per_horizon_meta, key=lambda x: -x["rate"]
        )[: self.worst_top_k]

        # health check
        reason = self._evaluate_reject_health(stats)
        if reason and self.health.fail_when_exceeded:
            raise RejectTransitionExceeded(
                f"reject_transition health 检查失败: {reason}; "
                f"dataset_reject_rate={stats.dataset_reject_rate:.3f}, "
                f"max_per_horizon={max(stats.per_horizon_reject_rate or [0]):.3f}"
            )
        return horizons, stats

    def _evaluate_reject_health(self, stats: RejectStats) -> Optional[str]:
        """判断是否超阈，返回原因字符串供异常使用。

        - ``dataset_reject_rate > max_dataset_reject_rate`` → ``"dataset_exceeded"``。
        - 任一 horizon ``per_horizon_reject_rate > max_horizon_reject_rate``
          → ``"horizon_exceeded"``。
        - 否则返回 ``None``。

        ``health.fail_when_exceeded=False`` 时即使返回非 None，也只在
        ``phase1_report.json`` 写 warning，不抛异常。
        """
        if stats.dataset_reject_rate > self.health.max_dataset_reject_rate:
            return "dataset_exceeded"
        if stats.per_horizon_reject_rate:
            max_rate = max(stats.per_horizon_reject_rate)
            if max_rate > self.health.max_horizon_reject_rate:
                return "horizon_exceeded"
        return None
