"""Demo 批生成器（含 reject_transition 监控）.

设计文档锚点: §5.3 与 §6.8.1。
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.preprocess_data.config import RejectTransitionHealthConfig
from src.preprocess_data.horizon_builder import HorizonRecord
from src.trading.cost_model import LobDepthCostModel
from src.trading.reward_alignment import RewardAlignment

from .single_trade_dp import DPInputs, DPResult, SingleTradeDPPlanner

_AUTO_PARALLEL_MIN_HORIZONS = 256
_DP_WORKER_PLANNER: SingleTradeDPPlanner | None = None


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
    - horizon 之间相互独立；生产数据可用多进程并行 ``planner.plan``。
    - 输出按输入 index 回填，保证相同输入下 actions/rewards/reject stats deterministic。
    """

    def __init__(
        self,
        planner: SingleTradeDPPlanner,
        health: RejectTransitionHealthConfig,
        worst_top_k: int = 10,
        max_workers: int | None = 1,
        worker_chunksize: int = 32,
        parallel_min_horizons: int = _AUTO_PARALLEL_MIN_HORIZONS,
    ) -> None:
        self.planner = planner
        self.health = health
        self.worst_top_k = worst_top_k
        self.max_workers = max_workers
        self.worker_chunksize = max(1, int(worker_chunksize))
        self.parallel_min_horizons = max(1, int(parallel_min_horizons))

    def generate(
        self,
        horizons: List[HorizonRecord],
    ) -> Tuple[List[HorizonRecord], RejectStats]:
        """对每个 horizon 跑 DP 并汇总 reject 统计。

        Steps
        -----
        1. 串行或多进程调 ``planner.plan(DPInputs)``，得到 ``DPResult``。
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
        for rec in horizons:
            if not rec.execution_books:
                raise ValueError(
                    f"horizon {rec.sample_id} 的 execution_books 为空; "
                    "构建 horizon 时漏掉了盘口列。"
                )

        worker_count = self._resolve_worker_count(len(horizons))
        if worker_count > 1:
            results = self._plan_parallel(horizons, worker_count)
        else:
            results = self._plan_serial(horizons)

        stats = self._apply_results_and_build_stats(horizons, results)

        # health check
        reason = self._evaluate_reject_health(stats)
        if reason and self.health.fail_when_exceeded:
            raise RejectTransitionExceeded(
                f"reject_transition health 检查失败: {reason}; "
                f"dataset_reject_rate={stats.dataset_reject_rate:.3f}, "
                f"max_per_horizon={max(stats.per_horizon_reject_rate or [0]):.3f}"
            )
        return horizons, stats

    def _resolve_worker_count(self, num_horizons: int) -> int:
        requested = 1 if self.max_workers is None else int(self.max_workers)
        if requested <= 0:
            cpu_count = os.cpu_count() or 1
            if num_horizons < self.parallel_min_horizons or cpu_count <= 1:
                return 1
            return max(1, min(cpu_count, num_horizons))
        return max(1, min(requested, num_horizons))

    def _plan_serial(
        self, horizons: Sequence[HorizonRecord]
    ) -> List[Tuple[int, DPResult]]:
        results: List[Tuple[int, DPResult]] = []
        for idx, rec in enumerate(horizons):
            inputs = DPInputs(
                prices=list(rec.prices),
                execution_books=list(rec.execution_books),
                horizon=len(rec.execution_books),
            )
            result: DPResult = self.planner.plan(inputs)
            results.append((idx, result))
        return results

    def _plan_parallel(
        self,
        horizons: Sequence[HorizonRecord],
        worker_count: int,
    ) -> List[Tuple[int, DPResult]]:
        if type(self.planner.cost_model) is not LobDepthCostModel:
            return self._plan_serial(horizons)
        payloads = (
            (idx, list(rec.prices), list(rec.execution_books))
            for idx, rec in enumerate(horizons)
        )
        initargs = (
            _cost_model_payload(self.planner.cost_model),
            self.planner.reward_alignment.mode,
            self.planner.max_position,
            self.planner.gamma,
        )
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_dp_worker,
            initargs=initargs,
        ) as executor:
            return list(
                executor.map(
                    _plan_horizon_in_worker,
                    payloads,
                    chunksize=self.worker_chunksize,
                )
            )

    def _apply_results_and_build_stats(
        self,
        horizons: Sequence[HorizonRecord],
        indexed_results: Sequence[Tuple[int, DPResult]],
    ) -> RejectStats:
        total_evaluated_transitions = 0
        total_rejected_transitions = 0
        per_horizon_reject_count: List[int] = []
        per_horizon_reject_rate: List[float] = []
        action_pair_counter: Counter = Counter()
        per_horizon_meta: List[dict] = []

        for idx, result in sorted(indexed_results, key=lambda item: item[0]):
            rec = horizons[idx]
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
        return stats

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


def _cost_model_payload(cost_model: LobDepthCostModel) -> Dict[str, Any]:
    return {
        "commission_rate": cost_model.commission_rate,
        "book_levels": cost_model.book_levels,
        "insufficient_depth_policy": cost_model.insufficient_depth_policy,
        "slippage_multiplier": cost_model.slippage_multiplier,
    }


def _init_dp_worker(
    cost_payload: Dict[str, Any],
    reward_alignment_mode: str,
    max_position: int,
    gamma: float,
) -> None:
    global _DP_WORKER_PLANNER
    cost_model = LobDepthCostModel(**cost_payload)
    _DP_WORKER_PLANNER = SingleTradeDPPlanner(
        cost_model=cost_model,
        reward_alignment=RewardAlignment(reward_alignment_mode),
        max_position=max_position,
        gamma=gamma,
    )


def _plan_horizon_in_worker(
    payload: Tuple[int, List[float], List[Any]],
) -> Tuple[int, DPResult]:
    if _DP_WORKER_PLANNER is None:
        raise RuntimeError("DP worker planner 未初始化")
    idx, prices, execution_books = payload
    inputs = DPInputs(
        prices=prices,
        execution_books=execution_books,
        horizon=len(execution_books),
    )
    return idx, _DP_WORKER_PLANNER.plan(inputs)
