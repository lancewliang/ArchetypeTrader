"""Phase II 评估编排器: val 快速评估 / 完整 walk-forward / rolling validation。

设计文档锚点: Phase II 执行计划 §Step 6。

职责:
- 支持 val 快速评估、完整 walk-forward 评估与 rolling validation 评估。
- train/val/test 的 walk-forward 均按时间正序、仓位连续执行。
- rolling validation 必须固定 fold 切法与种子。
- 产出 phase2_rolling_validation.json 与 phase2_rolling_validation_records.feather。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config.phase2_config import Phase2Config
from src.evaluation.phase2_metrics import phase2_composite_metrics
from src.evaluation.phase2_replay import Phase2BacktestRunner, Phase2HorizonReplayRecord


@dataclass
class Phase2EvalResult:
    """评估结果。"""
    metrics: Dict[str, Any] = field(default_factory=dict)
    per_horizon_records: List[Phase2HorizonReplayRecord] = field(default_factory=list)
    baseline_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RollingValidationResult:
    """Rolling validation 结果。"""
    fold_metrics: List[Dict[str, Any]] = field(default_factory=list)
    fold_mean: Dict[str, float] = field(default_factory=dict)
    worst_fold_quantile: Dict[str, float] = field(default_factory=dict)
    fold_volatility: Dict[str, float] = field(default_factory=dict)
    per_fold_records: List[List[Phase2HorizonReplayRecord]] = field(default_factory=list)


class Phase2Evaluator:
    """Phase II 评估编排器。

    使用方式::

        evaluator = Phase2Evaluator(config, backtest_runner, num_codes, dead_code_mask)
        result = evaluator.evaluate_val_fast(selector, update_idx)
        result = evaluator.evaluate_walk_forward(selector, split="val")
        rolling = evaluator.evaluate_rolling_validation(selector)
    """

    def __init__(
        self,
        config: Phase2Config,
        backtest_runner: Phase2BacktestRunner,
        num_codes: int = 10,
        dead_code_mask: Optional[List[bool]] = None,
    ) -> None:
        self.config = config
        self.backtest_runner = backtest_runner
        self.num_codes = num_codes
        self.dead_code_mask = dead_code_mask or [False] * num_codes

    def evaluate_val_fast(
        self,
        update_idx: int,
        ppo_stats: Optional[Dict[str, float]] = None,
    ) -> Phase2EvalResult:
        """Val 快速评估（子集）。"""
        records = self.backtest_runner.run_walk_forward(
            split="val", deterministic=True
        )
        horizon_dicts = [self._record_to_dict(r) for r in records]
        metrics = phase2_composite_metrics(
            horizon_dicts,
            ppo_stats or {},
            self.num_codes,
            self.dead_code_mask,
        )
        metrics["update_idx"] = update_idx

        warnings: List[str] = []
        if metrics.get("action_dominance_ratio", 0) > self.config.selection_policy.max_action_dominance_ratio:
            warnings.append(
                f"action_dominance_ratio={metrics['action_dominance_ratio']:.3f} 过高"
            )
        if metrics.get("active_archetype_ratio", 1) < self.config.selection_policy.min_active_archetype_ratio:
            warnings.append(
                f"active_archetype_ratio={metrics['active_archetype_ratio']:.3f} 过低"
            )

        return Phase2EvalResult(
            metrics=metrics,
            per_horizon_records=records,
            warnings=warnings,
        )

    def evaluate_walk_forward(
        self,
        split: str,
        ppo_stats: Optional[Dict[str, float]] = None,
    ) -> Phase2EvalResult:
        """完整 walk-forward 评估。

        Parameters
        ----------
        split : "train" / "val" / "test"。
        ppo_stats : PPO update 统计（可选）。

        Returns
        -------
        Phase2EvalResult : 包含 metrics / per_horizon_records / baselines。
        """
        records = self.backtest_runner.run_walk_forward(
            split=split, deterministic=True
        )
        horizon_dicts = [self._record_to_dict(r) for r in records]
        metrics = phase2_composite_metrics(
            horizon_dicts,
            ppo_stats or {},
            self.num_codes,
            self.dead_code_mask,
        )

        # 运行 baselines
        baseline_records = self.backtest_runner.run_baselines(split=split)
        baseline_results: Dict[str, Dict[str, Any]] = {}
        for name, bl_records in baseline_records.items():
            bl_dicts = [self._record_to_dict(r) for r in bl_records]
            bl_metrics = phase2_composite_metrics(
                bl_dicts, {}, self.num_codes, self.dead_code_mask
            )
            baseline_results[name] = bl_metrics

        return Phase2EvalResult(
            metrics=metrics,
            per_horizon_records=records,
            baseline_results=baseline_results,
        )

    def evaluate_rolling_validation(self) -> RollingValidationResult:
        """Rolling validation 评估。

        固定 fold 切法与种子，产出 fold 均值、最差分位、波动。

        Returns
        -------
        RollingValidationResult : rolling validation 结果。
        """
        if not self.config.rolling_validation.enabled:
            return RollingValidationResult()

        num_folds = self.config.rolling_validation.num_folds
        val_entries = [
            e for e in self.backtest_runner.dataset.horizon_entries
            if e.split == "val"
        ]
        if not val_entries or num_folds < 1:
            return RollingValidationResult()

        fold_size = max(len(val_entries) // num_folds, 1)
        fold_metrics_list: List[Dict[str, Any]] = []
        per_fold_records: List[List[Phase2HorizonReplayRecord]] = []

        for fold_idx in range(num_folds):
            # 每个 fold 使用 val 的一个子集
            start = fold_idx * fold_size
            end = min((fold_idx + 1) * fold_size, len(val_entries))
            # 运行 walk-forward on this fold subset
            records = self.backtest_runner.run_walk_forward(
                split="val", deterministic=True
            )
            # 只取对应 fold 的 records
            fold_records = records[start:end] if len(records) > start else []
            per_fold_records.append(fold_records)

            horizon_dicts = [self._record_to_dict(r) for r in fold_records]
            metrics = phase2_composite_metrics(
                horizon_dicts, {}, self.num_codes, self.dead_code_mask
            )
            fold_metrics_list.append(metrics)

        # 聚合
        fold_mean = self._aggregate_fold_metrics(fold_metrics_list, "mean")
        worst_fold = self._aggregate_fold_metrics(fold_metrics_list, "min")
        fold_vol = self._aggregate_fold_metrics(fold_metrics_list, "std")

        return RollingValidationResult(
            fold_metrics=fold_metrics_list,
            fold_mean=fold_mean,
            worst_fold_quantile=worst_fold,
            fold_volatility=fold_vol,
            per_fold_records=per_fold_records,
        )

    @staticmethod
    def _record_to_dict(r: Phase2HorizonReplayRecord) -> Dict[str, Any]:
        """将 replay record 转为 dict。"""
        return {
            "sample_id": r.sample_id,
            "env_id": r.env_id,
            "chosen_code": r.chosen_code,
            "final_position": r.final_position,
            "reward_raw": r.reward_raw,
            "reward_scaled": r.reward_scaled,
            "boundary_cost": r.boundary_cost,
            "cost_paid": r.cost_paid,
            "risk_triggered": r.risk_triggered,
            "step_returns": r.step_returns,
        }

    @staticmethod
    def _aggregate_fold_metrics(
        fold_metrics: List[Dict[str, Any]],
        mode: str,
    ) -> Dict[str, float]:
        """聚合 fold metrics。"""
        import math
        if not fold_metrics:
            return {}

        numeric_keys = [
            k for k in fold_metrics[0]
            if isinstance(fold_metrics[0][k], (int, float))
        ]
        result: Dict[str, float] = {}
        for key in numeric_keys:
            values = [float(fm[key]) for fm in fold_metrics if key in fm]
            if not values:
                continue
            if mode == "mean":
                result[key] = sum(values) / len(values)
            elif mode == "min":
                result[key] = min(values)
            elif mode == "std":
                mean = sum(values) / len(values)
                var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
                result[key] = math.sqrt(max(var, 0.0))
        return result
