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
import math

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
    fold_sizes: List[int] = field(default_factory=list)
    fold_initial_position_policy: str = "flat"
    fold_initial_positions: List[int] = field(default_factory=list)


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
        entry_indices = self._fast_eval_entry_indices("val")
        records = self.backtest_runner.run_walk_forward(
            split="val", deterministic=True, entry_indices=entry_indices
        )
        horizon_dicts = [self._record_to_dict(r) for r in records]
        metrics = phase2_composite_metrics(
            horizon_dicts,
            ppo_stats or {},
            self.num_codes,
            self.dead_code_mask,
            metric_weights=self.config.selection_policy.metric_weights,
        )
        metrics.update(
            self.compute_selector_diagnostics(
                split="val",
                records=records,
                entry_indices=entry_indices,
            )
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
            metric_weights=self.config.selection_policy.metric_weights,
        )
        metrics.update(
            self.compute_selector_diagnostics(
                split=split,
                records=records,
            )
        )

        # 运行 baselines
        baseline_records = self.backtest_runner.run_baselines(split=split)
        baseline_results: Dict[str, Dict[str, Any]] = {}
        for name, bl_records in baseline_records.items():
            bl_dicts = [self._record_to_dict(r) for r in bl_records]
            bl_metrics = phase2_composite_metrics(
                bl_dicts,
                {},
                self.num_codes,
                self.dead_code_mask,
                metric_weights=self.config.selection_policy.metric_weights,
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
        val_pairs = [
            (idx, e)
            for idx, e in enumerate(self.backtest_runner.dataset.horizon_entries)
            if e.split == "val"
        ]
        if not val_pairs or num_folds < 1:
            return RollingValidationResult()

        fold_size = max((len(val_pairs) + num_folds - 1) // num_folds, 1)
        fold_metrics_list: List[Dict[str, Any]] = []
        per_fold_records: List[List[Phase2HorizonReplayRecord]] = []
        fold_sizes: List[int] = []
        fold_initial_positions: List[int] = []
        initial_position = 0

        for fold_idx in range(num_folds):
            # 每个 fold 使用 val 的一个子集
            start = fold_idx * fold_size
            end = min((fold_idx + 1) * fold_size, len(val_pairs))
            fold_pairs = val_pairs[start:end]
            if not fold_pairs:
                continue
            fold_indices = [idx for idx, _entry in fold_pairs]
            fold_initial_positions.append(initial_position)
            records = self.backtest_runner.run_walk_forward(
                split="val",
                deterministic=True,
                entry_indices=fold_indices,
                initial_position=initial_position,
                fold_id=fold_idx,
            )
            fold_records = records
            per_fold_records.append(fold_records)
            fold_sizes.append(len(fold_records))
            if fold_records and self.config.horizon_schedule.position_continuity:
                initial_position = fold_records[-1].final_position

            horizon_dicts = [self._record_to_dict(r) for r in fold_records]
            metrics = phase2_composite_metrics(
                horizon_dicts,
                {},
                self.num_codes,
                self.dead_code_mask,
                metric_weights=self.config.selection_policy.metric_weights,
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
            fold_sizes=fold_sizes,
            fold_initial_position_policy=(
                "inherit_previous_fold"
                if self.config.horizon_schedule.position_continuity
                else "flat"
            ),
            fold_initial_positions=fold_initial_positions,
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
            "risk_trigger_step": r.risk_trigger_step,
            "risk_reason": r.risk_reason,
            "fold_id": r.fold_id,
            "fold_initial_position": r.fold_initial_position,
            "timestamp_start": r.timestamp_start,
            "step_returns": r.step_returns,
            "selector_confidence": r.selector_confidence,
            "throttle_triggered": r.throttle_triggered,
            "original_code": r.original_code,
            "throttled_code": r.throttled_code,
        }

    def compute_selector_diagnostics(
        self,
        split: str,
        records: List[Phase2HorizonReplayRecord],
        entry_indices: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """计算 selector logits/probability 诊断。

        诊断与 walk-forward replay 使用同一批 horizon，并按 replay 后的
        ``final_position`` 重建下一条 horizon 的 ``prev_terminal_position``。
        """
        if not records:
            return self._empty_selector_diagnostics()

        import torch

        runner = self.backtest_runner
        actor_critic = runner.actor_critic
        dataset = runner.dataset
        resolved_entries = runner._resolve_walk_forward_entries(split, entry_indices)
        index_by_sample_id = {
            entry.sample_id: actual_idx for actual_idx, entry in resolved_entries
        }

        obs_rows = []
        labels: List[Optional[int]] = []
        prev_position = 0
        for record in records:
            actual_idx = index_by_sample_id.get(record.sample_id)
            if actual_idx is None:
                continue
            obs_prev_position = (
                prev_position
                if self.config.horizon_schedule.position_continuity
                else 0
            )
            obs_rows.append(dataset.get_selector_state(actual_idx, obs_prev_position))
            entry = dataset.horizon_entries[actual_idx]
            if entry.is_labeled and entry.code_label is not None:
                labels.append(int(entry.code_label))
            else:
                labels.append(None)
            prev_position = int(record.final_position)

        if not obs_rows:
            return self._empty_selector_diagnostics()

        import numpy as np
        selector = actor_critic.selector
        device = next(selector.parameters()).device
        was_training = selector.training
        selector.eval()
        obs_tensor = torch.tensor(
            np.asarray(obs_rows, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            logits, _ = selector(obs_tensor)
            logits = actor_critic._mask_logits(logits)
            probs = torch.softmax(logits, dim=-1)
            actions = logits.argmax(dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1.0e-12))).sum(dim=-1)
            top_k = min(2, logits.shape[-1])
            top_values = torch.topk(probs, top_k, dim=-1).values
        if was_training:
            selector.train()

        action_values = [int(v) for v in actions.detach().cpu().tolist()]
        counts = [0 for _ in range(self.num_codes)]
        for action in action_values:
            if 0 <= action < self.num_codes:
                counts[action] += 1

        labeled_count = 0
        correct_count = 0
        label_counts = [0 for _ in range(self.num_codes)]
        for action, label in zip(action_values, labels):
            if label is None or not (0 <= label < self.num_codes):
                continue
            labeled_count += 1
            label_counts[label] += 1
            if action == label:
                correct_count += 1

        sample_count = len(action_values)
        dominance_count = max(counts) if counts else 0
        dominance_code = counts.index(dominance_count) if counts else -1
        mean_probs = probs.mean(dim=0).detach().cpu().tolist()
        top1_prob = top_values[:, 0]
        top2_prob = (
            top_values[:, 1]
            if top_k > 1
            else torch.zeros_like(top1_prob)
        )

        out: Dict[str, float] = {
            "selector_sample_count": float(sample_count),
            "selector_labeled_count": float(labeled_count),
            "selector_label_accuracy": (
                float(correct_count / labeled_count) if labeled_count else 0.0
            ),
            "selector_argmax_dominance_code": float(dominance_code),
            "selector_argmax_dominance_count": float(dominance_count),
            "selector_argmax_dominance_ratio": float(
                dominance_count / max(sample_count, 1)
            ),
            "selector_entropy_mean": float(entropy.mean().item()),
            "selector_top1_prob_mean": float(top1_prob.mean().item()),
            "selector_top2_prob_mean": float(top2_prob.mean().item()),
            "selector_top1_margin_mean": float((top1_prob - top2_prob).mean().item()),
        }
        for idx in range(self.num_codes):
            out[f"selector_argmax_count_code_{idx}"] = float(counts[idx])
            out[f"selector_label_count_code_{idx}"] = float(label_counts[idx])
            out[f"selector_mean_prob_code_{idx}"] = (
                float(mean_probs[idx]) if idx < len(mean_probs) else 0.0
            )
        return out

    def _empty_selector_diagnostics(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "selector_sample_count": 0.0,
            "selector_labeled_count": 0.0,
            "selector_label_accuracy": 0.0,
            "selector_argmax_dominance_code": -1.0,
            "selector_argmax_dominance_count": 0.0,
            "selector_argmax_dominance_ratio": 0.0,
            "selector_entropy_mean": 0.0,
            "selector_top1_prob_mean": 0.0,
            "selector_top2_prob_mean": 0.0,
            "selector_top1_margin_mean": 0.0,
        }
        for idx in range(self.num_codes):
            out[f"selector_argmax_count_code_{idx}"] = 0.0
            out[f"selector_label_count_code_{idx}"] = 0.0
            out[f"selector_mean_prob_code_{idx}"] = 0.0
        return out

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

    def _fast_eval_entry_indices(self, split: str) -> Optional[List[int]]:
        """返回 fast eval 使用的 dataset-global entry indices。"""
        max_horizons = self.config.fast_eval_max_horizons
        if max_horizons is None or max_horizons <= 0:
            return None
        pairs = [
            (idx, e)
            for idx, e in enumerate(self.backtest_runner.dataset.horizon_entries)
            if e.split == split
        ]
        if len(pairs) <= max_horizons:
            return None
        stride = self.config.fast_eval_stride
        if stride is None or stride <= 0:
            stride = max(int(math.ceil(len(pairs) / max_horizons)), 1)
        return [idx for idx, _entry in pairs[::stride]][:max_horizons]
