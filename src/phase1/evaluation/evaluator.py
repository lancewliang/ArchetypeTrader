"""Phase I 评估编排器.

设计文档锚点: §4.7。

evaluate_epoch 流程:
1. 模型 forward 收集 logits / code_id / z_e。
2. 计算 action / risk / archetype / behavior 指标。
3. 跑 student + teacher replay，得到 capture / regret / Sharpe。
4. 行为多样性: 固定一批 states，分别用 K 个 code 解码。
5. 边界 replay。
6. 收集 warnings。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.phase1.config import CausalOnlineValidationConfig
from src.phase1.data.dataset import Phase1DemoDataset, collate_phase1
from src.preprocess_data.horizon_builder import HorizonRecord

from src.evaluation.metrics.action import (
    action_confusion_matrix,
    switch_metrics,
    single_trade_consistency_rate,
    weighted_reconstruction_accuracy,
)
from src.evaluation.metrics.archetype import dp_teacher_quality, per_code_summary
from src.evaluation.metrics.behavior import (
    decoder_sensitivity_to_code,
    inter_code_action_diversity,
    inter_code_distance,
    latent_silhouette_score,
    per_code_action_entropy,
)
from src.evaluation.metrics.risk import (
    DEFAULT_ANNUALIZATION_FACTOR,
    calmar_ratio,
    cumulative_pnl_curve,
    equity_curve_from_step_returns,
    max_drawdown_abs,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    step_returns_from_pnl,
)
from .metrics import (
    codebook_displacement,
    code_usage_ratio,
    codebook_perplexity,
    epoch_code_stability,
    matched_epoch_code_stability,
    reconstruction_accuracy,
    return_capture_ratio,
    regret_to_dp,
    non_flat_accuracy,
)
from .replay import HorizonReplayRecord, Phase1ReplayEvaluator


@dataclass
class EpochMetrics:
    epoch: int
    metrics: Dict[str, float] = field(default_factory=dict)
    per_code_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    per_horizon_replay_records: List[dict] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class Phase1Evaluator:
    """编排所有评估。

    使用方式::

        evaluator = Phase1Evaluator(
            replay_evaluator=Phase1ReplayEvaluator(env_factory),
            annualization_factor=525_600,
            fast_probe_size=2048,
        )
        metrics = evaluator.evaluate_epoch(
            epoch=10, model=model, val_data=val_dataset,
            val_records=val_horizons, full_validation=True,
        )
    """

    def __init__(
        self,
        replay_evaluator: Phase1ReplayEvaluator,
        annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
        fast_probe_size: int = 2048,
        reward_normalizer=None,
        risk_capital_base: Optional[float] = None,
        online_validation_config: Optional[CausalOnlineValidationConfig] = None,
    ) -> None:
        self.replay_evaluator = replay_evaluator
        self.annualization_factor = annualization_factor
        self.fast_probe_size = fast_probe_size
        self.reward_normalizer = reward_normalizer
        self.risk_capital_base = risk_capital_base
        self.online_validation_config = (
            online_validation_config or CausalOnlineValidationConfig()
        )
        self._previous_code_ids: Optional[List[int]] = None
        self._previous_codebook: Optional[List[List[float]]] = None

    def evaluate_epoch(
        self,
        epoch: int,
        model,
        val_data: Phase1DemoDataset,
        val_records: Sequence[HorizonRecord],
        full_validation: bool,
    ) -> EpochMetrics:
        """跑一遍 validation 并计算所有指标。

        Steps
        -----
        1. 模型 forward 收集 ``logits / code_id / z_e``。
        2. action 指标: reconstruction / weighted / non-flat / switch metrics /
           single-trade consistency。
        3. VQ 指标: usage / perplexity / inter_code_distance / silhouette。
        4. teacher + student replay (per horizon) → 收益与 capture / regret。
        5. 风险指标: raw PnL 先按名义本金归一化，再算 Sharpe / MDD / Calmar。
        6. DP teacher quality: profitable_ratio / sharpe / 收益分布。
        7. per-archetype 汇总 + no_trade 集中度 + active_trade_code_count。
        8. 行为多样性: 固定一小批 states，分别用每个 code 解码，比较 logits 与
           argmax actions（``inter_code_action_diversity`` / ``decoder_sensitivity_to_code``）。
        9. teacher-free causal online validation。
        10. 收集 warnings（codebook collapse / drawdown / teacher 弱质量）。

        Parameters
        ----------
        full_validation : True 时遍历 ``val_records``；False 时仅取前
                          ``fast_probe_size`` 条做 fast probe。

        Returns
        -------
        EpochMetrics : 含 ``metrics / per_code_metrics / per_horizon_replay_records / warnings``。
        """
        out = EpochMetrics(epoch=epoch)

        teacher_ctx = self._evaluate_teacher_conditioned(
            model, val_data, val_records, full_validation, out
        )

        self._evaluate_online_validation(
            model, val_records, full_validation, out, teacher_ctx
        )

        self._build_diagnostics(out, teacher_ctx)

        self._previous_code_ids = list(teacher_ctx.code_ids)
        self._previous_codebook = [list(row) for row in teacher_ctx.codebook_list]
        return out

    def _evaluate_teacher_conditioned(
        self,
        model,
        val_data: Phase1DemoDataset,
        val_records: Sequence[HorizonRecord],
        full_validation: bool,
        out: EpochMetrics,
    ) -> _TeacherConditionedContext:
        """teacher-conditioned validation: student 使用 teacher 分配的 code_id 进行 replay。

        产生的指标前缀为 ``teacher_val_``。
        """
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:
            raise RuntimeError("evaluate_epoch 需要 torch")

        sample_size = len(val_data) if full_validation else min(self.fast_probe_size, len(val_data))
        probe_records = list(val_records[:sample_size])
        probe_dataset = Phase1DemoDataset(
            records=probe_records,
            reward_normalizer=self.reward_normalizer,
        )
        _device = next(model.parameters()).device
        _use_cuda = _device.type == "cuda"
        loader = DataLoader(
            probe_dataset,
            batch_size=256,
            shuffle=False,
            collate_fn=collate_phase1,
            num_workers=2 if _use_cuda else 0,
            pin_memory=_use_cuda,
        )
        all_logits: List = []
        all_actions: List = []
        all_code_ids: List[int] = []
        all_z_e: List = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                states = batch["states"].to(_device, non_blocking=True)
                actions = batch["actions"].to(_device, non_blocking=True)
                rewards = batch["rewards"].to(_device, non_blocking=True)
                outputs = model(states, actions, rewards)
                all_logits.append(outputs.action_logits)
                all_actions.append(actions)
                all_code_ids.extend(outputs.code_id.tolist())
                all_z_e.append(outputs.z_e)

        if all_logits:
            logits = torch.cat(all_logits, dim=0).cpu().tolist()
            true_actions = torch.cat(all_actions, dim=0).cpu().tolist()
            z_e_tensor = torch.cat(all_z_e, dim=0).cpu().tolist()
        else:
            logits = []
            true_actions = []
            z_e_tensor = []

        pred_actions = [
            [int(max(range(3), key=lambda i: step[i])) for step in row]
            for row in logits
        ]

        out.metrics["teacher_val_reconstruction_accuracy"] = reconstruction_accuracy(logits, true_actions)
        out.metrics["teacher_val_weighted_reconstruction_accuracy"] = weighted_reconstruction_accuracy(
            logits, true_actions, class_weights={0: 2.0, 1: 1.0, 2: 2.0}
        )
        out.metrics["teacher_val_non_flat_accuracy"] = non_flat_accuracy(logits, true_actions)
        out.metrics["teacher_val_single_trade_consistency_rate"] = single_trade_consistency_rate(pred_actions)
        sw = switch_metrics(true_actions, pred_actions)
        out.metrics["teacher_val_switch_point_recall"] = sw.switch_point_recall
        out.metrics["teacher_val_switch_direction_accuracy"] = (
            sw.switch_direction_accuracy
        )
        out.metrics["teacher_val_switch_timing_error_mean"] = (
            sw.switch_timing_error_mean
        )
        cm = action_confusion_matrix(true_actions, pred_actions)
        out.metrics["confusion_matrix"] = cm.matrix
        out.metrics["action_precision_recall_per_class"] = cm.per_class()

        K = model.quantizer.num_codes
        out.metrics["code_usage_ratio"] = code_usage_ratio(all_code_ids, K)
        out.metrics["perplexity"] = codebook_perplexity(all_code_ids, K)
        codebook_list = model.quantizer.codebook.detach().cpu().tolist()
        out.metrics["inter_code_distance"] = inter_code_distance(codebook_list)
        out.metrics["silhouette_score"] = latent_silhouette_score(z_e_tensor, all_code_ids)
        stability_measured = (
            self._previous_code_ids is not None
            and len(self._previous_code_ids) == len(all_code_ids)
        )
        out.metrics["epoch_code_stability_measured"] = stability_measured
        if stability_measured:
            out.metrics["epoch_code_stability"] = epoch_code_stability(
                self._previous_code_ids, all_code_ids
            )
            if (
                self._previous_codebook is not None
                and len(self._previous_codebook) == len(codebook_list)
            ):
                out.metrics["epoch_code_stability_matched"] = matched_epoch_code_stability(
                    self._previous_code_ids,
                    all_code_ids,
                    self._previous_codebook,
                    codebook_list,
                )
            else:
                out.metrics["epoch_code_stability_matched"] = out.metrics[
                    "epoch_code_stability"
                ]
        else:
            out.metrics["epoch_code_stability"] = 1.0
            out.metrics["epoch_code_stability_matched"] = 1.0
        displacement = (
            codebook_displacement(codebook_list, self._previous_codebook)
            if self._previous_codebook is not None
            else {}
        )
        if displacement:
            out.metrics["codebook_displacement_mean"] = sum(displacement.values()) / len(displacement)
            out.metrics["codebook_displacement_max"] = max(displacement.values())
        else:
            out.metrics["codebook_displacement_mean"] = 0.0
            out.metrics["codebook_displacement_max"] = 0.0

        teacher_records: List[HorizonReplayRecord] = []
        student_records: List[HorizonReplayRecord] = []
        per_horizon_replay: List[dict] = []
        decoded_actions: List[List[int]] = []
        risk_capital_base = self._risk_capital_base(probe_records)
        for rec, code_id in zip(probe_records, all_code_ids):
            teacher = self.replay_evaluator.replay_dp_teacher(rec)
            student = self.replay_evaluator.replay_student_online(
                rec, model.decoder, model.quantizer.codebook, code_id
            )
            student_step_return_rates = step_returns_from_pnl(
                student.student_step_returns, risk_capital_base
            )
            teacher_records.append(teacher)
            student_records.append(student)
            decoded_actions.append(list(student.student_actions))
            per_horizon_replay.append(
                {
                    "sample_id": rec.sample_id,
                    "code_id": code_id,
                    "teacher_actions": teacher.teacher_actions,
                    "student_actions": student.student_actions,
                    "teacher_net_return": teacher.teacher_net_return,
                    "student_net_return": student.student_net_return,
                    "regret_to_dp": regret_to_dp(student.student_net_return, teacher.teacher_net_return),
                    "cost_paid": student.cost_paid,
                    "teacher_is_no_trade": all(a == 1 for a in teacher.teacher_actions),
                    "student_turnover": sum(
                        1 for i in range(1, len(student.student_actions))
                        if student.student_actions[i] != student.student_actions[i - 1]
                    ),
                    "switch_timing_error": _switch_timing_diff(
                        teacher.teacher_actions, student.student_actions
                    ),
                    "step_rewards": student.student_step_returns,
                    "cumulative_pnl": cumulative_pnl_curve(student.student_step_returns),
                    "cumulative_returns": equity_curve_from_step_returns(student_step_return_rates),
                    "drawdowns": [],
                    "price_series": rec.prices,
                    "positions": [int(a) - 1 for a in student.student_actions],
                }
            )
        out.per_horizon_replay_records = per_horizon_replay

        teacher_total = sum(r.teacher_net_return for r in teacher_records)
        student_total = sum(r.student_net_return for r in student_records)
        out.metrics["teacher_val_dp_teacher_net_return"] = teacher_total
        out.metrics["teacher_val_student_net_return"] = student_total
        out.metrics["teacher_val_return_capture_ratio"] = return_capture_ratio(student_total, teacher_total)
        out.metrics["teacher_val_regret_to_dp"] = regret_to_dp(student_total, teacher_total)
        out.metrics["teacher_val_cost_paid"] = sum(r.cost_paid for r in student_records)
        boundary = self.replay_evaluator.evaluate_horizon_boundaries(
            probe_records, decoded_actions
        )
        out.metrics["horizon_boundary_turnover_cost"] = (
            boundary.horizon_boundary_turnover_cost
        )
        out.metrics["horizon_boundary_position_consistency"] = (
            boundary.horizon_boundary_position_consistency
        )

        flat_student_steps: List[float] = []
        for r in student_records:
            flat_student_steps.extend(r.student_step_returns)
        flat_student_return_rates = step_returns_from_pnl(
            flat_student_steps, risk_capital_base
        )
        out.metrics["teacher_val_risk_capital_base"] = risk_capital_base
        out.metrics["teacher_val_sharpe_ratio"] = sharpe_ratio(flat_student_return_rates, self.annualization_factor)
        out.metrics["teacher_val_sortino_ratio"] = sortino_ratio(flat_student_return_rates, self.annualization_factor)
        equity = equity_curve_from_step_returns(flat_student_return_rates)
        out.metrics["teacher_val_max_drawdown"] = max_drawdown(equity)
        out.metrics["teacher_val_max_drawdown_abs"] = max_drawdown_abs(
            cumulative_pnl_curve(flat_student_steps)
        )
        annual_ret = sum(flat_student_return_rates) * (
            self.annualization_factor / max(len(flat_student_return_rates), 1)
        )
        out.metrics["teacher_val_annual_return_ratio"] = annual_ret
        out.metrics["teacher_val_calmar_ratio"] = calmar_ratio(annual_ret, out.metrics["teacher_val_max_drawdown"])

        flat_teacher_steps: List[float] = []
        teacher_horizons: List[float] = []
        for r in teacher_records:
            flat_teacher_steps.extend(r.teacher_step_returns)
            teacher_horizons.append(r.teacher_net_return)
        tq = dp_teacher_quality(teacher_horizons, flat_teacher_steps, self.annualization_factor)
        out.metrics["teacher_val_dp_teacher_sharpe"] = tq.teacher_val_dp_teacher_sharpe
        out.metrics["teacher_val_dp_teacher_profitable_ratio"] = (
            tq.teacher_val_dp_teacher_profitable_ratio
        )
        out.metrics["dp_teacher_return_distribution"] = tq.return_distribution
        out.metrics["teacher_val_dp_teacher_return_distribution"] = tq.return_distribution

        diag = per_code_summary(
            horizon_returns=[r.student_net_return for r in student_records],
            code_ids=all_code_ids,
            no_trade_flags=[
                all(a == 1 for a in r.teacher_actions) for r in teacher_records
            ],
            switch_points=[
                next((i for i in range(1, len(r.student_actions)) if r.student_actions[i] != r.student_actions[i - 1]), -1)
                for r in student_records
            ],
        )
        for s in diag.per_code:
            out.per_code_metrics[s.code_id] = {
                "count": s.count,
                "avg_return": s.avg_return,
                "win_rate": s.win_rate,
                "no_trade_ratio": s.no_trade_ratio,
                "switch_point_distribution": s.switch_point_distribution,
            }
        out.metrics["per_code_switch_point_distribution"] = {
            str(s.code_id): s.switch_point_distribution for s in diag.per_code
        }
        out.metrics["active_trade_code_count"] = float(diag.active_trade_code_count)
        out.metrics["no_trade_code_concentration_top1"] = diag.no_trade_code_concentration["top1"]
        out.metrics["no_trade_code_concentration_top2"] = diag.no_trade_code_concentration["top2"]

        if probe_records and len(all_code_ids) > 0:
            decoded_logits_by_code, decoded_actions_by_code = self._behavior_probe(
                model, probe_records[: min(64, len(probe_records))]
            )
            out.metrics["per_code_action_entropy_mean"] = (
                sum(per_code_action_entropy(decoded_logits_by_code).values())
                / max(K, 1)
            )
            out.metrics["inter_code_action_diversity"] = inter_code_action_diversity(decoded_actions_by_code)
            out.metrics["decoder_sensitivity_to_code"] = decoder_sensitivity_to_code(decoded_logits_by_code)

        if out.metrics.get("code_usage_ratio", 1.0) < 0.7:
            out.warnings.append(f"code_usage_ratio={out.metrics['code_usage_ratio']:.3f} < 0.7")
        if out.metrics.get("teacher_val_max_drawdown", 0.0) > 0.2:
            out.warnings.append(f"teacher_val_max_drawdown={out.metrics['teacher_val_max_drawdown']:.3f} > 0.2")
        if out.metrics.get("teacher_val_dp_teacher_profitable_ratio", 1.0) < 0.3:
            out.warnings.append("DP teacher profitable_ratio 低，return_capture_ratio 不可单独解读为学得好")

        return _TeacherConditionedContext(
            teacher_total=teacher_total,
            risk_capital_base=risk_capital_base,
            code_ids=all_code_ids,
            codebook_list=codebook_list,
            probe_records=probe_records,
            K=K,
            confusion_matrix=cm,
            switch_metrics=sw,
            teacher_quality=tq,
            displacement=displacement,
        )

    def _evaluate_online_validation(
        self,
        model,
        val_records: Sequence[HorizonRecord],
        full_validation: bool,
        out: EpochMetrics,
        teacher_ctx: _TeacherConditionedContext,
    ) -> None:
        """teacher-free causal online validation: student 使用 state prefix 自编码的 code_id。

        产生的指标前缀为 ``online_val_``。
        """
        online_cfg = self.online_validation_config
        out.metrics["online_validation_measured"] = False
        out.metrics["online_code_prefix_steps"] = float(
            max(int(online_cfg.code_prefix_steps), 1)
        )
        out.metrics["online_code_usage_ratio"] = 0.0
        out.metrics["online_val_student_net_return"] = 0.0
        out.metrics["online_val_return_capture_ratio"] = 0.0
        out.metrics["online_val_regret_to_dp"] = 0.0
        out.metrics["online_val_cost_paid"] = 0.0
        out.metrics["online_val_sharpe_ratio"] = 0.0
        out.metrics["online_val_max_drawdown"] = 0.0
        out.metrics["online_val_max_drawdown_abs"] = 0.0
        out.metrics["online_horizon_boundary_turnover_cost"] = 0.0
        out.metrics["online_horizon_boundary_position_consistency"] = 1.0

        if not (online_cfg.enabled and full_validation and teacher_ctx.probe_records):
            return

        online_code_ids = self._online_state_prefix_code_ids(model, teacher_ctx.probe_records)
        online_student_records = (
            self.replay_evaluator.replay_student_online_sequence(
                teacher_ctx.probe_records,
                model.decoder,
                model.quantizer.codebook,
                online_code_ids,
            )
        )
        online_decoded_actions = [
            list(r.student_actions) for r in online_student_records
        ]
        online_total = sum(r.student_net_return for r in online_student_records)
        online_flat_steps: List[float] = []
        for r in online_student_records:
            online_flat_steps.extend(r.student_step_returns)
        online_return_rates = step_returns_from_pnl(
            online_flat_steps, teacher_ctx.risk_capital_base
        )
        online_equity = equity_curve_from_step_returns(online_return_rates)
        online_boundary = self.replay_evaluator.evaluate_horizon_boundaries(
            teacher_ctx.probe_records, online_decoded_actions
        )
        out.metrics["online_validation_measured"] = True
        out.metrics["online_code_usage_ratio"] = code_usage_ratio(online_code_ids, teacher_ctx.K)
        out.metrics["online_val_student_net_return"] = online_total
        out.metrics["online_val_return_capture_ratio"] = return_capture_ratio(
            online_total, teacher_ctx.teacher_total
        )
        out.metrics["online_val_regret_to_dp"] = regret_to_dp(
            online_total, teacher_ctx.teacher_total
        )
        out.metrics["online_val_cost_paid"] = sum(
            r.cost_paid for r in online_student_records
        )
        out.metrics["online_val_sharpe_ratio"] = sharpe_ratio(
            online_return_rates, self.annualization_factor
        )
        out.metrics["online_val_max_drawdown"] = max_drawdown(online_equity)
        out.metrics["online_val_max_drawdown_abs"] = max_drawdown_abs(
            cumulative_pnl_curve(online_flat_steps)
        )
        out.metrics["online_horizon_boundary_turnover_cost"] = (
            online_boundary.horizon_boundary_turnover_cost
        )
        out.metrics["online_horizon_boundary_position_consistency"] = (
            online_boundary.horizon_boundary_position_consistency
        )

    def _build_diagnostics(
        self,
        out: EpochMetrics,
        teacher_ctx: _TeacherConditionedContext,
    ) -> None:
        """构建 diagnostics 字典。"""
        out.diagnostics = {
            "action": {
                "confusion_matrix": teacher_ctx.confusion_matrix.matrix,
                "action_precision_recall_per_class": teacher_ctx.confusion_matrix.per_class(),
                "switch_timing_error_distribution": teacher_ctx.switch_metrics.switch_timing_error_distribution,
            },
            "risk": {
                "teacher_val_risk_capital_base": out.metrics[
                    "teacher_val_risk_capital_base"
                ],
                "teacher_val_sharpe_ratio": out.metrics["teacher_val_sharpe_ratio"],
                "teacher_val_sortino_ratio": out.metrics["teacher_val_sortino_ratio"],
                "teacher_val_max_drawdown": out.metrics[
                    "teacher_val_max_drawdown"
                ],
                "teacher_val_max_drawdown_abs": out.metrics[
                    "teacher_val_max_drawdown_abs"
                ],
                "teacher_val_annual_return_ratio": out.metrics[
                    "teacher_val_annual_return_ratio"
                ],
                "teacher_val_calmar_ratio": out.metrics["teacher_val_calmar_ratio"],
            },
            "archetype_separation": {
                "code_usage_ratio": out.metrics["code_usage_ratio"],
                "perplexity": out.metrics["perplexity"],
                "inter_code_distance": out.metrics["inter_code_distance"],
                "silhouette_score": out.metrics["silhouette_score"],
                "per_code_metrics": {str(k): v for k, v in out.per_code_metrics.items()},
                "dp_teacher_return_distribution": teacher_ctx.teacher_quality.return_distribution,
            },
            "archetype_behavior": {
                "active_trade_code_count": out.metrics["active_trade_code_count"],
                "no_trade_code_concentration_top1": out.metrics["no_trade_code_concentration_top1"],
                "no_trade_code_concentration_top2": out.metrics["no_trade_code_concentration_top2"],
                "per_code_switch_point_distribution": out.metrics["per_code_switch_point_distribution"],
                "inter_code_action_diversity": out.metrics.get("inter_code_action_diversity", 0.0),
                "decoder_sensitivity_to_code": out.metrics.get("decoder_sensitivity_to_code", 0.0),
            },
            "horizon_boundary": {
                "horizon_boundary_turnover_cost": out.metrics["horizon_boundary_turnover_cost"],
                "horizon_boundary_position_consistency": out.metrics["horizon_boundary_position_consistency"],
                "online_horizon_boundary_turnover_cost": out.metrics[
                    "online_horizon_boundary_turnover_cost"
                ],
                "online_horizon_boundary_position_consistency": out.metrics[
                    "online_horizon_boundary_position_consistency"
                ],
            },
            "online_validation": {
                "online_validation_measured": out.metrics["online_validation_measured"],
                "online_code_prefix_steps": out.metrics["online_code_prefix_steps"],
                "online_code_usage_ratio": out.metrics["online_code_usage_ratio"],
                "online_val_student_net_return": out.metrics[
                    "online_val_student_net_return"
                ],
                "online_val_return_capture_ratio": out.metrics[
                    "online_val_return_capture_ratio"
                ],
                "online_val_regret_to_dp": out.metrics["online_val_regret_to_dp"],
                "online_val_cost_paid": out.metrics["online_val_cost_paid"],
                "online_val_sharpe_ratio": out.metrics["online_val_sharpe_ratio"],
                "online_val_max_drawdown": out.metrics["online_val_max_drawdown"],
                "online_val_max_drawdown_abs": out.metrics[
                    "online_val_max_drawdown_abs"
                ],
            },
            "code_stability": {
                "epoch_code_stability_measured": out.metrics["epoch_code_stability_measured"],
                "epoch_code_stability": out.metrics["epoch_code_stability"],
                "epoch_code_stability_matched": out.metrics["epoch_code_stability_matched"],
                "codebook_displacement": teacher_ctx.displacement,
                "codebook_displacement_mean": out.metrics["codebook_displacement_mean"],
                "codebook_displacement_max": out.metrics["codebook_displacement_max"],
            },
        }

    def _online_state_prefix_code_ids(self, model, records: Sequence[HorizonRecord]) -> List[int]:
        """用 teacher-free state prefix 编码 online validation 的 code_id。"""
        import torch

        _device = next(model.parameters()).device
        prefix_steps_cfg = max(int(self.online_validation_config.code_prefix_steps), 1)
        code_ids: List[int] = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for rec in records:
                prefix_steps = max(1, min(prefix_steps_cfg, len(rec.states)))
                states = torch.tensor(
                    [rec.states[:prefix_steps]], dtype=torch.float32, device=_device
                )
                actions = torch.ones(
                    (1, prefix_steps), dtype=torch.long, device=_device
                )
                rewards = torch.zeros(
                    (1, prefix_steps), dtype=torch.float32, device=_device
                )
                fused = model.input_adapter(states, actions, rewards)
                z_e = model.encoder(fused)
                q = model.quantizer.quantize(z_e)
                code_ids.append(int(q.code_id.item()))
        if was_training:
            model.train()
        return code_ids

    def _risk_capital_base(self, records: Sequence[HorizonRecord]) -> float:
        """估算把 raw PnL 转成收益率所需的名义本金。"""
        if self.risk_capital_base is not None and self.risk_capital_base > 0:
            return float(self.risk_capital_base)

        prices: List[float] = []
        for rec in records:
            prices.extend(
                float(v) for v in rec.prices
                if v is not None and math.isfinite(float(v)) and float(v) > 0
            )
        if not prices:
            return 1.0
        prices.sort()
        mid = len(prices) // 2
        if len(prices) % 2:
            median_price = prices[mid]
        else:
            median_price = (prices[mid - 1] + prices[mid]) / 2.0

        try:
            env = self.replay_evaluator.env_factory()
            max_position = max(abs(float(getattr(env, "max_position", 1.0))), 1.0)
        except Exception:
            max_position = 1.0
        return max(median_price * max_position, 1.0)

    def _behavior_probe(self, model, records: List[HorizonRecord]):
        """固定一小批 states，分别用每个 code 解码，得到 logits / actions by code。"""
        import torch

        _device = next(model.parameters()).device
        states = torch.tensor([r.states for r in records], dtype=torch.float32).to(_device)
        K = model.quantizer.num_codes
        decoded_logits: Dict[int, list] = {}
        decoded_actions: Dict[int, list] = {}
        with torch.no_grad():
            for cid in range(K):
                code_tensor = torch.full((states.shape[0],), cid, dtype=torch.long, device=_device)
                logits = model.decode(states, code_tensor)
                decoded_logits[cid] = logits.cpu().tolist()
                decoded_actions[cid] = logits.argmax(dim=-1).cpu().tolist()
        return decoded_logits, decoded_actions


@dataclass
class _TeacherConditionedContext:
    """``_evaluate_teacher_conditioned`` 的返回值，供 online validation 和 diagnostics 使用。"""
    teacher_total: float
    risk_capital_base: float
    code_ids: List[int]
    codebook_list: List[List[float]]
    probe_records: List[HorizonRecord]
    K: int
    confusion_matrix: Any
    switch_metrics: Any
    teacher_quality: Any
    displacement: Dict[str, float]


def _switch_timing_diff(teacher_actions, student_actions) -> int:
    """teacher 切换点与 student 切换点的差值（绝对）。"""
    def _find(actions):
        for i in range(1, len(actions)):
            if actions[i] != actions[i - 1]:
                return i
        return -1
    t = _find(teacher_actions)
    s = _find(student_actions)
    if t == -1 and s == -1:
        return 0
    if t == -1 or s == -1:
        return abs(len(teacher_actions))
    return abs(t - s)
