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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.config.phase1_config import SelectionPolicyConfig
from src.data.dataset import Phase1DemoDataset, collate_phase1
from src.data.horizon_builder import HorizonRecord

from .metrics.action import (
    action_confusion_matrix,
    switch_metrics,
    single_trade_consistency_rate,
    weighted_reconstruction_accuracy,
)
from .metrics.archetype import dp_teacher_quality, per_code_summary
from .metrics.behavior import (
    decoder_sensitivity_to_code,
    inter_code_action_diversity,
    inter_code_distance,
    latent_silhouette_score,
    per_code_action_entropy,
)
from .metrics.risk import (
    DEFAULT_ANNUALIZATION_FACTOR,
    calmar_ratio,
    equity_curve_from_step_returns,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from .phase1_metrics import (
    codebook_displacement,
    code_usage_ratio,
    codebook_perplexity,
    epoch_code_stability,
    reconstruction_accuracy,
    return_capture_ratio,
    regret_to_dp,
    non_flat_accuracy,
)
from .phase1_replay import HorizonReplayRecord, Phase1ReplayEvaluator


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
    ) -> None:
        self.replay_evaluator = replay_evaluator
        self.annualization_factor = annualization_factor
        self.fast_probe_size = fast_probe_size
        # encoder 输入的 reward 必须使用与 train 一致的 normalizer 实例；
        # 否则 val/test code_id 会和 train 不在同一表征空间，所有 capture / regret
        # 都会被误读。
        self.reward_normalizer = reward_normalizer
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
        5. 风险指标: Sharpe / Sortino / MDD / Calmar (以 student 拼接的 step returns)。
        6. DP teacher quality: profitable_ratio / sharpe / 收益分布。
        7. per-archetype 汇总 + no_trade 集中度 + active_trade_code_count。
        8. 行为多样性: 固定一小批 states，分别用每个 code 解码，比较 logits 与
           argmax actions（``inter_code_action_diversity`` / ``decoder_sensitivity_to_code``）。
        9. 收集 warnings（codebook collapse / drawdown / teacher 弱质量）。

        Parameters
        ----------
        full_validation : True 时遍历 ``val_records``；False 时仅取前
                          ``fast_probe_size`` 条做 fast probe。

        Returns
        -------
        EpochMetrics : 含 ``metrics / per_code_metrics / per_horizon_replay_records / warnings``。
        """
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:
            raise RuntimeError("evaluate_epoch 需要 torch")

        out = EpochMetrics(epoch=epoch)

        # ---- 模型 forward 收集 logits / code_id / z_e ----
        sample_size = len(val_data) if full_validation else min(self.fast_probe_size, len(val_data))
        probe_records = list(val_records[:sample_size])
        # 必须把 normalizer 注入 dataset，确保 encoder 收到的 rewards 与训练一致。
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
            num_workers=2,
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

        out.metrics["reconstruction_accuracy"] = reconstruction_accuracy(logits, true_actions)
        out.metrics["val_weighted_reconstruction_accuracy"] = weighted_reconstruction_accuracy(
            logits, true_actions, class_weights={0: 2.0, 1: 1.0, 2: 2.0}
        )
        out.metrics["weighted_reconstruction_accuracy"] = out.metrics[
            "val_weighted_reconstruction_accuracy"
        ]
        out.metrics["non_flat_accuracy"] = non_flat_accuracy(logits, true_actions)
        out.metrics["single_trade_consistency_rate"] = single_trade_consistency_rate(pred_actions)
        sw = switch_metrics(true_actions, pred_actions)
        out.metrics["switch_point_recall"] = sw.switch_point_recall
        out.metrics["switch_direction_accuracy"] = sw.switch_direction_accuracy
        out.metrics["switch_timing_error_mean"] = sw.switch_timing_error_mean
        cm = action_confusion_matrix(true_actions, pred_actions)
        out.metrics["confusion_matrix"] = cm.matrix
        out.metrics["action_precision_recall_per_class"] = cm.per_class()

        # ---- VQ 指标 ----
        K = model.quantizer.num_codes
        out.metrics["code_usage_ratio"] = code_usage_ratio(all_code_ids, K)
        out.metrics["perplexity"] = codebook_perplexity(all_code_ids, K)
        codebook_list = model.quantizer.codebook.detach().cpu().tolist()
        out.metrics["inter_code_distance"] = inter_code_distance(codebook_list)
        out.metrics["silhouette_score"] = latent_silhouette_score(z_e_tensor, all_code_ids)
        if self._previous_code_ids is not None and len(self._previous_code_ids) == len(all_code_ids):
            out.metrics["epoch_code_stability"] = epoch_code_stability(
                self._previous_code_ids, all_code_ids
            )
        else:
            out.metrics["epoch_code_stability"] = 1.0
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

        # ---- replay: teacher / student ----
        teacher_records: List[HorizonReplayRecord] = []
        student_records: List[HorizonReplayRecord] = []
        per_horizon_replay: List[dict] = []
        decoded_actions: List[List[int]] = []
        for rec, code_id in zip(probe_records, all_code_ids):
            teacher = self.replay_evaluator.replay_dp_teacher(rec)
            student = self.replay_evaluator.replay_student_online(
                rec, model.decoder, model.quantizer.codebook, code_id
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
                    "cumulative_returns": equity_curve_from_step_returns(student.student_step_returns),
                    "drawdowns": [],  # 需要时再补
                    "price_series": rec.prices,
                    "positions": [int(a) - 1 for a in student.student_actions],  # action→position
                }
            )
        out.per_horizon_replay_records = per_horizon_replay

        teacher_total = sum(r.teacher_net_return for r in teacher_records)
        student_total = sum(r.student_net_return for r in student_records)
        out.metrics["val_dp_teacher_net_return"] = teacher_total
        out.metrics["val_student_online_net_return"] = student_total
        out.metrics["val_return_capture_ratio"] = return_capture_ratio(student_total, teacher_total)
        out.metrics["val_regret_to_dp"] = regret_to_dp(student_total, teacher_total)
        out.metrics["val_cost_paid"] = sum(r.cost_paid for r in student_records)
        boundary = self.replay_evaluator.evaluate_horizon_boundaries(
            probe_records, decoded_actions
        )
        out.metrics["horizon_boundary_turnover_cost"] = (
            boundary.horizon_boundary_turnover_cost
        )
        out.metrics["horizon_boundary_position_consistency"] = (
            boundary.horizon_boundary_position_consistency
        )

        # 风险指标: 把所有 student step returns 拼起来再算
        flat_student_steps: List[float] = []
        for r in student_records:
            flat_student_steps.extend(r.student_step_returns)
        out.metrics["val_sharpe_ratio"] = sharpe_ratio(flat_student_steps, self.annualization_factor)
        out.metrics["val_sortino_ratio"] = sortino_ratio(flat_student_steps, self.annualization_factor)
        equity = equity_curve_from_step_returns(flat_student_steps)
        out.metrics["val_max_drawdown"] = max_drawdown(equity)
        annual_ret = student_total * (self.annualization_factor / max(len(flat_student_steps), 1))
        out.metrics["val_calmar_ratio"] = calmar_ratio(annual_ret, out.metrics["val_max_drawdown"])

        # DP teacher quality
        flat_teacher_steps: List[float] = []
        teacher_horizons: List[float] = []
        for r in teacher_records:
            flat_teacher_steps.extend(r.teacher_step_returns)
            teacher_horizons.append(r.teacher_net_return)
        tq = dp_teacher_quality(teacher_horizons, flat_teacher_steps, self.annualization_factor)
        out.metrics["val_dp_teacher_sharpe"] = tq.val_dp_teacher_sharpe
        out.metrics["val_dp_teacher_profitable_ratio"] = tq.val_dp_teacher_profitable_ratio
        out.metrics["dp_teacher_return_distribution"] = tq.return_distribution

        # ---- per-archetype ----
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

        # ---- 行为多样性: 固定一小批 states，分别用每个 code 解码 ----
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

        # ---- warnings ----
        if out.metrics.get("code_usage_ratio", 1.0) < 0.7:
            out.warnings.append(f"code_usage_ratio={out.metrics['code_usage_ratio']:.3f} < 0.7")
        if out.metrics.get("val_max_drawdown", 0.0) > 0.2:
            out.warnings.append(f"val_max_drawdown={out.metrics['val_max_drawdown']:.3f} > 0.2")
        if out.metrics.get("val_dp_teacher_profitable_ratio", 1.0) < 0.3:
            out.warnings.append("DP teacher profitable_ratio 低，return_capture_ratio 不可单独解读为学得好")

        out.diagnostics = {
            "action": {
                "confusion_matrix": cm.matrix,
                "action_precision_recall_per_class": cm.per_class(),
                "switch_timing_error_distribution": sw.switch_timing_error_distribution,
            },
            "risk": {
                "val_sharpe_ratio": out.metrics["val_sharpe_ratio"],
                "val_sortino_ratio": out.metrics["val_sortino_ratio"],
                "val_max_drawdown": out.metrics["val_max_drawdown"],
                "val_calmar_ratio": out.metrics["val_calmar_ratio"],
            },
            "archetype_separation": {
                "code_usage_ratio": out.metrics["code_usage_ratio"],
                "perplexity": out.metrics["perplexity"],
                "inter_code_distance": out.metrics["inter_code_distance"],
                "silhouette_score": out.metrics["silhouette_score"],
                "per_code_metrics": {str(k): v for k, v in out.per_code_metrics.items()},
                "dp_teacher_return_distribution": tq.return_distribution,
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
            },
            "code_stability": {
                "epoch_code_stability": out.metrics["epoch_code_stability"],
                "codebook_displacement": displacement,
                "codebook_displacement_mean": out.metrics["codebook_displacement_mean"],
                "codebook_displacement_max": out.metrics["codebook_displacement_max"],
            },
        }
        self._previous_code_ids = list(all_code_ids)
        self._previous_codebook = [list(row) for row in codebook_list]
        return out

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
