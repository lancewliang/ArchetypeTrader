"""Phase I best checkpoint 选择策略 + guardrail（从 checkpoint manager 抽出）.

设计文档锚点: §4.15。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

from src.phase1.config import SelectionPolicyConfig
from src.phase1.evaluation.metrics import phase1_composite_score


SelectionDecision = Literal["promote_to_best", "reject", "keep_as_periodic", "fatal", "skipped"]


@dataclass
class SelectionVerdict:
    decision: SelectionDecision
    reasons: List[str] = field(default_factory=list)
    composite_score: float = 0.0
    composite_score_debug: dict = field(default_factory=dict)


@dataclass
class SelectionHistory:
    best_score: Optional[float] = None
    best_epoch: Optional[int] = None
    consecutive_collapse_epochs: int = 0
    last_dead_code_restart_epoch: Optional[int] = None


class Phase1SelectionPolicy:
    """计算 ``SelectionVerdict`` 的纯逻辑对象（无 IO）。

    使用方式::

        policy = Phase1SelectionPolicy(config.selection_policy)
        history = SelectionHistory()
        for epoch_metrics in trainer.iter_epochs():
            verdict = policy.evaluate(epoch_metrics, history)
            history = policy.update_history(history, epoch_metrics, verdict)
            checkpoint_manager.commit_verdict(state, epoch_metrics, verdict, epoch)

    边界
    ----
    - 无 IO；只接收 metrics 与历史 best，返回 verdict。
    - 不重新计算指标；阈值全部来自 ``SelectionPolicyConfig``。
    - 与 ``Phase1Evaluator`` 解耦: evaluator 输出指标，policy 把指标翻译成 best 决策。
    """

    def __init__(self, config: SelectionPolicyConfig) -> None:
        self.config = config

    # ---------- 主入口 ----------

    def evaluate(self, metrics: dict, history: SelectionHistory) -> SelectionVerdict:
        """根据 metrics 生成 verdict。

        Steps
        -----
        1. 检查 dead_code_restart 冷却期 → 若处于冷却期，``decision="keep_as_periodic"``。
        2. ``should_block_due_to_codebook`` → reject。
        3. ``should_block_due_to_risk`` → reject。
        4. ``should_block_due_to_behavior`` → reject。
        5. ``should_block_due_to_teacher_quality`` → 仅记 warning，不 reject。
        6. 若未被 reject 且 ``score > history.best_score`` → ``promote_to_best``。
        7. 检查 ``consecutive_collapse_epoch_limit`` → 命中即返回 ``fatal``。

        Notes
        -----
        - 多 guardrail 同时触发时，所有原因都会进入 ``verdict.reasons``，
          便于 manifest 记录。
        - 调用方需在 metrics 中注入 ``_consecutive_collapse_epochs`` 与
          ``_consecutive_collapse_limit``，由 trainer 维护。
        """
        score, debug = self.compute_composite_score(metrics)
        verdict = SelectionVerdict(
            decision="keep_as_periodic",
            composite_score=score,
            composite_score_debug=debug,
        )

        # 1. dead_code restart 冷却期。
        # cooldown 默认 3 epoch（设计 §6.5 ``restart_cooldown_epochs``）；
        # 真实值可由 trainer 通过 ``metrics["_dead_code_restart_cooldown_epochs"]`` 注入，
        # 否则使用默认。trainer 触发 ``quantizer.restart_dead_codes`` 后会在
        # ``update_history`` 中记录 ``last_dead_code_restart_epoch``。
        epoch = int(metrics.get("epoch", history.best_epoch or 0))
        cooldown = int(metrics.get("_dead_code_restart_cooldown_epochs", 3))
        if (
            history.last_dead_code_restart_epoch is not None
            and (epoch - history.last_dead_code_restart_epoch) <= cooldown
        ):
            verdict.decision = "keep_as_periodic"
            verdict.reasons.append("dead_code_restart_cooldown")
            return verdict

        # 2-5. guardrails
        for check_fn, reason_label in (
            (self.should_block_due_to_codebook, "codebook_collapse"),
            (self.should_block_due_to_risk, "risk_guardrail"),
            (self.should_block_due_to_behavior, "behavior_guardrail"),
        ):
            blocked, detail = check_fn(metrics)
            if blocked:
                verdict.decision = "reject"
                verdict.reasons.append(f"{reason_label}: {detail}")

        # teacher quality 不阻塞 promote，但记录 warning
        warn_teacher, detail_teacher = self.should_block_due_to_teacher_quality(metrics)
        if warn_teacher:
            verdict.reasons.append(f"teacher_quality_warning: {detail_teacher}")

        # 6. 复合分；如未被拒绝则比较历史 best
        if verdict.decision != "reject":
            if history.best_score is None or score > history.best_score:
                verdict.decision = "promote_to_best"

        # 7. consecutive_collapse: 在 metrics 中由 trainer 注入字段
        consec = int(metrics.get("_consecutive_collapse_epochs", 0))
        limit = int(metrics.get("_consecutive_collapse_limit", 10))
        if consec >= limit:
            verdict.decision = "fatal"
            verdict.reasons.append(
                f"consecutive_collapse_epoch_limit: {consec} >= {limit}"
            )
        return verdict

    # ---------- guardrail ----------

    def should_block_due_to_codebook(self, metrics: dict) -> Tuple[bool, Optional[str]]:
        """``code_usage_ratio < min_code_usage_ratio`` 时阻止 best。"""
        usage = float(metrics.get("code_usage_ratio", 1.0))
        if usage < self.config.min_code_usage_ratio:
            return True, f"code_usage_ratio={usage:.3f} < {self.config.min_code_usage_ratio}"
        return False, None

    def should_block_due_to_risk(self, metrics: dict) -> Tuple[bool, Optional[str]]:
        """``val_max_drawdown > risk.max_drawdown`` 或
        ``val_sharpe_ratio < risk.min_sharpe_ratio`` 时阻止 best。"""
        mdd = float(metrics.get("val_max_drawdown", 0.0))
        sharpe = float(metrics.get("val_sharpe_ratio", 0.0))
        if mdd > self.config.risk.max_drawdown:
            return True, f"val_max_drawdown={mdd:.3f} > {self.config.risk.max_drawdown}"
        if sharpe < self.config.risk.min_sharpe_ratio:
            return True, f"val_sharpe_ratio={sharpe:.3f} < {self.config.risk.min_sharpe_ratio}"
        return False, None

    def should_block_due_to_behavior(self, metrics: dict) -> Tuple[bool, Optional[str]]:
        """``inter_code_action_diversity`` / ``decoder_sensitivity_to_code`` /
        epoch 稳定性任一低于阈值即阻止 best；若稳定性尚未被实际测量
        （例如首次 VQ full validation），同样阻止 best。

        若 evaluator 提供 ``epoch_code_stability_matched``，guardrail 优先使用
        matched 一致率，避免纯 code-id 交换被误判为真实标签漂移；历史 metrics
        没有该字段时回退到原始 ``epoch_code_stability``。

        语义: codebook 在 latent space 看似分开，但 decoder 对 ``z_q`` 不敏感时，
        不同 archetype 实际产生几乎相同的 actions——这种 checkpoint 不能 sign-off。
        """
        diversity = float(metrics.get("inter_code_action_diversity", 1.0))
        sensitivity = float(metrics.get("decoder_sensitivity_to_code", 1.0))
        stability_measured = bool(metrics.get("epoch_code_stability_measured", True))
        stability_key = (
            "epoch_code_stability_matched"
            if "epoch_code_stability_matched" in metrics
            else "epoch_code_stability"
        )
        stability = float(
            metrics.get(stability_key, metrics.get("epoch_code_stability", 1.0))
        )
        if diversity < self.config.behavior.min_inter_code_action_diversity:
            return True, f"inter_code_action_diversity={diversity:.3f} < {self.config.behavior.min_inter_code_action_diversity}"
        if sensitivity < self.config.behavior.min_decoder_sensitivity_to_code:
            return True, f"decoder_sensitivity_to_code={sensitivity:.3f} < {self.config.behavior.min_decoder_sensitivity_to_code}"
        if not stability_measured:
            return True, "epoch_code_stability_measured=false; wait_for_next_full_validation"
        if stability < self.config.behavior.min_epoch_code_stability:
            return True, f"{stability_key}={stability:.3f} < {self.config.behavior.min_epoch_code_stability}"
        return False, None

    def should_block_due_to_teacher_quality(self, metrics: dict) -> Tuple[bool, Optional[str]]:
        """``val_dp_teacher_profitable_ratio`` 低于阈值时返回 ``True``。

        Notes
        -----
        当前实现 reject_status 只是 *warning*——主流程在 ``evaluate`` 中
        把它写进 ``verdict.reasons`` 但不阻止 promote；逻辑是：teacher 弱时
        ``return_capture_ratio`` 不该作为 tie-breaker，但 checkpoint 仍可能其他
        guardrail 全部通过。如需严格阻塞，请把 ``evaluate`` 的处理逻辑改为
        ``reject``。
        """
        ratio = float(metrics.get("val_dp_teacher_profitable_ratio", 1.0))
        if ratio < self.config.teacher.min_dp_teacher_profitable_ratio:
            return True, f"val_dp_teacher_profitable_ratio={ratio:.3f} < {self.config.teacher.min_dp_teacher_profitable_ratio}"
        return False, None

    # ---------- 历史维护 ----------

    def update_history(
        self,
        history: SelectionHistory,
        metrics: dict,
        verdict: SelectionVerdict,
    ) -> SelectionHistory:
        """根据本 epoch 结果更新历史。

        - ``promote_to_best`` → 更新 ``best_score`` / ``best_epoch``。
        - codebook collapse 计数维护: 当前 epoch usage 低于阈值则累加 +1，
          否则清零。
        - ``_dead_code_restart_triggered=True`` 由 trainer 在调用 quantizer
          ``restart_dead_codes`` 后注入，便于 cooldown 启动。
        """
        new_history = SelectionHistory(
            best_score=history.best_score,
            best_epoch=history.best_epoch,
            consecutive_collapse_epochs=history.consecutive_collapse_epochs,
            last_dead_code_restart_epoch=history.last_dead_code_restart_epoch,
        )
        epoch = int(metrics.get("epoch", history.best_epoch or 0))
        if verdict.decision == "promote_to_best":
            new_history.best_score = verdict.composite_score
            new_history.best_epoch = epoch
        # collapse 计数维护
        if metrics.get("code_usage_ratio", 1.0) < self.config.min_code_usage_ratio:
            new_history.consecutive_collapse_epochs = history.consecutive_collapse_epochs + 1
        else:
            new_history.consecutive_collapse_epochs = 0
        if metrics.get("_dead_code_restart_triggered"):
            new_history.last_dead_code_restart_epoch = epoch
        return new_history

    # ---------- 复合分 ----------

    def compute_composite_score(self, metrics: dict) -> Tuple[float, dict]:
        """返回 ``(score, debug_info)``。

        ``debug_info`` 含每项指标值与权重，供 ``composite_score_sensitivity.json``
        与 manifest 审计追溯使用。
        """
        return phase1_composite_score(metrics, self.config.metric_weights)
