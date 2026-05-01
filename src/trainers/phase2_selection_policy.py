"""Phase II best selector 选择策略与 guardrails。

设计文档锚点: Phase II 执行计划 §Step 6。

职责:
- 基于 val 主指标做 best verdict。
- guardrails: max_drawdown / min_sharpe / max_turnover_ratio /
  max_action_dominance_ratio / min_active_archetype_ratio。
- val_kl_to_demo / phase1_demo_label_selector_val_net_return 仅作 diagnostic。
- rolling validation sign-off 附加硬约束。

镜像 Phase I 的 Phase1SelectionPolicy 风格。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from src.config.phase2_config import Phase2SelectionPolicyConfig


Phase2SelectionDecision = Literal["promote_to_best", "reject", "keep"]


@dataclass
class Phase2SelectionVerdict:
    """选择决策。"""
    decision: Phase2SelectionDecision
    reasons: List[str] = field(default_factory=list)
    primary_metric_value: float = 0.0
    selection_metric_name: str = ""
    composite_score: Optional[float] = None


@dataclass
class Phase2SelectionHistory:
    """选择历史。"""
    best_metric: Optional[float] = None
    best_update_idx: Optional[int] = None


class Phase2SelectionPolicy:
    """Phase II best selector 选择策略。

    使用方式::

        policy = Phase2SelectionPolicy(config.selection_policy)
        history = Phase2SelectionHistory()
        verdict = policy.evaluate(metrics, history)
        history = policy.update_history(history, metrics, verdict)
    """

    def __init__(self, config: Phase2SelectionPolicyConfig) -> None:
        self.config = config

    def evaluate(
        self,
        metrics: Dict[str, float],
        history: Phase2SelectionHistory,
        rolling_result: Optional[Dict[str, float]] = None,
    ) -> Phase2SelectionVerdict:
        """根据 metrics 生成 verdict。

        Parameters
        ----------
        metrics : 当前 update 的评估指标。
        history : 历史 best 状态。
        rolling_result : rolling validation 结果（可选）。

        Returns
        -------
        Phase2SelectionVerdict : 选择决策。
        """
        metric_name = self.config.selection_metric
        if metric_name in metrics:
            primary_value = float(metrics.get(metric_name, 0.0))
        else:
            metric_name = self.config.primary_metric
            primary_value = float(metrics.get(metric_name, 0.0))
        verdict = Phase2SelectionVerdict(
            decision="keep",
            primary_metric_value=primary_value,
            selection_metric_name=metric_name,
            composite_score=(
                float(metrics["phase2_composite_score"])
                if "phase2_composite_score" in metrics
                else None
            ),
        )
        if metric_name != self.config.selection_metric:
            verdict.reasons.append(
                f"selection_metric_missing:{self.config.selection_metric};fallback:{metric_name}"
            )

        # 检查所有 guardrails
        checks = [
            self.should_block_due_to_drawdown(metrics),
            self.should_block_due_to_sharpe(metrics),
            self.should_block_due_to_turnover(metrics),
            self.should_block_due_to_action_dominance(metrics),
            self.should_block_due_to_active_archetype(metrics),
        ]

        for blocked, reason in checks:
            if blocked and reason:
                verdict.decision = "reject"
                verdict.reasons.append(reason)

        # 如果未被拒绝，检查是否优于历史 best
        if verdict.decision != "reject":
            if self.config.primary_mode == "max":
                is_better = (
                    history.best_metric is None
                    or primary_value > history.best_metric
                )
            else:
                is_better = (
                    history.best_metric is None
                    or primary_value < history.best_metric
                )
            if is_better:
                verdict.decision = "promote_to_best"

        return verdict

    def should_block_due_to_drawdown(
        self, metrics: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """max_drawdown 超阈拒绝 best。"""
        mdd = metrics.get("max_drawdown", 0.0)
        if mdd > self.config.max_drawdown:
            return True, f"max_drawdown={mdd:.3f} > {self.config.max_drawdown}"
        return False, None

    def should_block_due_to_sharpe(
        self, metrics: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """min_sharpe 不足拒绝 best。"""
        sharpe = metrics.get("sharpe_ratio", 0.0)
        if sharpe < self.config.min_sharpe:
            return True, f"sharpe_ratio={sharpe:.3f} < {self.config.min_sharpe}"
        return False, None

    def should_block_due_to_turnover(
        self, metrics: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """max_turnover_ratio 超阈拒绝 best。"""
        turnover = metrics.get("turnover", 0.0)
        if turnover > self.config.max_turnover_ratio:
            return True, f"turnover={turnover:.3f} > {self.config.max_turnover_ratio}"
        return False, None

    def should_block_due_to_action_dominance(
        self, metrics: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """action_dominance 过高拒绝 best。"""
        dominance = metrics.get("action_dominance_ratio", 0.0)
        if dominance > self.config.max_action_dominance_ratio:
            return True, f"action_dominance_ratio={dominance:.3f} > {self.config.max_action_dominance_ratio}"
        return False, None

    def should_block_due_to_active_archetype(
        self, metrics: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """active_archetype_ratio 过低拒绝 best。"""
        active = metrics.get("active_archetype_ratio", 1.0)
        if active < self.config.min_active_archetype_ratio:
            return True, f"active_archetype_ratio={active:.3f} < {self.config.min_active_archetype_ratio}"
        return False, None

    def update_history(
        self,
        history: Phase2SelectionHistory,
        metrics: Dict[str, float],
        verdict: Phase2SelectionVerdict,
    ) -> Phase2SelectionHistory:
        """更新选择历史。"""
        new_history = Phase2SelectionHistory(
            best_metric=history.best_metric,
            best_update_idx=history.best_update_idx,
        )
        if verdict.decision == "promote_to_best":
            new_history.best_metric = verdict.primary_metric_value
            new_history.best_update_idx = int(metrics.get("update_idx", 0))
        return new_history
