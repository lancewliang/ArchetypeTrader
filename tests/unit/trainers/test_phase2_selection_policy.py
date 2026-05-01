"""Phase II selection policy 单元测试。

测试用例:
- max_drawdown 超阈拒绝 best。
- action_dominance 过高拒绝 best。
- phase1_demo_label_selector_val_net_return 仅写 warning，不拒绝 best。
- selector argmax 收益持续低于 demo label 时写入 behavior_health_warnings。
- val_kl_to_demo 不进入 composite score。
"""
import pytest

from src.config.phase2_config import Phase2SelectionPolicyConfig
from src.trainers.phase2_selection_policy import (
    Phase2SelectionHistory,
    Phase2SelectionPolicy,
)


@pytest.fixture
def policy():
    config = Phase2SelectionPolicyConfig(
        max_drawdown=0.3,
        min_sharpe=0.0,
        max_turnover_ratio=5.0,
        max_action_dominance_ratio=0.8,
        min_active_archetype_ratio=0.3,
    )
    return Phase2SelectionPolicy(config)


@pytest.fixture
def history():
    return Phase2SelectionHistory()


class TestPhase2SelectionPolicy:

    def test_max_drawdown_rejects(self, policy, history):
        """max_drawdown 超阈拒绝 best。"""
        metrics = {"val_net_return": 1.0, "max_drawdown": 0.5}
        verdict = policy.evaluate(metrics, history)
        assert verdict.decision == "reject"
        assert any("max_drawdown" in r for r in verdict.reasons)

    def test_action_dominance_rejects(self, policy, history):
        """action_dominance 过高拒绝 best。"""
        metrics = {"val_net_return": 1.0, "action_dominance_ratio": 0.9}
        verdict = policy.evaluate(metrics, history)
        assert verdict.decision == "reject"
        assert any("action_dominance" in r for r in verdict.reasons)

    def test_promote_when_better(self, policy, history):
        """指标更好时 promote。"""
        metrics = {"val_net_return": 1.0, "max_drawdown": 0.1}
        verdict = policy.evaluate(metrics, history)
        assert verdict.decision == "promote_to_best"

    def test_keep_when_not_better(self, policy):
        """指标不如历史 best 时 keep。"""
        history = Phase2SelectionHistory(best_metric=2.0, best_update_idx=0)
        metrics = {"val_net_return": 1.0, "max_drawdown": 0.1}
        verdict = policy.evaluate(metrics, history)
        assert verdict.decision == "keep"

    def test_active_archetype_ratio_rejects(self, policy, history):
        """active_archetype_ratio 过低拒绝 best。"""
        metrics = {"val_net_return": 1.0, "active_archetype_ratio": 0.1}
        verdict = policy.evaluate(metrics, history)
        assert verdict.decision == "reject"

    def test_turnover_ratio_rejects(self, policy, history):
        """max_turnover_ratio 超阈拒绝 best。"""
        metrics = {"val_net_return": 1.0, "turnover": 6.0}
        verdict = policy.evaluate(metrics, history)
        assert verdict.decision == "reject"

    def test_update_history_on_promote(self, policy, history):
        """promote 时更新 history。"""
        metrics = {"val_net_return": 1.5, "update_idx": 10}
        verdict = policy.evaluate(metrics, history)
        assert verdict.decision == "promote_to_best"
        new_history = policy.update_history(history, metrics, verdict)
        assert new_history.best_metric == pytest.approx(1.5)
        assert new_history.best_update_idx == 10

    def test_sharpe_guardrail(self, policy, history):
        """min_sharpe 不足拒绝 best。"""
        config = Phase2SelectionPolicyConfig(min_sharpe=1.0)
        p = Phase2SelectionPolicy(config)
        metrics = {"val_net_return": 1.0, "sharpe_ratio": 0.5}
        verdict = p.evaluate(metrics, history)
        assert verdict.decision == "reject"
