"""Codebook collapse 处理 (轻量集成测试).

完整 fatal 行为已在 selection_policy 单元测试与 checkpoint 单元测试覆盖；
这里只做高层接口断言，避免长跑 trainer。
"""
from __future__ import annotations

import pytest

from src.phase1.config import (
    SelectionPolicyConfig,
)
from src.phase1.training.selection_policy import (
    Phase1SelectionPolicy,
    SelectionHistory,
)


def _good_metrics(extra: dict | None = None):
    m = {
        "epoch": 1,
        "code_usage_ratio": 0.9,
        "val_max_drawdown": 0.1,
        "val_sharpe_ratio": 1.0,
        "inter_code_action_diversity": 0.5,
        "decoder_sensitivity_to_code": 0.5,
        "epoch_code_stability": 0.9,
        "val_dp_teacher_profitable_ratio": 0.5,
    }
    if extra:
        m.update(extra)
    return m


def test_consecutive_collapse_eventually_fatal():
    policy = Phase1SelectionPolicy(SelectionPolicyConfig(min_code_usage_ratio=0.7))
    history = SelectionHistory()
    metrics = _good_metrics({
        "code_usage_ratio": 0.5,
        "_consecutive_collapse_epochs": 11,
        "_consecutive_collapse_limit": 10,
    })
    verdict = policy.evaluate(metrics, history)
    assert verdict.decision == "fatal"


def test_recovery_resets_collapse_counter():
    policy = Phase1SelectionPolicy(SelectionPolicyConfig(min_code_usage_ratio=0.7))
    history = SelectionHistory(consecutive_collapse_epochs=5)
    metrics = _good_metrics()
    verdict = policy.evaluate(metrics, history)
    new_history = policy.update_history(history, metrics, verdict)
    assert new_history.consecutive_collapse_epochs == 0
