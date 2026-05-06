"""``Phase1SelectionPolicy`` 单元测试 (高风险修复点)."""
from __future__ import annotations

import pytest

from src.phase1.config import (
    BehaviorGuardrailConfig,
    RiskGuardrailConfig,
    SelectionPolicyConfig,
    TeacherQualityGuardrailConfig,
)
from src.phase1.training.selection_policy import (
    Phase1SelectionPolicy,
    SelectionHistory,
)


def _policy() -> Phase1SelectionPolicy:
    cfg = SelectionPolicyConfig(
        min_code_usage_ratio=0.7,
        risk=RiskGuardrailConfig(max_drawdown=0.2, min_sharpe_ratio=0.0),
        behavior=BehaviorGuardrailConfig(
            min_inter_code_action_diversity=0.15,
            min_decoder_sensitivity_to_code=0.05,
            min_epoch_code_stability=0.8,
        ),
        teacher=TeacherQualityGuardrailConfig(min_dp_teacher_profitable_ratio=0.3),
    )
    return Phase1SelectionPolicy(cfg)


def _good_metrics(extra: dict | None = None):
    m = {
        "epoch": 1,
        "code_usage_ratio": 0.9,
        "val_max_drawdown": 0.1,
        "val_sharpe_ratio": 1.0,
        "inter_code_action_diversity": 0.5,
        "decoder_sensitivity_to_code": 0.5,
        "epoch_code_stability_measured": True,
        "epoch_code_stability": 0.9,
        "val_dp_teacher_profitable_ratio": 0.5,
        "switch_point_recall": 0.7,
        "switch_direction_accuracy": 0.7,
        "val_weighted_reconstruction_accuracy": 0.7,
        "val_return_capture_ratio": 0.6,
    }
    if extra:
        m.update(extra)
    return m


def test_block_when_code_usage_below_threshold():
    policy = _policy()
    metrics = _good_metrics({"code_usage_ratio": 0.5})
    verdict = policy.evaluate(metrics, SelectionHistory())
    assert verdict.decision == "reject"
    assert any("codebook_collapse" in r for r in verdict.reasons)


def test_block_when_drawdown_exceeds():
    policy = _policy()
    metrics = _good_metrics({"val_max_drawdown": 0.3})
    verdict = policy.evaluate(metrics, SelectionHistory())
    assert verdict.decision == "reject"


def test_block_when_sharpe_below_threshold():
    policy = _policy()
    metrics = _good_metrics({"val_sharpe_ratio": -0.1})
    verdict = policy.evaluate(metrics, SelectionHistory())
    assert verdict.decision == "reject"


def test_block_when_inter_code_action_diversity_low():
    policy = _policy()
    metrics = _good_metrics({"inter_code_action_diversity": 0.05})
    verdict = policy.evaluate(metrics, SelectionHistory())
    assert verdict.decision == "reject"


def test_block_when_decoder_sensitivity_low():
    policy = _policy()
    metrics = _good_metrics({"decoder_sensitivity_to_code": 0.01})
    verdict = policy.evaluate(metrics, SelectionHistory())
    assert verdict.decision == "reject"


def test_block_when_code_stability_low():
    policy = _policy()
    metrics = _good_metrics({"epoch_code_stability": 0.5})
    verdict = policy.evaluate(metrics, SelectionHistory())
    assert verdict.decision == "reject"


def test_prefers_matched_code_stability_when_available():
    policy = _policy()
    metrics = _good_metrics({
        "epoch_code_stability": 0.1,
        "epoch_code_stability_matched": 0.9,
    })
    verdict = policy.evaluate(metrics, SelectionHistory())
    assert verdict.decision == "promote_to_best"


def test_block_when_matched_code_stability_low():
    policy = _policy()
    metrics = _good_metrics({
        "epoch_code_stability": 0.9,
        "epoch_code_stability_matched": 0.5,
    })
    verdict = policy.evaluate(metrics, SelectionHistory())
    assert verdict.decision == "reject"
    assert any("epoch_code_stability_matched=0.500" in r for r in verdict.reasons)


def test_block_when_code_stability_unmeasured():
    policy = _policy()
    metrics = _good_metrics({
        "epoch_code_stability_measured": False,
        "epoch_code_stability": 1.0,
    })
    verdict = policy.evaluate(metrics, SelectionHistory())
    assert verdict.decision == "reject"
    assert any("epoch_code_stability_measured=false" in r for r in verdict.reasons)


def test_warn_when_teacher_quality_low():
    policy = _policy()
    metrics = _good_metrics({"val_dp_teacher_profitable_ratio": 0.1})
    verdict = policy.evaluate(metrics, SelectionHistory())
    assert any("teacher_quality_warning" in r for r in verdict.reasons)


def test_promote_when_first_good_epoch():
    policy = _policy()
    verdict = policy.evaluate(_good_metrics(), SelectionHistory())
    assert verdict.decision == "promote_to_best"


def test_keep_periodic_when_score_lower():
    policy = _policy()
    history = SelectionHistory(best_score=999.0, best_epoch=0)
    verdict = policy.evaluate(_good_metrics(), history)
    assert verdict.decision == "keep_as_periodic"


def test_consecutive_collapse_triggers_fatal():
    policy = _policy()
    metrics = _good_metrics({
        "code_usage_ratio": 0.5,
        "_consecutive_collapse_epochs": 11,
        "_consecutive_collapse_limit": 10,
    })
    verdict = policy.evaluate(metrics, SelectionHistory())
    assert verdict.decision == "fatal"


def test_update_history_on_promote():
    policy = _policy()
    history = SelectionHistory()
    metrics = _good_metrics()
    verdict = policy.evaluate(metrics, history)
    new_history = policy.update_history(history, metrics, verdict)
    assert new_history.best_score == verdict.composite_score
    assert new_history.best_epoch == metrics["epoch"]
