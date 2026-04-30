"""``src.evaluation.metrics.action`` 单元测试."""
from __future__ import annotations

import pytest

from src.evaluation.metrics.action import (
    action_confusion_matrix,
    non_flat_accuracy,
    reconstruction_accuracy,
    single_trade_consistency_rate,
    switch_metrics,
    weighted_reconstruction_accuracy,
)


def _logits_from_actions(actions):
    out = []
    for row in actions:
        row_logits = []
        for a in row:
            l = [0.0, 0.0, 0.0]
            l[a] = 10.0
            row_logits.append(l)
        out.append(row_logits)
    return out


def test_perfect_logits_yield_one():
    actions = [[1, 2, 2, 1]]
    assert reconstruction_accuracy(_logits_from_actions(actions), actions) == 1.0


def test_weighted_accuracy_balances_class_proportions():
    actions = [[1, 1, 1, 2]]
    pred_actions = [[1, 1, 1, 1]]  # 错了 long
    logits = _logits_from_actions(pred_actions)
    weights = {0: 2.0, 1: 1.0, 2: 2.0}
    score = weighted_reconstruction_accuracy(logits, actions, weights)
    # flat 三个 weight=1 全对 = 3；long 一个 weight=2 错了 = 0；分母 = 5；分子 = 3 → 0.6
    assert score == pytest.approx(0.6)


def test_non_flat_accuracy_only_evaluates_short_long():
    actions = [[1, 2, 1, 0]]
    pred_actions = [[1, 2, 1, 1]]  # short 错为 flat
    score = non_flat_accuracy(_logits_from_actions(pred_actions), actions)
    assert score == pytest.approx(0.5)


def test_confusion_matrix_basic():
    cm = action_confusion_matrix([[0, 1, 2]], [[0, 1, 1]])
    assert cm.matrix == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]


def test_switch_recall_when_match():
    sw = switch_metrics([[1, 1, 2, 2]], [[1, 1, 2, 2]])
    assert sw.switch_point_recall == 1.0
    assert sw.switch_direction_accuracy == 1.0
    assert sw.switch_timing_error_mean == 0.0


def test_switch_direction_wrong():
    sw = switch_metrics([[1, 1, 2, 2]], [[1, 1, 0, 0]])
    assert sw.switch_direction_accuracy == 0.0


def test_no_switch_in_both_treated_as_correct():
    sw = switch_metrics([[1, 1, 1, 1]], [[1, 1, 1, 1]])
    assert sw.switch_point_recall == 1.0


def test_single_trade_consistency_rate():
    pred = [[1, 1, 2, 2], [1, 0, 2, 0]]  # 第一行 1 切换，第二行 3 切换
    rate = single_trade_consistency_rate(pred)
    assert rate == 0.5
