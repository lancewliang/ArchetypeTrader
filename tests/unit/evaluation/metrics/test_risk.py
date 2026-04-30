"""``src.evaluation.metrics.risk`` 单元测试."""
from __future__ import annotations

import pytest

from src.evaluation.metrics.risk import (
    calmar_ratio,
    equity_curve_from_step_returns,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def test_sharpe_zero_when_constant():
    assert sharpe_ratio([0.001, 0.001, 0.001]) == 0.0


def test_sharpe_positive_for_consistent_positive_returns():
    s = sharpe_ratio([0.001, 0.0009, 0.0011, 0.0012], annualization_factor=525_600)
    assert s > 0


def test_sortino_zero_when_no_downside():
    assert sortino_ratio([0.001, 0.0001, 0.002]) == 0.0


def test_sortino_uses_only_downside_std():
    val = sortino_ratio([0.001, -0.002, 0.003, -0.001])
    assert val != 0.0


def test_max_drawdown_zero_when_monotonic_up():
    curve = [1.0, 1.01, 1.02, 1.03]
    assert max_drawdown(curve) == 0.0


def test_max_drawdown_positive_when_drop():
    curve = [1.0, 1.1, 0.9, 1.0]
    mdd = max_drawdown(curve)
    assert mdd == pytest.approx((1.1 - 0.9) / 1.1, rel=1e-3)


def test_calmar_zero_when_no_drawdown():
    assert calmar_ratio(0.5, 0.0) == 0.0


def test_equity_curve_starts_at_one_plus_first_return():
    curve = equity_curve_from_step_returns([0.01, 0.02])
    assert curve[0] == pytest.approx(1.01)
    assert curve[-1] == pytest.approx(1.03)
