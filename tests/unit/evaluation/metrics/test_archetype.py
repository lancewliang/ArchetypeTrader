"""``src.evaluation.metrics.archetype`` 单元测试."""
from __future__ import annotations

import pytest

from src.evaluation.metrics.archetype import (
    dp_teacher_quality,
    per_code_summary,
)


def test_per_code_summary_groups_by_code_id():
    diag = per_code_summary(
        horizon_returns=[0.1, -0.1, 0.0, 0.05, -0.02],
        code_ids=[0, 0, 1, 1, 2],
        no_trade_flags=[False, False, True, False, True],
        switch_points=[2, 3, -1, 4, -1],
    )
    code_to_count = {s.code_id: s.count for s in diag.per_code}
    assert code_to_count == {0: 2, 1: 2, 2: 1}


def test_no_trade_code_concentration_top1():
    diag = per_code_summary(
        horizon_returns=[0.0, 0.0, 0.0, 0.0],
        code_ids=[0, 0, 1, 1],
        no_trade_flags=[True, True, False, False],
        switch_points=[-1, -1, 2, 3],
    )
    assert diag.no_trade_code_concentration["top1"] >= 0.5


def test_active_trade_code_count():
    diag = per_code_summary(
        horizon_returns=[0.1, 0.2, 0.0, 0.0],
        code_ids=[0, 0, 1, 1],
        no_trade_flags=[False, False, True, True],
        switch_points=[2, 3, -1, -1],
    )
    assert diag.active_trade_code_count >= 1


def test_dp_teacher_profitable_ratio():
    quality = dp_teacher_quality(
        dp_horizon_returns=[0.1, -0.05, 0.02, -0.1, 0.05],
        dp_step_returns=[0.001, -0.001, 0.002, -0.002, 0.0005],
    )
    assert quality.val_dp_teacher_profitable_ratio == pytest.approx(3 / 5)


def test_dp_teacher_quality_distribution_keys():
    q = dp_teacher_quality(dp_horizon_returns=[0.1, 0.2, 0.3], dp_step_returns=[0.001])
    assert set(q.return_distribution) >= {"mean", "min", "max", "p50"}
