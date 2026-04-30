"""``src.evaluation.metrics.stability`` 单元测试."""
from __future__ import annotations

import pytest

from src.evaluation.metrics.stability import (
    codebook_displacement,
    epoch_code_stability,
    horizon_boundary_metrics,
    matched_epoch_code_stability,
)
from src.trading.cost_model import ExecutionBook, LobDepthCostModel


def test_epoch_code_stability_one_when_identical():
    assert epoch_code_stability([0, 1, 2], [0, 1, 2]) == 1.0


def test_matched_stability_higher_than_raw_when_swap():
    """last 把 best 的 code 0 ↔ 1 互换；matched 应当 > raw。"""
    best_codes = [0, 1, 0, 1]
    last_codes = [1, 0, 1, 0]
    best_cb = [[1.0, 0.0], [0.0, 1.0]]
    last_cb = [[0.0, 1.0], [1.0, 0.0]]  # 实际就是把 0,1 换了
    raw = epoch_code_stability(best_codes, last_codes)
    matched = matched_epoch_code_stability(best_codes, last_codes, best_cb, last_cb)
    assert matched > raw


def test_codebook_displacement_zero_when_unchanged():
    cb = [[1.0, 2.0], [3.0, 4.0]]
    out = codebook_displacement(cb, cb)
    for v in out.values():
        assert v == 0.0


def test_horizon_boundary_position_consistency_one_when_aligned():
    cm = LobDepthCostModel(commission_rate=0.0001)
    book = ExecutionBook(
        ask_prices=(101, 101.5, 102, 102.5, 103),
        ask_sizes=(10, 10, 10, 10, 10),
        bid_prices=(99, 98.5, 98, 97.5, 97),
        bid_sizes=(10, 10, 10, 10, 10),
        mark_price=100.0,
    )
    out = horizon_boundary_metrics(
        boundary_positions=[(1, 1), (-1, -1)],
        boundary_books=[book, book],
        cost_model=cm,
    )
    assert out["horizon_boundary_position_consistency"] == 1.0


def test_horizon_boundary_turnover_cost_when_misaligned():
    cm = LobDepthCostModel(commission_rate=0.0001)
    book = ExecutionBook(
        ask_prices=(101, 101.5, 102, 102.5, 103),
        ask_sizes=(10, 10, 10, 10, 10),
        bid_prices=(99, 98.5, 98, 97.5, 97),
        bid_sizes=(10, 10, 10, 10, 10),
        mark_price=100.0,
    )
    out = horizon_boundary_metrics(
        boundary_positions=[(1, -1)],
        boundary_books=[book],
        cost_model=cm,
    )
    assert out["horizon_boundary_turnover_cost"] > 0
