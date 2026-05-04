"""``SlidingWindowIndexer`` 单元测试."""
from __future__ import annotations

import math

import pytest

from src.data.window_indexer import (
    SlidingWindowIndexer,
    _compute_past_stats,
    _compute_window_stats,
)


def _make_frame(rows: int = 20):
    import polars as pl

    return pl.DataFrame({
        "timestamp": list(range(rows)),
        "close": [100.0 + i * 0.1 for i in range(rows)],
    })


def test_paper_formula_yields_n_minus_h_candidates():
    indexer = SlidingWindowIndexer(horizon=8, reward_alignment="paper_formula")
    entries = indexer.enumerate(_make_frame(20), stratification_mode="hindsight_horizon")
    assert len(entries) == 12  # 20 - 8


def test_next_row_yields_n_minus_h_minus_1_candidates():
    indexer = SlidingWindowIndexer(horizon=8, reward_alignment="next_row_execution")
    entries = indexer.enumerate(_make_frame(20), stratification_mode="hindsight_horizon")
    assert len(entries) == 11  # 20 - 8 - 1


def test_last_execution_and_markout_rows_paper():
    indexer = SlidingWindowIndexer(horizon=4, reward_alignment="paper_formula")
    entries = indexer.enumerate(_make_frame(10), stratification_mode="hindsight_horizon")
    e = entries[0]
    assert e.window_start == 0
    assert e.window_end == 3
    assert e.last_execution_row == 3
    assert e.last_markout_row == 4


def test_last_execution_and_markout_rows_next_row():
    indexer = SlidingWindowIndexer(horizon=4, reward_alignment="next_row_execution")
    entries = indexer.enumerate(_make_frame(10), stratification_mode="hindsight_horizon")
    e = entries[0]
    assert e.last_execution_row == 4
    assert e.last_markout_row == 5


def test_hindsight_horizon_return_uses_t_plus_h_markout():
    indexer = SlidingWindowIndexer(horizon=4, reward_alignment="paper_formula")
    entries = indexer.enumerate(_make_frame(10), stratification_mode="hindsight_horizon")
    e = entries[0]
    assert e.horizon_return == pytest.approx((100.4 - 100.0) / 100.0)


def test_prospective_past_return_includes_current_start_row():
    indexer = SlidingWindowIndexer(
        horizon=2,
        reward_alignment="paper_formula",
        prospective_lookback_minutes=2,
    )
    entries = indexer.enumerate(_make_frame(10), stratification_mode="prospective_past")
    e = entries[2]
    assert e.past_return == pytest.approx((100.2 - 100.0) / 100.0)


def test_invalid_alignment_raises():
    with pytest.raises(ValueError):
        SlidingWindowIndexer(horizon=8, reward_alignment="bad")  # type: ignore[arg-type]


def test_vectorized_stats_match_scalar_reference_on_edge_prices():
    import polars as pl

    close = [
        100.0,
        100.2,
        99.9,
        100.4,
        100.1,
        0.0,
        100.3,
        100.7,
        100.5,
        100.8,
        101.0,
        100.6,
        100.9,
        -1.0,
        101.2,
        101.4,
    ]
    horizon = 4
    lookback = 3
    indexer = SlidingWindowIndexer(
        horizon=horizon,
        reward_alignment="paper_formula",
        prospective_lookback_minutes=lookback,
    )
    entries = indexer.enumerate(
        pl.DataFrame({"close": close}),
        stratification_mode="prospective_past",
    )

    for start, entry in enumerate(entries):
        horizon_ret, realized_vol, draw, _, _ = _compute_window_stats(
            close, start, horizon
        )
        past_ret, past_vol, past_draw = _compute_past_stats(close, start, lookback)
        if math.isnan(horizon_ret):
            assert math.isnan(entry.horizon_return)
        else:
            assert entry.horizon_return == pytest.approx(horizon_ret)
        if math.isnan(realized_vol):
            assert math.isnan(entry.realized_volatility)
        else:
            assert entry.realized_volatility == pytest.approx(realized_vol)
        assert entry.draw_pattern == draw
        if math.isnan(past_ret):
            assert math.isnan(entry.past_return)
        else:
            assert entry.past_return == pytest.approx(past_ret)
        if math.isnan(past_vol):
            assert math.isnan(entry.past_realized_volatility)
        else:
            assert entry.past_realized_volatility == pytest.approx(past_vol)
        assert entry.past_draw_pattern == past_draw
