"""``SlidingWindowIndexer`` 单元测试."""
from __future__ import annotations

import pytest

from src.data.window_indexer import SlidingWindowIndexer


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
