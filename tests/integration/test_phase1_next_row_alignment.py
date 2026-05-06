"""验证 ``next_row_execution`` 行号映射的关键指标."""
from __future__ import annotations

import pytest

from src.preprocess_data.window_indexer import SlidingWindowIndexer


def _frame(rows: int = 20):
    pl = pytest.importorskip("polars")
    return pl.DataFrame({
        "timestamp": list(range(rows)),
        "close": [100.0 + i * 0.1 for i in range(rows)],
    })


def test_next_row_yields_n_minus_h_minus_1_candidates():
    indexer = SlidingWindowIndexer(horizon=8, reward_alignment="next_row_execution")
    entries = indexer.enumerate(_frame(20), stratification_mode="hindsight_horizon")
    assert len(entries) == 11


def test_next_row_split_boundary_embargo_default():
    """next_row_execution 默认 embargo 必须为 ``h+2``。"""
    from src.config.phase1_config import SamplingHealthConfig

    cfg = SamplingHealthConfig()
    h = 72
    assert cfg.split_boundary_embargo == h + 1
    assert cfg.next_row_split_boundary_embargo == h + 2
