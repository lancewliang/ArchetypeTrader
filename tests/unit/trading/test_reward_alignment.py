"""``RewardAlignment`` 单元测试."""
from __future__ import annotations

import pytest

from src.trading.reward_alignment import AlignmentRows, RewardAlignment


def test_paper_formula_rows_at_t_zero():
    """Given paper_formula / When rows(0) / Then decision=0, execution=0, markout=1。"""
    align = RewardAlignment("paper_formula")
    rows = align.rows(0)
    assert isinstance(rows, AlignmentRows)
    assert rows == AlignmentRows(decision_row=0, execution_row=0, markout_row=1)


def test_paper_formula_rows_for_general_t():
    """Given paper_formula / When rows(5) / Then decision=5, execution=5, markout=6。"""
    align = RewardAlignment("paper_formula")
    rows = align.rows(5)
    assert rows == AlignmentRows(decision_row=5, execution_row=5, markout_row=6)


def test_next_row_execution_rows_at_t_zero():
    """Given next_row_execution / When rows(0) / Then decision=0, execution=1, markout=2。"""
    align = RewardAlignment("next_row_execution")
    rows = align.rows(0)
    assert rows == AlignmentRows(decision_row=0, execution_row=1, markout_row=2)


def test_next_row_execution_rows_for_general_t():
    align = RewardAlignment("next_row_execution")
    rows = align.rows(5)
    assert rows == AlignmentRows(decision_row=5, execution_row=6, markout_row=7)


def test_invalid_mode_raises():
    """Given mode='unknown' / When __init__ / Then ValueError。"""
    with pytest.raises(ValueError):
        RewardAlignment("unknown")  # type: ignore[arg-type]


def test_required_lookahead_rows():
    """paper_formula → 1，next_row_execution → 2。"""
    assert RewardAlignment("paper_formula").required_lookahead_rows() == 1
    assert RewardAlignment("next_row_execution").required_lookahead_rows() == 2
