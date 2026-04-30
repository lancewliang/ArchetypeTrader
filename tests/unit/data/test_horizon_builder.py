"""``HorizonBuilder`` 单元测试."""
from __future__ import annotations

import pytest

from src.data.horizon_builder import HorizonBuilder
from src.data.schema import InputSchemaValidator
from src.data.stratified_sampler import SampledHorizon


def _frame(rows: int = 20):
    import polars as pl

    cols = {
        "timestamp": list(range(rows)),
        "close": [100.0 + i * 0.1 for i in range(rows)],
        "return_1m": [0.0] * rows,
        "ask1_price": [100.5 + i * 0.1 for i in range(rows)],
        "ask1_size": [10.0] * rows,
        "bid1_price": [99.5 + i * 0.1 for i in range(rows)],
        "bid1_size": [10.0] * rows,
    }
    for level in range(2, 6):
        cols[f"ask{level}_price"] = [100.5 + i * 0.1 + level * 0.05 for i in range(rows)]
        cols[f"ask{level}_size"] = [10.0] * rows
        cols[f"bid{level}_price"] = [99.5 + i * 0.1 - level * 0.05 for i in range(rows)]
        cols[f"bid{level}_size"] = [10.0] * rows
    return pl.DataFrame(cols)


def _sampled(start: int, h: int):
    return SampledHorizon(
        sample_id=f"s_{start}",
        window_start=start,
        window_end=start + h - 1,
        last_execution_row=start + h - 1,
        last_markout_row=start + h,
        strata_label="up|low|mixed",
    )


def test_states_shape_and_no_close_paper_formula():
    frame = _frame(20)
    schema = InputSchemaValidator().validate(frame)
    builder = HorizonBuilder(horizon=8, schema=schema, reward_alignment="paper_formula")
    records = builder.build(frame, [_sampled(0, 8)], pair="TEST", split="train")
    assert len(records) == 1
    rec = records[0]
    assert len(rec.states) == 8
    assert len(rec.states[0]) == schema.feature_dim()
    # close 不在 feature_dim 范围内
    assert "close" not in schema.feature_columns


def test_paper_formula_prices_length_h_plus_1():
    frame = _frame(20)
    schema = InputSchemaValidator().validate(frame)
    builder = HorizonBuilder(horizon=8, schema=schema, reward_alignment="paper_formula")
    rec = builder.build(frame, [_sampled(0, 8)], pair="TEST", split="train")[0]
    assert len(rec.prices) == 9


def test_next_row_prices_length_h_plus_2():
    frame = _frame(20)
    schema = InputSchemaValidator().validate(frame)
    builder = HorizonBuilder(horizon=8, schema=schema, reward_alignment="next_row_execution")
    sampled = SampledHorizon(
        sample_id="s_0",
        window_start=0,
        window_end=7,
        last_execution_row=8,
        last_markout_row=9,
        strata_label="up|low|mixed",
    )
    rec = builder.build(frame, [sampled], pair="TEST", split="train")[0]
    assert len(rec.prices) == 10
    assert rec.last_execution_row == 8
    assert rec.last_markout_row == 9


def test_sample_id_start_end_indices_consistent():
    frame = _frame(20)
    schema = InputSchemaValidator().validate(frame)
    builder = HorizonBuilder(horizon=8, schema=schema, reward_alignment="paper_formula")
    rec = builder.build(frame, [_sampled(2, 8)], pair="TEST", split="train")[0]
    assert rec.start_index == 2
    assert rec.end_index == 9
