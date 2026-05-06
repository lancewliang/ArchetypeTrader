"""``Phase1DemoStore`` 单元测试."""
from __future__ import annotations

import pytest

from src.preprocess_data.demo_store import HorizonLabel, Phase1DemoStore
from src.preprocess_data.horizon_builder import HorizonRecord
from src.trading.cost_model import ExecutionBook


def _book(mark: float = 100.0) -> ExecutionBook:
    return ExecutionBook(
        ask_prices=(mark + 0.1,) * 5,
        ask_sizes=(10.0,) * 5,
        bid_prices=(mark - 0.1,) * 5,
        bid_sizes=(10.0,) * 5,
        mark_price=mark,
    )


def _record(sample_id: str = "s0", h: int = 4):
    return HorizonRecord(
        sample_id=sample_id,
        start_index=0,
        end_index=h - 1,
        pair="TEST",
        split="train",
        strata_label="up|low|mixed",
        states=[[0.0] for _ in range(h)],
        prices=[100.0] * (h + 1),
        execution_books=[_book(100.0 + i) for i in range(h)],
        actions=[1] * h,
        rewards=[0.0] * h,
    )


def test_save_and_load_demos_roundtrip(tmp_path):
    store = Phase1DemoStore(tmp_path, config_hash="cfg", schema_hash="sch")
    path = store.save_demos([_record("a"), _record("b")])
    assert path.exists()
    loaded = store.load_demos()
    assert {r.sample_id for r in loaded} == {"a", "b"}
    assert len(loaded[0].execution_books) == 4
    assert loaded[0].execution_books[0].mark_price == 100.0


def test_load_with_mismatched_hash_raises(tmp_path):
    store = Phase1DemoStore(tmp_path, config_hash="cfg", schema_hash="sch")
    store.save_demos([_record("a")])
    other_store = Phase1DemoStore(tmp_path, config_hash="other", schema_hash="sch")
    with pytest.raises(ValueError):
        other_store.load_demos()


def test_save_horizon_labels_per_split(tmp_path):
    store = Phase1DemoStore(tmp_path, config_hash="cfg", schema_hash="sch")
    label = HorizonLabel(
        sample_id="x",
        start_index=0,
        end_index=7,
        last_execution_row=7,
        last_markout_row=8,
        strata_label="up|low|mixed",
        stratification_mode="hindsight_horizon",
        is_augmented=False,
        augmentation_type="none",
        code_label=2,
        demo_return=0.0,
        num_switches=1,
        is_no_trade=False,
    )
    path = store.save_labels([label], split="val")
    assert path.exists()
    loaded = store.load_labels("val")
    assert loaded[0].code_label == 2
