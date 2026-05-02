"""Phase I processed store tests."""
from __future__ import annotations

import pytest

from src.data.horizon_builder import HorizonRecord
from src.data.phase1_processed_store import (
    Phase1ProcessedStore,
    Phase1ProcessedStoreError,
    stable_hash,
)
from src.planners.demo_generator import RejectStats
from src.trading.cost_model import ExecutionBook
from src.utils.feather_io import atomic_write_json, read_json, write_ipc


def _book(mark: float = 100.0) -> ExecutionBook:
    return ExecutionBook(
        ask_prices=(mark + 0.1,) * 5,
        ask_sizes=(10.0,) * 5,
        bid_prices=(mark - 0.1,) * 5,
        bid_sizes=(10.0,) * 5,
        mark_price=mark,
    )


def _record(sample_id: str = "s0", split: str = "train", h: int = 4):
    return HorizonRecord(
        sample_id=sample_id,
        start_index=0,
        end_index=h - 1,
        pair="TEST",
        split=split,
        strata_label="up|low|mixed",
        states=[[float(i)] for i in range(h)],
        prices=[100.0] * (h + 1),
        execution_books=[_book(100.0 + i) for i in range(h)],
        actions=[0, 1, 2, 1][:h],
        rewards=[0.1, 0.0, -0.1, 0.2][:h],
    )


def _manifest(tmp_path, *, schema_hash: str, data_hash: str, teacher_hash: str, n: int):
    schema = {
        "timestamp_column": "timestamp",
        "price_column": "close",
        "feature_columns": ["feature_a"],
        "excluded_columns": ["timestamp", "close"],
        "orderbook_columns": [],
        "num_rows": 10,
        "feature_source": {"mode": "unit"},
    }
    actual_schema_hash = stable_hash(schema)
    assert actual_schema_hash == schema_hash
    atomic_write_json(schema, tmp_path / "input_schema.json")
    payload = {
        "version": 1,
        "phase": "phase1_data_process",
        "pair": "TEST",
        "data_batch_id": "processed",
        "artifact_dir": str(tmp_path),
        "created_at": "2026-05-02T00:00:00Z",
        "input_files": {"train": "train.feather", "val": "val.feather", "test": "test.feather"},
        "input_schema_path": "input_schema.json",
        "schema_hash": schema_hash,
        "data_process_hash": data_hash,
        "dp_teacher_hash": teacher_hash,
        "feature_source": {"mode": "unit"},
        "splits": {
            "train": {
                "window_index_path": "window_index_train.feather",
                "sampled_horizons_path": "sampled_horizons_train.feather",
                "dp_teacher_path": "dp_teacher_train.feather",
                "num_horizons": n,
            },
            "val": {
                "window_index_path": "window_index_val.feather",
                "sampled_horizons_path": "sampled_horizons_train.feather",
                "dp_teacher_path": "dp_teacher_train.feather",
                "num_horizons": n,
            },
            "test": {
                "window_index_path": "window_index_test.feather",
                "sampled_horizons_path": "sampled_horizons_train.feather",
                "dp_teacher_path": "dp_teacher_train.feather",
                "num_horizons": n,
            },
        },
    }
    return atomic_write_json(payload, tmp_path / "data_process_manifest.json")


def _write_valid_store(tmp_path, records=None):
    records = records or [_record("a"), _record("b")]
    store = Phase1ProcessedStore(tmp_path)
    schema_hash = stable_hash(
        {
            "timestamp_column": "timestamp",
            "price_column": "close",
            "feature_columns": ["feature_a"],
            "excluded_columns": ["timestamp", "close"],
            "orderbook_columns": [],
            "num_rows": 10,
            "feature_source": {"mode": "unit"},
        }
    )
    data_hash = "datahash"
    teacher_hash = "teacherhash"
    store.save_sampled_horizons(
        "train", records, schema_hash=schema_hash, data_process_hash=data_hash
    )
    store.save_dp_teacher(
        "train",
        records,
        RejectStats(
            dataset_reject_rate=0.0,
            per_horizon_reject_count=[0 for _ in records],
            per_horizon_reject_rate=[0.0 for _ in records],
        ),
        schema_hash=schema_hash,
        data_process_hash=data_hash,
        dp_teacher_hash=teacher_hash,
    )
    manifest = _manifest(
        tmp_path,
        schema_hash=schema_hash,
        data_hash=data_hash,
        teacher_hash=teacher_hash,
        n=len(records),
    )
    return store, manifest


def test_processed_store_saves_and_loads_records(tmp_path):
    store, manifest = _write_valid_store(tmp_path)

    loaded = store.load_records(manifest, "train")

    assert {r.sample_id for r in loaded} == {"a", "b"}
    assert loaded[0].actions is not None
    assert loaded[0].rewards is not None
    assert len(loaded[0].execution_books) == 4


def test_processed_store_joins_teacher_by_sample_id(tmp_path):
    store, manifest = _write_valid_store(tmp_path, records=[_record("z")])

    loaded = store.load_records(manifest, "train")

    assert loaded[0].sample_id == "z"
    assert loaded[0].actions == [0, 1, 2, 1]


def test_processed_store_rejects_missing_teacher_sample(tmp_path):
    store, manifest = _write_valid_store(tmp_path, records=[_record("a"), _record("b")])
    teacher = store.load_manifest(manifest).resolve("dp_teacher_train.feather")
    import polars as pl

    frame = pl.read_ipc(teacher).filter(pl.col("sample_id") == "a")
    write_ipc(frame, teacher)

    with pytest.raises(Phase1ProcessedStoreError, match="sample_id mismatch"):
        store.load_records(manifest, "train")


def test_processed_store_rejects_extra_teacher_sample(tmp_path):
    store, manifest = _write_valid_store(tmp_path, records=[_record("a")])
    teacher = store.load_manifest(manifest).resolve("dp_teacher_train.feather")
    import polars as pl

    frame = pl.read_ipc(teacher)
    extra = frame.with_columns(pl.lit("extra").alias("sample_id"))
    write_ipc(pl.concat([frame, extra], how="vertical"), teacher)

    with pytest.raises(Phase1ProcessedStoreError, match="sample_id mismatch"):
        store.load_records(manifest, "train")


def test_processed_store_rejects_hash_mismatch(tmp_path):
    store, manifest = _write_valid_store(tmp_path)
    payload = read_json(manifest)
    payload["dp_teacher_hash"] = "different"
    atomic_write_json(payload, manifest)

    with pytest.raises(Phase1ProcessedStoreError, match="_dp_teacher_hash"):
        store.load_records(manifest, "train")


def test_processed_store_rejects_wrong_split(tmp_path):
    store, manifest = _write_valid_store(tmp_path, records=[_record("a", split="val")])

    with pytest.raises(Phase1ProcessedStoreError, match="column split mismatch"):
        store.load_records(manifest, "train")


def test_processed_store_rejects_action_reward_length_mismatch(tmp_path):
    bad = _record("a")
    bad.actions = [1]
    store, manifest = _write_valid_store(tmp_path, records=[bad])

    with pytest.raises(Phase1ProcessedStoreError, match="length mismatch"):
        store.load_records(manifest, "train")
