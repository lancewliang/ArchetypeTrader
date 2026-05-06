"""``MarketFileReader`` 单元测试."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.preprocess_data.market_reader import MarketFileReader


def _build_simple_frame():
    import polars as pl

    return pl.DataFrame({"timestamp": [0, 1], "close": [100.0, 100.5]})


def test_should_return_polars_frame_when_reading_feather(tmp_path):
    pl_frame = _build_simple_frame()
    target = tmp_path / "x.feather"
    pl_frame.write_ipc(target)
    reader = MarketFileReader()
    result = reader.read(target)
    assert result.height == 2


def test_should_read_csv_for_debug(tmp_path):
    pl_frame = _build_simple_frame()
    target = tmp_path / "x.csv"
    pl_frame.write_csv(target)
    reader = MarketFileReader()
    result = reader.read(target)
    assert result.height == 2


def test_should_raise_when_unsupported_suffix(tmp_path):
    target = tmp_path / "x.parquet"
    target.write_text("dummy")
    with pytest.raises(ValueError):
        MarketFileReader().read(target)


def test_should_raise_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        MarketFileReader().read(tmp_path / "missing.feather")


def test_read_split_returns_three_frames(tmp_path):
    pl_frame = _build_simple_frame()
    files = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.feather"
        pl_frame.write_ipc(path)
        files[split] = path
    out = MarketFileReader().read_split(files["train"], files["val"], files["test"])
    assert set(out) == {"train", "val", "test"}
    assert all(v.height == 2 for v in out.values())
