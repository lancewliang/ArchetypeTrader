"""固定 seed 可复现性 (轻量版集成测试)."""
from __future__ import annotations

import pytest

from src.config.phase1_config import (
    Phase1Config,
    SamplingHealthConfig,
    StratificationConfig,
)
from src.data.stratified_sampler import StratifiedWindowSampler
from src.data.window_indexer import SlidingWindowIndexer


def _frame(rows: int = 200):
    pl = pytest.importorskip("polars")
    return pl.DataFrame({
        "timestamp": list(range(rows)),
        "close": [100.0 + i * 0.05 for i in range(rows)],
    })


def test_same_seed_yields_identical_sample_ids():
    indexer = SlidingWindowIndexer(horizon=8, reward_alignment="paper_formula")
    entries = indexer.enumerate(_frame(200), stratification_mode="hindsight_horizon")
    labels = [StratifiedWindowSampler.assign_strata(e, prospective=False) for e in entries]
    sampler = StratifiedWindowSampler(
        strategy="stratified_uniform",
        min_gap_between_samples=2,
        flat_low_vol_max_ratio=1.0,
        seed=42,
    )
    a = sampler.sample(entries, num_samples=20, strata_labels=labels)
    b = sampler.sample(entries, num_samples=20, strata_labels=labels)
    assert [s.sample_id for s in a] == [s.sample_id for s in b]


def test_config_hash_changes_when_field_changes():
    base = Phase1Config(
        pair="TEST",
        train_batch_id="b1",
        data_process_manifest="/tmp/m1.json",
    )
    h1 = base.config_hash()
    other = Phase1Config(
        pair="TEST",
        train_batch_id="b2",
        data_process_manifest="/tmp/m1.json",
    )
    assert h1 != other.config_hash()


def test_training_config_hash_ignores_identifiers():
    a = Phase1Config(
        pair="TEST",
        train_batch_id="b1",
        data_process_manifest="/tmp/m1.json",
    )
    b = Phase1Config(
        pair="OTHER",
        train_batch_id="b2",
        data_process_manifest="/tmp/m2.json",
    )
    assert a.training_config_hash() == b.training_config_hash()
