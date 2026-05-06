"""``StratifiedWindowSampler`` 单元测试."""
from __future__ import annotations

import pytest

from src.preprocess_data.stratified_sampler import StratifiedWindowSampler
from src.preprocess_data.window_indexer import WindowIndexEntry


def _entries(n: int):
    return [
        WindowIndexEntry(
            window_start=i,
            window_end=i + 7,
            last_execution_row=i + 7,
            last_markout_row=i + 8,
            horizon_return=0.001 * (i % 3 - 1),
            realized_volatility=0.001,
            draw_pattern="upward" if i % 2 == 0 else "downward",
            past_return=0.0,
            past_realized_volatility=0.0,
            past_draw_pattern="mixed",
        )
        for i in range(n)
    ]


def test_same_seed_yields_deterministic_window_starts():
    entries = _entries(200)
    labels = [
        StratifiedWindowSampler.assign_strata(e, prospective=False) for e in entries
    ]
    sampler = StratifiedWindowSampler(
        strategy="stratified_uniform",
        min_gap_between_samples=2,
        flat_low_vol_max_ratio=1.0,
        seed=123,
    )
    a = sampler.sample(entries, num_samples=20, strata_labels=labels)
    b = sampler.sample(entries, num_samples=20, strata_labels=labels)
    assert [s.window_start for s in a] == [s.window_start for s in b]


def test_min_gap_between_samples_enforced():
    entries = _entries(200)
    labels = [
        StratifiedWindowSampler.assign_strata(e, prospective=False) for e in entries
    ]
    sampler = StratifiedWindowSampler(
        strategy="stratified_uniform",
        min_gap_between_samples=10,
        flat_low_vol_max_ratio=1.0,
        seed=1,
    )
    samples = sampler.sample(entries, num_samples=15, strata_labels=labels)
    starts = sorted([s.window_start for s in samples])
    for i in range(1, len(starts)):
        assert starts[i] - starts[i - 1] >= 10


def test_unsupported_strategy_raises():
    with pytest.raises(ValueError):
        StratifiedWindowSampler(
            strategy="other",  # type: ignore[arg-type]
            min_gap_between_samples=1,
            flat_low_vol_max_ratio=1.0,
        )


def test_oversampling_raises():
    entries = _entries(20)
    labels = [
        StratifiedWindowSampler.assign_strata(e, prospective=False) for e in entries
    ]
    sampler = StratifiedWindowSampler(
        strategy="stratified_uniform",
        min_gap_between_samples=1,
        flat_low_vol_max_ratio=1.0,
    )
    with pytest.raises(ValueError):
        sampler.sample(entries, num_samples=30, strata_labels=labels)


def test_shortfall_refill_does_not_exceed_flat_low_cap():
    entries = _entries(6)
    labels = [
        "up|mid|mixed",
        "up|mid|mixed",
        "flat|low|mixed",
        "flat|low|mixed",
        "flat|low|mixed",
        "flat|low|mixed",
    ]
    sampler = StratifiedWindowSampler(
        strategy="stratified_uniform",
        min_gap_between_samples=1,
        flat_low_vol_max_ratio=0.25,
        seed=1,
    )

    with pytest.raises(RuntimeError, match="无法在 min_gap"):
        sampler.sample(entries, num_samples=4, strata_labels=labels)


def test_overlap_relaxation_records_effective_min_gap():
    entries = _entries(4)
    labels = ["up|mid|mixed"] * 4
    sampler = StratifiedWindowSampler(
        strategy="stratified_uniform",
        min_gap_between_samples=2,
        flat_low_vol_max_ratio=1.0,
        allow_overlap_relaxation=True,
        seed=1,
    )

    samples = sampler.sample(entries, num_samples=4, strata_labels=labels)

    assert len(samples) == 4
    assert sampler.last_overlap_relaxation_applied is True
    assert sampler.last_effective_min_gap_between_samples == 1
