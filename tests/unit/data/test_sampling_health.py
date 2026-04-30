"""``SamplingHealthChecker`` 单元测试."""
from __future__ import annotations

import pytest

from src.data.sampling_health import SamplingHealthChecker, SamplingHealthError
from src.data.stratified_sampler import SampledHorizon


def _sampled(starts):
    out = []
    for i, s in enumerate(starts):
        out.append(
            SampledHorizon(
                sample_id=f"s{i}",
                window_start=s,
                window_end=s + 7,
                last_execution_row=s + 7,
                last_markout_row=s + 8,
                strata_label="up|low|mixed",
            )
        )
    return out


def test_min_gap_calculations():
    checker = SamplingHealthChecker(
        horizon=8,
        max_overlap_ratio=1.0,
        min_gap_between_samples=2,
        split_boundary_embargo=0,
        flat_low_vol_max_ratio=1.0,
        warn_only=True,
    )
    sampled = _sampled([0, 5, 12])
    rep = checker.check(sampled, {"train_end_row": 100}, ["up|low|mixed"] * 3)
    assert rep.min_sample_gap == 5
    assert rep.mean_sample_gap == pytest.approx((5 + 7) / 2)


def test_warn_only_does_not_raise():
    checker = SamplingHealthChecker(
        horizon=8,
        max_overlap_ratio=0.0,  # 任何重叠都告警
        min_gap_between_samples=10,  # 5 < 10 → 告警
        split_boundary_embargo=0,
        flat_low_vol_max_ratio=1.0,
        warn_only=True,
    )
    rep = checker.check(_sampled([0, 5]), {"train_end_row": 100}, ["up|low|mixed"] * 2)
    assert rep.sampling_health_warnings


def test_warn_only_false_raises():
    checker = SamplingHealthChecker(
        horizon=8,
        max_overlap_ratio=0.0,
        min_gap_between_samples=10,
        split_boundary_embargo=0,
        flat_low_vol_max_ratio=1.0,
        warn_only=False,
    )
    with pytest.raises(SamplingHealthError):
        checker.check(_sampled([0, 5]), {"train_end_row": 100}, ["up|low|mixed"] * 2)


def test_split_boundary_warning_when_markout_row_close_to_train_end():
    checker = SamplingHealthChecker(
        horizon=8,
        max_overlap_ratio=1.0,
        min_gap_between_samples=1,
        split_boundary_embargo=10,
        flat_low_vol_max_ratio=1.0,
        warn_only=True,
    )
    # last_markout_row = 50 + 8 = 58；train_end_row = 60 → gap = 2 < 10
    rep = checker.check(_sampled([50]), {"train_end_row": 60}, ["up|low|mixed"])
    assert any("split_boundary_gap" in w for w in rep.sampling_health_warnings)


def test_flat_low_vol_warn_when_ratio_high():
    checker = SamplingHealthChecker(
        horizon=8,
        max_overlap_ratio=1.0,
        min_gap_between_samples=1,
        split_boundary_embargo=0,
        flat_low_vol_max_ratio=0.1,
        warn_only=True,
    )
    rep = checker.check(_sampled([0, 10, 20]), {"train_end_row": 100},
                        ["flat|low|mixed", "flat|low|mixed", "up|mid|mixed"])
    assert any("flat_low_vol_sample_ratio" in w for w in rep.sampling_health_warnings)
