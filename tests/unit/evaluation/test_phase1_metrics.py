"""``phase1_metrics`` 门面单元测试."""
from __future__ import annotations

import math

import pytest

from src.phase1.evaluation.metrics import (
    code_usage_ratio,
    codebook_perplexity,
    composite_score_sensitivity,
    composite_score_sensitivity_across_epochs,
    phase1_composite_score,
    return_capture_ratio,
)


def test_codebook_perplexity_uniform():
    """p_i = 1/K → perplexity ≈ K。"""
    code_ids = list(range(10)) * 10
    val = codebook_perplexity(code_ids, num_codes=10)
    assert val == pytest.approx(10.0, rel=1e-6)


def test_code_usage_ratio_basic():
    code_ids = [0, 1, 2, 0, 1]
    assert code_usage_ratio(code_ids, num_codes=5) == 3 / 5


def test_return_capture_ratio_stable_when_teacher_zero():
    val = return_capture_ratio(student_return=0.1, teacher_return=0.0)
    assert math.isfinite(val)


def test_phase1_composite_score_weighted_sum():
    metrics = {"a": 1.0, "b": 2.0, "c": 0.5}
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    score, debug = phase1_composite_score(metrics, weights)
    assert score == pytest.approx(0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 0.5)
    assert "missing_metrics" in debug


def test_phase1_composite_score_records_missing():
    metrics = {"a": 1.0}
    weights = {"a": 0.5, "b": 0.3}
    _, debug = phase1_composite_score(metrics, weights)
    assert "b" in debug["missing_metrics"]


def test_composite_sensitivity_returns_per_perturbation():
    metrics = {"a": 1.0, "b": 2.0}
    base_weights = {"a": 0.5, "b": 0.5}
    perts = [{"a": 0.1}, {"b": -0.1}]
    out = composite_score_sensitivity(metrics, base_weights, perts)
    assert len(out["results"]) == 2
    for r in out["results"]:
        assert "score" in r


def test_composite_sensitivity_across_epochs_reselects_best():
    epochs = [
        {"epoch": 0, "a": 1.0, "b": 0.0, "teacher_val_code_usage_ratio": 0.8},
        {"epoch": 1, "a": 0.0, "b": 2.0, "teacher_val_code_usage_ratio": 0.9},
    ]
    out = composite_score_sensitivity_across_epochs(
        epochs,
        base_weights={"a": 1.0, "b": 0.0},
        perturbations=[{"a": -1.0, "b": 1.0}],
    )
    assert out["base_best"]["best_epoch"] == 0
    assert out["results"][0]["best_epoch"] == 1
    assert out["best_epoch_drift"] is True
