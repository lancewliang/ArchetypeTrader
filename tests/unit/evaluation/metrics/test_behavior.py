"""``src.evaluation.metrics.behavior`` 单元测试."""
from __future__ import annotations

import pytest

from src.evaluation.metrics.behavior import (
    decoder_sensitivity_to_code,
    inter_code_action_diversity,
    inter_code_distance,
    latent_silhouette_score,
    per_code_action_entropy,
)


def test_per_code_action_entropy_low_when_single_action():
    """单一 action → 熵 ≈ 0。"""
    decoded = {0: [[[10.0, 0.0, 0.0]] * 4]}  # 全 short
    out = per_code_action_entropy(decoded)
    assert out[0] == pytest.approx(0.0)


def test_inter_code_action_diversity_zero_when_identical():
    actions = {0: [[1, 2, 1]], 1: [[1, 2, 1]]}
    assert inter_code_action_diversity(actions) == 0.0


def test_inter_code_action_diversity_nonzero_when_different():
    actions = {0: [[1, 2, 1]], 1: [[0, 2, 1]]}
    assert inter_code_action_diversity(actions) > 0.0


def test_decoder_sensitivity_zero_when_logits_identical():
    logits = [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]
    by_code = {0: logits, 1: logits}
    assert decoder_sensitivity_to_code(by_code) == pytest.approx(0.0)


def test_inter_code_distance_zero_when_collapsed():
    cb = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    assert inter_code_distance(cb) == pytest.approx(0.0)


def test_latent_silhouette_positive_when_well_separated():
    latents = [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]]
    code_ids = [0, 0, 1, 1]
    s = latent_silhouette_score(latents, code_ids)
    assert s > 0.5
