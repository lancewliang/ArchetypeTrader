"""``RewardNormalizer`` 单元测试."""
from __future__ import annotations

import math
import random

import pytest

from src.phase1.config import EncoderInputConfig
from src.phase1.models.encoder_inputs import RewardNormalizer


def _heavy_tail_rewards(n: int = 2000, seed: int = 1):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        u = rng.random()
        # 学生 t 风格的近似重尾: 90% 标准噪声 + 10% 大尾
        if u < 0.9:
            out.append(rng.gauss(0.0, 0.0001))
        else:
            out.append(rng.gauss(0.0, 0.005))
    return out


def test_robust_fits_median_and_mad():
    config = EncoderInputConfig(
        reward_normalization="train_reward_robust",
        reward_clip_value=8.0,
        fallback_to_standard_kurtosis_below=0.0,  # 关闭 fallback
    )
    norm = RewardNormalizer(config)
    rewards = _heavy_tail_rewards()
    stats = norm.fit_train(rewards)
    assert stats.method == "train_reward_robust"
    assert stats.scale > 0


def test_robust_transform_centers_median():
    config = EncoderInputConfig(reward_normalization="train_reward_robust", reward_clip_value=8.0,
                                fallback_to_standard_kurtosis_below=0.0)
    norm = RewardNormalizer(config)
    rewards = _heavy_tail_rewards()
    norm.fit_train(rewards)
    transformed = norm.transform(rewards)
    sorted_t = sorted(transformed)
    median = sorted_t[len(sorted_t) // 2]
    assert abs(median) < 0.5


def test_standard_when_explicit():
    config = EncoderInputConfig(reward_normalization="train_reward_standard")
    norm = RewardNormalizer(config)
    rewards = [0.001 * i for i in range(100)]
    stats = norm.fit_train(rewards)
    assert stats.method == "train_reward_standard"


def test_auto_fallback_when_kurtosis_low():
    """近似正态 → kurtosis 接近 0 → fallback 到 standard。"""
    config = EncoderInputConfig(
        reward_normalization="train_reward_robust",
        fallback_to_standard_kurtosis_below=6.0,
    )
    norm = RewardNormalizer(config)
    rng = random.Random(2)
    rewards = [rng.gauss(0.0, 0.001) for _ in range(2000)]
    stats = norm.fit_train(rewards)
    assert stats.method == "train_reward_standard"
    assert stats.fallback_reason == "kurtosis_below_threshold"


def test_clip_applied_in_transform():
    config = EncoderInputConfig(
        reward_normalization="train_reward_standard",
        reward_clip_value=2.0,
    )
    norm = RewardNormalizer(config)
    norm.fit_train([0.0, 1.0, -1.0, 0.5, -0.5])
    out = norm.transform([100.0, -100.0])
    assert max(out) <= 2.0
    assert min(out) >= -2.0


def test_val_test_must_not_refit():
    """fit_train 必须在 transform 前调用；transform 不应改变 stats。"""
    config = EncoderInputConfig()
    norm = RewardNormalizer(config)
    with pytest.raises(RuntimeError):
        norm.transform([0.0])


def test_zero_scale_uses_epsilon():
    config = EncoderInputConfig(reward_normalization="train_reward_standard")
    norm = RewardNormalizer(config)
    norm.fit_train([1.0, 1.0, 1.0])  # std 接近 0
    out = norm.transform([1.0])
    assert math.isfinite(out[0])


def test_to_dict_contains_all_audit_fields():
    config = EncoderInputConfig()
    norm = RewardNormalizer(config)
    norm.fit_train(_heavy_tail_rewards(n=200))
    d = norm.to_dict()
    for key in ("method", "center", "scale", "clip_value", "kurtosis", "skew"):
        assert key in d
