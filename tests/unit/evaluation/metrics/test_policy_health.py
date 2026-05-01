"""Policy health metrics 单元测试。

测试用例:
- approx_kl / clip_fraction / explained_variance 诊断正确。
- kl_demo_dominance_ratio 计算正确。
- per_archetype_reward_mean_and_std 聚合正确。
"""
import math

import pytest

from src.evaluation.metrics.policy_health import (
    compute_approx_kl,
    compute_clip_fraction,
    compute_explained_variance,
    compute_kl_demo_dominance_ratio,
    per_archetype_reward_stats,
)


class TestPolicyHealthMetrics:

    def test_approx_kl_identical(self):
        """相同 log_prob 时 KL=0。"""
        lp = [math.log(0.5)] * 10
        assert compute_approx_kl(lp, lp) == pytest.approx(0.0, abs=1e-6)

    def test_approx_kl_different(self):
        old = [math.log(0.5)] * 10
        new = [math.log(0.3)] * 10
        kl = compute_approx_kl(old, new)
        assert kl > 0

    def test_clip_fraction_no_clip(self):
        """ratio=1 时 clip_fraction=0。"""
        lp = [0.0] * 10
        assert compute_clip_fraction(lp, lp, 0.2) == 0.0

    def test_clip_fraction_all_clipped(self):
        old = [0.0] * 10
        new = [1.0] * 10  # ratio = e^1 ≈ 2.7, 远超 1+0.2
        frac = compute_clip_fraction(old, new, 0.2)
        assert frac == pytest.approx(1.0)

    def test_explained_variance_perfect(self):
        """value 完美预测 return 时 EV=1。"""
        vals = [1.0, 2.0, 3.0]
        rets = [1.0, 2.0, 3.0]
        assert compute_explained_variance(vals, rets) == pytest.approx(1.0)

    def test_explained_variance_zero(self):
        """value 完全不相关时 EV 接近 0 或负。"""
        vals = [0.0, 0.0, 0.0]
        rets = [1.0, 2.0, 3.0]
        ev = compute_explained_variance(vals, rets)
        assert ev <= 0.01

    def test_kl_demo_dominance_ratio(self):
        assert compute_kl_demo_dominance_ratio(0.5, 1.0) == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_kl_demo_dominance_ratio_zero(self):
        assert compute_kl_demo_dominance_ratio(0.0, 0.0) == 0.0

    def test_per_archetype_reward_stats(self):
        actions = [0, 0, 1, 1, 2]
        rewards = [0.1, 0.2, -0.1, -0.2, 0.5]
        stats = per_archetype_reward_stats(actions, rewards, 3)
        assert stats[0]["mean"] == pytest.approx(0.15)
        assert stats[0]["count"] == 2
        assert stats[1]["mean"] == pytest.approx(-0.15)
        assert stats[2]["count"] == 1
