"""Phase II GAE 无跨 env 泄漏集成测试。"""
from __future__ import annotations

import pytest

from src.rl.rollout_buffer import RolloutBuffer, RolloutSample


def _sample(env_id: int, reward: float) -> RolloutSample:
    return RolloutSample(
        obs=[0.0],
        env_id=env_id,
        action=0,
        log_prob=0.0,
        value=0.0,
        reward=reward,
        reward_raw=reward,
        done=False,
        truncated=False,
    )


class TestPhase2GAENoCrossEnvLeakage:

    @pytest.mark.integration
    def test_opposite_reward_envs(self, tmp_path):
        """2 个 reward 方向相反的 env，GAE 不跨 env 混算。"""
        buf = RolloutBuffer(num_envs=2, rollout_length=4, gamma=1.0, gae_lambda=1.0)
        for _ in range(4):
            buf.add([_sample(0, 1.0), _sample(1, -1.0)])
        buf.compute_gae([0.0, 0.0])
        assert (buf._advantages[:, 0] > 0).all()
        assert (buf._advantages[:, 1] < 0).all()

    @pytest.mark.integration
    def test_advantage_per_env_independent(self, tmp_path):
        """每个 env 的 advantage 只依赖自己的 reward/value。"""
        buf = RolloutBuffer(num_envs=2, rollout_length=2, gamma=1.0, gae_lambda=1.0)
        buf.add([_sample(0, 10.0), _sample(1, -1.0)])
        buf.add([_sample(0, 10.0), _sample(1, -1.0)])
        buf.compute_gae([0.0, 0.0])
        assert buf._advantages[0, 0] == pytest.approx(20.0)
        assert buf._advantages[0, 1] == pytest.approx(-2.0)
