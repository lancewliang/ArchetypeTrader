"""RolloutBuffer 单元测试。

测试用例:
- buffer 保存字段完整。
- done 与 truncated 区分正确。
- flatten minibatch 前按 env 分组。
- raw/scaled reward 同步记录。
"""
import numpy as np
import pytest

from src.rl.rollout_buffer import RolloutBuffer, RolloutSample


def _make_sample(env_id, reward=0.1, done=False, truncated=False, kl_label=None):
    return RolloutSample(
        obs=np.array([1.0, 2.0], dtype=np.float32),
        env_id=env_id,
        action=0,
        log_prob=-0.5,
        value=0.5,
        reward=reward,
        reward_raw=reward * 2,
        done=done,
        truncated=truncated,
        kl_label=kl_label,
        is_labeled=kl_label is not None,
    )


class TestRolloutBuffer:

    def test_fields_complete(self):
        """buffer 保存字段完整。"""
        buf = RolloutBuffer(num_envs=2, rollout_length=4)
        for _ in range(4):
            buf.add([_make_sample(0), _make_sample(1)])
        assert buf._step_count == 4
        assert len(buf._obs) == 4
        assert len(buf._rewards) == 4

    def test_done_truncated_distinction(self):
        """done 与 truncated 区分正确。"""
        buf = RolloutBuffer(num_envs=1, rollout_length=3)
        buf.add([_make_sample(0, done=False, truncated=False)])
        buf.add([_make_sample(0, done=True, truncated=False)])
        buf.add([_make_sample(0, done=False, truncated=True)])
        assert buf._dones[1][0] is True
        assert buf._truncateds[1][0] is False
        assert buf._dones[2][0] is False
        assert buf._truncateds[2][0] is True

    def test_gae_per_env_grouping(self):
        """GAE 按 env_id 分组计算: 不同 env 的 advantage 独立。"""
        buf = RolloutBuffer(num_envs=2, rollout_length=4, gamma=0.99, gae_lambda=0.95)
        # env 0: 正 reward; env 1: 负 reward
        for _ in range(4):
            buf.add([_make_sample(0, reward=1.0), _make_sample(1, reward=-1.0)])
        buf.compute_gae([0.0, 0.0])

        # env 0 的 advantage 应该全正，env 1 全负
        for t in range(4):
            assert buf._advantages[t, 0] > 0
            assert buf._advantages[t, 1] < 0

    def test_raw_scaled_reward_sync(self):
        """raw/scaled reward 同步记录。"""
        buf = RolloutBuffer(num_envs=1, rollout_length=2)
        buf.add([_make_sample(0, reward=0.5)])
        buf.add([_make_sample(0, reward=0.3)])
        stats = buf.get_stats()
        assert stats["reward_mean"] == pytest.approx(0.4)
        assert stats["reward_raw_mean"] == pytest.approx(0.8)  # raw = reward * 2

    def test_iterate_minibatches(self):
        """minibatch 迭代器生成正确大小的 batch。"""
        buf = RolloutBuffer(num_envs=2, rollout_length=4)
        for _ in range(4):
            buf.add([_make_sample(0), _make_sample(1)])
        buf.compute_gae([0.0, 0.0])

        total_samples = 0
        for mb in buf.iterate_minibatches(3):
            assert "obs" in mb
            assert "action" in mb
            assert "advantage" in mb
            total_samples += mb["obs"].shape[0]
        assert total_samples == 8  # 4 steps * 2 envs

    def test_reset(self):
        """reset 清空 buffer。"""
        buf = RolloutBuffer(num_envs=1, rollout_length=2)
        buf.add([_make_sample(0)])
        buf.reset()
        assert buf._step_count == 0
        assert len(buf._obs) == 0

    def test_gae_done_cuts_bootstrap(self):
        """done=True 切断 bootstrap: done 后的 advantage 不受之前 reward 影响。"""
        buf = RolloutBuffer(num_envs=1, rollout_length=4, gamma=0.99, gae_lambda=0.95)
        buf.add([_make_sample(0, reward=1.0, done=False)])
        buf.add([_make_sample(0, reward=1.0, done=True)])  # done 切断
        buf.add([_make_sample(0, reward=1.0, done=False)])  # 新 episode
        buf.add([_make_sample(0, reward=1.0, done=False)])
        buf.compute_gae([0.5])
        assert buf._advantages is not None
        # step 1 done=True，其 advantage 应该只是 reward - value = 1.0 - 0.5 = 0.5
        assert buf._advantages[1, 0] == pytest.approx(1.0 - 0.5, abs=0.01)
        # step 2 是新 episode 的第一步，不应受 step 0-1 的 reward 影响
        # step 2 的 advantage 应该基于 step 2-3 的 reward 和 bootstrap
        assert buf._advantages[2, 0] > 0  # 正 reward 应该有正 advantage
