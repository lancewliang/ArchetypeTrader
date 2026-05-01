"""Phase II GAE 无跨 env 泄漏集成测试。

目标: 确保完整 rollout + GAE 过程中不会跨 env 混算。

断言:
- 构造 2 个 reward 方向相反的 env。
- 跑一次完整 rollout + GAE。
- 每个 env 的 advantage 只依赖自己 env 的 reward/value 序列。
"""
import pytest


class TestPhase2GAENoCrossEnvLeakage:

    @pytest.mark.integration
    def test_opposite_reward_envs(self, tmp_path):
        """2 个 reward 方向相反的 env，GAE 不跨 env 混算。"""
        pass

    @pytest.mark.integration
    def test_advantage_per_env_independent(self, tmp_path):
        """每个 env 的 advantage 只依赖自己的 reward/value。"""
        pass
