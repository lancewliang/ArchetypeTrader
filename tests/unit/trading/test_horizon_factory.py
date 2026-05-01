"""HorizonFactory 单元测试。

测试用例:
- num_envs 连续时间分片正确。
- 每个 env 独立维护 cursor / prev_terminal_position。
- phase2_env_shards.feather 记录正确。
- rollover 模式仅在诊断配置下可启用。
"""
import pytest


class TestHorizonFactory:

    def test_continuous_time_shards(self):
        """num_envs 连续时间分片正确。"""
        pass

    def test_independent_env_state(self):
        """每个 env 独立维护 cursor / prev_terminal_position。"""
        pass

    def test_env_shards_feather_correct(self):
        """phase2_env_shards.feather 记录正确。"""
        pass

    def test_rollover_diagnostic_only(self):
        """rollover 模式仅在诊断配置下可启用。"""
        pass
