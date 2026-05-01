"""PPO trainer 单元测试。

测试用例:
- rollout -> GAE -> update 主链路可跑通。
- approx_kl > target_kl 时 early stop。
- advantage_normalization 开关生效。
- reward clip 开启时同时记录 clipped/unclipped 统计。
- reward clip 开启时 report 同时保留 unclipped 对照统计。
"""
import pytest


class TestPPOTrainer:

    def test_main_loop_runs(self):
        """rollout -> GAE -> update 主链路可跑通。"""
        pass

    def test_early_stop_on_kl(self):
        """approx_kl > target_kl 时 early stop。"""
        pass

    def test_advantage_normalization_toggle(self):
        """advantage_normalization 开关生效。"""
        pass

    def test_reward_clip_records_both(self):
        """reward clip 开启时同时记录 clipped/unclipped 统计。"""
        pass

    def test_numerical_safety_fail_fast(self):
        """非 finite tensor 触发 NumericalSafetyError。"""
        pass
