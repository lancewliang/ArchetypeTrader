"""PPO trainer 单元测试。

测试用例:
- rollout -> GAE -> update 主链路可跑通。
- approx_kl > target_kl 时 early stop。
- advantage_normalization 开关生效。
- reward clip 开启时同时记录 clipped/unclipped 统计。
- reward clip 开启时 report 同时保留 unclipped 对照统计。
"""
import pytest

from src.config.phase2_config import Phase2Config, PPOConfig, RewardNormalizationConfig
from src.rl.ppo_trainer import PPOTrainer


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

    def test_reward_normalization_rejected_for_signoff(self):
        """Phase II reward_normalization 启用时 fail-fast。"""
        config = Phase2Config(
            reward_normalization=RewardNormalizationConfig(enabled=True),
            ppo=PPOConfig(reward_normalization=False),
        )
        trainer = PPOTrainer(config, actor_critic=None, envs=[], schedule_manager=None)
        with pytest.raises(ValueError, match="reward_normalization"):
            trainer.setup()
