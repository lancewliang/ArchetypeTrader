"""PPOTrainer 单元测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.config.phase2_config import NumericalSafetyConfig, RewardScalingConfig
from src.rl.ppo_trainer import NumericalSafetyError, PPOTrainer
from src.rl.scheduling import ScheduleManager
from src.trading.horizon_factory import HorizonFactory
from tests.phase2_test_utils import (
    make_actor_critic,
    make_config,
    make_dataset,
    make_frozen_policy,
    make_trading_env,
)


def _trainer(tmp_path, **config_overrides) -> PPOTrainer:
    config = make_config(tmp_path, horizon=4, num_envs=1, rollout_length=2, total_timesteps=4)
    for key, value in config_overrides.items():
        config = replace(config, **{key: value})
    dataset = make_dataset(config, count=4, labeled=True)
    factory = HorizonFactory(config, dataset, make_frozen_policy(), make_trading_env)
    envs, _shards = factory.create_envs()
    actor_critic = make_actor_critic(state_dim=dataset.state_spec().total_dim)
    optimizer = torch.optim.Adam(actor_critic.selector.parameters(), lr=config.ppo.lr)
    schedule = ScheduleManager(config, optimizer, total_updates=2)
    trainer = PPOTrainer(config, actor_critic, envs, schedule)
    trainer.setup(optimizer)
    return trainer


class TestPPOTrainer:

    def test_rollout_gae_update_runs(self, tmp_path):
        """rollout -> GAE -> update 主链路可跑通。"""
        trainer = _trainer(tmp_path)
        stats = trainer.collect_and_update()
        assert isinstance(stats.policy_loss, float)
        assert stats.rollout_truncated_count >= 1

    def test_approx_kl_early_stop(self, tmp_path):
        """approx_kl > target_kl 时 early stop。"""
        config = make_config(tmp_path, horizon=4, num_envs=1, rollout_length=2)
        config = replace(config, ppo=replace(config.ppo, target_kl=-1.0, update_epochs=3))
        trainer = _trainer(tmp_path, ppo=config.ppo)
        stats = trainer.collect_and_update()
        assert stats.early_stopped is True
        assert stats.early_stop_epoch == 0

    def test_advantage_normalization_can_be_disabled(self, tmp_path):
        """advantage_normalization 开关生效且关闭后仍可训练。"""
        config = make_config(tmp_path)
        trainer = _trainer(tmp_path, ppo=replace(config.ppo, advantage_normalization=False))
        stats = trainer.collect_and_update()
        assert stats.policy_loss == pytest.approx(stats.policy_loss)

    def test_reward_clip_records_clipped_unclipped_stats(self, tmp_path):
        """reward clip 开启时同时记录 clipped/unclipped 统计。"""
        trainer = _trainer(
            tmp_path,
            reward_scaling=RewardScalingConfig(method="raw", clip_range=0.001),
        )
        original_step = trainer.envs[0].step

        def high_reward_step(action: int):
            next_obs, _reward, done, truncated, info = original_step(action)
            return next_obs, 1.0, done, truncated, info

        trainer.envs[0].step = high_reward_step
        trainer.collect_rollout()
        assert trainer._buffer is not None
        stats = trainer._buffer.get_stats()
        assert stats["reward_clipped_ratio"] > 0.0
        assert stats["reward_mean"] == pytest.approx(0.001)
        assert stats["reward_unclipped_mean"] == pytest.approx(1.0)
        assert "reward_unclipped_mean" in stats

    def test_numerical_safety_fail_fast(self, tmp_path):
        """非 finite loss 触发 fail-fast 并导出 snapshot。"""
        trainer = _trainer(tmp_path)
        with pytest.raises(NumericalSafetyError):
            trainer._check_numerical_safety(torch.tensor(float("nan")))
        snapshots = list(trainer.config.artifacts_dir().glob("debug_snapshots/*.pt"))
        assert snapshots

    def test_gradient_safety_fail_fast(self, tmp_path):
        """gradient norm 爆炸触发 fail-fast。"""
        trainer = _trainer(
            tmp_path,
            numerical_safety=NumericalSafetyConfig(max_gradient_norm=0.000001),
        )
        for param in trainer.actor_critic.selector.parameters():
            param.grad = torch.ones_like(param) * 10.0
            break
        with pytest.raises(NumericalSafetyError):
            trainer._check_gradient_safety()
