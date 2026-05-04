"""PPOTrainer 单元测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.config.phase2_config import (
    NumericalSafetyConfig,
    RewardScalingConfig,
    RolloutCollectionConfig,
)
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
        assert stats.rollout_collect_seconds > 0.0
        assert stats.rollout_samples_per_second > 0.0

    def test_threaded_rollout_update_runs(self, tmp_path):
        """thread rollout 采样器可完成 rollout -> update。"""
        config = make_config(
            tmp_path,
            horizon=4,
            num_envs=2,
            rollout_length=2,
            total_timesteps=4,
        )
        config = replace(
            config,
            rollout_collection=RolloutCollectionConfig(
                mode="thread",
                max_workers=2,
                fail_fast=True,
            ),
        )
        dataset = make_dataset(config, count=4, labeled=True)
        factory = HorizonFactory(config, dataset, make_frozen_policy(), make_trading_env)
        envs, _shards = factory.create_envs()
        actor_critic = make_actor_critic(state_dim=dataset.state_spec().total_dim)
        optimizer = torch.optim.Adam(actor_critic.selector.parameters(), lr=config.ppo.lr)
        schedule = ScheduleManager(config, optimizer, total_updates=2)
        trainer = PPOTrainer(config, actor_critic, envs, schedule)
        trainer.setup(optimizer)

        stats = trainer.collect_and_update()

        assert isinstance(stats.policy_loss, float)
        assert stats.rollout_samples_per_second > 0.0
        assert trainer.rollout_collection_info()["max_workers"] == 2

    def test_process_rollout_update_runs_and_checkpoint_state_roundtrips(self, tmp_path):
        """process rollout 采样器可完成 update，并通过 worker state checkpoint。"""
        config = make_config(
            tmp_path,
            horizon=4,
            num_envs=2,
            rollout_length=2,
            total_timesteps=4,
        )
        config = replace(
            config,
            rollout_collection=RolloutCollectionConfig(
                mode="process",
                max_workers=2,
                fail_fast=True,
                worker_startup_timeout_seconds=20.0,
                worker_step_timeout_seconds=20.0,
            ),
        )
        dataset = make_dataset(config, count=4, labeled=True)
        factory = HorizonFactory(config, dataset, make_frozen_policy(), make_trading_env)
        worker_specs, _shards = factory.create_worker_specs(
            phase1_decoder_path=config.phase1_dir() / "decoder.pt",
            phase1_codebook_path=config.phase1_dir() / "codebook.pt",
            cost_config={"commission_rate": 0.0, "book_levels": 5},
            reward_alignment_name="paper_formula",
        )
        actor_critic = make_actor_critic(state_dim=dataset.state_spec().total_dim)
        optimizer = torch.optim.Adam(actor_critic.selector.parameters(), lr=config.ppo.lr)
        schedule = ScheduleManager(config, optimizer, total_updates=2)
        trainer = PPOTrainer(
            config,
            actor_critic,
            [],
            schedule,
            worker_specs=worker_specs,
        )
        trainer.setup(optimizer)
        try:
            stats = trainer.collect_and_update()
            assert isinstance(stats.policy_loss, float)
            assert stats.rollout_samples_per_second > 0.0
            state = trainer.get_state()
            assert len(state["env_states"]) == 2
            trainer.load_state(state)
        finally:
            trainer.close()

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

    def test_default_gradient_safety_allows_pre_clip_spike(self, tmp_path):
        """默认 safety 阈值允许有限的裁剪前梯度尖峰。"""
        trainer = _trainer(tmp_path)
        for param in trainer.actor_critic.selector.parameters():
            param.grad = torch.zeros_like(param)
            param.grad.view(-1)[0] = 101.0
            break
        trainer._check_gradient_safety()
