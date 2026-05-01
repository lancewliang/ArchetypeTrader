"""HorizonFactory 单元测试。"""
from __future__ import annotations

from dataclasses import replace

import polars as pl

from src.trading.horizon_factory import HorizonFactory
from tests.phase2_test_utils import (
    make_config,
    make_dataset,
    make_frozen_policy,
    make_trading_env,
)


class TestHorizonFactory:

    def test_num_envs_contiguous_shards(self, tmp_path):
        """num_envs 连续时间分片正确。"""
        config = make_config(tmp_path, horizon=4, num_envs=2)
        dataset = make_dataset(config, count=5)
        factory = HorizonFactory(config, dataset, make_frozen_policy(), make_trading_env)
        envs, shards = factory.create_envs()
        assert len(envs) == 2
        assert [e.horizon_indices for e in envs] == [[0, 1, 2], [3, 4]]
        assert [s.num_horizons for s in shards] == [3, 2]

    def test_round_robin_shards(self, tmp_path):
        """round_robin 模式按 env id 交错分片。"""
        config = make_config(tmp_path, horizon=4, num_envs=2)
        config = replace(config, env_shards=replace(config.env_shards, mode="round_robin"))
        dataset = make_dataset(config, count=5)
        factory = HorizonFactory(config, dataset, make_frozen_policy(), make_trading_env)
        envs, _shards = factory.create_envs()
        assert [e.horizon_indices for e in envs] == [[0, 2, 4], [1, 3]]

    def test_envs_keep_independent_state(self, tmp_path):
        """每个 env 独立维护 cursor / prev_terminal_position。"""
        config = make_config(tmp_path, horizon=4, num_envs=2)
        dataset = make_dataset(config, count=4)
        factory = HorizonFactory(config, dataset, make_frozen_policy(), make_trading_env)
        envs, _shards = factory.create_envs()
        envs[0].reset(prev_terminal_position=1, cursor=1)
        envs[1].reset(prev_terminal_position=-1, cursor=0)
        assert envs[0].cursor == 1
        assert envs[1].cursor == 0
        assert envs[0].prev_terminal_position == 1
        assert envs[1].prev_terminal_position == -1

    def test_write_env_shards_feather(self, tmp_path):
        """phase2_env_shards.feather 记录正确。"""
        config = make_config(tmp_path, horizon=4, num_envs=2)
        dataset = make_dataset(config, count=4)
        factory = HorizonFactory(config, dataset, make_frozen_policy(), make_trading_env)
        _envs, shards = factory.create_envs()
        path = factory.write_shards(shards, tmp_path / "phase2_env_shards.feather")
        df = pl.read_ipc(path)
        assert df.height == 2
        assert set(df.columns) >= {"env_id", "start_horizon_idx", "end_horizon_idx", "num_horizons"}
