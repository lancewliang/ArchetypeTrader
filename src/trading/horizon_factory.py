"""Multi-env 时间分片工厂: 按 num_envs 生成时间分片并实例化多个 HorizonEnv。

设计文档锚点: Phase II 执行计划 §Step 3。

职责:
- 按 num_envs 将 horizon index 切分为连续时间分片。
- 实例化多个 HorizonEnv，每个 env 独立维护 cursor / prev_terminal_position。
- 输出 phase2_env_shards.feather（边界位置、horizon 数、时间区间、regime 摘要）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import polars as pl

from src.config.phase2_config import Phase2Config
from src.data.phase2_dataset import Phase2Dataset
from src.models.phase1_frozen_policy import Phase1FrozenPolicy
from src.trading.env import TradingEnv
from src.trading.horizon_env import HorizonEnv
from src.utils.feather_io import write_ipc


@dataclass
class EnvShardInfo:
    """单个 env 的时间分片信息。"""
    env_id: int
    start_horizon_idx: int
    end_horizon_idx: int
    num_horizons: int
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    regime_summary: Optional[Dict[str, Any]] = None


class HorizonFactory:
    """Multi-env 工厂。

    使用方式::

        factory = HorizonFactory(config, dataset, frozen_policy, trading_env_factory)
        envs, shard_infos = factory.create_envs()
        factory.write_shards(shard_infos, output_path)
    """

    def __init__(
        self,
        config: Phase2Config,
        dataset: Phase2Dataset,
        frozen_policy: Phase1FrozenPolicy,
        trading_env_factory: Callable[[], TradingEnv],
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.frozen_policy = frozen_policy
        self.trading_env_factory = trading_env_factory

    def create_envs(self) -> tuple[List[HorizonEnv], List[EnvShardInfo]]:
        """创建 num_envs 个 HorizonEnv 及其分片信息。

        时间分片策略: 将 horizon index 按时间顺序均匀切分为 num_envs 段。
        每个 env 独立维护自己的 cursor 和 prev_terminal_position。

        Returns
        -------
        envs : HorizonEnv 列表。
        shard_infos : 每个 env 的分片信息。
        """
        num_envs = self.config.num_envs
        total_horizons = len(self.dataset)
        if total_horizons == 0:
            return [], []

        # 均匀切分
        shard_size = max(total_horizons // num_envs, 1)
        envs: List[HorizonEnv] = []
        shard_infos: List[EnvShardInfo] = []

        for env_id in range(num_envs):
            start_idx = env_id * shard_size
            if env_id == num_envs - 1:
                end_idx = total_horizons
            else:
                end_idx = min((env_id + 1) * shard_size, total_horizons)

            if start_idx >= total_horizons:
                break

            horizon_indices = list(range(start_idx, end_idx))
            trading_env = self.trading_env_factory()

            env = HorizonEnv(
                env_id=env_id,
                dataset=self.dataset,
                frozen_policy=self.frozen_policy,
                trading_env=trading_env,
                config=self.config,
                horizon_indices=horizon_indices,
            )
            envs.append(env)

            shard_infos.append(EnvShardInfo(
                env_id=env_id,
                start_horizon_idx=start_idx,
                end_horizon_idx=end_idx - 1,
                num_horizons=len(horizon_indices),
            ))

        return envs, shard_infos

    def write_shards(
        self,
        shard_infos: List[EnvShardInfo],
        output_path: Path,
    ) -> Path:
        """将分片信息写入 phase2_env_shards.feather。"""
        data = {
            "env_id": [s.env_id for s in shard_infos],
            "start_horizon_idx": [s.start_horizon_idx for s in shard_infos],
            "end_horizon_idx": [s.end_horizon_idx for s in shard_infos],
            "num_horizons": [s.num_horizons for s in shard_infos],
            "timestamp_start": [s.timestamp_start or "" for s in shard_infos],
            "timestamp_end": [s.timestamp_end or "" for s in shard_infos],
        }
        df = pl.DataFrame(data)
        return write_ipc(df, output_path)
