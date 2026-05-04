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


@dataclass
class HorizonEnvWorkerSpec:
    """Process rollout worker 构造单个 HorizonEnv 所需的可 pickle 信息。"""
    env_id: int
    dataset: Phase2Dataset
    config: Phase2Config
    horizon_indices: List[int]
    phase1_decoder_path: str
    phase1_codebook_path: str
    cost_config: Dict[str, Any]
    reward_alignment_name: str


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

    def _iter_shards(self) -> List[tuple[int, List[int], int, int]]:
        """返回 env 分片: (env_id, horizon_indices, start_idx, end_idx_exclusive)。"""
        num_envs = self.config.num_envs
        total_horizons = len(self.dataset)
        if total_horizons == 0:
            return []

        shards: List[tuple[int, List[int], int, int]] = []
        for env_id in range(num_envs):
            if self.config.env_shards.mode == "round_robin":
                horizon_indices = list(range(env_id, total_horizons, num_envs))
                if not horizon_indices:
                    continue
                start_idx = horizon_indices[0]
                end_idx = horizon_indices[-1] + 1
            else:
                # contiguous / rollover 当前都使用连续时间分片；rollover 保留为后续扩展。
                shard_size = max((total_horizons + num_envs - 1) // num_envs, 1)
                start_idx = env_id * shard_size
                end_idx = min((env_id + 1) * shard_size, total_horizons)

                if start_idx >= total_horizons:
                    break
                horizon_indices = list(range(start_idx, end_idx))

            shards.append((env_id, horizon_indices, start_idx, end_idx))
        return shards

    def _shard_info(
        self,
        env_id: int,
        horizon_indices: List[int],
        start_idx: int,
    ) -> EnvShardInfo:
        first_entry = self.dataset.horizon_entries[horizon_indices[0]]
        last_entry = self.dataset.horizon_entries[horizon_indices[-1]]
        return EnvShardInfo(
            env_id=env_id,
            start_horizon_idx=start_idx,
            end_horizon_idx=horizon_indices[-1],
            num_horizons=len(horizon_indices),
            timestamp_start=getattr(first_entry, "timestamp_start", None),
            timestamp_end=getattr(last_entry, "timestamp_start", None),
        )

    def create_envs(self) -> tuple[List[HorizonEnv], List[EnvShardInfo]]:
        """创建 num_envs 个 HorizonEnv 及其分片信息。

        时间分片策略: 将 horizon index 按时间顺序均匀切分为 num_envs 段。
        每个 env 独立维护自己的 cursor 和 prev_terminal_position。

        Returns
        -------
        envs : HorizonEnv 列表。
        shard_infos : 每个 env 的分片信息。
        """
        envs: List[HorizonEnv] = []
        shard_infos: List[EnvShardInfo] = []

        for env_id, horizon_indices, start_idx, _end_idx in self._iter_shards():
            trading_env = self.trading_env_factory()

            env = HorizonEnv(
                env_id=env_id,
                dataset=self.dataset,
                frozen_policy=self.frozen_policy.spawn_worker_policy(),
                trading_env=trading_env,
                config=self.config,
                horizon_indices=horizon_indices,
            )
            envs.append(env)
            shard_infos.append(self._shard_info(env_id, horizon_indices, start_idx))

        return envs, shard_infos

    def create_worker_specs(
        self,
        *,
        phase1_decoder_path: Path,
        phase1_codebook_path: Path,
        cost_config: Dict[str, Any],
        reward_alignment_name: str,
    ) -> tuple[List[HorizonEnvWorkerSpec], List[EnvShardInfo]]:
        """创建 process rollout worker specs 及其分片信息。"""
        specs: List[HorizonEnvWorkerSpec] = []
        shard_infos: List[EnvShardInfo] = []

        for env_id, horizon_indices, start_idx, _end_idx in self._iter_shards():
            specs.append(HorizonEnvWorkerSpec(
                env_id=env_id,
                dataset=self.dataset,
                config=self.config,
                horizon_indices=horizon_indices,
                phase1_decoder_path=str(phase1_decoder_path),
                phase1_codebook_path=str(phase1_codebook_path),
                cost_config=dict(cost_config),
                reward_alignment_name=reward_alignment_name,
            ))
            shard_infos.append(self._shard_info(env_id, horizon_indices, start_idx))

        return specs, shard_infos

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
