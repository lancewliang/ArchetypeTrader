"""PyTorch Dataset / DataLoader 适配.

设计文档锚点: §4.3。

边界（极重要）:
- 不读原始文件、不调 DP、不做采样、不做 schema 校验。
- 仅把已经构建好的 ``HorizonRecord`` 转成 tensor，并支持 contrastive pair index 注入。
- 可选传入 ``reward_normalizer``: train 拟合后 train/val/test 共用同一个实例，
  在 ``__getitem__`` 中把原始 rewards 转为 encoder 输入；记录本身的 ``rec.rewards``
  保持原始（actual）值，便于 demo_return 等下游统计仍能拿到真实收益。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc, assignment]

from .data_augmentation import ContrastivePair
from .horizon_builder import HorizonRecord


class _SupportsTransform(Protocol):
    """任何实现 ``transform(rewards) -> Sequence[float]`` 的对象都可作为 normalizer。

    主要为 ``src.models.encoder_inputs.RewardNormalizer``；
    用 Protocol 而不是直接 import 是为了避免 dataset 模块依赖 torch 之外的 models。
    """

    def transform(self, rewards): ...


class Phase1DemoDataset(Dataset):
    """每个 ``__getitem__`` 返回一个训练样本。

    返回结构::

        {
            "states": Tensor[h, feature_dim],
            "actions": Tensor[h]  (long),
            "rewards": Tensor[h]  (float32, 经 normalizer.transform 后的 encoder 输入),
            "trajectory_return": float,  # 原始 reward 求和，不做 normalizer transform
            "sample_id": str,
            "contrastive_pair_id": str,  # 没有 pair 时为空字符串
        }

    DataLoader collate 规则: states/actions/rewards 直接 ``stack``；
    sample_id / pair id 保留为 list 字段。详见 ``collate_phase1``。

    边界（极重要）:
    - 不读原始文件、不调 DP、不做采样、不做 schema 校验。
    - 仅把已经构建好的 ``HorizonRecord`` 转成 tensor。
    - 不修改 ``rec.rewards``；normalizer transform 在 __getitem__ 中即时应用，
      以便同一组 records 既可作 encoder 输入（normalized）又可作 demo_return 来源（actual）。
    """

    def __init__(
        self,
        records: List[HorizonRecord],
        contrastive_pairs: Optional[List[ContrastivePair]] = None,
        reward_normalizer: Optional[_SupportsTransform] = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("Phase1DemoDataset 需要 torch")
        self.records = records
        self.contrastive_pairs = contrastive_pairs or []
        self.reward_normalizer = reward_normalizer
        self._sample_to_pair: Dict[str, str] = self._build_pair_index(self.contrastive_pairs)

    @staticmethod
    def _build_pair_index(pairs: List[ContrastivePair]) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for p in pairs:
            index[p.sample_id_original] = p.pair_id
            index[p.sample_id_shifted] = p.pair_id
        return index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        if rec.actions is None or rec.rewards is None:
            raise RuntimeError(
                f"sample {rec.sample_id} 还未填入 actions/rewards; "
                "请确保 DemoGenerator 已生成 demonstrations。"
            )
        # rewards：必须经 normalizer 后再喂 encoder；缺 normalizer 时
        # 等价于使用原始数据（仅在单测/调试中允许）。
        rewards = rec.rewards
        trajectory_return = float(sum(rewards))
        if self.reward_normalizer is not None:
            rewards = self.reward_normalizer.transform(rewards)
            # transform 输入是 list 时返回 list；包成 list 让 torch.tensor 接受。
            if not isinstance(rewards, list):
                rewards = list(rewards)
        return {
            "states": torch.tensor(rec.states, dtype=torch.float32),
            "actions": torch.tensor(rec.actions, dtype=torch.long),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "trajectory_return": trajectory_return,
            "sample_id": rec.sample_id,
            "contrastive_pair_id": self._sample_to_pair.get(rec.sample_id, ""),
        }


def collate_phase1(batch: List[dict]) -> dict:
    """Phase1DemoDataset 的 collate fn。

    DataLoader 默认 collate 不能很好处理 string；这里手动拼。
    """
    states = torch.stack([b["states"] for b in batch], dim=0)
    actions = torch.stack([b["actions"] for b in batch], dim=0)
    rewards = torch.stack([b["rewards"] for b in batch], dim=0)
    trajectory_returns = torch.tensor(
        [b["trajectory_return"] for b in batch], dtype=torch.float32
    )
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "trajectory_returns": trajectory_returns,
        "sample_ids": [b["sample_id"] for b in batch],
        "contrastive_pair_ids": [b["contrastive_pair_id"] for b in batch],
    }
