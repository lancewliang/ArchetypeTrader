"""模型训练阶段共用 Tensor 类型定义。

本文件是 ``data_types.py`` 的 PyTorch Tensor 版本，描述进入模型后的
数据结构、形状和 dtype 约定，并提供 numpy trajectory 到 PyTorch
``TensorDataset`` 的标准转换和 batch device 搬运工具。

核心数据流:
    ``HorizonDataset`` / ``TrajectoryDataset`` 的 numpy 产物
        -> PyTorch Dataset / collate
        -> 本文件定义的 Tensor batch
        -> Phase I VQ encoder-decoder

为什么需要单独的 Tensor 类型:
    ``data_types.py`` 面向数据准备和落盘，使用 ``np.ndarray`` 更方便和
    Polars、feather、pickle 等数据流程衔接；模型训练阶段则需要
    ``torch.Tensor`` 来参与 GPU 计算、自动求导和 batch 化训练。
    单独定义 Tensor 版本可以避免在模型代码里反复猜测形状和 dtype。
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .data_types import TrajectoryDataset


HorizonTensorDataset: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
"""固定 horizon 的 Tensor 数据集。

结构:
    ``HorizonTensorDataset = (states, relative_states, trend_states, prices, depthprices)``

形状:
    ``states``: ``[x, h, feature_dim]``
        ``x`` 表示 horizon 样本数量。
        ``h`` 表示每个 horizon 的时间步长度，论文实验默认 72。
        ``feature_dim`` 表示状态特征数量。

    ``relative_states``: ``[x, h, relative_feature_dim]``
        相对状态特征，来自 ``relative_need_normalization`` 和 ``relative`` block。

    ``trend_states``: ``[x, h, trend_feature_dim]``
        趋势状态特征，来自长短周期 trend block。

    ``prices``: ``[x, h, 1]``
        价格来自原始 feature ``DataFrame`` 的 ``close`` 列。

    ``depthprices``: ``[x, h, 20]``
        深度行情来自 ``states`` 中的五档 ask/bid price 和 size 特征。

dtype:
    ``states`` 通常为 ``torch.float32``。
    ``relative_states`` 通常为 ``torch.float32``。
    ``trend_states`` 通常为 ``torch.float32``。
    ``prices`` 通常为 ``torch.float32``。
    ``depthprices`` 通常为 ``torch.float32``。

含义:
    这是 ``HorizonDataset`` 进入 PyTorch 后的表示，主要用于 batch 化、
    GPU 搬运，以及后续生成 demonstration trajectory。
"""

DemonstrationTrajectoryTensor: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
"""单条 demonstration trajectory 的 Tensor 表示。

结构:
    ``DemonstrationTrajectoryTensor = (
        s_demo, relative_s_demo, trend_s_demo, a_demo, r_demo
    )``

形状:
    ``s_demo``: ``[h, feature_dim]``
        单个 horizon 的状态序列。

    ``relative_s_demo``: ``[h, relative_feature_dim]``
        单个 horizon 的相对状态序列。

    ``trend_s_demo``: ``[h, trend_feature_dim]``
        单个 horizon 的趋势状态序列。

    ``a_demo``: ``[h]``
        DP teacher 动作序列。

    ``r_demo``: ``[h]``
        单个 horizon 的逐步 reward 序列。

dtype:
    ``s_demo`` 通常为 ``torch.float32``。
    ``relative_s_demo`` 通常为 ``torch.float32``。
    ``trend_s_demo`` 通常为 ``torch.float32``。
    ``a_demo`` 必须为 ``torch.long``，用于 ``Embedding`` 或
    ``cross_entropy`` target。
    ``r_demo`` 通常为 ``torch.float32``。

动作约定:
    ``a_demo`` 取值为 ``{0, 1, 2}``，分别表示 short、flat、long。

为什么:
    Phase I 的 encoder 不只看市场状态，还看 teacher 动作和 reward 路径。
    这些 Tensor 一起构成论文里的 ``tau``，用于蒸馏离散 trading archetype。
"""

TrajectoryTensorDataset: TypeAlias = list[DemonstrationTrajectoryTensor]
"""Demonstration trajectory Tensor 数据集。

结构:
    ``TrajectoryTensorDataset = [tau_0, tau_1, ..., tau_{n-1}]``
    ``tau = (s_demo, relative_s_demo, trend_s_demo, a_demo, r_demo)``

含义:
    这是 ``TrajectoryDataset`` 转成 Tensor 后的逐样本集合。
    适合在自定义 ``Dataset`` 中按 index 返回单条 trajectory。
"""

TrajectoryTensorBatch: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
"""Phase I 训练 batch。

结构:
    ``TrajectoryTensorBatch = (states, relative_states, trend_states, actions, rewards)``

形状:
    ``states``: ``[batch, h, feature_dim]``
        batch 内多条 demonstration trajectory 的状态序列。

    ``relative_states``: ``[batch, h, relative_feature_dim]``
        batch 内多条 demonstration trajectory 的相对状态序列。

    ``trend_states``: ``[batch, h, trend_feature_dim]``
        batch 内多条 demonstration trajectory 的趋势状态序列。

    ``actions``: ``[batch, h]``
        batch 内多条 DP teacher 动作序列。

    ``rewards``: ``[batch, h]``
        batch 内多条逐步 reward 序列。 

dtype:
    ``states`` 通常为 ``torch.float32``。
    ``relative_states`` 通常为 ``torch.float32``。
    ``trend_states`` 通常为 ``torch.float32``。
    ``actions`` 必须为 ``torch.long``。
    ``rewards`` 通常为 ``torch.float32``。

用途:
    这是 ``VQArchetype`` 模型 ``forward`` 最自然的输入形式。
    当前 encoder 用主 ``states/actions/rewards`` 生成连续 latent，VQ codebook
    把 latent 离散化为 archetype label，decoder 再根据
    ``states`` 和选中的 archetype 重构 ``actions``。
"""


def build_trajectory_tensor_dataset(
    trajectory_dataset: TrajectoryDataset,
) -> TensorDataset:
    """将 numpy ``TrajectoryDataset`` 转为 Phase I 训练用 ``TensorDataset``。

    返回的 ``TensorDataset`` 包含第六列稳定 ``sample_ids``。模型训练和评估通过
    ``move_trajectory_batch_to_device()`` 消费前五列 Tensor；codebook validation
    evaluator 可读取第六列做 assignment churn 和 prices 对齐。
    """

    states = torch.as_tensor(
        np.stack([trajectory[0] for trajectory in trajectory_dataset]),
        dtype=torch.float32,
    )
    relative_states = torch.as_tensor(
        np.stack([trajectory[1] for trajectory in trajectory_dataset]),
        dtype=torch.float32,
    )
    trend_states = torch.as_tensor(
        np.stack([trajectory[2] for trajectory in trajectory_dataset]),
        dtype=torch.float32,
    )
    actions = torch.as_tensor(
        np.stack([trajectory[3] for trajectory in trajectory_dataset]),
        dtype=torch.long,
    )
    rewards = torch.as_tensor(
        np.stack([trajectory[4] for trajectory in trajectory_dataset]),
        dtype=torch.float32,
    )
    sample_ids = torch.arange(len(trajectory_dataset), dtype=torch.long)
    return TensorDataset(
        states,
        relative_states,
        trend_states,
        actions,
        rewards,
        sample_ids,
    )


def move_trajectory_batch_to_device(
    batch: tuple[torch.Tensor, ...],
    device: torch.device | str,
) -> TrajectoryTensorBatch:
    """将 Phase I trajectory batch 搬到目标 device。

    输入 batch 可包含第六列 ``sample_ids``。返回值保持和
    ``DemonstrationTrajectoryTensor`` 对齐的前五列稳定契约。
    为兼容旧产物，本函数也接受 ``(states, actions, rewards[, sample_ids])``，
    并补出空的 ``relative_states`` / ``trend_states``。
    """

    if len(batch) >= 5:
        states, relative_states, trend_states, actions, rewards = batch[:5]
    else:
        raise ValueError("trajectory batch must contain at least states, actions, rewards")
    return (
        states.to(device),
        relative_states.to(device),
        trend_states.to(device),
        actions.to(device),
        rewards.to(device),
    )


ArchetypeLabelTensor: TypeAlias = torch.Tensor
"""VQ encoder 分配的 archetype label。

形状:
    ``[batch]`` 或 ``[x]``

dtype:
    必须为 ``torch.long``。

含义:
    每个元素是一个 codebook index，表示对应 horizon 被归入哪个
    trading archetype。该 label 后续可作为 Phase II archetype selector
    的 supervised signal 或 KL 约束目标。
"""

ActionLogitTensor: TypeAlias = torch.Tensor
"""decoder 输出的动作 logits。

形状:
    ``[batch, h, 3]``

dtype:
    通常为 ``torch.float32``。

含义:
    第三维对应 short、flat、long 三个动作类别。
    训练时可直接传入 ``torch.nn.functional.cross_entropy``；
    推理时通常通过 ``argmax(dim=-1)`` 得到基础 archetype 动作。
"""

LatentTensor: TypeAlias = torch.Tensor
"""VQ encoder 和 codebook 使用的 latent / archetype embedding。

形状:
    ``[batch, code_dim]`` 或 ``[num_archetypes, code_dim]``

dtype:
    通常为 ``torch.float32``。

含义:
    ``z_e`` 是 encoder 输出的连续 latent；
    ``z_q`` 是从 codebook 中选出的离散 archetype 向量。
"""


__all__ = [
    "ActionLogitTensor",
    "ArchetypeLabelTensor",
    "DemonstrationTrajectoryTensor",
    "HorizonTensorDataset",
    "LatentTensor",
    "TrajectoryTensorBatch",
    "TrajectoryTensorDataset",
    "build_trajectory_tensor_dataset",
    "move_trajectory_batch_to_device",
]
