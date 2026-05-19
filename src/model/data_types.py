"""数据准备阶段共用类型定义。

本文件只定义数据结构的类型别名和语义说明，不负责具体数据生成或保存。

核心数据流:
    feature 文件
        -> ``HorizonDataset``
        -> ``TrajectoryDataset``

其中 feature 文件由 ``DataLoad`` 读取为 ``pl.DataFrame``；
``HorizonDataset`` 由 ``HorizonBuilder`` 生成；
``TrajectoryDataset`` 由 ``SingleTrade_DP_Planner`` 生成。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


HorizonDataset = tuple[np.ndarray, np.ndarray, np.ndarray]
"""固定 horizon 的中间数据集。

结构:
    ``HorizonDataset = (states, prices, depthprices)``

形状:
    ``states``: ``[x, h, feature_dim]``
        ``x`` 表示 horizon 样本数量。
        ``h`` 表示每个 horizon 的时间步长度，论文实验默认 72。
        ``feature_dim`` 表示状态特征数量。

    ``prices``: ``[x, h, 1]``
        价格来自 feature ``DataFrame`` 的 ``close`` 列。
        最后一维为 1，是为了保留和状态张量一致的三维结构。

    ``depthprices``: ``[x, h, 20]``
        深度行情来自 ``states`` 中的五档 ask/bid price 和 size 特征。
        列顺序为 ``ask1_price`` ... ``ask5_price``、
        ``ask1_size`` ... ``ask5_size``、
        ``bid1_price`` ... ``bid5_price``、
        ``bid1_size`` ... ``bid5_size``。

含义:
    ``states`` 是模型观察到的市场状态序列，不包含 ``close`` 价格列。
    ``prices`` 是 DP teacher 和 reward 计算所需的结算价格序列。
    ``depthprices`` 是从 ``states`` 中额外切出的盘口深度行情序列。

为什么:
    状态输入和价格序列用途不同。状态给模型学习，
    价格给 ``SingleTrade_DP_Planner`` 生成 ``a_demo`` 和 ``r_demo``。
"""

DemonstrationTrajectory = tuple[np.ndarray, np.ndarray, np.ndarray]
"""单条 demonstration trajectory。

结构:
    ``DemonstrationTrajectory = (s_demo, a_demo, r_demo)``

形状:
    ``s_demo``: ``[h, feature_dim]``
        单个 horizon 的状态序列。

    ``a_demo``: ``[h]``
        单个 horizon 的 DP teacher 动作序列。
        动作取值约定为 ``{0, 1, 2}``，分别表示 short、flat、long。

    ``r_demo``: ``[h]``
        单个 horizon 的逐步 reward 序列。

含义:
    ``DemonstrationTrajectory`` 是论文中的 ``tau``，
    是 Phase I VQ encoder-decoder 的单个训练样本。
"""

TrajectoryDataset = list[DemonstrationTrajectory]
"""Demonstration trajectory 数据集。

结构:
    ``TrajectoryDataset = [tau_0, tau_1, ..., tau_{n-1}]``
    ``tau = (s_demo, a_demo, r_demo)``

形状:
    每个 ``tau`` 的形状见 ``DemonstrationTrajectory``。

含义:
    每个 ``tau`` 是论文中的 demonstration trajectory，
    用作 Phase I VQ encoder-decoder 的训练样本。

为什么:
    Phase I 不是只学习动作序列，而是从状态、teacher 动作和 reward 路径中
    提取可复用的 trading archetype。
"""

ArtifactPaths = dict[str, Path]
"""数据准备产出物路径集合。

结构:
    ``ArtifactPaths`` 是从产物名称到文件路径的字典。

推荐键:
    ``horizon_dataset``:
        ``HorizonDataset`` 的保存路径。

    ``trajectory_dataset``:
        ``TrajectoryDataset`` 的保存路径。

    ``state_normalizer``:
        ``state`` 特征归一化参数的保存路径。

含义:
    描述一次数据准备流程会写出的中间产物位置。

为什么:
    ``DataStore`` 统一管理产物路径，可以避免 train/test/validation
    使用不一致的命名规则或目录结构。
"""
