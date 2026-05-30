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


HorizonDataset = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
"""固定 horizon 的中间数据集。

结构:
    ``HorizonDataset = (states, relative_states, trend_states, prices, depthprices)``

形状:
    ``states``: ``[x, h, feature_dim]``
        ``x`` 表示 horizon 样本数量。
        ``h`` 表示每个 horizon 的时间步长度，论文实验默认 72。
        ``feature_dim`` 表示市场状态特征数量。

    ``relative_states``: ``[x, h, relative_feature_dim]``
        相对状态特征，来自 ``relative_need_normalization`` 和 ``relative`` block。

    ``trend_states``: ``[x, h, trend_feature_dim]``
        趋势状态特征，来自长短周期 trend block。

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
    ``states``、``relative_states`` 和 ``trend_states`` 是模型观察到的三路状态
    序列，不包含 ``close`` 价格列。
    ``prices`` 是 DP teacher 和 reward 计算所需的结算价格序列。
    ``depthprices`` 是从原始 LOB 行情列切出的盘口深度行情序列，保持未归一化尺度。

为什么:
    状态输入和价格序列用途不同。三路状态给模型学习，价格和未归一化 LOB
    深度给 ``SingleTrade_DP_Planner`` 生成 ``a_demo`` 和 ``r_demo``。
"""

DemonstrationTrajectory = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]
"""单条 demonstration trajectory。

结构:
    ``DemonstrationTrajectory = (
        s_demo, relative_s_demo, trend_s_demo, a_demo, r_demo, sample_id
    )``

形状:
    ``s_demo``: ``[h, feature_dim]``
        单个 horizon 的市场状态序列。

    ``relative_s_demo``: ``[h, relative_feature_dim]``
        单个 horizon 的相对状态序列。

    ``trend_s_demo``: ``[h, trend_feature_dim]``
        单个 horizon 的趋势状态序列。

    ``a_demo``: ``[h]``
        单个 horizon 的 DP teacher 动作序列。
        动作取值约定为 ``{0, 1, 2}``，分别表示 short、flat、long。

    ``r_demo``: ``[h]``
        单个 horizon 的逐步 reward 序列。

    ``sample_id``: scalar
        当前 trajectory 在 split 内的稳定 horizon 样本编号。

含义:
    ``DemonstrationTrajectory`` 是论文中的 ``tau``，
    是 Phase I VQ encoder-decoder 的单个训练样本。
"""

TrajectoryDataset = list[DemonstrationTrajectory]
"""Demonstration trajectory 数据集。

结构:
    ``TrajectoryDataset = [tau_0, tau_1, ..., tau_{n-1}]``
    ``tau = (s_demo, relative_s_demo, trend_s_demo, a_demo, r_demo, sample_id)``

形状:
    每个 ``tau`` 的形状见 ``DemonstrationTrajectory``。

含义:
    每个 ``tau`` 是论文中的 demonstration trajectory，
    用作 Phase I VQ encoder-decoder 的训练样本。

为什么:
    Phase I 不是只学习动作序列，而是从状态、teacher 动作和 reward 路径中
    提取可复用的 trading archetype。
"""

TSize = int
"""Phase II selector 可见状态的 t 步长。

含义:
    ``TSize`` 定义 ``VisibleStatesDataset`` 中当前分片 t 状态窗口的长度。若
    ``TSize == 4``，则当前分片可见状态包含 4 个连续 timestep。
"""

VisibleStates = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]

VisibleStatesBatch = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]

VisibleStatesDataset = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]
"""Phase II selector 输入模型的可见状态数据集。

结构:
    ``VisibleStatesDataset = (
        previous_t_states,
        previous_t_relative_states,
        previous_t_trend_states,
        current_t_states,
        current_t_relative_states,
        current_t_trend_states,
    )``

形状:
    ``previous_t_states``: ``[x - 1, horizon, feature_dim]``
        上一个分片的所有 t 状态。第 0 条样本没有上一分片，因此不会形成
        selector 训练样本。

    ``previous_t_relative_states``: ``[x - 1, horizon, relative_feature_dim]``
        上一个分片的所有相对状态。

    ``previous_t_trend_states``: ``[x - 1, horizon, trend_feature_dim]``
        上一个分片的所有趋势状态。

    ``current_t_states``: ``[x - 1, TSize, feature_dim]``
        当前分片的所有 t 状态，窗口长度由 ``TSize`` 定义。

    ``current_t_relative_states``: ``[x - 1, TSize, relative_feature_dim]``
        当前分片可见窗口内的相对状态。

    ``current_t_trend_states``: ``[x - 1, TSize, trend_feature_dim]``
        当前分片可见窗口内的趋势状态。

含义:
    这是 Phase II selector 选择模型的直接输入类型。它只包含 selector 在线可见
    的状态信息，不包含当前分片未来状态、价格、teacher action 或 reward。
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

DemonstrationHorizonLabel = tuple[int, int]
DemonstrationHorizonLabelBatch = tuple[np.ndarray, np.ndarray]
DemonstrationHorizonLabelDataset = tuple[np.ndarray, np.ndarray]
"""Phase I demonstration horizon 的标签向量数据集。

结构:
    ``DemonstrationHorizonLabelDataset = (sample_ids, code_labels)``

来源:
    由 ``HorizonTrainLabelBuilder`` 生成的 ``HorizonTrainLabelRow`` 序列派生而来。
    其中 ``sample_ids`` 对应 ``HorizonTrainLabelRow.sample_id`` 字段，
    ``code_labels`` 对应 ``HorizonTrainLabelRow.code_label`` 字段。

形状:
    ``sample_ids``: ``[x]``
        horizon 样本位置索引，通常为完整的零基连续区间 ``0..x-1``。

    ``code_labels``: ``[x]``
        Phase I VQ codebook 分配给每个 horizon 的 archetype id。

含义:
    这是 Phase I 离线导出的 horizon-level supervision。Phase II selector
    可用 ``sample_ids`` 将 label 与 ``HorizonDataset`` 的样本行对齐，并用
    ``code_labels`` 作为选择模型的 supervised/imitation target。
"""

VisibleStatesLabelDataset = tuple[np.ndarray, np.ndarray]
"""
Phase II selector 输出的的标签向量数据集。
结构:
    ``HorizonLabelDataset = (sample_ids, code_labels)``
"""