"""Single-trade DP planner 的接口骨架。"""

from __future__ import annotations

import numpy as np

from ..model.data_types import DemonstrationTrajectory, HorizonDataset, TrajectoryDataset


class SingleTrade_DP_Planner:
    """为固定 horizon 生成 single-trade demonstration trajectories。

    该类对应论文中的 Single-trade DP planner。
    它的目标是在每个长度为 ``h`` 的 horizon 内，根据价格序列寻找一条
    最优或高质量的 teacher action 序列，并把状态、动作和奖励组合成：

    ``tau = (s_demo, a_demo, r_demo)``

    其中:
        ``s_demo`` 的 shape 为 ``[h, feature_dim]``。
        ``a_demo`` 的 shape 为 ``[h]``，动作取值为 ``{0, 1, 2}``。
        ``r_demo`` 的 shape 为 ``[h]``，表示逐步 reward。

    为什么需要这个类:
        Phase I 的 VQ encoder-decoder 需要 demonstration trajectories
        作为训练数据。DP teacher 只在离线训练数据生成阶段使用，避免在训练、
        验证、测试或推理阶段重新调用未来信息。
    """

    def __init__(
        self,
        horizon: int = 72,
        action_set: tuple[int, ...] = (0, 1, 2),
        initial_action: int = 1,
        gamma: float = 1.0,
    ) -> None:
        """初始化 Single-trade DP planner。

        参数:
            horizon: 每条 demonstration trajectory 的时间窗口长度 ``h``。
            action_set: 可选动作集合，默认 ``(0, 1, 2)``，分别表示 short、flat、long。
            initial_action: 每个 horizon 开始时的初始仓位动作，默认 ``1`` 表示 flat。
            gamma: DP 规划中的折扣系数，用于累计未来 reward。

        输出:
            无返回值。

        方法作用:
            记录 DP planner 的基本配置，确保后续
            ``build_trajectory`` 使用一致的 horizon、动作空间和初始仓位假设。

        为什么:
            demonstration trajectories 必须由同一套交易语义生成，否则 Phase I
            学到的 archetype 会混合不一致的动作和 reward 定义。
        """
        ...
   
    def build_trajectory(
        self,
        states: np.ndarray,
        prices: np.ndarray,
    ) -> DemonstrationTrajectory:
        """为单个 horizon 生成 demonstration trajectory。

        参数:
            states: 单个 horizon 的市场状态矩阵，shape 为 ``[h, feature_dim]``。
            prices: 单个 horizon 的价格序列，shape 为 ``[h]``。

        输出:
            返回 ``DemonstrationTrajectory``，即 ``tau = (s_demo, a_demo, r_demo)``。
            ``s_demo`` 的 shape 为 ``[h, feature_dim]``。
            ``a_demo`` 的 shape 为 ``[h]``。
            ``r_demo`` 的 shape 为 ``[h]``。

        方法作用:
            先调用 ``plan`` 生成 teacher actions，再调用 ``compute_rewards`` 生成
            step rewards，最后把 ``states``、``actions``、``rewards`` 组合成论文中的
            demonstration tuple。

        为什么:
            VQ encoder-decoder 的训练样本单位是完整的 ``tau``，
            而不是孤立的动作序列。该方法把 DP teacher 输出整理成模型训练契约。
        """
        ...

    def build_trajectory_dataset(
        self,
        horizon_dataset: HorizonDataset,
    ) -> TrajectoryDataset:
        """批量生成 demonstration trajectory 数据集 ``D``。

        参数:
            horizon_dataset: ``HorizonBuilder`` 的输出，即 ``(states, prices)``。
                ``states`` 的 shape 为 ``[n, h, feature_dim]``。
                ``prices`` 的 shape 为 ``[n, h, 1]``，来自 feature 文件的 ``close`` 列。

        输出:
            返回 ``TrajectoryDataset``，即 ``D = [tau_0, tau_1, ..., tau_{n-1}]``。
            每个 ``tau_i`` 都是 ``(s_demo, a_demo, r_demo)``。

        方法作用:
            从 ``horizon_dataset`` 中取出每个 horizon 的 ``states`` 和 ``prices``，
            对每个 sampled horizon 调用 ``build_trajectory``，
            批量生成 Phase I 训练所需的 demonstration trajectories。

        为什么:
            论文中的训练数据集定义为 ``D = {tau_i}_{i=0}^{n-1}``。
            该方法把单条 trajectory 生成逻辑扩展到完整训练集。
        """
        ...
 
