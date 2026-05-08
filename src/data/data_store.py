"""数据准备产出物存储类的接口骨架。"""

from __future__ import annotations

from pathlib import Path

from .data_types import ArtifactPaths, HorizonDataset, TrajectoryDataset


class DataStore:
    """负责计算产出物路径并保存数据准备中间产物。

    为什么需要这个类:
        ``horizon_dataset`` 和 ``trajectory_dataset`` 都是需要持久化的产出物。
        将路径规划和保存逻辑从 ``DataPreparer`` 中拆出，可以让
        ``DataPreparer`` 专注于流程编排，避免 I/O 细节散落在训练、测试、
        校验三个入口里。
    """

    def build_artifact_paths(
        self,
        path: str | Path,
        split_name: str,
    ) -> ArtifactPaths:
        """计算数据准备产出物的存储路径。

        参数:
            path: feature 输入文件路径。
            split_name: 数据集名称，例如 ``train``、``test`` 或 ``validation``。

        输出:
            返回产出物路径字典。
            至少包含:
                ``horizon_dataset``: horizon 中间数据的保存路径。
                ``trajectory_dataset``: demonstration trajectory 数据的保存路径。

        方法作用:
            根据输入文件和 split 名称，提前规划数据准备阶段的产物位置。

        为什么:
            路径命名规则应集中管理，避免 train/test/validation 各自拼路径，
            导致产物位置和命名不一致。
        """
        ...

    def save_horizon_dataset(
        self,
        horizon_dataset: HorizonDataset,
        output_path: str | Path,
    ) -> None:
        """保存 horizon 中间数据。

        参数:
            horizon_dataset: ``HorizonBuilder`` 的输出。
                通常为 ``(states, prices)``。
                ``states`` shape 为 ``[x, h, len(states)]``。
                ``prices`` shape 为 ``[x, h, 1]``。
            output_path: horizon 中间数据保存路径。

        输出:
            无返回值。

        方法作用:
            将 ``HorizonBuilder`` 生成的 horizon 数据作为独立产物保存。

        为什么:
            horizon 数据是从 feature 文件到 trajectory 数据之间的重要中间结果。
            单独保存可以支持复查 ``close`` 价格切分、状态 shape 和后续 DP 输入。
        """
        ...

    def save_trajectory_dataset(
        self,
        trajectory_dataset: TrajectoryDataset,
        output_path: str | Path,
    ) -> None:
        """保存 demonstration trajectory 数据集。

        参数:
            trajectory_dataset: ``SingleTrade_DP_Planner`` 的输出。
                数据形式为 ``D = [tau_0, tau_1, ..., tau_{n-1}]``。
                每个 ``tau`` 都是 ``(s_demo, a_demo, r_demo)``。
            output_path: trajectory 数据集保存路径。

        输出:
            无返回值。

        方法作用:
            将 DP teacher 生成的 demonstration trajectories 保存为训练产物。

        为什么:
            Phase I 训练应消费已经固化的 trajectory 数据集，
            而不是在训练过程中重新读取 feature 文件或重新运行 DP。
        """
        ...
