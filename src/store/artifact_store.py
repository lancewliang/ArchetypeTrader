"""数据准备产出物读写类的接口骨架。"""

from __future__ import annotations

from pathlib import Path

from ..model.data_types import (
    ArtifactPaths,
    HorizonDataset,
    TrajectoryDataset,
)


class DataFileStore:
    """负责计算数据准备产出物路径，并读写中间数据集。

    为什么需要这个类:
        ``horizon_dataset`` 和 ``trajectory_dataset`` 都是需要持久化的产出物。
        将路径规划、保存和读取逻辑从 ``DataPreparer`` 中拆出，可以让
        ``DataPreparer`` 专注于流程编排，避免 I/O 细节散落在训练、测试、
        校验三个入口里。
    """

    def __init__(
        self,
        pair: str | None = None,
        batchid: str | None = None,
        artifacts_root: str | Path = "artifacts",
    ) -> None:
        """初始化数据产物 store。

        参数:
            pair: 可选交易标的，用于子类按标的隔离产物目录。
            batchid: 可选批次 ID，用于子类按批次隔离产物目录。
            artifacts_root: 全阶段产物根目录，默认 ``artifacts``。
        """

        self.pair = pair
        self.batchid = batchid
        self.artifacts_root = Path(artifacts_root)
        self.artifact_paths: ArtifactPaths = {}

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

    def load_horizon_dataset(
        self,
        split_name: str | Path,
    ) -> HorizonDataset:
        """读取 horizon 中间数据。

        参数:
            split_name: split 名称，例如 ``train``、``val``、``test``。

        输出:
            返回 ``HorizonDataset``，通常为 ``(states, prices)``。
            ``states`` shape 为 ``[x, h, len(states)]``。
            ``prices`` shape 为 ``[x, h, 1]``。

        方法作用:
            从已保存的产出物中恢复 ``HorizonBuilder`` 生成的 horizon 数据。

        为什么:
            数据准备产物需要可复用。后续调试、审计或重新生成
            ``trajectory_dataset`` 时，应能直接读取已固化的 horizon 数据，
            而不是重新读取 feature 文件并重新切分。
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

    def load_trajectory_dataset(
        self,
        split_name: str | Path,
    ) -> TrajectoryDataset:
        """读取 demonstration trajectory 数据集。

        参数:
            split_name: split 名称，例如 ``train``、``val``、``test``。

        输出:
            返回 ``TrajectoryDataset``。
            数据形式为 ``D = [tau_0, tau_1, ..., tau_{n-1}]``，
            每个 ``tau`` 都是 ``(s_demo, a_demo, r_demo)``。

        方法作用:
            从已保存的产出物中恢复 DP teacher 生成的 demonstration trajectories。

        为什么:
            Phase I 训练应直接消费固化后的 ``trajectory_dataset``。
            读取方法可以让训练流程、验证流程和审计流程复用同一份产物，
            避免重复运行 DP teacher。
        """
        ...


DataStore = DataFileStore
