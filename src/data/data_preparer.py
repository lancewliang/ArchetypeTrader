"""Phase I 数据集准备流程编排类的接口骨架。"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .data_load import DataLoad
from .feature_spec import build_feature_input_spec
from .horizon_builder import HorizonBuilder
from .resolve_factor import FACTORS_ROOT
from .state_normalizer import StateNormalizer
from ..model.data_types import TrajectoryDataset
from ..store.artifact_store import DataStore
from ..tool.SingleTrade_DP_Planner import SingleTrade_DP_Planner


class DataPreparer:
    """编排训练、测试和校验数据集准备流程。

    该类不直接负责文件读取和产物保存。
    它只负责把以下组件串起来：
        1. ``DataLoad``: 根据 ``path`` 读取 feature 文件，得到 ``pl.DataFrame``。
        2. ``HorizonBuilder``: 从 ``pl.DataFrame`` 生成 ``horizon_dataset``。
        3. ``SingleTrade_DP_Planner``: 从 ``horizon_dataset`` 生成 ``trajectory_dataset``。
        4. ``DataStore``: 计算产物路径，并读写 ``horizon_dataset`` 和
           ``trajectory_dataset``。

    数据集的核心样本是论文中的 demonstration trajectory：

    ``tau = (s_demo, a_demo, r_demo)``

    其中:
        ``s_demo`` 是 horizon 状态序列，shape 为 ``[h, feature_dim]``。
        ``a_demo`` 是 DP teacher 动作序列，shape 为 ``[h]``。
        ``r_demo`` 是逐步 reward 序列，shape 为 ``[h]``。

    为什么需要这个类:
        训练、测试和校验数据集应使用同一套数据准备语义。
        ``DataPreparer`` 作为流程入口，可以避免三个 split 各自拼装流程。
        文件 I/O 和产物存储拆到独立类后，该类只保留数据准备的业务顺序。
    """

    def __init__(
        self,
        horizon: int = 72,
        pair: str | None = None,
        data_store: DataStore | None = None,
    ) -> None:
        """初始化 DataPreparer。

        参数:
            horizon: 每个样本的固定时间窗口长度 ``h``，默认 72。
            pair: 交易标的。DataPreparer 会在内部构建 ``FeatureInputSpec``，
                并用其 required columns 初始化 ``DataLoad``。
            data_store: 数据产物读写组件。用于计算产物路径、保存中间产物，
                以及读取已经固化的产出物。

        输出:
            无返回值。

        方法作用:
            保存数据准备流程所需的共用配置和组件引用。

        为什么:
            train/test/validation 必须使用相同的 horizon 长度和同一套
            数据读取、产物读写规则，否则数据集之间的 schema 和审计方式会不一致。
        """
        self.horizon = horizon
        self.factors_root = Path(FACTORS_ROOT)
        self.pair = pair
        self.feature_spec = build_feature_input_spec(
            pair=pair,
            factors_root=self.factors_root,
        )
        feature_columns = list(self.feature_spec.required_columns)
        self.data_load = DataLoad(feature_columns=feature_columns)
        self.data_store = data_store or DataStore(artifacts_root="data")
        self.horizon_builder = HorizonBuilder(
            horizon=horizon,
            feature_spec=self.feature_spec,
        )
        self.feature_normalizers: dict[str, StateNormalizer] | None = None
        self.dp_planner = SingleTrade_DP_Planner(horizon=horizon)

    def _fit_feature_normalizers(
        self,
        dataframe: pl.DataFrame,
    ) -> dict[str, StateNormalizer]:
        """Fit block-level normalizers required by the three-input feature spec."""

        normalizers: dict[str, StateNormalizer] = {}
        for block in self.feature_spec.iter_blocks():
            if not block.normalize:
                continue
            key = block.effective_normalizer_key
            if key in normalizers:
                if tuple(block.columns) != normalizers[key].feature_columns:
                    raise ValueError(
                        f"normalizer key {key!r} is reused for different columns"
                    )
                continue
            normalizers[key] = StateNormalizer.fit(dataframe, block.columns)
        return normalizers

    def _validate_feature_normalizers(
        self,
        normalizers: dict[str, StateNormalizer],
    ) -> None:
        """Validate that all normalized feature blocks have fitted parameters."""

        missing = [
            key
            for key in self.feature_spec.normalizer_keys
            if key not in normalizers
        ]
        if missing:
            raise ValueError(
                "missing feature normalizers: " + ", ".join(missing)
            )

        for block in self.feature_spec.iter_blocks():
            if not block.normalize:
                continue
            key = block.effective_normalizer_key
            if tuple(block.columns) != normalizers[key].feature_columns:
                raise ValueError(
                    f"feature normalizer {key!r} columns do not match block columns"
                )

    def _prepare_dataset(
        self,
        path: str | Path,
        split_name: str,
    ) -> TrajectoryDataset:
        """准备单个 split 的数据集。

        参数:
            path: 当前 split 的 feature 输入文件路径。
            split_name: 数据集名称，例如 ``train``、``test`` 或 ``validation``。

        输出:
            返回 ``trajectory_dataset``。
            同时会将 ``horizon_dataset`` 和 ``trajectory_dataset`` 分别保存为产出物。

        方法作用:
            作为 ``prepare_train_dataset``、``prepare_test_dataset`` 和
            ``prepare_validation_dataset`` 的共用流程骨架。

        基本流程:
            1. 调用 ``DataLoad.load_feature_file(path)`` 得到 ``pl.DataFrame``。
            2. 调用 ``DataStore.build_artifact_paths(path, split_name)`` 计算产物路径。
            3. 在 train split 上拟合 feature block normalizers，或在非 train split
               上读取已保存的归一化参数。
            4. 将 feature block normalizers 注入 ``HorizonBuilder``。
            5. 调用 ``HorizonBuilder.build(dataframe)`` 得到 ``horizon_dataset``。
            6. 调用 ``SingleTrade_DP_Planner.build_trajectory_dataset(horizon_dataset)``
               得到 ``trajectory_dataset``。
            7. 调用 ``DataStore.save_horizon_dataset(...)`` 保存 horizon 中间数据。
            8. 调用 ``DataStore.save_trajectory_dataset(...)`` 保存 trajectory 数据集。
            9. 若是 train split，则保存 feature block normalizers。

        读取产物:
            如果后续流程需要复用已保存的中间产物，可通过
            ``DataStore.load_horizon_dataset(...)`` 和
            ``DataStore.load_trajectory_dataset(...)`` 读取，不需要在本类中重复实现 I/O。

        为什么:
            三个 split 的准备流程相同，只是输入文件和 split 名称不同。
            抽成内部方法可以减少重复逻辑，并确保 train/test/validation
            使用完全一致的数据准备契约。
        """
        dataframe = self.data_load.load_feature_file(path)
        artifact_paths = self.data_store.build_artifact_paths(path, split_name)
        if split_name == "train":
            self.feature_normalizers = self._fit_feature_normalizers(dataframe)
        elif self.feature_normalizers is None:
            self.feature_normalizers = self.data_store.load_feature_normalizers(
                artifact_paths["feature_normalizers"],
            )
        if self.feature_normalizers is None:
            raise RuntimeError("feature_normalizers must be initialized")
        self._validate_feature_normalizers(self.feature_normalizers)
        self.horizon_builder.set_feature_normalizers(self.feature_normalizers)
        horizon_dataset = self.horizon_builder.build(dataframe)
        trajectory_dataset = self.dp_planner.build_trajectory_dataset(horizon_dataset)
        self.data_store.save_horizon_dataset(
            horizon_dataset,
            artifact_paths["horizon_dataset"],
        )
        self.data_store.save_trajectory_dataset(
            trajectory_dataset,
            artifact_paths["trajectory_dataset"],
        )
        if split_name == "train":
            self.data_store.save_feature_normalizers(
                self.feature_normalizers,
                artifact_paths["feature_normalizers"],
            )
        return trajectory_dataset

    def prepare_train_dataset(
        self,
        path: str | Path,
    ) -> TrajectoryDataset:
        """准备训练数据集。

        参数:
            path: 训练 split 的 feature 输入文件路径。

        输出:
            返回训练用 ``D_train``。
            ``D_train = [tau_0, tau_1, ..., tau_{n-1}]``，
            每个 ``tau`` 都是 ``(s_demo, a_demo, r_demo)``。
            同时保存训练 split 的 ``horizon_dataset`` 和 ``trajectory_dataset`` 产物。

        方法作用:
            使用训练 feature 文件生成 Phase I 训练所需的 demonstration trajectories。

        为什么:
            VQ encoder-decoder 的参数学习只依赖训练数据集。
            训练集准备必须独立、可复现，避免混入测试或校验数据。
        """
        return self._prepare_dataset(path, "train")

    def prepare_test_dataset(
        self,
        path: str | Path,
    ) -> TrajectoryDataset:
        """准备测试数据集。

        参数:
            path: 测试 split 的 feature 输入文件路径。

        输出:
            返回测试用 ``D_test``。
            ``D_test = [tau_0, tau_1, ..., tau_{n-1}]``，
            每个 ``tau`` 都是 ``(s_demo, a_demo, r_demo)``。
            同时保存测试 split 的 ``horizon_dataset`` 和 ``trajectory_dataset`` 产物。

        方法作用:
            使用测试 feature 文件生成与训练集同 schema 的 demonstration trajectories。

        为什么:
            测试集用于最终评估模型泛化能力。
            它必须和训练集格式一致，但不能参与模型训练或超参数选择。
        """
        return self._prepare_dataset(path, "test")

    def prepare_validation_dataset(
        self,
        path: str | Path,
    ) -> TrajectoryDataset:
        """准备校验数据集。

        参数:
            path: 校验 split 的 feature 输入文件路径。

        输出:
            返回校验用 ``D_validation``。
            ``D_validation = [tau_0, tau_1, ..., tau_{n-1}]``，
            每个 ``tau`` 都是 ``(s_demo, a_demo, r_demo)``。
            同时保存校验 split 的 ``horizon_dataset`` 和 ``trajectory_dataset`` 产物。

        方法作用:
            使用校验 feature 文件生成与训练集同 schema 的 demonstration trajectories。

        为什么:
            校验集用于训练过程中的模型选择、早停和超参数判断。
            它需要独立于训练集和测试集，避免评估结果被数据泄漏污染。
        """
        return self._prepare_dataset(path, "val")
