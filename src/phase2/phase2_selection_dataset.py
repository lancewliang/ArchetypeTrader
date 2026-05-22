"""Phase II selector dataset builder.

本模块负责把数据准备阶段产出的 ``HorizonDataset`` 和 Phase I 离线导出的
archetype label 表对齐，生成 Phase II horizon-level selector 训练数据。

输入:
    horizon_dataset:
        ``HorizonDataset = (states, relative_states, trend_states, prices, depthprices)``。
        ``states`` 形状为 ``[sample, horizon, feature_dim]``；
        ``relative_states`` 形状为 ``[sample, horizon, relative_feature_dim]``；
        ``trend_states`` 形状为 ``[sample, horizon, trend_feature_dim]``；
        ``prices`` 形状为 ``[sample, horizon, price_dim]``；
        ``depthprices`` 形状为 ``[sample, horizon, depth_dim]`` 或 ``None``。
    label_table:
        Phase I 离线导出的 ``polars.DataFrame``，至少包含
        ``sample_id`` 和 ``code_label`` 两列。``sample_id`` 必须是完整的
        ``[0, sample_count)``，表示 Phase I label 表中与 ``HorizonDataset``
        行位置对齐的 horizon 样本索引；``code_label`` 是对应 horizon 的
        assigned archetype id。

输出:
    Phase2SelectionDataset:
        Phase II selector 使用的 numpy dataset，包含 selector 可见的
        ``visible_states``、环境模拟需要的 ``horizon_dataset``，以及 imitation
        regularization 使用的 ``demonstration_horizon_label_dataset``。

边界:
    - 不加载 Phase I checkpoint。
    - 不调用 Phase I encoder。
    - 不读取 ``TrajectoryDataset`` 或 DP teacher trajectory。
    - selector observation 只使用上一分片完整三路状态序列和当前分片三路 t 状态窗口。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch.utils.data import TensorDataset

from ..model.data_types import (
    DemonstrationHorizonLabelDataset,
    HorizonDataset,
    TSize,
    VisibleStatesDataset,
)


@dataclass(frozen=True)
class Phase2SelectionDataset:
    """Phase II archetype selector 的 numpy 数据集。

    ``visible_states`` 是 selector 在线推理可见的 observation，结构为
    ``(previous_t_states, previous_t_relative_states, previous_t_trend_states,
    current_t_states, current_t_relative_states, current_t_trend_states)``；
    ``horizon_dataset`` 只供 frozen decoder 和 reward/environment 计算使用，
    不应拼入 selector observation。

    字段:
        visible_states: 输入给 selector、用于训练选择模型的可见状态，即
            ``VisibleStatesDataset``。前三个元素是上一分片的完整三路状态序列，
            后三个元素是当前分片的三路 t 状态窗口，t 步长由 builder 的
            ``tsize`` 定义。
        horizon_dataset: 完整 horizon 数据，结构为
            ``(horizon_states, relative_states, trend_states, prices, depthprices)``，
            即 ``HorizonDataset``。其中 ``horizon_states`` 用于 frozen decoder
            和环境模拟，``prices/depthprices`` 用于交易执行和 reward 计算。
        demonstration_horizon_label_dataset: Phase I VQ 导出的 horizon-level
            label dataset，结构为 ``(sample_ids, code_labels)``，即
            ``DemonstrationHorizonLabelDataset``。``sample_ids`` 来自
            ``HorizonTrainLabelRow.sample_id``，``code_labels`` 来自
            ``HorizonTrainLabelRow.code_label``，用于 imitation loss 和诊断。
    """

    visible_states: VisibleStatesDataset
    horizon_dataset: HorizonDataset
    demonstration_horizon_label_dataset: DemonstrationHorizonLabelDataset


class Phase2SelectionDatasetBuilder:
    """从 horizon dataset 和 Phase I label 表构建 Phase II selector dataset.

    输入来源:
        - ``HorizonDataset`` 来自数据准备阶段或 dataset cache。
        - ``label_table`` 来自 Phase I ``HorizonTrainLabelBuilder`` 的离线导出。

    输出:
        ``Phase2SelectionDataset`` 或 PyTorch ``TensorDataset``。
    """

    REQUIRED_LABEL_COLUMNS = ("sample_id", "code_label")

    def __init__(self, *, tsize: TSize = 1) -> None:
        """初始化 Phase II dataset builder。

        输入:
            tsize: selector 可见状态的 t 步长。previous 三路状态使用上一分片
                完整状态序列，current 三路状态使用当前分片前 ``tsize`` 个状态。

        输出:
            无返回值。builder 不持有 Phase I model，也不持有可变训练状态。
        """
        if tsize <= 0:
            raise ValueError("tsize must be greater than 0")
        self.tsize = int(tsize)

    def build_from_horizon_and_labels(
        self,
        horizon_dataset: HorizonDataset,
        label_table: pl.DataFrame,
    ) -> Phase2SelectionDataset:
        """从 horizon 数据和 Phase I 导出的 label 表生成 selector dataset。

        输入:
            horizon_dataset: 五元组
                ``(horizon_states, relative_states, trend_states, prices, depthprices)``。
                ``horizon_states`` 是完整 horizon 状态；
                ``prices`` 和 ``depthprices`` 供环境模拟与 reward 计算使用。
            label_table: Phase I 导出的 label 表，至少包含 ``sample_id`` 和
                ``code_label``。本方法会按 ``sample_id`` 排序后生成
                ``DemonstrationHorizonLabelDataset``。

        输出:
            ``Phase2SelectionDataset``。其中 ``visible_states`` 是六路可见状态；
            ``sample_ids`` 是当前分片
            在原始 horizon dataset 中的样本行号，即 ``1..sample_count-1``；
            ``code_labels[i]`` 来自 ``label_table`` 中
            ``sample_id == sample_ids[i]`` 的 ``code_label``。

        异常:
            当 horizon 数据形状不合法、label 表缺列、sample_id 不完整或重复时
            抛出 ``ValueError``。
        """

        (
            horizon_states,
            relative_states,
            trend_states,
            prices,
            depthprices,
        ) = _unpack_horizon_dataset(horizon_dataset)
        sample_count = int(horizon_states.shape[0])

        self.validate_horizon_dataset(horizon_dataset)
        full_demonstration_horizon_label_dataset = (
            self.extract_demonstration_horizon_label_dataset(
                label_table=label_table,
                sample_count=sample_count,
            )
        )
        visible_states = self.build_visible_states(
            horizon_states,
            relative_states,
            trend_states,
        )
        sample_ids, code_labels = full_demonstration_horizon_label_dataset
        current_horizon_dataset = (
            horizon_states[1:].copy(),
            relative_states[1:].copy(),
            trend_states[1:].copy(),
            prices[1:].copy(),
            None if depthprices is None else depthprices[1:].copy(),
        )
        demonstration_horizon_label_dataset = (
            sample_ids[1:].copy(),
            code_labels[1:].copy(),
        )

        dataset = Phase2SelectionDataset(
            visible_states=visible_states,
            horizon_dataset=current_horizon_dataset,
            demonstration_horizon_label_dataset=demonstration_horizon_label_dataset,
        )
        self.validate_no_future_leakage(dataset)
        return dataset

    def validate_horizon_dataset(self, horizon_dataset: HorizonDataset) -> None:
        """检查 horizon dataset 的基础形状契约。

        输入:
            horizon_dataset: 待校验的 ``HorizonDataset``。

        输出:
            无返回值。校验通过即静默返回。

        异常:
            当 ``states/relative_states/trend_states/prices/depthprices`` 不是
            三维数组，或 sample/horizon 维度不一致时抛出 ``ValueError``。
        """

        (
            horizon_states,
            relative_states,
            trend_states,
            prices,
            depthprices,
        ) = _unpack_horizon_dataset(horizon_dataset)
        if horizon_states.ndim != 3:
            raise ValueError(
                "horizon states must have shape [sample, horizon, feature_dim]"
            )
        if relative_states.ndim != 3:
            raise ValueError(
                "relative_states must have shape "
                "[sample, horizon, relative_feature_dim]"
            )
        if trend_states.ndim != 3:
            raise ValueError(
                "trend_states must have shape [sample, horizon, trend_feature_dim]"
            )
        if prices.ndim != 3:
            raise ValueError("prices must have shape [sample, horizon, price_dim]")
        if (
            relative_states.shape[:2] != horizon_states.shape[:2]
            or trend_states.shape[:2] != horizon_states.shape[:2]
            or prices.shape[:2] != horizon_states.shape[:2]
        ):
            raise ValueError(
                "horizon dataset arrays must have the same sample/horizon shape"
            )
        if depthprices is not None:
            if depthprices.ndim != 3:
                raise ValueError(
                    "depthprices must have shape [sample, horizon, depth_dim]"
                )
            if horizon_states.shape[:2] != depthprices.shape[:2]:
                raise ValueError(
                    "horizon states and depthprices must have the same "
                    "sample/horizon shape"
                )

    def extract_demonstration_horizon_label_dataset(
        self,
        label_table: pl.DataFrame,
        sample_count: int,
    ) -> DemonstrationHorizonLabelDataset:
        """从 label 表读取 ``sample_id/code_label``，并按 ``sample_id`` 排序。

        输入:
            label_table: Phase I 离线 label 表，至少包含 ``sample_id`` 和
                ``code_label``。
            sample_count: horizon dataset 的样本数，用于校验 label 完整性。

        输出:
            ``DemonstrationHorizonLabelDataset``，即 ``(sample_ids, code_labels)``。
            两个数组形状均为 ``[sample_count]``，dtype 为 ``int64``。
        """

        self.validate_label_alignment(
            label_table=label_table,
            sample_count=sample_count,
        )
        sorted_labels = label_table.sort("sample_id").select("sample_id", "code_label")
        sample_ids = np.array(
            sorted_labels.get_column("sample_id").to_numpy(),
            dtype=np.int64,
            copy=True,
        )
        code_labels = np.array(
            sorted_labels.get_column("code_label").to_numpy(),
            dtype=np.int64,
            copy=True,
        )
        return sample_ids, code_labels

    def validate_label_alignment(
        self,
        label_table: pl.DataFrame,
        sample_count: int,
    ) -> None:
        """检查 label 表 sample_id 是否完整、唯一，并与 horizon 样本数一致。

        输入:
            label_table: Phase I 离线 label 表。
            sample_count: horizon dataset 的样本数。

        输出:
            无返回值。校验通过即静默返回。

        异常:
            缺少必要列、行数不匹配、``sample_id`` 重复、``sample_id`` 不是完整
            零基连续区间，或 ``code_label`` 存在 null 时抛出 ``ValueError``。
        """

        missing_columns = [
            column
            for column in self.REQUIRED_LABEL_COLUMNS
            if column not in label_table.columns
        ]
        if missing_columns:
            raise ValueError(
                "label_table is missing required columns: "
                f"{', '.join(missing_columns)}"
            )
        if label_table.height != sample_count:
            raise ValueError(
                "label_table row count must match horizon sample count: "
                f"{label_table.height} != {sample_count}"
            )

        sample_ids = label_table.get_column("sample_id")
        if sample_ids.n_unique() != sample_count:
            raise ValueError("label_table sample_id values must be unique")

        expected_ids = pl.Series("sample_id", range(sample_count), dtype=sample_ids.dtype)
        actual_ids = sample_ids.sort()
        if not actual_ids.equals(expected_ids):
            raise ValueError(
                "label_table sample_id must be a complete zero-based range "
                f"[0, {sample_count})"
            )

        if label_table.get_column("code_label").null_count() > 0:
            raise ValueError("label_table code_label must not contain null values")

    def build_visible_states(
        self,
        horizon_states: np.ndarray,
        relative_states: np.ndarray,
        trend_states: np.ndarray,
    ) -> VisibleStatesDataset:
        """提取 selector 可见状态。

        输入:
            horizon_states: 完整 horizon 状态，形状
                ``[sample, horizon, feature_dim]``。
            relative_states: 完整 horizon 相对状态，形状
                ``[sample, horizon, relative_feature_dim]``。
            trend_states: 完整 horizon 趋势状态，形状
                ``[sample, horizon, trend_feature_dim]``。

        输出:
            ``VisibleStatesDataset``，即 previous/current 各三路状态。
            ``previous_t_states`` 形状为 ``[sample - 1, horizon, feature_dim]``；
            ``current_t_states`` 形状为 ``[sample - 1, tsize, feature_dim]``。

        说明:
            selector 在线推理只能读取上一分片完整状态序列和当前分片 t 状态窗口，
            不能读取当前分片 t 之后的未来状态、价格、teacher action 或 reward。
        """

        self._validate_visible_source_array(
            "horizon_states",
            horizon_states,
            "feature_dim",
        )
        self._validate_visible_source_array(
            "relative_states",
            relative_states,
            "relative_feature_dim",
        )
        self._validate_visible_source_array(
            "trend_states",
            trend_states,
            "trend_feature_dim",
        )
        if relative_states.shape[:2] != horizon_states.shape[:2]:
            raise ValueError(
                "relative_states and horizon_states must share [sample, horizon]"
            )
        if trend_states.shape[:2] != horizon_states.shape[:2]:
            raise ValueError(
                "trend_states and horizon_states must share [sample, horizon]"
            )
        if horizon_states.shape[1] < self.tsize:
            raise ValueError(
                "horizon_states horizon length must be greater than or equal to tsize"
            )
        if horizon_states.shape[0] < 2:
            raise ValueError("horizon_states must contain at least two samples")
        previous_t_states = horizon_states[:-1].copy()
        previous_t_relative_states = relative_states[:-1].copy()
        previous_t_trend_states = trend_states[:-1].copy()
        current_t_states = horizon_states[1:, : self.tsize, :].copy()
        current_t_relative_states = relative_states[1:, : self.tsize, :].copy()
        current_t_trend_states = trend_states[1:, : self.tsize, :].copy()
        return (
            previous_t_states,
            previous_t_relative_states,
            previous_t_trend_states,
            current_t_states,
            current_t_relative_states,
            current_t_trend_states,
        )

    def validate_no_future_leakage(
        self,
        dataset: Phase2SelectionDataset,
    ) -> None:
        """检查 visible_states 来源，防止未来 states/prices 混入 observation。

        输入:
            dataset: 待校验的 ``Phase2SelectionDataset``。

        输出:
            无返回值。校验通过即静默返回。

        异常:
            当 ``visible_states`` 形状或来源不符合 ``VisibleStatesDataset``
            契约，或 ``sample_ids/code_labels`` 无法一一对应时抛出
            ``ValueError``。
        """

        horizon_states, relative_states, trend_states, _, _ = _unpack_horizon_dataset(
            dataset.horizon_dataset
        )
        (
            previous_t_states,
            previous_t_relative_states,
            previous_t_trend_states,
            current_t_states,
            current_t_relative_states,
            current_t_trend_states,
        ) = dataset.visible_states
        self._validate_previous_visible_state(
            "previous_t_states",
            previous_t_states,
            horizon_states,
        )
        self._validate_previous_visible_state(
            "previous_t_relative_states",
            previous_t_relative_states,
            relative_states,
        )
        self._validate_previous_visible_state(
            "previous_t_trend_states",
            previous_t_trend_states,
            trend_states,
        )
        self._validate_current_visible_state(
            "current_t_states",
            current_t_states,
            horizon_states,
        )
        self._validate_current_visible_state(
            "current_t_relative_states",
            current_t_relative_states,
            relative_states,
        )
        self._validate_current_visible_state(
            "current_t_trend_states",
            current_t_trend_states,
            trend_states,
        )
        sample_ids, code_labels = dataset.demonstration_horizon_label_dataset
        if code_labels.shape != sample_ids.shape:
            raise ValueError("code_labels and sample_ids must have the same shape")
        if sample_ids.shape != (current_t_states.shape[0],):
            raise ValueError("sample_ids must match horizon sample count")

    def to_tensor_dataset(
        self,
        dataset: Phase2SelectionDataset,
    ) -> TensorDataset:
        """把 numpy dataset 转成 PyTorch ``TensorDataset``。

        输入:
            dataset: ``Phase2SelectionDataset``。各 numpy 数组应已通过
                ``build_from_horizon_and_labels`` 的契约校验。

        输出:
            ``torch.utils.data.TensorDataset``，固定包含 13 列。

        返回列顺序:
            ``previous_t_states, previous_t_relative_states, previous_t_trend_states,
            current_t_states, current_t_relative_states, current_t_trend_states,
            horizon_states, relative_states, trend_states, prices, depthprices,
            assigned_labels, sample_ids``。

        dtype:
            前 11 个状态/市场数据 tensor 为 ``torch.float32``；
            ``assigned_labels/sample_ids`` 为 ``torch.long``。

        如果 ``dataset.horizon_dataset`` 中的 ``depthprices`` 为 ``None``，
        会生成形状 ``[sample, horizon, 0]`` 的空 float tensor，保持
        ``TensorDataset`` 的固定列数。
        """

        (
            previous_t_states_np,
            previous_t_relative_states_np,
            previous_t_trend_states_np,
            current_t_states_np,
            current_t_relative_states_np,
            current_t_trend_states_np,
        ) = dataset.visible_states
        previous_t_states = torch.as_tensor(
            previous_t_states_np,
            dtype=torch.float32,
        )
        previous_t_relative_states = torch.as_tensor(
            previous_t_relative_states_np,
            dtype=torch.float32,
        )
        previous_t_trend_states = torch.as_tensor(
            previous_t_trend_states_np,
            dtype=torch.float32,
        )
        current_t_states = torch.as_tensor(current_t_states_np, dtype=torch.float32)
        current_t_relative_states = torch.as_tensor(
            current_t_relative_states_np,
            dtype=torch.float32,
        )
        current_t_trend_states = torch.as_tensor(
            current_t_trend_states_np,
            dtype=torch.float32,
        )
        (
            horizon_states_np,
            horizon_relative_states_np,
            horizon_trend_states_np,
            prices_np,
            depthprices_np,
        ) = _unpack_horizon_dataset(dataset.horizon_dataset)
        horizon_states = torch.as_tensor(horizon_states_np, dtype=torch.float32)
        horizon_relative_states = torch.as_tensor(
            horizon_relative_states_np,
            dtype=torch.float32,
        )
        horizon_trend_states = torch.as_tensor(
            horizon_trend_states_np,
            dtype=torch.float32,
        )
        prices = torch.as_tensor(prices_np, dtype=torch.float32)
        if depthprices_np is None:
            depthprices = torch.empty(
                (
                    horizon_states_np.shape[0],
                    horizon_states_np.shape[1],
                    0,
                ),
                dtype=torch.float32,
            )
        else:
            depthprices = torch.as_tensor(depthprices_np, dtype=torch.float32)
        sample_ids_np, code_labels_np = dataset.demonstration_horizon_label_dataset
        assigned_labels = torch.as_tensor(code_labels_np, dtype=torch.long)
        sample_ids = torch.as_tensor(sample_ids_np, dtype=torch.long)
        return TensorDataset(
            previous_t_states,
            previous_t_relative_states,
            previous_t_trend_states,
            current_t_states,
            current_t_relative_states,
            current_t_trend_states,
            horizon_states,
            horizon_relative_states,
            horizon_trend_states,
            prices,
            depthprices,
            assigned_labels,
            sample_ids,
        )

    def _validate_visible_source_array(
        self,
        name: str,
        values: np.ndarray,
        feature_name: str,
    ) -> None:
        """校验构造 visible states 的单路源数组。"""

        if values.ndim != 3:
            raise ValueError(
                f"{name} must have shape [sample, horizon, {feature_name}]"
            )

    def _validate_previous_visible_state(
        self,
        name: str,
        values: np.ndarray,
        current_horizon_values: np.ndarray,
    ) -> None:
        """校验上一分片 visible state 的形状。"""

        if values.ndim != 3:
            raise ValueError(f"{name} must be a 3D array")
        if values.shape[0] != current_horizon_values.shape[0]:
            raise ValueError(f"{name} must match current horizon sample count")
        if values.shape[1:] != current_horizon_values.shape[1:]:
            raise ValueError(f"{name} must match current horizon sequence shape")

    def _validate_current_visible_state(
        self,
        name: str,
        values: np.ndarray,
        current_horizon_values: np.ndarray,
    ) -> None:
        """校验当前分片 visible state 的形状和来源。"""

        expected_shape = (
            current_horizon_values.shape[0],
            self.tsize,
            current_horizon_values.shape[2],
        )
        if values.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if not np.array_equal(values, current_horizon_values[:, : self.tsize, :]):
            raise ValueError(f"{name} must match current horizon t states")


def _unpack_horizon_dataset(
    horizon_dataset: HorizonDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """返回标准五元组，并兼容旧三元组产物。"""

    if len(horizon_dataset) == 5:
        return horizon_dataset
    raise ValueError(
        "horizon_dataset must be "
        "(states, relative_states, trend_states, prices, depthprices)"
    )


__all__ = [
    "Phase2SelectionDataset",
    "Phase2SelectionDatasetBuilder",
]
