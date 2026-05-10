"""生成并保存 Phase I 输出的 horizon-level archetype 训练标签。

论文背景:
    ArchetypeTrader 的 Phase I 是 Archetype Discovery。它先从训练数据中切出
    固定长度 ``h`` 的 horizon，再用 single-trade DP planner 为每个 horizon
    生成 demonstration trajectory:

        ``tau = (s_demo, a_demo, r_demo)``

    其中 ``s_demo`` 是 horizon 内的市场状态序列，``a_demo`` 是 DP teacher
    给出的 short/flat/long 动作序列，``r_demo`` 是执行该动作序列得到的逐步
    reward。Phase I 的 VQ encoder-decoder 训练完成后，encoder 会把整条
    trajectory 编码为连续 latent ``z_e``，VQ codebook 再把 ``z_e`` 分配给
    最近的离散 archetype code。

本模块用途:
    本模块负责把训练好的 Phase I VQ 模型应用到已固化的
    ``TrajectoryDataset`` 上，为每个 fixed horizon 生成一个离散
    ``code_label``。这个 ``code_label`` 就是论文 Phase II selector 目标中的
    ``hat{a}^{sel}``，用于训练 horizon-level RL selector 时提供监督标签或
    KL 一致性约束。

使用场景:
    1. Phase I 训练完成并选出 best checkpoint 后，由
       ``Phase1MainFlow.export_horizon_labels`` 调用。
    2. Phase II 训练前，离线读取该 label 文件，把每个 horizon 起点状态
       映射到对应 archetype 标签。
    3. 离线诊断 codebook 使用率、latent 分布、collapse 风险时，可读取本模块
       写出的 ``latent_*`` 和 ``demo_return`` 辅助分析。

设计边界:
    本模块只做离线 label 生成和保存，不重新运行 DP planner，不训练模型，也不
    决定 best checkpoint。调用方必须传入已经加载好 best checkpoint 权重的
    ``ArchetypeVQModel``。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from ..model.data_types import TrajectoryDataset
from ..model.tensor_data_types import (
    build_trajectory_tensor_dataset,
    move_trajectory_batch_to_device,
)
from ..model.vq_archetype import ArchetypeVQModel
from ..store.artifact_store import DataFileStore


@dataclass(frozen=True)
class HorizonTrainLabelBuilderConfig:
    """离线 horizon label 生成配置。

    参数:
        horizon: 每个样本的固定窗口长度 ``h``。论文实验默认 ``h=72``。
            这里用于恢复 ``sample_id`` 到原始时间序列位置的近似映射:
            ``horizon_start_idx = sample_id * horizon``。
        batch_size: 批量编码 trajectory 时的 batch size。只影响离线生成速度和
            显存/内存占用，不改变 label 结果。
        device: 执行 encoder/codebook 推理的设备。通常与 Phase I 主流程使用的
            device 保持一致。
    """

    horizon: int = 72
    batch_size: int = 256
    device: str | torch.device = "cpu"


class HorizonTrainLabelBuilder:
    """使用训练好的 Phase I VQ encoder 生成 Phase II 训练标签。

    论文中的算法含义:
        Phase I 已经通过 VQ bottleneck 学到有限个 trading archetypes:
        ``epsilon = {e_0, e_1, ..., e_{K-1}}``。对于每条 DP demonstration
        trajectory ``tau_i``:

        1. encoder 计算连续表示 ``z_e = q_theta_e(tau_i)``。
        2. VectorQuantizer 在 codebook 中寻找距离 ``z_e`` 最近的 code。
        3. 最近 code 的 index ``k`` 被记为该 horizon 的 ``code_label``。

        因此输出 label 不是人工规则，也不是未来在线推理结果，而是 Phase I
        从 DP demonstrations 中自监督发现出来的离散 archetype 分配。

    使用方式:
        ``Phase1MainFlow.build_components`` 中初始化 builder，使其绑定
        ``DataFileStore`` 和生成配置；``export_horizon_labels`` 中再传入已加载
        best checkpoint 权重的 ``model``、对应 split 的 ``trajectory_dataset``
        和 ``split_name`` 调用 ``build_and_store``。
    """

    def __init__(
        self,
        *,
        data_store: DataFileStore,
        config: HorizonTrainLabelBuilderConfig | None = None,
    ) -> None:
        self.data_store = data_store
        self.config = config or HorizonTrainLabelBuilderConfig()
        self.device = torch.device(self.config.device)

    def build_and_store(
        self,
        *,
        model: ArchetypeVQModel,
        trajectory_dataset: TrajectoryDataset,
        split_name: str = "train",
        output_path: str | Path | None = None,
    ) -> pl.DataFrame:
        """构建 horizon-level archetype labels，并通过 ``DataFileStore`` 保存。

        输入:
            model: 已加载 best checkpoint 权重的 ``ArchetypeVQModel``。本方法只
                调用 ``model.encode``，不会执行 decoder，也不会更新参数。
            trajectory_dataset: DP planner 已经生成并固化的 demonstration
                trajectories，数据形式为 ``[(s_demo, a_demo, r_demo), ...]``。
            split_name: 当前数据 split 名称，例如 ``train``、``val``、``test``。
                该字段会写入 label 表，方便后续阶段审计来源。
            output_path: 可选输出路径。不传时由 ``DataFileStore`` 的 Phase I
                标准产物路径决定。

        论文算法步骤:
            1. 保持 ``trajectory_dataset`` 原始顺序，禁止 shuffle。这样
               ``sample_id`` 可以稳定对应到第几个 fixed horizon chunk。
            2. 将 numpy trajectory 转成 PyTorch ``TensorDataset``，批量送入
               与模型相同的 device。
            3. 在 ``torch.no_grad`` 和 ``model.eval`` 下调用
               ``model.encode((states, actions, rewards))``。
            4. ``model.encode`` 内部先用 trajectory encoder 生成 ``z_e``，
               再通过 VQ codebook 取最近 code，返回 ``code_labels``。
            5. 为每个 horizon 写出一行 label:
               ``sample_id``、``horizon_start_idx``、``horizon_end_idx``、
               ``code_label``、``demo_return`` 和 ``latent_*``。
            6. 调用 ``DataFileStore.save_phase1_horizon_labels`` 保存为 feather、
               parquet 或 csv 文件。

        输出:
            返回写出的 ``polars.DataFrame``。最核心字段是:
                ``sample_id``: horizon 样本序号。
                ``code_label``: Phase I VQ codebook 分配的 archetype id。
                ``demo_return``: DP demonstration 在该 horizon 内的累计 reward。

        使用场景:
            Phase II selector 训练时，``code_label`` 可作为监督目标或 KL penalty
            的 reference label；诊断时，``latent_*`` 可用于检查 codebook 是否
            collapse、不同 archetype 是否覆盖了不同收益/行为模式。
        """

        if not trajectory_dataset:
            raise ValueError("trajectory_dataset must not be empty")

        tensor_dataset = build_trajectory_tensor_dataset(trajectory_dataset)
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # 离线生成 label 时必须关闭 dropout/训练态行为；结束后恢复调用前状态。
        previous_training = model.training
        model.to(self.device)
        model.eval()

        rows: list[dict[str, object]] = []
        sample_offset = 0
        try:
            with torch.no_grad():
                for batch in dataloader:
                    states, actions, rewards = move_trajectory_batch_to_device(
                        batch,
                        self.device,
                    )
                    # encode 对应论文中的 q_theta_e + VQ nearest-code assignment。
                    code_labels, latent = model.encode((states, actions, rewards))
                    code_labels_cpu = code_labels.detach().cpu()
                    latent_cpu = latent.detach().cpu()
                    rewards_cpu = rewards.detach().cpu()

                    batch_size = int(code_labels_cpu.shape[0])
                    for batch_index in range(batch_size):
                        sample_id = sample_offset + batch_index
                        row: dict[str, object] = {
                            "split": split_name,
                            "sample_id": sample_id,
                            # HorizonBuilder 当前按固定长度连续切块，因此可用
                            # sample_id 和 horizon 恢复该样本覆盖的相对 bar 范围。
                            "horizon_start_idx": sample_id * self.config.horizon,
                            "horizon_end_idx": (sample_id + 1) * self.config.horizon - 1,
                            "code_label": int(code_labels_cpu[batch_index].item()),
                            "demo_return": float(rewards_cpu[batch_index].sum().item()),
                        }
                        for latent_index, value in enumerate(
                            latent_cpu[batch_index].tolist()
                        ):
                            row[f"latent_{latent_index}"] = float(value)
                        rows.append(row)
                    sample_offset += batch_size
        finally:
            if previous_training:
                model.train()

        labels = pl.DataFrame(rows)
        path = (
            Path(output_path)
            if output_path is not None
            else self._default_output_path(split_name)
        )
        self.data_store.save_phase1_horizon_labels(labels, path)
        return labels

    def _default_output_path(self, split_name: str) -> Path:
        """根据 Phase I 产物目录约定生成默认 label 输出路径。"""

        if split_name == "train":
            train_labels = self.data_store.artifact_paths.get("horizon_train_labels")
            if train_labels is not None:
                return Path(train_labels)
        label_dir = self.data_store.artifact_paths.get("labels")
        if label_dir is not None:
            return Path(label_dir) / f"sampled_horizon_labels_{split_name}.feather"
        output_dir = self.data_store.artifact_paths.get("output_dir")
        if output_dir is None:
            raise ValueError("data_store must be initialized with phase1 artifact paths")
        return Path(output_dir) / f"sampled_horizon_labels_{split_name}.feather"


def build_and_store(
    *,
    model: ArchetypeVQModel,
    data_store: DataFileStore,
    trajectory_dataset: TrajectoryDataset,
    split_name: str = "train",
    horizon: int = 72,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    output_path: str | Path | None = None,
) -> pl.DataFrame:
    """模块级便捷入口，用于一次性构建并保存 horizon labels。

    使用场景:
        当调用方不需要长期持有 ``HorizonTrainLabelBuilder`` 实例时，可以直接
        调用这个函数。Phase I 主流程中因为 builder 会在 ``build_components``
        阶段初始化并复用，所以更适合直接使用类方法。
    """

    builder = HorizonTrainLabelBuilder(
        data_store=data_store,
        config=HorizonTrainLabelBuilderConfig(
            horizon=horizon,
            batch_size=batch_size,
            device=device,
        ),
    )
    return builder.build_and_store(
        model=model,
        trajectory_dataset=trajectory_dataset,
        split_name=split_name,
        output_path=output_path,
    )


__all__ = [
    "HorizonTrainLabelBuilder",
    "HorizonTrainLabelBuilderConfig",
    "build_and_store",
]
