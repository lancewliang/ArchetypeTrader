"""数据准备产出物读写类的接口骨架。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..model.data_types import ArtifactPaths, HorizonDataset, TrajectoryDataset
from ..model.vq_archetype import (
    ArchetypeActionDecoder,
    ArchetypeTrajectoryEncoder,
    VectorQuantizer,
)


class DataFileStore:
    """负责计算产出物路径，并读写各阶段中间产物。

    为什么需要这个类:
        ``horizon_dataset`` 和 ``trajectory_dataset`` 都是需要持久化的产出物。
        将路径规划、保存和读取逻辑从 ``DataPreparer`` 中拆出，可以让
        ``DataPreparer`` 专注于流程编排，避免 I/O 细节散落在训练、测试、
        校验三个入口里。Phase I 训练产物也通过同一个 store 管理，避免
        checkpoint、模型导出、label 和报告各自拼路径、各自决定 I/O 语义。
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

    def save_phase1_checkpoint(
        self,
        checkpoint: Any,
        output_path: str | Path,
    ) -> None:
        """保存 Phase I checkpoint。

        参数:
            checkpoint: checkpoint 内容。后续实现通常是包含 model/optimizer/
                scheduler/epoch/metrics/config hash 的字典。
            output_path: checkpoint 保存路径，例如 ``last.pt`` 或 ``best.pt``。

        方法作用:
            统一封装 Phase I checkpoint 持久化。正式实现应使用原子写入，并
            记录 sha256 供审计。
        """

        ...

    def load_phase1_checkpoint(
        self,
        input_path: str | Path,
    ) -> Any:
        """读取 Phase I checkpoint。

        参数:
            input_path: checkpoint 文件路径。

        输出:
            返回 checkpoint 内容，供恢复训练、best 选择和模型导出使用。
        """

        ...

    def save_phase1_encoder(
        self,
        encoder: ArchetypeTrajectoryEncoder,
        output_path: str | Path,
    ) -> None:
        """保存 Phase I encoder 导出产物。

        参数:
            encoder: 训练完成后从 best checkpoint 导出的 trajectory encoder。
            output_path: encoder 保存路径，例如 ``encoder.pt``。

        方法作用:
            将 Phase II 离线 label 生成需要复用的 encoder 以统一方式写出。
        """

        ...

    def load_phase1_encoder(
        self,
        input_path: str | Path,
    ) -> ArchetypeTrajectoryEncoder:
        """读取 Phase I encoder 导出产物。

        参数:
            input_path: ``encoder.pt`` 路径。

        输出:
            返回 trajectory encoder，供 Phase II label 生成或离线诊断加载。
        """

        ...

    def save_phase1_decoder(
        self,
        decoder: ArchetypeActionDecoder,
        output_path: str | Path,
    ) -> None:
        """保存 Phase I decoder 导出产物。

        参数:
            decoder: 训练完成后从 best checkpoint 导出的 causal action decoder。
            output_path: decoder 保存路径，例如 ``decoder.pt``。

        方法作用:
            将 Phase II/III 在线生成基础动作需要复用的 decoder 以统一方式写出。
        """

        ...

    def load_phase1_decoder(
        self,
        input_path: str | Path,
    ) -> ArchetypeActionDecoder:
        """读取 Phase I decoder 导出产物。

        参数:
            input_path: ``decoder.pt`` 路径。

        输出:
            返回 causal action decoder，供 Phase II/III 或离线诊断加载。
        """

        ...

    def save_phase1_codebook(
        self,
        codebook: VectorQuantizer,
        output_path: str | Path,
    ) -> None:
        """保存 Phase I codebook 导出产物。

        参数:
            codebook: 训练完成后从 best checkpoint 导出的 VQ codebook。
            output_path: codebook 保存路径，例如 ``codebook.pt``。

        方法作用:
            将 Phase II/III 离散 archetype 选择和解码需要复用的 codebook
            以统一方式写出。
        """

        ...

    def load_phase1_codebook(
        self,
        input_path: str | Path,
    ) -> VectorQuantizer:
        """读取 Phase I codebook 导出产物。

        参数:
            input_path: ``codebook.pt`` 路径。

        输出:
            返回 VQ codebook，供 Phase II/III 或离线诊断加载。
        """

        ...

    def save_phase1_horizon_labels(
        self,
        labels: Any,
        output_path: str | Path,
    ) -> None:
        """保存 Phase I horizon-level archetype labels。

        参数:
            labels: horizon label 表。后续实现可使用 polars/pandas DataFrame，
                至少应包含 ``sample_id`` 和 ``code_label``。
            output_path: label 文件路径，例如
                ``sampled_horizon_labels_train.feather``。

        方法作用:
            保存由 best checkpoint 生成的离线 archetype label，供 Phase II/III
            训练和审计复用。
        """

        ...

    def load_phase1_horizon_labels(
        self,
        input_path: str | Path,
    ) -> Any:
        """读取 Phase I horizon-level archetype labels。

        参数:
            input_path: horizon label 文件路径。

        输出:
            返回 label 表，供 Phase II/III 训练、评估或审计使用。
        """

        ...

    def save_phase1_report(
        self,
        report: dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """保存 Phase I 训练报告。

        参数:
            report: 训练报告内容，包含配置摘要、best checkpoint、评估指标、
                guardrail 结果、warning 和 collapse 诊断。
            output_path: 报告保存路径，通常为 ``phase1_report.json``。

        方法作用:
            统一封装 Phase I report 写入。正式实现应在写入前校验 report
            schema，并使用稳定 JSON 格式。
        """

        ...

    def load_phase1_report(
        self,
        input_path: str | Path,
    ) -> dict[str, Any]:
        """读取 Phase I 训练报告。

        参数:
            input_path: ``phase1_report.json`` 路径。

        输出:
            返回训练报告字典，供复查、resume 和后续阶段读取。
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
        input_path: str | Path,
    ) -> HorizonDataset:
        """读取 horizon 中间数据。

        参数:
            input_path: horizon 中间数据产出物路径。

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
        input_path: str | Path,
    ) -> TrajectoryDataset:
        """读取 demonstration trajectory 数据集。

        参数:
            input_path: trajectory 数据集产出物路径。

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
