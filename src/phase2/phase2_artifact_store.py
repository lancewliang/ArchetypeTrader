"""Phase II 训练产物 store 骨架。

本文件只定义 Phase II artifact store 的职责边界、方法签名和路径语义注释。
不实现目录创建、文件读写、序列化、反序列化、checkpoint 保存或指标保存逻辑。
"""

from __future__ import annotations

from pathlib import Path

from ..store.artifact_store import DataFileStore
from .checkpoint.phase2_checkpoint import Phase2Checkpoint
from .metrics.phase2_metric_results import Phase2ValidationResult
from .phase2_selection_data_schema import Phase2SelectionDataset


class Phase2ArtifactStore(DataFileStore):
    """Phase II 产物路径与读写入口骨架。

    适用场景:
        由 ``Phase2MainFlow`` 持有，用于集中管理 Phase II selector 训练期间的
        dataset cache、checkpoint、metrics 和 report 路径。

    设计边界:
        本类复用 ``DataFileStore`` 的基础数据读写能力，但 Phase II 自有产物
        必须放在 ``artifacts/{pair}/{train_batch_id}/phase2/`` 命名空间下。
        本骨架不执行真实 I/O，只固定后续实现应提供的接口。
    """

    def __init__(
        self,
        pair: str,
        train_batch_id: str,
        artifacts_root: Path,
    ) -> None:
        """初始化 Phase II artifact store。

        适用场景:
            在 ``Phase2MainFlow`` 初始化时创建，绑定一次 Phase II 训练的交易
            标的、训练批次和产物根目录。

        参数:
            pair: 交易标的或数据域名称，例如 ``BTCUSDT``。
            train_batch_id: 当前训练批次 ID。
            artifacts_root: 全阶段产物根目录。

        字段解释:
            后续实现应将 ``train_batch_id`` 映射到 ``DataFileStore.batchid``，
            并把 Phase II 的标准路径写入 ``artifact_paths``。
        """

        pass

    def initialize_phase2_artifact_dirs(self) -> None:
        """初始化 Phase II 标准产物目录骨架。

        适用场景:
            在训练开始前调用，规划 checkpoint、dataset cache、metrics 和 report
            等路径，供训练、评估、checkpoint selector 和 report 复用。

        目录语义:
            ``output_dir``:
                Phase II 根目录，推荐为 ``artifacts/{pair}/{batch_id}/phase2``。
            ``dataset_cache``:
                保存按 split 生成的 ``Phase2SelectionDataset`` cache。
            ``checkpoints``:
                保存周期性 epoch checkpoint 和 best checkpoint。
            ``metrics``:
                保存 train/validation/test selection metrics JSON。
            ``reports``:
                保存 Phase II selection report 的 JSON/HTML 输出。
            ``diagnostics``:
                保存 code usage、label consistency、reward distribution 等诊断产物。

        设计边界:
            本方法正式实现时只负责目录和路径契约，不负责训练、不计算指标、
            不生成空 checkpoint 或 report。
        """

        pass

    def save_phase2_checkpoint(
        self,
        checkpoint: Phase2Checkpoint,
        *,
        checkpoint_name: str | None = None,
    ) -> Path:
        """保存 Phase II model checkpoint 的骨架入口。

        功能说明:
            后续实现应将 ``Phase2Checkpoint`` 转换为 torch/save 友好的 payload，
            并保存到 Phase II model checkpoint 目录。该 checkpoint 只包含
            Q-network、optimizer 和恢复训练所需配置，不包含 validation metrics。

        适用场景:
            ``Phase2DoubleDqnTrainer`` 在 checkpoint interval 或训练结束时调用，
            保存某个 epoch 的 selector 模型状态。

        参数:
            checkpoint: 待保存的 Phase II model checkpoint payload。
            checkpoint_name: 可选文件名或稳定 ID；为空时后续实现可按 epoch 生成。

        返回:
            保存后的 checkpoint 路径。

        设计边界:
            当前方法只固定接口，不创建目录、不序列化、不写文件。
        """

        raise NotImplementedError("Phase2 checkpoint saving is not implemented yet.")

    def load_phase2_checkpoint(
        self,
        *,
        checkpoint_path: Path | None = None,
        epoch: int | None = None,
        best: bool = False,
    ) -> Phase2Checkpoint:
        """读取 Phase II model checkpoint 的骨架入口。

        功能说明:
            后续实现应从指定路径、epoch 或 best 标记解析 Phase II model checkpoint
            文件，并恢复为 ``Phase2Checkpoint`` 强类型对象。

        适用场景:
            恢复训练、加载 best selector 做 test evaluation，或后续阶段加载
            Phase II selector 时调用。

        参数:
            checkpoint_path: 显式 checkpoint 文件路径。
            epoch: 按训练 epoch 读取 checkpoint。
            best: 为 True 时读取 best model checkpoint。

        返回:
            ``Phase2Checkpoint``。

        设计边界:
            当前方法只固定接口，不读取文件、不反序列化、不校验路径。
        """

        raise NotImplementedError("Phase2 checkpoint loading is not implemented yet.")

    def save_phase2_validation_result(
        self,
        validation_result: Phase2ValidationResult,
        *,
        split_name: str = "validation",
        epoch: int | None = None,
    ) -> Path:
        """保存 Phase II validation result 的骨架入口。

        功能说明:
            后续实现应把 ``Phase2ValidationResult.metrics`` 和
            ``Phase2ValidationResult.diagnostics`` 保存为 JSON 友好的评估结果文件。
            它只保存评估结果，不保存模型权重。

        适用场景:
            ``Phase2Evaluator.evaluate()`` 产出 validation/test 结果后调用，供
            report、checkpoint selector 和审计流程复用。

        参数:
            validation_result: 待保存的 validation/test 评估结果。
            split_name: 数据 split 名称，例如 ``"validation"`` 或 ``"test"``。
            epoch: 结果对应的训练 epoch；离线 test 或未知 epoch 时可为 None。

        返回:
            保存后的 validation result 路径。

        设计边界:
            当前方法只固定接口，不创建目录、不转 dict、不写 JSON。
        """

        raise NotImplementedError("Phase2 validation result saving is not implemented yet.")

    def load_phase2_validation_result(
        self,
        *,
        result_path: Path | None = None,
        split_name: str = "validation",
        epoch: int | None = None,
        best: bool = False,
    ) -> Phase2ValidationResult:
        """读取 Phase II validation result 的骨架入口。

        功能说明:
            后续实现应从指定路径、split/epoch 或 best 标记读取评估结果文件，并恢复
            为 ``Phase2ValidationResult`` 强类型对象。

        适用场景:
            checkpoint selector 读取历史 validation 结果做排序，report 读取
            validation/test 结果构建展示上下文，或主流程恢复已完成评估结果时调用。

        参数:
            result_path: 显式 validation result 文件路径。
            split_name: 数据 split 名称。
            epoch: 结果对应的训练 epoch。
            best: 为 True 时读取 best checkpoint 对应的 validation result。

        返回:
            ``Phase2ValidationResult``。

        设计边界:
            当前方法只固定接口，不读取文件、不解析 JSON、不应用选择逻辑。
        """

        raise NotImplementedError("Phase2 validation result loading is not implemented yet.")
