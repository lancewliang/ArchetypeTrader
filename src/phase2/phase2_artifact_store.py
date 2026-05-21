"""Phase II 训练产物 store 骨架。

本文件只定义 Phase II artifact store 的职责边界、方法签名和路径语义注释。
不实现目录创建、文件读写、序列化、反序列化、checkpoint 保存或指标保存逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..store.artifact_store import DataFileStore
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
