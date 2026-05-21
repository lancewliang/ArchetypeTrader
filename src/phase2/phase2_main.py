"""Phase II archetype selection 主流程骨架。

文件功能说明:
    本文件定义 Phase II 的主流程编排入口。Phase II 的目标是训练一个
    horizon-level selector：输入在线可见状态 ``(previous_t_states,
    current_t_states)``，输出一个 archetype id，并通过冻结的 Phase I decoder
    生成基础动作序列，最后用 horizon-level trading reward 训练 Double DQN。

设计边界:
    - 只负责主流程编排和组件 wiring；
    - 不实现 Q-network forward、decoder 解码、reward 计算、replay 采样或 TD loss；
    - 不重新训练 Phase I，不调用 Phase I encoder 在线生成 label；
    - 不把当前 horizon 的未来状态、价格、teacher action 或 reward 拼入 selector
      observation；
    - checkpoint 文件保存、validation result 保存和 report 渲染委托给各自模块。

使用场景:
    ``scripts/train_phase2.py`` 创建 ``Phase2MainFlow`` 并调用 ``run()``。后续
    Phase III 或离线评估应读取本流程固化的 Phase II 产物，而不是直接复用训练中
    的临时对象。
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import torch

from ..model.vq_archetype import ArchetypeVQModel
from ..phase1.checkpoint import Phase1Checkpoint
from ..phase1.phase1_main import Phase1MainConfig
from ..phase1.phase1_artifact_store import Phase1ArtifactStore
from ..utils import RuntimeUtils
from .checkpoint.phase2_checkpoint import Phase2Checkpoint, Phase2ValidationCheckpoint
from .checkpoint.phase2_checkpoint_selector import (
    Phase2CheckpointSelectionResult,
    Phase2CheckpointSelector,
)
from .evaluators.phase2_evaluator import Phase2Evaluator
from .model.phase2_decoder_policy import FrozenArchetypeDecoderPolicy
from .phase2_artifact_store import Phase2ArtifactStore
from .phase2_config import (
    Phase2DatasetConfig,
    Phase2MainConfig,
    Phase2ModelConfig,
    Phase2RewardConfig,
    Phase2TrainConfig,
)
from .phase2_env import ArchetypeSelectionEnv
from .phase2_selection_dataset import (
    Phase2SelectionDataset,
    Phase2SelectionDatasetBuilder,
)
from .rl import Phase2ReplayBuffer
from .rl.phase2_double_dqn_trainer import Phase2DoubleDqnTrainer
from .model.phase2_q_network import Phase2QNetwork


logger = logging.getLogger("archetype_trader.phase2")


class Phase2FatalError(RuntimeError):
    """Phase II 主流程致命错误。

    输入:
        由 ``Phase2MainFlow.run()`` 捕获底层异常后包装得到。

    输出:
        无返回值；通过抛出异常通知入口脚本或调度系统 Phase II 失败。

    使用场景:
        当 Phase I checkpoint、horizon dataset、Phase I exported labels、训练组件
        或 checkpoint/report 流程出现不可恢复错误时，统一转换为本异常，避免后续
        阶段使用不完整的 selector 产物。
    """


class Phase2MainFlow:
    """Phase II Archetype Selection 主流程编排骨架。

    输入:
        ``Phase2MainConfig`` 和可选的 dataset/train/reward 子配置。

    输出:
        ``run()`` 返回 ``Phase2TrainingResult``；训练过程中的 dataset cache、
        model checkpoint、validation result 和 report 路径由 ``Phase2ArtifactStore``
        统一管理。

    使用场景:
        入口脚本只需要创建本类并调用 ``run()``。本类内部按设计文档串联 Phase I
        产物加载、Phase II dataset 构建、Double DQN 训练、validation checkpoint
        选择和 report 输出。
    """

    def __init__(
        self,
        config: Phase2MainConfig,
        phase1_config: Phase1MainConfig,
        *,
        dataset_config: Phase2DatasetConfig | None = None,
        train_config: Phase2TrainConfig | None = None,
        reward_config: Phase2RewardConfig | None = None,
    ) -> None:
        """初始化 Phase II 主流程对象。

        输入:
            config: 主流程入口配置，包含交易标的、训练批次、Phase I checkpoint
                路径、产物根目录和设备。
            dataset_config: Phase II selector dataset 构建配置；为空时使用默认值。
            train_config: Double DQN 训练配置；为空时使用默认值。
            reward_config: reward 与 imitation regularization 配置；为空时使用默认值。

        输出:
            无返回值。初始化后对象持有 artifact store、Phase I store、dataset
            builder、checkpoint selector 和运行设备。

        使用场景:
            ``scripts/train_phase2.py`` 或测试用例构造主流程对象。构造阶段不读取
            checkpoint、不加载 dataset、不启动训练，便于单独测试 wiring。
        """
        self.phase1_config = phase1_config
        self.config = config
        self.dataset_config = dataset_config or Phase2DatasetConfig()
        self.train_config = train_config or Phase2TrainConfig()
        self.reward_config = reward_config or Phase2RewardConfig()
        self.device = RuntimeUtils.resolve_device(config.device, logger=logger)

        self.artifact_store = Phase2ArtifactStore(
            pair=config.pair,
            train_batch_id=config.train_batch_id,
            artifacts_root=config.artifacts_root,
        )
        self.phase1_store = Phase1ArtifactStore(
            pair=config.pair,
            batchid=config.train_batch_id,
            artifacts_root=config.artifacts_root,
        )
        self.dataset_builder = Phase2SelectionDatasetBuilder(
            tsize=self.dataset_config.tsize,
        )
        self.checkpoint_selector = Phase2CheckpointSelector()

        self.phase1_model: ArchetypeVQModel | None = None
        self.q_network: Phase2QNetwork | None = None
        self.datasets: dict[str, Phase2SelectionDataset] = {}
        self.validation_checkpoints: list[Phase2ValidationCheckpoint] = []
    

    def run(self) -> None:
        """执行 Phase II 完整训练流程。

        输入:
            无显式输入，消费构造函数中保存的配置和路径。

        输出:
            无返回值。训练结果保存在 ``self.training_result``；若流程失败，
            抛出 ``Phase2FatalError``。

        使用场景:
            Phase II 对外唯一执行入口。脚本、notebook 或调度系统不应绕过本方法
            直接调用私有步骤，否则容易跳过 lineage、checkpoint 和 report 约束。

        流程顺序:
            1. 准备 Phase I/Phase II 产物目录和路径；
            2. 加载并冻结 Phase I best checkpoint；
            3. 构建 train/val/test selector dataset；
            4. 创建 selector Q-network；
            5. 执行 Double DQN 训练和 validation checkpoint selection；
            6. 执行 test split 评估；
            7. 写出最终 report。
        """

        try:
            # Step 1: 准备 Phase II/Phase I 产物目录和路径契约。
            self.prepare()
            # Step 2: 加载 Phase I best checkpoint，并构建 train/val/test selector dataset。
            self.load_inputs()
            # Step 3: 创建 Phase II selector Q-network 和训练所需主组件。
            self.build_components()
            # Step 4: 集中校验训练、评估和 checkpoint 所需组件。
            self.validate_components()
            logger.info("Phase II components validated")
            # Step 5: 训练 Double DQN selector，并产出 validation checkpoint selection 摘要。
            self.train()
            logger.info("Phase II training completed")
            # Step 6: 使用 test split 做最终离线评估，不参与 best checkpoint 选择。            
            self.select_and_save_best_checkpoint()
        except Phase2FatalError:
            raise
        except Exception as exc:
            raise Phase2FatalError("phase2 main flow failed") from exc

    def prepare(self) -> None:
        """初始化 Phase II 和 Phase I 产物路径契约。

        输入:
            无显式输入，使用 ``self.config`` 中的 pair、train_batch_id 和
            artifacts_root。

        输出:
            无返回值。调用后 ``artifact_store`` 和 ``phase1_store`` 应完成标准目录
            与 artifact path key 初始化。

        使用场景:
            ``run()`` 的第一步。后续读取 horizon dataset、Phase I labels、Phase I
            checkpoint 和保存 Phase II checkpoint/report 都依赖本步骤。
        """

        self.artifact_store.initialize_phase2_artifact_dirs()
        self.phase1_store.initialize_phase1_artifact_dirs()

    def load_inputs(self) -> None:
        """加载 Phase II 训练输入。

        输入:
            无显式输入，读取 ``self.config.phase1_checkpoint_path``、Phase I horizon
            labels，以及 DataFileStore 中的 train/val/test horizon dataset。

        输出:
            无返回值。调用后 ``self.phase1_model`` 保存冻结的 Phase I model，
            ``self.datasets`` 至少包含 ``train``、``val`` 和 ``test``。

        使用场景:
            ``run()`` 的第二步。该步骤只准备 Phase II 所需离线输入，不创建
            Q-network，不启动训练，也不调用 Phase I encoder 在线生成 label。
        """

        self.phase1_model = self._load_phase1_model()
        for split_name in ("train", "val"):
            self.datasets[split_name] = self._build_or_load_dataset(split_name)

    def build_components(self) -> None:
        """构建 Phase II selector 训练组件。

        输入:
            无显式输入，消费 ``self.datasets["train"]`` 和 ``self.phase1_model``。

        输出:
            无返回值。调用后 ``self.q_network`` 保存 online selector Q-network。

        使用场景:
            ``run()`` 的第三步。当前骨架只创建 Q-network；env、target network、
            replay buffer、trainer 和 evaluator 在 ``train()`` 内按当前训练批次组装。
        """

        train_dataset = self.datasets.get("train")
        if train_dataset is None:
            raise Phase2FatalError("train dataset must be loaded before build_components")
        self.q_network = self._create_q_network(train_dataset)

    def validate_components(self) -> None:
        """集中校验 Phase II 后续训练和评估步骤依赖的组件。

        输入:
            无显式输入，检查 ``self`` 上已经初始化的组件状态。

        输出:
            无返回值。校验失败时抛出 ``Phase2FatalError``。

        使用场景:
            ``run()`` 在训练前调用，避免缺失 Phase I model、dataset 或 Q-network 时
            进入更深层模块才失败。
        """

        if self.phase1_model is None:
            raise Phase2FatalError("phase1 model must be loaded")
        if self.q_network is None:
            raise Phase2FatalError("q_network must be initialized")
        for split_name in ("train", "val", "test"):
            if split_name not in self.datasets:
                raise Phase2FatalError(f"{split_name} dataset is required")

    def train(self) -> None:
        """训练 Phase II Double DQN selector。

        输入:
            无显式输入，消费 ``self.q_network``、``self.phase1_model``、
            ``self.datasets["train"]`` 和 ``self.datasets["val"]``。

        输出:
            无返回值。训练摘要保存到 ``self.training_result``。

        使用场景:
            ``run()`` 的训练步骤。具体 replay、env step、loss、validation 和 checkpoint
            保存仍由下游 trainer/evaluator/artifact store 模块负责。
        """
        self._train_double_dqn(
            q_network=self.q_network,
            train_dataset=self.datasets["train"],
            val_dataset=self.datasets["val"],
            phase1_model=self.phase1_model,
        )

    
    def _load_phase1_model(self) -> ArchetypeVQModel:
        """加载并冻结 Phase I best checkpoint。

        输入:
            无显式输入，读取 ``self.config.phase1_checkpoint_path`` 指向的 checkpoint。

        输出:
            已加载权重、移动到 ``self.device``、切换到 eval、关闭梯度的
            ``ArchetypeVQModel``。

        使用场景:
            ``FrozenArchetypeDecoderPolicy`` 使用该模型的 codebook/decoder 将 selector
            选出的 archetype id 解码为基础动作序列。Phase II 不允许更新 Phase I
            模型参数。
        """
        checkpoint = self.phase1_store.load_best_checkpoint()
         
        state_dim = self._infer_phase1_state_dim(checkpoint.model_state_dict)
        self.state_dim = state_dim
        model = ArchetypeVQModel(
            state_dim=state_dim,
            action_dim=self.phase1_config.action_dim,
            hidden_dim=self.phase1_config.hidden_dim,
            latent_dim=self.phase1_config.latent_dim,
            num_archetypes=self.phase1_config.num_archetypes,
            commitment_cost=self.phase1_config.commitment_cost,
            num_layers=self.phase1_config.num_layers,
            dropout=self.phase1_config.dropout,
        ).to(self.device)
        model.load_state_dict(checkpoint.model_state_dict)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False) 
        return model

    def _build_or_load_dataset(
        self,
        split_name: str,
    ) -> Phase2SelectionDataset:
        """构建或读取 Phase II selector dataset。

        输入:
            split_name: 数据 split 名称，例如 ``"train"``、``"val"`` 或 ``"test"``。

        输出:
            ``Phase2SelectionDataset``，包含 selector 可见状态、当前 horizon dataset
            和 Phase I assigned labels。

        使用场景:
            ``run()`` 分别为 train/val/test 调用本方法。train dataset 供 env/trainer
            采样 transition；val dataset 用于 checkpoint selection；test dataset
            只用于最终离线评估。

        设计约束:
            本方法只读取 Phase I 离线导出的 horizon label 表，不调用 Phase I
            encoder，不读取 DP teacher trajectory，也不把未来状态或价格拼入 selector
            observation。
        """
        horizon_dataset = self.phase1_store.load_horizon_dataset(split_name)
        label_table = self.phase1_store.load_phase1_horizon_labels(split_name)
        dataset = self.dataset_builder.build_from_horizon_and_labels(
            horizon_dataset=horizon_dataset,
            label_table=label_table,
        )         
        self.datasets[split_name] = dataset
        return dataset

    def _create_q_network(
        self,
        dataset: Phase2SelectionDataset,
    ) -> "Phase2QNetwork":
        """创建 Phase II selector Q-network。

        输入:
            dataset: Phase II selection dataset。方法从 visible state shape 推断
                ``state_dim``，并从 Phase I model 或 label 兜底推断 ``num_archetypes``。

        输出:
            移动到 ``self.device`` 的 ``Phase2QNetwork``。

        使用场景:
            ``run()`` 在 dataset 与 Phase I model 就绪后调用。该网络作为 online
            Q-network 传给 ``_train_double_dqn()``，target network 在训练方法中用
            同一配置复制创建。

        设计边界:
            本方法只创建模型对象，不实现 forward/select_action，不加载 Phase II
            checkpoint。模型结构细节属于 ``model/phase2_q_network.py``。
        """ 

        model_config = Phase2ModelConfig(
            state_dim = self.state_dim,
            num_archetypes=self.phase1_config.num_archetypes,
        )
        return Phase2QNetwork(model_config).to(self.device)

    def _train_double_dqn(
        self,
        q_network: Phase2QNetwork,
        train_dataset: Phase2SelectionDataset,
        val_dataset: Phase2SelectionDataset,
        phase1_model: ArchetypeVQModel,
    ) -> None:
        """编排 Double DQN 训练、validation 和 checkpoint selection。

        输入:
            q_network: online selector Q-network。
            train_dataset: 用于 env transition 采样和 replay 更新的数据。
            val_dataset: 用于 deterministic validation 和 best checkpoint selection 的数据。
            phase1_model: 冻结的 Phase I VQ model。

        输出:
            ``Phase2TrainingResult``，包含训练末态和 checkpoint selector 摘要。

        使用场景:
            ``run()`` 在 Phase I model、dataset 和 Q-network 都准备好后调用。
            本方法负责组装 decoder policy、target network、optimizer、replay buffer、
            env、trainer 和 evaluator，并控制 epoch 外层循环。

        设计边界:
            不在本方法中实现 reward、Double DQN target、TD loss、imitation loss 或
            replay sampling。这些逻辑分别属于 env/reward/loss/replay/trainer 模块。
        """
        decoder_policy = FrozenArchetypeDecoderPolicy(
            phase1_model=phase1_model,
            device=self.device,
        )
        target_q_network = Phase2QNetwork(q_network.config).to(self.device)
        target_q_network.load_state_dict(q_network.state_dict())
        optimizer = torch.optim.Adam(
            q_network.parameters(),
            lr=self.train_config.learning_rate,
        )

        previous_t_states, current_t_states = train_dataset.visible_states
        replay_buffer = Phase2ReplayBuffer(
            capacity=self.train_config.replay_capacity,
            visible_state_shapes=(
                tuple(previous_t_states.shape[1:]),
                tuple(current_t_states.shape[1:]),
            ),
            seed=self.train_config.seed,
        )
        env = ArchetypeSelectionEnv(
            dataset=train_dataset,
            decoder_policy=decoder_policy,
            reward_config=self.reward_config,
        )
        trainer = Phase2DoubleDqnTrainer(
            online_q_network=q_network,
            target_q_network=target_q_network,
            env=env,
            replay_buffer=replay_buffer,
            train_config=self.train_config,
            reward_config=self.reward_config,
            optimizer=optimizer,
            device=self.device,
        )
        evaluator = Phase2Evaluator(
            reward_config=self.reward_config,
            device=self.device,
        )
        for epoch in range(1, self.train_config.epochs + 1):
            latest_checkpoint = trainer.train_one_epoch(epoch=epoch)
            self.artifact_store.save_phase2_checkpoint(latest_checkpoint)
            validation_result = evaluator.evaluate_checkpoint(
                dataset=val_dataset,
                deterministic=True,
                split_name="validation",
                epoch=epoch,
            )
            self.artifact_store.save_phase2_validation_result(
                validation_result,
                split_name="validation",
                epoch=epoch,
            )
            self.validation_checkpoints.append(
                Phase2ValidationCheckpoint(
                    epoch=epoch,
                    validation_result=validation_result,
                )
            )
    def select_and_save_best_checkpoint(self) -> None:     
        pass
    
     

    def _infer_phase1_state_dim(self, state_dict: Mapping[str, Any]) -> int:
        """从 Phase I checkpoint state dict 推断 state_dim。

        输入:
            state_dict: Phase I checkpoint 中的 model state dict。

        输出:
            ``int``，单步 state feature 维度。

        使用场景:
            ``_load_phase1_model()`` 重建 ``ArchetypeVQModel`` 时调用。当前 Phase I
            checkpoint config 不保证保存 ``state_dim``，因此从 encoder 第一层权重
            的输入维度反推。
        """

        state_weight = state_dict.get("encoder.state_adapter.0.weight")
        if not isinstance(state_weight, torch.Tensor):
            raise ValueError(
                "phase1 checkpoint is missing encoder.state_adapter.0.weight"
            )
        return int(state_weight.shape[1])

__all__ = [
    "Phase2FatalError",
    "Phase2MainFlow",
    "Phase2TrainingResult",
]
