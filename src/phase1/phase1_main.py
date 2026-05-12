from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch.utils.data import DataLoader

from ..data.data_load import DataLoad
from ..data.horizon_builder import HorizonBuilder
from ..model.data_types import (
    HorizonDataset,
    TrajectoryDataset,
)
from ..model.tensor_data_types import (
    TrajectoryTensorBatch,
    build_trajectory_tensor_dataset,
    move_trajectory_batch_to_device,
)
from ..model.vq_archetype import ArchetypeVQModel
from .phase1_artifact_store import Phase1ArtifactStore
from ..tool.SingleTrade_DP_Planner import SingleTrade_DP_Planner
from .checkpoint import (
    Phase1CheckpointSelectionResult,
    Phase1CheckpointSelector,
)
from .evaluators import Phase1CodebookEvaluator, Phase1Evaluator
from .horizon_train_label_builder import (
    HorizonTrainLabelBuilder,
    HorizonTrainLabelBuilderConfig,
)
from .metrics import CodeAssignmentSnapshot, Phase1Metrics, Phase1ValidationResult
from .report import Phase1CodebookReport


class Phase1FatalError(RuntimeError):
    """Phase I 主流程的致命错误。

    功能描述:
        将数据读取、DP 示范轨迹生成、VQ 训练、checkpoint 选择和产物导出
        中出现的非预期异常统一包装为 Phase I 失败，便于入口脚本和调度系统
        用单一异常类型处理失败重试、告警和报告。

    论文描述:
        Phase I 是 Archetype Discovery 的离线训练阶段。该阶段输出的 encoder、
        decoder、codebook 和 horizon labels 会被 Phase II/III 复用，因此任何
        训练或导出失败都应阻断后续阶段，避免使用不完整的 archetype 产物。
    """

    pass


@dataclass(frozen=True)
class Phase1MainConfig:
    """Phase I 主流程配置。

    功能描述:
        保存一次 Phase I 训练所需的交易标的、训练批次、数据 split、训练轮数、
        预训练轮数、batch size、学习率和设备等参数。该配置只表达流程入口的
        运行契约，具体模型结构、损失细节和数据 schema 由后续组件实现承载。

    论文描述:
        论文在 Archetype Discovery 中将 Phase I 定义为固定 horizon 上的
        demonstration trajectory 生成与 VQ encoder-decoder 训练。实验设置中
        使用 30k 条 DP trajectories、horizon length h=72、100 个 epoch、
        128-unit VQ encoder-decoder、archetype dimension=16、K=10 和
        beta_0=0.25；这些论文超参数应在正式实现中通过本配置或下游组件配置
        显式记录。
    """

    pair: str
    train_batch_id: str
    epochs: int = 50
    pretrain_epochs: int = 10
    batch_size: int = 1024
    learning_rate: float = 1e-3
    device: str = "cuda"
    horizon: int = 72
    hidden_dim: int = 128
    latent_dim: int = 16
    num_archetypes: int = 10
    action_dim: int = 3
    commitment_cost: float = 0.25
    num_layers: int = 1
    dropout: float = 0.0
    gamma: float = 0.9
    validation_interval: int = 5

    def __post_init__(self) -> None:
        if self.validation_interval < 1:
            raise ValueError("validation_interval must be >= 1")


class Phase1MainFlow:
    """Phase I Archetype Discovery 的主流程编排骨架。

    功能描述:
        串联 Phase I 的离线训练步骤: 初始化产物目录、加载输入数据、构建 DP
        与 VQ 组件、训练 encoder-decoder、选择 best checkpoint、导出 Phase II
        可复用产物、生成 horizon-level archetype labels，并写出训练报告。

    论文描述:
        该流程对应论文三阶段框架中的第一阶段。Phase I 先用 single-trade DP
        planner 在固定 horizon 上生成高质量示范轨迹 tau=(s_demo, a_demo,
        r_demo)，再用 LSTM encoder 将轨迹编码为连续 latent z_e，经 VQ codebook
        离散化为 z_q，最后由 decoder 根据状态和 z_q 重构动作序列。训练完成后，
        codebook 中的离散向量即论文所称可复用 trading archetypes，并为 Phase II
        的 horizon-level selector 提供可选择的 archetype 集合。
    """

    def __init__(self, config: Phase1MainConfig) -> None:
        """绑定训练配置并创建 Phase I 产物 store。

        功能描述:
            根据 ``pair`` 和 ``train_batch_id`` 绑定一次训练批次的产物命名空间，
            后续所有 checkpoint、模型导出、label 和报告路径都通过
            ``Phase1ArtifactStore`` 统一管理。

        论文描述:
            Phase I 的离线产物是 Phase II 选择策略和 Phase III refinement 的
            共同前置条件。按交易标的和批次隔离产物，可以确保每组 archetypes
            都能追溯到对应的训练数据、DP demonstrations 和 VQ 训练配置。
        """

        self.config = config
        self.device = self._resolve_device(config.device)        
        self.horizon_datasets: dict[str, HorizonDataset] = {}
        self.trajectory_datasets: dict[str, TrajectoryDataset] = {}   
        self.data_load: DataLoad | None = None
        self.horizon_builder: HorizonBuilder | None = None
        self.dp_planner: SingleTrade_DP_Planner | None = None
        self.model: ArchetypeVQModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.dataloaders: dict[str, DataLoader[TrajectoryTensorBatch]] = {}
        self.evaluation_dataloaders: dict[str, DataLoader[TrajectoryTensorBatch]] = {}
        self.data_store: Phase1ArtifactStore | None = None
        self.evaluator: Phase1Evaluator | None = None
        self.codebook_evaluator: Phase1CodebookEvaluator | None = None
        self.report: Phase1CodebookReport | None = None
        self.selector: Phase1CheckpointSelector | None = None
        self.best_checkpoint_selection: Phase1CheckpointSelectionResult | None = None
        self.validation_results: dict[int, Phase1ValidationResult] = {}
        self.assignment_history: list[CodeAssignmentSnapshot] = []
        self.horizon_train_label_builder: HorizonTrainLabelBuilder | None = None
 
    def _resolve_device(self, device: str) -> torch.device:
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested_device

    def run(self) -> None:
        """按论文 Phase I 顺序执行 Archetype Discovery。

        功能描述:
            以线性流程执行各个骨架步骤，并将非预期异常统一转换为
            ``Phase1FatalError``。当前类只定义流程契约，具体数据处理、模型构建、
            训练和导出逻辑由各步骤的后续实现补齐。

        论文描述:
            论文中的 Phase I 不是在线交易过程，而是离线发现 archetypes 的训练
            管线。执行顺序需要先生成/读取 demonstration 数据，再训练 VQ
            encoder-decoder，最后固定 best checkpoint 并导出离散 archetype
            codebook 与 horizon labels，供 Phase II 的 RL selector 使用。
        """

        try:
            # Step 1: 准备 Phase I 产物目录和路径契约。
            self.prepare()
            # Step 2: 加载市场数据 split，并准备固定 horizon 的 DP 示范样本。
            self.load_inputs()
            # Step 3: 构建 DP planner、trajectory dataset、encoder、decoder 和 VQ codebook。
            self.build_components()
            # Step 4: 集中校验训练、评估和 checkpoint 所需组件。
            self.validate_components()
            # Step 5: 可选预训练 encoder-decoder，使模型具备基础动作重构能力。
            self.pretrain()
            # Step 6: 训练 VQ encoder-decoder，使 codebook 学到可复用 trading archetypes。
            self.train()
            # Step 7: 根据验证指标选择最能代表稳定 archetype 发现结果的 checkpoint。
            self.select_and_save_best_checkpoint()
            # Step 9: 用训练好的 encoder/codebook 为 sampled horizons 生成 archetype labels。
            self.export_horizon_labels()
        except Phase1FatalError:
            raise
        except Exception as exc:
            raise Phase1FatalError("phase1 main flow failed") from exc

    def prepare(self) -> None:
        """初始化 Phase I 标准产物目录。

        功能描述:
            创建当前 ``pair`` / ``train_batch_id`` 对应的 Phase I 根目录、
            checkpoints、diagnostics、tensorboard、latent_snapshots 和
            failure_cases 等目录，并记录标准产物路径。

        论文描述:
            Archetype Discovery 会产生多类离线产物: DP demonstrations、VQ
            checkpoint、learned codebook、decoder、horizon labels 和诊断指标。
            这些产物必须在训练前有稳定路径，才能保证后续 Phase II selector
            使用的是同一组离散 archetypes，而不是混用不同实验批次。
        """
        self.data_store = Phase1ArtifactStore(
            pair=self.config.pair,
            batchid=self.config.train_batch_id,
        )   
        self.data_store.initialize_phase1_artifact_dirs()
        self.report = Phase1CodebookReport()
        self.data_load = DataLoad()
        self.horizon_builder = HorizonBuilder(horizon=self.config.horizon)
        self.dp_planner = SingleTrade_DP_Planner(
            horizon=self.config.horizon,
            action_set=tuple(range(self.config.action_dim)),
            initial_action=1,
            gamma=self.config.gamma,
        )

    def load_inputs(self) -> None:
        """加载输入 split，并形成 Phase I 训练数据基础。

        功能描述:
            读取前置数据处理已经固化的 train/val/test horizon 与 trajectory
            数据集。训练入口不再接收原始 feature split 文件路径，避免训练阶段
            重新读 feature 文件或重新运行 DP teacher。

        论文描述:
            论文先从训练数据中采样 n 个长度为 h 的 data chunks，再对每个 chunk
            应用 single-trade DP planner。每个 horizon 的市场观测记为
            s_demo=(s_0,...,s_{h-1})，DP 输出动作序列 a_demo，执行动作得到
            reward 序列 r_demo，最终组成 tau=(s_demo, a_demo, r_demo) 作为
            VQ archetype extraction 的训练数据集 D。
        """
      
        for split_name in ("train", "val", "test"):
            self.horizon_datasets[split_name] = (
                    self.data_store.load_horizon_dataset(split_name)
                )
            self.trajectory_datasets[split_name] = (
                    self.data_store.load_trajectory_dataset(split_name)
                )
          

    def build_components(self) -> None:
        """构建 DP 示范生成与 VQ 训练组件。

        功能描述:
            初始化 single-trade DP planner、trajectory/horizon datasets、
            dataloaders、LSTM trajectory encoder、causal action decoder、
            VectorQuantizer codebook、optimizer、scheduler、metric recorder 和
            checkpoint writer 等训练部件。

        论文描述:
            论文 Algorithm 1 的 DP planner 在每个 horizon 中限制最多一次交易，
            以过滤短暂噪声并突出主要价格运动。随后 LSTM encoder
            q_theta_e(z_e | s_demo, a_demo, r_demo) 将完整示范轨迹压缩为
            连续 latent，VQ 模块从 codebook epsilon={e_0,...,e_{K-1}} 中选择
            最近向量 z_q，decoder p_theta_d(a_hat_demo | s_demo, z_q) 再根据
            状态和离散 archetype 重构动作序列。
        """

        train_dataset = self.trajectory_datasets.get("train")
        if not train_dataset:
            return
        for split_name, trajectory_dataset in self.trajectory_datasets.items():
            tensor_dataset = build_trajectory_tensor_dataset(trajectory_dataset)
            self.dataloaders[split_name] = DataLoader(
                tensor_dataset,
                batch_size=self.config.batch_size,
                shuffle=(split_name == "train"),
            )
            self.evaluation_dataloaders[split_name] = DataLoader(
                tensor_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        first_states, _, _ = train_dataset[0]
        state_dim = int(first_states.shape[-1])
        model = ArchetypeVQModel(
            state_dim=state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            num_archetypes=self.config.num_archetypes,
            commitment_cost=self.config.commitment_cost,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)
        self.model = model
        self.evaluator = Phase1Evaluator(model=model, device=self.device)
        self.codebook_evaluator = Phase1CodebookEvaluator(
            model=model,
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
        )
        self.selector = Phase1CheckpointSelector()
        self.horizon_train_label_builder = HorizonTrainLabelBuilder(
            config=HorizonTrainLabelBuilderConfig(
                horizon=self.config.horizon,
                batch_size=self.config.batch_size,
                device=self.device,
            ),
        )

    def validate_components(self) -> None:
        """集中校验 Phase I 后续训练和评估步骤依赖的组件。"""

        if self.data_store is None:
            raise Phase1FatalError("data store must be initialized")
        if not self.trajectory_datasets.get("train"):
            raise Phase1FatalError("train trajectory dataset is required")
        if "train" not in self.dataloaders:
            raise Phase1FatalError("train dataloader is required")
        if self.model is None:
            raise Phase1FatalError("model must be initialized")
        if self.optimizer is None:
            raise Phase1FatalError("optimizer must be initialized")
        if self.evaluator is None:
            raise Phase1FatalError("evaluator must be initialized")
        if self.codebook_evaluator is None:
            raise Phase1FatalError("codebook evaluator must be initialized")
        if self.report is None:
            raise Phase1FatalError("report must be initialized")
        if self.selector is None:
            raise Phase1FatalError("checkpoint selector must be initialized")
        if "val" not in self.dataloaders:
            raise Phase1FatalError("val dataloader is required")
        if "train" not in self.evaluation_dataloaders:
            raise Phase1FatalError("train evaluation dataloader is required")
        if "val" not in self.evaluation_dataloaders:
            raise Phase1FatalError("val evaluation dataloader is required")

    def pretrain(self) -> None:
        """预训练 Phase I encoder-decoder 并保存 checkpoint。

        功能描述:
            执行可选 reconstruction pretrain，让 encoder/decoder 具备基础动作
            重构能力。checkpoint 保存委托给 ``Phase1ArtifactStore``，具体持久化策略
            后续在 store 层实现。

        论文描述:
            预训练阶段先跳过 VQ quantization，仅优化动作重构，使后续离散
            archetype 学习从较稳定的 encoder/decoder 表示开始。
        """

        train_loader = self.dataloaders["train"]
        for epoch in range(1, self.config.pretrain_epochs + 1):
            train_metrics = self._run_epoch(
                train_loader,
                use_vq=False,
                stage="pretrain",
                split="train",
                epoch=epoch,
            )
            self.data_store.save_phase1_checkpoint(
                stage="pretrain",
                epoch=epoch,
                config=asdict(self.config),
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict()               
            )

    def train(self) -> None:
        """训练 Phase I VQ encoder-decoder 并保存 checkpoint。

        功能描述:
            启用 VQ codebook 训练，按 epoch 计算训练/验证指标和五层 codebook
            validation。模型状态通过 checkpoint 保存，指标通过 datastore JSON 保存。

        论文描述:
            训练目标对应论文式 (4):
            L = L_rec + ||sg[z_e]-z_q||^2 + beta_0 ||z_e-sg[z_q]||^2。
            其中 L_rec 约束 decoder 重构 DP demonstration actions，codebook loss
            使被选中的 archetype 向 encoder 输出靠近，commitment loss 约束
            encoder 稳定地提交到离散 code。优化后，codebook vectors 学到可复用
            trading archetypes，而不是连续、难以由 Phase II selector 探索的
            latent manifold。
        """

        train_loader = self.dataloaders["train"]
                
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(
                train_loader,
                use_vq=True,
                stage="vq",
                split="train",
                epoch=epoch,
            )
            epoch_metrics = self._evaluate_checkpoint(
                epoch=epoch,
                train_metrics=train_metrics,
            )
            self.data_store.save_phase1_epoch_metrics(
                stage="vq",
                epoch=epoch,
                metrics=epoch_metrics,
            )
            self.data_store.save_phase1_checkpoint(
                stage="vq",
                epoch=epoch,
                config=asdict(self.config),
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict()             
            )
            self.report.write_html(
                validation_result=self.validation_results[epoch],
                output_path=self.data_store.artifact_paths[
                    "phase1_codebook_validation_html"
                ],
                config=asdict(self.config),
                artifacts=self.data_store.artifact_paths,
            )


    def select_and_save_best_checkpoint(self) -> None:       
        self.best_checkpoint_selection = self.selector.select_best_from_dir(
            self.data_store.artifact_paths["checkpoints"],
        )
        self.data_store.save_best_checkpoint(self.best_checkpoint_selection.checkpoint)
        
    def export_horizon_labels(self) -> None:
        """导出 Phase I horizon-level archetype labels。

        论文中的 Phase II selector 需要每个固定 horizon 的 VQ archetype
        label ``hat{a}^{sel}`` 作为监督/一致性信号。这里使用 best
        checkpoint 的 encoder 与 codebook 对 DP demonstration trajectories
        离线编码，再通过 ``Phase1ArtifactStore`` 写出 label 文件。
        """

        if self.best_checkpoint_selection is None:
            return

        self.model.load_state_dict(
            self.best_checkpoint_selection.checkpoint.model_state_dict
        )
        for split_name, trajectory_dataset in self.trajectory_datasets.items():
            labels = self.horizon_train_label_builder.build(
                model=self.model,
                trajectory_dataset=trajectory_dataset,
                split_name=split_name,
            )
            self.data_store.save_phase1_horizon_labels(
                labels,
                split_name=split_name,
            )
        
    def _run_epoch(
        self,
        dataloader: DataLoader[TrajectoryTensorBatch],
        *,
        use_vq: bool,
        stage: str | None = None,
        split: str | None = None,
        epoch: int | None = None,
    ) -> Phase1Metrics:
        self.model.train()
        totals = Phase1Metrics(stage=stage, split=split, epoch=epoch)
        for batch in dataloader:
            batch = move_trajectory_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = (
                self.model(batch)
                if use_vq
                else self.model.forward_pretrain(batch)
            )
            outputs.total_loss.backward()
            self.optimizer.step()
            totals.add_batch(batch_size=batch[0].shape[0], outputs=outputs, actions=batch[1])
        return totals.averaged()

    def _evaluate_checkpoint(
        self,
        *,
        epoch: int,
        train_metrics: Phase1Metrics,
    ) -> dict[str, object]:
        """运行 checkpoint 评估，并返回本轮完整 metrics payload。"""

        val_metrics: Phase1Metrics = self.evaluator.evaluate(
            self.dataloaders["val"],
            stage="vq",
            split="val",
            epoch=epoch,
        )

        validation_result = self.codebook_evaluator.evaluate_checkpoint(
            train_loader=self.evaluation_dataloaders["train"],
            val_loader=self.evaluation_dataloaders["val"],
            epoch=epoch,
            checkpoint_id=f"vq_epoch_{epoch:04d}",
            stage="vq",
            train_horizon_dataset=self.horizon_datasets.get("train"),
            val_horizon_dataset=self.horizon_datasets.get("val"),
            assignment_history=tuple(self.assignment_history),
        )
        self.validation_results[epoch] = validation_result

        current_assignment = self.codebook_evaluator.last_assignment_snapshot
        if current_assignment is not None:
            self.assignment_history.append(current_assignment)
        return {
            "train": train_metrics.to_dict(include_context=True),
            "val": val_metrics.to_dict(include_context=True),
            "codebook_validation": validation_result.to_dict(),
            "codebook_validation_flat": validation_result.to_flat_dict(),
        }
