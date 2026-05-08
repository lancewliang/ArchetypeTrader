from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..data.data_load import DataLoad
from ..data.horizon_builder import HorizonBuilder
from ..model.data_types import ArtifactPaths, HorizonDataset, TrajectoryDataset
from ..model.vq_archetype import ArchetypeVQModel
from ..store.artifact_store import DataFileStore
from ..tool.SingleTrade_DP_Planner import SingleTrade_DP_Planner


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
    train_file: Path | None = None
    val_file: Path | None = None
    test_file: Path | None = None
    epochs: int = 100
    pretrain_epochs: int = 10
    batch_size: int = 256
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
    gamma: float = 1.0


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
            ``DataFileStore`` 统一管理。

        论文描述:
            Phase I 的离线产物是 Phase II 选择策略和 Phase III refinement 的
            共同前置条件。按交易标的和批次隔离产物，可以确保每组 archetypes
            都能追溯到对应的训练数据、DP demonstrations 和 VQ 训练配置。
        """

        self.config = config
        
        self.horizon_datasets: dict[str, HorizonDataset] = {}
        self.trajectory_datasets: dict[str, TrajectoryDataset] = {}   
        self.data_load: DataLoad | None = None
        self.horizon_builder: HorizonBuilder | None = None
        self.dp_planner: SingleTrade_DP_Planner | None = None
        self.model: ArchetypeVQModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.dataloaders: dict[str, DataLoader[tuple[torch.Tensor, ...]]] = {}
        self.data_store: DataFileStore | None = None
        self.best_metric: float | None = None
        self.device = torch.device("cpu")
 

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
            # Step 4: 可选预训练 encoder-decoder，使模型具备基础动作重构能力。
            self.pretrain()
            # Step 5: 训练 VQ encoder-decoder，使 codebook 学到可复用 trading archetypes。
            self.train()
            # Step 6: 根据验证指标选择最能代表稳定 archetype 发现结果的 checkpoint。
            self.select_best_checkpoint()
            # Step 7: 从 best checkpoint 导出 Phase II/III 复用的 encoder、decoder 和 codebook。
            self.export_phase2_artifacts()
            # Step 8: 用训练好的 encoder/codebook 为 sampled horizons 生成 archetype labels。
            self.export_horizon_labels()
            # Step 9: 写出配置、指标、诊断和产物索引，支撑复现实验与后续阶段审计。
            self.write_report()
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
        self.data_store = DataFileStore(
            pair=self.config.pair,
            batchid=self.config.train_batch_id,
        )   
        self.data_store.initialize_phase1_artifact_dirs()
        self.data_load = DataLoad()
        self.horizon_builder = HorizonBuilder(horizon=self.config.horizon)
        self.dp_planner = SingleTrade_DP_Planner(
            horizon=self.config.horizon,
            action_set=tuple(range(self.config.action_dim)),
            initial_action=1,
            gamma=self.config.gamma,
        )
        
        requested_device = torch.device(self.config.device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device



    def load_inputs(self) -> None:
        """加载输入 split，并形成 Phase I 训练数据基础。

        功能描述:
            读取 train/val/test 文件或默认数据源，校验 feature schema，构建固定
            horizon 的市场状态窗口，并为训练集采样 demonstration chunks。正式
            实现应避免跨 split 泄漏，并保存输入 schema、feature provenance 和
            normalizer 信息。

        论文描述:
            论文先从训练数据中采样 n 个长度为 h 的 data chunks，再对每个 chunk
            应用 single-trade DP planner。每个 horizon 的市场观测记为
            s_demo=(s_0,...,s_{h-1})，DP 输出动作序列 a_demo，执行动作得到
            reward 序列 r_demo，最终组成 tau=(s_demo, a_demo, r_demo) 作为
            VQ archetype extraction 的训练数据集 D。
        """
      
        split_files: dict[str, Path | None] = {
            "train": self.config.train_file,
            "val": self.config.val_file,
            "test": self.config.test_file,
        }
        for split_name, path in split_files.items():
            if path is None:
                continue
            if self.data_store is None:
                raise Phase1FatalError("data store must be initialized")
            self.horizon_datasets[split_name] = self.data_store.load_horizon_dataset(split_name)
            self.trajectory_datasets[split_name] = self.data_store.load_trajectory_dataset(split_name)

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
        if train_dataset is None:
            raise Phase1FatalError("train trajectory dataset is required")
        for split_name, trajectory_dataset in self.trajectory_datasets.items():
            tensor_dataset = self._build_tensor_dataset(trajectory_dataset)
            self.dataloaders[split_name] = DataLoader(
                tensor_dataset,
                batch_size=self.config.batch_size,
                shuffle=(split_name == "train"),
            )

        first_states, _, _ = train_dataset[0]
        state_dim = int(first_states.shape[-1])
        self.model = ArchetypeVQModel(
            state_dim=state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            num_archetypes=self.config.num_archetypes,
            commitment_cost=self.config.commitment_cost,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

    def pretrain(self) -> None:
        """预训练 Phase I encoder-decoder 并保存 checkpoint。

        功能描述:
            执行可选 reconstruction pretrain，让 encoder/decoder 具备基础动作
            重构能力。checkpoint 保存委托给 ``DataFileStore``，具体持久化策略
            后续在 store 层实现。

        论文描述:
            预训练阶段先跳过 VQ quantization，仅优化动作重构，使后续离散
            archetype 学习从较稳定的 encoder/decoder 表示开始。
        """

        train_loader = self.dataloaders.get("train")     
        val_loader = self.dataloaders.get("val")
        if train_loader is None:
            raise Phase1FatalError("train dataloader is required")
        if self.data_store is None:
            raise Phase1FatalError("data store must be initialized")
        if self.model is None or self.optimizer is None:
            raise Phase1FatalError("model and optimizer must be initialized")

        for epoch in range(1, self.config.pretrain_epochs + 1):
            train_metrics = self._run_epoch(train_loader, use_vq=False)
            eval_loader = val_loader or train_loader
            val_metrics = self._evaluate(eval_loader, use_vq=False)
            self.data_store.save_phase1_checkpoint(
                {
                    "stage": "pretrain",
                    "epoch": epoch,
                    "is_best": False,
                    "config": asdict(self.config),
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": {"train": train_metrics, "val": val_metrics},
                }
            )

    def train(self) -> None:
        """训练 Phase I VQ encoder-decoder 并保存 checkpoint。

        功能描述:
            启用 VQ codebook 训练，按 epoch 计算训练/验证指标，并通过
            ``DataFileStore.save_phase1_checkpoint`` 交给 store 层保存 checkpoint。

        论文描述:
            训练目标对应论文式 (4):
            L = L_rec + ||sg[z_e]-z_q||^2 + beta_0 ||z_e-sg[z_q]||^2。
            其中 L_rec 约束 decoder 重构 DP demonstration actions，codebook loss
            使被选中的 archetype 向 encoder 输出靠近，commitment loss 约束
            encoder 稳定地提交到离散 code。优化后，codebook vectors 学到可复用
            trading archetypes，而不是连续、难以由 Phase II selector 探索的
            latent manifold。
        """

        train_loader = self.dataloaders.get("train")     
        val_loader = self.dataloaders.get("val")
        if train_loader is None:
            raise Phase1FatalError("train dataloader is required")
        if self.data_store is None:
            raise Phase1FatalError("data store must be initialized")
        if self.model is None or self.optimizer is None:
            raise Phase1FatalError("model and optimizer must be initialized")
        best_metric = float("inf")

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, use_vq=True)
            eval_loader = val_loader or train_loader
            val_metrics = self._evaluate(eval_loader, use_vq=True)
            metrics = {"train": train_metrics, "val": val_metrics}
            self.data_store.save_phase1_checkpoint(
                {
                    "stage": "vq",
                    "epoch": epoch,
                    "is_best": False,
                    "config": asdict(self.config),
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": metrics,
                }
            )

            metric = float(val_metrics["total_loss"])
            if metric < best_metric:
                best_metric = metric
                self.best_metric = metric
                self.data_store.save_phase1_checkpoint(
                    {
                        "stage": "vq",
                        "epoch": epoch,
                        "is_best": True,
                        "config": asdict(self.config),
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": metrics,
                    }
                )

    def _build_tensor_dataset(
        self,
        trajectory_dataset: TrajectoryDataset,
    ) -> TensorDataset:
        states = torch.as_tensor(
            np.stack([trajectory[0] for trajectory in trajectory_dataset]),
            dtype=torch.float32,
        )
        actions = torch.as_tensor(
            np.stack([trajectory[1] for trajectory in trajectory_dataset]),
            dtype=torch.long,
        )
        rewards = torch.as_tensor(
            np.stack([trajectory[2] for trajectory in trajectory_dataset]),
            dtype=torch.float32,
        )
        return TensorDataset(states, actions, rewards)

    def _run_epoch(
        self,
        dataloader: DataLoader[tuple[torch.Tensor, ...]],
        *,
        use_vq: bool,
    ) -> dict[str, float]:
        if self.model is None or self.optimizer is None:
            raise Phase1FatalError("model and optimizer must be initialized")

        self.model.train()
        totals = self._empty_metric_totals()
        for batch in dataloader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch) if use_vq else self.model.forward_pretrain(batch)
            outputs.total_loss.backward()
            self.optimizer.step()
            self._accumulate_metrics(totals, outputs, batch_size=batch[0].shape[0])
        return self._finalize_metrics(totals)

    @torch.no_grad()
    def _evaluate(
        self,
        dataloader: DataLoader[tuple[torch.Tensor, ...]],
        *,
        use_vq: bool,
    ) -> dict[str, float]:
        if self.model is None:
            raise Phase1FatalError("model must be initialized")

        self.model.eval()
        totals = self._empty_metric_totals()
        for batch in dataloader:
            batch = self._move_batch(batch)
            outputs = self.model(batch) if use_vq else self.model.forward_pretrain(batch)
            self._accumulate_metrics(totals, outputs, batch_size=batch[0].shape[0])
        return self._finalize_metrics(totals)

    def _move_batch(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards = batch
        return (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
        )

    def _empty_metric_totals(self) -> dict[str, float]:
        return {
            "samples": 0.0,
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "vq_loss": 0.0,
            "codebook_loss": 0.0,
            "commitment_loss": 0.0,
        }

    def _accumulate_metrics(
        self,
        totals: dict[str, float],
        outputs: Any,
        *,
        batch_size: int,
    ) -> None:
        totals["samples"] += batch_size
        totals["total_loss"] += float(outputs.total_loss.detach().cpu()) * batch_size
        totals["reconstruction_loss"] += (
            float(outputs.reconstruction_loss.detach().cpu()) * batch_size
        )
        totals["vq_loss"] += float(outputs.vq_loss.detach().cpu()) * batch_size
        totals["codebook_loss"] += (
            float(outputs.codebook_loss.detach().cpu()) * batch_size
        )
        totals["commitment_loss"] += (
            float(outputs.commitment_loss.detach().cpu()) * batch_size
        )

    def _finalize_metrics(self, totals: dict[str, float]) -> dict[str, float]:
        samples = max(totals.pop("samples"), 1.0)
        return {name: value / samples for name, value in totals.items()}
    def select_best_checkpoint(self) -> None:
        """选择 Phase I best checkpoint。

        功能描述:
            汇总训练和验证指标，按 checkpoint selection policy 选择最终导出的
            best VQ 模型。选择策略应同时考虑动作重构质量、VQ code 使用率、
            archetype 分离度、验证 split 稳定性和必要的风控/边界诊断。

        论文描述:
            论文目标是得到 compact and reusable trading archetypes，而不只是最小
            训练 loss。best checkpoint 应优先保留能稳定压缩 demonstration
            trajectories、形成有限离散 code，并能被后续 selector 清晰选择的模型。
            这对应实验设置中“保留验证表现最佳 checkpoint”的离线选择原则。
        """

        raise NotImplementedError(
            "implement Phase I checkpoint selection policy here"
        )

    def export_phase2_artifacts(self) -> None:
        """从 best checkpoint 导出 Phase II/III 复用模型产物。

        功能描述:
            加载 best checkpoint，分别导出 ``encoder.pt``、``decoder.pt`` 和
            ``codebook.pt``。encoder 用于离线生成 demonstration horizon 的
            archetype label；decoder 和 codebook 用于 Phase II 在选定 archetype
            后重构未来 horizon 内的 step-wise base actions，也会被 Phase III
            refinement 作为 base policy 来源。

        论文描述:
            论文 Phase II 的动作是选择离散 archetype id
            a_sel in {0,...,K-1}。选中后，将对应 codebook 向量 e_{a_sel} 输入
            frozen decoder p_theta_d(a_base | s, e_{a_sel})，得到整个 horizon 的
            micro action sequence。因此 Phase I 导出的 decoder/codebook 是后续
            层级 RL 能否复用 learned archetypes 的核心接口。
        """

        raise NotImplementedError(
            "export encoder.pt, decoder.pt and codebook.pt from best checkpoint here"
        )

    def export_horizon_labels(self) -> None:
        """为 sampled horizons 导出离散 archetype labels。

        功能描述:
            使用 best encoder 和 codebook 对 train/val/test sampled horizons 编码，
            将每个 demonstration chunk 分配到最近 codebook index，并写出
            ``sampled_horizon_labels_{split}.feather`` 等 label 文件。正式实现还应
            记录 label 分布、code 使用率和 split 间稳定性诊断。

        论文描述:
            在 Phase II 目标式 (5) 中，hat_a_t^sel 是 VQ encoder 为该 horizon 的
            demonstration chunk 分配的 ground-truth archetype label。RL selector
            一方面最大化 horizon return，另一方面通过 KL penalty 保持接近这些
            demonstration archetype choices。因此 Phase I 必须导出可靠的 horizon
            labels，作为 Phase II imitation/regularization 信号。
        """

        raise NotImplementedError(
            "generate sampled_horizon_labels_{split}.feather files here"
        )

    def write_report(self) -> None:
        """写出 Phase I 训练报告。

        功能描述:
            生成 ``phase1_report.json``，记录配置、输入数据摘要、DP 采样统计、
            训练曲线、best checkpoint 指标、codebook 使用率、archetype 行为诊断、
            horizon boundary 诊断、导出文件路径和失败样本索引等复盘信息。

        论文描述:
            论文强调 DP 仅用于训练阶段，推理时禁用以避免 future information
            leakage。报告应明确记录 DP demonstrations 的生成范围、horizon 采样
            方式和导出 label 的 split 边界，使 Phase II/III 使用 Phase I 产物时
            能验证没有跨时间或跨 split 泄漏，并可复现实验中的 Archetype
            Discovery 结果。
        """

        raise NotImplementedError("write phase1_report.json here")
