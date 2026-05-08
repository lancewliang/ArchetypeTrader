"""Phase I Archetype Discovery 主流程骨架。

论文语义:
    ArchetypeTrader 第一阶段先用 Single-trade DP planner 生成
    demonstration trajectory ``tau = (s_demo, a_demo, r_demo)``，再用
    VQ encoder-decoder 将每个 horizon 压缩成有限 codebook 中的离散
    trading archetype。后续 Phase II 只选择 archetype id，Phase III 再做
    step-level refinement；因此 Phase I 产物是后两阶段的策略空间基础。

工程边界:
    这个模块只负责编排第一阶段训练，不直接实现数据采样、DP teacher 生成或
    模型细节。按照技术设计，DP 和 horizon 采样必须在离线数据处理阶段完成；
    Phase I 训练只消费已经固化、可审计的 manifest/feather 产物，训练 VQ
    archetype encoder-decoder，并导出 Phase II/III 需要的 ``codebook.pt``、
    ``decoder.pt`` 和 horizon-level archetype labels。

主链路:
    ``data_process_manifest.json``
    -> train/val/test demonstration records
    -> train-only normalizer
    -> VQ encoder-decoder training
    -> guardrail/checkpoint selection
    -> labels/model/report export。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..model.data_types import ArtifactPaths
from ..store.artifact_store import DataFileStore


class Phase1FatalError(RuntimeError):
    """Phase I 训练遇到不可恢复错误。"""


@dataclass(frozen=True)
class Phase1MainConfig:
    """Phase I 主流程配置。

    这里保留主流程必需的最小配置。更细的模型、loss、checkpoint 和评估配置
    后续可以拆到独立 config 模块，再由入口脚本组装后传入。

    字段语义:
        pair: 交易标的，用于产物目录、报告和跨阶段一致性校验。
        train_batch_id: 本次 Phase I 训练批次 ID。Phase II 通过该 ID 绑定
            对应的 decoder/codebook/labels，避免误用旧产物。     
        train_file/val_file/test_file: 旧式直接文件入口，仅作为兼容路径；正式
            实现应优先使用 manifest，并在此路径上避免重新采样或重新运行 DP。
        epochs/pretrain_epochs: 对应技术设计中的 Phase B VQ 训练轮数和 Phase A
            reconstruction 预训练轮数。严格论文复现模式应关闭 Phase A。
        
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


class Phase1MainFlow:
    """第一阶段训练主编排器。

    职责定位:
        本类是 orchestration layer。它只决定步骤顺序、保存步骤间状态和统一
        包装不可恢复错误；真正的数据集、模型、loss、trainer、evaluator、
        checkpoint policy 和 report writer 应由各自模块实现。

    论文到工程的顺序映射:
        1. 论文中的 fixed-length chunks 和 DP demonstrations 在
           ``load_inputs()`` 中从 manifest 读取，而不是现场生成。
        2. 论文公式 (1) 的 encoder、公式 (2) 的 VQ codebook、公式 (3) 的
           decoder 和公式 (4) 的训练损失在 ``build_components()`` 组装。
        3. ``train()`` 先执行工程增强的 Phase A reconstruction pretrain，
           再执行完整 VQ training，以学习离散、可复用 archetypes。
        4. ``select_best_checkpoint()`` 按技术设计的 composite score 和
           guardrail 选择可签收 checkpoint。
        5. ``export_phase2_artifacts()`` 和 ``export_horizon_labels()`` 将
           best checkpoint 冻结成 Phase II/III 的稳定接口。

    推荐调用顺序:
        1. ``prepare()``
        2. ``load_inputs()``
        3. ``build_components()``
        4. ``train()``
        5. ``select_best_checkpoint()``
        6. ``export_phase2_artifacts()``
        7. ``export_horizon_labels()``
        8. ``write_report()``

    ``run()`` 会按以上顺序执行。关键产物路径由内部状态、report 和 artifact
    store 统一记录。
    """

    def __init__(self, config: Phase1MainConfig) -> None:
        self.config = config
        self.data_store = DataFileStore()

        self._artifact_paths: ArtifactPaths = {}

    def run(self) -> None:
        """执行 Phase I 主流程。

        流程说明:
            ``run()`` 是唯一的顶层入口，保持步骤顺序与技术设计一致。每一步
            都只依赖前一步写入的内部状态:

            - ``prepare`` 建立运行目录和可复现上下文。
            - ``load_inputs`` 将离线 demonstration 产物加载为训练输入。
            - ``build_components`` 组装 dataset/model/loss/trainer/evaluator。
            - ``train`` 产出 epoch checkpoint 与评估指标。
            - ``select_best_checkpoint`` 固化 best checkpoint，失败时阻断导出。
            - ``export_phase2_artifacts`` 导出 frozen encoder/decoder/codebook。
            - ``export_horizon_labels`` 生成各 split 的 archetype label。
            - ``write_report`` 写入审计、guardrail 和 Phase II eligible 状态。

        失败边界:
            已知的 ``Phase1FatalError`` 原样抛出，便于 CLI 返回非零退出码。
            其他异常统一包装为 ``Phase1FatalError``，避免上层误判为成功。
        """

        try:
            # 1. 准备运行上下文。这里不触碰训练数据
            self.prepare()
            # 2. 加载论文中的 demonstration dataset D={tau_i}。工程上应优先
            #    从 data_process_manifest 读取 sampled horizons 和 DP teacher，
            #    严禁在训练阶段重新枚举窗口或重新调用 SingleTradeDPPlanner。
            self.load_inputs()
            # 3. 构建公式 (1)-(4) 对应的训练组件：encoder、VQ codebook、
            #    causal decoder、loss、optimizer、scheduler、evaluator 等。
            self.build_components()
            # 4. 执行训练。技术设计中包含 Phase A reconstruction pretrain、
            #    K-means warmup，以及 Phase B 完整 VQ 训练和周期评估。
            self.train()
            # 5. 用 selection policy 和 guardrail 固化 best checkpoint。
            #    如果发生 fatal collapse，后续导出必须被阻断。
            self.select_best_checkpoint()
            # 6. 从 best checkpoint 导出跨阶段接口模型。Phase II/III 在线路径
            #    主要依赖 decoder.pt + codebook.pt，encoder.pt 服务离线编码。
            self.export_phase2_artifacts()
            # 7. 用 best encoder/VQ 为 horizon 分配 code_label，作为 Phase II
            #    KL/demo consistency 的监督标签和评估审计输入。
            self.export_horizon_labels()
            # 8. 写最终报告，记录配置、best epoch、指标、guardrail、warning、
            #    sign-off 状态和与论文设定不同的工程增强。
            self.write_report()
        except Phase1FatalError:
            raise
        except Exception as exc:  # pragma: no cover - 顶层错误包装
            raise Phase1FatalError("phase1 main flow failed") from exc

    def prepare(self) -> None:
        """创建输出目录和训练主流程上下文。

        功能流程:
            1. 通过 ``DataFileStore`` 初始化 Phase I 标准产物目录。
            2. 保存标准产物路径字典，供 checkpoint、model export、label 和
               report 步骤复用。

        职责边界:
            日志初始化和随机种子初始化属于入口脚本的 runtime bootstrap，
            不在训练主流程中执行。
        """

        artifacts_root = self._infer_artifacts_root()
        self._artifact_paths = self.data_store.initialize_phase1_artifact_dirs(
            pair=self.config.pair,
            batch_id=self.config.train_batch_id,
            artifacts_root=artifacts_root,
        )

    def _infer_artifacts_root(self) -> Path:
        """从 ``output_dir`` 推断 artifacts 根目录。

        ``Phase1MainConfig.output_dir`` 当前仍保留为兼容字段；目录创建规则由
        ``DataFileStore.initialize_phase1_artifact_dirs(pair, batch_id, root)``
        统一管理。
        """

        output_dir = self.config.output_dir
        if output_dir.name == "phase1" and len(output_dir.parents) >= 3:
            return output_dir.parents[2]
        return output_dir


    def load_inputs(self) -> None:
        """加载 Phase I 训练输入。

        论文语义:
            输入是 demonstration trajectory 集合
            ``D = {tau_i}_{i=0}^{n-1}``，其中每个
            ``tau = (s_demo, a_demo, r_demo)`` 来自固定长度 horizon。DP teacher
            action ``a_demo`` 表示 short/flat/long，论文中强调每个 horizon
            捕捉主要单次交易机会；工程数据还允许 no-trade 低机会样本，但必须
            在报告中标注比例。

        功能流程:
            1. 优先读取 ``data_process_manifest``，验证
               ``phase == "phase1_data_process"``。
            2. 加载 input schema、feature provenance、train/val/test sampled
               horizons 和对应 DP teacher actions/rewards。
            3. 校验 schema hash、data process hash、DP teacher hash、pair/split
               元信息一致，避免训练集与 teacher 或旧 schema 错配。
            4. 校验 sampled horizons 与 DP teacher 的 ``sample_id`` 集合完全一致，
               且 actions/rewards 长度等于 horizon length。
            5. 只用 train split 拟合 state/reward normalizer，再变换所有 split，
               防止 validation/test 信息泄漏到训练统计量中。
            6. 兼容旧 ``train_file``/``val_file``/``test_file`` 时，也只能做读取和
               适配，不能重新采样、重建 horizon 或重新运行 DP。

        输出状态:
            应写入 ``self._inputs``，例如 manifest、schema、normalized records、
            normalizer 参数和可选 non-overlap/full-time label 来源。
        """
        raise NotImplementedError(
            "build Phase I datasets, model and training components here"
        )

    def build_components(self) -> None:
        """构建 dataset、dataloader、model、optimizer、scheduler 和 evaluator。

        论文语义:
            - Encoder 对应公式 (1)，输入 ``s_demo/a_demo/r_demo`` 并输出连续
              latent ``z_e``。
            - Vector quantizer 对应公式 (2)，把 ``z_e`` 映射到最近 codebook
              entry ``z_q=e_k``，形成离散 archetype id。
            - Decoder 对应公式 (3)，在因果约束下根据 state 序列和 ``z_q``
              重构 demonstration action。
            - Loss 对应公式 (4)，核心为 action reconstruction、codebook loss
              和 commitment loss；技术设计中的 usage regularization、EMA、
              K-means warmup 等属于工程增强，必须可配置并写入报告。

        功能流程:
            1. 用 ``self._inputs`` 构建 ``Phase1DemoDataset`` 和 collate 函数，
               batch 应保留 states/actions/rewards/sample_id/meta。
            2. 构建 VQ archetype model，保证 decoder 是 causal decoder，不能在
               第 t 步使用未来 action/reward 信息。
            3. 构建 ``Phase1Loss``，并根据 strict reproduction 开关决定是否关闭
               usage/alignment/dead-code 等增强项。
            4. 构建 optimizer、scheduler、evaluator、checkpoint manager 和
               selection policy。
            5. 将组件写入 ``self._components``，供 ``train`` 和导出步骤复用。
        """

        raise NotImplementedError(
            "build Phase I datasets, model and training components here"
        )

    def train(self) -> None:
        """执行 Phase A 预训练和 Phase B VQ 训练。

        Phase A:
            技术设计中的稳定性增强。跳过 VQ 离散化，直接使用 encoder 输出的
            ``z_e`` 条件化 decoder，只优化 action reconstruction，使 encoder 和
            decoder 先学会解释 demonstration 行为。严格论文复现模式应将
            ``pretrain_epochs`` 设为 0。

        Warmup:
            Phase A 后可用 train batches 的 ``z_e`` 做 K-means codebook warmup。
            论文未指定该初始化方式，因此报告中需要标记为工程增强。

        Phase B:
            完整启用 VQ，按公式 (4) 训练 ``z_e -> z_q -> action logits``。正式
            实现应在每个 epoch 记录 reconstruction、code usage、perplexity、
            replay return、risk 和 behavior diversity 等指标，并按配置周期调用
            evaluator。

        Checkpoint:
            每轮至少应保存 last checkpoint；通过 selection policy 的 epoch 可
            promote 为 candidate/best。若 dead-code restart 连续失败导致 fatal
            collapse，应抛出 ``Phase1FatalError`` 并阻断后续导出。
        """

        raise NotImplementedError(
            "implement pretrain loop, VQ training loop and checkpoint saving here"
        )

    def select_best_checkpoint(self) -> None:
        """根据评估指标和 guardrail 选择 best checkpoint。

        功能流程:
            1. 汇总训练过程中 evaluator 产生的 epoch metrics。
            2. 调用 selection policy 进行 guardrail 检查，包括 code usage、
               drawdown/sharpe、inter-code action diversity、decoder sensitivity
               和 epoch code stability。
            3. 对通过 guardrail 的 checkpoint 计算 composite score，选择验证集
               表现最好的版本作为 ``best_vq_model.pt``。
            4. 记录 reject/warning/fatal verdict、checkpoint hash 和 best epoch，
               供 ``phase1_report.json`` 和 Phase II sign-off 使用。

        失败边界:
            没有 best checkpoint 时，不能导出 encoder/decoder/codebook/labels。
            fatal collapse 时应显式标记 ``fatal_collapse=true``。
        """

        raise NotImplementedError(
            "implement Phase I checkpoint selection policy here"
        )

    def export_phase2_artifacts(self) -> None:
        """导出 Phase II/III 复用的 encoder、decoder 和 codebook。

        论文语义:
            Phase II selector 的动作空间就是 Phase I 学到的 codebook index。
            Phase II/III 选中某个 code 后，将 ``e_k`` 输入 frozen decoder，生成
            horizon 内 step-wise base actions。

        功能流程:
            1. 读取 ``self._best_checkpoint_path`` 指向的 best checkpoint。
            2. 从完整 VQ model 中拆出 encoder、decoder 和 codebook。
            3. 导出 ``encoder.pt``、``decoder.pt``、``codebook.pt``，并记录到
               ``self._exported_paths``。
            4. 同步导出或校验 ``input_schema.json``、``state_normalizer.json`` 和
               ``phase1_config.yaml``，确保 Phase II 使用同一特征顺序和成本语义。

        使用边界:
            ``decoder.pt`` 和 ``codebook.pt`` 是 Phase II/III 在线主路径；encoder
            主要用于离线编码、label 复现和诊断，不应成为线上推理依赖。
        """

        raise NotImplementedError(
            "export encoder.pt, decoder.pt and codebook.pt from best checkpoint here"
        )

    def export_horizon_labels(self) -> None:
        """使用 best checkpoint 为各 split 导出 horizon archetype labels。

        论文语义:
            每条 demonstration trajectory 经过 encoder 和 VQ 后得到离散
            ``code_label``。Phase II 的 KL/demo consistency 项使用这些 label
            约束 selector 不要过早偏离 Phase I 发现的 archetype 选择。

        功能流程:
            1. 加载 best encoder/VQ，并切到 eval/frozen 模式。
            2. 对 sampled train/val/test records 编码，生成
               ``sampled_horizon_labels_{split}.feather``。
            3. 对 manifest 中可选的 full-time train 和 non-overlap split 生成
               对应 label。技术设计要求 Phase II validation/evaluation 优先使用
               non-overlap val label 来降低 horizon 重叠泄漏风险。
            4. label 文件至少包含 ``sample_id``、``start_index``、``end_index``、
               ``code_label``、``demo_return``、``num_switches``、``is_no_trade``、
               ``sample_source``、``_config_hash`` 和 ``_schema_hash``。
            5. 将输出路径写入 ``self._label_paths``，供 report 和后续阶段读取。
        """

        raise NotImplementedError(
            "generate sampled_horizon_labels_{split}.feather files here"
        )

    def write_report(self) -> None:
        """写入 Phase I 训练报告。

        功能流程:
            1. 汇总配置、manifest provenance、normalizer、best checkpoint、导出路径
               和 label 路径。
            2. 写入 action/VQ/replay/risk/behavior/boundary 等评估指标，以及
               composite score 和 sensitivity 检查结果。
            3. 明确记录与论文默认设定不同的工程增强，例如 Phase A pretrain、
               EMA codebook、K-means warmup、usage regularization、no-trade 样本
               和 reward alignment 选择。
            4. 写入 guardrail verdict、warning、fatal collapse 状态和
               ``phase1_checkpoint_eligible_for_phase2``。
            5. 以稳定 JSON 格式保存 ``phase1_report.json``，作为 Phase II
               artifact validator 和 sign-off 的输入。
        """

        raise NotImplementedError("write phase1_report.json here")
