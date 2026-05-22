"""Phase I VQ archetype 模型。

这个文件实现 ArchetypeTrader 第一阶段使用的 VQ encoder-decoder。

类的功能:
    模型接收一条或一批 DP demonstration trajectory：
    ``(states, relative_states, trend_states, actions, rewards, sample_ids)``，
    encoder 使用三路市场状态、actions 和 rewards 压缩成连续 latent ``z_e``，
    再通过 Vector Quantizer 把 ``z_e`` 分配到有限 codebook 中的一个离散
    archetype，最后由 decoder 根据三路市场状态和 archetype 向量重构每个
    时间步的 teacher action。

为什么设计这个类:
    交易策略不应该只是一条固定动作序列。Phase I 的目标是从 DP teacher
    生成的大量轨迹中蒸馏出一组可复用的高层交易行为，例如趋势持有、
    均值回复、反转捕捉等。VQ codebook 提供了一个离散瓶颈，迫使模型把
    相似轨迹归入同一个 archetype，而不是记住每条轨迹的细节。

用途:
    1. Phase I 训练时，用 ``forward`` 学习 action reconstruction 和 VQ loss。
    2. Phase I 预训练时，用 ``forward_pretrain`` 跳过量化，先稳定
       encoder/decoder 的基本重构能力。
    3. Phase II 训练前，用 ``encode`` 给每个 horizon 生成 archetype label。
    4. Phase II/III 推理时，用 ``decode`` 根据 selector 选中的 code
       生成基础动作序列，再交给后续策略选择或细化模块。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .archetype_decoder import ArchetypeActionDecoder
from .archetype_encoder import ArchetypeTrajectoryEncoder
from .codebook import (
    CodebookInitResult,
    QuantizeOutput,
    VectorQuantizer,
    classify_trajectory_directions,
)
from .market_state_input import MarketStateInputEncoder
from .tensor_data_types import (
    ActionLogitTensor,
    ArchetypeLabelTensor,
    LatentTensor,
    TrajectoryTensorBatch,
)
from .trajectory_batch import normalize_trajectory_batch as _normalize_trajectory_batch


@dataclass(frozen=True)
class VqModelOutputs:
    """VQ archetype 模型的一次前向输出。

    ``action_logits`` 用于重构 DP teacher action；
    ``code_id`` 是每条 trajectory 被分配到的 archetype label；
    ``total_loss`` 是基础训练损失，包含 action reconstruction 和 VQ loss。
    更复杂的 usage regularization、contrastive loss、alignment loss 应该在
    trainer/loss 模块中额外组合，避免模型类承担训练策略职责。
    """

    action_logits: ActionLogitTensor
    z_e: LatentTensor
    z_q: LatentTensor
    z_q_no_grad: LatentTensor
    code_id: ArchetypeLabelTensor
    code_indices: ArchetypeLabelTensor
    reconstruction_loss: torch.Tensor
    vq_loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    total_loss: torch.Tensor


class ArchetypeVQModel(nn.Module):
    """Phase I VQ encoder-decoder 总模型。

    这个类把 trajectory encoder、VQ codebook 和 causal decoder 串起来。
    它既能作为 Phase I 训练模型，也能在训练完成后作为 label generator
    和 base action generator 被 Phase II/III 复用。

    模块说明:
        1. ``encoder``:
           输入 ``(states, relative_states, trend_states, actions, rewards,
           sample_ids)``，
           使用三路市场状态、actions 和 rewards 输出连续 latent ``z_e``。
           它负责从 DP teacher trajectory 中抽取整段交易行为模式。
        2. ``quantizer``:
           把 ``z_e`` 映射到最近的 codebook 向量 ``z_q``，并输出离散
           ``code_indices``。这是模型从连续表示变成 archetype label 的核心。
        3. ``decoder``:
           输入 ``states/relative_states/trend_states`` 和 ``z_q``，重构逐步
           动作 logits。它验证 code 是否真的包含足够信息来指导 action generation。
        4. ``_build_outputs``:
           统一计算 action reconstruction loss 和 VQ loss，并把训练/诊断
           需要的中间 Tensor 打包返回。
    """

    def __init__(
        self,
        state_dim: int,
        relative_state_dim: int,
        trend_state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        num_archetypes: int = 10,
        commitment_cost: float = 0.25,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """初始化 Phase I VQ archetype 模型。

        为什么需要:
            顶层模型需要把 encoder、quantizer、decoder 用同一组维度参数
            组装起来，保证 ``latent_dim``、``hidden_dim``、``action_dim``
            在三个子模块之间一致，避免训练时出现隐式 shape mismatch。

        功能说明:
            保存模型超参数，并创建三个核心子模块：
            ``encoder`` 负责把 demonstration trajectory 压缩成 ``z_e``；
            ``quantizer`` 负责把 ``z_e`` 离散化为 archetype code；
            ``decoder`` 负责根据三路市场状态和 archetype code 重构动作 logits。

        使用场景:
            在 Phase I 训练脚本、checkpoint 加载、离线 label 生成工具中创建
            模型实例时调用。调用方通常只需要提供 ``state_dim``，其余参数
            可来自实验配置。
        """

        super().__init__()
        self.state_dim = state_dim
        self.relative_state_dim = relative_state_dim
        self.trend_state_dim = trend_state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_archetypes = num_archetypes

        # Trajectory -> 连续 latent：学习 DP teacher 的整段行为摘要。
        self.encoder = ArchetypeTrajectoryEncoder(
            state_dim=state_dim,
            relative_state_dim=relative_state_dim,
            trend_state_dim=trend_state_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        # 连续 latent -> 离散 archetype：形成 Phase II 可监督的 code label。
        self.quantizer = VectorQuantizer(
            num_archetypes=num_archetypes,
            latent_dim=latent_dim,
            commitment_cost=commitment_cost,
        )
        # states + archetype -> action logits：检验并复用离散原型的执行能力。
        self.decoder = ArchetypeActionDecoder(
            state_dim=state_dim,
            relative_state_dim=relative_state_dim,
            trend_state_dim=trend_state_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, batch: TrajectoryTensorBatch) -> VqModelOutputs:
        """完整 VQ 训练前向。

        为什么需要:
            Phase I 的核心目标是同时学习“如何把 teacher trajectory 聚类成
            离散 archetype”和“每个 archetype 是否能重构 teacher action”。
            ``forward`` 把这两个目标放在同一次前向里，方便 trainer 直接
            使用 ``total_loss`` 做标准 VQ-VAE 风格训练。

        功能说明:
            1. 标准化 ``batch`` 的形状。
            2. 用 ``encoder`` 从 trajectory batch 得到连续 latent ``z_e``。
            3. 用 ``quantizer`` 把 ``z_e`` 映射成离散 code 和 STE 后的
               ``z_q``。
            4. 用 ``decoder`` 根据三路市场状态和 ``z_q`` 重构 ``action_logits``。
            5. 调用 ``_build_outputs`` 计算 reconstruction loss、VQ loss，
               并打包所有训练和诊断需要的 Tensor。

        使用场景:
            Phase I 正式 VQ 训练阶段调用。此时 ``actions`` 同时作为 encoder
            输入和 reconstruction target，模型会学习 codebook 分配和动作
            重构。
        """

        batch = _normalize_trajectory_batch(batch)
        states, relative_states, trend_states, actions, _, _ = batch
        z_e = self.encoder(batch)
        quantize_output = self.quantizer(z_e)
        action_logits = self.decoder(
            states,
            relative_states,
            trend_states,
            quantize_output.quantized,
        )
        return self._build_outputs(
            action_logits=action_logits,
            actions=actions,
            z_e=z_e,
            z_q=quantize_output.quantized,
            z_q_no_grad=quantize_output.z_q_no_grad,
            code_indices=quantize_output.code_indices,
            vq_loss=quantize_output.vq_loss,
            codebook_loss=quantize_output.codebook_loss,
            commitment_loss=quantize_output.commitment_loss,
        )

    def forward_pretrain(self, batch: TrajectoryTensorBatch) -> VqModelOutputs:
        """Phase A 预训练前向。

        为什么需要:
            VQ 训练一开始如果 encoder/decoder 还没有基本重构能力，codebook
            很容易被噪声 latent 牵引，导致 code 使用不稳定或塌缩。预训练
            先跳过离散量化，让连续 ``z_e`` 直接条件化 decoder，降低正式
            VQ 阶段的优化难度。

        功能说明:
            1. 用 ``encoder`` 得到连续 latent ``z_e``。
            2. 直接把 ``z_e`` 传给 ``decoder``，不经过 ``quantizer``。
            3. VQ 相关 loss 返回 0，``code_indices`` 用占位 0 填充。
            4. 仍然复用 ``_build_outputs``，保证返回结构和正式
               ``forward`` 一致。

        使用场景:
            Phase I 的 warmup / Phase A 预训练阶段调用。训练目标只关注
            action reconstruction，用来稳定 encoder/decoder 的基础表示。
        """

        batch = _normalize_trajectory_batch(batch)
        states, relative_states, trend_states, actions, _, _ = batch
        z_e = self.encoder(batch)
        action_logits = self.decoder(states, relative_states, trend_states, z_e)
        zero = z_e.new_zeros(())
        code_indices = z_e.new_zeros((z_e.shape[0],), dtype=torch.long)
        return self._build_outputs(
            action_logits=action_logits,
            actions=actions,
            z_e=z_e,
            z_q=z_e,
            z_q_no_grad=z_e.detach(),
            code_indices=code_indices,
            vq_loss=zero,
            codebook_loss=zero,
            commitment_loss=zero,
        )

    @torch.no_grad()
    def encode(
        self,
        batch: TrajectoryTensorBatch,
    ) -> tuple[ArchetypeLabelTensor, LatentTensor]:
        """离线生成 archetype label。

        为什么需要:
            Phase II 的 archetype selector 需要监督信号或参考标签，而这些
            标签来自 Phase I 训练好的 VQ codebook。``encode`` 提供一个只做
            编码和量化的入口，不执行 decoder，也不计算训练 loss。

        功能说明:
            1. 标准化 trajectory batch。
            2. 用 ``encoder`` 得到连续 latent ``z_e``。
            3. 用 ``quantizer`` 找到最近的 archetype code。
            4. 返回 ``code_indices`` 和 ``z_e``。方法带 ``torch.no_grad``，
               适合批量离线处理。

        使用场景:
            Phase II 训练前，为每个 horizon 生成 archetype label；也可用于
            诊断 encoder latent 分布、codebook 使用率和 trajectory 聚类效果。

        返回:
            ``(code_id, z_e)``。``code_id`` 可作为 Phase II selector 的
            demonstration label，``z_e`` 可用于诊断 latent 分布。
        """

        batch = _normalize_trajectory_batch(batch)
        z_e = self.encoder(batch)
        quantize_output = self.quantizer(z_e)
        return quantize_output.code_indices, z_e

    @torch.no_grad()
    def decode(
        self,
        states_seq: torch.Tensor,
        relative_states_seq: torch.Tensor,
        trend_states_seq: torch.Tensor,
        code_id: ArchetypeLabelTensor,
    ) -> tuple[torch.Tensor, ActionLogitTensor]:
        """在线生成，根据指定 archetype 生成基础动作。

        为什么需要:
            Phase II/III 中 selector 选择的是离散 archetype id，而真正下单
            或继续细化策略时需要逐时间步动作。``decode`` 是从 archetype id
            回到单个 horizon 内可执行基础动作序列的推理入口。

        功能说明:
            1. 接收一个 horizon 内的部分市场状态序列 ``states_seq`` 和一个
               指定 ``code_id``。
            2. 从 codebook 中取出对应 embedding ``z_q``。
            3. 内部临时补一个 batch 维度，复用 batch-first 的 ``decoder``。
            4. 对 logits 做 ``argmax``，得到基础动作 ``base_actions``。
               方法带 ``torch.no_grad``，默认用于推理/离线生成。

        输入形状:
            ``states_seq``: ``[partial_horizon, state_dim]``
                单个 horizon 内已经观察到或准备解码的状态片段。
                ``partial_horizon`` 可以小于完整 horizon 长度；``state_dim``
                必须和模型初始化时的 ``state_dim`` 一致。

            ``code_id``: 标量或单元素 Tensor
                该状态片段要使用的 archetype id。这个接口只解码一个
                horizon 片段，因此不接受 ``[batch]`` 形式的多个 code。

        内部形状:
            ``states_batch``: ``[1, partial_horizon, state_dim]``
            ``z_q``: ``[1, latent_dim]``
            ``decode_logits_batch``: ``[1, partial_horizon, action_dim]``

        使用场景:
            Phase II/III 推理时，根据 selector 选中的 archetype 生成基础交易
            动作；也可在一个 horizon 还没走完时，用当前已经有的
            ``partial_horizon`` 状态片段生成基础动作。

        返回:
            ``(base_actions, decode_logits)``，其中 ``base_actions`` 是
            ``argmax`` 后的动作 id，形状为 ``[partial_horizon]``；
            ``decode_logits`` 形状为 ``[partial_horizon, action_dim]``。
        """

        if states_seq.ndim != 2:
            raise ValueError("states_seq must have shape [partial_horizon, state_dim]")
        if states_seq.shape[-1] != self.state_dim:
            raise ValueError(
                f"states_seq last dim must be {self.state_dim}, got {states_seq.shape[-1]}"
            )
        if relative_states_seq.ndim != 2:
            raise ValueError(
                "relative_states_seq must have shape "
                "[partial_horizon, relative_state_dim]"
            )
        if relative_states_seq.shape[-1] != self.relative_state_dim:
            raise ValueError(
                "relative_states_seq last dim must be "
                f"{self.relative_state_dim}, got {relative_states_seq.shape[-1]}"
            )
        if trend_states_seq.ndim != 2:
            raise ValueError(
                "trend_states_seq must have shape [partial_horizon, trend_state_dim]"
            )
        if trend_states_seq.shape[-1] != self.trend_state_dim:
            raise ValueError(
                "trend_states_seq last dim must be "
                f"{self.trend_state_dim}, got {trend_states_seq.shape[-1]}"
            )
        if relative_states_seq.shape[0] != states_seq.shape[0]:
            raise ValueError("relative_states_seq and states_seq must share horizon")
        if trend_states_seq.shape[0] != states_seq.shape[0]:
            raise ValueError("trend_states_seq and states_seq must share horizon")

        if not torch.is_tensor(code_id):
            code_id = torch.as_tensor(code_id, dtype=torch.long, device=states_seq.device)
        else:
            code_id = code_id.to(states_seq.device)
        if code_id.numel() != 1:
            raise ValueError("code_id must be a scalar or single-element tensor")
        code_id = code_id.reshape(1).long()

        states_batch = states_seq.unsqueeze(0)
        relative_states_batch = relative_states_seq.unsqueeze(0)
        trend_states_batch = trend_states_seq.unsqueeze(0)
        z_q = self.quantizer.embedding_from_code(code_id.to(states_seq.device))
        decode_logits = self.decoder(
            states_batch,
            relative_states_batch,
            trend_states_batch,
            z_q,
        ).squeeze(0)
        base_actions = decode_logits.argmax(dim=-1)
        return base_actions, decode_logits

    def _build_outputs(
        self,
        *,
        action_logits: ActionLogitTensor,
        actions: torch.Tensor,
        z_e: LatentTensor,
        z_q: LatentTensor,
        z_q_no_grad: LatentTensor,
        code_indices: ArchetypeLabelTensor,
        vq_loss: torch.Tensor,
        codebook_loss: torch.Tensor,
        commitment_loss: torch.Tensor,
    ) -> VqModelOutputs:
        """统一计算 loss 并打包模型输出。

        为什么需要:
            ``forward`` 和 ``forward_pretrain`` 的前向路径不同，但输出结构
            应该完全一致。把 loss 计算和 ``VqModelOutputs`` 构造集中到一个
            私有方法，可以减少重复代码，也能避免两个训练入口返回字段不一致。

        功能说明:
            1. 把 ``action_logits`` 和 ``actions`` 展平成
               ``[batch * horizon, action_dim]`` 与 ``[batch * horizon]``。
            2. 使用 ``cross_entropy`` 计算 teacher action reconstruction loss。
            3. 将 reconstruction loss 与传入的 ``vq_loss`` 相加得到
               ``total_loss``。
            4. 返回包含 logits、latent、code id、分项 loss 和总 loss 的
               ``VqModelOutputs``。

        使用场景:
            仅供本类内部调用。正式 VQ 训练和预训练都复用它，trainer 可以
            依赖稳定的返回字段读取 loss、code label 和诊断信息。
        """

        reconstruction_loss = F.cross_entropy(
            action_logits.reshape(-1, self.action_dim),
            actions.reshape(-1).long(),
        )
        total_loss = reconstruction_loss + vq_loss
        return VqModelOutputs(
            action_logits=action_logits,
            z_e=z_e,
            z_q=z_q,
            z_q_no_grad=z_q_no_grad,
            code_id=code_indices,
            code_indices=code_indices,
            reconstruction_loss=reconstruction_loss,
            vq_loss=vq_loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            total_loss=total_loss,
        )


# 设计文档里的新名字。
ArchetypeEncoder = ArchetypeTrajectoryEncoder
ArchetypeDecoder = ArchetypeActionDecoder
VQArchetypeModel = ArchetypeVQModel


__all__ = [
    "ArchetypeActionDecoder",
    "ArchetypeDecoder",
    "ArchetypeEncoder",
    "ArchetypeTrajectoryEncoder",
    "ArchetypeVQModel",
    "classify_trajectory_directions",
    "CodebookInitResult",
    "MarketStateInputEncoder",
    "VqModelOutputs",
    "QuantizeOutput",
    "VQArchetypeModel",
    "VectorQuantizer",
]
